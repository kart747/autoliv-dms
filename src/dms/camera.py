"""Camera sources and frame normalisation.

Every source yields `Frame(image, t)`:
  * image: HxW uint8 (grayscale / IR) or HxWx3 uint8 BGR
  * t:     seconds on a clock that only moves forward within that source
           (capture time for live cameras, media time for video files)

Downstream code never assumes 3 channels. `to_rgb()` / `to_bgr()` adapt a
frame for MediaPipe (needs SRGB) and for drawing.

Sources:
  opencv    - USB/V4L2 webcams (also USB variants of the OV9281)
  file      - recorded video, timestamps from the container (repeatable tests)
  picamera2 - Pi 5 CSI cameras via libcamera. picamera2 comes from apt
              (python3-picamera2), so the venv must be created with
              `uv venv --system-site-packages`
  rpicam    - Pi 5 CSI cameras via an `rpicam-vid` YUV420 pipe. No Python deps.
              Always delivers the Y plane, so frames are grayscale.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol

import cv2
import numpy as np

from dms.config import CameraConfig


@dataclass
class Frame:
    image: np.ndarray
    t: float

    @property
    def is_gray(self) -> bool:
        return self.image.ndim == 2


def to_rgb(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def to_bgr(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img.copy()


class Preprocessor:
    """Brings a raw frame to the configured format: gray squeeze, IR CLAHE."""

    def __init__(self, cfg: CameraConfig):
        self.to_gray = cfg.mono or cfg.force_grayscale
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if cfg.clahe else None

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if img.ndim == 3 and img.shape[2] == 1:
            img = img[:, :, 0]
        if img.ndim == 3 and self.to_gray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if self.clahe is not None:
            if img.ndim == 2:
                img = self.clahe.apply(img)
            else:
                lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
                img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return np.ascontiguousarray(img)


class Source(Protocol):
    def frames(self) -> Iterator[Frame]: ...
    def close(self) -> None: ...


class OpenCVSource:
    def __init__(self, cfg: CameraConfig):
        dev = cfg.device
        if isinstance(dev, str) and dev.isdigit():
            dev = int(dev)
        backend = cv2.CAP_V4L2 if isinstance(dev, int) and _is_linux() else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(dev, backend)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open camera {cfg.device!r}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        self.cap.set(cv2.CAP_PROP_FPS, cfg.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # keep latency low

    def frames(self) -> Iterator[Frame]:
        while True:
            ok, img = self.cap.read()
            if not ok:
                return
            yield Frame(img, time.monotonic())

    def close(self) -> None:
        self.cap.release()


class FileSource:
    def __init__(self, cfg: CameraConfig):
        if not Path(cfg.path).is_file():
            raise FileNotFoundError(cfg.path)
        self.cap = cv2.VideoCapture(cfg.path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or float(cfg.fps)

    def frames(self) -> Iterator[Frame]:
        idx, last_t = 0, -1.0
        while True:
            ok, img = self.cap.read()
            if not ok:
                return
            t = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if t <= last_t:  # some containers report 0 / repeats
                t = idx / self.fps
            last_t = t
            idx += 1
            yield Frame(img, t)

    def close(self) -> None:
        self.cap.release()


class Picamera2Source:
    def __init__(self, cfg: CameraConfig):
        try:
            from picamera2 import Picamera2  # type: ignore[import-not-found]
        except ImportError as e:
            raise RuntimeError(
                "picamera2 not importable. On the Pi: `sudo apt install python3-picamera2` "
                "and create the venv with `uv venv --system-site-packages`, "
                "or use camera.source = 'rpicam'."
            ) from e
        self.w, self.h, self.mono = cfg.width, cfg.height, cfg.mono
        self.cam = Picamera2(cfg.camera_index)
        # YUV420 for mono sensors: the Y plane is the IR intensity image.
        # "RGB888" in libcamera naming is BGR byte order, which is what OpenCV wants.
        fmt = "YUV420" if cfg.mono else "RGB888"
        conf = self.cam.create_video_configuration(
            main={"size": (cfg.width, cfg.height), "format": fmt},
            controls={"FrameRate": float(cfg.fps)},
            buffer_count=4,
        )
        self.cam.configure(conf)
        self.cam.start()

    def frames(self) -> Iterator[Frame]:
        while True:
            req = self.cam.capture_request()
            try:
                arr = req.make_array("main")
                ts_ns = req.get_metadata().get("SensorTimestamp")
            finally:
                req.release()
            img = arr[: self.h, : self.w] if self.mono else arr[:, : self.w]
            t = ts_ns / 1e9 if ts_ns else time.monotonic()
            yield Frame(np.ascontiguousarray(img), t)

    def close(self) -> None:
        self.cam.stop()
        self.cam.close()


class RpicamSource:
    """Reads raw YUV420 from `rpicam-vid`. Width should be a multiple of 64 so
    rows carry no stride padding (the OV9281's 1280 and 640 both qualify)."""

    def __init__(self, cfg: CameraConfig):
        exe = shutil.which("rpicam-vid")
        if exe is None:
            raise RuntimeError("rpicam-vid not found (install rpicam-apps)")
        if cfg.width % 64:
            print(f"[camera] warning: width {cfg.width} is not a multiple of 64; frames may shear")
        self.w, self.h = cfg.width, cfg.height
        self.frame_bytes = cfg.width * cfg.height * 3 // 2
        cmd = [
            exe, "-t", "0", "-n", "--flush",
            "--camera", str(cfg.camera_index),
            "--width", str(cfg.width), "--height", str(cfg.height),
            "--framerate", str(cfg.fps),
            "--codec", "yuv420", "-o", "-",
        ]  # fmt: skip
        self.proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=self.frame_bytes
        )

    def frames(self) -> Iterator[Frame]:
        buf = bytearray(self.frame_bytes)
        view = memoryview(buf)
        ysize = self.w * self.h
        while True:
            got = 0
            while got < self.frame_bytes:
                n = self.proc.stdout.readinto(view[got:])  # type: ignore[union-attr]
                if not n:
                    return
                got += n
            y = np.frombuffer(buf, np.uint8, ysize).reshape(self.h, self.w).copy()
            yield Frame(y, time.monotonic())

    def close(self) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def _is_linux() -> bool:
    import sys

    return sys.platform.startswith("linux")


def apply_source_override(cfg: CameraConfig, source: str | None) -> None:
    """CLI shorthand: '0' or '/dev/video2' -> opencv, a video path -> file,
    'picamera2' / 'rpicam' -> that backend."""
    if not source:
        return
    if source in {"picamera2", "rpicam", "opencv"}:
        cfg.source = source
    elif source.isdigit() or source.startswith("/dev/video") or "://" in source:
        cfg.source, cfg.device = "opencv", int(source) if source.isdigit() else source
    elif Path(source).is_file():
        cfg.source, cfg.path = "file", source
    else:
        raise ValueError(f"cannot interpret --source {source!r}")


def open_source(cfg: CameraConfig) -> Source:
    return {
        "opencv": OpenCVSource,
        "file": FileSource,
        "picamera2": Picamera2Source,
        "rpicam": RpicamSource,
    }[cfg.source](cfg)
