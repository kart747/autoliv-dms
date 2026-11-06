"""MediaPipe Face Landmarker wrapper (VIDEO running mode).

VIDEO mode is synchronous and tracks landmarks between frames, so it's cheaper
than IMAGE mode. Unlike LIVE_STREAM it never silently drops frames, which keeps
the per-frame CSV log complete. The input must be SRGB, so grayscale IR frames
are expanded to 3 channels before inference (see camera.to_rgb).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from dms.config import ModelConfig

# Pinned model version (not "latest") so results are reproducible.
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODEL_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"


class FaceLandmarker:
    def __init__(self, cfg: ModelConfig):
        path = Path(cfg.path)
        if not path.is_file():
            raise FileNotFoundError(f"{path} missing. Run `uv run dms-download-model` first.")
        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=cfg.min_face_detection_confidence,
            min_face_presence_confidence=cfg.min_face_presence_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
        )
        self._lm = vision.FaceLandmarker.create_from_options(opts)
        self._last_ms = -1

    def detect(self, rgb: np.ndarray, t: float) -> np.ndarray | None:
        """Return (478, 2) landmark pixel coordinates, or None when there's no face."""
        ts = int(t * 1000)
        if ts <= self._last_ms:  # MediaPipe requires strictly increasing timestamps
            ts = self._last_ms + 1
        self._last_ms = ts
        h, w = rgb.shape[:2]
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._lm.detect_for_video(image, ts)
        if not result.face_landmarks:
            return None
        lms = result.face_landmarks[0]
        return np.array([(p.x * w, p.y * h) for p in lms], dtype=np.float32)

    def close(self) -> None:
        self._lm.close()


def download_model(dest: Path, force: bool = False) -> Path:
    if dest.is_file() and not force:
        if _sha256(dest) == MODEL_SHA256:
            print(f"{dest} already present and verified")
            return dest
        print(f"{dest} has an unexpected checksum, re-downloading")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    print(f"downloading {MODEL_URL}")
    urllib.request.urlretrieve(MODEL_URL, tmp)
    digest = _sha256(tmp)
    if digest != MODEL_SHA256:
        tmp.unlink()
        raise RuntimeError(f"checksum mismatch: got {digest}")
    tmp.replace(dest)
    print(f"saved {dest} (sha256 ok)")
    return dest


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_model_cli() -> None:
    ap = argparse.ArgumentParser(description="Download the MediaPipe Face Landmarker model")
    ap.add_argument("--dest", default=ModelConfig().path)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    try:
        download_model(Path(args.dest), args.force)
    except Exception as e:  # noqa: BLE001 - CLI boundary
        sys.exit(f"error: {e}")
