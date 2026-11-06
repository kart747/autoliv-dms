"""Drowsiness baseline: camera -> Face Landmarker -> EAR/MAR/PERCLOS -> alert level.

    uv run dms-drowsiness                         # laptop webcam, config/drowsiness.toml
    uv run dms-drowsiness --source 2              # another /dev/video index
    uv run dms-drowsiness --source clip.mp4       # recorded video (media-time PERCLOS)
    uv run dms-drowsiness --source rpicam --no-display   # Pi 5 + OV9281 over SSH

Press q or Esc in the preview window to quit.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

from dms.alerts import AlertLevel, AlertStateMachine
from dms.camera import Preprocessor, apply_source_override, open_source, to_bgr, to_rgb
from dms.config import Config, load_config
from dms.csvlog import CsvLogger
from dms.face import FaceLandmarker
from dms.metrics import face_metrics
from dms.overlay import LEVEL_STYLE, draw_hud, draw_landmarks
from dms.perclos import Perclos
from dms.perf import PerfMonitor

DEFAULT_CONFIG = "config/drowsiness.toml"
WINDOW = "DMS drowsiness baseline"


def _has_display() -> bool:
    if sys.platform.startswith("linux"):
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return True


def run(cfg: Config, config_path: str | None, show: bool, log: bool,
        max_frames: int | None, duration_s: float | None) -> int:  # fmt: skip
    thr = cfg.thresholds
    pre = Preprocessor(cfg.camera)
    landmarker = FaceLandmarker(cfg.model)
    source = open_source(cfg.camera)
    perclos = Perclos(cfg.perclos.window_s)
    sm = AlertStateMachine(cfg.alerts, cfg.perclos.min_coverage_s)
    perf = PerfMonitor()
    logger = CsvLogger(cfg.logging.csv_dir) if log else None
    if logger and config_path:  # keep the thresholds used next to the data
        shutil.copy(config_path, logger.path.with_suffix(".toml"))

    print(f"[dms] source={cfg.camera.source} {cfg.camera.width}x{cfg.camera.height}"
          f" display={'on' if show else 'off'} log={logger.path if logger else 'off'}")  # fmt: skip

    start = time.perf_counter()
    last_print = start
    level = AlertLevel.ALERT
    frame_idx = 0
    try:
        for frame in source.frames():
            img = pre(frame.image)
            t0 = time.perf_counter()
            pts = landmarker.detect(to_rgb(img), frame.t)
            infer_ms = (time.perf_counter() - t0) * 1000.0

            face = pts is not None
            ear_l = ear_r = ear = mar = None
            eye_closed = mouth_open = False
            if face:
                ear_l, ear_r, ear, mar = face_metrics(pts)
                eye_closed = ear < thr.ear_closed
                mouth_open = mar > thr.mar_yawn

            p = perclos.update(frame.t, eye_closed if face else None)
            status = sm.update(frame.t, face, eye_closed, mouth_open, p, perclos.coverage_s)
            perf.tick()

            if status.level != level:
                print(f"[dms] t={frame.t:8.2f}s  {LEVEL_STYLE[level][0]} -> "
                      f"{LEVEL_STYLE[status.level][0]}  ({', '.join(status.reasons) or 'recovered'})")  # fmt: skip
                level = status.level

            if logger:
                logger.write({
                    "wall_time": datetime.now().isoformat(timespec="milliseconds"),
                    "t": frame.t, "frame": frame_idx, "face": int(face),
                    "ear_l": ear_l, "ear_r": ear_r, "ear": ear, "mar": mar,
                    "eye_closed": int(eye_closed), "mouth_open": int(mouth_open),
                    "perclos": p, "perclos_valid": int(status.perclos_valid),
                    "closure_s": status.closure_s, "blinks": status.blink_count,
                    "long_closures": status.long_closure_count, "yawns": status.yawn_count,
                    "yawns_window": status.yawns_in_window, "no_face_s": status.no_face_s,
                    "level": int(status.level), "reasons": ";".join(status.reasons),
                    "fps": perf.fps, "infer_ms": infer_ms, "cpu_proc": perf.cpu_proc,
                    "cpu_sys": perf.cpu_sys, "temp_c": perf.temp_c,
                })  # fmt: skip

            now = time.perf_counter()
            if now - last_print >= cfg.logging.print_interval_s:
                last_print = now
                ear_s = f"{ear:.3f}" if ear is not None else "  -- "
                print(f"[perf] {perf.summary()} | infer {infer_ms:5.1f} ms | EAR {ear_s} "
                      f"PERCLOS {p:5.1%} | {LEVEL_STYLE[status.level][0]}")  # fmt: skip

            if show:
                vis = to_bgr(img)
                if face and cfg.display.draw_landmarks:
                    draw_landmarks(vis, pts, eye_closed)
                if cfg.camera.flip_horizontal:
                    vis = cv2.flip(vis, 1)
                draw_hud(vis, face=face, ear=ear, mar=mar, perclos=p, status=status,
                         fps=perf.fps, infer_ms=infer_ms, ear_thr=thr.ear_closed)  # fmt: skip
                cv2.imshow(WINDOW, vis)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break

            frame_idx += 1
            if max_frames is not None and frame_idx >= max_frames:
                break
            if duration_s is not None and now - start >= duration_s:
                break
    except KeyboardInterrupt:
        pass
    finally:
        source.close()
        landmarker.close()
        if logger:
            logger.close()
        if show:
            cv2.destroyAllWindows()

    elapsed = time.perf_counter() - start
    print(f"[dms] {frame_idx} frames in {elapsed:.1f}s ({frame_idx / max(elapsed, 1e-9):.1f} FPS avg), "
          f"blinks={sm.blink_count} long_closures={sm.long_closure_count} yawns={sm.yawn_count}")  # fmt: skip
    if logger:
        print(f"[dms] log: {logger.path}")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=DEFAULT_CONFIG if Path(DEFAULT_CONFIG).is_file() else None)
    ap.add_argument("--source", help="camera index, /dev/videoN, video file, 'picamera2' or 'rpicam'")
    ap.add_argument("--no-display", action="store_true", help="headless (SSH / benchmarking)")
    ap.add_argument("--no-log", action="store_true", help="don't write the CSV")
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--duration", type=float, help="stop after N seconds of wall time")
    args = ap.parse_args()

    try:
        cfg = load_config(args.config)
        apply_source_override(cfg.camera, args.source)
    except (OSError, ValueError) as e:
        sys.exit(f"config error: {e}")

    show = cfg.display.show and not args.no_display
    if show and not _has_display():
        print("[dms] no display detected, running headless")
        show = False
    try:
        sys.exit(run(cfg, args.config, show, not args.no_log, args.max_frames, args.duration))
    except (RuntimeError, FileNotFoundError) as e:
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()
