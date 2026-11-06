# DMS: drowsiness baseline

Vision-only drowsiness baseline for the driver monitoring system. It runs on the
Raspberry Pi 5, the main compute node. It runs MediaPipe Face Landmarker on the
driver camera and computes **EAR** (both eyes), **MAR** (yawn) and **PERCLOS**
over a rolling 60 s window. From those it drives a Level 0/1/2 alert state
machine. Each frame is drawn with a HUD and logged to CSV.

This is one specialised model in the late-fusion architecture. The STM32 safety
ECU makes the final alert decision.

## Setup

```bash
uv sync                    # Python 3.11 from .python-version, deps from uv.lock
uv run dms-download-model  # models/face_landmarker.task (pinned + sha256-checked)
uv run pytest              # unit tests
```

## Run

```bash
uv run dms-drowsiness                        # laptop webcam (device 0)
uv run dms-drowsiness --source 2             # other /dev/video index
uv run dms-drowsiness --source clip.mp4      # recorded video; timing uses media time
uv run dms-drowsiness --no-display --duration 60   # headless benchmark
```

`q`/`Esc` quits. The console prints FPS, process/system CPU, SoC temperature
(on the Pi) and inference time every `logging.print_interval_s`. Every level
change is printed with the reason.

Every run writes `logs/drowsiness_<timestamp>.csv` plus a `.toml` copy of the
config it used.

To test the IR/grayscale path on a webcam, set `camera.force_grayscale = true`.

## Raspberry Pi 5 + OV9281

Set in `config/drowsiness.toml`:

```toml
[camera]
source = "rpicam"      # or "picamera2"
width = 1280           # keep a multiple of 64
height = 800
fps = 60
camera_index = 0
mono = true            # OV9281 is monochrome: frames stay single-channel
clahe = true           # optional, helps under 850 nm IR
```

* **`rpicam`** pipes YUV420 from `rpicam-vid` and keeps the Y plane. It needs
  no Python packages, so it works with a plain `uv sync` venv.
* **`picamera2`** gives sensor timestamps (useful later for syncing with
  CAN/GNSS). picamera2 comes from apt, so create the venv with system packages:
  `sudo apt install python3-picamera2 && uv venv --system-site-packages && uv sync`.
  Caveat: the apt `simplejpeg` is built against numpy 1.x, so it may not import
  with the numpy 2.x in this venv. If picamera2 fails to import, use `rpicam`.

Run over SSH with `--no-display`. Grayscale frames are expanded to 3 channels
only for MediaPipe (which requires SRGB) and for drawing.

## Config (`config/drowsiness.toml`)

All thresholds live here. Unknown keys are rejected, so typos fail loudly.

| key | default | meaning |
|---|---|---|
| `thresholds.ear_closed` | 0.22 | mean EAR below this = eye closed |
| `thresholds.mar_yawn` | 0.5 | MAR above this = mouth wide open |
| `perclos.window_s` / `min_coverage_s` | 60 / 20 | rolling window; PERCLOS ignored until 20 s of face time |
| `alerts.min_closure_s` | 0.5 | shorter closures are blinks and are ignored |
| `alerts.warning_closure_s` / `critical_closure_s` | 1.0 / 2.0 | continuous closure, L1 / L2 |
| `alerts.perclos_warning` / `perclos_critical` | 0.15 / 0.25 | L1 / L2 |
| `alerts.yawn_min_s`, `yawn_window_s`, `yawn_warning_count` | 1.5, 300, 3 | yawn = MAR high ≥1.5 s; 3 in 5 min, L1 |
| `alerts.no_face_warning_s` | 3.0 | face lost (head drop / looking away), L1 |
| `alerts.recovery_s` | 3.0 | step down one level only after 3 s clear |

EAR depends on camera angle, lens and the person, so 0.22 is only a starting
point. Check the `ear` column in the CSV for open versus closed eyes on the
real OV9281 mount and adjust.

## Design notes

* **PERCLOS is time-weighted**, not frame-counted. It stays correct when FPS
  changes (Pi throttling, slow first inference). No-face time counts as
  unobserved, so it neither raises nor dilutes the ratio.
* **Escalate fast, recover slow.** Escalation is immediate. De-escalation
  steps down one level per `recovery_s`.
* A single-frame face dropout doesn't reset an ongoing closure.
* EAR/MAR are computed in pixel space, because MediaPipe's normalised x/y are
  anisotropic on non-square frames.
* `num_faces = 1`. In the car, a rear passenger could in principle be
  picked up instead of the driver; a driver-ROI filter is a later step.

## Layout

```
config/drowsiness.toml   thresholds + camera/model settings
src/dms/
  config.py      TOML -> typed dataclasses
  camera.py      opencv | file | picamera2 | rpicam sources, gray/BGR handling
  face.py        MediaPipe Face Landmarker (VIDEO mode) + model download
  metrics.py     EAR / MAR
  perclos.py     rolling time-weighted PERCLOS
  alerts.py      Level 0/1/2 state machine
  overlay.py     HUD
  csvlog.py      per-frame CSV
  perf.py        FPS / CPU / SoC temperature
  drowsiness.py  app entry point (dms-drowsiness)
tests/           unit tests (no camera or model needed)
```
