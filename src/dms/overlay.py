"""On-frame HUD: metrics, FPS and alert state."""

from __future__ import annotations

import cv2
import numpy as np

from dms.alerts import AlertLevel, AlertStatus
from dms.metrics import LEFT_EYE, MOUTH_CORNERS, MOUTH_PAIRS, RIGHT_EYE

LEVEL_STYLE = {
    AlertLevel.ALERT: ("L0 ALERT", (60, 180, 60)),
    AlertLevel.WARNING: ("L1 WARNING", (0, 165, 255)),
    AlertLevel.CRITICAL: ("L2 CRITICAL", (0, 0, 255)),
}

_MOUTH_IDX = MOUTH_CORNERS + tuple(i for pair in MOUTH_PAIRS for i in pair)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_landmarks(img: np.ndarray, pts: np.ndarray, eye_closed: bool) -> None:
    eye_col = (0, 0, 255) if eye_closed else (0, 255, 0)
    for idx, col in ((LEFT_EYE, eye_col), (RIGHT_EYE, eye_col), (_MOUTH_IDX, (255, 200, 0))):
        for i in idx:
            x, y = pts[i]
            cv2.circle(img, (int(x), int(y)), 2, col, -1, cv2.LINE_AA)


def draw_hud(
    img: np.ndarray,
    *,
    face: bool,
    ear: float | None,
    mar: float | None,
    perclos: float,
    status: AlertStatus,
    fps: float,
    infer_ms: float,
    ear_thr: float,
) -> None:
    label, colour = LEVEL_STYLE[status.level]
    h, w = img.shape[:2]
    scale = max(0.45, w / 1280)
    line = int(26 * scale / 0.6)

    lines = [
        f"EAR  {ear:.3f}  (thr {ear_thr:.2f})" if ear is not None else "EAR  --",
        f"MAR  {mar:.3f}" if mar is not None else "MAR  --",
        f"PERCLOS {perclos:5.1%}" + ("" if status.perclos_valid else " (warming)"),
        f"closure {status.closure_s:.1f}s  blinks {status.blink_count}  yawns {status.yawn_count}",
        f"FPS {fps:4.1f}  infer {infer_ms:4.1f} ms",
    ]
    if not face:
        lines.insert(0, f"NO FACE {status.no_face_s:.1f}s")

    panel_h = line * (len(lines) + 2) + 10
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (int(430 * scale / 0.6), panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, img, 0.55, 0, dst=img)

    cv2.putText(img, label, (10, line), _FONT, scale * 1.4, colour, 2, cv2.LINE_AA)
    for k, text in enumerate(lines):
        cv2.putText(img, text, (10, line * (k + 2) + 4), _FONT, scale, (255, 255, 255), 1, cv2.LINE_AA)
    if status.reasons:
        cv2.putText(img, ", ".join(status.reasons), (10, h - 12), _FONT, scale, colour, 2, cv2.LINE_AA)
    if status.level > AlertLevel.ALERT:
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), colour, 6 if status.level == AlertLevel.CRITICAL else 3)
