"""Eye and mouth aspect ratios from MediaPipe Face Landmarker points.

All distances must be computed in pixel space, not in MediaPipe's normalised
[0,1] x/y, because the normalised axes have different scales whenever the frame
is not square. Callers pass an (N, 2) array of pixel coordinates.

Index sets are for the 478-point MediaPipe face mesh. "Right"/"left" are the
subject's own sides (the subject's right eye appears on the image left).
"""

from __future__ import annotations

import numpy as np

# 6-point EAR layout (Soukupova & Cech 2016): p1/p4 are the corners,
# (p2, p6) and (p3, p5) are the upper/lower lid pairs.
RIGHT_EYE = (33, 160, 158, 133, 153, 144)
LEFT_EYE = (362, 385, 387, 263, 373, 380)

# Inner-lip MAR: corners plus three vertical upper/lower pairs.
MOUTH_CORNERS = (78, 308)
MOUTH_PAIRS = ((81, 178), (13, 14), (311, 402))


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def eye_aspect_ratio(pts: np.ndarray, idx: tuple[int, ...]) -> float:
    """EAR = (|p2-p6| + |p3-p5|) / (2 |p1-p4|). About 0.25-0.35 open, near 0 closed."""
    p1, p2, p3, p4, p5, p6 = (pts[i] for i in idx)
    width = _dist(p1, p4)
    if width < 1e-6:
        return 0.0
    return (_dist(p2, p6) + _dist(p3, p5)) / (2.0 * width)


def mouth_aspect_ratio(pts: np.ndarray) -> float:
    """MAR = mean inner-lip opening / mouth width. About 0 closed, above 0.5 yawning."""
    width = _dist(pts[MOUTH_CORNERS[0]], pts[MOUTH_CORNERS[1]])
    if width < 1e-6:
        return 0.0
    opening = np.mean([_dist(pts[u], pts[l]) for u, l in MOUTH_PAIRS])
    return float(opening / width)


def face_metrics(pts: np.ndarray) -> tuple[float, float, float, float]:
    """Return (ear_left, ear_right, ear_mean, mar)."""
    ear_l = eye_aspect_ratio(pts, LEFT_EYE)
    ear_r = eye_aspect_ratio(pts, RIGHT_EYE)
    return ear_l, ear_r, (ear_l + ear_r) / 2.0, mouth_aspect_ratio(pts)
