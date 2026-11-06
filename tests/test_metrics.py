import numpy as np
import pytest

from dms.metrics import (
    LEFT_EYE,
    MOUTH_CORNERS,
    MOUTH_PAIRS,
    RIGHT_EYE,
    eye_aspect_ratio,
    face_metrics,
    mouth_aspect_ratio,
)


def _eye(pts, idx, cx, width, height):
    """Place a synthetic eye: corners at +-width/2, lids at +-height/2."""
    p1, p2, p3, p4, p5, p6 = idx
    half_w, half_h = width / 2, height / 2
    pts[p1] = (cx - half_w, 0)
    pts[p4] = (cx + half_w, 0)
    pts[p2] = (cx - half_w / 3, -half_h)
    pts[p3] = (cx + half_w / 3, -half_h)
    pts[p6] = (cx - half_w / 3, half_h)
    pts[p5] = (cx + half_w / 3, half_h)


def _mouth(pts, width, opening):
    pts[MOUTH_CORNERS[0]] = (-width / 2, 100)
    pts[MOUTH_CORNERS[1]] = (width / 2, 100)
    for k, (u, l) in enumerate(MOUTH_PAIRS):
        x = (k - 1) * width / 4
        pts[u] = (x, 100 - opening / 2)
        pts[l] = (x, 100 + opening / 2)


def _face(eye_h=9.0, mouth_open=0.0):
    pts = np.zeros((478, 2))
    _eye(pts, RIGHT_EYE, -40, 30, eye_h)
    _eye(pts, LEFT_EYE, 40, 30, eye_h)
    _mouth(pts, 50, mouth_open)
    return pts


def test_ear_open_eye():
    # height/width = 9/30 -> EAR = (9 + 9) / (2 * 30) = 0.3
    assert eye_aspect_ratio(_face(9.0), LEFT_EYE) == pytest.approx(0.3)


def test_ear_closed_eye():
    assert eye_aspect_ratio(_face(0.0), RIGHT_EYE) == pytest.approx(0.0)


def test_ear_is_scale_invariant():
    pts = _face(9.0)
    assert eye_aspect_ratio(pts * 3.7, LEFT_EYE) == pytest.approx(0.3)


def test_degenerate_eye_returns_zero():
    assert eye_aspect_ratio(np.zeros((478, 2)), LEFT_EYE) == 0.0


def test_mar_closed_and_yawning():
    assert mouth_aspect_ratio(_face(mouth_open=0.0)) == pytest.approx(0.0)
    assert mouth_aspect_ratio(_face(mouth_open=35.0)) == pytest.approx(0.7)


def test_face_metrics_mean():
    ear_l, ear_r, ear, mar = face_metrics(_face(6.0, 10.0))
    assert ear_l == pytest.approx(0.2)
    assert ear_r == pytest.approx(0.2)
    assert ear == pytest.approx(0.2)
    assert mar == pytest.approx(0.2)
