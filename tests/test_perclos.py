import pytest

from dms.perclos import Perclos


def feed(p, t0, t1, closed, fps=30.0):
    """Feed samples on [t0, t1) at a fixed rate; return the next timestamp."""
    n = round((t1 - t0) * fps)
    for k in range(n):
        p.update(t0 + k / fps, closed)
    return t0 + n / fps


def test_all_open_is_zero():
    p = Perclos(60)
    feed(p, 0, 10, False)
    assert p.value == pytest.approx(0.0)
    assert p.coverage_s == pytest.approx(10, abs=0.05)


def test_fraction_closed():
    p = Perclos(60)
    t = feed(p, 0, 9, False)
    t = feed(p, t, 10, True)
    p.update(t, False)
    assert p.value == pytest.approx(0.1, abs=0.01)


def test_independent_of_frame_rate():
    a, b = Perclos(60), Perclos(60)
    for p, fps in ((a, 10.0), (b, 60.0)):
        t = feed(p, 0, 8, False, fps)
        t = feed(p, t, 10, True, fps)
        p.update(t, False)
    assert a.value == pytest.approx(b.value, abs=0.01)
    assert a.value == pytest.approx(0.2, abs=0.01)


def test_window_forgets_old_closures():
    p = Perclos(60)
    t = feed(p, 0, 10, True)  # 10 s closed at the start
    t = feed(p, t, 80, False)  # then 70 s open, so the closures have left the window
    assert p.value == pytest.approx(0.0, abs=1e-6)
    assert p.coverage_s == pytest.approx(60, abs=0.05)


def test_partial_window_trim():
    p = Perclos(60)
    t = feed(p, 0, 20, True)
    t = feed(p, t, 70, False)  # window = [10, 70): 10 s closed of 60 s
    assert p.value == pytest.approx(10 / 60, abs=0.01)


def test_no_face_gap_is_unobserved():
    p = Perclos(60)
    t = feed(p, 0, 10, False)
    t = feed(p, t + 30, t + 40, True)  # 30 s with no face, then 10 s closed
    p.update(t, False)
    assert p.coverage_s == pytest.approx(20, abs=0.1)
    assert p.value == pytest.approx(0.5, abs=0.01)


def test_none_breaks_hold():
    p = Perclos(60)
    p.update(0.0, True)
    p.update(0.1, None)  # face lost: the 0.0-0.1 interval still counts as closed
    p.update(0.2, False)  # 0.1-0.2 is unobserved
    assert p.coverage_s == pytest.approx(0.1)
    assert p.value == pytest.approx(1.0)
