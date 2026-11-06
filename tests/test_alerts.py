from dms.alerts import AlertLevel, AlertStateMachine
from dms.config import AlertConfig

FPS = 30.0


class Driver:
    """Drives the state machine at a fixed frame rate."""

    def __init__(self, cfg=None, min_coverage=20.0):
        self.sm = AlertStateMachine(cfg or AlertConfig(), min_coverage)
        self.t = 0.0
        self.last = None

    def run(self, seconds, **kw):
        kw.setdefault("face_present", True)
        for _ in range(round(seconds * FPS)):
            self.last = self.sm.update(self.t, **kw)
            self.t += 1 / FPS
        return self.last

    def max_level(self, seconds, **kw):
        kw.setdefault("face_present", True)
        worst = AlertLevel.ALERT
        for _ in range(round(seconds * FPS)):
            self.last = self.sm.update(self.t, **kw)
            worst = max(worst, self.last.level)
            self.t += 1 / FPS
        return worst


def test_awake_driver_stays_alert():
    d = Driver()
    assert d.max_level(10) == AlertLevel.ALERT


def test_blinks_are_ignored_and_counted():
    d = Driver()
    worst = AlertLevel.ALERT
    for _ in range(20):
        worst = max(worst, d.max_level(0.3, eye_closed=True))  # 300 ms blink
        worst = max(worst, d.max_level(2.0))
    assert worst == AlertLevel.ALERT
    assert d.last.blink_count == 20
    assert d.last.long_closure_count == 0


def test_closure_just_under_min_is_a_blink():
    d = Driver()
    d.run(0.45, eye_closed=True)
    d.run(0.1)
    assert d.last.blink_count == 1


def test_warning_then_critical_on_long_closure():
    d = Driver()
    assert d.max_level(0.9, eye_closed=True) == AlertLevel.ALERT
    assert d.max_level(0.2, eye_closed=True) == AlertLevel.WARNING  # 1.1 s
    assert d.max_level(1.0, eye_closed=True) == AlertLevel.CRITICAL  # 2.1 s
    assert "microsleep" in d.last.reasons[0]


def test_stepwise_recovery_with_hysteresis():
    cfg = AlertConfig(recovery_s=3.0)
    d = Driver(cfg)
    d.run(2.5, eye_closed=True)
    assert d.last.level == AlertLevel.CRITICAL
    d.run(2.9)
    assert d.last.level == AlertLevel.CRITICAL  # not yet recovered
    d.run(0.2)
    assert d.last.level == AlertLevel.WARNING  # one step down, not straight to 0
    d.run(3.1)
    assert d.last.level == AlertLevel.ALERT
    assert d.last.long_closure_count == 1


def test_perclos_ignored_until_coverage():
    d = Driver(min_coverage=20.0)
    assert d.max_level(1, perclos=0.5, perclos_coverage_s=10.0) == AlertLevel.ALERT
    assert d.max_level(1, perclos=0.5, perclos_coverage_s=25.0) == AlertLevel.CRITICAL


def test_perclos_warning_level():
    d = Driver()
    st = d.run(0.1, perclos=0.18, perclos_coverage_s=60.0)
    assert st.level == AlertLevel.WARNING


def test_yawns_trigger_warning():
    d = Driver(AlertConfig(yawn_min_s=1.5, yawn_warning_count=3))
    for _ in range(2):
        d.run(2.0, mouth_open=True)
        d.run(5.0)
    assert d.last.level == AlertLevel.ALERT
    assert d.last.yawn_count == 2
    d.run(2.0, mouth_open=True)
    assert d.last.level == AlertLevel.WARNING


def test_short_mouth_opening_is_not_a_yawn():
    d = Driver()
    d.run(1.0, mouth_open=True)  # talking, not yawning
    d.run(1.0)
    assert d.last.yawn_count == 0


def test_no_face_warning():
    d = Driver()
    d.run(2.0)
    assert d.max_level(2.5, face_present=False) == AlertLevel.ALERT
    assert d.max_level(1.0, face_present=False) == AlertLevel.WARNING


def test_single_frame_dropout_does_not_reset_closure():
    d = Driver()
    d.run(0.8, eye_closed=True)
    d.run(1 / FPS, face_present=False)
    st = d.run(0.4, eye_closed=True)
    assert st.closure_s > 1.1
    assert st.level == AlertLevel.WARNING
