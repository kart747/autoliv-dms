"""Three-level drowsiness alert state machine.

Level 0 ALERT    : no drowsiness indicators.
Level 1 WARNING  : closure >= warning_closure_s, PERCLOS >= perclos_warning,
                   frequent yawning, or face lost for no_face_warning_s.
Level 2 CRITICAL : closure >= critical_closure_s (microsleep) or
                   PERCLOS >= perclos_critical.

Rules:
* Closures shorter than min_closure_s are blinks. They are counted but never
  trigger anything.
* Escalation is immediate. De-escalation steps down one level at a time, only
  after the target level has stayed lower for recovery_s (hysteresis), so the
  alert doesn't flicker.
* PERCLOS only counts once the window holds min_coverage_s of observed time.
* A short face dropout (<= dropout_tolerance_s, e.g. one missed detection)
  doesn't reset an ongoing closure. A longer one does, and the no-face timer
  takes over.

This is the Pi-side vision baseline. The STM32 safety ECU makes the final
alert decision. This module's output is one input to that fusion.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum

from dms.config import AlertConfig


class AlertLevel(IntEnum):
    ALERT = 0
    WARNING = 1
    CRITICAL = 2


@dataclass
class AlertStatus:
    level: AlertLevel = AlertLevel.ALERT
    reasons: list[str] = field(default_factory=list)
    closure_s: float = 0.0
    yawns_in_window: int = 0
    no_face_s: float = 0.0
    perclos_valid: bool = False
    blink_count: int = 0
    long_closure_count: int = 0
    yawn_count: int = 0


class AlertStateMachine:
    def __init__(
        self,
        cfg: AlertConfig,
        perclos_min_coverage_s: float,
        dropout_tolerance_s: float = 0.3,
    ):
        self.cfg = cfg
        self.perclos_min_coverage_s = perclos_min_coverage_s
        self.dropout_tolerance_s = dropout_tolerance_s
        self.level = AlertLevel.ALERT
        self._clear_since: float | None = None
        self._closure_start: float | None = None
        self._yawn_start: float | None = None
        self._yawn_counted = False
        self._yawn_times: deque[float] = deque()
        self._no_face_start: float | None = None
        self._last_face_t: float | None = None
        self.blink_count = 0
        self.long_closure_count = 0
        self.yawn_count = 0

    # -- per-signal trackers ---------------------------------------------------

    def _end_closure(self, t_end: float) -> None:
        if self._closure_start is None:
            return
        dur = t_end - self._closure_start
        if dur < self.cfg.min_closure_s:
            self.blink_count += 1
        else:
            self.long_closure_count += 1
        self._closure_start = None

    def _track_eyes(self, t: float, eye_closed: bool) -> float:
        if eye_closed:
            if self._closure_start is None:
                self._closure_start = t
            return t - self._closure_start
        self._end_closure(t)
        return 0.0

    def _track_yawn(self, t: float, mouth_open: bool) -> None:
        if not mouth_open:
            self._yawn_start = None
            self._yawn_counted = False
            return
        if self._yawn_start is None:
            self._yawn_start = t
        if not self._yawn_counted and t - self._yawn_start >= self.cfg.yawn_min_s:
            self._yawn_counted = True
            self._yawn_times.append(t)
            self.yawn_count += 1

    # -- main update ----------------------------------------------------------

    def update(
        self,
        t: float,
        face_present: bool,
        eye_closed: bool = False,
        mouth_open: bool = False,
        perclos: float = 0.0,
        perclos_coverage_s: float = 0.0,
    ) -> AlertStatus:
        c = self.cfg
        closure_s = 0.0
        no_face_s = 0.0

        if face_present:
            if self._no_face_start is not None:
                gap = t - self._no_face_start
                if gap > self.dropout_tolerance_s and self._last_face_t is not None:
                    # Long dropout: close out any closure at the last time we saw the face.
                    self._end_closure(self._last_face_t)
                    self._yawn_start, self._yawn_counted = None, False
            self._no_face_start = None
            self._last_face_t = t
            closure_s = self._track_eyes(t, eye_closed)
            self._track_yawn(t, mouth_open)
        else:
            if self._no_face_start is None:
                self._no_face_start = t
            no_face_s = t - self._no_face_start

        while self._yawn_times and t - self._yawn_times[0] > c.yawn_window_s:
            self._yawn_times.popleft()
        yawns = len(self._yawn_times)
        perclos_valid = face_present and perclos_coverage_s >= self.perclos_min_coverage_s

        reasons: list[str] = []
        target = AlertLevel.ALERT
        if closure_s >= c.critical_closure_s:
            target = AlertLevel.CRITICAL
            reasons.append(f"microsleep {closure_s:.1f}s")
        elif closure_s >= c.warning_closure_s:
            target = AlertLevel.WARNING
            reasons.append(f"closure {closure_s:.1f}s")
        if perclos_valid and perclos >= c.perclos_critical:
            target = AlertLevel.CRITICAL
            reasons.append(f"PERCLOS {perclos:.0%}")
        elif perclos_valid and perclos >= c.perclos_warning:
            target = max(target, AlertLevel.WARNING)
            reasons.append(f"PERCLOS {perclos:.0%}")
        if yawns >= c.yawn_warning_count:
            target = max(target, AlertLevel.WARNING)
            reasons.append(f"{yawns} yawns")
        if no_face_s >= c.no_face_warning_s:
            target = max(target, AlertLevel.WARNING)
            reasons.append(f"no face {no_face_s:.1f}s")

        self._apply(t, AlertLevel(target))

        return AlertStatus(
            level=self.level,
            reasons=reasons,
            closure_s=closure_s,
            yawns_in_window=yawns,
            no_face_s=no_face_s,
            perclos_valid=perclos_valid,
            blink_count=self.blink_count,
            long_closure_count=self.long_closure_count,
            yawn_count=self.yawn_count,
        )

    def _apply(self, t: float, target: AlertLevel) -> None:
        if target > self.level:
            self.level = target
            self._clear_since = None
        elif target < self.level:
            if self._clear_since is None:
                self._clear_since = t
            elif t - self._clear_since >= self.cfg.recovery_s:
                self.level = AlertLevel(self.level - 1)
                self._clear_since = t if target < self.level else None
        else:
            self._clear_since = None
