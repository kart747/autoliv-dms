"""Time-weighted rolling PERCLOS.

PERCLOS = time with eyes closed / time observed, over the last `window_s`
seconds. It is weighted by real elapsed time rather than by frame count, so it
stays correct when the frame rate varies (thermal throttling on the Pi, a slow
first inference, dropped frames).

Each sample's state is held until the next sample arrives. Gaps longer than
`max_gap_s` (for example, no face detected) count as unobserved time. They are
neither open nor closed, so they don't dilute the ratio. `coverage_s` reports
how much observed time the window holds, so callers can ignore PERCLOS until
the window is meaningfully full.
"""

from __future__ import annotations

from collections import deque


class Perclos:
    def __init__(self, window_s: float = 60.0, max_gap_s: float = 0.5):
        self.window_s = window_s
        self.max_gap_s = max_gap_s
        self._segs: deque[list] = deque()  # [start, end, closed]
        self._last: tuple[float, bool] | None = None
        self._closed = 0.0
        self._total = 0.0

    def update(self, t: float, closed: bool | None) -> float:
        """Add a sample at time `t` (seconds). `closed=None` means no face."""
        if self._last is not None:
            t0, c0 = self._last
            dt = t - t0
            if 0.0 < dt <= self.max_gap_s:
                self._segs.append([t0, t, c0])
                self._total += dt
                if c0:
                    self._closed += dt
        self._last = None if closed is None else (t, closed)
        self._prune(t)
        return self.value

    def _prune(self, now: float) -> None:
        start = now - self.window_s
        while self._segs and self._segs[0][1] <= start:
            s, e, c = self._segs.popleft()
            self._remove(e - s, c)
        if self._segs and self._segs[0][0] < start:
            seg = self._segs[0]
            self._remove(start - seg[0], seg[2])
            seg[0] = start
        if not self._segs:  # resync running sums to kill float drift
            self._closed = self._total = 0.0

    def _remove(self, dt: float, closed: bool) -> None:
        self._total = max(0.0, self._total - dt)
        if closed:
            self._closed = max(0.0, self._closed - dt)

    @property
    def value(self) -> float:
        return self._closed / self._total if self._total > 0 else 0.0

    @property
    def coverage_s(self) -> float:
        return self._total
