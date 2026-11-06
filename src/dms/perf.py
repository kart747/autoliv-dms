"""FPS, CPU and SoC temperature for checking headroom on the Pi.

FPS here is *processing* throughput in wall-clock time (frames handled per
second), not the camera's nominal rate. CPU is sampled at most once per
second because psutil calls aren't free.
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path

import psutil

_THERMAL = Path("/sys/class/thermal/thermal_zone0/temp")


class PerfMonitor:
    def __init__(self, fps_window_s: float = 1.0):
        self.fps_window_s = fps_window_s
        self._ticks: deque[float] = deque()
        self._proc = psutil.Process()
        self._proc.cpu_percent(None)  # prime the counters
        psutil.cpu_percent(None)
        self.n_cpu = psutil.cpu_count() or 1
        self._last_sample = 0.0
        self.fps = 0.0
        self.cpu_proc = 0.0  # this process, % of one core (can exceed 100 on multi-core)
        self.cpu_sys = 0.0  # whole system, % of all cores
        self.temp_c: float | None = None

    def tick(self) -> None:
        now = time.perf_counter()
        self._ticks.append(now)
        while self._ticks and now - self._ticks[0] > self.fps_window_s:
            self._ticks.popleft()
        if len(self._ticks) >= 2:
            span = self._ticks[-1] - self._ticks[0]
            self.fps = (len(self._ticks) - 1) / span if span > 0 else 0.0
        if now - self._last_sample >= 1.0:
            self._last_sample = now
            self.cpu_proc = self._proc.cpu_percent(None)
            self.cpu_sys = psutil.cpu_percent(None)
            self.temp_c = _read_temp()

    def summary(self) -> str:
        temp = f" | SoC {self.temp_c:.1f}C" if self.temp_c is not None else ""
        return (
            f"FPS {self.fps:5.1f} | CPU proc {self.cpu_proc:5.1f}% "
            f"(= {self.cpu_proc / self.n_cpu:4.1f}% of {self.n_cpu} cores) "
            f"| sys {self.cpu_sys:5.1f}%{temp}"
        )


def _read_temp() -> float | None:
    try:
        return int(_THERMAL.read_text().strip()) / 1000.0
    except (OSError, ValueError):
        return None
