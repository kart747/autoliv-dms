"""Per-frame CSV logger.

Flushes at most once a second instead of every frame, which keeps SD-card
writes on the Pi cheap.
"""

from __future__ import annotations

import csv
import time
from datetime import datetime
from pathlib import Path

FIELDS = [
    "wall_time", "t", "frame", "face",
    "ear_l", "ear_r", "ear", "mar", "eye_closed", "mouth_open",
    "perclos", "perclos_valid", "closure_s", "blinks", "long_closures",
    "yawns", "yawns_window", "no_face_s", "level", "reasons",
    "fps", "infer_ms", "cpu_proc", "cpu_sys", "temp_c",
]  # fmt: skip


class CsvLogger:
    def __init__(self, directory: str | Path, stem: str = "drowsiness"):
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.path = d / f"{stem}_{datetime.now():%Y%m%d_%H%M%S}.csv"
        self._f = open(self.path, "w", newline="")
        self._w = csv.DictWriter(self._f, fieldnames=FIELDS)
        self._w.writeheader()
        self._last_flush = time.monotonic()

    def write(self, row: dict) -> None:
        out = {}
        for k in FIELDS:
            v = row.get(k)
            out[k] = f"{v:.4f}" if isinstance(v, float) else ("" if v is None else v)
        self._w.writerow(out)
        now = time.monotonic()
        if now - self._last_flush >= 1.0:
            self._f.flush()
            self._last_flush = now

    def close(self) -> None:
        self._f.close()
