import csv

import numpy as np

from dms.alerts import AlertLevel, AlertStatus
from dms.csvlog import FIELDS, CsvLogger
from dms.overlay import draw_hud
from dms.perf import PerfMonitor


def test_csv_logger_roundtrip(tmp_path):
    log = CsvLogger(tmp_path)
    log.write({"frame": 1, "ear": 0.25, "face": True, "level": 2, "temp_c": None})
    log.close()
    rows = list(csv.DictReader(open(log.path)))
    assert list(rows[0].keys()) == FIELDS
    assert rows[0]["ear"] == "0.2500"
    assert rows[0]["level"] == "2"
    assert rows[0]["temp_c"] == ""


def test_perf_monitor_runs():
    p = PerfMonitor()
    for _ in range(5):
        p.tick()
    assert p.fps >= 0
    assert "FPS" in p.summary()


def test_hud_draws_on_frame():
    img = np.zeros((480, 640, 3), np.uint8)
    st = AlertStatus(level=AlertLevel.CRITICAL, reasons=["microsleep 2.1s"])
    draw_hud(img, face=True, ear=0.1, mar=0.2, perclos=0.3, status=st, fps=25, infer_ms=12, ear_thr=0.22)
    assert img.any()
