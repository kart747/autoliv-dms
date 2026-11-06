from pathlib import Path

import pytest

from dms.config import load_config

REPO_CONFIG = Path(__file__).resolve().parents[1] / "config" / "drowsiness.toml"


def test_repo_config_loads():
    cfg = load_config(REPO_CONFIG)
    assert cfg.thresholds.ear_closed == pytest.approx(0.22)
    assert cfg.perclos.window_s == 60.0
    assert cfg.alerts.min_closure_s == 0.5


def test_defaults_without_file():
    cfg = load_config(None)
    assert cfg.camera.source == "opencv"


def test_unknown_key_rejected(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text("[thresholds]\near_close = 0.2\n")
    with pytest.raises(ValueError, match="ear_close"):
        load_config(p)


def test_inconsistent_closure_times_rejected(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text("[alerts]\nwarning_closure_s = 3.0\ncritical_closure_s = 2.0\n")
    with pytest.raises(ValueError):
        load_config(p)
