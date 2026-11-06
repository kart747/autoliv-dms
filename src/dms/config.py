"""Typed configuration loaded from a TOML file.

Unknown keys are rejected so a typo in the config file fails loudly instead of
silently falling back to a default threshold.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any


@dataclass
class CameraConfig:
    source: str = "opencv"
    device: int | str = 0
    path: str = ""
    width: int = 640
    height: int = 480
    fps: int = 30
    camera_index: int = 0
    mono: bool = False
    force_grayscale: bool = False
    clahe: bool = False
    flip_horizontal: bool = False


@dataclass
class ModelConfig:
    path: str = "models/face_landmarker.task"
    min_face_detection_confidence: float = 0.5
    min_face_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


@dataclass
class ThresholdConfig:
    ear_closed: float = 0.22
    mar_yawn: float = 0.5


@dataclass
class PerclosConfig:
    window_s: float = 60.0
    min_coverage_s: float = 20.0


@dataclass
class AlertConfig:
    min_closure_s: float = 0.5
    warning_closure_s: float = 1.0
    critical_closure_s: float = 2.0
    perclos_warning: float = 0.15
    perclos_critical: float = 0.25
    yawn_min_s: float = 1.5
    yawn_window_s: float = 300.0
    yawn_warning_count: int = 3
    no_face_warning_s: float = 3.0
    recovery_s: float = 3.0


@dataclass
class DisplayConfig:
    show: bool = True
    draw_landmarks: bool = True


@dataclass
class LoggingConfig:
    csv_dir: str = "logs"
    print_interval_s: float = 2.0


@dataclass
class Config:
    camera: CameraConfig = field(default_factory=CameraConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    perclos: PerclosConfig = field(default_factory=PerclosConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def validate(self) -> None:
        a = self.alerts
        if not 0 < a.min_closure_s <= a.warning_closure_s <= a.critical_closure_s:
            raise ValueError("need 0 < min_closure_s <= warning_closure_s <= critical_closure_s")
        if not 0 < a.perclos_warning <= a.perclos_critical <= 1:
            raise ValueError("need 0 < perclos_warning <= perclos_critical <= 1")
        if self.perclos.min_coverage_s > self.perclos.window_s:
            raise ValueError("perclos.min_coverage_s must be <= perclos.window_s")
        if self.camera.source not in {"opencv", "file", "picamera2", "rpicam"}:
            raise ValueError(f"unknown camera.source {self.camera.source!r}")


def _build(cls: type, data: dict[str, Any], section: str):
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown key(s) in [{section}]: {sorted(unknown)}")
    return cls(**data)


def load_config(path: str | Path | None) -> Config:
    """Load config from TOML; missing sections/keys take the dataclass defaults."""
    raw: dict[str, Any] = {}
    if path is not None:
        with open(path, "rb") as f:
            raw = tomllib.load(f)
    sections = {f.name: f.type for f in fields(Config)}
    unknown = set(raw) - set(sections)
    if unknown:
        raise ValueError(f"unknown section(s): {sorted(unknown)}")
    defaults = Config()
    kwargs = {
        name: _build(type(getattr(defaults, name)), raw.get(name, {}), name)
        for name in sections
    }
    cfg = Config(**kwargs)
    cfg.validate()
    return cfg
