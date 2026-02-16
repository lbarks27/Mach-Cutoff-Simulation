"""Dedicated guidance configuration and JSON loader."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class GuidanceLateralConfig:
    lookahead_time_s: float = 45.0
    min_lookahead_m: float = 3_000.0
    max_lookahead_m: float = 55_000.0
    terminal_lookahead_scale: float = 0.45
    cross_track_gain: float = 4.0e-5


@dataclass(slots=True)
class GuidanceVerticalConfig:
    altitude_time_constant_s: float = 80.0
    climb_target_altitude_m: float = 16_500.0
    terminal_altitude_blend_distance_m: float = 60_000.0


@dataclass(slots=True)
class GuidanceSpeedConfig:
    target_mach: float = 1.05
    min_mach: float = 1.01
    max_mach: float = 1.35
    effective_mach_target: float = 1.03
    effective_mach_gain: float = 0.35
    speed_time_constant_s: float = 20.0


@dataclass(slots=True)
class GuidanceConstraintsConfig:
    max_bank_deg: float = 35.0
    max_pitch_deg: float = 18.0
    max_turn_rate_deg_s: float = 3.5
    max_pitch_rate_deg_s: float = 2.5
    max_longitudinal_accel_mps2: float = 4.0
    min_speed_mps: float = 220.0
    max_speed_mps: float = 490.0
    min_altitude_m: float = 0.0
    max_altitude_m: float = 20_000.0
    max_climb_rate_mps: float = 30.0
    max_descent_rate_mps: float = 35.0


@dataclass(slots=True)
class GuidanceModeSwitchConfig:
    enable_takeoff_climb: bool = True
    enable_terminal: bool = True
    takeoff_distance_m: float = 35_000.0
    climb_completion_altitude_m: float = 14_000.0
    terminal_distance_m: float = 50_000.0
    abort_cross_track_m: float = 100_000.0
    abort_recovery_cross_track_m: float = 40_000.0
    abort_min_altitude_m: float = 50.0


@dataclass(slots=True)
class GuidanceBoomOptimizerConfig:
    enabled: bool = True
    horizon_steps: int = 8
    horizon_step_s: float = 20.0
    candidate_altitude_offsets_m: list[float] = field(
        default_factory=lambda: [-1500.0, -900.0, -450.0, 0.0, 450.0, 900.0, 1500.0, 2200.0]
    )
    candidate_mach_offsets: list[float] = field(
        default_factory=lambda: [-0.06, -0.04, -0.02, 0.0, 0.02, 0.04]
    )
    effective_mach_ratio_smoothing: float = 0.25
    ground_hit_fraction_smoothing: float = 0.35
    source_cutoff_smoothing: float = 0.40
    effective_mach_margin: float = 1.015
    effective_mach_altitude_gain_per_km: float = 0.015
    ground_risk_decay: float = 0.82
    ground_risk_mach_gain: float = 1.40
    ground_risk_altitude_relief_per_km: float = 0.20
    ground_risk_track_gain: float = 0.25
    weight_ground_risk: float = 7.0
    weight_cutoff_risk: float = 10.0
    weight_tracking: float = 1.0
    weight_altitude_deviation: float = 0.6
    weight_mach_deviation: float = 2.0
    weight_terminal_altitude_bias: float = 1.2
    max_altitude_adjustment_m: float = 3_500.0
    max_altitude_step_m_per_cycle: float = 400.0
    max_mach_adjustment: float = 0.06
    max_mach_step_per_cycle: float = 0.015

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GuidanceBoomOptimizerConfig":
        if not isinstance(data, dict):
            raise ValueError("boom_avoidance.optimizer must be a JSON object")
        payload = dict(data)
        if "candidate_altitude_offsets_m" in payload:
            payload["candidate_altitude_offsets_m"] = [float(v) for v in payload["candidate_altitude_offsets_m"]]
        if "candidate_mach_offsets" in payload:
            payload["candidate_mach_offsets"] = [float(v) for v in payload["candidate_mach_offsets"]]
        return cls(**payload)


@dataclass(slots=True)
class GuidanceBoomAvoidanceConfig:
    enable_ground_hit_feedback: bool = True
    ground_hit_fraction_threshold: float = 0.05
    altitude_gain_m_per_hit_fraction: float = 8_000.0
    altitude_bias_decay_m_per_step: float = 150.0
    max_altitude_bias_m: float = 6_000.0
    mach_reduction_per_hit_fraction: float = 0.08
    source_cutoff_recovery_mach_step: float = 0.004
    optimizer: GuidanceBoomOptimizerConfig = field(default_factory=GuidanceBoomOptimizerConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GuidanceBoomAvoidanceConfig":
        if not isinstance(data, dict):
            raise ValueError("boom_avoidance must be a JSON object")
        payload = dict(data)
        optimizer_cfg = GuidanceBoomOptimizerConfig.from_dict(payload.pop("optimizer", {}))
        return cls(**payload, optimizer=optimizer_cfg)


@dataclass(slots=True)
class GuidanceConfig:
    enabled: bool = True
    integration_dt_s: float = 1.0
    max_wind_mps: float = 120.0
    lateral: GuidanceLateralConfig = field(default_factory=GuidanceLateralConfig)
    vertical: GuidanceVerticalConfig = field(default_factory=GuidanceVerticalConfig)
    speed: GuidanceSpeedConfig = field(default_factory=GuidanceSpeedConfig)
    constraints: GuidanceConstraintsConfig = field(default_factory=GuidanceConstraintsConfig)
    mode_switch: GuidanceModeSwitchConfig = field(default_factory=GuidanceModeSwitchConfig)
    boom_avoidance: GuidanceBoomAvoidanceConfig = field(default_factory=GuidanceBoomAvoidanceConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GuidanceConfig":
        if not isinstance(data, dict):
            raise ValueError("Guidance config root must be a JSON object")
        return cls(
            enabled=bool(data.get("enabled", True)),
            integration_dt_s=float(data.get("integration_dt_s", 1.0)),
            max_wind_mps=float(data.get("max_wind_mps", 120.0)),
            lateral=GuidanceLateralConfig(**data.get("lateral", {})),
            vertical=GuidanceVerticalConfig(**data.get("vertical", {})),
            speed=GuidanceSpeedConfig(**data.get("speed", {})),
            constraints=GuidanceConstraintsConfig(**data.get("constraints", {})),
            mode_switch=GuidanceModeSwitchConfig(**data.get("mode_switch", {})),
            boom_avoidance=GuidanceBoomAvoidanceConfig.from_dict(data.get("boom_avoidance", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_guidance_config(path: str | Path | None) -> GuidanceConfig:
    if path is None:
        return GuidanceConfig()

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Guidance config file not found: {config_path}")
    if config_path.suffix.lower() != ".json":
        raise ValueError("Only JSON guidance config files are supported")

    with config_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError("Guidance config root must be a JSON object")

    # Allow either a direct guidance config object or {"guidance": {...}}.
    payload = raw.get("guidance", raw)
    if not isinstance(payload, dict):
        raise ValueError("Guidance config payload must be a JSON object")
    return GuidanceConfig.from_dict(payload)
