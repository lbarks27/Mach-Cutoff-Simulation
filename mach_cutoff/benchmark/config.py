"""Benchmark manifest model and loader."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class GPWConfig:
    product: str = "gpwv4_popcount_r11"
    year: int = 2020
    auto_download: bool = True
    raw_cache_dir: str = ".cache/population/raw"
    prepared_cache_dir: str = ".cache/population/prepared"
    prepared_cell_deg: float = 0.05
    conus_bounds: list[float] = field(default_factory=lambda: [-125.0, 24.0, -66.0, 50.0])
    download_url: str | None = None


@dataclass(slots=True)
class CorridorConfig:
    corridor_id: str
    origin_lat_deg: float
    origin_lon_deg: float
    destination_lat_deg: float
    destination_lon_deg: float
    origin_name: str = ""
    destination_name: str = ""
    notes: str = ""


@dataclass(slots=True)
class RouteClassConfig:
    route_class_id: str
    config_overrides: dict[str, Any] = field(default_factory=dict)
    guidance_overrides: dict[str, Any] = field(default_factory=dict)
    notes: str = ""


@dataclass(slots=True)
class SensitivityProfileConfig:
    profile_id: str
    config_overrides: dict[str, Any] = field(default_factory=dict)
    guidance_overrides: dict[str, Any] = field(default_factory=dict)
    route_weight_multipliers: dict[str, float] = field(default_factory=dict)
    notes: str = ""


@dataclass(slots=True)
class ResearchObjectiveConfig:
    baseline_route_class_id: str = "fastest"
    boom_exposure_limit_people: float | None = None
    max_time_penalty_pct: float | None = None
    min_cutoff_emission_fraction: float | None = None
    max_abort_samples: int = 0
    max_distance_to_destination_m: float = 5_000.0


@dataclass(slots=True)
class BenchmarkConfig:
    name: str
    output_root: str
    base_config_path: str
    base_guidance_config_path: str
    population_cache_dir: str = ".cache/population"
    gpw: GPWConfig = field(default_factory=GPWConfig)
    route_classes: dict[str, RouteClassConfig] = field(default_factory=dict)
    corridors: dict[str, CorridorConfig] = field(default_factory=dict)
    timestamps_utc: list[str] = field(default_factory=list)
    anchor_scenario_ids: list[str] = field(default_factory=list)
    sensitivity_profiles: dict[str, SensitivityProfileConfig] = field(default_factory=dict)
    research_objectives: ResearchObjectiveConfig = field(default_factory=ResearchObjectiveConfig)
    jobs: int = 1
    resume: bool = True
    retain_anchor_npz: bool = True

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "BenchmarkConfig":
        if not isinstance(raw, dict):
            raise ValueError("Benchmark config root must be a JSON object")

        gpw = GPWConfig(**raw.get("gpw", {}))
        research_objectives = ResearchObjectiveConfig(**raw.get("research_objectives", {}))

        route_classes: dict[str, RouteClassConfig] = {}
        for key, value in raw.get("route_classes", {}).items():
            payload = dict(value or {})
            route_classes[key] = RouteClassConfig(
                route_class_id=key,
                config_overrides=dict(payload.get("config_overrides", {})),
                guidance_overrides=dict(payload.get("guidance_overrides", {})),
                notes=str(payload.get("notes", "")),
            )

        corridors: dict[str, CorridorConfig] = {}
        for key, value in raw.get("corridors", {}).items():
            payload = dict(value or {})
            corridors[key] = CorridorConfig(
                corridor_id=key,
                origin_lat_deg=float(payload["origin_lat_deg"]),
                origin_lon_deg=float(payload["origin_lon_deg"]),
                destination_lat_deg=float(payload["destination_lat_deg"]),
                destination_lon_deg=float(payload["destination_lon_deg"]),
                origin_name=str(payload.get("origin_name", "")),
                destination_name=str(payload.get("destination_name", "")),
                notes=str(payload.get("notes", "")),
            )

        sensitivity_profiles: dict[str, SensitivityProfileConfig] = {}
        for key, value in raw.get("sensitivity_profiles", {}).items():
            payload = dict(value or {})
            multipliers = {
                str(k): float(v)
                for k, v in dict(payload.get("route_weight_multipliers", {})).items()
            }
            sensitivity_profiles[key] = SensitivityProfileConfig(
                profile_id=key,
                config_overrides=dict(payload.get("config_overrides", {})),
                guidance_overrides=dict(payload.get("guidance_overrides", {})),
                route_weight_multipliers=multipliers,
                notes=str(payload.get("notes", "")),
            )

        return cls(
            name=str(raw["name"]),
            output_root=str(raw.get("output_root", f"benchmarks/{raw['name']}")),
            base_config_path=str(raw["base_config_path"]),
            base_guidance_config_path=str(raw["base_guidance_config_path"]),
            population_cache_dir=str(raw.get("population_cache_dir", ".cache/population")),
            gpw=gpw,
            route_classes=route_classes,
            corridors=corridors,
            timestamps_utc=[str(v) for v in raw.get("timestamps_utc", [])],
            anchor_scenario_ids=[str(v) for v in raw.get("anchor_scenario_ids", [])],
            sensitivity_profiles=sensitivity_profiles,
            research_objectives=research_objectives,
            jobs=max(1, int(raw.get("jobs", 1))),
            resume=bool(raw.get("resume", True)),
            retain_anchor_npz=bool(raw.get("retain_anchor_npz", True)),
        )


def load_benchmark_config(path: str | Path) -> BenchmarkConfig:
    cfg_path = Path(path).expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Benchmark config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    cfg = BenchmarkConfig.from_dict(raw)

    root = cfg_path.parent
    cfg.output_root = str((root / cfg.output_root).resolve()) if not Path(cfg.output_root).is_absolute() else cfg.output_root
    cfg.base_config_path = (
        str((root / cfg.base_config_path).resolve())
        if not Path(cfg.base_config_path).is_absolute()
        else cfg.base_config_path
    )
    cfg.base_guidance_config_path = (
        str((root / cfg.base_guidance_config_path).resolve())
        if not Path(cfg.base_guidance_config_path).is_absolute()
        else cfg.base_guidance_config_path
    )

    if not Path(cfg.gpw.raw_cache_dir).is_absolute():
        cfg.gpw.raw_cache_dir = str((root / cfg.gpw.raw_cache_dir).resolve())
    if not Path(cfg.gpw.prepared_cache_dir).is_absolute():
        cfg.gpw.prepared_cache_dir = str((root / cfg.gpw.prepared_cache_dir).resolve())
    if not Path(cfg.population_cache_dir).is_absolute():
        cfg.population_cache_dir = str((root / cfg.population_cache_dir).resolve())

    return cfg
