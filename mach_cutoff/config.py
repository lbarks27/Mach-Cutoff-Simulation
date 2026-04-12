"""Configuration model for the Mach cutoff simulation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class HRRRConfig:
    cache_dir: str = ".cache/hrrr"
    domain: str = "conus"
    product: str = "wrfprsf"
    use_s3fs: bool = True
    http_fallback: bool = True
    download_workers: int = 6
    local_grib_dir: str | None = None
    keep_downloaded_files: bool = True


@dataclass(slots=True)
class GridConfig:
    half_width_east_m: float = 200_000.0
    half_width_north_m: float = 200_000.0
    min_altitude_m: float = -1_000.0
    max_altitude_m: float = 30_000.0
    nx: int = 120
    ny: int = 120
    nz: int = 80
    gradient_step_m: float = 50.0
    wind_projection_mode: str = "heading"


@dataclass(slots=True)
class AircraftConfig:
    mach: float = 1.05
    constant_altitude_m: float = 16_500.0
    reference_sound_speed_mps: float = 340.29


@dataclass(slots=True)
class ShockConfig:
    emission_interval_s: float = 5.0
    rays_per_emission: int = 40
    downward_only: bool = True
    azimuth_offset_deg: float = 0.0
    direction_reference: str = "aircraft_forward"


@dataclass(slots=True)
class RaytraceConfig:
    ds_m: float = 250.0
    max_steps: int = 4_000
    adaptive: bool = True
    tol: float = 1e-3
    min_step_m: float | None = 20.0
    max_step_m: float | None = 1_500.0
    stop_on_exit: bool = True


@dataclass(slots=True)
class RuntimeConfig:
    start_time_iso: str | None = None
    end_time_iso: str | None = None
    max_emissions: int | None = None
    ray_batch_size: int = 0


@dataclass(slots=True)
class PopulationConfig:
    enabled: bool = False
    dataset_path: str | None = None
    heatmap_cell_deg: float = 0.12
    hit_radius_km: float = 25.0
    trace_half_width_km: float = 12.0
    overflight_half_width_km: float = 20.0
    max_segment_length_km: float = 220.0
    domain_pad_deg: float = 0.8
    max_grid_cells: int = 600_000


@dataclass(slots=True)
class VisualizationConfig:
    enable_matplotlib: bool = True
    enable_plotly: bool = True
    enable_pyvista: bool = True
    output_dir: str = "outputs"
    make_animation: bool = True
    write_html: bool = True
    include_atmosphere: bool = True
    map_style: str = "topographic"


@dataclass(slots=True)
class RouteOptimizationConfig:
    enabled: bool = False
    max_wall_time_s: float = 300.0
    reserve_time_for_full_fidelity_s: float = 120.0
    seed: int = 17
    batch_size: int = 4
    finalists: int = 3
    elite_pool_size: int = 6
    max_duplicate_attempts: int = 10
    min_control_waypoints: int = 9
    enable_fuel_objective: bool = True
    weight_population: float = 1.0
    weight_speed: float = 0.35
    weight_fuel: float = 0.20
    weight_populated_ground_hits: float = 8.0
    weight_total_ground_hits: float = 0.35
    weight_overflight_population: float = 0.0
    weight_overflight_area: float = 0.0
    boom_exposure_limit_people: float | None = None
    weight_boom_exposure_limit: float = 0.0
    min_cutoff_emission_fraction: float | None = None
    weight_cutoff_shortfall: float = 0.0
    unpopulated_speed_weight_bonus: float = 0.8
    min_fuel_weight_scale: float = 0.10
    artifact_mode: str = "full"
    populated_hit_people_threshold: float = 1.0
    populated_cell_people_threshold: float = 1.0
    population_guard_hit_radius_km: float = 60.0
    abort_sample_penalty_s: float = 90.0
    low_fidelity_emission_interval_scale: float = 2.0
    low_fidelity_rays_scale: float = 0.35
    low_fidelity_grid_scale: float = 0.55
    low_fidelity_step_scale: float = 1.8
    low_fidelity_max_steps_scale: float = 0.45
    low_fidelity_max_emissions: int = 96


@dataclass(slots=True)
class ExperimentConfig:
    hrrr: HRRRConfig = field(default_factory=HRRRConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    aircraft: AircraftConfig = field(default_factory=AircraftConfig)
    shock: ShockConfig = field(default_factory=ShockConfig)
    raytrace: RaytraceConfig = field(default_factory=RaytraceConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    population: PopulationConfig = field(default_factory=PopulationConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    route_optimization: RouteOptimizationConfig = field(default_factory=RouteOptimizationConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentConfig":
        return cls(
            hrrr=HRRRConfig(**data.get("hrrr", {})),
            grid=GridConfig(**data.get("grid", {})),
            aircraft=AircraftConfig(**data.get("aircraft", {})),
            shock=ShockConfig(**data.get("shock", {})),
            raytrace=RaytraceConfig(**data.get("raytrace", {})),
            runtime=RuntimeConfig(**data.get("runtime", {})),
            population=PopulationConfig(**data.get("population", {})),
            visualization=VisualizationConfig(**data.get("visualization", {})),
            route_optimization=RouteOptimizationConfig(**data.get("route_optimization", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path | None) -> ExperimentConfig:
    if path is None:
        return ExperimentConfig()

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    if config_path.suffix.lower() != ".json":
        raise ValueError("Only JSON config files are supported")

    with config_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError("Config root must be a JSON object")
    return ExperimentConfig.from_dict(raw)
