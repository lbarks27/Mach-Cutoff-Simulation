"""Iterative route optimization with low-fidelity reruns and full-fidelity promotion."""

from __future__ import annotations

import copy
import csv
import json
import time
from datetime import datetime, timezone
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..config import ExperimentConfig
from ..core.geodesy import geodetic_to_ecef, normalize_lon_deg
from ..flight.waypoints import FlightPath, Waypoint
from ..guidance_config import GuidanceConfig
from .engine import MachCutoffSimulator
from .outputs import SimulationResult

_M_PER_DEG = 111_320.0
_EPS = 1e-9


@dataclass(slots=True)
class RouteOptimizationSettings:
    max_wall_time_s: float = 300.0
    reserve_time_for_full_fidelity_s: float = 120.0
    reserve_time_for_mid_fidelity_s: float = 60.0
    seed: int = 17
    batch_size: int = 4
    semifinalists: int = 6
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
    weight_route_stretch: float = 0.0
    weight_route_heading_change: float = 0.0
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
    mid_fidelity_emission_interval_scale: float = 1.35
    mid_fidelity_rays_scale: float = 0.65
    mid_fidelity_grid_scale: float = 0.8
    mid_fidelity_step_scale: float = 1.25
    mid_fidelity_max_steps_scale: float = 0.75
    mid_fidelity_max_emissions: int = 160


@dataclass(slots=True)
class ObjectiveMetrics:
    exposure_people: float
    overflight_population: float
    ground_hit_count: int
    populated_ground_hit_count: int
    cutoff_emission_fraction: float
    source_cutoff_emission_fraction: float
    populated_hit_population: float
    populated_exposed_area_km2: float
    overflight_area_km2: float
    elapsed_time_s: float
    time_proxy_s: float
    fuel_proxy: float
    mean_ground_speed_mps: float
    route_distance_km: float
    route_excess_ratio: float
    route_heading_change_deg: float
    distance_to_destination_m: float
    abort_samples: int


@dataclass(slots=True)
class ObjectiveNormalizer:
    exposure_scale: float
    overflight_population_scale: float
    ground_hit_scale: float
    populated_ground_hit_scale: float
    populated_hit_population_scale: float
    populated_exposed_area_scale: float
    overflight_area_scale: float
    time_scale: float
    fuel_scale: float
    speed_scale: float
    route_excess_scale: float
    route_heading_change_scale: float


@dataclass(slots=True)
class RouteOptimizationOutcome:
    result: SimulationResult
    best_candidate_id: str
    optimized_waypoints_path: Path
    report_path: Path
    iteration_csv_path: Path


@dataclass(slots=True)
class _Candidate:
    candidate_id: str
    generation: int
    parent_candidate_id: str | None
    waypoints: list[Waypoint]
    guidance_config: GuidanceConfig
    phase_mutation_counts: dict[str, int]
    mutated_waypoint_count: int
    guidance_mutation: dict[str, float]
    low_metrics: ObjectiveMetrics | None = None
    low_score: float | None = None
    mid_metrics: ObjectiveMetrics | None = None
    mid_score: float | None = None
    full_metrics: ObjectiveMetrics | None = None
    full_score: float | None = None


def _clone_guidance_config(cfg: GuidanceConfig) -> GuidanceConfig:
    return GuidanceConfig.from_dict(copy.deepcopy(cfg.to_dict()))


def _clone_experiment_config(cfg: ExperimentConfig) -> ExperimentConfig:
    return ExperimentConfig.from_dict(copy.deepcopy(cfg.to_dict()))


def _clip(value: float, lo: float, hi: float) -> float:
    return float(np.clip(float(value), float(lo), float(hi)))


def _wrap_lon_deg(lon_deg: float) -> float:
    wrapped = normalize_lon_deg(np.asarray([lon_deg], dtype=np.float64))
    return float(wrapped.reshape(-1)[0])


def _haversine_km(lat0_deg: np.ndarray, lon0_deg: np.ndarray, lat1_deg: np.ndarray, lon1_deg: np.ndarray) -> np.ndarray:
    lat0 = np.deg2rad(np.asarray(lat0_deg, dtype=np.float64))
    lon0 = np.deg2rad(np.asarray(lon0_deg, dtype=np.float64))
    lat1 = np.deg2rad(np.asarray(lat1_deg, dtype=np.float64))
    lon1 = np.deg2rad(np.asarray(lon1_deg, dtype=np.float64))
    dlat = lat1 - lat0
    dlon = lon1 - lon0
    a = np.sin(dlat * 0.5) ** 2 + np.cos(lat0) * np.cos(lat1) * np.sin(dlon * 0.5) ** 2
    return 6371.0 * (2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.clip(1.0 - a, 0.0, 1.0))))


def _compute_cumulative_lengths_m(waypoints: list[Waypoint]) -> tuple[np.ndarray, float]:
    ecef = np.asarray([geodetic_to_ecef(wp.lat_deg, wp.lon_deg, wp.alt_m) for wp in waypoints], dtype=np.float64)
    if ecef.shape[0] < 2:
        return np.asarray([0.0], dtype=np.float64), 0.0
    segment = np.linalg.norm(np.diff(ecef, axis=0), axis=1)
    cumulative = np.concatenate([np.asarray([0.0], dtype=np.float64), np.cumsum(segment)])
    return cumulative, float(cumulative[-1])


def _route_shape_metrics(waypoints: list[Waypoint]) -> tuple[float, float]:
    if len(waypoints) < 2:
        return 0.0, 0.0

    lat = np.asarray([wp.lat_deg for wp in waypoints], dtype=np.float64)
    lon = normalize_lon_deg(np.asarray([wp.lon_deg for wp in waypoints], dtype=np.float64))
    seg_km = _haversine_km(lat[:-1], lon[:-1], lat[1:], lon[1:])
    route_distance_km = float(np.sum(seg_km))
    direct_distance_km = float(_haversine_km(lat[0], lon[0], lat[-1], lon[-1]))
    route_excess_ratio = max(0.0, route_distance_km / max(direct_distance_km, 1e-6) - 1.0)

    heading_change_deg = 0.0
    for idx in range(1, len(waypoints) - 1):
        lat_mid = float(waypoints[idx].lat_deg)
        meters_per_lon = max(1.0, _M_PER_DEG * float(np.cos(np.deg2rad(lat_mid))))
        in_e = float((waypoints[idx].lon_deg - waypoints[idx - 1].lon_deg) * meters_per_lon)
        in_n = float((waypoints[idx].lat_deg - waypoints[idx - 1].lat_deg) * _M_PER_DEG)
        out_e = float((waypoints[idx + 1].lon_deg - waypoints[idx].lon_deg) * meters_per_lon)
        out_n = float((waypoints[idx + 1].lat_deg - waypoints[idx].lat_deg) * _M_PER_DEG)
        in_norm = float(np.hypot(in_e, in_n))
        out_norm = float(np.hypot(out_e, out_n))
        if in_norm <= 1.0 or out_norm <= 1.0:
            continue
        in_heading = float(np.arctan2(in_e, in_n))
        out_heading = float(np.arctan2(out_e, out_n))
        delta = float((out_heading - in_heading + np.pi) % (2.0 * np.pi) - np.pi)
        heading_change_deg += abs(float(np.rad2deg(delta)))

    return float(route_excess_ratio), float(heading_change_deg)


def _population_cell_areas_km2(lat_edges_deg: np.ndarray, lon_edges_deg: np.ndarray) -> np.ndarray:
    lat_edges = np.asarray(lat_edges_deg, dtype=np.float64)
    lon_edges = np.asarray(lon_edges_deg, dtype=np.float64)
    if lat_edges.size < 2 or lon_edges.size < 2:
        return np.zeros((0, 0), dtype=np.float64)
    lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    dlat_km = np.abs(np.diff(lat_edges)) * 111.32
    dlon_km = np.abs(np.diff(lon_edges))[None, :] * (111.32 * np.cos(np.deg2rad(lat_centers)))[:, None]
    return np.abs(dlat_km[:, None] * dlon_km)


def _expand_waypoints_for_optimization(base_waypoints: list[Waypoint], min_points: int) -> list[Waypoint]:
    target_points = max(2, int(min_points))
    if len(base_waypoints) >= target_points:
        return list(base_waypoints)
    if len(base_waypoints) < 2:
        return list(base_waypoints)

    cumulative, total_m = _compute_cumulative_lengths_m(base_waypoints)
    if total_m <= _EPS:
        return list(base_waypoints)

    times = np.asarray([wp.time_utc.timestamp() for wp in base_waypoints], dtype=np.float64)
    expanded: list[Waypoint] = []
    for idx in range(target_points):
        frac = float(idx / max(target_points - 1, 1))
        target_s = frac * total_m
        seg_idx = int(np.searchsorted(cumulative, target_s, side="right") - 1)
        seg_idx = max(0, min(seg_idx, len(base_waypoints) - 2))

        s0 = float(cumulative[seg_idx])
        s1 = float(cumulative[seg_idx + 1])
        seg_frac = 0.0 if s1 <= s0 else float(np.clip((target_s - s0) / (s1 - s0), 0.0, 1.0))

        wp0 = base_waypoints[seg_idx]
        wp1 = base_waypoints[seg_idx + 1]
        dlon = float(normalize_lon_deg(np.asarray([wp1.lon_deg - wp0.lon_deg], dtype=np.float64)).reshape(-1)[0])
        lon = _wrap_lon_deg(float(wp0.lon_deg) + seg_frac * dlon)
        lat = float(wp0.lat_deg + seg_frac * (wp1.lat_deg - wp0.lat_deg))
        alt = float(wp0.alt_m + seg_frac * (wp1.alt_m - wp0.alt_m))
        t = float(times[seg_idx] + seg_frac * (times[seg_idx + 1] - times[seg_idx]))
        time_utc = datetime.fromtimestamp(t, tz=timezone.utc)
        expanded.append(Waypoint(lat_deg=lat, lon_deg=lon, alt_m=alt, time_utc=time_utc))

    expanded[0] = base_waypoints[0]
    expanded[-1] = base_waypoints[-1]
    return expanded


def _phase_for_waypoint(waypoint: Waypoint, along_m: float, total_m: float, guidance_cfg: GuidanceConfig) -> str:
    mode_cfg = guidance_cfg.mode_switch
    remaining_m = max(0.0, total_m - along_m)
    in_takeoff = mode_cfg.enable_takeoff_climb and (
        along_m <= float(mode_cfg.takeoff_distance_m) or waypoint.alt_m < float(mode_cfg.climb_completion_altitude_m)
    )
    if in_takeoff:
        return "takeoff_climb"
    if mode_cfg.enable_terminal and remaining_m <= float(mode_cfg.terminal_distance_m):
        return "terminal"
    return "enroute"


def _phase_mutation_params(phase: str, total_m: float) -> tuple[float, float, float]:
    route_scale_m = max(float(total_m), 1.0)
    if phase == "takeoff_climb":
        return (
            _clip(0.015 * route_scale_m, 2_500.0, 15_000.0),
            450.0,
            0.45,
        )
    if phase == "terminal":
        return (
            _clip(0.040 * route_scale_m, 6_500.0, 40_000.0),
            700.0,
            0.60,
        )
    return (
        _clip(0.100 * route_scale_m, 20_000.0, 120_000.0),
        1_600.0,
        0.82,
    )


def _mutate_waypoints(
    base_waypoints: list[Waypoint],
    guidance_cfg: GuidanceConfig,
    rng: np.random.Generator,
) -> tuple[list[Waypoint], dict[str, int], int]:
    if len(base_waypoints) <= 2:
        return list(base_waypoints), {"takeoff_climb": 0, "enroute": 0, "terminal": 0, "abort_failsafe": 0}, 0

    alt_lo = float(guidance_cfg.constraints.min_altitude_m)
    alt_hi = float(guidance_cfg.constraints.max_altitude_m)
    cumulative_m, total_m = _compute_cumulative_lengths_m(base_waypoints)
    waypoints = list(base_waypoints)

    phase_counts = {"takeoff_climb": 0, "enroute": 0, "terminal": 0, "abort_failsafe": 0}
    mutated_count = 0
    interior_indices = list(range(1, len(base_waypoints) - 1))

    for idx in interior_indices:
        wp = base_waypoints[idx]
        phase = _phase_for_waypoint(wp, float(cumulative_m[idx]), total_m, guidance_cfg)
        lateral_max_m, altitude_max_m, mutation_probability = _phase_mutation_params(phase, total_m)
        if float(rng.random()) > mutation_probability:
            continue

        prev_wp = base_waypoints[idx - 1]
        next_wp = base_waypoints[idx + 1]
        meters_per_lon = max(1.0, _M_PER_DEG * float(np.cos(np.deg2rad(wp.lat_deg))))
        east_track_m = (next_wp.lon_deg - prev_wp.lon_deg) * meters_per_lon
        north_track_m = (next_wp.lat_deg - prev_wp.lat_deg) * _M_PER_DEG
        norm_track = float(np.hypot(east_track_m, north_track_m))
        if norm_track <= 1.0:
            east_track_m = 1.0
            north_track_m = 0.0
            norm_track = 1.0
        east_track_m /= norm_track
        north_track_m /= norm_track
        left_east = -north_track_m
        left_north = east_track_m

        lateral_offset_m = _clip(float(rng.normal(0.0, lateral_max_m * 0.35)), -lateral_max_m, lateral_max_m)
        east_offset_m = left_east * lateral_offset_m
        north_offset_m = left_north * lateral_offset_m

        new_lat = _clip(wp.lat_deg + north_offset_m / _M_PER_DEG, -89.8, 89.8)
        new_meters_per_lon = max(1.0, _M_PER_DEG * float(np.cos(np.deg2rad(new_lat))))
        new_lon = _wrap_lon_deg(wp.lon_deg + east_offset_m / new_meters_per_lon)

        altitude_delta_m = _clip(float(rng.normal(0.0, altitude_max_m * 0.35)), -altitude_max_m, altitude_max_m)
        new_alt = _clip(wp.alt_m + altitude_delta_m, alt_lo, alt_hi)

        waypoints[idx] = Waypoint(lat_deg=float(new_lat), lon_deg=float(new_lon), alt_m=float(new_alt), time_utc=wp.time_utc)
        phase_counts[phase] += 1
        mutated_count += 1

    if mutated_count == 0 and interior_indices:
        idx = int(interior_indices[int(rng.integers(0, len(interior_indices)))])
        wp = base_waypoints[idx]
        phase = _phase_for_waypoint(wp, float(cumulative_m[idx]), total_m, guidance_cfg)
        lateral_max_m, altitude_max_m, _ = _phase_mutation_params(phase, total_m)
        new_lat = _clip(wp.lat_deg + float(rng.normal(0.0, lateral_max_m * 0.12)) / _M_PER_DEG, -89.8, 89.8)
        meters_per_lon = max(1.0, _M_PER_DEG * float(np.cos(np.deg2rad(new_lat))))
        new_lon = _wrap_lon_deg(wp.lon_deg + float(rng.normal(0.0, lateral_max_m * 0.12)) / meters_per_lon)
        new_alt = _clip(wp.alt_m + float(rng.normal(0.0, altitude_max_m * 0.12)), alt_lo, alt_hi)
        waypoints[idx] = Waypoint(lat_deg=float(new_lat), lon_deg=float(new_lon), alt_m=float(new_alt), time_utc=wp.time_utc)
        phase_counts[phase] += 1
        mutated_count = 1

    return waypoints, phase_counts, mutated_count


def _mutate_guidance(
    base_guidance: GuidanceConfig,
    rng: np.random.Generator,
    settings: RouteOptimizationSettings,
) -> tuple[GuidanceConfig, dict[str, float]]:
    cfg = _clone_guidance_config(base_guidance)
    mutation: dict[str, float] = {}
    if not cfg.enabled:
        return cfg, mutation

    min_mach = float(cfg.speed.min_mach)
    max_mach = float(cfg.speed.max_mach)
    current_target = float(cfg.speed.target_mach)
    slower_bias = 0.0
    if settings.min_cutoff_emission_fraction is not None:
        slower_bias += 0.05
    slower_bias += 0.012 * float(min(settings.weight_total_ground_hits, 4.0))
    target_delta = float(rng.normal(-slower_bias, 0.05))
    if float(rng.random()) < 0.20:
        target_delta += float(rng.normal(-0.5 * slower_bias, 0.08))
    target_delta = _clip(target_delta, -0.20, 0.12)
    new_target = _clip(current_target + target_delta, min_mach, max_mach)
    cfg.speed.target_mach = float(new_target)
    mutation["target_mach_delta"] = float(new_target - current_target)

    effective_target = float(cfg.speed.effective_mach_target)
    effective_delta = _clip(float(rng.normal(-0.35 * slower_bias, 0.03)), -0.10, 0.08)
    new_effective_target = _clip(effective_target + effective_delta, 0.95, new_target)
    cfg.speed.effective_mach_target = float(new_effective_target)
    mutation["effective_mach_target_delta"] = float(new_effective_target - effective_target)
    return cfg, mutation


def _estimate_required_low_fidelity_emissions(flight_path: FlightPath, emission_interval_s: float) -> int:
    route_duration_s = max(60.0, float(flight_path.duration_s))
    interval_s = max(float(emission_interval_s), 1.0)
    return int(np.ceil(route_duration_s / interval_s)) + 3


def _compute_metrics(
    result: SimulationResult,
    flight_path: FlightPath,
    cfg: ExperimentConfig,
    settings: RouteOptimizationSettings,
) -> ObjectiveMetrics:
    route_excess_ratio, route_heading_change_deg = _route_shape_metrics(list(flight_path.waypoints))
    ground_hit_count = int(sum(1 for emission in result.emissions for ray in emission.rays if ray.ground_hit))
    exposure_people = 0.0
    overflight_population = 0.0
    populated_ground_hit_count = 0
    populated_hit_population = 0.0
    populated_exposed_area_km2 = 0.0
    overflight_area_km2 = 0.0
    num_emissions = max(1, len(result.emissions))
    cutoff_emission_fraction = float(sum(1 for emission in result.emissions if emission.mach_cutoff) / num_emissions)
    source_cutoff_emission_fraction = float(
        sum(1 for emission in result.emissions if emission.source_mach_cutoff) / num_emissions
    )

    if result.population_impact is not None:
        pop = result.population_impact
        exposure_people = float(pop.total_exposed_population)
        overflight_population = float(pop.total_overflight_population)
        hit_population = np.asarray(pop.hit_population_within_radius, dtype=np.float64)
        if hit_population.size:
            populated_mask = hit_population >= float(max(settings.populated_hit_people_threshold, 0.0))
            populated_ground_hit_count = int(np.sum(populated_mask))
            populated_hit_population = float(np.sum(hit_population[populated_mask]))

        heatmap = np.asarray(pop.heatmap_population, dtype=np.float64)
        exposed_mask = np.asarray(pop.exposed_cell_mask, dtype=bool)
        cell_areas_km2 = _population_cell_areas_km2(pop.heatmap_lat_edges_deg, pop.heatmap_lon_edges_deg)
        if (
            heatmap.ndim == 2
            and exposed_mask.shape == heatmap.shape
            and cell_areas_km2.shape == heatmap.shape
        ):
            populated_cells = heatmap >= float(max(settings.populated_cell_people_threshold, 0.0))
            populated_exposed_area_km2 = float(np.sum(cell_areas_km2[exposed_mask & populated_cells]))
            overflight_mask = np.asarray(pop.overflight_cell_mask, dtype=bool)
            if overflight_mask.shape == heatmap.shape:
                overflight_area_km2 = float(np.sum(cell_areas_km2[overflight_mask]))
    else:
        # Without population data, treat any ground hit as populated risk.
        exposure_people = float(ground_hit_count)
        overflight_population = float(ground_hit_count)
        populated_ground_hit_count = int(ground_hit_count)
        populated_hit_population = float(ground_hit_count)
        overflight_area_km2 = float(ground_hit_count)

    if len(result.emissions) < 2:
        mean_speed_mps = max(1.0, float(cfg.aircraft.mach) * float(cfg.aircraft.reference_sound_speed_mps))
        route_distance_km = float(flight_path.total_length_m / 1000.0)
        distance_to_destination_m = float(flight_path.total_length_m)
        abort_samples = 0
        time_proxy_s = float(flight_path.total_length_m / max(mean_speed_mps, 1.0))
        fuel_proxy = time_proxy_s if settings.enable_fuel_objective else 0.0
        return ObjectiveMetrics(
            exposure_people=exposure_people,
            overflight_population=overflight_population,
            ground_hit_count=int(ground_hit_count),
            populated_ground_hit_count=int(populated_ground_hit_count),
            cutoff_emission_fraction=float(cutoff_emission_fraction),
            source_cutoff_emission_fraction=float(source_cutoff_emission_fraction),
            populated_hit_population=float(populated_hit_population),
            populated_exposed_area_km2=float(populated_exposed_area_km2),
            overflight_area_km2=float(overflight_area_km2),
            elapsed_time_s=float(time_proxy_s),
            time_proxy_s=float(time_proxy_s),
            fuel_proxy=float(fuel_proxy),
            mean_ground_speed_mps=float(mean_speed_mps),
            route_distance_km=float(route_distance_km),
            route_excess_ratio=float(route_excess_ratio),
            route_heading_change_deg=float(route_heading_change_deg),
            distance_to_destination_m=float(distance_to_destination_m),
            abort_samples=abort_samples,
        )

    lat = np.asarray([e.aircraft_lat_deg for e in result.emissions], dtype=np.float64)
    lon = normalize_lon_deg(np.asarray([e.aircraft_lon_deg for e in result.emissions], dtype=np.float64))
    alt = np.asarray([e.aircraft_alt_m for e in result.emissions], dtype=np.float64)
    times = np.asarray([e.emission_time_utc.timestamp() for e in result.emissions], dtype=np.float64)
    dt = np.diff(times)
    valid = dt > _EPS
    if not np.any(valid):
        mean_speed_mps = max(1.0, float(cfg.aircraft.mach) * float(cfg.aircraft.reference_sound_speed_mps))
        route_distance_km = float(flight_path.total_length_m / 1000.0)
        distance_to_destination_m = 0.0
        abort_samples = 0
        time_proxy_s = float(flight_path.total_length_m / max(mean_speed_mps, 1.0))
        fuel_proxy = time_proxy_s if settings.enable_fuel_objective else 0.0
        return ObjectiveMetrics(
            exposure_people=exposure_people,
            overflight_population=overflight_population,
            ground_hit_count=int(ground_hit_count),
            populated_ground_hit_count=int(populated_ground_hit_count),
            cutoff_emission_fraction=float(cutoff_emission_fraction),
            source_cutoff_emission_fraction=float(source_cutoff_emission_fraction),
            populated_hit_population=float(populated_hit_population),
            populated_exposed_area_km2=float(populated_exposed_area_km2),
            overflight_area_km2=float(overflight_area_km2),
            elapsed_time_s=float(time_proxy_s),
            time_proxy_s=float(time_proxy_s),
            fuel_proxy=float(fuel_proxy),
            mean_ground_speed_mps=float(mean_speed_mps),
            route_distance_km=float(route_distance_km),
            route_excess_ratio=float(route_excess_ratio),
            route_heading_change_deg=float(route_heading_change_deg),
            distance_to_destination_m=float(distance_to_destination_m),
            abort_samples=abort_samples,
        )

    ground_km = _haversine_km(lat[:-1], lon[:-1], lat[1:], lon[1:])
    dalt = np.diff(alt)
    segment_distance_m = np.sqrt((ground_km * 1000.0) ** 2 + dalt**2)
    dt_valid = dt[valid]
    segment_distance_valid = segment_distance_m[valid]
    total_time_s = float(np.sum(dt_valid))
    total_distance_m = float(np.sum(segment_distance_valid))
    mean_speed_mps = total_distance_m / max(total_time_s, 1e-6)
    route_distance_km = total_distance_m / 1000.0

    distance_to_destination_m = 0.0
    abort_samples = 0
    for emission in result.emissions:
        if emission.guidance is None:
            continue
        if emission.guidance.mode == "abort_failsafe":
            abort_samples += 1
        distance_to_destination_m = float(emission.guidance.distance_to_destination_m)

    path_time_proxy_s = float(flight_path.total_length_m / max(mean_speed_mps, 1.0))
    remaining_time_proxy_s = float(distance_to_destination_m / max(mean_speed_mps, 1.0))
    abort_penalty = float(abort_samples * max(settings.abort_sample_penalty_s, 0.0))
    time_proxy_s = path_time_proxy_s + remaining_time_proxy_s + abort_penalty

    if settings.enable_fuel_objective:
        speed_mid = segment_distance_valid / np.maximum(dt_valid, 1e-6)
        mach_mid = speed_mid / max(float(cfg.aircraft.reference_sound_speed_mps), 1e-6)
        climb_rate = np.abs(dalt[valid] / np.maximum(dt_valid, 1e-6))
        alt_mid = 0.5 * (alt[:-1][valid] + alt[1:][valid])
        mach_penalty = np.maximum(mach_mid - 1.0, 0.0) ** 2
        low_altitude_penalty = np.maximum(0.0, (11_000.0 - alt_mid) / 1000.0)
        fuel_rate_proxy = 1.0 + 0.65 * mach_penalty + 0.018 * climb_rate + 0.05 * low_altitude_penalty
        fuel_proxy = float(np.sum(fuel_rate_proxy * dt_valid))
    else:
        fuel_proxy = 0.0

    return ObjectiveMetrics(
        exposure_people=float(exposure_people),
        overflight_population=float(overflight_population),
        ground_hit_count=int(ground_hit_count),
        populated_ground_hit_count=int(populated_ground_hit_count),
        cutoff_emission_fraction=float(cutoff_emission_fraction),
        source_cutoff_emission_fraction=float(source_cutoff_emission_fraction),
        populated_hit_population=float(populated_hit_population),
        populated_exposed_area_km2=float(populated_exposed_area_km2),
        overflight_area_km2=float(overflight_area_km2),
        elapsed_time_s=float(total_time_s),
        time_proxy_s=float(time_proxy_s),
        fuel_proxy=float(fuel_proxy),
        mean_ground_speed_mps=float(mean_speed_mps),
        route_distance_km=float(route_distance_km),
        route_excess_ratio=float(route_excess_ratio),
        route_heading_change_deg=float(route_heading_change_deg),
        distance_to_destination_m=float(distance_to_destination_m),
        abort_samples=int(abort_samples),
    )


def _normalizer_from_baseline(baseline: ObjectiveMetrics) -> ObjectiveNormalizer:
    return ObjectiveNormalizer(
        exposure_scale=max(1.0, float(baseline.exposure_people)),
        overflight_population_scale=max(1.0, float(baseline.overflight_population)),
        ground_hit_scale=max(1.0, float(baseline.ground_hit_count)),
        populated_ground_hit_scale=max(1.0, float(baseline.populated_ground_hit_count)),
        populated_hit_population_scale=max(1.0, float(baseline.populated_hit_population)),
        populated_exposed_area_scale=max(1.0, float(baseline.populated_exposed_area_km2)),
        overflight_area_scale=max(1.0, float(baseline.overflight_area_km2)),
        time_scale=max(1.0, float(baseline.time_proxy_s)),
        fuel_scale=max(1.0, float(baseline.fuel_proxy)),
        speed_scale=max(1.0, float(baseline.mean_ground_speed_mps)),
        route_excess_scale=max(0.01, float(baseline.route_excess_ratio), 0.01),
        route_heading_change_scale=max(1.0, float(baseline.route_heading_change_deg), 1.0),
    )


def _score_metrics(metrics: ObjectiveMetrics, norm: ObjectiveNormalizer, settings: RouteOptimizationSettings) -> float:
    pop_norm = float(metrics.exposure_people / norm.exposure_scale)
    overflight_pop_norm = float(metrics.overflight_population / norm.overflight_population_scale)
    total_hits_norm = float(metrics.ground_hit_count / norm.ground_hit_scale)
    populated_hits_norm = float(metrics.populated_ground_hit_count / norm.populated_ground_hit_scale)
    populated_hit_pop_norm = float(metrics.populated_hit_population / norm.populated_hit_population_scale)
    populated_area_norm = float(metrics.populated_exposed_area_km2 / norm.populated_exposed_area_scale)
    overflight_area_norm = float(metrics.overflight_area_km2 / norm.overflight_area_scale)
    route_excess_norm = float(metrics.route_excess_ratio / norm.route_excess_scale)
    route_heading_change_norm = float(metrics.route_heading_change_deg / norm.route_heading_change_scale)

    pop_term = float(settings.weight_population) * pop_norm
    overflight_pop_term = float(settings.weight_overflight_population) * overflight_pop_norm
    overflight_area_term = float(settings.weight_overflight_area) * overflight_area_norm
    route_stretch_term = float(settings.weight_route_stretch) * route_excess_norm
    route_heading_term = float(settings.weight_route_heading_change) * route_heading_change_norm
    populated_hit_term = float(settings.weight_populated_ground_hits) * (
        populated_hits_norm + 0.5 * populated_hit_pop_norm
    )
    populated_area_term = 0.5 * float(settings.weight_populated_ground_hits) * populated_area_norm
    total_ground_hit_term = float(settings.weight_total_ground_hits) * total_hits_norm

    populated_guard_term = 0.0
    if metrics.populated_ground_hit_count > 0 or metrics.populated_hit_population > 0.0:
        populated_guard_term = 5.0 * float(settings.weight_populated_ground_hits) * (
            1.0 + populated_hits_norm + populated_hit_pop_norm
        )

    boom_exposure_limit_term = 0.0
    if settings.boom_exposure_limit_people is not None and float(settings.weight_boom_exposure_limit) > 0.0:
        exposure_limit_people = max(1.0, float(settings.boom_exposure_limit_people))
        exposure_violation_people = max(0.0, float(metrics.exposure_people) - exposure_limit_people)
        if exposure_violation_people > 0.0:
            violation_ratio = exposure_violation_people / exposure_limit_people
            boom_exposure_limit_term = float(settings.weight_boom_exposure_limit) * float(
                violation_ratio * (1.0 + violation_ratio)
            )

    cutoff_shortfall_term = 0.0
    if settings.min_cutoff_emission_fraction is not None and float(settings.weight_cutoff_shortfall) > 0.0:
        cutoff_shortfall = max(0.0, float(settings.min_cutoff_emission_fraction) - float(metrics.cutoff_emission_fraction))
        if cutoff_shortfall > 0.0:
            cutoff_shortfall_term = float(settings.weight_cutoff_shortfall) * float(
                cutoff_shortfall * (1.0 + cutoff_shortfall)
            )

    populated_risk = _clip(max(pop_norm, populated_hits_norm, populated_hit_pop_norm, populated_area_norm), 0.0, 1.0)
    unpopulated_factor = 1.0 - populated_risk
    speed_weight = float(settings.weight_speed) * (
        1.0 + float(settings.unpopulated_speed_weight_bonus) * float(unpopulated_factor**1.5)
    )
    fuel_weight_scale = _clip(float(settings.min_fuel_weight_scale), 0.0, 1.0)
    fuel_weight_scale += (1.0 - fuel_weight_scale) * float(populated_risk * populated_risk)

    time_term = speed_weight * float(metrics.time_proxy_s / norm.time_scale)
    fuel_term = 0.0
    if settings.enable_fuel_objective and float(settings.weight_fuel) > 0.0:
        fuel_term = float(settings.weight_fuel) * fuel_weight_scale * float(metrics.fuel_proxy / norm.fuel_scale)
    speed_reward = (
        0.25
        * float(settings.unpopulated_speed_weight_bonus)
        * unpopulated_factor
        * float(metrics.mean_ground_speed_mps / norm.speed_scale)
    )
    return float(
        pop_term
        + overflight_pop_term
        + overflight_area_term
        + route_stretch_term
        + route_heading_term
        + populated_hit_term
        + populated_area_term
        + total_ground_hit_term
        + populated_guard_term
        + boom_exposure_limit_term
        + cutoff_shortfall_term
        + time_term
        + fuel_term
        - speed_reward
    )


def _constraint_violation_count(metrics: ObjectiveMetrics, settings: RouteOptimizationSettings) -> int:
    count = 0
    if settings.boom_exposure_limit_people is not None and float(metrics.exposure_people) > float(settings.boom_exposure_limit_people):
        count += 1
    if (
        settings.min_cutoff_emission_fraction is not None
        and float(metrics.cutoff_emission_fraction) + _EPS < float(settings.min_cutoff_emission_fraction)
    ):
        count += 1
    return count


def _candidate_rank_key(
    candidate: _Candidate,
    *,
    use_full: bool,
    settings: RouteOptimizationSettings,
) -> tuple[float, float, float, float, str]:
    metrics = candidate.full_metrics if use_full else candidate.low_metrics
    score = candidate.full_score if use_full else candidate.low_score
    if metrics is None or score is None:
        return (1.0, 1.0, 1.0, float("inf"), candidate.candidate_id)
    constraint_violation = float(_constraint_violation_count(metrics, settings))
    populated_violation = 1.0 if (metrics.populated_ground_hit_count > 0 or metrics.populated_hit_population > 0.0) else 0.0
    cutoff_shortfall = 0.0
    if settings.min_cutoff_emission_fraction is not None:
        cutoff_shortfall = max(0.0, float(settings.min_cutoff_emission_fraction) - float(metrics.cutoff_emission_fraction))
    return (
        constraint_violation,
        populated_violation,
        float(cutoff_shortfall),
        float(score),
        candidate.candidate_id,
    )


def _build_low_fidelity_config(
    full_cfg: ExperimentConfig,
    flight_path: FlightPath,
    settings: RouteOptimizationSettings,
) -> ExperimentConfig:
    cfg = _clone_experiment_config(full_cfg)
    cfg.shock.emission_interval_s = max(1.0, float(cfg.shock.emission_interval_s) * float(settings.low_fidelity_emission_interval_scale))
    cfg.shock.rays_per_emission = max(8, int(round(float(cfg.shock.rays_per_emission) * float(settings.low_fidelity_rays_scale))))
    cfg.grid.nx = max(20, int(round(float(cfg.grid.nx) * float(settings.low_fidelity_grid_scale))))
    cfg.grid.ny = max(20, int(round(float(cfg.grid.ny) * float(settings.low_fidelity_grid_scale))))
    cfg.grid.nz = max(20, int(round(float(cfg.grid.nz) * float(settings.low_fidelity_grid_scale))))
    cfg.raytrace.ds_m = max(30.0, float(cfg.raytrace.ds_m) * float(settings.low_fidelity_step_scale))
    cfg.raytrace.max_steps = max(300, int(round(float(cfg.raytrace.max_steps) * float(settings.low_fidelity_max_steps_scale))))
    required_emissions = _estimate_required_low_fidelity_emissions(flight_path, float(cfg.shock.emission_interval_s))
    min_low_fidelity_emissions = int(max(8, settings.low_fidelity_max_emissions))
    required_emissions = max(required_emissions, min_low_fidelity_emissions)
    if cfg.runtime.max_emissions is None:
        cfg.runtime.max_emissions = int(required_emissions)
    else:
        cfg.runtime.max_emissions = int(max(cfg.runtime.max_emissions, required_emissions))
    cfg.visualization.enable_matplotlib = False
    cfg.visualization.enable_plotly = False
    cfg.visualization.enable_pyvista = False
    cfg.visualization.make_animation = False
    cfg.visualization.write_html = False
    cfg.visualization.include_atmosphere = False
    cfg.population.enabled = True
    cfg.population.hit_radius_km = max(float(cfg.population.hit_radius_km), float(settings.population_guard_hit_radius_km))
    return cfg


def _build_mid_fidelity_config(
    full_cfg: ExperimentConfig,
    flight_path: FlightPath,
    settings: RouteOptimizationSettings,
) -> ExperimentConfig:
    cfg = _clone_experiment_config(full_cfg)
    cfg.shock.emission_interval_s = max(
        1.0,
        float(cfg.shock.emission_interval_s) * float(settings.mid_fidelity_emission_interval_scale),
    )
    cfg.shock.rays_per_emission = max(
        8,
        int(round(float(cfg.shock.rays_per_emission) * float(settings.mid_fidelity_rays_scale))),
    )
    cfg.grid.nx = max(24, int(round(float(cfg.grid.nx) * float(settings.mid_fidelity_grid_scale))))
    cfg.grid.ny = max(24, int(round(float(cfg.grid.ny) * float(settings.mid_fidelity_grid_scale))))
    cfg.grid.nz = max(24, int(round(float(cfg.grid.nz) * float(settings.mid_fidelity_grid_scale))))
    cfg.raytrace.ds_m = max(25.0, float(cfg.raytrace.ds_m) * float(settings.mid_fidelity_step_scale))
    cfg.raytrace.max_steps = max(
        500,
        int(round(float(cfg.raytrace.max_steps) * float(settings.mid_fidelity_max_steps_scale))),
    )
    required_emissions = _estimate_required_low_fidelity_emissions(flight_path, float(cfg.shock.emission_interval_s))
    min_mid_fidelity_emissions = int(max(12, settings.mid_fidelity_max_emissions))
    required_emissions = max(required_emissions, min_mid_fidelity_emissions)
    if cfg.runtime.max_emissions is None:
        cfg.runtime.max_emissions = int(required_emissions)
    else:
        cfg.runtime.max_emissions = int(max(cfg.runtime.max_emissions, required_emissions))
    cfg.visualization.enable_matplotlib = False
    cfg.visualization.enable_plotly = False
    cfg.visualization.enable_pyvista = False
    cfg.visualization.make_animation = False
    cfg.visualization.write_html = False
    cfg.visualization.include_atmosphere = False
    cfg.population.enabled = True
    cfg.population.hit_radius_km = max(float(cfg.population.hit_radius_km), float(settings.population_guard_hit_radius_km))
    return cfg


def _serialize_metrics(metrics: ObjectiveMetrics) -> dict[str, float | int]:
    return {
        "exposure_people": float(metrics.exposure_people),
        "overflight_population": float(metrics.overflight_population),
        "ground_hit_count": int(metrics.ground_hit_count),
        "populated_ground_hit_count": int(metrics.populated_ground_hit_count),
        "cutoff_emission_fraction": float(metrics.cutoff_emission_fraction),
        "source_cutoff_emission_fraction": float(metrics.source_cutoff_emission_fraction),
        "populated_hit_population": float(metrics.populated_hit_population),
        "populated_exposed_area_km2": float(metrics.populated_exposed_area_km2),
        "overflight_area_km2": float(metrics.overflight_area_km2),
        "elapsed_time_s": float(metrics.elapsed_time_s),
        "time_proxy_s": float(metrics.time_proxy_s),
        "fuel_proxy": float(metrics.fuel_proxy),
        "mean_ground_speed_mps": float(metrics.mean_ground_speed_mps),
        "route_distance_km": float(metrics.route_distance_km),
        "route_excess_ratio": float(metrics.route_excess_ratio),
        "route_heading_change_deg": float(metrics.route_heading_change_deg),
        "distance_to_destination_m": float(metrics.distance_to_destination_m),
        "abort_samples": int(metrics.abort_samples),
    }


def _write_waypoints_json(path: Path, waypoints: list[Waypoint]):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "waypoints": [
            {
                "lat": float(wp.lat_deg),
                "lon": float(wp.lon_deg),
                "alt_m": float(wp.alt_m),
                "time_utc": wp.time_utc.isoformat().replace("+00:00", "Z"),
            }
            for wp in waypoints
        ]
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _candidate_signature(waypoints: list[Waypoint], guidance_cfg: GuidanceConfig) -> tuple[float, ...]:
    values: list[float] = []
    for wp in waypoints:
        values.extend(
            [
                round(float(wp.lat_deg), 5),
                round(float(wp.lon_deg), 5),
                round(float(wp.alt_m), 1),
            ]
        )
    values.append(round(float(guidance_cfg.speed.target_mach), 4))
    values.append(round(float(guidance_cfg.speed.effective_mach_target), 4))
    return tuple(values)


def _evaluate_candidate(
    candidate: _Candidate,
    *,
    sim_cfg: ExperimentConfig,
    settings: RouteOptimizationSettings,
) -> tuple[SimulationResult, ObjectiveMetrics]:
    path = FlightPath(candidate.waypoints)
    simulator = MachCutoffSimulator(sim_cfg, guidance_config=candidate.guidance_config)
    result = simulator.run(path)
    metrics = _compute_metrics(result, path, sim_cfg, settings)
    return result, metrics


def _candidate_to_history_row(candidate: _Candidate, low_dir: Path, full_dir: Path | None = None) -> dict[str, object]:
    row: dict[str, object] = {
        "candidate_id": candidate.candidate_id,
        "generation": int(candidate.generation),
        "parent_candidate_id": candidate.parent_candidate_id or "",
        "mutated_waypoint_count": int(candidate.mutated_waypoint_count),
        "phase_takeoff_climb_mutations": int(candidate.phase_mutation_counts.get("takeoff_climb", 0)),
        "phase_enroute_mutations": int(candidate.phase_mutation_counts.get("enroute", 0)),
        "phase_terminal_mutations": int(candidate.phase_mutation_counts.get("terminal", 0)),
        "phase_abort_failsafe_mutations": int(candidate.phase_mutation_counts.get("abort_failsafe", 0)),
        "guidance_target_mach_delta": float(candidate.guidance_mutation.get("target_mach_delta", 0.0)),
        "guidance_effective_target_delta": float(candidate.guidance_mutation.get("effective_mach_target_delta", 0.0)),
        "guidance_target_mach": float(candidate.guidance_config.speed.target_mach),
        "low_score": float(candidate.low_score) if candidate.low_score is not None else None,
        "mid_score": float(candidate.mid_score) if candidate.mid_score is not None else None,
        "full_score": float(candidate.full_score) if candidate.full_score is not None else None,
        "low_output_dir": str(low_dir),
        "full_output_dir": str(full_dir) if full_dir is not None else "",
    }
    if candidate.low_metrics is not None:
        row.update({f"low_{k}": v for k, v in _serialize_metrics(candidate.low_metrics).items()})
    if candidate.mid_metrics is not None:
        row.update({f"mid_{k}": v for k, v in _serialize_metrics(candidate.mid_metrics).items()})
    if candidate.full_metrics is not None:
        row.update({f"full_{k}": v for k, v in _serialize_metrics(candidate.full_metrics).items()})
    return row


def _write_iteration_csv(path: Path, rows: list[dict[str, object]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def optimize_route_with_reruns(
    *,
    flight_path: FlightPath,
    config: ExperimentConfig,
    guidance_config: GuidanceConfig,
    output_dir: str | Path,
    settings: RouteOptimizationSettings | None = None,
) -> RouteOptimizationOutcome:
    settings = settings or RouteOptimizationSettings()
    settings.max_wall_time_s = max(5.0, float(settings.max_wall_time_s))
    settings.batch_size = max(1, int(settings.batch_size))
    settings.semifinalists = max(1, int(settings.semifinalists))
    settings.finalists = max(1, int(settings.finalists))
    settings.finalists = min(settings.finalists, settings.semifinalists)
    settings.elite_pool_size = max(1, int(settings.elite_pool_size))
    settings.max_duplicate_attempts = max(2, int(settings.max_duplicate_attempts))
    settings.min_control_waypoints = max(2, int(settings.min_control_waypoints))
    settings.reserve_time_for_full_fidelity_s = max(0.0, float(settings.reserve_time_for_full_fidelity_s))
    settings.reserve_time_for_mid_fidelity_s = max(0.0, float(settings.reserve_time_for_mid_fidelity_s))
    settings.population_guard_hit_radius_km = max(1.0, float(settings.population_guard_hit_radius_km))
    settings.weight_populated_ground_hits = max(0.0, float(settings.weight_populated_ground_hits))
    settings.weight_total_ground_hits = max(0.0, float(settings.weight_total_ground_hits))
    settings.weight_overflight_population = max(0.0, float(settings.weight_overflight_population))
    settings.weight_overflight_area = max(0.0, float(settings.weight_overflight_area))
    settings.weight_route_stretch = max(0.0, float(settings.weight_route_stretch))
    settings.weight_route_heading_change = max(0.0, float(settings.weight_route_heading_change))
    settings.weight_boom_exposure_limit = max(0.0, float(settings.weight_boom_exposure_limit))
    settings.weight_cutoff_shortfall = max(0.0, float(settings.weight_cutoff_shortfall))
    if settings.boom_exposure_limit_people is not None:
        settings.boom_exposure_limit_people = max(1.0, float(settings.boom_exposure_limit_people))
    if settings.min_cutoff_emission_fraction is not None:
        settings.min_cutoff_emission_fraction = _clip(float(settings.min_cutoff_emission_fraction), 0.0, 1.0)
    settings.unpopulated_speed_weight_bonus = max(0.0, float(settings.unpopulated_speed_weight_bonus))
    settings.min_fuel_weight_scale = _clip(float(settings.min_fuel_weight_scale), 0.0, 1.0)
    if str(settings.artifact_mode).strip().lower() not in {"full", "benchmark"}:
        settings.artifact_mode = "full"
    else:
        settings.artifact_mode = str(settings.artifact_mode).strip().lower()

    rng = np.random.default_rng(int(settings.seed))
    deadline = time.monotonic() + float(settings.max_wall_time_s)
    full_reserve_s = min(float(settings.reserve_time_for_full_fidelity_s), float(settings.max_wall_time_s) * 0.7)
    mid_reserve_s = min(
        float(settings.reserve_time_for_mid_fidelity_s),
        max(0.0, float(settings.max_wall_time_s) * 0.6 - full_reserve_s),
    )
    reserve_s = full_reserve_s + mid_reserve_s
    low_fidelity_deadline = deadline - reserve_s
    if low_fidelity_deadline <= time.monotonic():
        low_fidelity_deadline = deadline
    mid_fidelity_deadline = deadline - full_reserve_s
    if mid_fidelity_deadline < low_fidelity_deadline:
        mid_fidelity_deadline = low_fidelity_deadline

    full_cfg = _clone_experiment_config(config)
    full_cfg.population.enabled = True
    full_cfg.population.hit_radius_km = max(
        float(full_cfg.population.hit_radius_km),
        float(settings.population_guard_hit_radius_km),
    )
    low_cfg = _build_low_fidelity_config(full_cfg, flight_path, settings)
    mid_cfg = _build_mid_fidelity_config(full_cfg, flight_path, settings)

    optimization_dir = Path(output_dir) / "optimization"
    low_dir = optimization_dir / "low_fidelity_candidates"
    mid_dir = optimization_dir / "mid_fidelity_semifinalists"
    full_dir = optimization_dir / "full_fidelity_finalists"
    low_dir.mkdir(parents=True, exist_ok=True)
    mid_dir.mkdir(parents=True, exist_ok=True)
    full_dir.mkdir(parents=True, exist_ok=True)

    control_waypoints = _expand_waypoints_for_optimization(list(flight_path.waypoints), settings.min_control_waypoints)
    baseline_candidate = _Candidate(
        candidate_id="cand_0000_baseline",
        generation=0,
        parent_candidate_id=None,
        waypoints=control_waypoints,
        guidance_config=_clone_guidance_config(guidance_config),
        phase_mutation_counts={"takeoff_climb": 0, "enroute": 0, "terminal": 0, "abort_failsafe": 0},
        mutated_waypoint_count=0,
        guidance_mutation={},
    )

    print("[opt] evaluating low-fidelity baseline candidate")
    baseline_result, baseline_metrics = _evaluate_candidate(baseline_candidate, sim_cfg=low_cfg, settings=settings)
    baseline_candidate.low_metrics = baseline_metrics
    baseline_norm = _normalizer_from_baseline(baseline_metrics)
    baseline_candidate.low_score = _score_metrics(baseline_metrics, baseline_norm, settings)
    baseline_dir = low_dir / baseline_candidate.candidate_id
    baseline_dir.mkdir(parents=True, exist_ok=True)
    _write_waypoints_json(baseline_dir / "waypoints.json", baseline_candidate.waypoints)
    baseline_result.save_json_summary(baseline_dir / "simulation_summary_low.json")
    with (baseline_dir / "objective_low.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "candidate_id": baseline_candidate.candidate_id,
                "score": float(baseline_candidate.low_score),
                "metrics": _serialize_metrics(baseline_metrics),
            },
            f,
            indent=2,
        )

    candidates: list[_Candidate] = [baseline_candidate]
    signatures = {_candidate_signature(baseline_candidate.waypoints, baseline_candidate.guidance_config)}
    generation = 0
    stop_reason = "wall_clock_budget_reached"

    while time.monotonic() < low_fidelity_deadline:
        generation += 1
        ranked = sorted(
            [c for c in candidates if c.low_score is not None],
            key=lambda c: _candidate_rank_key(c, use_full=False, settings=settings),
        )
        elite = ranked[: min(len(ranked), settings.elite_pool_size)]
        if not elite:
            break

        accepted_in_generation = 0
        for _ in range(settings.batch_size):
            if time.monotonic() >= low_fidelity_deadline:
                break

            parent = elite[int(rng.integers(0, len(elite)))]
            candidate: _Candidate | None = None
            for _attempt in range(settings.max_duplicate_attempts):
                mutated_waypoints, phase_counts, mutated_count = _mutate_waypoints(
                    parent.waypoints,
                    parent.guidance_config,
                    rng,
                )
                mutated_guidance, guidance_mutation = _mutate_guidance(parent.guidance_config, rng, settings)
                signature = _candidate_signature(mutated_waypoints, mutated_guidance)
                if signature in signatures:
                    continue
                signatures.add(signature)
                candidate_index = len(candidates)
                candidate_id = f"cand_{candidate_index:04d}"
                candidate = _Candidate(
                    candidate_id=candidate_id,
                    generation=generation,
                    parent_candidate_id=parent.candidate_id,
                    waypoints=mutated_waypoints,
                    guidance_config=mutated_guidance,
                    phase_mutation_counts=phase_counts,
                    mutated_waypoint_count=int(mutated_count),
                    guidance_mutation=guidance_mutation,
                )
                break

            if candidate is None:
                continue

            print(f"[opt] low-fidelity evaluate {candidate.candidate_id} (generation={generation})")
            result, metrics = _evaluate_candidate(candidate, sim_cfg=low_cfg, settings=settings)
            candidate.low_metrics = metrics
            candidate.low_score = _score_metrics(metrics, baseline_norm, settings)
            candidates.append(candidate)
            accepted_in_generation += 1

            candidate_dir = low_dir / candidate.candidate_id
            candidate_dir.mkdir(parents=True, exist_ok=True)
            _write_waypoints_json(candidate_dir / "waypoints.json", candidate.waypoints)
            result.save_json_summary(candidate_dir / "simulation_summary_low.json")
            with (candidate_dir / "objective_low.json").open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "candidate_id": candidate.candidate_id,
                        "generation": generation,
                        "parent_candidate_id": candidate.parent_candidate_id,
                        "score": float(candidate.low_score),
                        "metrics": _serialize_metrics(metrics),
                    },
                    f,
                    indent=2,
                )

        if accepted_in_generation == 0:
            stop_reason = "duplicate_candidates_exhausted"
            break

    if stop_reason == "wall_clock_budget_reached" and reserve_s > 0.0 and time.monotonic() >= low_fidelity_deadline:
        stop_reason = "reserved_time_for_higher_fidelity"

    ranked_low = sorted(
        [c for c in candidates if c.low_score is not None],
        key=lambda c: _candidate_rank_key(c, use_full=False, settings=settings),
    )
    best_low = ranked_low[0]

    semifinalists: list[_Candidate] = sorted(
        ranked_low[: min(len(ranked_low), settings.semifinalists)],
        key=lambda c: _candidate_rank_key(c, use_full=False, settings=settings),
    )
    mid_norm: ObjectiveNormalizer | None = None
    ranked_mid: list[_Candidate] = []
    mid_candidate_dirs: dict[str, Path] = {}
    for idx, candidate in enumerate(semifinalists):
        if idx > 0 and time.monotonic() >= mid_fidelity_deadline:
            break
        print(f"[opt] mid-fidelity evaluate {candidate.candidate_id} ({idx + 1}/{len(semifinalists)})")
        mid_result, mid_metrics = _evaluate_candidate(candidate, sim_cfg=mid_cfg, settings=settings)
        candidate.mid_metrics = mid_metrics
        if mid_norm is None:
            mid_norm = _normalizer_from_baseline(mid_metrics)
        candidate.mid_score = _score_metrics(mid_metrics, mid_norm, settings)
        candidate_mid_dir = mid_dir / candidate.candidate_id
        candidate_mid_dir.mkdir(parents=True, exist_ok=True)
        mid_candidate_dirs[candidate.candidate_id] = candidate_mid_dir
        _write_waypoints_json(candidate_mid_dir / "waypoints.json", candidate.waypoints)
        mid_result.save_json_summary(candidate_mid_dir / "simulation_summary_mid.json")
        with (candidate_mid_dir / "objective_mid.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "candidate_id": candidate.candidate_id,
                    "score": float(candidate.mid_score),
                    "metrics": _serialize_metrics(mid_metrics),
                },
                f,
                indent=2,
            )

    ranked_mid = sorted(
        [c for c in semifinalists if c.mid_score is not None],
        key=lambda c: _candidate_rank_key(c, use_full=False, settings=settings) if c.mid_metrics is None else (
            float(_constraint_violation_count(c.mid_metrics, settings)),
            1.0 if (c.mid_metrics.populated_ground_hit_count > 0 or c.mid_metrics.populated_hit_population > 0.0) else 0.0,
            0.0 if settings.min_cutoff_emission_fraction is None else max(0.0, float(settings.min_cutoff_emission_fraction) - float(c.mid_metrics.cutoff_emission_fraction)),
            float(c.mid_score),
            c.candidate_id,
        ),
    )
    finalists: list[_Candidate] = sorted(
        (ranked_mid if ranked_mid else semifinalists)[: min(len(ranked_mid if ranked_mid else semifinalists), settings.finalists)],
        key=lambda c: (
            float(_constraint_violation_count(c.mid_metrics, settings)) if c.mid_metrics is not None else float(_constraint_violation_count(c.low_metrics, settings) if c.low_metrics is not None else 1),
            1.0 if ((c.mid_metrics or c.low_metrics) and ((c.mid_metrics or c.low_metrics).populated_ground_hit_count > 0 or (c.mid_metrics or c.low_metrics).populated_hit_population > 0.0)) else 0.0,
            float(c.mid_score if c.mid_score is not None else c.low_score if c.low_score is not None else float("inf")),
            c.candidate_id,
        ),
    )
    evaluation_queue = list(finalists)
    if baseline_candidate not in evaluation_queue:
        evaluation_queue.append(baseline_candidate)

    full_norm: ObjectiveNormalizer | None = None
    full_results: dict[str, SimulationResult] = {}
    full_candidate_dirs: dict[str, Path] = {}
    for idx, candidate in enumerate(evaluation_queue):
        if idx > 0 and time.monotonic() >= deadline:
            break
        print(f"[opt] full-fidelity evaluate {candidate.candidate_id} ({idx + 1}/{len(evaluation_queue)})")
        full_result, full_metrics = _evaluate_candidate(candidate, sim_cfg=full_cfg, settings=settings)
        candidate.full_metrics = full_metrics
        if full_norm is None:
            full_norm = _normalizer_from_baseline(full_metrics)
        candidate.full_score = _score_metrics(full_metrics, full_norm, settings)
        full_results[candidate.candidate_id] = full_result
        candidate_full_dir = full_dir / candidate.candidate_id
        candidate_full_dir.mkdir(parents=True, exist_ok=True)
        full_candidate_dirs[candidate.candidate_id] = candidate_full_dir
        _write_waypoints_json(candidate_full_dir / "waypoints.json", candidate.waypoints)
        full_result.save_json_summary(candidate_full_dir / "simulation_summary.json")
        if settings.artifact_mode != "benchmark":
            full_result.save_npz(candidate_full_dir / "simulation_hits.npz")
            full_result.save_kml(candidate_full_dir / "google_earth_overlay.kml")
        with (candidate_full_dir / "objective_full.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "candidate_id": candidate.candidate_id,
                    "score": float(candidate.full_score),
                    "metrics": _serialize_metrics(full_metrics),
                },
                f,
                indent=2,
            )

    ranked_full = sorted(
        [c for c in evaluation_queue if c.full_score is not None],
        key=lambda c: _candidate_rank_key(c, use_full=True, settings=settings),
    )

    if ranked_full:
        best_candidate = ranked_full[0]
        best_result = full_results[best_candidate.candidate_id]
    else:
        # If time budget prevented finalist runs, guarantee at least one full-fidelity output.
        print(f"[opt] full-fidelity fallback evaluate {best_low.candidate_id}")
        full_result, full_metrics = _evaluate_candidate(best_low, sim_cfg=full_cfg, settings=settings)
        norm = _normalizer_from_baseline(full_metrics)
        best_low.full_metrics = full_metrics
        best_low.full_score = _score_metrics(full_metrics, norm, settings)
        best_candidate = best_low
        best_result = full_result
        full_results[best_candidate.candidate_id] = full_result
        candidate_full_dir = full_dir / best_candidate.candidate_id
        candidate_full_dir.mkdir(parents=True, exist_ok=True)
        full_candidate_dirs[best_candidate.candidate_id] = candidate_full_dir
        _write_waypoints_json(candidate_full_dir / "waypoints.json", best_candidate.waypoints)
        full_result.save_json_summary(candidate_full_dir / "simulation_summary.json")
        if settings.artifact_mode != "benchmark":
            full_result.save_npz(candidate_full_dir / "simulation_hits.npz")
            full_result.save_kml(candidate_full_dir / "google_earth_overlay.kml")

    optimized_waypoints_path = optimization_dir / "optimized_waypoints.json"
    _write_waypoints_json(optimized_waypoints_path, best_candidate.waypoints)

    history_rows = [
        _candidate_to_history_row(
            c,
            low_dir=low_dir / c.candidate_id,
            full_dir=full_candidate_dirs.get(c.candidate_id),
        )
        for c in candidates
    ]

    leaderboard_low = []
    for candidate in ranked_low[: min(20, len(ranked_low))]:
        if candidate.low_metrics is None or candidate.low_score is None:
            continue
        leaderboard_low.append(
            {
                "candidate_id": candidate.candidate_id,
                "low_score": float(candidate.low_score),
                "metrics": _serialize_metrics(candidate.low_metrics),
            }
        )

    leaderboard_full = []
    for candidate in ranked_full:
        if candidate.full_metrics is None or candidate.full_score is None:
            continue
        leaderboard_full.append(
            {
                "candidate_id": candidate.candidate_id,
                "full_score": float(candidate.full_score),
                "metrics": _serialize_metrics(candidate.full_metrics),
            }
        )

    report = {
        "seed": int(settings.seed),
        "wall_clock_budget_s": float(settings.max_wall_time_s),
        "wall_clock_elapsed_s": float(max(0.0, settings.max_wall_time_s - max(0.0, deadline - time.monotonic()))),
        "stop_reason": stop_reason,
        "settings": asdict(settings),
        "objective_weights": {
            "population_exposure": float(settings.weight_population),
            "overflight_population": float(settings.weight_overflight_population),
            "overflight_area": float(settings.weight_overflight_area),
            "populated_ground_hits": float(settings.weight_populated_ground_hits),
            "total_ground_hits": float(settings.weight_total_ground_hits),
            "boom_exposure_limit_penalty": float(settings.weight_boom_exposure_limit),
            "cutoff_shortfall_penalty": float(settings.weight_cutoff_shortfall),
            "speed": float(settings.weight_speed),
            "fuel": float(settings.weight_fuel if settings.enable_fuel_objective else 0.0),
            "fuel_enabled": bool(settings.enable_fuel_objective),
            "unpopulated_speed_weight_bonus": float(settings.unpopulated_speed_weight_bonus),
            "min_fuel_weight_scale": float(settings.min_fuel_weight_scale),
            "artifact_mode": str(settings.artifact_mode),
        },
        "objective_constraints": {
            "boom_exposure_limit_people": (
                None if settings.boom_exposure_limit_people is None else float(settings.boom_exposure_limit_people)
            ),
            "min_cutoff_emission_fraction": (
                None
                if settings.min_cutoff_emission_fraction is None
                else float(settings.min_cutoff_emission_fraction)
            ),
        },
        "normalization": asdict(baseline_norm),
        "population_dataset_path": str(full_cfg.population.dataset_path or "bundled:us_metro_population_sample.csv"),
        "low_fidelity_config": low_cfg.to_dict(),
        "mid_fidelity_config": mid_cfg.to_dict(),
        "baseline_candidate_id": baseline_candidate.candidate_id,
        "best_low_fidelity_candidate_id": best_low.candidate_id,
        "best_mid_fidelity_candidate_id": None if not ranked_mid else ranked_mid[0].candidate_id,
        "best_final_candidate_id": best_candidate.candidate_id,
        "best_candidate_id": best_candidate.candidate_id,
        "best_low_score": None if best_low.low_score is None else float(best_low.low_score),
        "best_mid_score": None if not ranked_mid or ranked_mid[0].mid_score is None else float(ranked_mid[0].mid_score),
        "best_full_score": None if best_candidate.full_score is None else float(best_candidate.full_score),
        "best_low_metrics": None if best_low.low_metrics is None else _serialize_metrics(best_low.low_metrics),
        "best_mid_metrics": None if not ranked_mid or ranked_mid[0].mid_metrics is None else _serialize_metrics(ranked_mid[0].mid_metrics),
        "best_full_metrics": None
        if best_candidate.full_metrics is None
        else _serialize_metrics(best_candidate.full_metrics),
        "num_low_fidelity_candidates": int(len(candidates)),
        "num_mid_fidelity_candidates": int(len(ranked_mid)),
        "num_full_fidelity_candidates": int(len(ranked_full)),
        "selected_waypoints_path": str(optimized_waypoints_path),
        "low_fidelity_leaderboard": leaderboard_low,
        "mid_fidelity_leaderboard": [
            {
                "candidate_id": candidate.candidate_id,
                "mid_score": float(candidate.mid_score),
                "metrics": _serialize_metrics(candidate.mid_metrics),
            }
            for candidate in ranked_mid[: min(20, len(ranked_mid))]
            if candidate.mid_metrics is not None and candidate.mid_score is not None
        ],
        "full_fidelity_leaderboard": leaderboard_full,
        "candidate_history": history_rows,
    }

    report_path = optimization_dir / "optimization_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    iteration_csv_path = optimization_dir / "optimization_iteration_metrics.csv"
    _write_iteration_csv(iteration_csv_path, history_rows)

    print(
        "[opt] selected candidate "
        f"{best_candidate.candidate_id} | "
        f"low_score={best_candidate.low_score if best_candidate.low_score is not None else float('nan'):.4f} "
        f"full_score={best_candidate.full_score if best_candidate.full_score is not None else float('nan'):.4f}"
    )
    print(f"[opt] optimized waypoints: {optimized_waypoints_path}")
    print(f"[opt] optimization report: {report_path}")
    print(f"[opt] iteration metrics:  {iteration_csv_path}")

    return RouteOptimizationOutcome(
        result=best_result,
        best_candidate_id=best_candidate.candidate_id,
        optimized_waypoints_path=optimized_waypoints_path,
        report_path=report_path,
        iteration_csv_path=iteration_csv_path,
    )
