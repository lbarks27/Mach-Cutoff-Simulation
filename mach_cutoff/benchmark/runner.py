"""Benchmark orchestration."""

from __future__ import annotations

import copy
import json
import shutil
import time
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
from typing import Any

from ..config import ExperimentConfig, RouteOptimizationConfig, load_config
from ..flight.waypoints import FlightPath, Waypoint
from ..guidance_config import GuidanceConfig, load_guidance_config
from ..simulation.engine import MachCutoffSimulator
from ..simulation.route_optimizer import (
    RouteOptimizationSettings,
    optimize_route_with_reruns,
)
from ..visualization import (
    render_matplotlib_bundle,
    render_plotly_bundle,
    render_pyvista_bundle,
)
from .analysis import write_aggregate_artifacts
from .config import BenchmarkConfig, RouteClassConfig, SensitivityProfileConfig
from .gpw import ensure_prepared_population_dataset
from .scenarios import BenchmarkScenario, build_core_scenarios


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _clone_experiment_config(cfg: ExperimentConfig) -> ExperimentConfig:
    return ExperimentConfig.from_dict(copy.deepcopy(cfg.to_dict()))


def _clone_guidance_config(cfg: GuidanceConfig) -> GuidanceConfig:
    return GuidanceConfig.from_dict(copy.deepcopy(cfg.to_dict()))


def _haversine_km(lat0: float, lon0: float, lat1: float, lon1: float) -> float:
    import math

    lat0r = math.radians(lat0)
    lon0r = math.radians(lon0)
    lat1r = math.radians(lat1)
    lon1r = math.radians(lon1)
    dlat = lat1r - lat0r
    dlon = lon1r - lon0r
    a = math.sin(dlat * 0.5) ** 2 + math.cos(lat0r) * math.cos(lat1r) * math.sin(dlon * 0.5) ** 2
    return 6371.0 * (2.0 * math.atan2(math.sqrt(a), math.sqrt(max(1e-12, 1.0 - a))))


def _build_path_for_scenario(s: BenchmarkScenario, cfg: ExperimentConfig) -> FlightPath:
    start = s.timestamp_dt
    speed_mps = max(1.0, float(cfg.aircraft.mach) * float(cfg.aircraft.reference_sound_speed_mps))
    dist_m = _haversine_km(s.origin_lat_deg, s.origin_lon_deg, s.destination_lat_deg, s.destination_lon_deg) * 1000.0
    dur_s = max(60.0, dist_m / speed_mps)
    end = start + timedelta(seconds=float(dur_s))

    waypoints = [
        Waypoint(
            lat_deg=float(s.origin_lat_deg),
            lon_deg=float(s.origin_lon_deg),
            alt_m=float(cfg.aircraft.constant_altitude_m),
            time_utc=start,
        ),
        Waypoint(
            lat_deg=float(s.destination_lat_deg),
            lon_deg=float(s.destination_lon_deg),
            alt_m=float(cfg.aircraft.constant_altitude_m),
            time_utc=end,
        ),
    ]
    return FlightPath(waypoints)


def _settings_from_route_cfg(ro_cfg: RouteOptimizationConfig) -> RouteOptimizationSettings:
    return RouteOptimizationSettings(
        max_wall_time_s=float(ro_cfg.max_wall_time_s),
        reserve_time_for_full_fidelity_s=float(ro_cfg.reserve_time_for_full_fidelity_s),
        reserve_time_for_mid_fidelity_s=float(ro_cfg.reserve_time_for_mid_fidelity_s),
        seed=int(ro_cfg.seed),
        batch_size=int(ro_cfg.batch_size),
        semifinalists=int(ro_cfg.semifinalists),
        finalists=int(ro_cfg.finalists),
        elite_pool_size=int(ro_cfg.elite_pool_size),
        max_duplicate_attempts=int(ro_cfg.max_duplicate_attempts),
        min_control_waypoints=int(ro_cfg.min_control_waypoints),
        enable_fuel_objective=bool(ro_cfg.enable_fuel_objective),
        weight_population=float(ro_cfg.weight_population),
        weight_speed=float(ro_cfg.weight_speed),
        weight_fuel=float(ro_cfg.weight_fuel),
        weight_populated_ground_hits=float(ro_cfg.weight_populated_ground_hits),
        weight_total_ground_hits=float(ro_cfg.weight_total_ground_hits),
        weight_overflight_population=float(ro_cfg.weight_overflight_population),
        weight_overflight_area=float(ro_cfg.weight_overflight_area),
        weight_route_stretch=float(ro_cfg.weight_route_stretch),
        weight_route_heading_change=float(ro_cfg.weight_route_heading_change),
        boom_exposure_limit_people=(
            None if ro_cfg.boom_exposure_limit_people is None else float(ro_cfg.boom_exposure_limit_people)
        ),
        weight_boom_exposure_limit=float(ro_cfg.weight_boom_exposure_limit),
        min_cutoff_emission_fraction=(
            None if ro_cfg.min_cutoff_emission_fraction is None else float(ro_cfg.min_cutoff_emission_fraction)
        ),
        weight_cutoff_shortfall=float(ro_cfg.weight_cutoff_shortfall),
        unpopulated_speed_weight_bonus=float(ro_cfg.unpopulated_speed_weight_bonus),
        min_fuel_weight_scale=float(ro_cfg.min_fuel_weight_scale),
        artifact_mode=str(ro_cfg.artifact_mode),
        populated_hit_people_threshold=float(ro_cfg.populated_hit_people_threshold),
        populated_cell_people_threshold=float(ro_cfg.populated_cell_people_threshold),
        population_guard_hit_radius_km=float(ro_cfg.population_guard_hit_radius_km),
        abort_sample_penalty_s=float(ro_cfg.abort_sample_penalty_s),
        low_fidelity_emission_interval_scale=float(ro_cfg.low_fidelity_emission_interval_scale),
        low_fidelity_rays_scale=float(ro_cfg.low_fidelity_rays_scale),
        low_fidelity_grid_scale=float(ro_cfg.low_fidelity_grid_scale),
        low_fidelity_step_scale=float(ro_cfg.low_fidelity_step_scale),
        low_fidelity_max_steps_scale=float(ro_cfg.low_fidelity_max_steps_scale),
        low_fidelity_max_emissions=int(ro_cfg.low_fidelity_max_emissions),
        mid_fidelity_emission_interval_scale=float(ro_cfg.mid_fidelity_emission_interval_scale),
        mid_fidelity_rays_scale=float(ro_cfg.mid_fidelity_rays_scale),
        mid_fidelity_grid_scale=float(ro_cfg.mid_fidelity_grid_scale),
        mid_fidelity_step_scale=float(ro_cfg.mid_fidelity_step_scale),
        mid_fidelity_max_steps_scale=float(ro_cfg.mid_fidelity_max_steps_scale),
        mid_fidelity_max_emissions=int(ro_cfg.mid_fidelity_max_emissions),
    )


def _configure_defaults_for_benchmark(cfg: ExperimentConfig, benchmark_cfg: BenchmarkConfig):
    cfg.population.enabled = True
    cfg.population.heatmap_cell_deg = 0.05
    cfg.population.hit_radius_km = 90.0
    cfg.population.trace_half_width_km = 14.0
    cfg.population.overflight_half_width_km = 20.0

    cfg.route_optimization.enabled = True
    cfg.route_optimization.max_wall_time_s = 240.0
    cfg.route_optimization.reserve_time_for_mid_fidelity_s = 70.0
    cfg.route_optimization.reserve_time_for_full_fidelity_s = 70.0
    cfg.route_optimization.batch_size = 4
    cfg.route_optimization.semifinalists = 6
    cfg.route_optimization.finalists = 4
    cfg.route_optimization.elite_pool_size = 4
    cfg.route_optimization.min_control_waypoints = 11
    cfg.route_optimization.artifact_mode = "full" if benchmark_cfg.save_google_earth_kml else "benchmark"

    if benchmark_cfg.enable_visual_outputs:
        cfg.visualization.enable_matplotlib = True
        cfg.visualization.enable_plotly = True
        cfg.visualization.enable_pyvista = True
        cfg.visualization.make_animation = bool(benchmark_cfg.make_animation)
        cfg.visualization.write_html = bool(benchmark_cfg.write_html)
        cfg.visualization.include_atmosphere = bool(benchmark_cfg.include_atmosphere)
    else:
        cfg.visualization.enable_matplotlib = False
        cfg.visualization.enable_plotly = False
        cfg.visualization.enable_pyvista = False
        cfg.visualization.make_animation = False
        cfg.visualization.write_html = False
        cfg.visualization.include_atmosphere = False


def _apply_route_class(
    cfg: ExperimentConfig,
    guidance: GuidanceConfig,
    route_class: RouteClassConfig,
):
    cfg_dict = cfg.to_dict()
    guidance_dict = guidance.to_dict()
    _deep_update(cfg_dict, route_class.config_overrides)
    _deep_update(guidance_dict, route_class.guidance_overrides)

    cfg2 = ExperimentConfig.from_dict(cfg_dict)
    guidance2 = GuidanceConfig.from_dict(guidance_dict)
    return cfg2, guidance2


def _apply_sensitivity(
    cfg: ExperimentConfig,
    guidance: GuidanceConfig,
    profile: SensitivityProfileConfig,
):
    cfg_dict = cfg.to_dict()
    guidance_dict = guidance.to_dict()
    _deep_update(cfg_dict, profile.config_overrides)
    _deep_update(guidance_dict, profile.guidance_overrides)

    if profile.route_weight_multipliers:
        ro = cfg_dict.setdefault("route_optimization", {})
        for key, factor in profile.route_weight_multipliers.items():
            if key in ro:
                ro[key] = float(ro[key]) * float(factor)

    cfg2 = ExperimentConfig.from_dict(cfg_dict)
    guidance2 = GuidanceConfig.from_dict(guidance_dict)
    return cfg2, guidance2


def _render_visual_outputs(
    *,
    result,
    cfg: ExperimentConfig,
    run_dir: Path,
):
    if cfg.visualization.enable_matplotlib:
        try:
            render_matplotlib_bundle(
                result,
                run_dir,
                make_animation=bool(cfg.visualization.make_animation),
                include_atmosphere=bool(cfg.visualization.include_atmosphere),
                show_window=False,
                map_style=cfg.visualization.map_style,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[bench] warn: matplotlib render failed for {run_dir.name}: {exc}")

    if cfg.visualization.enable_plotly:
        try:
            render_plotly_bundle(
                result,
                run_dir,
                write_html=bool(cfg.visualization.write_html),
                include_atmosphere=bool(cfg.visualization.include_atmosphere),
                open_browser=False,
                map_style=cfg.visualization.map_style,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[bench] warn: plotly render failed for {run_dir.name}: {exc}")

    if cfg.visualization.enable_pyvista:
        try:
            render_pyvista_bundle(
                result,
                run_dir,
                make_animation=bool(cfg.visualization.make_animation),
                show_window=False,
                map_style=cfg.visualization.map_style,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[bench] warn: pyvista render failed for {run_dir.name}: {exc}")


def run_benchmark(
    *,
    benchmark_cfg: BenchmarkConfig,
    output_root_override: str | None = None,
    jobs_override: int | None = None,
    resume_override: bool | None = None,
    force: bool = False,
    skip_sensitivity: bool = False,
    scenario_filter: set[str] | None = None,
    route_class_filter: set[str] | None = None,
) -> dict[str, Any]:
    output_root = Path(output_root_override or benchmark_cfg.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    base_cfg = load_config(benchmark_cfg.base_config_path)
    base_guidance = load_guidance_config(benchmark_cfg.base_guidance_config_path)

    prepared_population_path = ensure_prepared_population_dataset(benchmark_cfg.gpw)

    resume = benchmark_cfg.resume if resume_override is None else bool(resume_override)
    jobs = benchmark_cfg.jobs if jobs_override is None else max(1, int(jobs_override))
    if jobs > 1:
        print(f"[bench] jobs={jobs} requested; running serially in this implementation")

    scenarios = build_core_scenarios(benchmark_cfg)
    if scenario_filter:
        scenarios = [s for s in scenarios if s.scenario_id in scenario_filter]

    route_classes = {
        k: v for k, v in benchmark_cfg.route_classes.items() if (not route_class_filter or k in route_class_filter)
    }

    run_records: list[dict[str, Any]] = []

    for scenario in scenarios:
        for route_class_id, route_class in route_classes.items():
            run_records.append(
                _run_single(
                    benchmark_cfg=benchmark_cfg,
                    output_root=output_root,
                    scenario=scenario,
                    route_class_id=route_class_id,
                    sensitivity_profile_id="core",
                    route_class=route_class,
                    sensitivity_profile=None,
                    base_cfg=base_cfg,
                    base_guidance=base_guidance,
                    prepared_population_path=prepared_population_path,
                    resume=resume,
                    force=force,
                    retain_anchor_npz=benchmark_cfg.retain_anchor_npz,
                    anchor_scenario_ids=set(benchmark_cfg.anchor_scenario_ids),
                )
            )

    if not skip_sensitivity:
        core_by_id = {s.scenario_id: s for s in scenarios}
        for scenario_id in benchmark_cfg.anchor_scenario_ids:
            scenario = core_by_id.get(scenario_id)
            if scenario is None:
                continue
            if scenario_filter and scenario.scenario_id not in scenario_filter:
                continue
            for route_class_id, route_class in route_classes.items():
                for profile_id, profile in benchmark_cfg.sensitivity_profiles.items():
                    run_records.append(
                        _run_single(
                            benchmark_cfg=benchmark_cfg,
                            output_root=output_root,
                            scenario=scenario,
                            route_class_id=route_class_id,
                            sensitivity_profile_id=profile_id,
                            route_class=route_class,
                            sensitivity_profile=profile,
                            base_cfg=base_cfg,
                            base_guidance=base_guidance,
                            prepared_population_path=prepared_population_path,
                            resume=resume,
                            force=force,
                            retain_anchor_npz=benchmark_cfg.retain_anchor_npz,
                            anchor_scenario_ids=set(benchmark_cfg.anchor_scenario_ids),
                        )
                    )

    aggregate_paths = write_aggregate_artifacts(
        output_root=output_root,
        run_records=run_records,
        anchor_scenario_ids=benchmark_cfg.anchor_scenario_ids,
        research_objectives=benchmark_cfg.research_objectives,
    )

    manifest = {
        "name": benchmark_cfg.name,
        "output_root": str(output_root),
        "prepared_population_dataset": str(prepared_population_path),
        "num_runs": len(run_records),
        "aggregate_paths": {k: str(v) for k, v in aggregate_paths.items()},
    }
    manifest_path = output_root / "benchmark_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "manifest_path": manifest_path,
        "output_root": output_root,
        "run_records": run_records,
        "aggregate_paths": aggregate_paths,
    }


def _run_single(
    *,
    benchmark_cfg: BenchmarkConfig,
    output_root: Path,
    scenario: BenchmarkScenario,
    route_class_id: str,
    sensitivity_profile_id: str,
    route_class: RouteClassConfig,
    sensitivity_profile: SensitivityProfileConfig | None,
    base_cfg: ExperimentConfig,
    base_guidance: GuidanceConfig,
    prepared_population_path: Path,
    resume: bool,
    force: bool,
    retain_anchor_npz: bool,
    anchor_scenario_ids: set[str],
) -> dict[str, Any]:
    if sensitivity_profile_id == "core":
        run_dir = output_root / "core" / scenario.scenario_id / route_class_id
    else:
        run_dir = output_root / "sensitivity" / scenario.scenario_id / route_class_id / sensitivity_profile_id
    run_dir.mkdir(parents=True, exist_ok=True)

    run_complete_path = run_dir / "run_complete.json"
    if resume and not force and run_complete_path.exists():
        try:
            completed = json.loads(run_complete_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            completed = {}
        if bool(completed.get("success", False)):
            return {
                "scenario_id": scenario.scenario_id,
                "corridor_id": scenario.corridor_id,
                "timestamp_utc": scenario.timestamp_utc,
                "route_class": route_class_id,
                "sensitivity_profile": sensitivity_profile_id,
                "run_dir": str(run_dir),
                "summary_path": str(run_dir / "simulation_summary.json"),
                "optimization_report_path": str(run_dir / "optimization_report.json"),
                "wall_clock_run_s": float(completed.get("wall_clock_run_s", 0.0)),
                "skipped": True,
            }

    cfg = _clone_experiment_config(base_cfg)
    guidance = _clone_guidance_config(base_guidance)
    _configure_defaults_for_benchmark(cfg, benchmark_cfg)
    cfg.population.dataset_path = str(prepared_population_path)

    cfg, guidance = _apply_route_class(cfg, guidance, route_class)
    if sensitivity_profile is not None:
        cfg, guidance = _apply_sensitivity(cfg, guidance, sensitivity_profile)

    cfg.visualization.output_dir = str(run_dir)

    path = _build_path_for_scenario(scenario, cfg)
    run_config_path = run_dir / "run_config_resolved.json"
    run_config_path.write_text(
        json.dumps(
            {
                "scenario": asdict(scenario),
                "route_class": route_class_id,
                "sensitivity_profile": sensitivity_profile_id,
                "config": cfg.to_dict(),
                "guidance_config": guidance.to_dict(),
                "prepared_population_dataset": str(prepared_population_path),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    t0 = time.monotonic()
    success = False
    error_message = ""
    summary_path = run_dir / "simulation_summary.json"
    optimization_report_out = run_dir / "optimization_report.json"
    optimized_waypoints_out = run_dir / "optimized_waypoints.json"
    try:
        if cfg.route_optimization.enabled:
            settings = _settings_from_route_cfg(cfg.route_optimization)
            outcome = optimize_route_with_reruns(
                flight_path=path,
                config=cfg,
                guidance_config=guidance,
                output_dir=run_dir,
                settings=settings,
            )
            result = outcome.result
            if outcome.report_path.exists():
                shutil.copy2(outcome.report_path, optimization_report_out)
            if outcome.optimized_waypoints_path.exists():
                shutil.copy2(outcome.optimized_waypoints_path, optimized_waypoints_out)
        else:
            simulator = MachCutoffSimulator(cfg, guidance_config=guidance)
            result = simulator.run(path)

        result.save_json_summary(summary_path)
        should_save_npz = bool(benchmark_cfg.save_npz_for_all_runs) or (
            retain_anchor_npz and scenario.scenario_id in anchor_scenario_ids
        )
        if should_save_npz:
            result.save_npz(run_dir / "simulation_hits.npz")
        if benchmark_cfg.save_google_earth_kml:
            result.save_kml(run_dir / "google_earth_overlay.kml")
        if benchmark_cfg.enable_visual_outputs:
            _render_visual_outputs(result=result, cfg=cfg, run_dir=run_dir)
        success = True
    except Exception as exc:  # noqa: BLE001
        error_message = str(exc)

    wall_clock = float(time.monotonic() - t0)
    run_complete_payload = {
        "success": bool(success),
        "scenario_id": scenario.scenario_id,
        "route_class": route_class_id,
        "sensitivity_profile": sensitivity_profile_id,
        "wall_clock_run_s": wall_clock,
        "summary_path": str(summary_path),
        "optimization_report_path": str(optimization_report_out),
        "optimized_waypoints_path": str(optimized_waypoints_out),
        "error": error_message,
    }
    run_complete_path.write_text(json.dumps(run_complete_payload, indent=2), encoding="utf-8")

    return {
        "scenario_id": scenario.scenario_id,
        "corridor_id": scenario.corridor_id,
        "timestamp_utc": scenario.timestamp_utc,
        "route_class": route_class_id,
        "sensitivity_profile": sensitivity_profile_id,
        "run_dir": str(run_dir),
        "summary_path": str(summary_path),
        "optimization_report_path": str(optimization_report_out),
        "wall_clock_run_s": wall_clock,
        "success": bool(success),
        "error": error_message,
    }
