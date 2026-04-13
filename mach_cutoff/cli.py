"""Command-line entrypoint for Mach cutoff experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .flight.waypoints import FlightPath, load_waypoints_json
from .guidance_config import load_guidance_config
from .simulation.engine import MachCutoffSimulator
from .simulation.route_optimizer import (
    RouteOptimizationSettings,
    optimize_route_with_reruns,
)
from .visualization.basemap import MAP_STYLE_CHOICES, normalize_map_style
from .visualization import (
    render_matplotlib_bundle,
    render_plotly_bundle,
    render_pyvista_bundle,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mach cutoff simulation")
    parser.add_argument("--waypoints", required=True, help="Path to waypoint JSON")
    parser.add_argument("--config", default=None, help="Path to config JSON")
    parser.add_argument("--guidance-config", default=None, help="Path to guidance config JSON")
    parser.add_argument("--output-dir", default=None, help="Output directory override")
    parser.add_argument(
        "--map-style",
        choices=MAP_STYLE_CHOICES,
        default=None,
        help="Basemap style for map/terrain visuals",
    )

    parser.add_argument("--skip-matplotlib", action="store_true", help="Disable matplotlib outputs")
    parser.add_argument("--skip-plotly", action="store_true", help="Disable plotly outputs")
    parser.add_argument("--skip-pyvista", action="store_true", help="Disable pyvista outputs")
    parser.add_argument("--no-animation", action="store_true", help="Disable animation outputs")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Open live render windows (matplotlib/pyvista) and open plotly 3D in browser",
    )
    parser.add_argument(
        "--optimize-routing",
        action="store_true",
        help="Run iterative low-fidelity rerouting and promote top candidates to full fidelity",
    )
    parser.add_argument(
        "--no-optimize-routing",
        action="store_true",
        help="Force-disable route optimization even if enabled in config",
    )
    parser.add_argument(
        "--optimizer-max-wall-time-s",
        type=float,
        default=None,
        help="Override wall-clock budget for the optimization loop in seconds",
    )
    parser.add_argument(
        "--optimizer-seed",
        type=int,
        default=None,
        help="Override deterministic seed for candidate generation",
    )
    parser.add_argument(
        "--optimizer-finalists",
        type=int,
        default=None,
        help="Override number of low-fidelity top candidates promoted to full-fidelity evaluation",
    )
    parser.add_argument(
        "--optimizer-batch-size",
        type=int,
        default=None,
        help="Override number of candidates sampled per optimization generation",
    )
    parser.add_argument(
        "--optimizer-weight-population",
        type=float,
        default=None,
        help="Override objective weight for exposed population (lower is better)",
    )
    parser.add_argument(
        "--optimizer-weight-speed",
        type=float,
        default=None,
        help="Override objective weight for speed/time proxy (lower is better)",
    )
    parser.add_argument(
        "--optimizer-weight-fuel",
        type=float,
        default=None,
        help="Override objective weight for fuel proxy (lower is better)",
    )
    parser.add_argument(
        "--optimizer-disable-fuel",
        action="store_true",
        default=None,
        help="Disable fuel proxy term in route optimization objective (config override)",
    )
    return parser


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    guidance_cfg = load_guidance_config(args.guidance_config)
    if args.output_dir is not None:
        cfg.visualization.output_dir = args.output_dir
    if args.map_style is not None:
        cfg.visualization.map_style = args.map_style
    cfg.visualization.map_style = normalize_map_style(cfg.visualization.map_style)

    output_dir = Path(cfg.visualization.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    optimize_routing = bool(cfg.route_optimization.enabled)
    if args.optimize_routing:
        optimize_routing = True
    if args.no_optimize_routing:
        optimize_routing = False

    if optimize_routing:
        waypoints = load_waypoints_json(args.waypoints)
        flight_path = FlightPath(waypoints)

        ro_cfg = cfg.route_optimization
        enable_fuel_objective = bool(ro_cfg.enable_fuel_objective)
        if args.optimizer_disable_fuel is True:
            enable_fuel_objective = False

        optimization_settings = RouteOptimizationSettings(
            max_wall_time_s=float(
                ro_cfg.max_wall_time_s if args.optimizer_max_wall_time_s is None else args.optimizer_max_wall_time_s
            ),
            reserve_time_for_full_fidelity_s=float(ro_cfg.reserve_time_for_full_fidelity_s),
            reserve_time_for_mid_fidelity_s=float(ro_cfg.reserve_time_for_mid_fidelity_s),
            seed=int(ro_cfg.seed if args.optimizer_seed is None else args.optimizer_seed),
            batch_size=int(ro_cfg.batch_size if args.optimizer_batch_size is None else args.optimizer_batch_size),
            semifinalists=int(ro_cfg.semifinalists),
            finalists=int(ro_cfg.finalists if args.optimizer_finalists is None else args.optimizer_finalists),
            elite_pool_size=int(ro_cfg.elite_pool_size),
            max_duplicate_attempts=int(ro_cfg.max_duplicate_attempts),
            min_control_waypoints=int(ro_cfg.min_control_waypoints),
            enable_fuel_objective=enable_fuel_objective,
            weight_population=float(
                ro_cfg.weight_population if args.optimizer_weight_population is None else args.optimizer_weight_population
            ),
            weight_speed=float(ro_cfg.weight_speed if args.optimizer_weight_speed is None else args.optimizer_weight_speed),
            weight_fuel=float(ro_cfg.weight_fuel if args.optimizer_weight_fuel is None else args.optimizer_weight_fuel),
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
        optimization_outcome = optimize_route_with_reruns(
            flight_path=flight_path,
            config=cfg,
            guidance_config=guidance_cfg,
            output_dir=output_dir,
            settings=optimization_settings,
        )
        result = optimization_outcome.result
        print(f"[done] optimization report: {optimization_outcome.report_path}")
        print(f"[done] optimized route:      {optimization_outcome.optimized_waypoints_path}")
        print(f"[done] selected candidate:   {optimization_outcome.best_candidate_id}")
    else:
        simulator = MachCutoffSimulator(cfg, guidance_config=guidance_cfg)
        result = simulator.run_from_waypoint_file(args.waypoints)

    summary_path = output_dir / "simulation_summary.json"
    npz_path = output_dir / "simulation_hits.npz"
    kml_path = output_dir / "google_earth_overlay.kml"
    result.save_json_summary(summary_path)
    result.save_npz(npz_path)
    result.save_kml(kml_path)

    print(f"[done] summary: {summary_path}")
    print(f"[done] hits:    {npz_path}")
    print(f"[done] kml:     {kml_path}")
    if result.population_impact is not None:
        pop = result.population_impact
        print(
            "[done] population: "
            f"dataset={pop.dataset_name!r} "
            f"total={pop.total_population_in_heatmap:,.0f} "
            f"exposed={pop.total_exposed_population:,.0f}"
        )

    generated = {}

    backend_runners = {
        "matplotlib": (
            cfg.visualization.enable_matplotlib and not args.skip_matplotlib,
            lambda: render_matplotlib_bundle(
                result,
                output_dir,
                make_animation=(cfg.visualization.make_animation and not args.no_animation),
                include_atmosphere=cfg.visualization.include_atmosphere,
                show_window=args.interactive,
                map_style=cfg.visualization.map_style,
            ),
        ),
        "plotly": (
            cfg.visualization.enable_plotly and not args.skip_plotly,
            lambda: render_plotly_bundle(
                result,
                output_dir,
                write_html=cfg.visualization.write_html,
                include_atmosphere=cfg.visualization.include_atmosphere,
                open_browser=args.interactive,
                map_style=cfg.visualization.map_style,
            ),
        ),
        "pyvista": (
            cfg.visualization.enable_pyvista and not args.skip_pyvista,
            lambda: render_pyvista_bundle(
                result,
                output_dir,
                make_animation=(cfg.visualization.make_animation and not args.no_animation),
                show_window=args.interactive,
                map_style=cfg.visualization.map_style,
            ),
        ),
    }

    # Matplotlib window display is blocking; run it last in interactive mode.
    run_order = ["plotly", "pyvista", "matplotlib"] if args.interactive else ["matplotlib", "plotly", "pyvista"]

    for name in run_order:
        enabled, runner = backend_runners[name]
        if not enabled:
            continue
        try:
            generated[name] = runner()
        except Exception as exc:
            print(f"[warn] {name} backend failed: {exc}")

    if generated:
        print("[done] generated visual outputs:")
        for backend, outputs in generated.items():
            for key, path in outputs.items():
                print(f"  - {backend}:{key} -> {path}")


if __name__ == "__main__":
    main()
