"""Command-line entrypoint for Mach cutoff experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import ExperimentConfig, load_config
from .simulation.engine import MachCutoffSimulator
from .visualization import (
    render_matplotlib_bundle,
    render_plotly_bundle,
    render_pyvista_bundle,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mach cutoff simulation")
    parser.add_argument("--waypoints", required=True, help="Path to waypoint JSON")
    parser.add_argument("--config", default=None, help="Path to config JSON")
    parser.add_argument("--output-dir", default=None, help="Output directory override")

    parser.add_argument("--skip-matplotlib", action="store_true", help="Disable matplotlib outputs")
    parser.add_argument("--skip-plotly", action="store_true", help="Disable plotly outputs")
    parser.add_argument("--skip-pyvista", action="store_true", help="Disable pyvista outputs")
    parser.add_argument("--no-animation", action="store_true", help="Disable animation outputs")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Open live render windows (matplotlib/pyvista) and open plotly 3D in browser",
    )
    return parser


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.output_dir is not None:
        cfg.visualization.output_dir = args.output_dir

    output_dir = Path(cfg.visualization.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    simulator = MachCutoffSimulator(cfg)
    result = simulator.run_from_waypoint_file(args.waypoints)

    summary_path = output_dir / "simulation_summary.json"
    npz_path = output_dir / "simulation_hits.npz"
    result.save_json_summary(summary_path)
    result.save_npz(npz_path)

    print(f"[done] summary: {summary_path}")
    print(f"[done] hits:    {npz_path}")

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
            ),
        ),
        "pyvista": (
            cfg.visualization.enable_pyvista and not args.skip_pyvista,
            lambda: render_pyvista_bundle(
                result,
                output_dir,
                make_animation=(cfg.visualization.make_animation and not args.no_animation),
                show_window=args.interactive,
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
