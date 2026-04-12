"""CLI for benchmark orchestration."""

from __future__ import annotations

import argparse

from .config import load_benchmark_config
from .runner import run_benchmark


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mach cutoff benchmark runner")
    parser.add_argument("--benchmark-config", required=True, help="Path to benchmark config JSON")
    parser.add_argument("--output-root", default=None, help="Override benchmark output root")
    parser.add_argument("--jobs", type=int, default=None, help="Requested worker count")
    parser.add_argument("--resume", action="store_true", help="Resume from existing successful runs")
    parser.add_argument("--force", action="store_true", help="Force rerun even if run_complete.json is present")
    parser.add_argument("--skip-sensitivity", action="store_true", help="Skip sensitivity profiles")
    parser.add_argument("--scenario-id", action="append", default=None, help="Run only selected scenario id(s)")
    parser.add_argument("--route-class", action="append", default=None, help="Run only selected route class(es)")
    return parser


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    cfg = load_benchmark_config(args.benchmark_config)
    scenario_filter = set(args.scenario_id or []) or None
    route_class_filter = set(args.route_class or []) or None

    result = run_benchmark(
        benchmark_cfg=cfg,
        output_root_override=args.output_root,
        jobs_override=args.jobs,
        resume_override=(True if args.resume else None),
        force=bool(args.force),
        skip_sensitivity=bool(args.skip_sensitivity),
        scenario_filter=scenario_filter,
        route_class_filter=route_class_filter,
    )

    print(f"[bench] output root: {result['output_root']}")
    print(f"[bench] manifest:    {result['manifest_path']}")
    for key, path in result["aggregate_paths"].items():
        print(f"[bench] aggregate:{key} -> {path}")


if __name__ == "__main__":
    main()
