"""Utilities for running repeated parameterized trials."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from ..config import ExperimentConfig
from .engine import MachCutoffSimulator


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def run_parameter_sweep(
    waypoints_path: str | Path,
    base_config: ExperimentConfig,
    variants: list[dict[str, Any]],
    output_root: str | Path,
):
    """Run multiple simulation variants for quick experimental sweeps."""
    out_root = Path(output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    manifest = []

    for i, variant in enumerate(variants):
        cfg_dict = copy.deepcopy(base_config.to_dict())
        _deep_update(cfg_dict, variant)
        cfg = ExperimentConfig.from_dict(cfg_dict)

        trial_dir = out_root / f"trial_{i:03d}"
        cfg.visualization.output_dir = str(trial_dir)

        simulator = MachCutoffSimulator(cfg)
        result = simulator.run_from_waypoint_file(waypoints_path)

        result.save_json_summary(trial_dir / "simulation_summary.json")
        result.save_npz(trial_dir / "simulation_hits.npz")

        manifest.append({
            "trial": i,
            "overrides": variant,
            "output_dir": str(trial_dir),
            "num_emissions": len(result.emissions),
            "num_rays": int(sum(len(e.rays) for e in result.emissions)),
        })

    manifest_path = out_root / "sweep_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest_path
