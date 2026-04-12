"""Aggregate analysis for benchmark runs."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from .config import ResearchObjectiveConfig


_METRICS_FOR_PAIRWISE = [
    "elapsed_time_s",
    "time_proxy_s",
    "route_distance_km",
    "fuel_proxy",
    "ground_hit_count",
    "populated_ground_hit_count",
    "boom_exposed_population",
    "boom_exposed_area_km2",
    "overflight_population",
    "overflight_area_km2",
    "cutoff_emission_fraction",
    "distance_to_destination_m",
    "abort_samples",
]

_DOMINANCE_METRICS = [
    "overflight_population",
    "boom_exposed_population",
    "elapsed_time_s",
    "fuel_proxy",
]

_PRIMARY_METRIC_BY_CLASS = {
    "fastest": "elapsed_time_s",
    "population_avoidant": "overflight_population",
    "cutoff_optimized": "boom_exposed_population",
}

_ROUTE_CLASS_ORDER = ["fastest", "population_avoidant", "cutoff_optimized"]


def write_aggregate_artifacts(
    *,
    output_root: Path,
    run_records: list[dict[str, Any]],
    anchor_scenario_ids: list[str],
    research_objectives: ResearchObjectiveConfig | None = None,
) -> dict[str, Path]:
    aggregate_dir = output_root / "aggregate"
    plots_dir = aggregate_dir / "plots"
    exposure_map_dir = aggregate_dir / "exposure_maps"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    exposure_map_dir.mkdir(parents=True, exist_ok=True)

    rows = [_extract_run_metrics(record) for record in run_records]
    rows = [row for row in rows if row is not None]
    _annotate_relative_metrics(rows, research_objectives=research_objectives)

    run_metrics_csv = aggregate_dir / "run_metrics.csv"
    _write_csv(run_metrics_csv, rows)

    efficiency_rows = _efficiency_summary(rows, research_objectives=research_objectives)
    efficiency_csv = aggregate_dir / "route_efficiency_summary.csv"
    _write_csv(efficiency_csv, efficiency_rows)

    pairwise_rows, pairwise_json = _pairwise_statistics(rows)
    pairwise_csv = aggregate_dir / "paired_statistics.csv"
    pairwise_json_path = aggregate_dir / "paired_statistics.json"
    _write_csv(pairwise_csv, pairwise_rows)
    pairwise_json_path.write_text(json.dumps(pairwise_json, indent=2), encoding="utf-8")

    dominance_rows, dominance_summary = _dominance_analysis(rows)
    dominance_csv = aggregate_dir / "dominance_matrix.csv"
    dominance_json_path = aggregate_dir / "dominance_summary.json"
    _write_csv(dominance_csv, dominance_rows)
    dominance_json_path.write_text(json.dumps(dominance_summary, indent=2), encoding="utf-8")

    robustness_summary = _robustness_analysis(
        rows=rows,
        anchor_scenario_ids=anchor_scenario_ids,
        pairwise_rows=pairwise_rows,
        dominance_rows=dominance_rows,
    )
    robustness_json_path = aggregate_dir / "robustness_summary.json"
    robustness_json_path.write_text(json.dumps(robustness_summary, indent=2), encoding="utf-8")

    feasibility_summary = _feasibility_summary(rows, research_objectives=research_objectives)
    feasibility_json_path = aggregate_dir / "feasibility_summary.json"
    feasibility_json_path.write_text(json.dumps(feasibility_summary, indent=2), encoding="utf-8")

    report_path = aggregate_dir / "benchmark_report.md"
    report_path.write_text(
        _render_report(
            rows=rows,
            efficiency_rows=efficiency_rows,
            pairwise_rows=pairwise_rows,
            dominance_summary=dominance_summary,
            robustness_summary=robustness_summary,
            feasibility_summary=feasibility_summary,
        ),
        encoding="utf-8",
    )

    _write_plots(rows=rows, pairwise_rows=pairwise_rows, robustness_summary=robustness_summary, plots_dir=plots_dir)
    exposure_map_manifest = _write_exposure_map_artifacts(
        run_records=run_records,
        anchor_scenario_ids=anchor_scenario_ids,
        output_dir=exposure_map_dir,
    )
    exposure_map_manifest_path = aggregate_dir / "exposure_map_manifest.json"
    exposure_map_manifest_path.write_text(json.dumps(exposure_map_manifest, indent=2), encoding="utf-8")

    return {
        "run_metrics_csv": run_metrics_csv,
        "route_efficiency_csv": efficiency_csv,
        "pairwise_csv": pairwise_csv,
        "pairwise_json": pairwise_json_path,
        "dominance_csv": dominance_csv,
        "dominance_json": dominance_json_path,
        "robustness_json": robustness_json_path,
        "feasibility_json": feasibility_json_path,
        "exposure_map_manifest": exposure_map_manifest_path,
        "report": report_path,
    }


def _extract_run_metrics(record: dict[str, Any]) -> dict[str, Any] | None:
    summary_path = Path(record["summary_path"])
    if not summary_path.exists():
        return None

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    optimization_metrics = _extract_best_objective_metrics(record.get("optimization_report_path"))

    pop = summary.get("population_analysis", {})
    num_emissions = int(summary.get("num_emissions", 0))
    num_cutoff = int(summary.get("num_cutoff_emissions", 0))
    cutoff_fraction = float(num_cutoff / max(1, num_emissions))

    row: dict[str, Any] = {
        "scenario_id": str(record.get("scenario_id", "")),
        "corridor_id": str(record.get("corridor_id", "")),
        "timestamp_utc": str(record.get("timestamp_utc", "")),
        "route_class": str(record.get("route_class", "")),
        "sensitivity_profile": str(record.get("sensitivity_profile", "core")),
        "elapsed_time_s": float(optimization_metrics.get("elapsed_time_s", 0.0)),
        "time_proxy_s": float(optimization_metrics.get("time_proxy_s", 0.0)),
        "route_distance_km": float(optimization_metrics.get("route_distance_km", 0.0)),
        "fuel_proxy": float(optimization_metrics.get("fuel_proxy", 0.0)),
        "ground_hit_count": int(summary.get("num_ground_hits", 0)),
        "populated_ground_hit_count": int(optimization_metrics.get("populated_ground_hit_count", 0)),
        "boom_exposed_population": float(pop.get("total_exposed_population", 0.0)),
        "boom_exposed_area_km2": float(pop.get("total_exposed_area_km2", 0.0)),
        "overflight_population": float(pop.get("total_overflight_population", 0.0)),
        "overflight_area_km2": float(pop.get("total_overflight_area_km2", 0.0)),
        "cutoff_emission_fraction": cutoff_fraction,
        "full_route_cutoff": bool(summary.get("full_route_cutoff", False)),
        "distance_to_destination_m": float(optimization_metrics.get("distance_to_destination_m", 0.0)),
        "abort_samples": int(optimization_metrics.get("abort_samples", 0)),
        "wall_clock_run_s": float(record.get("wall_clock_run_s", 0.0)),
    }
    return row


def _extract_best_objective_metrics(report_path: str | None) -> dict[str, Any]:
    if not report_path:
        return {}
    path = Path(report_path)
    if not path.exists():
        return {}
    report = json.loads(path.read_text(encoding="utf-8"))
    best_id = str(report.get("best_final_candidate_id", ""))

    leaderboard = list(report.get("full_fidelity_leaderboard", []) or [])
    for row in leaderboard:
        if str(row.get("candidate_id", "")) == best_id:
            return dict(row.get("metrics", {}))

    if leaderboard:
        return dict(leaderboard[0].get("metrics", {}))

    low_leaderboard = list(report.get("low_fidelity_leaderboard", []) or [])
    if low_leaderboard:
        return dict(low_leaderboard[0].get("metrics", {}))
    return {}


def _write_csv(path: Path, rows: list[dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
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


def _ordered_route_classes(route_classes: set[str]) -> list[str]:
    known = [route_class for route_class in _ROUTE_CLASS_ORDER if route_class in route_classes]
    extras = sorted(route_classes - set(known))
    return known + extras


def _relative_delta_pct(value: float, baseline: float) -> float:
    baseline_abs = abs(float(baseline))
    if baseline_abs <= 1e-9:
        return 0.0 if abs(float(value)) <= 1e-9 else float("nan")
    return float((float(value) - float(baseline)) / baseline_abs * 100.0)


def _reduction_pct(value: float, baseline: float) -> float:
    baseline_abs = abs(float(baseline))
    if baseline_abs <= 1e-9:
        return 0.0 if abs(float(value)) <= 1e-9 else float("nan")
    return float((float(baseline) - float(value)) / baseline_abs * 100.0)


def _format_threshold(value: float | None, *, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.1f}{suffix}"


def _annotate_relative_metrics(
    rows: list[dict[str, Any]],
    *,
    research_objectives: ResearchObjectiveConfig | None,
):
    baseline_route_class = (
        str(research_objectives.baseline_route_class_id)
        if research_objectives is not None
        else "fastest"
    )
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (str(row["scenario_id"]), str(row.get("sensitivity_profile", "core")))
        grouped[key][str(row["route_class"])] = row

    for key, case_rows in grouped.items():
        baseline = case_rows.get(baseline_route_class)
        for row in case_rows.values():
            if baseline is None:
                row["time_delta_vs_fastest_s"] = float("nan")
                row["time_penalty_vs_fastest_pct"] = float("nan")
                row["boom_exposure_delta_vs_fastest"] = float("nan")
                row["boom_exposure_reduction_vs_fastest_pct"] = float("nan")
                row["overflight_delta_vs_fastest"] = float("nan")
                row["overflight_reduction_vs_fastest_pct"] = float("nan")
                continue

            row["time_delta_vs_fastest_s"] = float(row.get("elapsed_time_s", 0.0)) - float(
                baseline.get("elapsed_time_s", 0.0)
            )
            row["time_penalty_vs_fastest_pct"] = _relative_delta_pct(
                float(row.get("elapsed_time_s", 0.0)),
                float(baseline.get("elapsed_time_s", 0.0)),
            )
            row["boom_exposure_delta_vs_fastest"] = float(row.get("boom_exposed_population", 0.0)) - float(
                baseline.get("boom_exposed_population", 0.0)
            )
            row["boom_exposure_reduction_vs_fastest_pct"] = _reduction_pct(
                float(row.get("boom_exposed_population", 0.0)),
                float(baseline.get("boom_exposed_population", 0.0)),
            )
            row["overflight_delta_vs_fastest"] = float(row.get("overflight_population", 0.0)) - float(
                baseline.get("overflight_population", 0.0)
            )
            row["overflight_reduction_vs_fastest_pct"] = _reduction_pct(
                float(row.get("overflight_population", 0.0)),
                float(baseline.get("overflight_population", 0.0)),
            )

    for row in rows:
        feasible, reasons = _evaluate_feasibility(row, research_objectives=research_objectives)
        row["feasible_under_research_objective"] = bool(feasible)
        row["feasibility_status"] = "feasible" if feasible else "infeasible"
        row["feasibility_reasons"] = "; ".join(reasons)


def _evaluate_feasibility(
    row: dict[str, Any],
    *,
    research_objectives: ResearchObjectiveConfig | None,
) -> tuple[bool, list[str]]:
    if research_objectives is None:
        return True, []

    reasons: list[str] = []
    boom_exposure_limit = research_objectives.boom_exposure_limit_people
    if boom_exposure_limit is not None and float(row.get("boom_exposed_population", 0.0)) > float(boom_exposure_limit):
        reasons.append("boom_exposure_limit_exceeded")

    max_time_penalty_pct = research_objectives.max_time_penalty_pct
    time_penalty_pct = float(row.get("time_penalty_vs_fastest_pct", 0.0))
    if (
        max_time_penalty_pct is not None
        and str(row.get("route_class", "")) != str(research_objectives.baseline_route_class_id)
        and np.isfinite(time_penalty_pct)
        and time_penalty_pct > float(max_time_penalty_pct)
    ):
        reasons.append("time_penalty_limit_exceeded")

    if float(row.get("distance_to_destination_m", 0.0)) > float(research_objectives.max_distance_to_destination_m):
        reasons.append("destination_not_reached")

    if int(row.get("abort_samples", 0)) > int(research_objectives.max_abort_samples):
        reasons.append("abort_samples_exceeded")

    if (
        research_objectives.min_cutoff_emission_fraction is not None
        and str(row.get("route_class", "")) == "cutoff_optimized"
        and float(row.get("cutoff_emission_fraction", 0.0)) + 1e-9
        < float(research_objectives.min_cutoff_emission_fraction)
    ):
        reasons.append("cutoff_fraction_below_target")

    return len(reasons) == 0, reasons


def _efficiency_summary(
    rows: list[dict[str, Any]],
    *,
    research_objectives: ResearchObjectiveConfig | None,
) -> list[dict[str, Any]]:
    core_rows = [r for r in rows if str(r.get("sensitivity_profile", "core")) == "core"]
    summary_rows: list[dict[str, Any]] = []
    classes = _ordered_route_classes({str(r.get("route_class", "")) for r in core_rows})
    for route_class in classes:
        class_rows = [r for r in core_rows if str(r.get("route_class")) == route_class]
        if not class_rows:
            continue
        elapsed = np.asarray([float(r.get("elapsed_time_s", 0.0)) for r in class_rows], dtype=np.float64)
        boom = np.asarray([float(r.get("boom_exposed_population", 0.0)) for r in class_rows], dtype=np.float64)
        overflight = np.asarray([float(r.get("overflight_population", 0.0)) for r in class_rows], dtype=np.float64)
        time_penalty = np.asarray(
            [float(r.get("time_penalty_vs_fastest_pct", 0.0)) for r in class_rows if np.isfinite(r.get("time_penalty_vs_fastest_pct", 0.0))],
            dtype=np.float64,
        )
        feasible_runs = int(sum(1 for r in class_rows if bool(r.get("feasible_under_research_objective", False))))
        summary_rows.append(
            {
                "route_class": route_class,
                "n_runs": int(len(class_rows)),
                "median_elapsed_time_s": float(np.median(elapsed)),
                "median_time_penalty_vs_fastest_pct": float(np.median(time_penalty)) if time_penalty.size else 0.0,
                "median_boom_exposed_population": float(np.median(boom)),
                "median_boom_reduction_vs_fastest_pct": float(
                    np.median([float(r.get("boom_exposure_reduction_vs_fastest_pct", 0.0)) for r in class_rows])
                ),
                "median_overflight_population": float(np.median(overflight)),
                "median_overflight_reduction_vs_fastest_pct": float(
                    np.median([float(r.get("overflight_reduction_vs_fastest_pct", 0.0)) for r in class_rows])
                ),
                "median_cutoff_emission_fraction": float(
                    np.median([float(r.get("cutoff_emission_fraction", 0.0)) for r in class_rows])
                ),
                "feasible_runs": feasible_runs,
                "feasibility_rate": float(feasible_runs / len(class_rows)) if class_rows else 0.0,
                "research_threshold_boom_exposure_limit_people": (
                    None
                    if research_objectives is None or research_objectives.boom_exposure_limit_people is None
                    else float(research_objectives.boom_exposure_limit_people)
                ),
            }
        )
    return summary_rows


def _feasibility_summary(
    rows: list[dict[str, Any]],
    *,
    research_objectives: ResearchObjectiveConfig | None,
) -> dict[str, Any]:
    core_rows = [r for r in rows if str(r.get("sensitivity_profile", "core")) == "core"]
    if research_objectives is None:
        return {
            "criteria": None,
            "by_route_class": {},
        }

    by_route_class: dict[str, Any] = {}
    classes = _ordered_route_classes({str(r.get("route_class", "")) for r in core_rows})
    for route_class in classes:
        class_rows = [r for r in core_rows if str(r.get("route_class")) == route_class]
        if not class_rows:
            continue
        feasible_rows = [r for r in class_rows if bool(r.get("feasible_under_research_objective", False))]
        reason_counts: dict[str, int] = defaultdict(int)
        for row in class_rows:
            reasons = str(row.get("feasibility_reasons", "")).strip()
            if not reasons:
                continue
            for reason in reasons.split(";"):
                cleaned = reason.strip()
                if cleaned:
                    reason_counts[cleaned] += 1
        by_route_class[route_class] = {
            "n_runs": int(len(class_rows)),
            "feasible_runs": int(len(feasible_rows)),
            "feasibility_rate": float(len(feasible_rows) / len(class_rows)) if class_rows else 0.0,
            "median_boom_exposed_population": float(
                np.median([float(r.get("boom_exposed_population", 0.0)) for r in class_rows])
            ),
            "median_time_penalty_vs_fastest_pct": float(
                np.median([float(r.get("time_penalty_vs_fastest_pct", 0.0)) for r in class_rows])
            ),
            "median_cutoff_emission_fraction": float(
                np.median([float(r.get("cutoff_emission_fraction", 0.0)) for r in class_rows])
            ),
            "failure_reason_counts": dict(sorted(reason_counts.items())),
        }

    return {
        "criteria": {
            "baseline_route_class_id": str(research_objectives.baseline_route_class_id),
            "boom_exposure_limit_people": (
                None
                if research_objectives.boom_exposure_limit_people is None
                else float(research_objectives.boom_exposure_limit_people)
            ),
            "max_time_penalty_pct": (
                None if research_objectives.max_time_penalty_pct is None else float(research_objectives.max_time_penalty_pct)
            ),
            "min_cutoff_emission_fraction": (
                None
                if research_objectives.min_cutoff_emission_fraction is None
                else float(research_objectives.min_cutoff_emission_fraction)
            ),
            "max_abort_samples": int(research_objectives.max_abort_samples),
            "max_distance_to_destination_m": float(research_objectives.max_distance_to_destination_m),
        },
        "by_route_class": by_route_class,
    }


def _pairwise_statistics(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    comparisons = [
        ("population_avoidant", "fastest"),
        ("cutoff_optimized", "fastest"),
        ("cutoff_optimized", "population_avoidant"),
    ]

    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (str(row["scenario_id"]), str(row.get("sensitivity_profile", "core")))
        grouped[key][str(row["route_class"])] = row

    out_rows: list[dict[str, Any]] = []
    out_json: dict[str, Any] = {"comparisons": []}
    for left, right in comparisons:
        comp_payload: dict[str, Any] = {"left": left, "right": right, "metrics": []}
        for metric in _METRICS_FOR_PAIRWISE:
            diffs: list[float] = []
            for pair in grouped.values():
                if left not in pair or right not in pair:
                    continue
                a = float(pair[left].get(metric, 0.0))
                b = float(pair[right].get(metric, 0.0))
                diffs.append(a - b)
            if not diffs:
                continue
            arr = np.asarray(diffs, dtype=np.float64)
            hl = _hodges_lehmann(arr)
            p_value, rank_biserial = _wilcoxon_signed_rank(arr)
            wins = int(np.sum(arr < 0.0))
            losses = int(np.sum(arr > 0.0))
            ties = int(np.sum(np.isclose(arr, 0.0)))
            row = {
                "comparison": f"{left}_vs_{right}",
                "metric": metric,
                "n_pairs": int(arr.size),
                "median_delta": float(np.median(arr)),
                "mean_delta": float(np.mean(arr)),
                "hodges_lehmann": float(hl),
                "wilcoxon_p_value": float(p_value),
                "rank_biserial": float(rank_biserial),
                "wins": wins,
                "losses": losses,
                "ties": ties,
            }
            out_rows.append(row)
            comp_payload["metrics"].append(row)
        out_json["comparisons"].append(comp_payload)
    return out_rows, out_json


def _hodges_lehmann(diffs: np.ndarray) -> float:
    d = np.asarray(diffs, dtype=np.float64).reshape(-1)
    if d.size == 0:
        return 0.0
    vals: list[float] = []
    for i in range(d.size):
        for j in range(i, d.size):
            vals.append(float(0.5 * (d[i] + d[j])))
    return float(median(vals))


def _wilcoxon_signed_rank(diffs: np.ndarray) -> tuple[float, float]:
    d = np.asarray(diffs, dtype=np.float64).reshape(-1)
    d = d[~np.isclose(d, 0.0)]
    n = int(d.size)
    if n == 0:
        return 1.0, 0.0

    abs_d = np.abs(d)
    ranks = _average_ranks(abs_d)
    w_pos = float(np.sum(ranks[d > 0.0]))
    w_neg = float(np.sum(ranks[d < 0.0]))
    w = min(w_pos, w_neg)

    mean_w = n * (n + 1.0) / 4.0
    tie_corr = 0.0
    _, counts = np.unique(abs_d, return_counts=True)
    for c in counts:
        if c > 1:
            tie_corr += float(c * (c - 1) * (c + 1))
    var_w = n * (n + 1.0) * (2.0 * n + 1.0) / 24.0 - tie_corr / 48.0
    if var_w <= 1e-12:
        p_value = 1.0
    else:
        z = (w - mean_w - 0.5) / math.sqrt(var_w)
        p_value = float(math.erfc(abs(z) / math.sqrt(2.0)))

    denom = max(w_pos + w_neg, 1e-9)
    rank_biserial = float((w_pos - w_neg) / denom)
    return p_value, rank_biserial


def _average_ranks(values: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(vals, kind="mergesort")
    ranks = np.zeros(vals.size, dtype=np.float64)
    i = 0
    while i < vals.size:
        j = i + 1
        while j < vals.size and np.isclose(vals[order[j]], vals[order[i]]):
            j += 1
        avg_rank = 0.5 * (i + 1 + j)
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def _dominates(a: dict[str, Any], b: dict[str, Any]) -> bool:
    all_leq = True
    any_lt = False
    for metric in _DOMINANCE_METRICS:
        av = float(a.get(metric, 0.0))
        bv = float(b.get(metric, 0.0))
        if av > bv:
            all_leq = False
            break
        if av < bv:
            any_lt = True
    return all_leq and any_lt


def _dominance_analysis(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (str(row["scenario_id"]), str(row.get("sensitivity_profile", "core")))
        grouped[key].append(row)

    matrix_rows: list[dict[str, Any]] = []
    dominates_count: dict[str, int] = defaultdict(int)
    dominated_count: dict[str, int] = defaultdict(int)
    pair_counts: dict[str, int] = defaultdict(int)

    for (scenario_id, sensitivity_profile), group_rows in grouped.items():
        for a in group_rows:
            for b in group_rows:
                if a is b:
                    continue
                a_class = str(a["route_class"])
                b_class = str(b["route_class"])
                dom = _dominates(a, b)
                matrix_rows.append(
                    {
                        "scenario_id": scenario_id,
                        "sensitivity_profile": sensitivity_profile,
                        "class_a": a_class,
                        "class_b": b_class,
                        "dominates": bool(dom),
                    }
                )
                pair_key = f"{a_class}->{b_class}"
                if dom:
                    pair_counts[pair_key] += 1
                    dominates_count[a_class] += 1
                    dominated_count[b_class] += 1

    summary = {
        "dominates_count": dict(dominates_count),
        "dominated_count": dict(dominated_count),
        "pairwise_dominance_counts": dict(pair_counts),
    }
    return matrix_rows, summary


def _robustness_analysis(
    *,
    rows: list[dict[str, Any]],
    anchor_scenario_ids: list[str],
    pairwise_rows: list[dict[str, Any]],
    dominance_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    anchors = set(anchor_scenario_ids)
    filtered = [r for r in rows if r["scenario_id"] in anchors]

    # Sign consistency for paired deltas against core baseline.
    sign_consistency: dict[str, dict[str, float]] = {}
    grouped: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in filtered:
        key = (str(row["scenario_id"]), str(row.get("sensitivity_profile", "core")), str(row["route_class"]))
        grouped[key] = row

    comparisons = [
        ("population_avoidant", "fastest"),
        ("cutoff_optimized", "fastest"),
        ("cutoff_optimized", "population_avoidant"),
    ]
    for left, right in comparisons:
        comp_key = f"{left}_vs_{right}"
        sign_consistency[comp_key] = {}
        for metric in _METRICS_FOR_PAIRWISE:
            same = 0
            total = 0
            for scenario_id in anchors:
                core_left = grouped.get((scenario_id, "core", left))
                core_right = grouped.get((scenario_id, "core", right))
                if core_left is None or core_right is None:
                    continue
                core_delta = float(core_left.get(metric, 0.0)) - float(core_right.get(metric, 0.0))
                core_sign = 0 if np.isclose(core_delta, 0.0) else (1 if core_delta > 0.0 else -1)
                for sensitivity in sorted({str(r.get("sensitivity_profile", "core")) for r in filtered}):
                    if sensitivity == "core":
                        continue
                    left_row = grouped.get((scenario_id, sensitivity, left))
                    right_row = grouped.get((scenario_id, sensitivity, right))
                    if left_row is None or right_row is None:
                        continue
                    delta = float(left_row.get(metric, 0.0)) - float(right_row.get(metric, 0.0))
                    sign = 0 if np.isclose(delta, 0.0) else (1 if delta > 0.0 else -1)
                    if sign == core_sign:
                        same += 1
                    total += 1
            sign_consistency[comp_key][metric] = float(same / total) if total > 0 else 0.0

    # Dominance retention.
    core_dom: dict[tuple[str, str, str], bool] = {}
    sens_dom: list[tuple[tuple[str, str, str], bool]] = []
    for row in dominance_rows:
        sid = str(row["scenario_id"])
        if sid not in anchors:
            continue
        key = (sid, str(row["class_a"]), str(row["class_b"]))
        sens = str(row.get("sensitivity_profile", "core"))
        dominates = bool(row.get("dominates", False))
        if sens == "core":
            core_dom[key] = dominates
        else:
            sens_dom.append((key, dominates))

    same_dom = 0
    total_dom = 0
    for key, dom in sens_dom:
        if key not in core_dom:
            continue
        if dom == core_dom[key]:
            same_dom += 1
        total_dom += 1
    dominance_retention_rate = float(same_dom / total_dom) if total_dom > 0 else 0.0

    # Primary-metric winner retention by class.
    primary_wins: dict[str, int] = defaultdict(int)
    primary_total = 0
    by_case: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in filtered:
        sens = str(row.get("sensitivity_profile", "core"))
        if sens == "core":
            continue
        by_case[(str(row["scenario_id"]), sens)].append(row)

    for case_rows in by_case.values():
        by_class = {str(r["route_class"]): r for r in case_rows}
        if any(c not in by_class for c in _PRIMARY_METRIC_BY_CLASS):
            continue
        winners: list[str] = []
        for route_class, metric in _PRIMARY_METRIC_BY_CLASS.items():
            class_value = float(by_class[route_class].get(metric, 0.0))
            all_values = [float(by_class[c].get(metric, 0.0)) for c in by_class]
            if np.isclose(class_value, min(all_values)):
                winners.append(route_class)
        if not winners:
            continue
        primary_total += 1
        for w in winners:
            primary_wins[w] += 1

    primary_rate = {
        route_class: float(primary_wins.get(route_class, 0) / primary_total) if primary_total > 0 else 0.0
        for route_class in _PRIMARY_METRIC_BY_CLASS
    }

    return {
        "anchor_scenarios": sorted(anchors),
        "sign_consistency": sign_consistency,
        "dominance_retention_rate": dominance_retention_rate,
        "primary_metric_best_rate": primary_rate,
    }


def _render_report(
    *,
    rows: list[dict[str, Any]],
    efficiency_rows: list[dict[str, Any]],
    pairwise_rows: list[dict[str, Any]],
    dominance_summary: dict[str, Any],
    robustness_summary: dict[str, Any],
    feasibility_summary: dict[str, Any],
) -> str:
    core_rows = [r for r in rows if str(r.get("sensitivity_profile", "core")) == "core"]
    classes = _ordered_route_classes({str(r.get("route_class", "")) for r in core_rows})

    lines: list[str] = []
    lines.append("# Benchmark Report")
    lines.append("")
    lines.append("## Dataset and benchmark manifest")
    lines.append(f"- Total runs: {len(rows)}")
    lines.append(f"- Core runs: {len(core_rows)}")
    lines.append("")
    lines.append("## Core scenario inventory")
    lines.append(f"- Route classes: {', '.join(classes)}")
    lines.append(f"- Unique scenarios: {len(set(str(r['scenario_id']) for r in core_rows))}")
    lines.append("")
    lines.append("## Route efficiency comparisons")
    for row in efficiency_rows:
        lines.append(
            f"- `{row['route_class']}`: median elapsed={float(row['median_elapsed_time_s']):.1f}s, "
            f"time penalty vs fastest={float(row['median_time_penalty_vs_fastest_pct']):.1f}%, "
            f"boom reduction vs fastest={float(row['median_boom_reduction_vs_fastest_pct']):.1f}%, "
            f"overflight reduction vs fastest={float(row['median_overflight_reduction_vs_fastest_pct']):.1f}%"
        )
    lines.append("")

    criteria = feasibility_summary.get("criteria")
    lines.append("## Feasibility assessment")
    if criteria is None:
        lines.append("- No research-objective thresholds were declared.")
    else:
        lines.append(
            "- Thresholds: "
            f"boom_exposure<={_format_threshold(criteria.get('boom_exposure_limit_people'))} people, "
            f"time_penalty<={_format_threshold(criteria.get('max_time_penalty_pct'), suffix='%')}, "
            f"cutoff_fraction>={_format_threshold(criteria.get('min_cutoff_emission_fraction'))}, "
            f"abort_samples<={int(criteria.get('max_abort_samples', 0))}, "
            f"distance_to_destination<={_format_threshold(criteria.get('max_distance_to_destination_m'), suffix=' m')}"
        )
        for route_class in classes:
            payload = feasibility_summary.get("by_route_class", {}).get(route_class)
            if not payload:
                continue
            lines.append(
                f"- `{route_class}`: feasible {int(payload['feasible_runs'])}/{int(payload['n_runs'])} "
                f"({float(payload['feasibility_rate']):.3f}), median boom={float(payload['median_boom_exposed_population']):.1f}, "
                f"median time penalty={float(payload['median_time_penalty_vs_fastest_pct']):.1f}%"
            )
            reason_counts = payload.get("failure_reason_counts", {})
            if reason_counts:
                reasons = ", ".join(f"{reason}={count}" for reason, count in sorted(reason_counts.items()))
                lines.append(f"- `{route_class}` failures: {reasons}")
    lines.append("")

    lines.append("## Pairwise findings")
    for row in pairwise_rows[:18]:
        lines.append(
            f"- {row['comparison']} / {row['metric']}: median_delta={row['median_delta']:.3f}, p={row['wilcoxon_p_value']:.3g}, wins={row['wins']}, losses={row['losses']}"
        )
    lines.append("")

    lines.append("## Dominance findings")
    for key, value in sorted(dominance_summary.get("pairwise_dominance_counts", {}).items()):
        lines.append(f"- {key}: {value}")
    lines.append("")

    lines.append("## Sensitivity robustness findings")
    lines.append(
        f"- Dominance retention rate: {float(robustness_summary.get('dominance_retention_rate', 0.0)):.3f}"
    )
    for route_class, rate in sorted(robustness_summary.get("primary_metric_best_rate", {}).items()):
        lines.append(f"- Primary-metric best rate `{route_class}`: {float(rate):.3f}")
    lines.append("")

    lines.append("## Failures / incomplete runs")
    lines.append("- See run directories for `run_complete.json` status and any captured error traces.")
    lines.append("")

    lines.append("## Threats to validity")
    lines.append("- Optimization stochasticity may influence route ranking; compare across seeds for stronger confidence.")
    lines.append("- Atmospheric sampling at selected timestamps may not capture full climatological variability.")
    lines.append("- Population metrics are model-based proxies, not direct measured exposure outcomes.")
    lines.append("")

    return "\n".join(lines)


def _write_exposure_map_artifacts(
    *,
    run_records: list[dict[str, Any]],
    anchor_scenario_ids: list[str],
    output_dir: Path,
) -> dict[str, Any]:
    try:
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return {"maps": {}, "note": "matplotlib unavailable"}

    manifests: dict[str, str] = {}
    for scenario_id in anchor_scenario_ids:
        case_records = [
            r
            for r in run_records
            if str(r.get("scenario_id", "")) == scenario_id and str(r.get("sensitivity_profile", "core")) == "core"
        ]
        case_records = sorted(
            case_records,
            key=lambda r: (
                _ordered_route_classes({str(rec.get("route_class", "")) for rec in case_records}).index(
                    str(r.get("route_class", ""))
                )
                if str(r.get("route_class", "")) in {str(rec.get("route_class", "")) for rec in case_records}
                else 999
            ),
        )

        payloads: list[dict[str, Any]] = []
        lat_candidates: list[float] = []
        lon_candidates: list[float] = []
        for record in case_records:
            npz_path = Path(str(record.get("run_dir", ""))) / "simulation_hits.npz"
            if not npz_path.exists():
                continue
            with np.load(npz_path) as raw:
                heatmap = np.asarray(raw["population_heatmap_people"], dtype=np.float64) if "population_heatmap_people" in raw else None
                lat_edges = (
                    np.asarray(raw["population_heatmap_lat_edges_deg"], dtype=np.float64)
                    if "population_heatmap_lat_edges_deg" in raw
                    else None
                )
                lon_edges = (
                    np.asarray(raw["population_heatmap_lon_edges_deg"], dtype=np.float64)
                    if "population_heatmap_lon_edges_deg" in raw
                    else None
                )
                exposed_mask = (
                    np.asarray(raw["population_exposed_cell_mask"], dtype=bool)
                    if "population_exposed_cell_mask" in raw
                    else None
                )
                hit_lat = np.asarray(raw["hit_lat_deg"], dtype=np.float64) if "hit_lat_deg" in raw else np.asarray([], dtype=np.float64)
                hit_lon = np.asarray(raw["hit_lon_deg"], dtype=np.float64) if "hit_lon_deg" in raw else np.asarray([], dtype=np.float64)

            if lat_edges is not None and lat_edges.size:
                lat_candidates.extend([float(np.min(lat_edges)), float(np.max(lat_edges))])
            if lon_edges is not None and lon_edges.size:
                lon_candidates.extend([float(np.min(lon_edges)), float(np.max(lon_edges))])
            if hit_lat.size:
                lat_candidates.extend(hit_lat.tolist())
            if hit_lon.size:
                lon_candidates.extend(hit_lon.tolist())

            summary_payload: dict[str, Any] = {}
            summary_path = Path(str(record.get("summary_path", "")))
            if summary_path.exists():
                summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
            payloads.append(
                {
                    "route_class": str(record.get("route_class", "")),
                    "heatmap": heatmap,
                    "lat_edges": lat_edges,
                    "lon_edges": lon_edges,
                    "exposed_mask": exposed_mask,
                    "hit_lat": hit_lat,
                    "hit_lon": hit_lon,
                    "summary": summary_payload,
                }
            )

        if not payloads:
            continue

        lat_min = min(lat_candidates) if lat_candidates else -1.0
        lat_max = max(lat_candidates) if lat_candidates else 1.0
        lon_min = min(lon_candidates) if lon_candidates else -1.0
        lon_max = max(lon_candidates) if lon_candidates else 1.0
        lat_pad = max(0.15, 0.05 * max(lat_max - lat_min, 1.0))
        lon_pad = max(0.15, 0.05 * max(lon_max - lon_min, 1.0))

        fig, axes = plt.subplots(1, len(payloads), figsize=(5.2 * len(payloads), 5.0), squeeze=False)
        route_colors = {
            "fastest": "#1f1f1f",
            "population_avoidant": "#0e7490",
            "cutoff_optimized": "#b91c1c",
        }

        for ax, payload in zip(axes[0], payloads, strict=False):
            heatmap = payload["heatmap"]
            lat_edges = payload["lat_edges"]
            lon_edges = payload["lon_edges"]
            exposed_mask = payload["exposed_mask"]
            hit_lat = payload["hit_lat"]
            hit_lon = payload["hit_lon"]
            route_class = payload["route_class"]
            color = route_colors.get(route_class, "#7c3aed")

            if (
                heatmap is not None
                and lat_edges is not None
                and lon_edges is not None
                and heatmap.ndim == 2
                and heatmap.shape == (max(0, lat_edges.size - 1), max(0, lon_edges.size - 1))
            ):
                ax.pcolormesh(
                    lon_edges,
                    lat_edges,
                    np.log10(np.maximum(heatmap, 0.0) + 1.0),
                    shading="auto",
                    cmap="Greys",
                    alpha=0.70,
                )
                if exposed_mask is not None and exposed_mask.shape == heatmap.shape and np.any(exposed_mask):
                    overlay = np.where(exposed_mask, 1.0, np.nan)
                    ax.pcolormesh(
                        lon_edges,
                        lat_edges,
                        overlay,
                        shading="auto",
                        cmap="Reds",
                        alpha=0.35,
                        vmin=0.0,
                        vmax=1.0,
                    )

            if hit_lat.size and hit_lon.size:
                ax.scatter(hit_lon, hit_lat, s=14, c=color, alpha=0.75, edgecolors="none")
            else:
                ax.text(0.5, 0.5, "No ground-reaching rays", transform=ax.transAxes, ha="center", va="center")

            summary = payload["summary"]
            pop = dict(summary.get("population_analysis", {}) or {})
            ax.set_title(
                f"{route_class}\n"
                f"exposed={float(pop.get('total_exposed_population', 0.0)):.0f}, "
                f"hits={int(summary.get('num_ground_hits', 0))}"
            )
            ax.set_xlim(lon_min - lon_pad, lon_max + lon_pad)
            ax.set_ylim(lat_min - lat_pad, lat_max + lat_pad)
            ax.set_xlabel("Longitude (deg)")
            ax.set_ylabel("Latitude (deg)")
            ax.grid(alpha=0.15, linewidth=0.5)

        fig.suptitle(f"Predicted Boom Exposure Map: {scenario_id}")
        fig.tight_layout()
        output_path = output_dir / f"{scenario_id}_boom_exposure.png"
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        manifests[scenario_id] = str(output_path)

    return {"maps": manifests}


def _write_plots(
    *,
    rows: list[dict[str, Any]],
    pairwise_rows: list[dict[str, Any]],
    robustness_summary: dict[str, Any],
    plots_dir: Path,
):
    try:
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return

    core_rows = [r for r in rows if str(r.get("sensitivity_profile", "core")) == "core"]
    classes = _ordered_route_classes({str(r.get("route_class", "")) for r in core_rows})
    if not core_rows or not classes:
        return

    def _boxplot(metric: str, output_name: str, title: str):
        data = [
            [float(r.get(metric, 0.0)) for r in core_rows if str(r.get("route_class")) == route_class]
            for route_class in classes
        ]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.boxplot(data, labels=classes, showfliers=False)
        ax.set_title(title)
        ax.set_ylabel(metric)
        fig.tight_layout()
        fig.savefig(plots_dir / output_name, dpi=140)
        plt.close(fig)

    _boxplot("overflight_population", "overflight_population_boxplot.png", "Overflight Population by Class")
    _boxplot("boom_exposed_population", "boom_population_boxplot.png", "Boom-Exposed Population by Class")

    # Time tradeoff scatter.
    fig, ax = plt.subplots(figsize=(9, 6))
    for route_class in classes:
        subset = [r for r in core_rows if str(r.get("route_class")) == route_class]
        x = [float(r.get("overflight_population", 0.0)) for r in subset]
        y = [float(r.get("boom_exposed_population", 0.0)) for r in subset]
        s = [20.0 + 0.02 * float(r.get("elapsed_time_s", 0.0)) for r in subset]
        ax.scatter(x, y, s=s, alpha=0.65, label=route_class)
    ax.set_xlabel("overflight_population")
    ax.set_ylabel("boom_exposed_population")
    ax.set_title("Time-Weighted Tradeoff Scatter (core)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "time_tradeoff_scatter.png", dpi=140)
    plt.close(fig)

    # Pareto front view.
    fig, ax = plt.subplots(figsize=(9, 6))
    for route_class in classes:
        subset = [r for r in core_rows if str(r.get("route_class")) == route_class]
        x = [float(r.get("elapsed_time_s", 0.0)) for r in subset]
        y = [float(r.get("boom_exposed_population", 0.0)) for r in subset]
        ax.scatter(x, y, alpha=0.65, label=route_class)
    ax.set_xlabel("elapsed_time_s")
    ax.set_ylabel("boom_exposed_population")
    ax.set_title("Pareto View (core)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "pareto_front_core.png", dpi=140)
    plt.close(fig)

    # Robustness heatmap from sign consistency.
    sign_consistency = robustness_summary.get("sign_consistency", {})
    if sign_consistency:
        comp_keys = sorted(sign_consistency.keys())
        metric_keys = sorted({m for comp in sign_consistency.values() for m in comp.keys()})
        matrix = np.zeros((len(comp_keys), len(metric_keys)), dtype=np.float64)
        for i, comp in enumerate(comp_keys):
            for j, metric in enumerate(metric_keys):
                matrix[i, j] = float(sign_consistency.get(comp, {}).get(metric, 0.0))

        fig, ax = plt.subplots(figsize=(max(8, 0.5 * len(metric_keys)), 3 + 0.4 * len(comp_keys)))
        im = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_yticks(np.arange(len(comp_keys)))
        ax.set_yticklabels(comp_keys)
        ax.set_xticks(np.arange(len(metric_keys)))
        ax.set_xticklabels(metric_keys, rotation=70, ha="right")
        ax.set_title("Sign Consistency Robustness")
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        fig.tight_layout()
        fig.savefig(plots_dir / "robustness_heatmap.png", dpi=140)
        plt.close(fig)
