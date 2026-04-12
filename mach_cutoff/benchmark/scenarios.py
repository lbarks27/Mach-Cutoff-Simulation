"""Scenario expansion helpers for the benchmark pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from .config import BenchmarkConfig, CorridorConfig


@dataclass(slots=True)
class BenchmarkScenario:
    scenario_id: str
    corridor_id: str
    timestamp_utc: str
    timestamp_dt: datetime
    origin_lat_deg: float
    origin_lon_deg: float
    destination_lat_deg: float
    destination_lon_deg: float
    origin_name: str
    destination_name: str


def parse_utc_timestamp(value: str) -> datetime:
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def format_scenario_timestamp(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y%m%dT%H%MZ")


def scenario_id(corridor_id: str, timestamp_utc: str) -> str:
    dt = parse_utc_timestamp(timestamp_utc)
    return f"{corridor_id}__{format_scenario_timestamp(dt)}"


def build_core_scenarios(cfg: BenchmarkConfig) -> list[BenchmarkScenario]:
    scenarios: list[BenchmarkScenario] = []
    for corridor_id, corridor in cfg.corridors.items():
        scenarios.extend(_scenarios_for_corridor(corridor_id, corridor, cfg.timestamps_utc))
    return scenarios


def _scenarios_for_corridor(
    corridor_id: str,
    corridor: CorridorConfig,
    timestamps_utc: list[str],
) -> list[BenchmarkScenario]:
    rows: list[BenchmarkScenario] = []
    for ts in timestamps_utc:
        dt = parse_utc_timestamp(ts)
        sid = f"{corridor_id}__{format_scenario_timestamp(dt)}"
        rows.append(
            BenchmarkScenario(
                scenario_id=sid,
                corridor_id=corridor_id,
                timestamp_utc=dt.isoformat().replace("+00:00", "Z"),
                timestamp_dt=dt,
                origin_lat_deg=float(corridor.origin_lat_deg),
                origin_lon_deg=float(corridor.origin_lon_deg),
                destination_lat_deg=float(corridor.destination_lat_deg),
                destination_lon_deg=float(corridor.destination_lon_deg),
                origin_name=str(corridor.origin_name),
                destination_name=str(corridor.destination_name),
            )
        )
    return rows
