"""Waypoint ingestion and time interpolation utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

from .geodesy import ecef_to_geodetic, geodetic_to_ecef


@dataclass(frozen=True, slots=True)
class Waypoint:
    lat_deg: float
    lon_deg: float
    alt_m: float
    time_utc: datetime


def _parse_time_iso(value: str) -> datetime:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def load_waypoints_json(path: str | Path) -> list[Waypoint]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Waypoint file not found: {p}")

    with p.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        points = raw.get("waypoints")
    else:
        points = raw

    if not isinstance(points, list):
        raise ValueError("Waypoint JSON must be a list or an object with key 'waypoints'")

    waypoints: list[Waypoint] = []
    for i, item in enumerate(points):
        if not isinstance(item, dict):
            raise ValueError(f"Waypoint {i} must be a JSON object")
        waypoints.append(
            Waypoint(
                lat_deg=float(item["lat"]),
                lon_deg=float(item["lon"]),
                alt_m=float(item["alt_m"]),
                time_utc=_parse_time_iso(str(item["time_utc"])),
            )
        )

    if len(waypoints) < 2:
        raise ValueError("At least two waypoints are required")

    waypoints = sorted(waypoints, key=lambda w: w.time_utc)
    return waypoints


class FlightPath:
    """Time-parameterized flight path from geodetic waypoints."""

    def __init__(self, waypoints: Iterable[Waypoint]):
        self.waypoints = list(waypoints)
        if len(self.waypoints) < 2:
            raise ValueError("FlightPath requires at least two waypoints")

        self._times = np.array(
            [wp.time_utc.timestamp() for wp in self.waypoints],
            dtype=float,
        )
        if not np.all(np.diff(self._times) > 0.0):
            raise ValueError("Waypoint times must be strictly increasing")

        ecef = [geodetic_to_ecef(wp.lat_deg, wp.lon_deg, wp.alt_m) for wp in self.waypoints]
        self._ecef = np.asarray(ecef, dtype=float)

    @property
    def start_time(self) -> datetime:
        return self.waypoints[0].time_utc

    @property
    def end_time(self) -> datetime:
        return self.waypoints[-1].time_utc

    @property
    def duration_s(self) -> float:
        return float(self._times[-1] - self._times[0])

    def _segment_index(self, t_epoch: float) -> int:
        if t_epoch <= self._times[0]:
            return 0
        if t_epoch >= self._times[-1]:
            return len(self._times) - 2
        idx = int(np.searchsorted(self._times, t_epoch, side="right") - 1)
        return max(0, min(idx, len(self._times) - 2))

    def state_at(self, time_utc: datetime):
        t = time_utc.timestamp()
        i = self._segment_index(t)

        t0 = self._times[i]
        t1 = self._times[i + 1]
        f = 0.0 if t1 == t0 else np.clip((t - t0) / (t1 - t0), 0.0, 1.0)

        p0 = self._ecef[i]
        p1 = self._ecef[i + 1]
        p = (1.0 - f) * p0 + f * p1

        v = p1 - p0
        v_norm = np.linalg.norm(v)
        tangent = v / v_norm if v_norm > 0.0 else np.array([1.0, 0.0, 0.0], dtype=float)

        lat, lon, alt = ecef_to_geodetic(p[0], p[1], p[2])
        return {
            "ecef_m": p,
            "lat_deg": float(lat),
            "lon_deg": float(lon),
            "alt_m": float(alt),
            "tangent_ecef": tangent,
            "segment_index": i,
            "segment_fraction": float(f),
        }

    def sample_times(self, step_s: float, start: datetime | None = None, end: datetime | None = None):
        if step_s <= 0:
            raise ValueError("step_s must be positive")

        start_epoch = self._times[0] if start is None else max(self._times[0], start.timestamp())
        end_epoch = self._times[-1] if end is None else min(self._times[-1], end.timestamp())
        if end_epoch < start_epoch:
            return []

        values = np.arange(start_epoch, end_epoch + 1e-9, step_s)
        return [datetime.fromtimestamp(t, tz=timezone.utc) for t in values]
