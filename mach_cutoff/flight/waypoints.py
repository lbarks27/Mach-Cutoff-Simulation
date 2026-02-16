"""Waypoint ingestion and time interpolation utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

from ..core.geodesy import ecef_to_geodetic, geodetic_to_ecef


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
        self._segment_vectors = self._ecef[1:] - self._ecef[:-1]
        self._segment_lengths_m = np.linalg.norm(self._segment_vectors, axis=1)
        self._cum_length_m = np.concatenate([[0.0], np.cumsum(self._segment_lengths_m)])

    @property
    def start_time(self) -> datetime:
        return self.waypoints[0].time_utc

    @property
    def end_time(self) -> datetime:
        return self.waypoints[-1].time_utc

    @property
    def duration_s(self) -> float:
        return float(self._times[-1] - self._times[0])

    @property
    def total_length_m(self) -> float:
        return float(self._cum_length_m[-1])

    @property
    def segment_count(self) -> int:
        return int(len(self._segment_vectors))

    def _segment_index(self, t_epoch: float) -> int:
        if t_epoch <= self._times[0]:
            return 0
        if t_epoch >= self._times[-1]:
            return len(self._times) - 2
        idx = int(np.searchsorted(self._times, t_epoch, side="right") - 1)
        return max(0, min(idx, len(self._times) - 2))

    def _segment_index_for_distance(self, distance_m: float) -> int:
        if distance_m <= 0.0:
            return 0
        if distance_m >= self._cum_length_m[-1]:
            return len(self._cum_length_m) - 2
        idx = int(np.searchsorted(self._cum_length_m, distance_m, side="right") - 1)
        return max(0, min(idx, len(self._cum_length_m) - 2))

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

    def state_at_distance(self, distance_m: float):
        s = float(np.clip(distance_m, 0.0, self._cum_length_m[-1]))
        i = self._segment_index_for_distance(s)

        s0 = self._cum_length_m[i]
        s1 = self._cum_length_m[i + 1]
        f = 0.0 if s1 == s0 else np.clip((s - s0) / (s1 - s0), 0.0, 1.0)

        p0 = self._ecef[i]
        p1 = self._ecef[i + 1]
        p = (1.0 - f) * p0 + f * p1

        v = self._segment_vectors[i]
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
            "distance_m": s,
        }

    def project_ecef(self, ecef_m: np.ndarray):
        """Project an ECEF point onto the piecewise-linear route."""
        p = np.asarray(ecef_m, dtype=float).reshape(3)
        best_i = 0
        best_f = 0.0
        best_point = self._ecef[0]
        best_dist2 = np.inf

        for i, seg in enumerate(self._segment_vectors):
            p0 = self._ecef[i]
            seg_len2 = float(np.dot(seg, seg))
            if seg_len2 <= 1e-12:
                f = 0.0
                candidate = p0
            else:
                f = float(np.clip(np.dot(p - p0, seg) / seg_len2, 0.0, 1.0))
                candidate = p0 + f * seg
            dist2 = float(np.dot(p - candidate, p - candidate))
            if dist2 < best_dist2:
                best_dist2 = dist2
                best_i = i
                best_f = f
                best_point = candidate

        seg_vec = self._segment_vectors[best_i]
        seg_norm = float(np.linalg.norm(seg_vec))
        tangent = seg_vec / seg_norm if seg_norm > 0.0 else np.array([1.0, 0.0, 0.0], dtype=float)
        along_distance_m = float(self._cum_length_m[best_i] + best_f * self._segment_lengths_m[best_i])
        cross_track_m = float(np.sqrt(max(best_dist2, 0.0)))

        lat, lon, alt = ecef_to_geodetic(best_point[0], best_point[1], best_point[2])
        return {
            "distance_m": along_distance_m,
            "cross_track_m": cross_track_m,
            "segment_index": best_i,
            "segment_fraction": float(best_f),
            "nearest_ecef_m": np.asarray(best_point, dtype=float),
            "nearest_lat_deg": float(lat),
            "nearest_lon_deg": float(lon),
            "nearest_alt_m": float(alt),
            "tangent_ecef": tangent,
        }

    def sample_times(
        self,
        step_s: float,
        start: datetime | None = None,
        end: datetime | None = None,
        *,
        clamp_to_path_bounds: bool = True,
    ):
        if step_s <= 0:
            raise ValueError("step_s must be positive")

        start_epoch = self._times[0] if start is None else float(start.timestamp())
        end_epoch = self._times[-1] if end is None else float(end.timestamp())
        if clamp_to_path_bounds:
            start_epoch = max(self._times[0], start_epoch)
            end_epoch = min(self._times[-1], end_epoch)
        if end_epoch < start_epoch:
            return []

        values = np.arange(start_epoch, end_epoch + 1e-9, step_s, dtype=float)
        if values.size == 0:
            values = np.array([start_epoch], dtype=float)
        if end_epoch - values[-1] > 1e-6:
            values = np.append(values, end_epoch)
        return [datetime.fromtimestamp(float(t), tz=timezone.utc) for t in values]
