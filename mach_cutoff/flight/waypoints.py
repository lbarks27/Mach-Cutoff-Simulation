"""Waypoint ingestion and constant-altitude geodetic path interpolation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

from ..core.constants import WGS84_A_M
from ..core.geodesy import ecef_to_geodetic, geodetic_to_ecef, normalize_lon_deg


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


def _latlon_to_unit(lat_deg: float, lon_deg: float) -> np.ndarray:
    lat = np.deg2rad(float(lat_deg))
    lon = np.deg2rad(float(lon_deg))
    cos_lat = np.cos(lat)
    return np.array(
        [cos_lat * np.cos(lon), cos_lat * np.sin(lon), np.sin(lat)],
        dtype=float,
    )


def _unit_to_latlon(unit: np.ndarray) -> tuple[float, float]:
    u = np.asarray(unit, dtype=float).reshape(3)
    norm = float(np.linalg.norm(u))
    if norm <= 0.0:
        return 0.0, 0.0
    u = u / norm
    lat = float(np.arcsin(np.clip(u[2], -1.0, 1.0)))
    lon = float(np.arctan2(u[1], u[0]))
    return float(np.rad2deg(lat)), float(normalize_lon_deg(np.rad2deg(lon)))


def _angular_distance_rad(u0: np.ndarray, u1: np.ndarray) -> float:
    cos_a = float(np.clip(np.dot(u0, u1), -1.0, 1.0))
    return float(np.arccos(cos_a))


def _great_circle_unit(u0: np.ndarray, u1: np.ndarray, fraction: float) -> np.ndarray:
    """Spherical linear interpolation of unit vectors (great-circle)."""
    f = float(np.clip(fraction, 0.0, 1.0))
    u0 = np.asarray(u0, dtype=float).reshape(3)
    u1 = np.asarray(u1, dtype=float).reshape(3)
    n0 = float(np.linalg.norm(u0))
    n1 = float(np.linalg.norm(u1))
    if n0 <= 0.0 or n1 <= 0.0:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    u0 = u0 / n0
    u1 = u1 / n1
    cos_a = float(np.clip(np.dot(u0, u1), -1.0, 1.0))
    angle = float(np.arccos(cos_a))
    if angle < 1e-12:
        return u0.copy()
    # Nearly antipodal: fall back to linear then renorm.
    if abs(np.pi - angle) < 1e-8:
        u = (1.0 - f) * u0 + f * u1
        n = float(np.linalg.norm(u))
        return u / n if n > 0.0 else u0.copy()
    sin_a = float(np.sin(angle))
    s0 = float(np.sin((1.0 - f) * angle) / sin_a)
    s1 = float(np.sin(f * angle) / sin_a)
    u = s0 * u0 + s1 * u1
    n = float(np.linalg.norm(u))
    return u / n if n > 0.0 else u0.copy()


def _geodetic_on_segment(
    lat0: float,
    lon0: float,
    alt0: float,
    lat1: float,
    lon1: float,
    alt1: float,
    fraction: float,
    u0: np.ndarray | None = None,
    u1: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Great-circle lat/lon + linear altitude between two geodetic waypoints."""
    f = float(np.clip(fraction, 0.0, 1.0))
    if u0 is None:
        u0 = _latlon_to_unit(lat0, lon0)
    if u1 is None:
        u1 = _latlon_to_unit(lat1, lon1)
    lat, lon = _unit_to_latlon(_great_circle_unit(u0, u1, f))
    alt = float((1.0 - f) * alt0 + f * alt1)
    return lat, lon, alt


def _tangent_ecef_on_segment(
    lat0: float,
    lon0: float,
    alt0: float,
    lat1: float,
    lon1: float,
    alt1: float,
    fraction: float,
    u0: np.ndarray,
    u1: np.ndarray,
) -> np.ndarray:
    """Unit ECEF tangent along the constant-alt (linear-alt) great-circle path."""
    f = float(np.clip(fraction, 0.0, 1.0))
    # Finite difference along the geodetic path (not ECEF chord).
    df = 1e-5
    f_a = max(0.0, f - df)
    f_b = min(1.0, f + df)
    if f_b <= f_a:
        f_a, f_b = 0.0, min(1.0, 1e-5)
    lat_a, lon_a, alt_a = _geodetic_on_segment(lat0, lon0, alt0, lat1, lon1, alt1, f_a, u0, u1)
    lat_b, lon_b, alt_b = _geodetic_on_segment(lat0, lon0, alt0, lat1, lon1, alt1, f_b, u0, u1)
    p_a = geodetic_to_ecef(lat_a, lon_a, alt_a).reshape(3)
    p_b = geodetic_to_ecef(lat_b, lon_b, alt_b).reshape(3)
    v = p_b - p_a
    n = float(np.linalg.norm(v))
    if n <= 0.0:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return v / n


class FlightPath:
    """Time-parameterized flight path with constant-altitude geodetic segments.

    Horizontal motion follows a great-circle between waypoint lat/lon. Altitude is
    linearly interpolated between endpoints, so equal endpoint altitudes stay
    constant along the segment (no ECEF-chord altitude dip).
    """

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

        self._lats = np.asarray([wp.lat_deg for wp in self.waypoints], dtype=float)
        self._lons = np.asarray([wp.lon_deg for wp in self.waypoints], dtype=float)
        self._alts = np.asarray([wp.alt_m for wp in self.waypoints], dtype=float)
        self._unit = np.asarray(
            [_latlon_to_unit(lat, lon) for lat, lon in zip(self._lats, self._lons, strict=True)],
            dtype=float,
        )
        # Keep endpoint ECEF for projection bookkeeping / convenience.
        self._ecef = np.asarray(
            [
                geodetic_to_ecef(wp.lat_deg, wp.lon_deg, wp.alt_m).reshape(3)
                for wp in self.waypoints
            ],
            dtype=float,
        )

        n_seg = len(self.waypoints) - 1
        segment_lengths = np.zeros(n_seg, dtype=float)
        angles = np.zeros(n_seg, dtype=float)
        for i in range(n_seg):
            angle = _angular_distance_rad(self._unit[i], self._unit[i + 1])
            angles[i] = angle
            # Arc length at mean altitude (spherical approximation of WGS84 radius).
            mean_alt = 0.5 * (float(self._alts[i]) + float(self._alts[i + 1]))
            segment_lengths[i] = float(angle * (WGS84_A_M + mean_alt))

        self._segment_angles_rad = angles
        self._segment_lengths_m = segment_lengths
        self._cum_length_m = np.concatenate([[0.0], np.cumsum(segment_lengths)])

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
        return int(len(self._segment_lengths_m))

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

    def _state_on_segment(self, i: int, fraction: float) -> dict:
        f = float(np.clip(fraction, 0.0, 1.0))
        lat0 = float(self._lats[i])
        lon0 = float(self._lons[i])
        alt0 = float(self._alts[i])
        lat1 = float(self._lats[i + 1])
        lon1 = float(self._lons[i + 1])
        alt1 = float(self._alts[i + 1])
        u0 = self._unit[i]
        u1 = self._unit[i + 1]

        lat, lon, alt = _geodetic_on_segment(lat0, lon0, alt0, lat1, lon1, alt1, f, u0, u1)
        ecef = geodetic_to_ecef(lat, lon, alt).reshape(3)
        tangent = _tangent_ecef_on_segment(lat0, lon0, alt0, lat1, lon1, alt1, f, u0, u1)
        return {
            "ecef_m": np.asarray(ecef, dtype=float),
            "lat_deg": float(lat),
            "lon_deg": float(lon),
            "alt_m": float(alt),
            "tangent_ecef": np.asarray(tangent, dtype=float),
            "segment_index": int(i),
            "segment_fraction": float(f),
        }

    def state_at(self, time_utc: datetime):
        t = time_utc.timestamp()
        i = self._segment_index(t)

        t0 = self._times[i]
        t1 = self._times[i + 1]
        f = 0.0 if t1 == t0 else float(np.clip((t - t0) / (t1 - t0), 0.0, 1.0))
        return self._state_on_segment(i, f)

    def state_at_distance(self, distance_m: float):
        s = float(np.clip(distance_m, 0.0, self._cum_length_m[-1]))
        i = self._segment_index_for_distance(s)

        s0 = float(self._cum_length_m[i])
        s1 = float(self._cum_length_m[i + 1])
        f = 0.0 if s1 == s0 else float(np.clip((s - s0) / (s1 - s0), 0.0, 1.0))
        state = self._state_on_segment(i, f)
        state["distance_m"] = s
        return state

    def project_ecef(self, ecef_m: np.ndarray):
        """Project an ECEF point onto the piecewise-geodesic route."""
        p = np.asarray(ecef_m, dtype=float).reshape(3)
        # Spherical unit direction of query (for along-track fraction).
        lat_q, lon_q, alt_q = ecef_to_geodetic(p[0], p[1], p[2])
        u_q = _latlon_to_unit(float(lat_q), float(lon_q))

        best_i = 0
        best_f = 0.0
        best_point = self._ecef[0].copy()
        best_dist2 = float(np.inf)

        for i in range(self.segment_count):
            u0 = self._unit[i]
            u1 = self._unit[i + 1]
            angle = float(self._segment_angles_rad[i])
            if angle < 1e-12:
                candidate_state = self._state_on_segment(i, 0.0)
                candidate = candidate_state["ecef_m"]
                f = 0.0
            else:
                # Project query onto the great-circle plane of the segment.
                normal = np.cross(u0, u1)
                n_norm = float(np.linalg.norm(normal))
                if n_norm <= 1e-12:
                    f = 0.0
                else:
                    normal = normal / n_norm
                    u_plane = u_q - float(np.dot(u_q, normal)) * normal
                    plane_norm = float(np.linalg.norm(u_plane))
                    if plane_norm <= 1e-12:
                        f = 0.0
                    else:
                        u_plane = u_plane / plane_norm
                        # Fraction along arc from angle from u0 toward u1.
                        # Use atan2 form for robust signed progress on the arc.
                        e1 = u0
                        e2 = np.cross(normal, u0)
                        e2 = e2 / max(float(np.linalg.norm(e2)), 1e-15)
                        ang_q = float(np.arctan2(np.dot(u_plane, e2), np.dot(u_plane, e1)))
                        # Segment spans [0, angle] in this basis (positive toward u1).
                        if ang_q < 0.0:
                            # Prefer the shorter direction; clamp to endpoints.
                            if abs(ang_q) <= abs(ang_q - angle):
                                ang_q = 0.0
                            else:
                                ang_q = angle
                        f = float(np.clip(ang_q / angle, 0.0, 1.0))
                candidate_state = self._state_on_segment(i, f)
                candidate = candidate_state["ecef_m"]

            delta = p - candidate
            dist2 = float(np.dot(delta, delta))
            if dist2 < best_dist2:
                best_dist2 = dist2
                best_i = i
                best_f = f
                best_point = np.asarray(candidate, dtype=float)

        state = self._state_on_segment(best_i, best_f)
        along_distance_m = float(self._cum_length_m[best_i] + best_f * self._segment_lengths_m[best_i])
        return {
            "distance_m": along_distance_m,
            "cross_track_m": float(np.sqrt(max(best_dist2, 0.0))),
            "segment_index": int(best_i),
            "segment_fraction": float(best_f),
            "nearest_ecef_m": np.asarray(best_point, dtype=float),
            "nearest_lat_deg": float(state["lat_deg"]),
            "nearest_lon_deg": float(state["lon_deg"]),
            "nearest_alt_m": float(state["alt_m"]),
            "tangent_ecef": np.asarray(state["tangent_ecef"], dtype=float),
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
