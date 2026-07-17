"""Aircraft point-mass and initial shock-ray generation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from ..config import AircraftConfig, ShockConfig
from ..core.geodesy import enu_basis
from .waypoints import FlightPath


@dataclass(slots=True)
class AircraftState:
    time_utc: datetime
    lat_deg: float
    lon_deg: float
    alt_m: float
    position_ecef_m: np.ndarray
    velocity_ecef_mps: np.ndarray
    speed_mps: float
    mach: float


class PointMassAircraft:
    """Open-loop point-mass aircraft that follows a straight-line waypoint path.

    Position is piecewise-linear between waypoints. Segment ground speed is
    derived from waypoint times (segment length / segment duration). There is
    no closed-loop guidance: the aircraft simply tracks the prescribed schedule.
    """

    def __init__(self, flight_path: FlightPath, config: AircraftConfig):
        self.flight_path = flight_path
        self.config = config
        self._ref_sound_speed_mps = float(config.reference_sound_speed_mps)
        if self._ref_sound_speed_mps <= 0.0:
            raise ValueError("aircraft.reference_sound_speed_mps must be positive")

        segment_lengths_m = np.asarray(flight_path._segment_lengths_m, dtype=float)
        times = np.asarray(flight_path._times, dtype=float)
        dt_s = np.diff(times)
        if np.any(dt_s <= 0.0):
            raise ValueError("Waypoint times must be strictly increasing for open-loop flight")

        # Ground speed per straight-line segment from the schedule.
        self._segment_speed_mps = (segment_lengths_m / dt_s).astype(float)
        if np.any(~np.isfinite(self._segment_speed_mps)):
            raise ValueError("Non-finite segment speeds from waypoint schedule")

        self._start_time = self.flight_path.start_time
        self._end_time = self.flight_path.end_time
        self._duration_s = float(self.flight_path.duration_s)
        if self._duration_s <= 0.0:
            raise ValueError("Flight path duration must be positive")

    @property
    def start_time(self) -> datetime:
        return self._start_time

    @property
    def end_time(self) -> datetime:
        return self._end_time

    @property
    def duration_s(self) -> float:
        return self._duration_s

    def state_at(self, time_utc: datetime) -> AircraftState:
        # Clamp sampling to the scheduled route window; hold endpoints outside it.
        t_epoch = float(time_utc.timestamp())
        t0 = float(self.flight_path._times[0])
        t1 = float(self.flight_path._times[-1])
        sample_epoch = float(np.clip(t_epoch, t0, t1))
        sample_time = datetime.fromtimestamp(sample_epoch, tz=time_utc.tzinfo or self._start_time.tzinfo)

        path_state = self.flight_path.state_at(sample_time)
        segment_index = int(path_state["segment_index"])
        segment_index = max(0, min(segment_index, len(self._segment_speed_mps) - 1))

        speed = float(self._segment_speed_mps[segment_index])
        # Zero-length segments (hover / hold) produce zero speed.
        if not np.isfinite(speed) or speed < 0.0:
            speed = 0.0
        mach = float(speed / self._ref_sound_speed_mps)

        tangent = np.asarray(path_state["tangent_ecef"], dtype=float)
        tangent_norm = float(np.linalg.norm(tangent))
        if tangent_norm <= 0.0:
            unit_tangent = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            unit_tangent = tangent / tangent_norm
        velocity_ecef = unit_tangent * speed if speed > 0.0 else np.zeros(3, dtype=float)

        return AircraftState(
            time_utc=time_utc,
            lat_deg=float(path_state["lat_deg"]),
            lon_deg=float(path_state["lon_deg"]),
            alt_m=float(path_state["alt_m"]),
            position_ecef_m=np.asarray(path_state["ecef_m"], dtype=float),
            velocity_ecef_mps=np.asarray(velocity_ecef, dtype=float),
            speed_mps=float(speed),
            mach=float(mach),
        )


def _orthonormal_basis_from_axis(axis: np.ndarray):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)

    if abs(axis[2]) < 0.9:
        ref = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        ref = np.array([1.0, 0.0, 0.0], dtype=float)

    b1 = np.cross(axis, ref)
    b1 /= np.linalg.norm(b1)
    b2 = np.cross(axis, b1)
    b2 /= np.linalg.norm(b2)
    return axis, b1, b2


def generate_shock_directions(
    aircraft_state: AircraftState,
    shock_config: ShockConfig,
):
    """Generate cone-distributed rays in the selected reference frame."""
    rays_per_emission = int(shock_config.rays_per_emission)
    if rays_per_emission <= 0:
        raise ValueError("rays_per_emission must be positive")

    mach = aircraft_state.mach
    if mach <= 1.0:
        raise ValueError("generate_shock_directions requires supersonic aircraft mach > 1.0")
    mach_angle = np.arcsin(1.0 / mach)
    launch_angle = 0.5 * np.pi - mach_angle

    mode = str(shock_config.direction_reference).strip().lower()
    if mode in {"earth_down", "earth"}:
        _, _, up = enu_basis(aircraft_state.lat_deg, aircraft_state.lon_deg)
        axis = -up
    elif mode in {"aircraft_forward", "aircraft", "legacy"}:
        vel_axis = aircraft_state.velocity_ecef_mps / np.linalg.norm(aircraft_state.velocity_ecef_mps)
        axis = vel_axis
    elif mode in {"aircraft_aft"}:
        vel_axis = aircraft_state.velocity_ecef_mps / np.linalg.norm(aircraft_state.velocity_ecef_mps)
        axis = -vel_axis
    else:
        raise ValueError(
            f"Unsupported shock.direction_reference '{shock_config.direction_reference}'. "
            "Use 'earth_down', 'aircraft_forward', or 'aircraft_aft'."
        )

    axis, b1, b2 = _orthonormal_basis_from_axis(axis)

    az0 = np.deg2rad(shock_config.azimuth_offset_deg)
    azimuths = az0 + np.linspace(0.0, 2.0 * np.pi, rays_per_emission, endpoint=False)

    dirs = []
    for az in azimuths:
        radial = np.cos(az) * b1 + np.sin(az) * b2
        d = np.cos(launch_angle) * axis + np.sin(launch_angle) * radial
        d = d / np.linalg.norm(d)
        dirs.append(d)

    directions = np.asarray(dirs, dtype=float)

    if shock_config.downward_only:
        _, _, up = enu_basis(aircraft_state.lat_deg, aircraft_state.lon_deg)
        down_mask = np.dot(directions, up) < 0.0
        filtered = directions[down_mask]
        if len(filtered) == 0:
            return directions
        return filtered

    return directions
