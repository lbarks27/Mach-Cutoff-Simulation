"""Aircraft point-mass and initial shock-ray generation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from ..config import AircraftConfig, ShockConfig
from ..core.geodesy import enu_basis, geodetic_to_ecef
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
    def __init__(self, flight_path: FlightPath, config: AircraftConfig):
        self.flight_path = flight_path
        self.config = config
        if config.mach <= 1.0:
            raise ValueError("Aircraft Mach must be > 1.0 for sonic boom simulation")

    def state_at(self, time_utc: datetime) -> AircraftState:
        path_state = self.flight_path.state_at(time_utc)
        lat = path_state["lat_deg"]
        lon = path_state["lon_deg"]

        alt = self.config.constant_altitude_m
        position_ecef = geodetic_to_ecef(lat, lon, alt)

        speed = self.config.mach * self.config.reference_sound_speed_mps
        tangent = np.asarray(path_state["tangent_ecef"], dtype=float)
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm == 0.0:
            tangent = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            tangent = tangent / tangent_norm

        velocity_ecef = tangent * speed
        return AircraftState(
            time_utc=time_utc,
            lat_deg=float(lat),
            lon_deg=float(lon),
            alt_m=float(alt),
            position_ecef_m=np.asarray(position_ecef, dtype=float),
            velocity_ecef_mps=np.asarray(velocity_ecef, dtype=float),
            speed_mps=float(speed),
            mach=float(self.config.mach),
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
    """Generate cone-distributed rays as wavefront normals trailing the aircraft."""
    rays_per_emission = int(shock_config.rays_per_emission)
    if rays_per_emission <= 0:
        raise ValueError("rays_per_emission must be positive")

    mach = aircraft_state.mach
    mach_angle = np.arcsin(1.0 / mach)
    launch_angle = 0.5 * np.pi - mach_angle

    vel_axis = aircraft_state.velocity_ecef_mps / np.linalg.norm(aircraft_state.velocity_ecef_mps)
    aft_axis = -vel_axis
    axis, b1, b2 = _orthonormal_basis_from_axis(aft_axis)

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
