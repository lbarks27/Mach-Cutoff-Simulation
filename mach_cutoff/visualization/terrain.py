"""Terrain helpers shared across visualization backends."""

from __future__ import annotations

import numpy as np

from ..simulation.outputs import SimulationResult


def terrain_grid_from_result(result: SimulationResult):
    lat = result.terrain_lat_deg
    lon = result.terrain_lon_deg
    elev = result.terrain_elevation_m
    if lat is None or lon is None or elev is None:
        return None

    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    elev = np.asarray(elev, dtype=float)

    if lat.ndim != 2 or lon.ndim != 2 or elev.ndim != 2:
        return None
    if not (lat.shape == lon.shape == elev.shape):
        return None

    finite = np.isfinite(elev)
    if not np.any(finite):
        return None
    if not np.all(finite):
        elev = np.where(finite, elev, np.nanmedian(elev[finite]))

    return lat, lon, elev


def downsample_terrain_grid(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    elevation_m: np.ndarray,
    *,
    max_points_per_axis: int = 140,
):
    if max_points_per_axis <= 0:
        raise ValueError("max_points_per_axis must be > 0")

    step_y = max(1, int(np.ceil(lat_deg.shape[0] / max_points_per_axis)))
    step_x = max(1, int(np.ceil(lat_deg.shape[1] / max_points_per_axis)))
    return (
        lat_deg[::step_y, ::step_x],
        lon_deg[::step_y, ::step_x],
        elevation_m[::step_y, ::step_x],
    )
