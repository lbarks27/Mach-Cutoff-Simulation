"""WGS84 geodesy utilities used throughout the simulation."""

from __future__ import annotations

import numpy as np

from .constants import WGS84_A_M, WGS84_B_M, WGS84_E2, WGS84_EP2


def geodetic_to_ecef(lat_deg, lon_deg, alt_m):
    lat = np.deg2rad(np.asarray(lat_deg, dtype=float))
    lon = np.deg2rad(np.asarray(lon_deg, dtype=float))
    alt = np.asarray(alt_m, dtype=float)

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    cos_lon = np.cos(lon)
    sin_lon = np.sin(lon)

    N = WGS84_A_M / np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    x = (N + alt) * cos_lat * cos_lon
    y = (N + alt) * cos_lat * sin_lon
    z = (N * (1.0 - WGS84_E2) + alt) * sin_lat
    return np.stack([x, y, z], axis=-1)


def ecef_to_geodetic(x, y, z):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    lon = np.arctan2(y, x)
    p = np.sqrt(x * x + y * y)

    theta = np.arctan2(z * WGS84_A_M, p * WGS84_B_M)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    lat = np.arctan2(
        z + WGS84_EP2 * WGS84_B_M * sin_theta**3,
        p - WGS84_E2 * WGS84_A_M * cos_theta**3,
    )

    sin_lat = np.sin(lat)
    N = WGS84_A_M / np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    alt = p / np.cos(lat) - N

    for _ in range(3):
        sin_lat = np.sin(lat)
        N = WGS84_A_M / np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
        alt = p / np.cos(lat) - N
        lat = np.arctan2(z, p * (1.0 - WGS84_E2 * N / (N + alt)))

    return np.rad2deg(lat), np.rad2deg(lon), alt


def enu_basis(lat_deg: float, lon_deg: float):
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)

    east = np.array([-sin_lon, cos_lon, 0.0], dtype=float)
    north = np.array([-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat], dtype=float)
    up = np.array([cos_lat * cos_lon, cos_lat * sin_lon, sin_lat], dtype=float)
    return east, north, up


def enu_to_ecef(origin_lat_deg, origin_lon_deg, origin_alt_m, enu_xyz):
    origin_ecef = geodetic_to_ecef(origin_lat_deg, origin_lon_deg, origin_alt_m)
    east, north, up = enu_basis(origin_lat_deg, origin_lon_deg)

    enu = np.asarray(enu_xyz, dtype=float)
    e = enu[..., 0]
    n = enu[..., 1]
    u = enu[..., 2]

    out = (
        origin_ecef
        + np.expand_dims(e, axis=-1) * east
        + np.expand_dims(n, axis=-1) * north
        + np.expand_dims(u, axis=-1) * up
    )
    return out


def ecef_to_enu(origin_lat_deg, origin_lon_deg, origin_alt_m, ecef_xyz):
    origin_ecef = geodetic_to_ecef(origin_lat_deg, origin_lon_deg, origin_alt_m)
    east, north, up = enu_basis(origin_lat_deg, origin_lon_deg)

    vec = np.asarray(ecef_xyz, dtype=float) - origin_ecef
    e = np.tensordot(vec, east, axes=([-1], [0]))
    n = np.tensordot(vec, north, axes=([-1], [0]))
    u = np.tensordot(vec, up, axes=([-1], [0]))
    return np.stack([e, n, u], axis=-1)


def normalize_lon_deg(lon_deg):
    lon = np.asarray(lon_deg, dtype=float)
    return (lon + 180.0) % 360.0 - 180.0


def ray_ellipsoid_intersection(origin_ecef, direction_ecef):
    o = np.asarray(origin_ecef, dtype=float)
    d = np.asarray(direction_ecef, dtype=float)
    d = d / np.linalg.norm(d)

    a2 = WGS84_A_M**2
    b2 = WGS84_B_M**2

    A = (d[0] ** 2 + d[1] ** 2) / a2 + (d[2] ** 2) / b2
    B = 2.0 * ((o[0] * d[0] + o[1] * d[1]) / a2 + (o[2] * d[2]) / b2)
    C = (o[0] ** 2 + o[1] ** 2) / a2 + (o[2] ** 2) / b2 - 1.0

    disc = B * B - 4.0 * A * C
    if disc < 0.0:
        return None

    sqrt_disc = np.sqrt(disc)
    t1 = (-B - sqrt_disc) / (2.0 * A)
    t2 = (-B + sqrt_disc) / (2.0 * A)

    ts = [t for t in (t1, t2) if t > 0.0]
    if not ts:
        return None
    return min(ts)
