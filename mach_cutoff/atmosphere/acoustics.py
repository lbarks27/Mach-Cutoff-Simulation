"""Acoustic effective-index field construction."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import GridConfig
from ..core.constants import GAMMA_DRY_AIR, R_DRY_AIR
from ..core.geodesy import ecef_to_geodetic, ecef_to_enu, enu_to_ecef
from .interpolation import HRRRInterpolator


def saturation_vapor_pressure_pa(temperature_k: np.ndarray) -> np.ndarray:
    # Bolton-like fit, valid for tropospheric ranges.
    t_c = temperature_k - 273.15
    return 611.2 * np.exp((17.67 * t_c) / (t_c + 243.5))


def compute_sound_speed_mps(
    temperature_k: np.ndarray,
    pressure_hpa: np.ndarray,
    relative_humidity_pct: np.ndarray,
) -> np.ndarray:
    """Approximate moist-air speed of sound."""
    t = np.asarray(temperature_k, dtype=np.float32)
    p_pa = np.asarray(pressure_hpa, dtype=np.float32) * 100.0
    rh = np.clip(np.asarray(relative_humidity_pct, dtype=np.float32), 0.0, 100.0) / 100.0

    e_sat = saturation_vapor_pressure_pa(t)
    e = np.clip(rh * e_sat, 0.0, 0.99 * p_pa)
    mixing_ratio = 0.622 * e / np.maximum(p_pa - e, 1.0)
    specific_humidity = mixing_ratio / (1.0 + mixing_ratio)

    r_moist = R_DRY_AIR * (1.0 + 0.61 * specific_humidity)
    return np.sqrt(GAMMA_DRY_AIR * r_moist * t)


def _manual_trilinear(grid_x, grid_y, grid_z, values, point):
    x, y, z = point

    def locate(grid, q):
        if q <= grid[0]:
            return 0, 1, 0.0
        if q >= grid[-1]:
            return len(grid) - 2, len(grid) - 1, 1.0
        i1 = int(np.searchsorted(grid, q, side="right"))
        i0 = i1 - 1
        w = (q - grid[i0]) / (grid[i1] - grid[i0])
        return i0, i1, float(w)

    x0, x1, wx = locate(grid_x, x)
    y0, y1, wy = locate(grid_y, y)
    z0, z1, wz = locate(grid_z, z)

    c000 = values[x0, y0, z0]
    c001 = values[x0, y0, z1]
    c010 = values[x0, y1, z0]
    c011 = values[x0, y1, z1]
    c100 = values[x1, y0, z0]
    c101 = values[x1, y0, z1]
    c110 = values[x1, y1, z0]
    c111 = values[x1, y1, z1]

    c00 = c000 * (1.0 - wx) + c100 * wx
    c01 = c001 * (1.0 - wx) + c101 * wx
    c10 = c010 * (1.0 - wx) + c110 * wx
    c11 = c011 * (1.0 - wx) + c111 * wx

    c0 = c00 * (1.0 - wy) + c10 * wy
    c1 = c01 * (1.0 - wy) + c11 * wy

    return float(c0 * (1.0 - wz) + c1 * wz)


@dataclass(slots=True)
class AcousticGridField:
    x_m: np.ndarray
    y_m: np.ndarray
    z_m: np.ndarray
    n_grid: np.ndarray
    grad_x: np.ndarray
    grad_y: np.ndarray
    grad_z: np.ndarray
    domain_bounds: np.ndarray = field(init=False, repr=False)
    _interp: object = field(init=False, default=None, repr=False)
    _gx_interp: object = field(init=False, default=None, repr=False)
    _gy_interp: object = field(init=False, default=None, repr=False)
    _gz_interp: object = field(init=False, default=None, repr=False)

    def __post_init__(self):
        self.domain_bounds = np.array(
            [
                [self.x_m[0], self.x_m[-1]],
                [self.y_m[0], self.y_m[-1]],
                [self.z_m[0], self.z_m[-1]],
            ],
            dtype=float,
        )

        try:
            from scipy.interpolate import RegularGridInterpolator

            opts = {"bounds_error": False, "fill_value": None}
            self._interp = RegularGridInterpolator((self.x_m, self.y_m, self.z_m), self.n_grid, **opts)
            self._gx_interp = RegularGridInterpolator((self.x_m, self.y_m, self.z_m), self.grad_x, **opts)
            self._gy_interp = RegularGridInterpolator((self.x_m, self.y_m, self.z_m), self.grad_y, **opts)
            self._gz_interp = RegularGridInterpolator((self.x_m, self.y_m, self.z_m), self.grad_z, **opts)
        except Exception:
            self._interp = None

    def n(self, xyz_local_m: np.ndarray) -> float:
        p = np.asarray(xyz_local_m, dtype=float).reshape(-1)
        if self._interp is not None:
            return float(self._interp(p[None, :])[0])
        return _manual_trilinear(self.x_m, self.y_m, self.z_m, self.n_grid, p)

    def grad_n(self, xyz_local_m: np.ndarray) -> np.ndarray:
        p = np.asarray(xyz_local_m, dtype=float).reshape(-1)
        if self._gx_interp is not None:
            gx = float(self._gx_interp(p[None, :])[0])
            gy = float(self._gy_interp(p[None, :])[0])
            gz = float(self._gz_interp(p[None, :])[0])
            return np.array([gx, gy, gz], dtype=float)

        gx = _manual_trilinear(self.x_m, self.y_m, self.z_m, self.grad_x, p)
        gy = _manual_trilinear(self.x_m, self.y_m, self.z_m, self.grad_y, p)
        gz = _manual_trilinear(self.x_m, self.y_m, self.z_m, self.grad_z, p)
        return np.array([gx, gy, gz], dtype=float)


def _wind_projection(
    u_wind_mps: np.ndarray,
    v_wind_mps: np.ndarray,
    velocity_enu: np.ndarray,
    mode: str,
) -> np.ndarray:
    mode = mode.lower()
    if mode == "none":
        return np.zeros_like(u_wind_mps)
    if mode == "eastward":
        return u_wind_mps
    if mode == "northward":
        return v_wind_mps

    vx, vy, vz = velocity_enu
    if mode == "along_track_3d":
        norm = np.linalg.norm([vx, vy, vz])
        if norm <= 1e-6:
            return np.zeros_like(u_wind_mps)
        return u_wind_mps * (vx / norm) + v_wind_mps * (vy / norm)

    horiz = np.array([vx, vy], dtype=np.float32)
    norm_h = np.linalg.norm(horiz)
    if norm_h <= 1e-6:
        return np.zeros_like(u_wind_mps)
    hx, hy = horiz / norm_h
    return u_wind_mps * hx + v_wind_mps * hy


def build_acoustic_grid_field(
    hrrr_interp: HRRRInterpolator,
    aircraft_lat_deg: float,
    aircraft_lon_deg: float,
    aircraft_alt_m: float,
    aircraft_velocity_ecef_mps: np.ndarray,
    grid_config: GridConfig,
    reference_speed_mps: float,
):
    x = np.linspace(-grid_config.half_width_east_m, grid_config.half_width_east_m, grid_config.nx, dtype=np.float32)
    y = np.linspace(-grid_config.half_width_north_m, grid_config.half_width_north_m, grid_config.ny, dtype=np.float32)
    alt_abs = np.linspace(grid_config.min_altitude_m, grid_config.max_altitude_m, grid_config.nz, dtype=np.float32)
    z_rel = alt_abs - aircraft_alt_m

    X, Y = np.meshgrid(x, y, indexing="xy")
    local_xy0 = np.column_stack([X.reshape(-1), Y.reshape(-1), np.zeros(X.size, dtype=np.float32)])
    ecef_xy = enu_to_ecef(aircraft_lat_deg, aircraft_lon_deg, aircraft_alt_m, local_xy0)
    lat_xy, lon_xy, _ = ecef_to_geodetic(ecef_xy[:, 0], ecef_xy[:, 1], ecef_xy[:, 2])

    cols = hrrr_interp.sample_columns(lat_xy, lon_xy, alt_abs)

    temp = cols["temperature_k"]
    rh = cols["relative_humidity_pct"]
    u = cols["u_wind_mps"]
    v = cols["v_wind_mps"]
    p = cols["pressure_hpa"]

    c = compute_sound_speed_mps(temp, p, rh)

    origin_ecef = enu_to_ecef(
        aircraft_lat_deg,
        aircraft_lon_deg,
        aircraft_alt_m,
        np.array([0.0, 0.0, 0.0]),
    )
    velocity_enu = ecef_to_enu(
        aircraft_lat_deg,
        aircraft_lon_deg,
        aircraft_alt_m,
        origin_ecef + aircraft_velocity_ecef_mps,
    )
    velocity_enu = np.asarray(velocity_enu, dtype=np.float32)

    w_proj = _wind_projection(u, v, velocity_enu.reshape(-1), grid_config.wind_projection_mode)
    c_eff = np.clip(c + w_proj, 50.0, None)

    n_cols = np.clip(reference_speed_mps / c_eff, 0.5, 2.0).astype(np.float32)
    n_grid = n_cols.reshape(grid_config.ny, grid_config.nx, grid_config.nz).transpose(1, 0, 2)

    grad_x, grad_y, grad_z = np.gradient(
        n_grid,
        x.astype(np.float64),
        y.astype(np.float64),
        z_rel.astype(np.float64),
        edge_order=1,
    )

    field = AcousticGridField(
        x_m=x.astype(np.float64),
        y_m=y.astype(np.float64),
        z_m=z_rel.astype(np.float64),
        n_grid=n_grid.astype(np.float64),
        grad_x=grad_x.astype(np.float64),
        grad_y=grad_y.astype(np.float64),
        grad_z=grad_z.astype(np.float64),
    )

    metadata = {
        "altitude_abs_m": alt_abs,
        "x_m": x,
        "y_m": y,
        "z_rel_m": z_rel,
        "n_grid": n_grid,
        "c_eff_mps": c_eff.reshape(grid_config.ny, grid_config.nx, grid_config.nz).transpose(1, 0, 2),
    }
    return field, metadata
