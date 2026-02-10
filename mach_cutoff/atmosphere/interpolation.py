"""Interpolation helpers for HRRR atmospheric snapshots."""

from __future__ import annotations

import numpy as np

from .hrrr import HRRRSnapshot


class HRRRInterpolator:
    """Nearest-horizontal, linear-vertical interpolation on HRRR snapshots."""

    def __init__(self, snapshot: HRRRSnapshot):
        self.snapshot = snapshot
        self.levels_hpa = np.asarray(snapshot.levels_hpa, dtype=np.float32)

        lat2d = np.asarray(snapshot.lat_deg, dtype=np.float32)
        lon2d = np.asarray(snapshot.lon_deg, dtype=np.float32)

        self.ny, self.nx = lat2d.shape
        self.nxy = self.nx * self.ny

        self.lat_flat = lat2d.reshape(-1)
        self.lon_flat = lon2d.reshape(-1)

        # Scaled lon to reduce distortion in nearest-neighbor search.
        cos_lat = np.cos(np.deg2rad(self.lat_flat))
        self._xy = np.column_stack([self.lat_flat, self.lon_flat * cos_lat])

        self._tree = None
        try:
            from scipy.spatial import cKDTree

            self._tree = cKDTree(self._xy)
        except Exception:
            self._tree = None

        L = self.levels_hpa.shape[0]
        self.temperature_flat = np.asarray(snapshot.temperature_k, dtype=np.float32).reshape(L, -1)
        self.rh_flat = np.asarray(snapshot.relative_humidity_pct, dtype=np.float32).reshape(L, -1)
        self.u_flat = np.asarray(snapshot.u_wind_mps, dtype=np.float32).reshape(L, -1)
        self.v_flat = np.asarray(snapshot.v_wind_mps, dtype=np.float32).reshape(L, -1)
        self.height_flat = np.asarray(snapshot.geopotential_height_m, dtype=np.float32).reshape(L, -1)

    def _nearest_indices(self, lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
        lat = np.asarray(lat_deg, dtype=np.float32).reshape(-1)
        lon = np.asarray(lon_deg, dtype=np.float32).reshape(-1)
        xy = np.column_stack([lat, lon * np.cos(np.deg2rad(lat))])

        if self._tree is not None:
            _, idx = self._tree.query(xy, k=1)
            return np.asarray(idx, dtype=np.int64).reshape(lat_deg.shape)

        # Fallback without scipy.
        d2 = np.sum((xy[:, None, :] - self._xy[None, :, :]) ** 2, axis=2)
        idx = np.argmin(d2, axis=1)
        return idx.reshape(lat_deg.shape)

    @staticmethod
    def _interp_profile(z_profile: np.ndarray, v_profile: np.ndarray, z_targets: np.ndarray) -> np.ndarray:
        order = np.argsort(z_profile)
        z_sorted = z_profile[order]
        v_sorted = v_profile[order]
        return np.interp(z_targets, z_sorted, v_sorted, left=v_sorted[0], right=v_sorted[-1])

    def sample_columns(
        self,
        lat_deg: np.ndarray,
        lon_deg: np.ndarray,
        altitudes_m: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """
        Sample atmospheric variables at N horizontal points and M altitude levels.

        Parameters
        ----------
        lat_deg, lon_deg : array-like shape (N,)
        altitudes_m : array-like shape (M,)

        Returns
        -------
        dict with arrays shape (N, M)
        """
        lat = np.asarray(lat_deg, dtype=np.float32).reshape(-1)
        lon = np.asarray(lon_deg, dtype=np.float32).reshape(-1)
        z_targets = np.asarray(altitudes_m, dtype=np.float32).reshape(-1)

        if lat.shape != lon.shape:
            raise ValueError("lat_deg and lon_deg must have matching shape")

        idx = self._nearest_indices(lat, lon).reshape(-1)

        n = lat.shape[0]
        m = z_targets.shape[0]

        out_t = np.empty((n, m), dtype=np.float32)
        out_rh = np.empty((n, m), dtype=np.float32)
        out_u = np.empty((n, m), dtype=np.float32)
        out_v = np.empty((n, m), dtype=np.float32)
        out_p = np.empty((n, m), dtype=np.float32)

        for i, j in enumerate(idx):
            z_prof = self.height_flat[:, j]

            out_t[i, :] = self._interp_profile(z_prof, self.temperature_flat[:, j], z_targets)
            out_rh[i, :] = self._interp_profile(z_prof, self.rh_flat[:, j], z_targets)
            out_u[i, :] = self._interp_profile(z_prof, self.u_flat[:, j], z_targets)
            out_v[i, :] = self._interp_profile(z_prof, self.v_flat[:, j], z_targets)
            out_p[i, :] = self._interp_profile(z_prof, self.levels_hpa, z_targets)

        return {
            "temperature_k": out_t,
            "relative_humidity_pct": out_rh,
            "u_wind_mps": out_u,
            "v_wind_mps": out_v,
            "pressure_hpa": out_p,
        }

    def sample_points(self, lat_deg: np.ndarray, lon_deg: np.ndarray, alt_m: np.ndarray) -> dict[str, np.ndarray]:
        lat = np.asarray(lat_deg, dtype=np.float32).reshape(-1)
        lon = np.asarray(lon_deg, dtype=np.float32).reshape(-1)
        alt = np.asarray(alt_m, dtype=np.float32).reshape(-1)
        if not (lat.shape == lon.shape == alt.shape):
            raise ValueError("lat, lon, alt must have matching shape")

        idx = self._nearest_indices(lat, lon).reshape(-1)

        out_t = np.empty_like(lat, dtype=np.float32)
        out_rh = np.empty_like(lat, dtype=np.float32)
        out_u = np.empty_like(lat, dtype=np.float32)
        out_v = np.empty_like(lat, dtype=np.float32)
        out_p = np.empty_like(lat, dtype=np.float32)

        for i, j in enumerate(idx):
            z_prof = self.height_flat[:, j]
            zt = np.array([alt[i]], dtype=np.float32)

            out_t[i] = self._interp_profile(z_prof, self.temperature_flat[:, j], zt)[0]
            out_rh[i] = self._interp_profile(z_prof, self.rh_flat[:, j], zt)[0]
            out_u[i] = self._interp_profile(z_prof, self.u_flat[:, j], zt)[0]
            out_v[i] = self._interp_profile(z_prof, self.v_flat[:, j], zt)[0]
            out_p[i] = self._interp_profile(z_prof, self.levels_hpa, zt)[0]

        return {
            "temperature_k": out_t,
            "relative_humidity_pct": out_rh,
            "u_wind_mps": out_u,
            "v_wind_mps": out_v,
            "pressure_hpa": out_p,
        }
