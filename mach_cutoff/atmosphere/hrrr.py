"""HRRR file resolution, download, and snapshot loading."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable
from urllib.request import urlretrieve

import numpy as np

from ..config import HRRRConfig
from ..core.geodesy import normalize_lon_deg


def _to_hour_floor(dt: datetime) -> datetime:
    dt = dt.astimezone(timezone.utc)
    return dt.replace(minute=0, second=0, microsecond=0)


def _to_hour_ceil(dt: datetime) -> datetime:
    floor = _to_hour_floor(dt)
    if dt == floor:
        return floor
    return floor + timedelta(hours=1)


@dataclass(frozen=True, slots=True)
class HRRRFileSpec:
    cycle_time_utc: datetime
    forecast_hour: int
    domain: str
    product: str

    @property
    def relative_key(self) -> str:
        date_str = self.cycle_time_utc.strftime("%Y%m%d")
        hour_str = self.cycle_time_utc.strftime("%H")
        return (
            f"hrrr.{date_str}/{self.domain}/"
            f"hrrr.t{hour_str}z.{self.product}{self.forecast_hour:02d}.grib2"
        )

    @property
    def valid_time_utc(self) -> datetime:
        return self.cycle_time_utc + timedelta(hours=self.forecast_hour)


@dataclass(slots=True)
class HRRRSnapshot:
    valid_time_utc: datetime
    levels_hpa: np.ndarray
    lat_deg: np.ndarray
    lon_deg: np.ndarray
    temperature_k: np.ndarray
    relative_humidity_pct: np.ndarray
    u_wind_mps: np.ndarray
    v_wind_mps: np.ndarray
    geopotential_height_m: np.ndarray


def specs_for_times(
    times_utc: Iterable[datetime],
    *,
    domain: str,
    product: str,
) -> list[HRRRFileSpec]:
    specs: dict[tuple[datetime, int, str, str], HRRRFileSpec] = {}
    for t in times_utc:
        floor = _to_hour_floor(t)
        ceil = _to_hour_ceil(t)
        for hour_dt in (floor, ceil):
            spec = HRRRFileSpec(
                cycle_time_utc=hour_dt,
                forecast_hour=0,
                domain=domain,
                product=product,
            )
            key = (spec.cycle_time_utc, spec.forecast_hour, spec.domain, spec.product)
            specs[key] = spec
    return sorted(specs.values(), key=lambda s: s.valid_time_utc)


def _select_var(ds, candidates: list[str]):
    for name in candidates:
        if name in ds:
            return ds[name]
    raise KeyError(f"Missing variable. Tried: {candidates}")


def _subset_by_bbox(ds, bbox):
    if bbox is None:
        return ds

    lat = None
    lon = None
    for name in ("latitude", "lat"):
        if name in ds:
            lat = ds[name]
            break
    for name in ("longitude", "lon"):
        if name in ds:
            lon = ds[name]
            break

    if lat is None or lon is None:
        return ds

    lat2 = np.asarray(lat.values)
    lon2 = normalize_lon_deg(np.asarray(lon.values))

    min_lat, max_lat, min_lon, max_lon = bbox
    mask = (
        (lat2 >= min_lat)
        & (lat2 <= max_lat)
        & (lon2 >= min_lon)
        & (lon2 <= max_lon)
    )

    if not np.any(mask):
        return ds

    y_idx, x_idx = np.where(mask)
    y0, y1 = int(y_idx.min()), int(y_idx.max()) + 1
    x0, x1 = int(x_idx.min()), int(x_idx.max()) + 1

    y_dim, x_dim = lat.dims
    return ds.isel({y_dim: slice(y0, y1), x_dim: slice(x0, x1)})


def load_snapshot_from_grib(path: Path, valid_time_utc: datetime, bbox=None) -> HRRRSnapshot:
    try:
        import xarray as xr
    except ImportError as exc:
        raise ImportError("xarray is required to load HRRR GRIB files") from exc

    backend_kwargs = {
        "filter_by_keys": {"typeOfLevel": "isobaricInhPa"},
        "indexpath": "",
    }

    ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs=backend_kwargs)
    ds = _subset_by_bbox(ds, bbox)
    ds.load()

    level_coord = None
    for name in ("isobaricInhPa", "isobaricInPa"):
        if name in ds.coords:
            level_coord = ds.coords[name]
            break
    if level_coord is None:
        raise KeyError("isobaric level coordinate not found")

    levels = np.asarray(level_coord.values, dtype=float)
    if level_coord.name == "isobaricInPa":
        levels = levels / 100.0

    t_var = _select_var(ds, ["t", "temperature", "Temperature"])
    rh_var = _select_var(ds, ["r", "relativeHumidity", "Relative humidity"])
    u_var = _select_var(ds, ["u", "u_component_of_wind", "u-component_of_wind"])
    v_var = _select_var(ds, ["v", "v_component_of_wind", "v-component_of_wind"])
    gh_var = _select_var(ds, ["gh", "ghl", "z", "geopotential_height"])

    lat_var = _select_var(ds, ["latitude", "lat"])
    lon_var = _select_var(ds, ["longitude", "lon"])

    level_dim = level_coord.dims[0]
    if lat_var.ndim != 2:
        raise ValueError("Expected 2D HRRR latitude grid")
    y_dim, x_dim = lat_var.dims

    t = t_var.transpose(level_dim, y_dim, x_dim).values.astype(np.float32)
    rh = rh_var.transpose(level_dim, y_dim, x_dim).values.astype(np.float32)
    u = u_var.transpose(level_dim, y_dim, x_dim).values.astype(np.float32)
    v = v_var.transpose(level_dim, y_dim, x_dim).values.astype(np.float32)
    gh = gh_var.transpose(level_dim, y_dim, x_dim).values.astype(np.float32)

    gh_units = str(getattr(gh_var, "units", "")).lower().strip()
    if "m2" in gh_units or "m^2" in gh_units:
        gh = gh / 9.80665

    lat = lat_var.values.astype(np.float32)
    lon = normalize_lon_deg(lon_var.values.astype(np.float32))

    ds.close()

    return HRRRSnapshot(
        valid_time_utc=valid_time_utc,
        levels_hpa=levels.astype(np.float32),
        lat_deg=lat,
        lon_deg=lon,
        temperature_k=t,
        relative_humidity_pct=rh,
        u_wind_mps=u,
        v_wind_mps=v,
        geopotential_height_m=gh,
    )


class HRRRDatasetManager:
    """Manages HRRR retrieval and cached snapshot loading."""

    def __init__(self, config: HRRRConfig, bbox: tuple[float, float, float, float] | None = None):
        self.config = config
        self.bbox = bbox
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._snapshot_cache: dict[Path, HRRRSnapshot] = {}

    def build_specs_for_times(self, times_utc: Iterable[datetime]) -> list[HRRRFileSpec]:
        return specs_for_times(
            times_utc,
            domain=self.config.domain,
            product=self.config.product,
        )

    def _local_path_for_spec(self, spec: HRRRFileSpec) -> Path:
        return self.cache_dir / spec.relative_key

    def _candidate_local_path(self, spec: HRRRFileSpec) -> Path:
        if self.config.local_grib_dir is None:
            return self._local_path_for_spec(spec)

        local_dir = Path(self.config.local_grib_dir)
        path = local_dir / spec.relative_key
        if path.exists():
            return path
        return self._local_path_for_spec(spec)

    def _download_one(self, spec: HRRRFileSpec) -> Path:
        target = self._local_path_for_spec(spec)
        if target.exists():
            return target

        alt_local = self._candidate_local_path(spec)
        if alt_local.exists():
            return alt_local

        target.parent.mkdir(parents=True, exist_ok=True)

        if self.config.use_s3fs:
            try:
                import s3fs

                fs = s3fs.S3FileSystem(anon=True)
                remote = f"s3://noaa-hrrr-pds/{spec.relative_key}"
                fs.get(remote, str(target))
                return target
            except Exception:
                pass

        if self.config.http_fallback:
            url = f"https://noaa-hrrr-bdp-pds.s3.amazonaws.com/{spec.relative_key}"
            urlretrieve(url, target)
            return target

        raise FileNotFoundError(
            "Unable to retrieve HRRR file. Configure local_grib_dir or enable use_s3fs/http_fallback"
        )

    def ensure_files(self, specs: Iterable[HRRRFileSpec]) -> list[Path]:
        unique_specs = {spec.relative_key: spec for spec in specs}
        ordered_specs = [unique_specs[key] for key in sorted(unique_specs.keys())]

        if self.config.download_workers <= 1:
            return [self._download_one(spec) for spec in ordered_specs]

        with ThreadPoolExecutor(max_workers=self.config.download_workers) as pool:
            futures = [pool.submit(self._download_one, spec) for spec in ordered_specs]
            return [f.result() for f in futures]

    def load_snapshots(self, specs: Iterable[HRRRFileSpec]) -> dict[datetime, HRRRSnapshot]:
        specs = list(specs)
        self.ensure_files(specs)

        snapshots: dict[datetime, HRRRSnapshot] = {}
        for spec in specs:
            path = self._candidate_local_path(spec)
            if not path.exists():
                path = self._local_path_for_spec(spec)
            if not path.exists():
                raise FileNotFoundError(f"HRRR file missing after download: {path}")

            if path in self._snapshot_cache:
                snap = self._snapshot_cache[path]
            else:
                snap = load_snapshot_from_grib(path, spec.valid_time_utc, bbox=self.bbox)
                self._snapshot_cache[path] = snap

            snapshots[spec.valid_time_utc] = snap

        return dict(sorted(snapshots.items(), key=lambda x: x[0]))

    def snapshots_for_times(self, times_utc: Iterable[datetime]) -> dict[datetime, HRRRSnapshot]:
        specs = self.build_specs_for_times(times_utc)
        return self.load_snapshots(specs)
