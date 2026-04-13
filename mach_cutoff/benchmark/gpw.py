"""GPWv4 acquisition and prepared-cache generation helpers."""

from __future__ import annotations

import json
import netrc
import os
import zipfile
from importlib import resources
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import (
    HTTPBasicAuthHandler,
    HTTPCookieProcessor,
    HTTPPasswordMgrWithDefaultRealm,
    Request,
    build_opener,
)

import numpy as np

from .census_blockgroups import ensure_prepared_census_blockgroup_dataset
from .config import GPWConfig

_DEFAULT_URL_TEMPLATE = (
    "https://sedac.ciesin.columbia.edu/downloads/data/gpw-v4/"
    "gpw-v4-population-count-rev11/gpw-v4-population-count_rev11_{year}_30_sec_tif.zip"
)


def ensure_prepared_population_dataset(cfg: GPWConfig) -> Path:
    source = _normalize_population_source(getattr(cfg, "source", "auto"))
    if source in {"census", "census_blockgroup", "census_blockgroup_2020"}:
        try:
            return ensure_prepared_census_blockgroup_dataset(cfg)
        except Exception as exc:  # noqa: BLE001
            fallback = _bundled_proxy_dataset_path()
            print(
                "[bench] warn: unable to prepare requested Census block-group population dataset "
                f"({exc}); falling back to bundled metro proxy CSV {fallback}"
            )
            return fallback

    prepared_dir = Path(cfg.prepared_cache_dir).expanduser().resolve()
    prepared_dir.mkdir(parents=True, exist_ok=True)
    prepared_name = f"gpwv4_r11_{int(cfg.year)}_conus_{_format_cell_deg(cfg.prepared_cell_deg)}.npz"
    prepared_path = prepared_dir / prepared_name
    if prepared_path.exists() and _prepared_dataset_matches_request(prepared_path, cfg):
        return prepared_path

    raw_path = _find_raw_dataset(Path(cfg.raw_cache_dir).expanduser().resolve(), year=int(cfg.year))
    try:
        if raw_path is None:
            if not cfg.auto_download:
                raise FileNotFoundError(
                    "Prepared GPW cache missing and auto_download=false. "
                    f"Expected prepared cache at {prepared_path}"
                )
            raw_path = _download_raw_dataset(cfg)

        geotiff_path = _ensure_geotiff(raw_path)
        _prepare_from_geotiff(geotiff_path, prepared_path, cfg)
        return prepared_path
    except Exception as exc:  # noqa: BLE001
        try:
            census_fallback = ensure_prepared_census_blockgroup_dataset(cfg)
            print(
                "[bench] warn: unable to prepare requested GPW population dataset "
                f"({exc}); falling back to Census {_format_cell_deg(min(float(cfg.prepared_cell_deg), 0.02))} "
                f"block-group grid {census_fallback}"
            )
            return census_fallback
        except Exception as census_exc:  # noqa: BLE001
            fallback = _bundled_proxy_dataset_path()
            print(
                "[bench] warn: unable to prepare requested GPW population dataset "
                f"({exc}); Census block-group fallback failed ({census_exc}); "
                f"falling back to bundled metro proxy CSV {fallback}"
            )
            return fallback


def _normalize_population_source(value: object) -> str:
    return str(value or "auto").strip().lower()


def _format_cell_deg(cell_deg: float) -> str:
    return str(float(cell_deg)).replace(".", "p") + "deg"


def _bundled_proxy_dataset_path() -> Path:
    resource = resources.files("mach_cutoff").joinpath("data").joinpath("us_metro_population_sample.csv")
    return Path(str(resource)).resolve()


def _prepared_dataset_matches_request(path: Path, cfg: GPWConfig) -> bool:
    metadata = _read_prepared_metadata(path)
    if not metadata:
        return True
    product = str(metadata.get("product", "")).strip().lower()
    if product.startswith("metro_proxy"):
        return False
    year = metadata.get("year")
    if year is not None and int(year) != int(cfg.year):
        return False
    cell_deg = metadata.get("prepared_cell_deg")
    if cell_deg is not None and abs(float(cell_deg) - float(cfg.prepared_cell_deg)) > 1e-6:
        return False
    return True


def _read_prepared_metadata(path: Path) -> dict | None:
    try:
        with np.load(path) as raw:
            if "source_metadata" not in raw:
                return None
            payload = raw["source_metadata"]
            if payload.size == 0:
                return None
            return json.loads(str(payload.reshape(-1)[0]))
    except Exception:  # noqa: BLE001
        return None


def _find_raw_dataset(raw_dir: Path, year: int) -> Path | None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    preferred = sorted(raw_dir.glob(f"*{year}*30_sec*.tif"))
    if preferred:
        return preferred[0]
    preferred_zip = sorted(raw_dir.glob(f"*{year}*30_sec*.zip"))
    if preferred_zip:
        return preferred_zip[0]
    generic_tif = sorted(raw_dir.glob("*.tif"))
    if generic_tif:
        return generic_tif[0]
    generic_zip = sorted(raw_dir.glob("*.zip"))
    if generic_zip:
        return generic_zip[0]
    return None


def _resolve_credentials(download_url: str) -> tuple[str, str] | None:
    env_user = os.getenv("EARTHDATA_USERNAME")
    env_pass = os.getenv("EARTHDATA_PASSWORD")
    if env_user and env_pass:
        return env_user, env_pass

    host = urlparse(download_url).hostname or ""
    try:
        auth = netrc.netrc()
    except OSError:
        auth = None
    if auth is None:
        return None

    for machine in [host, "urs.earthdata.nasa.gov", "earthdata.nasa.gov"]:
        if not machine:
            continue
        creds = auth.authenticators(machine)
        if creds is None:
            continue
        login, _account, password = creds
        if login and password:
            return login, password
    return None


def _download_raw_dataset(cfg: GPWConfig) -> Path:
    raw_dir = Path(cfg.raw_cache_dir).expanduser().resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)

    download_url = cfg.download_url or _DEFAULT_URL_TEMPLATE.format(year=int(cfg.year))
    local_zip = raw_dir / Path(urlparse(download_url).path).name
    if local_zip.exists():
        return local_zip

    creds = _resolve_credentials(download_url)
    if creds is None:
        raise RuntimeError(
            "GPW download requires Earthdata credentials. Set EARTHDATA_USERNAME/EARTHDATA_PASSWORD "
            "or configure ~/.netrc for urs.earthdata.nasa.gov."
        )

    user, password = creds
    mgr = HTTPPasswordMgrWithDefaultRealm()
    mgr.add_password(None, "https://urs.earthdata.nasa.gov", user, password)
    opener = build_opener(HTTPBasicAuthHandler(mgr), HTTPCookieProcessor())

    req = Request(download_url, headers={"User-Agent": "mach-cutoff-sim/0.2 benchmark"})
    try:
        with opener.open(req, timeout=180) as resp:
            with local_zip.open("wb") as f:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to download GPW dataset from {download_url}: {exc}") from exc

    return local_zip


def _ensure_geotiff(raw_path: Path) -> Path:
    suffix = raw_path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        return raw_path
    if suffix != ".zip":
        raise ValueError(f"Unsupported GPW raw artifact: {raw_path}")

    with zipfile.ZipFile(raw_path, "r") as zf:
        members = [name for name in zf.namelist() if name.lower().endswith((".tif", ".tiff"))]
        if not members:
            raise ValueError(f"ZIP archive does not include a GeoTIFF: {raw_path}")
        member = members[0]
        out_path = raw_path.with_name(Path(member).name)
        if out_path.exists():
            return out_path
        with zf.open(member, "r") as src, out_path.open("wb") as dst:
            while True:
                chunk = src.read(1024 * 1024)
                if not chunk:
                    break
                dst.write(chunk)
        return out_path


def _prepare_from_geotiff(geotiff_path: Path, prepared_path: Path, cfg: GPWConfig):
    try:
        from PIL import Image, TiffImagePlugin
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Pillow is required to preprocess GPW GeoTIFF into prepared NPZ cache. "
            "Install optional visualization dependencies or provide a prepared dataset path directly."
        ) from exc

    with Image.open(geotiff_path) as img:
        arr = np.asarray(img, dtype=np.float64)
        if arr.ndim == 3:
            arr = arr[..., 0]
        if arr.ndim != 2:
            raise ValueError(f"Unsupported GPW GeoTIFF shape: {arr.shape}")

        tags = img.tag_v2 if isinstance(img.tag_v2, TiffImagePlugin.ImageFileDirectory_v2) else img.tag_v2
        scale = tags.get(33550)
        tie = tags.get(33922)
        if scale is None or tie is None:
            raise ValueError("GeoTIFF missing ModelPixelScaleTag(33550) or ModelTiepointTag(33922)")

        sx = float(scale[0])
        sy = float(scale[1])
        lon0 = float(tie[3])
        lat0 = float(tie[4])

        height, width = arr.shape
        lon_edges = lon0 + np.arange(width + 1, dtype=np.float64) * sx
        lat_edges_desc = lat0 - np.arange(height + 1, dtype=np.float64) * abs(sy)
        if lat_edges_desc[1] < lat_edges_desc[0]:
            lat_edges = lat_edges_desc[::-1]
            arr = arr[::-1, :]
        else:
            lat_edges = lat_edges_desc

    lon_min, lat_min, lon_max, lat_max = [float(v) for v in cfg.conus_bounds]
    clipped_lat_edges, clipped_lon_edges, clipped_grid = _clip_grid(
        lat_edges,
        lon_edges,
        arr,
        lat_min,
        lat_max,
        lon_min,
        lon_max,
    )

    target_lat_edges, target_lon_edges, target_grid = _regrid_to_cell_deg(
        clipped_lat_edges,
        clipped_lon_edges,
        clipped_grid,
        lat_min,
        lat_max,
        lon_min,
        lon_max,
        float(cfg.prepared_cell_deg),
    )

    metadata = {
        "product": cfg.product,
        "year": int(cfg.year),
        "source_geotiff": str(geotiff_path),
        "prepared_cell_deg": float(cfg.prepared_cell_deg),
        "conus_bounds": [lon_min, lat_min, lon_max, lat_max],
        "shape": [int(v) for v in target_grid.shape],
    }

    prepared_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        prepared_path,
        lat_edges_deg=target_lat_edges.astype(np.float32),
        lon_edges_deg=target_lon_edges.astype(np.float32),
        population_grid=target_grid.astype(np.float32),
        source_metadata=np.asarray([json.dumps(metadata)], dtype="<U4096"),
    )


def _clip_grid(
    lat_edges: np.ndarray,
    lon_edges: np.ndarray,
    grid: np.ndarray,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat_lo = int(np.searchsorted(lat_edges, lat_min, side="right") - 1)
    lat_hi = int(np.searchsorted(lat_edges, lat_max, side="left"))
    lon_lo = int(np.searchsorted(lon_edges, lon_min, side="right") - 1)
    lon_hi = int(np.searchsorted(lon_edges, lon_max, side="left"))

    lat_lo = max(0, min(lat_lo, lat_edges.size - 2))
    lat_hi = max(lat_lo + 1, min(lat_hi, lat_edges.size - 1))
    lon_lo = max(0, min(lon_lo, lon_edges.size - 2))
    lon_hi = max(lon_lo + 1, min(lon_hi, lon_edges.size - 1))

    lat_edges_out = lat_edges[lat_lo:lat_hi + 1]
    lon_edges_out = lon_edges[lon_lo:lon_hi + 1]
    grid_out = np.asarray(grid[lat_lo:lat_hi, lon_lo:lon_hi], dtype=np.float64)
    grid_out = np.where(np.isfinite(grid_out) & (grid_out > 0.0), grid_out, 0.0)
    return lat_edges_out, lon_edges_out, grid_out


def _regrid_to_cell_deg(
    lat_edges: np.ndarray,
    lon_edges: np.ndarray,
    grid: np.ndarray,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    cell_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cell_deg = max(float(cell_deg), 1e-4)

    ny = max(1, int(np.ceil((lat_max - lat_min) / cell_deg)))
    nx = max(1, int(np.ceil((lon_max - lon_min) / cell_deg)))
    target_lat_edges = np.linspace(lat_min, lat_max, ny + 1, dtype=np.float64)
    target_lon_edges = np.linspace(lon_min, lon_max, nx + 1, dtype=np.float64)

    src_lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    src_lon_centers = 0.5 * (lon_edges[:-1] + lon_edges[1:])

    lat_idx = np.searchsorted(target_lat_edges, src_lat_centers, side="right") - 1
    lon_idx = np.searchsorted(target_lon_edges, src_lon_centers, side="right") - 1
    valid_lat = (lat_idx >= 0) & (lat_idx < ny)
    valid_lon = (lon_idx >= 0) & (lon_idx < nx)

    out = np.zeros((ny, nx), dtype=np.float64)
    if not np.any(valid_lat) or not np.any(valid_lon):
        return target_lat_edges, target_lon_edges, out

    lon_idx_valid = lon_idx[valid_lon]
    for src_row_idx, lat_ok in enumerate(valid_lat):
        if not lat_ok:
            continue
        li = int(lat_idx[src_row_idx])
        row = np.asarray(grid[src_row_idx], dtype=np.float64)
        row = row[valid_lon]
        row_valid = np.isfinite(row) & (row > 0.0)
        if not np.any(row_valid):
            continue
        np.add.at(out[li], lon_idx_valid[row_valid], row[row_valid])

    return target_lat_edges, target_lon_edges, out
