"""Prepare a higher-resolution CONUS population grid from official Census block-group data."""

from __future__ import annotations

import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np

from .config import GPWConfig

_KM_PER_DEG = 111.32
_SQMI_TO_SQKM = 2.58999
_API_TIMEOUT_S = 120.0
_API_RETRIES = 3
_API_USER_AGENT = "mach-cutoff-sim/0.2 census-blockgroup-prep"
_DECENNIAL_YEAR = 2020
_PREPARED_CELL_DEG_MAX = 0.02
_DECENNIAL_PL_URL = f"https://api.census.gov/data/{_DECENNIAL_YEAR}/dec/pl"
_GEOINFO_URL = f"https://api.census.gov/data/{_DECENNIAL_YEAR}/geoinfo"
_CONUS_STATE_FIPS: tuple[str, ...] = (
    "01",
    "04",
    "05",
    "06",
    "08",
    "09",
    "10",
    "11",
    "12",
    "13",
    "16",
    "17",
    "18",
    "19",
    "20",
    "21",
    "22",
    "23",
    "24",
    "25",
    "26",
    "27",
    "28",
    "29",
    "30",
    "31",
    "32",
    "33",
    "34",
    "35",
    "36",
    "37",
    "38",
    "39",
    "40",
    "41",
    "42",
    "44",
    "45",
    "46",
    "47",
    "48",
    "49",
    "50",
    "51",
    "53",
    "54",
    "55",
    "56",
)


def ensure_prepared_census_blockgroup_dataset(cfg: GPWConfig) -> Path:
    prepared_dir = Path(cfg.prepared_cache_dir).expanduser().resolve()
    prepared_dir.mkdir(parents=True, exist_ok=True)
    cell_deg = min(float(cfg.prepared_cell_deg), _PREPARED_CELL_DEG_MAX)
    prepared_name = f"census_blockgroup_{_DECENNIAL_YEAR}_conus_{_format_cell_deg(cell_deg)}.npz"
    prepared_path = prepared_dir / prepared_name
    if prepared_path.exists():
        return prepared_path

    lon_min, lat_min, lon_max, lat_max = [float(v) for v in cfg.conus_bounds]
    lat_edges = _grid_edges(lat_min, lat_max, cell_deg)
    lon_edges = _grid_edges(lon_min, lon_max, cell_deg)
    population_grid = np.zeros((lat_edges.size - 1, lon_edges.size - 1), dtype=np.float32)

    max_workers = min(8, max(2, len(_CONUS_STATE_FIPS) // 6))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_fetch_state_blockgroup_records, state_fips): state_fips for state_fips in _CONUS_STATE_FIPS
        }
        for future in as_completed(futures):
            state_fips = futures[future]
            lat_deg, lon_deg, population, area_sqmi = future.result()
            _accumulate_blockgroups_to_grid(
                population_grid,
                lat_edges,
                lon_edges,
                lat_deg=lat_deg,
                lon_deg=lon_deg,
                population=population,
                area_sqmi=area_sqmi,
            )
            print(
                "[bench] prepared Census block-group population "
                f"state={state_fips} records={int(population.size):,}"
            )

    metadata = {
        "product": "census_blockgroup_decennial_pl",
        "year": _DECENNIAL_YEAR,
        "prepared_cell_deg": cell_deg,
        "conus_bounds": [lon_min, lat_min, lon_max, lat_max],
        "source_population_endpoint": _DECENNIAL_PL_URL,
        "source_geoinfo_endpoint": _GEOINFO_URL,
        "state_fips": list(_CONUS_STATE_FIPS),
    }
    np.savez_compressed(
        prepared_path,
        lat_edges_deg=lat_edges.astype(np.float32),
        lon_edges_deg=lon_edges.astype(np.float32),
        population_grid=population_grid.astype(np.float32),
        source_metadata=np.asarray(json.dumps(metadata)),
    )
    return prepared_path


def _format_cell_deg(cell_deg: float) -> str:
    return str(float(cell_deg)).replace(".", "p") + "deg"


def _grid_edges(lo: float, hi: float, cell_deg: float) -> np.ndarray:
    lo = float(lo)
    hi = float(hi)
    cell_deg = float(max(cell_deg, 1e-4))
    count = max(1, int(math.ceil((hi - lo) / cell_deg)))
    return np.linspace(lo, hi, count + 1, dtype=np.float64)


def _fetch_state_blockgroup_records(state_fips: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pop_rows = _fetch_api_json(
        _DECENNIAL_PL_URL,
        params={
            "get": "P1_001N",
            "for": "block group:*",
            "in": [f"state:{state_fips}", "county:*", "tract:*"],
        },
    )
    geoinfo_rows = _fetch_api_json(
        _GEOINFO_URL,
        params={
            "get": "INTPTLAT,INTPTLON,AREALAND_SQMI",
            "for": "block group:*",
            "in": [f"state:{state_fips}", "county:*", "tract:*"],
        },
    )

    pop_by_key: dict[tuple[str, str, str, str], float] = {}
    for row in pop_rows[1:]:
        pop = _parse_float(row[0])
        if pop is None or pop <= 0.0:
            continue
        key = (row[1], row[2], row[3], row[4])
        pop_by_key[key] = pop

    lat_vals: list[float] = []
    lon_vals: list[float] = []
    pop_vals: list[float] = []
    area_vals: list[float] = []
    for row in geoinfo_rows[1:]:
        key = (row[3], row[4], row[5], row[6])
        pop = pop_by_key.get(key)
        if pop is None or pop <= 0.0:
            continue
        lat = _parse_float(row[0])
        lon = _parse_float(row[1])
        area_sqmi = _parse_float(row[2])
        if lat is None or lon is None or area_sqmi is None:
            continue
        lat_vals.append(lat)
        lon_vals.append(lon)
        pop_vals.append(pop)
        area_vals.append(max(area_sqmi, 1e-6))

    return (
        np.asarray(lat_vals, dtype=np.float64),
        np.asarray(lon_vals, dtype=np.float64),
        np.asarray(pop_vals, dtype=np.float64),
        np.asarray(area_vals, dtype=np.float64),
    )


def _fetch_api_json(url: str, *, params: dict[str, str | list[str]]) -> list[list[str]]:
    encoded_items: list[tuple[str, str]] = []
    for key, value in params.items():
        if isinstance(value, list):
            for item in value:
                encoded_items.append((key, item))
        else:
            encoded_items.append((key, value))
    query = urlencode(encoded_items)
    request = Request(f"{url}?{query}", headers={"User-Agent": _API_USER_AGENT})
    last_error: Exception | None = None
    for attempt in range(_API_RETRIES):
        try:
            with urlopen(request, timeout=_API_TIMEOUT_S) as response:
                payload = json.load(response)
            if not isinstance(payload, list) or not payload:
                raise RuntimeError(f"Unexpected Census API response shape from {url}")
            return payload
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt + 1 >= _API_RETRIES:
                break
            time.sleep(1.0 + attempt)
    raise RuntimeError(f"Failed to fetch Census API data from {url}: {last_error}") from last_error


def _parse_float(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(parsed):
        return None
    return parsed


def _accumulate_blockgroups_to_grid(
    population_grid: np.ndarray,
    lat_edges: np.ndarray,
    lon_edges: np.ndarray,
    *,
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    population: np.ndarray,
    area_sqmi: np.ndarray,
) -> None:
    if population.size == 0:
        return

    lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    lon_centers = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    cell_diag_km = math.sqrt(
        (float(np.mean(np.diff(lat_edges))) * _KM_PER_DEG) ** 2
        + (float(np.mean(np.diff(lon_edges))) * _KM_PER_DEG * math.cos(math.radians(float(np.mean(lat_edges))))) ** 2
    )

    ny, nx = population_grid.shape
    for lat_i, lon_i, pop_i, area_sqmi_i in zip(lat_deg, lon_deg, population, area_sqmi, strict=False):
        radius_km = max(0.25, math.sqrt(max(area_sqmi_i, 1e-6) * _SQMI_TO_SQKM / math.pi))
        support_radius_km = max(radius_km, 0.6 * cell_diag_km)
        lat_radius_deg = support_radius_km / _KM_PER_DEG
        lon_radius_deg = support_radius_km / max(1e-3, _KM_PER_DEG * math.cos(math.radians(float(lat_i))))

        lat_lo = max(0, int(np.searchsorted(lat_centers, lat_i - lat_radius_deg, side="left")))
        lat_hi = min(ny, int(np.searchsorted(lat_centers, lat_i + lat_radius_deg, side="right")))
        lon_lo = max(0, int(np.searchsorted(lon_centers, lon_i - lon_radius_deg, side="left")))
        lon_hi = min(nx, int(np.searchsorted(lon_centers, lon_i + lon_radius_deg, side="right")))

        if lat_lo >= lat_hi or lon_lo >= lon_hi:
            lat_idx = int(np.clip(np.searchsorted(lat_edges, lat_i, side="right") - 1, 0, ny - 1))
            lon_idx = int(np.clip(np.searchsorted(lon_edges, lon_i, side="right") - 1, 0, nx - 1))
            population_grid[lat_idx, lon_idx] += np.float32(pop_i)
            continue

        patch_lat = lat_centers[lat_lo:lat_hi]
        patch_lon = lon_centers[lon_lo:lon_hi]
        patch_lon_grid, patch_lat_grid = np.meshgrid(patch_lon, patch_lat)
        dist_km = _haversine_km(
            float(lat_i),
            float(lon_i),
            patch_lat_grid.reshape(-1),
            patch_lon_grid.reshape(-1),
        ).reshape(lat_hi - lat_lo, lon_hi - lon_lo)
        weights = np.maximum(0.0, 1.0 - np.square(dist_km / max(support_radius_km, 1e-6)))
        weight_sum = float(np.sum(weights))
        if weight_sum <= 0.0:
            lat_idx = int(np.clip(np.searchsorted(lat_edges, lat_i, side="right") - 1, 0, ny - 1))
            lon_idx = int(np.clip(np.searchsorted(lon_edges, lon_i, side="right") - 1, 0, nx - 1))
            population_grid[lat_idx, lon_idx] += np.float32(pop_i)
            continue
        population_grid[lat_lo:lat_hi, lon_lo:lon_hi] += np.float32(pop_i) * (weights / weight_sum).astype(np.float32)


def _haversine_km(lat0: float, lon0: float, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    lat0r = np.deg2rad(float(lat0))
    lon0r = np.deg2rad(float(lon0))
    latr = np.deg2rad(np.asarray(lat, dtype=np.float64))
    lonr = np.deg2rad(np.asarray(lon, dtype=np.float64))
    dlat = latr - lat0r
    dlon = lonr - lon0r
    a = np.sin(dlat * 0.5) ** 2 + np.cos(lat0r) * np.cos(latr) * np.sin(dlon * 0.5) ** 2
    return 6371.0 * (2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.clip(1.0 - a, 0.0, 1.0))))
