"""Population dataset loading and boom-footprint impact analysis."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import TextIO

import numpy as np

from ..config import PopulationConfig
from ..core.geodesy import normalize_lon_deg
from .outputs import EmissionResult, PopulationImpactResult

_KM_PER_DEG = 111.32
_DEFAULT_DATASET_NAME = "us_metro_population_sample.csv"


@dataclass(slots=True)
class _PopulationPoints:
    source_name: str
    source_path: str
    lat_deg: np.ndarray
    lon_deg: np.ndarray
    population: np.ndarray


@dataclass(slots=True)
class _PopulationGrid:
    source_name: str
    source_path: str
    lat_edges_deg: np.ndarray
    lon_edges_deg: np.ndarray
    population_grid: np.ndarray


@dataclass(slots=True)
class PopulationCorridorSampler:
    source_name: str
    lat_edges_deg: np.ndarray | None
    lon_edges_deg: np.ndarray | None
    density_grid_people_per_km2: np.ndarray | None
    point_lat_deg: np.ndarray | None
    point_lon_deg: np.ndarray | None
    point_population: np.ndarray | None

    def local_density_people_per_km2(
        self,
        lat_deg: float,
        lon_deg: float,
        *,
        half_width_km: float,
    ) -> float:
        half_width_km = max(float(half_width_km), 1e-3)
        lon_deg = float(normalize_lon_deg(np.asarray([lon_deg], dtype=np.float64)).reshape(-1)[0])
        lat_deg = float(lat_deg)

        if self.density_grid_people_per_km2 is not None and self.lat_edges_deg is not None and self.lon_edges_deg is not None:
            lat_edges = np.asarray(self.lat_edges_deg, dtype=np.float64)
            lon_edges = np.asarray(self.lon_edges_deg, dtype=np.float64)
            density = np.asarray(self.density_grid_people_per_km2, dtype=np.float64)
            lat_radius_deg = half_width_km / _KM_PER_DEG
            lon_radius_deg = half_width_km / max(1e-3, _KM_PER_DEG * np.cos(np.deg2rad(lat_deg)))
            lat_mask = (lat_edges[:-1] < lat_deg + lat_radius_deg) & (lat_edges[1:] > lat_deg - lat_radius_deg)
            lon_mask = (lon_edges[:-1] < lon_deg + lon_radius_deg) & (lon_edges[1:] > lon_deg - lon_radius_deg)
            if np.any(lat_mask) and np.any(lon_mask):
                window = density[np.ix_(lat_mask, lon_mask)]
                if window.size:
                    return float(np.mean(window))
            return 0.0

        if self.point_lat_deg is not None and self.point_lon_deg is not None and self.point_population is not None:
            lat = np.asarray(self.point_lat_deg, dtype=np.float64)
            lon = np.asarray(self.point_lon_deg, dtype=np.float64)
            pop = np.asarray(self.point_population, dtype=np.float64)
            lat_radius_deg = half_width_km / _KM_PER_DEG
            lon_radius_deg = half_width_km / max(1e-3, _KM_PER_DEG * np.cos(np.deg2rad(lat_deg)))
            mask = (np.abs(lat - lat_deg) <= lat_radius_deg) & (np.abs(lon - lon_deg) <= lon_radius_deg)
            if not np.any(mask):
                return 0.0
            area_km2 = max((2.0 * half_width_km) ** 2, 1.0)
            return float(np.sum(pop[mask]) / area_km2)

        return 0.0


def _read_population_csv(stream: TextIO, source_name: str, source_path: str) -> _PopulationPoints:
    reader = csv.DictReader(stream)
    if not reader.fieldnames:
        raise ValueError("Population CSV is missing a header row")

    field_lookup = {name.strip().lower(): name for name in reader.fieldnames}

    def _pick(*aliases: str) -> str | None:
        for alias in aliases:
            key = alias.strip().lower()
            if key in field_lookup:
                return field_lookup[key]
        return None

    lat_key = _pick("lat", "lat_deg", "latitude", "latitude_deg")
    lon_key = _pick("lon", "lon_deg", "longitude", "longitude_deg")
    pop_key = _pick("population", "pop", "people", "value")
    if lat_key is None or lon_key is None or pop_key is None:
        raise ValueError(
            "Population CSV must include latitude, longitude, and population columns "
            "(e.g. lat_deg, lon_deg, population)"
        )

    lat_vals: list[float] = []
    lon_vals: list[float] = []
    pop_vals: list[float] = []
    for row in reader:
        try:
            lat = float(row[lat_key])
            lon = float(row[lon_key])
            pop = float(row[pop_key])
        except (TypeError, ValueError, KeyError):
            continue
        if not (np.isfinite(lat) and np.isfinite(lon) and np.isfinite(pop)):
            continue
        if pop <= 0.0:
            continue
        lat_vals.append(lat)
        lon_vals.append(lon)
        pop_vals.append(pop)

    lat_arr = np.asarray(lat_vals, dtype=np.float64)
    lon_arr = normalize_lon_deg(np.asarray(lon_vals, dtype=np.float64))
    pop_arr = np.asarray(pop_vals, dtype=np.float64)
    return _PopulationPoints(
        source_name=source_name,
        source_path=source_path,
        lat_deg=lat_arr,
        lon_deg=lon_arr,
        population=pop_arr,
    )


def _npz_pick(raw: np.lib.npyio.NpzFile, *keys: str):
    for key in keys:
        if key in raw:
            return raw[key]
    return None


def _load_population_npz(path: Path) -> _PopulationPoints | _PopulationGrid:
    with np.load(path) as raw:
        lat_edges = _npz_pick(raw, "lat_edges_deg", "heatmap_lat_edges_deg", "lat_edges")
        lon_edges = _npz_pick(raw, "lon_edges_deg", "heatmap_lon_edges_deg", "lon_edges")
        grid = _npz_pick(raw, "population_grid", "heatmap_population", "population_heatmap")

        if lat_edges is not None and lon_edges is not None and grid is not None:
            lat_edges_arr = np.asarray(lat_edges, dtype=np.float64).reshape(-1)
            lon_edges_arr = normalize_lon_deg(np.asarray(lon_edges, dtype=np.float64).reshape(-1))
            pop_grid = np.asarray(grid, dtype=np.float64)
            if pop_grid.ndim != 2:
                raise ValueError("Population NPZ grid payload must be 2D")
            if pop_grid.shape != (max(0, lat_edges_arr.size - 1), max(0, lon_edges_arr.size - 1)):
                raise ValueError("Population NPZ grid shape does not match lat/lon edges")
            pop_grid = np.where(np.isfinite(pop_grid) & (pop_grid > 0.0), pop_grid, 0.0)
            return _PopulationGrid(
                source_name=path.name,
                source_path=str(path),
                lat_edges_deg=lat_edges_arr,
                lon_edges_deg=lon_edges_arr,
                population_grid=pop_grid,
            )

        lat = _npz_pick(raw, "lat_deg", "lat")
        lon = _npz_pick(raw, "lon_deg", "lon")
        pop = _npz_pick(raw, "population", "pop")
        if lat is None or lon is None or pop is None:
            raise ValueError(
                "Population NPZ must contain point arrays (lat/lon/pop) or gridded arrays "
                "(lat_edges/lon_edges/population_grid)"
            )
        lat_arr = np.asarray(lat, dtype=np.float64).reshape(-1)
        lon_arr = normalize_lon_deg(np.asarray(lon, dtype=np.float64).reshape(-1))
        pop_arr = np.asarray(pop, dtype=np.float64).reshape(-1)
        n = int(min(lat_arr.size, lon_arr.size, pop_arr.size))
        if n <= 0:
            raise ValueError("Population NPZ does not contain valid lat/lon/pop data")
        lat_arr = lat_arr[:n]
        lon_arr = lon_arr[:n]
        pop_arr = pop_arr[:n]
        finite = np.isfinite(lat_arr) & np.isfinite(lon_arr) & np.isfinite(pop_arr) & (pop_arr > 0.0)
        return _PopulationPoints(
            source_name=path.name,
            source_path=str(path),
            lat_deg=lat_arr[finite],
            lon_deg=lon_arr[finite],
            population=pop_arr[finite],
        )


def _load_population_source(dataset_path: str | None) -> _PopulationPoints | _PopulationGrid:
    if dataset_path is None:
        resource = resources.files("mach_cutoff").joinpath("data").joinpath(_DEFAULT_DATASET_NAME)
        with resource.open("r", encoding="utf-8", newline="") as f:
            return _read_population_csv(
                f,
                source_name=f"bundled:{_DEFAULT_DATASET_NAME}",
                source_path=f"bundled:{_DEFAULT_DATASET_NAME}",
            )

    path = Path(dataset_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Population dataset not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        with path.open("r", encoding="utf-8", newline="") as f:
            return _read_population_csv(f, source_name=path.name, source_path=str(path))

    if suffix == ".npz":
        return _load_population_npz(path)

    raise ValueError("Population dataset must be .csv, .txt, or .npz")


def build_population_corridor_sampler(dataset_path: str | None) -> PopulationCorridorSampler | None:
    try:
        source = _load_population_source(dataset_path)
    except Exception:
        return None

    if isinstance(source, _PopulationGrid):
        lat_edges = np.asarray(source.lat_edges_deg, dtype=np.float64)
        lon_edges = np.asarray(source.lon_edges_deg, dtype=np.float64)
        lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])
        dlat_km = np.abs(np.diff(lat_edges)) * _KM_PER_DEG
        dlon_km = np.abs(np.diff(lon_edges))[None, :] * (
            _KM_PER_DEG * np.cos(np.deg2rad(lat_centers))
        )[:, None]
        cell_area_km2 = np.maximum(dlat_km[:, None] * dlon_km, 1e-6)
        density = np.asarray(source.population_grid, dtype=np.float64) / cell_area_km2
        return PopulationCorridorSampler(
            source_name=source.source_name,
            lat_edges_deg=lat_edges,
            lon_edges_deg=lon_edges,
            density_grid_people_per_km2=density,
            point_lat_deg=None,
            point_lon_deg=None,
            point_population=None,
        )

    return PopulationCorridorSampler(
        source_name=source.source_name,
        lat_edges_deg=None,
        lon_edges_deg=None,
        density_grid_people_per_km2=None,
        point_lat_deg=np.asarray(source.lat_deg, dtype=np.float64),
        point_lon_deg=np.asarray(source.lon_deg, dtype=np.float64),
        point_population=np.asarray(source.population, dtype=np.float64),
    )


def _analysis_domain_bounds(emissions: list[EmissionResult], pad_deg: float) -> tuple[float, float, float, float]:
    lats: list[float] = []
    lons: list[float] = []
    for emission in emissions:
        lats.append(float(emission.aircraft_lat_deg))
        lons.append(float(emission.aircraft_lon_deg))
        for ray in emission.rays:
            if ray.ground_hit and ray.ground_hit_lat_lon is not None:
                lat, lon = ray.ground_hit_lat_lon
                lats.append(float(lat))
                lons.append(float(lon))

    if not lats or not lons:
        return -90.0, 90.0, -180.0, 180.0

    lat_arr = np.asarray(lats, dtype=np.float64)
    lon_arr = normalize_lon_deg(np.asarray(lons, dtype=np.float64))
    lat_min = float(max(-89.9, np.min(lat_arr) - pad_deg))
    lat_max = float(min(89.9, np.max(lat_arr) + pad_deg))
    lon_min = float(max(-180.0, np.min(lon_arr) - pad_deg))
    lon_max = float(min(180.0, np.max(lon_arr) + pad_deg))

    if lat_max <= lat_min:
        center = float(lat_arr[0])
        lat_min = center - 0.25
        lat_max = center + 0.25
    if lon_max <= lon_min:
        center = float(lon_arr[0])
        lon_min = center - 0.25
        lon_max = center + 0.25

    return lat_min, lat_max, lon_min, lon_max


def _adaptive_grid(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    *,
    cell_deg: float,
    max_cells: int,
) -> tuple[np.ndarray, np.ndarray]:
    step = float(max(cell_deg, 1e-3))
    span_lat = float(max(1e-6, lat_max - lat_min))
    span_lon = float(max(1e-6, lon_max - lon_min))
    max_cells = int(max(max_cells, 10_000))

    while True:
        ny = int(np.ceil(span_lat / step))
        nx = int(np.ceil(span_lon / step))
        ny = max(1, ny)
        nx = max(1, nx)
        if ny * nx <= max_cells:
            break
        scale = np.sqrt((ny * nx) / max_cells)
        step *= max(1.2, scale)

    lat_edges = np.linspace(lat_min, lat_max, ny + 1, dtype=np.float64)
    lon_edges = np.linspace(lon_min, lon_max, nx + 1, dtype=np.float64)
    return lat_edges, lon_edges


def _aggregate_population_to_grid(
    points: _PopulationPoints,
    lat_edges: np.ndarray,
    lon_edges: np.ndarray,
) -> np.ndarray:
    ny = int(max(0, lat_edges.size - 1))
    nx = int(max(0, lon_edges.size - 1))
    grid = np.zeros((ny, nx), dtype=np.float64)
    if ny == 0 or nx == 0 or points.population.size == 0:
        return grid

    lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    lon_centers = 0.5 * (lon_edges[:-1] + lon_edges[1:])

    for lat_deg, lon_deg, population in zip(points.lat_deg, points.lon_deg, points.population, strict=False):
        sigma_km = _point_population_spread_sigma_km(float(population))
        support_radius_km = max(12.0, 2.8 * sigma_km)
        lat_radius_deg = support_radius_km / _KM_PER_DEG
        lon_radius_deg = support_radius_km / max(1e-3, _KM_PER_DEG * np.cos(np.deg2rad(lat_deg)))

        lat_lo = max(0, int(np.searchsorted(lat_centers, lat_deg - lat_radius_deg, side="left")))
        lat_hi = min(ny, int(np.searchsorted(lat_centers, lat_deg + lat_radius_deg, side="right")))
        lon_lo = max(0, int(np.searchsorted(lon_centers, lon_deg - lon_radius_deg, side="left")))
        lon_hi = min(nx, int(np.searchsorted(lon_centers, lon_deg + lon_radius_deg, side="right")))

        if lat_lo >= lat_hi or lon_lo >= lon_hi:
            lat_idx = int(np.clip(np.searchsorted(lat_edges, lat_deg, side="right") - 1, 0, ny - 1))
            lon_idx = int(np.clip(np.searchsorted(lon_edges, lon_deg, side="right") - 1, 0, nx - 1))
            grid[lat_idx, lon_idx] += float(population)
            continue

        patch_lat = lat_centers[lat_lo:lat_hi]
        patch_lon = lon_centers[lon_lo:lon_hi]
        patch_lon_grid, patch_lat_grid = np.meshgrid(patch_lon, patch_lat)
        dist_km = _haversine_km(
            float(lat_deg),
            float(lon_deg),
            patch_lat_grid.reshape(-1),
            patch_lon_grid.reshape(-1),
        ).reshape(lat_hi - lat_lo, lon_hi - lon_lo)
        weights = np.exp(-0.5 * np.square(dist_km / max(sigma_km, 1e-3)))
        weights = np.where(dist_km <= support_radius_km, weights, 0.0)
        weight_sum = float(np.sum(weights))
        if weight_sum <= 0.0:
            lat_idx = int(np.clip(np.searchsorted(lat_edges, lat_deg, side="right") - 1, 0, ny - 1))
            lon_idx = int(np.clip(np.searchsorted(lon_edges, lon_deg, side="right") - 1, 0, nx - 1))
            grid[lat_idx, lon_idx] += float(population)
            continue

        grid[lat_lo:lat_hi, lon_lo:lon_hi] += float(population) * (weights / weight_sum)
    return grid


def _point_population_spread_sigma_km(population: float) -> float:
    pop = max(float(population), 1.0)
    sigma_km = 4.0 + 2.6 * float(np.log10(pop))
    return float(np.clip(sigma_km, 5.0, 24.0))


def _clip_population_grid_to_domain(
    source: _PopulationGrid,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat_edges = np.asarray(source.lat_edges_deg, dtype=np.float64).reshape(-1)
    lon_edges = normalize_lon_deg(np.asarray(source.lon_edges_deg, dtype=np.float64).reshape(-1))
    grid = np.asarray(source.population_grid, dtype=np.float64)
    if (
        lat_edges.size < 2
        or lon_edges.size < 2
        or grid.ndim != 2
        or grid.shape != (lat_edges.size - 1, lon_edges.size - 1)
    ):
        raise ValueError("Population grid is malformed (lat/lon edges and grid shape mismatch)")

    lat_lo_idx = int(np.searchsorted(lat_edges, lat_min, side="right") - 1)
    lat_hi_idx = int(np.searchsorted(lat_edges, lat_max, side="left"))
    lon_lo_idx = int(np.searchsorted(lon_edges, lon_min, side="right") - 1)
    lon_hi_idx = int(np.searchsorted(lon_edges, lon_max, side="left"))

    lat_lo_idx = max(0, min(lat_lo_idx, lat_edges.size - 2))
    lat_hi_idx = max(lat_lo_idx + 1, min(lat_hi_idx, lat_edges.size - 1))
    lon_lo_idx = max(0, min(lon_lo_idx, lon_edges.size - 2))
    lon_hi_idx = max(lon_lo_idx + 1, min(lon_hi_idx, lon_edges.size - 1))

    clipped_lat_edges = lat_edges[lat_lo_idx:lat_hi_idx + 1]
    clipped_lon_edges = lon_edges[lon_lo_idx:lon_hi_idx + 1]
    clipped_grid = grid[lat_lo_idx:lat_hi_idx, lon_lo_idx:lon_hi_idx]
    clipped_grid = np.where(np.isfinite(clipped_grid) & (clipped_grid > 0.0), clipped_grid, 0.0)
    return clipped_lat_edges, clipped_lon_edges, clipped_grid


def _haversine_km(lat0: float, lon0: float, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    lat0r = np.deg2rad(float(lat0))
    lon0r = np.deg2rad(float(lon0))
    latr = np.deg2rad(np.asarray(lat, dtype=np.float64))
    lonr = np.deg2rad(np.asarray(lon, dtype=np.float64))
    dlat = latr - lat0r
    dlon = lonr - lon0r
    a = np.sin(dlat * 0.5) ** 2 + np.cos(lat0r) * np.cos(latr) * np.sin(dlon * 0.5) ** 2
    return 6371.0 * (2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.clip(1.0 - a, 0.0, 1.0))))


def _project_equirect_km(lat_deg: np.ndarray, lon_deg: np.ndarray, lat0_deg: float, lon0_deg: float) -> np.ndarray:
    lat = np.asarray(lat_deg, dtype=np.float64)
    lon = normalize_lon_deg(np.asarray(lon_deg, dtype=np.float64))
    lon0 = float(normalize_lon_deg(np.asarray(lon0_deg, dtype=np.float64)).item())
    dlon = normalize_lon_deg(lon - lon0)
    x = dlon * (_KM_PER_DEG * np.cos(np.deg2rad(lat0_deg)))
    y = (lat - lat0_deg) * _KM_PER_DEG
    return np.column_stack([x, y])


def _point_to_segment_distance_km(points_xy: np.ndarray, a_xy: np.ndarray, b_xy: np.ndarray) -> np.ndarray:
    p = np.asarray(points_xy, dtype=np.float64)
    a = np.asarray(a_xy, dtype=np.float64)
    b = np.asarray(b_xy, dtype=np.float64)
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-12:
        return np.linalg.norm(p - a, axis=1)
    t = np.sum((p - a) * ab, axis=1) / denom
    t = np.clip(t, 0.0, 1.0)
    closest = a + t[:, None] * ab[None, :]
    return np.linalg.norm(p - closest, axis=1)


def _convex_hull_xy(points_xy: np.ndarray) -> np.ndarray:
    pts = np.unique(np.asarray(points_xy, dtype=np.float64), axis=0)
    if pts.shape[0] <= 2:
        return pts

    order = np.lexsort((pts[:, 1], pts[:, 0]))
    pts = pts[order]

    def _cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

    lower: list[np.ndarray] = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0.0:
            lower.pop()
        lower.append(p)

    upper: list[np.ndarray] = []
    for p in pts[::-1]:
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0.0:
            upper.pop()
        upper.append(p)

    hull = np.array(lower[:-1] + upper[:-1], dtype=np.float64)
    return hull


def _points_in_polygon_xy(points_xy: np.ndarray, polygon_xy: np.ndarray) -> np.ndarray:
    if polygon_xy.shape[0] < 3 or points_xy.shape[0] == 0:
        return np.zeros(points_xy.shape[0], dtype=bool)

    px = points_xy[:, 0]
    py = points_xy[:, 1]
    vx = polygon_xy[:, 0]
    vy = polygon_xy[:, 1]

    inside = np.zeros(points_xy.shape[0], dtype=bool)
    j = polygon_xy.shape[0] - 1
    for i in range(polygon_xy.shape[0]):
        yi = vy[i]
        yj = vy[j]
        xi = vx[i]
        xj = vx[j]
        denom = (yj - yi) + 1e-12
        intersects = ((yi > py) != (yj > py)) & (px < ((xj - xi) * (py - yi) / denom + xi))
        inside ^= intersects
        j = i
    return inside


def _emission_coverage_mask(
    hit_lat: np.ndarray,
    hit_lon: np.ndarray,
    hit_ray_id: np.ndarray,
    cell_lat_flat: np.ndarray,
    cell_lon_flat: np.ndarray,
    cfg: PopulationConfig,
) -> np.ndarray:
    if hit_lat.size == 0:
        return np.zeros(cell_lat_flat.size, dtype=bool)

    width_km = float(max(cfg.trace_half_width_km, 0.5))
    mask = np.zeros(cell_lat_flat.size, dtype=bool)
    for lat_i, lon_i in zip(hit_lat, hit_lon, strict=False):
        dist = _haversine_km(float(lat_i), float(lon_i), cell_lat_flat, cell_lon_flat)
        mask |= dist <= width_km
    if hit_lat.size == 1:
        return mask

    sort_idx = np.argsort(hit_ray_id, kind="mergesort")
    hit_lat = hit_lat[sort_idx]
    hit_lon = hit_lon[sort_idx]
    lat0 = float(np.mean(hit_lat))
    lon0 = float(np.mean(hit_lon))
    hit_xy = _project_equirect_km(hit_lat, hit_lon, lat0, lon0)
    cell_xy = _project_equirect_km(cell_lat_flat, cell_lon_flat, lat0, lon0)

    max_seg_km = float(max(cfg.max_segment_length_km, 1.0))
    for i in range(hit_xy.shape[0] - 1):
        if float(np.linalg.norm(hit_xy[i + 1] - hit_xy[i])) > max_seg_km:
            continue
        dist = _point_to_segment_distance_km(cell_xy, hit_xy[i], hit_xy[i + 1])
        mask |= dist <= width_km
    return mask


def _flight_track_coverage_mask(
    track_lat: np.ndarray,
    track_lon: np.ndarray,
    cell_lat_flat: np.ndarray,
    cell_lon_flat: np.ndarray,
    cfg: PopulationConfig,
) -> np.ndarray:
    if track_lat.size == 0:
        return np.zeros(cell_lat_flat.size, dtype=bool)

    width_km = float(max(cfg.overflight_half_width_km, 0.5))
    if track_lat.size == 1:
        dist = _haversine_km(float(track_lat[0]), float(track_lon[0]), cell_lat_flat, cell_lon_flat)
        return dist <= width_km

    lat0 = float(np.mean(track_lat))
    lon0 = float(np.mean(track_lon))
    track_xy = _project_equirect_km(track_lat, track_lon, lat0, lon0)
    cell_xy = _project_equirect_km(cell_lat_flat, cell_lon_flat, lat0, lon0)
    max_seg_km = float(max(cfg.max_segment_length_km, 1.0))

    mask = np.zeros(cell_xy.shape[0], dtype=bool)
    for i in range(track_xy.shape[0] - 1):
        if float(np.linalg.norm(track_xy[i + 1] - track_xy[i])) > max_seg_km:
            continue
        dist = _point_to_segment_distance_km(cell_xy, track_xy[i], track_xy[i + 1])
        mask |= dist <= width_km

    if not np.any(mask):
        for lat_i, lon_i in zip(track_lat, track_lon, strict=False):
            dist = _haversine_km(float(lat_i), float(lon_i), cell_lat_flat, cell_lon_flat)
            mask |= dist <= width_km
    return mask


def analyze_population_impact(
    emissions: list[EmissionResult],
    cfg: PopulationConfig,
) -> PopulationImpactResult | None:
    if not cfg.enabled:
        return None

    source = _load_population_source(cfg.dataset_path)
    lat_min, lat_max, lon_min, lon_max = _analysis_domain_bounds(emissions, float(max(cfg.domain_pad_deg, 0.0)))

    if isinstance(source, _PopulationGrid):
        dataset_name = source.source_name
        dataset_path = source.source_path
        lat_edges, lon_edges, heatmap_pop = _clip_population_grid_to_domain(
            source,
            lat_min,
            lat_max,
            lon_min,
            lon_max,
        )
    else:
        points = source
        if points.population.size == 0:
            return None

        points_mask = (
            (points.lat_deg >= lat_min)
            & (points.lat_deg <= lat_max)
            & (points.lon_deg >= lon_min)
            & (points.lon_deg <= lon_max)
        )
        if np.any(points_mask):
            points = _PopulationPoints(
                source_name=points.source_name,
                source_path=points.source_path,
                lat_deg=points.lat_deg[points_mask],
                lon_deg=points.lon_deg[points_mask],
                population=points.population[points_mask],
            )
        else:
            points = _PopulationPoints(
                source_name=points.source_name,
                source_path=points.source_path,
                lat_deg=np.asarray([], dtype=np.float64),
                lon_deg=np.asarray([], dtype=np.float64),
                population=np.asarray([], dtype=np.float64),
            )

        lat_edges, lon_edges = _adaptive_grid(
            lat_min,
            lat_max,
            lon_min,
            lon_max,
            cell_deg=float(cfg.heatmap_cell_deg),
            max_cells=int(cfg.max_grid_cells),
        )
        heatmap_pop = _aggregate_population_to_grid(points, lat_edges, lon_edges)
        dataset_name = points.source_name
        dataset_path = points.source_path

    lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    lon_centers = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    lon_grid, lat_grid = np.meshgrid(lon_centers, lat_centers)
    cell_lat_flat = lat_grid.reshape(-1)
    cell_lon_flat = lon_grid.reshape(-1)
    pop_flat = heatmap_pop.reshape(-1)

    dlat_km = np.abs(np.diff(lat_edges)) * _KM_PER_DEG
    dlon_km = np.abs(np.diff(lon_edges))[None, :] * (_KM_PER_DEG * np.cos(np.deg2rad(lat_centers)))[:, None]
    cell_area_km2 = np.abs(dlat_km[:, None] * dlon_km)
    cell_area_flat = cell_area_km2.reshape(-1)

    hit_lat: list[float] = []
    hit_lon: list[float] = []
    hit_time_utc: list[str] = []
    hit_emission_idx: list[int] = []
    hit_ray_id: list[int] = []
    emission_hit_groups: list[list[int]] = [[] for _ in emissions]

    for emission_idx, emission in enumerate(emissions):
        for ray in emission.rays:
            if not ray.ground_hit or ray.ground_hit_lat_lon is None:
                continue
            lat_i, lon_i = ray.ground_hit_lat_lon
            idx = len(hit_lat)
            hit_lat.append(float(lat_i))
            hit_lon.append(float(lon_i))
            hit_time_utc.append(emission.emission_time_utc.isoformat())
            hit_emission_idx.append(int(emission_idx))
            hit_ray_id.append(int(ray.ray_id))
            emission_hit_groups[emission_idx].append(idx)

    hit_lat_arr = np.asarray(hit_lat, dtype=np.float64)
    hit_lon_arr = normalize_lon_deg(np.asarray(hit_lon, dtype=np.float64))
    hit_ray_id_arr = np.asarray(hit_ray_id, dtype=np.int32)
    hit_emission_idx_arr = np.asarray(hit_emission_idx, dtype=np.int32)
    hit_pop_local = np.zeros(hit_lat_arr.size, dtype=np.float64)

    positive_pop = pop_flat > 0.0
    if hit_lat_arr.size and np.any(positive_pop):
        pop_lat = cell_lat_flat[positive_pop]
        pop_lon = cell_lon_flat[positive_pop]
        pop_val = pop_flat[positive_pop]
        hit_radius_km = float(max(cfg.hit_radius_km, 0.1))
        for i in range(hit_lat_arr.size):
            dist = _haversine_km(float(hit_lat_arr[i]), float(hit_lon_arr[i]), pop_lat, pop_lon)
            hit_pop_local[i] = float(np.sum(pop_val[dist <= hit_radius_km]))

    emission_exposed_pop = np.zeros(len(emissions), dtype=np.float64)
    emission_exposed_area_km2 = np.zeros(len(emissions), dtype=np.float64)
    emission_ground_hit_count = np.zeros(len(emissions), dtype=np.int32)
    exposed_union_flat = np.zeros(cell_lat_flat.size, dtype=bool)

    for emission_idx, group in enumerate(emission_hit_groups):
        emission_ground_hit_count[emission_idx] = int(len(group))
        if not group:
            continue
        g_idx = np.asarray(group, dtype=np.int64)
        mask = _emission_coverage_mask(
            hit_lat_arr[g_idx],
            hit_lon_arr[g_idx],
            hit_ray_id_arr[g_idx],
            cell_lat_flat,
            cell_lon_flat,
            cfg,
        )
        if not np.any(mask):
            continue
        emission_exposed_pop[emission_idx] = float(np.sum(pop_flat[mask]))
        emission_exposed_area_km2[emission_idx] = float(np.sum(cell_area_flat[mask]))
        exposed_union_flat |= mask

    track_lat = np.asarray([float(e.aircraft_lat_deg) for e in emissions], dtype=np.float64)
    track_lon = normalize_lon_deg(np.asarray([float(e.aircraft_lon_deg) for e in emissions], dtype=np.float64))
    overflight_union_flat = _flight_track_coverage_mask(track_lat, track_lon, cell_lat_flat, cell_lon_flat, cfg)

    return PopulationImpactResult(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        heatmap_cell_deg=float(np.mean(np.diff(lat_edges))) if lat_edges.size > 1 else float(cfg.heatmap_cell_deg),
        hit_radius_km=float(cfg.hit_radius_km),
        trace_half_width_km=float(cfg.trace_half_width_km),
        overflight_half_width_km=float(cfg.overflight_half_width_km),
        total_population_in_heatmap=float(np.sum(heatmap_pop)),
        total_exposed_population=float(np.sum(pop_flat[exposed_union_flat])),
        total_exposed_area_km2=float(np.sum(cell_area_flat[exposed_union_flat])),
        total_overflight_population=float(np.sum(pop_flat[overflight_union_flat])),
        total_overflight_area_km2=float(np.sum(cell_area_flat[overflight_union_flat])),
        heatmap_lat_edges_deg=lat_edges.astype(np.float32),
        heatmap_lon_edges_deg=lon_edges.astype(np.float32),
        heatmap_population=heatmap_pop.astype(np.float32),
        exposed_cell_mask=exposed_union_flat.reshape(heatmap_pop.shape),
        overflight_cell_mask=overflight_union_flat.reshape(heatmap_pop.shape),
        hit_lat_deg=hit_lat_arr.astype(np.float32),
        hit_lon_deg=hit_lon_arr.astype(np.float32),
        hit_emission_index=hit_emission_idx_arr,
        hit_ray_id=hit_ray_id_arr,
        hit_time_utc=hit_time_utc,
        hit_population_within_radius=hit_pop_local.astype(np.float32),
        emission_exposed_population=emission_exposed_pop.astype(np.float32),
        emission_exposed_area_km2=emission_exposed_area_km2.astype(np.float32),
        emission_ground_hit_count=emission_ground_hit_count,
    )
