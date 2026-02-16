"""Basemap style and tile helpers for visualization backends."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import base64
import math
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen

import numpy as np

MAP_STYLE_TOPOGRAPHIC = "topographic"
MAP_STYLE_ROAD = "road"
MAP_STYLE_SATELLITE = "satellite"
MAP_STYLE_CHOICES = (MAP_STYLE_TOPOGRAPHIC, MAP_STYLE_ROAD, MAP_STYLE_SATELLITE)

_MAX_WEB_MERCATOR_LAT = 85.05112878


@dataclass(slots=True)
class BasemapTile:
    image_rgba: np.ndarray
    extent_lon_lat: tuple[float, float, float, float]
    attribution: str


def normalize_map_style(map_style: str | None) -> str:
    if not map_style:
        return MAP_STYLE_TOPOGRAPHIC
    style = str(map_style).strip().lower()
    aliases = {
        "topo": MAP_STYLE_TOPOGRAPHIC,
        "topography": MAP_STYLE_TOPOGRAPHIC,
        "osm": MAP_STYLE_ROAD,
        "street": MAP_STYLE_ROAD,
        "streets": MAP_STYLE_ROAD,
        "roads": MAP_STYLE_ROAD,
        "imagery": MAP_STYLE_SATELLITE,
    }
    style = aliases.get(style, style)
    if style not in MAP_STYLE_CHOICES:
        raise ValueError(f"Unsupported map_style: {map_style!r}. Expected one of {MAP_STYLE_CHOICES}.")
    return style


def terrain_matplotlib_cmap(map_style: str) -> str:
    style = normalize_map_style(map_style)
    if style == MAP_STYLE_ROAD:
        return "Greys"
    if style == MAP_STYLE_SATELLITE:
        return "gist_earth"
    return "terrain"


def terrain_plotly_colorscale(map_style: str) -> str:
    style = normalize_map_style(map_style)
    if style == MAP_STYLE_ROAD:
        return "Greys"
    return "Earth"


def terrain_pyvista_cmap(map_style: str) -> str:
    # PyVista uses matplotlib colormap names.
    return terrain_matplotlib_cmap(map_style)


def _clip_lat_deg(lat_deg: float) -> float:
    return max(-_MAX_WEB_MERCATOR_LAT, min(_MAX_WEB_MERCATOR_LAT, float(lat_deg)))


def _lon_to_xtile(lon_deg: float, zoom: int) -> float:
    n = 2.0**zoom
    return n * (float(lon_deg) + 180.0) / 360.0


def _lat_to_ytile(lat_deg: float, zoom: int) -> float:
    n = 2.0**zoom
    lat_rad = math.radians(_clip_lat_deg(lat_deg))
    return n * (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0


def _xtile_to_lon(x_tile: float, zoom: int) -> float:
    n = 2.0**zoom
    return float(x_tile / n * 360.0 - 180.0)


def _ytile_to_lat(y_tile: float, zoom: int) -> float:
    n = 2.0**zoom
    lat_rad = math.atan(math.sinh(math.pi * (1.0 - 2.0 * y_tile / n)))
    return float(math.degrees(lat_rad))


def _choose_zoom(lon_span_deg: float, lat_span_deg: float, min_zoom: int = 2, max_zoom: int = 13) -> int:
    span = max(lon_span_deg, lat_span_deg, 1e-4)
    zoom = int(math.floor(math.log2(360.0 / span)) + 1)
    return max(min_zoom, min(max_zoom, zoom))


def _tile_provider(style: str) -> tuple[str, str]:
    style = normalize_map_style(style)
    if style == MAP_STYLE_ROAD:
        return (
            "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
            "Map data and tiles: OpenStreetMap contributors",
        )
    if style == MAP_STYLE_SATELLITE:
        return (
            "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            "Imagery: Esri, Maxar, Earthstar Geographics, and GIS User Community",
        )
    raise ValueError("Topographic style does not use remote basemap tiles.")


def fetch_basemap_tile(
    *,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    map_style: str,
    max_tiles: int = 36,
    timeout_s: float = 6.0,
) -> BasemapTile | None:
    style = normalize_map_style(map_style)
    if style == MAP_STYLE_TOPOGRAPHIC:
        return None

    lon_min_f = float(min(lon_min, lon_max))
    lon_max_f = float(max(lon_min, lon_max))
    lat_min_f = _clip_lat_deg(min(lat_min, lat_max))
    lat_max_f = _clip_lat_deg(max(lat_min, lat_max))

    lon_pad = max(0.05, (lon_max_f - lon_min_f) * 0.08)
    lat_pad = max(0.05, (lat_max_f - lat_min_f) * 0.08)
    lon_min_f -= lon_pad
    lon_max_f += lon_pad
    lat_min_f = _clip_lat_deg(lat_min_f - lat_pad)
    lat_max_f = _clip_lat_deg(lat_max_f + lat_pad)

    zoom = _choose_zoom(lon_max_f - lon_min_f, lat_max_f - lat_min_f)
    url_template, attribution = _tile_provider(style)

    x_min = x_max = y_min = y_max = 0
    for z in range(zoom, 1, -1):
        x0 = int(math.floor(_lon_to_xtile(lon_min_f, z)))
        x1 = int(math.floor(_lon_to_xtile(lon_max_f, z)))
        y0 = int(math.floor(_lat_to_ytile(lat_max_f, z)))
        y1 = int(math.floor(_lat_to_ytile(lat_min_f, z)))
        n = 2**z
        x0 = max(0, min(n - 1, x0))
        x1 = max(0, min(n - 1, x1))
        y0 = max(0, min(n - 1, y0))
        y1 = max(0, min(n - 1, y1))
        nx = x1 - x0 + 1
        ny = y1 - y0 + 1
        if nx <= 0 or ny <= 0:
            continue
        if nx * ny <= max_tiles or z <= 2:
            zoom = z
            x_min, x_max, y_min, y_max = x0, x1, y0, y1
            break

    try:
        from PIL import Image
    except Exception:
        return None

    tile_size = 256
    nx = x_max - x_min + 1
    ny = y_max - y_min + 1
    if nx <= 0 or ny <= 0:
        return None

    canvas = np.zeros((ny * tile_size, nx * tile_size, 4), dtype=np.uint8)
    canvas[:, :, :3] = 245
    canvas[:, :, 3] = 255
    ok_tiles = 0

    user_agent = "mach-cutoff-sim/0.1 (+https://local)"
    for iy, y in enumerate(range(y_min, y_max + 1)):
        for ix, x in enumerate(range(x_min, x_max + 1)):
            url = url_template.format(z=zoom, x=x, y=y)
            req = Request(url, headers={"User-Agent": user_agent})
            try:
                with urlopen(req, timeout=timeout_s) as resp:
                    raw = resp.read()
                with Image.open(BytesIO(raw)) as img:
                    tile = np.asarray(img.convert("RGBA"), dtype=np.uint8)
                if tile.shape[0] != tile_size or tile.shape[1] != tile_size:
                    continue
                y0 = iy * tile_size
                x0 = ix * tile_size
                canvas[y0 : y0 + tile_size, x0 : x0 + tile_size, :] = tile
                ok_tiles += 1
            except (URLError, HTTPError, TimeoutError, OSError):
                continue

    if ok_tiles == 0:
        return None

    lon_left = _xtile_to_lon(x_min, zoom)
    lon_right = _xtile_to_lon(x_max + 1, zoom)
    lat_top = _ytile_to_lat(y_min, zoom)
    lat_bottom = _ytile_to_lat(y_max + 1, zoom)

    return BasemapTile(
        image_rgba=canvas,
        extent_lon_lat=(lon_left, lon_right, lat_bottom, lat_top),
        attribution=attribution,
    )


def rgba_to_png_data_uri(image_rgba: np.ndarray) -> str | None:
    try:
        from PIL import Image
    except Exception:
        return None

    try:
        img = Image.fromarray(image_rgba.astype(np.uint8), mode="RGBA")
    except Exception:
        return None

    buf = BytesIO()
    img.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"
