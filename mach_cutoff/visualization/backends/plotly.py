"""Plotly visualizations."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ...simulation.outputs import SimulationResult
from ..terrain import downsample_terrain_grid, terrain_grid_from_result


def render_plotly_bundle(
    result: SimulationResult,
    output_dir: str | Path,
    *,
    write_html: bool = True,
    max_rays_per_emission: int = 24,
):
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError("plotly is required for plotly backend") from exc

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    generated = {}

    flight_lats = [e.aircraft_lat_deg for e in result.emissions]
    flight_lons = [e.aircraft_lon_deg for e in result.emissions]
    flight_alt = [e.aircraft_alt_m for e in result.emissions]
    terrain_3d = None
    terrain_2d = None
    terrain_grid = terrain_grid_from_result(result)
    if terrain_grid is not None:
        terrain_3d = downsample_terrain_grid(*terrain_grid, max_points_per_axis=120)
        terrain_2d = downsample_terrain_grid(*terrain_grid, max_points_per_axis=180)

    # 3D figure
    fig3d = go.Figure()
    if terrain_3d is not None:
        t_lat, t_lon, t_elev = terrain_3d
        fig3d.add_trace(
            go.Surface(
                x=t_lon,
                y=t_lat,
                z=t_elev,
                colorscale="Earth",
                opacity=0.55,
                name="Terrain",
                showscale=True,
                colorbar=dict(title="Terrain (m MSL)"),
            )
        )
    fig3d.add_trace(
        go.Scatter3d(
            x=flight_lons,
            y=flight_lats,
            z=flight_alt,
            mode="lines",
            line=dict(color="black", width=5),
            name="Flight",
        )
    )

    for emission in result.emissions:
        for ray in emission.rays[:max_rays_per_emission]:
            g = ray.trajectory_geodetic
            fig3d.add_trace(
                go.Scatter3d(
                    x=g[:, 1],
                    y=g[:, 0],
                    z=g[:, 2],
                    mode="lines",
                    line=dict(color="royalblue", width=2),
                    opacity=0.25,
                    showlegend=False,
                )
            )

    fig3d.update_layout(
        title="Mach Cutoff 3D Flight and Rays",
        scene=dict(
            xaxis_title="Longitude (deg)",
            yaxis_title="Latitude (deg)",
            zaxis_title="Altitude (m)",
        ),
    )

    if write_html:
        p3d = out_dir / "plotly_3d.html"
        fig3d.write_html(p3d, include_plotlyjs="cdn")
        generated["3d_html"] = str(p3d)

    # Ground hits
    lat_hits, lon_hits, _ = result.all_ground_hits()
    fig2d = go.Figure()
    if terrain_2d is not None:
        t_lat2, t_lon2, t_elev2 = terrain_2d
        fig2d.add_trace(
            go.Scattergl(
                x=t_lon2.ravel(),
                y=t_lat2.ravel(),
                mode="markers",
                marker=dict(
                    size=3,
                    opacity=0.45,
                    color=t_elev2.ravel(),
                    colorscale="Earth",
                    showscale=True,
                    colorbar=dict(title="Terrain (m MSL)"),
                ),
                name="Terrain",
                hoverinfo="skip",
            )
        )
    fig2d.add_trace(
        go.Scatter(
            x=flight_lons,
            y=flight_lats,
            mode="lines",
            line=dict(color="black", width=2),
            name="Flight",
        )
    )
    if lat_hits.size:
        fig2d.add_trace(
            go.Scatter(
                x=lon_hits,
                y=lat_hits,
                mode="markers",
                marker=dict(color="crimson", size=5, opacity=0.7),
                name="Ground hits",
            )
        )
    fig2d.update_layout(
        title="Ground Intersection Footprint",
        xaxis_title="Longitude (deg)",
        yaxis_title="Latitude (deg)",
    )

    if write_html:
        pg = out_dir / "plotly_ground_hits.html"
        fig2d.write_html(pg, include_plotlyjs="cdn")
        generated["ground_hits_html"] = str(pg)

    # Vertical profiles
    figv = go.Figure()
    if result.emissions:
        emission = result.emissions[0]
        for ray in emission.rays[:max_rays_per_emission]:
            local = ray.trajectory_local_m
            horizontal = np.linalg.norm(local[:, :2], axis=1) / 1000.0
            altitude = (local[:, 2] + emission.aircraft_alt_m) / 1000.0
            figv.add_trace(
                go.Scatter(
                    x=horizontal,
                    y=altitude,
                    mode="lines",
                    line=dict(color="green", width=1),
                    opacity=0.45,
                    showlegend=False,
                )
            )
    figv.update_layout(
        title="Vertical Movement of Rays (Emission 0)",
        xaxis_title="Horizontal distance (km)",
        yaxis_title="Altitude (km)",
    )

    if write_html:
        pv = out_dir / "plotly_vertical.html"
        figv.write_html(pv, include_plotlyjs="cdn")
        generated["vertical_html"] = str(pv)

    return generated
