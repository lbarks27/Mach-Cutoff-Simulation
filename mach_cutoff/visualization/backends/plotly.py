"""Plotly visualizations."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ...simulation.outputs import SimulationResult


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

    # 3D figure
    fig3d = go.Figure()
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
