"""Plotly visualizations."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ...core.geodesy import geodetic_to_ecef
from ...simulation.outputs import SimulationResult
from ..basemap import (
    MAP_STYLE_TOPOGRAPHIC,
    fetch_basemap_tile,
    normalize_map_style,
    rgba_to_png_data_uri,
    terrain_plotly_colorscale,
)
from ..terrain import downsample_terrain_grid, terrain_grid_from_result


def render_plotly_bundle(
    result: SimulationResult,
    output_dir: str | Path,
    *,
    write_html: bool = True,
    max_rays_per_emission: int = 24,
    include_atmosphere: bool = True,
    open_browser: bool = False,
    map_style: str = MAP_STYLE_TOPOGRAPHIC,
):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise ImportError("plotly is required for plotly backend") from exc

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    generated = {}
    map_style = normalize_map_style(map_style)
    terrain_colorscale = terrain_plotly_colorscale(map_style)

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
    lon_min = float(np.min(flight_lons)) if flight_lons else -1.0
    lon_max = float(np.max(flight_lons)) if flight_lons else 1.0
    lat_min = float(np.min(flight_lats)) if flight_lats else -1.0
    lat_max = float(np.max(flight_lats)) if flight_lats else 1.0
    z_min_m = float(np.min(flight_alt)) if flight_alt else 0.0
    z_max_m = float(np.max(flight_alt)) if flight_alt else 1.0
    if terrain_3d is not None:
        t_lat, t_lon, t_elev = terrain_3d
        lon_min = min(lon_min, float(np.min(t_lon)))
        lon_max = max(lon_max, float(np.max(t_lon)))
        lat_min = min(lat_min, float(np.min(t_lat)))
        lat_max = max(lat_max, float(np.max(t_lat)))
        z_min_m = min(z_min_m, float(np.min(t_elev)))
        z_max_m = max(z_max_m, float(np.max(t_elev)))
        fig3d.add_trace(
            go.Surface(
                x=t_lon,
                y=t_lat,
                z=t_elev,
                colorscale=terrain_colorscale,
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
            lon_min = min(lon_min, float(np.min(g[:, 1])))
            lon_max = max(lon_max, float(np.max(g[:, 1])))
            lat_min = min(lat_min, float(np.min(g[:, 0])))
            lat_max = max(lat_max, float(np.max(g[:, 0])))
            z_min_m = min(z_min_m, float(np.min(g[:, 2])))
            z_max_m = max(z_max_m, float(np.max(g[:, 2])))
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

    mean_lat_deg = 0.5 * (lat_min + lat_max)
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = max(1.0, 111_320.0 * float(np.cos(np.deg2rad(mean_lat_deg))))
    x_span_m = max(1.0, (lon_max - lon_min) * meters_per_deg_lon)
    y_span_m = max(1.0, (lat_max - lat_min) * meters_per_deg_lat)
    z_span_m = max(1.0, z_max_m - z_min_m)
    span_ref = max(x_span_m, y_span_m, z_span_m)

    fig3d.update_layout(
        title="Mach Cutoff 3D Flight and Rays",
        scene=dict(
            xaxis=dict(title="Longitude (deg)", range=[lon_min, lon_max]),
            yaxis=dict(title="Latitude (deg)", range=[lat_min, lat_max]),
            zaxis=dict(title="Altitude (m)", range=[z_min_m, z_max_m]),
            aspectmode="manual",
            aspectratio=dict(x=x_span_m / span_ref, y=y_span_m / span_ref, z=z_span_m / span_ref),
        ),
    )

    if write_html:
        p3d = out_dir / "plotly_3d.html"
        fig3d.write_html(p3d, include_plotlyjs="cdn")
        generated["3d_html"] = str(p3d)
        if open_browser:
            import webbrowser

            opened = webbrowser.open_new_tab(p3d.resolve().as_uri())
            if not opened:
                fig3d.show(renderer="browser")
    elif open_browser:
        fig3d.show()

    # Interactive globe figure (ECEF)
    globe_fig = go.Figure()
    globe_lats = np.linspace(-90.0, 90.0, 90)
    globe_lons = np.linspace(-180.0, 180.0, 180)
    globe_lon_grid, globe_lat_grid = np.meshgrid(globe_lons, globe_lats)
    globe_alt = np.zeros_like(globe_lat_grid)
    globe_xyz = geodetic_to_ecef(globe_lat_grid, globe_lon_grid, globe_alt)
    globe_fig.add_trace(
        go.Surface(
            x=globe_xyz[..., 0],
            y=globe_xyz[..., 1],
            z=globe_xyz[..., 2],
            surfacecolor=globe_lat_grid,
            colorscale=[
                [0.0, "#2d6a8b"],
                [0.3, "#5f8f5d"],
                [0.5, "#8eaa72"],
                [0.7, "#c7bb8a"],
                [1.0, "#f2f0de"],
            ],
            cmin=-90.0,
            cmax=90.0,
            showscale=False,
            name="Earth",
            hoverinfo="skip",
            lighting=dict(ambient=0.65, diffuse=0.85, roughness=0.9, specular=0.15),
            lightposition=dict(x=80_000.0, y=60_000.0, z=95_000.0),
            opacity=0.95,
        )
    )

    if flight_lats:
        flight_lat_arr = np.asarray(flight_lats, dtype=float)
        flight_lon_arr = np.asarray(flight_lons, dtype=float)
        flight_alt_arr = np.asarray(flight_alt, dtype=float)
        flight_xyz = geodetic_to_ecef(flight_lat_arr, flight_lon_arr, flight_alt_arr)
        globe_fig.add_trace(
            go.Scatter3d(
                x=flight_xyz[:, 0],
                y=flight_xyz[:, 1],
                z=flight_xyz[:, 2],
                mode="lines",
                line=dict(color="black", width=6),
                name="Flight",
                customdata=np.column_stack([flight_lat_arr, flight_lon_arr, flight_alt_arr]),
                hovertemplate="Lat: %{customdata[0]:.3f}<br>Lon: %{customdata[1]:.3f}<br>Alt: %{customdata[2]:.0f} m<extra>Flight</extra>",
            )
        )

    max_altitude_candidates = [0.0, *flight_alt]
    for emission in result.emissions:
        for ray in emission.rays[:max_rays_per_emission]:
            g = ray.trajectory_geodetic
            ray_xyz = geodetic_to_ecef(g[:, 0], g[:, 1], g[:, 2])
            if g.size:
                max_altitude_candidates.append(float(np.nanmax(g[:, 2])))
            globe_fig.add_trace(
                go.Scatter3d(
                    x=ray_xyz[:, 0],
                    y=ray_xyz[:, 1],
                    z=ray_xyz[:, 2],
                    mode="lines",
                    line=dict(color="royalblue", width=2),
                    opacity=0.2,
                    showlegend=False,
                    customdata=g,
                    hovertemplate="Lat: %{customdata[0]:.3f}<br>Lon: %{customdata[1]:.3f}<br>Alt: %{customdata[2]:.0f} m<extra>Ray</extra>",
                )
            )

    lat_hits, lon_hits, _ = result.all_ground_hits()
    if lat_hits.size:
        hit_xyz = geodetic_to_ecef(lat_hits, lon_hits, np.zeros_like(lat_hits))
        globe_fig.add_trace(
            go.Scatter3d(
                x=hit_xyz[:, 0],
                y=hit_xyz[:, 1],
                z=hit_xyz[:, 2],
                mode="markers",
                marker=dict(color="crimson", size=3, opacity=0.75),
                name="Ground hits",
                customdata=np.column_stack([lat_hits, lon_hits]),
                hovertemplate="Lat: %{customdata[0]:.3f}<br>Lon: %{customdata[1]:.3f}<extra>Ground hit</extra>",
            )
        )

    max_altitude_m = float(np.nanmax(np.asarray(max_altitude_candidates, dtype=float)))
    if not np.isfinite(max_altitude_m):
        max_altitude_m = 0.0
    earth_extent = float(np.max(np.abs(globe_xyz)))
    axis_limit = earth_extent + max(10_000.0, 1.1 * max_altitude_m)
    globe_fig.update_layout(
        title="Mach Cutoff Interactive Globe",
        scene=dict(
            xaxis=dict(visible=False, showgrid=False, zeroline=False, range=[-axis_limit, axis_limit]),
            yaxis=dict(visible=False, showgrid=False, zeroline=False, range=[-axis_limit, axis_limit]),
            zaxis=dict(visible=False, showgrid=False, zeroline=False, range=[-axis_limit, axis_limit]),
            aspectmode="data",
        ),
        legend=dict(y=0.96, x=0.01),
        margin=dict(l=0, r=0, t=60, b=0),
    )

    if write_html:
        pglobe = out_dir / "plotly_globe.html"
        globe_fig.write_html(pglobe, include_plotlyjs="cdn")
        generated["globe_html"] = str(pglobe)

    # Ground hits
    fig2d = go.Figure()
    lon_candidates = list(flight_lons)
    lat_candidates = list(flight_lats)
    if lon_hits.size:
        lon_candidates.extend(lon_hits.tolist())
    if lat_hits.size:
        lat_candidates.extend(lat_hits.tolist())
    if terrain_2d is not None:
        t_lat2, t_lon2, t_elev2 = terrain_2d
        lon_candidates.extend([float(np.min(t_lon2)), float(np.max(t_lon2))])
        lat_candidates.extend([float(np.min(t_lat2)), float(np.max(t_lat2))])

    drew_remote_basemap = False
    if map_style != MAP_STYLE_TOPOGRAPHIC and lon_candidates and lat_candidates:
        basemap = fetch_basemap_tile(
            lon_min=min(lon_candidates),
            lon_max=max(lon_candidates),
            lat_min=min(lat_candidates),
            lat_max=max(lat_candidates),
            map_style=map_style,
        )
        if basemap is not None:
            source = rgba_to_png_data_uri(basemap.image_rgba)
            if source is not None:
                lon_left, lon_right, lat_bottom, lat_top = basemap.extent_lon_lat
                fig2d.add_layout_image(
                    dict(
                        source=source,
                        xref="x",
                        yref="y",
                        x=lon_left,
                        y=lat_top,
                        sizex=lon_right - lon_left,
                        sizey=lat_top - lat_bottom,
                        sizing="stretch",
                        opacity=1.0,
                        layer="below",
                    )
                )
                fig2d.add_annotation(
                    xref="paper",
                    yref="paper",
                    x=0.01,
                    y=0.01,
                    text=basemap.attribution,
                    showarrow=False,
                    font=dict(size=9, color="#222"),
                    bgcolor="rgba(255,255,255,0.7)",
                )
                drew_remote_basemap = True

    if terrain_2d is not None and not drew_remote_basemap:
        fig2d.add_trace(
            go.Scattergl(
                x=t_lon2.ravel(),
                y=t_lat2.ravel(),
                mode="markers",
                marker=dict(
                    size=3,
                    opacity=0.45,
                    color=t_elev2.ravel(),
                    colorscale=terrain_colorscale,
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
        xaxis=dict(scaleanchor="y", scaleratio=1),
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

    # Atmospheric time-series diagnostics
    if include_atmosphere and result.atmospheric_time_series is not None:
        ts = result.atmospheric_time_series
        x = ts.emission_times_utc if ts.emission_times_utc else np.arange(len(ts.temperature_k), dtype=float)
        x_label = "Emission time (UTC)" if ts.emission_times_utc else "Emission index"

        figa = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            specs=[[{"secondary_y": True}], [{"secondary_y": False}], [{"secondary_y": True}]],
        )
        figa.add_trace(
            go.Scatter(x=x, y=ts.temperature_k - 273.15, name="Temperature (C)", line=dict(color="#d62728", width=2)),
            row=1,
            col=1,
            secondary_y=False,
        )
        figa.add_trace(
            go.Scatter(
                x=x,
                y=ts.relative_humidity_pct,
                name="RH (%)",
                line=dict(color="#1f77b4", width=1.6),
            ),
            row=1,
            col=1,
            secondary_y=True,
        )

        figa.add_trace(
            go.Scatter(x=x, y=ts.u_wind_mps, name="U wind (m/s)", line=dict(color="#2ca02c", width=1.5)),
            row=2,
            col=1,
        )
        figa.add_trace(
            go.Scatter(x=x, y=ts.v_wind_mps, name="V wind (m/s)", line=dict(color="#17becf", width=1.5)),
            row=2,
            col=1,
        )
        figa.add_trace(
            go.Scatter(
                x=x,
                y=ts.wind_projection_mps,
                name="Wind projection (m/s)",
                line=dict(color="#9467bd", width=1.5, dash="dash"),
            ),
            row=2,
            col=1,
        )
        if np.size(x) > 0:
            figa.add_shape(
                type="line",
                x0=x[0],
                x1=x[-1],
                y0=0.0,
                y1=0.0,
                line=dict(color="gray", width=1, dash="dot"),
                row=2,
                col=1,
            )

        figa.add_trace(
            go.Scatter(
                x=x,
                y=ts.sound_speed_mps,
                name="Sound speed (m/s)",
                line=dict(color="#8c564b", width=1.5, dash="dash"),
            ),
            row=3,
            col=1,
            secondary_y=False,
        )
        figa.add_trace(
            go.Scatter(
                x=x,
                y=ts.effective_sound_speed_mps,
                name="Effective sound speed (m/s)",
                line=dict(color="#ff7f0e", width=2),
            ),
            row=3,
            col=1,
            secondary_y=False,
        )
        figa.add_trace(
            go.Scatter(
                x=x,
                y=ts.pressure_hpa,
                name="Pressure (hPa)",
                line=dict(color="#7f7f7f", width=1.4),
            ),
            row=3,
            col=1,
            secondary_y=True,
        )
        figa.update_layout(title="Atmospheric Conditions at Aircraft Position", legend=dict(orientation="h", y=-0.14))
        figa.update_yaxes(title_text="Temperature (C)", row=1, col=1, secondary_y=False)
        figa.update_yaxes(title_text="Relative humidity (%)", row=1, col=1, secondary_y=True)
        figa.update_yaxes(title_text="Wind speed (m/s)", row=2, col=1)
        figa.update_yaxes(title_text="Speed (m/s)", row=3, col=1, secondary_y=False)
        figa.update_yaxes(title_text="Pressure (hPa)", row=3, col=1, secondary_y=True)
        figa.update_xaxes(title_text=x_label, row=3, col=1)

        if write_html:
            pa = out_dir / "plotly_atmospheric_timeseries.html"
            figa.write_html(pa, include_plotlyjs="cdn")
            generated["atmospheric_timeseries_html"] = str(pa)

    # Atmospheric vertical profile diagnostics
    if include_atmosphere and result.atmospheric_vertical_profile is not None:
        profile = result.atmospheric_vertical_profile
        altitude_km = profile.altitude_m / 1000.0

        figp = make_subplots(
            rows=2,
            cols=3,
            shared_yaxes=True,
            horizontal_spacing=0.08,
            vertical_spacing=0.12,
            subplot_titles=(
                "Temperature",
                "Relative humidity",
                "Pressure",
                "Wind components",
                "Acoustic speeds",
                "Metadata",
            ),
        )

        figp.add_trace(
            go.Scatter(x=profile.temperature_k - 273.15, y=altitude_km, name="Temperature (C)", line=dict(color="#d62728", width=2)),
            row=1,
            col=1,
        )
        figp.add_trace(
            go.Scatter(x=profile.relative_humidity_pct, y=altitude_km, name="RH (%)", line=dict(color="#1f77b4", width=2)),
            row=1,
            col=2,
        )
        figp.add_trace(
            go.Scatter(x=profile.pressure_hpa, y=altitude_km, name="Pressure (hPa)", line=dict(color="#7f7f7f", width=2)),
            row=1,
            col=3,
        )
        figp.add_trace(
            go.Scatter(x=profile.u_wind_mps, y=altitude_km, name="U wind (m/s)", line=dict(color="#2ca02c", width=1.6)),
            row=2,
            col=1,
        )
        figp.add_trace(
            go.Scatter(x=profile.v_wind_mps, y=altitude_km, name="V wind (m/s)", line=dict(color="#17becf", width=1.6)),
            row=2,
            col=1,
        )
        figp.add_trace(
            go.Scatter(
                x=profile.wind_projection_mps,
                y=altitude_km,
                name="Wind projection (m/s)",
                line=dict(color="#9467bd", width=1.5, dash="dash"),
            ),
            row=2,
            col=1,
        )
        figp.add_shape(
            type="line",
            x0=0.0,
            x1=0.0,
            y0=float(np.min(altitude_km)),
            y1=float(np.max(altitude_km)),
            line=dict(color="gray", width=1, dash="dot"),
            row=2,
            col=1,
        )
        figp.add_trace(
            go.Scatter(
                x=profile.sound_speed_mps,
                y=altitude_km,
                name="Sound speed (m/s)",
                line=dict(color="#8c564b", width=1.5, dash="dash"),
            ),
            row=2,
            col=2,
        )
        figp.add_trace(
            go.Scatter(
                x=profile.effective_sound_speed_mps,
                y=altitude_km,
                name="Effective sound speed (m/s)",
                line=dict(color="#ff7f0e", width=2),
            ),
            row=2,
            col=2,
        )

        figp.update_xaxes(title_text="Temperature (C)", row=1, col=1)
        figp.update_xaxes(title_text="Relative humidity (%)", row=1, col=2)
        figp.update_xaxes(title_text="Pressure (hPa)", row=1, col=3)
        figp.update_xaxes(title_text="Wind speed (m/s)", row=2, col=1)
        figp.update_xaxes(title_text="Speed (m/s)", row=2, col=2)
        figp.update_yaxes(title_text="Altitude (km MSL)", row=1, col=1)
        figp.update_yaxes(title_text="Altitude (km MSL)", row=2, col=1)

        figp.update_xaxes(visible=False, row=2, col=3)
        figp.update_yaxes(visible=False, row=2, col=3)
        figp.add_annotation(
            text=(
                "<b>Profile metadata</b><br>"
                f"Emission: {profile.emission_time_utc.isoformat()}<br>"
                f"Lat: {profile.aircraft_lat_deg:.4f} deg<br>"
                f"Lon: {profile.aircraft_lon_deg:.4f} deg<br>"
                f"Alt: {profile.aircraft_alt_m:.0f} m"
            ),
            x=0.0,
            y=1.0,
            xref="x6 domain",
            yref="y6 domain",
            showarrow=False,
            align="left",
        )

        figp.update_layout(
            title="Atmospheric Vertical Profile at First Emission",
            legend=dict(orientation="h", y=-0.08),
        )

        if write_html:
            pap = out_dir / "plotly_atmospheric_profile.html"
            figp.write_html(pap, include_plotlyjs="cdn")
            generated["atmospheric_profile_html"] = str(pap)

    # Atmospheric 3D grid overlay with variable toggles
    if include_atmosphere and result.atmospheric_grid_3d is not None:
        grid = result.atmospheric_grid_3d
        nx, ny, nz = grid.temperature_k.shape

        def _slice_steps(nx0: int, ny0: int, nz0: int, tx: int, ty: int, tz: int):
            sx = max(1, int(np.ceil(nx0 / max(tx, 1))))
            sy = max(1, int(np.ceil(ny0 / max(ty, 1))))
            sz = max(1, int(np.ceil(nz0 / max(tz, 1))))
            return slice(0, nx0, sx), slice(0, ny0, sy), slice(0, nz0, sz)

        slx, sly, slz = _slice_steps(nx, ny, nz, tx=24, ty=24, tz=14)
        lon_xy = grid.lon_grid_deg[slx, sly]
        lat_xy = grid.lat_grid_deg[slx, sly]
        alt_1d = grid.altitude_m[slz]

        lon_3d = np.repeat(lon_xy[:, :, None], alt_1d.size, axis=2)
        lat_3d = np.repeat(lat_xy[:, :, None], alt_1d.size, axis=2)
        alt_3d = np.broadcast_to(alt_1d[None, None, :], lon_3d.shape)

        lon_flat = lon_3d.reshape(-1)
        lat_flat = lat_3d.reshape(-1)
        alt_flat = alt_3d.reshape(-1)

        temp_c = grid.temperature_k[slx, sly, slz] - 273.15
        rh = grid.relative_humidity_pct[slx, sly, slz]
        pressure = grid.pressure_hpa[slx, sly, slz]
        ceff = grid.effective_sound_speed_mps[slx, sly, slz]
        wind_speed = np.sqrt(grid.u_wind_mps[slx, sly, slz] ** 2 + grid.v_wind_mps[slx, sly, slz] ** 2)

        metric_specs = [
            ("Temperature (C)", temp_c, "Turbo", "C"),
            ("Relative Humidity (%)", rh, "Blues", "%"),
            ("Pressure (hPa)", pressure, "Viridis", "hPa"),
            ("Effective Sound Speed (m/s)", ceff, "Plasma", "m/s"),
            ("Wind Speed (m/s)", wind_speed, "YlGnBu", "m/s"),
        ]

        figa3d = go.Figure()
        if terrain_3d is not None:
            t_lat, t_lon, t_elev = terrain_3d
            figa3d.add_trace(
                go.Surface(
                    x=t_lon,
                    y=t_lat,
                    z=t_elev,
                    colorscale=terrain_colorscale,
                    opacity=0.45,
                    name="Terrain",
                    showscale=False,
                    hoverinfo="skip",
                )
            )

        figa3d.add_trace(
            go.Scatter3d(
                x=flight_lons,
                y=flight_lats,
                z=flight_alt,
                mode="lines",
                line=dict(color="black", width=5),
                name="Flight",
            )
        )

        slx_w, sly_w, slz_w = _slice_steps(nx, ny, nz, tx=12, ty=12, tz=8)
        lon_w_xy = grid.lon_grid_deg[slx_w, sly_w]
        lat_w_xy = grid.lat_grid_deg[slx_w, sly_w]
        alt_w_1d = grid.altitude_m[slz_w]
        u_w = grid.u_wind_mps[slx_w, sly_w, slz_w]
        v_w = grid.v_wind_mps[slx_w, sly_w, slz_w]

        lon_w_3d = np.repeat(lon_w_xy[:, :, None], alt_w_1d.size, axis=2)
        lat_w_3d = np.repeat(lat_w_xy[:, :, None], alt_w_1d.size, axis=2)
        alt_w_3d = np.broadcast_to(alt_w_1d[None, None, :], lon_w_3d.shape)

        meters_per_deg_lat = 111_320.0
        meters_per_deg_lon = np.maximum(111_320.0 * np.cos(np.deg2rad(lat_w_3d)), 1.0)
        wind_visual_window_s = 150.0
        dlon = (u_w * wind_visual_window_s) / meters_per_deg_lon
        dlat = (v_w * wind_visual_window_s) / meters_per_deg_lat

        tails_lon = lon_w_3d.reshape(-1)
        tails_lat = lat_w_3d.reshape(-1)
        tails_alt = alt_w_3d.reshape(-1)
        heads_lon = (lon_w_3d + dlon).reshape(-1)
        heads_lat = (lat_w_3d + dlat).reshape(-1)
        heads_alt = alt_w_3d.reshape(-1)

        line_x = np.empty(tails_lon.size * 3, dtype=float)
        line_y = np.empty(tails_lat.size * 3, dtype=float)
        line_z = np.empty(tails_alt.size * 3, dtype=float)
        line_x[0::3], line_x[1::3], line_x[2::3] = tails_lon, heads_lon, np.nan
        line_y[0::3], line_y[1::3], line_y[2::3] = tails_lat, heads_lat, np.nan
        line_z[0::3], line_z[1::3], line_z[2::3] = tails_alt, heads_alt, np.nan

        figa3d.add_trace(
            go.Scatter3d(
                x=line_x,
                y=line_y,
                z=line_z,
                mode="lines",
                line=dict(color="#0b3c5d", width=3),
                opacity=0.7,
                name="Wind vectors",
                hoverinfo="skip",
            )
        )
        wind_trace_idx = len(figa3d.data) - 1
        base_trace_count = len(figa3d.data)

        for idx, (label, values, colorscale, unit) in enumerate(metric_specs):
            finite = np.isfinite(values)
            if np.any(finite):
                cmin = float(np.nanpercentile(values[finite], 2.0))
                cmax = float(np.nanpercentile(values[finite], 98.0))
            else:
                cmin, cmax = 0.0, 1.0
            if cmax <= cmin:
                cmax = cmin + 1e-6

            figa3d.add_trace(
                go.Scatter3d(
                    x=lon_flat,
                    y=lat_flat,
                    z=alt_flat,
                    mode="markers",
                    marker=dict(
                        size=2.4,
                        opacity=0.34,
                        color=values.reshape(-1),
                        colorscale=colorscale,
                        cmin=cmin,
                        cmax=cmax,
                        colorbar=dict(title=f"{label} [{unit}]", x=1.02),
                    ),
                    name=label,
                    showlegend=False,
                    visible=idx == 0,
                    hovertemplate=f"Lon: %{{x:.3f}}<br>Lat: %{{y:.3f}}<br>Alt: %{{z:.0f}} m<br>{label}: %{{marker.color:.2f}} {unit}<extra></extra>",
                )
            )

        measure_count = len(metric_specs)
        buttons = []
        for i, (label, _, _, _) in enumerate(metric_specs):
            visible = [True] * base_trace_count + [False] * measure_count
            visible[base_trace_count + i] = True
            buttons.append(
                dict(
                    label=label,
                    method="update",
                    args=[
                        {"visible": visible},
                        {"title": f"3D Atmospheric Overlay on Terrain ({label})"},
                    ],
                )
            )

        figa3d.update_layout(
            title=f"3D Atmospheric Overlay on Terrain ({metric_specs[0][0]})",
            scene=dict(
                xaxis_title="Longitude (deg)",
                yaxis_title="Latitude (deg)",
                zaxis_title="Altitude (m MSL)",
                aspectmode="data",
            ),
            margin=dict(l=0, r=0, t=80, b=0),
            legend=dict(y=0.98, x=0.01),
            updatemenus=[
                dict(
                    type="dropdown",
                    direction="down",
                    x=0.01,
                    y=1.12,
                    showactive=True,
                    buttons=buttons,
                ),
                dict(
                    type="buttons",
                    direction="right",
                    x=0.43,
                    y=1.12,
                    showactive=True,
                    buttons=[
                        dict(label="Wind On", method="restyle", args=[{"visible": True}, [wind_trace_idx]]),
                        dict(label="Wind Off", method="restyle", args=[{"visible": False}, [wind_trace_idx]]),
                    ],
                ),
            ],
            annotations=[
                dict(
                    text=f"Snapshot: {grid.emission_time_utc.isoformat()}",
                    x=0.01,
                    y=1.04,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    align="left",
                )
            ],
        )

        if write_html:
            pa3 = out_dir / "plotly_atmosphere_3d.html"
            figa3d.write_html(pa3, include_plotlyjs="cdn")
            generated["atmosphere_3d_html"] = str(pa3)

    return generated
