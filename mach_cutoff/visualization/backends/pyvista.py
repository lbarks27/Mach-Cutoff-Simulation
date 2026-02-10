"""PyVista visualizations."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ...simulation.outputs import SimulationResult
from ..terrain import downsample_terrain_grid, terrain_grid_from_result


def _polyline(points: np.ndarray):
    import pyvista as pv

    pts = np.asarray(points, dtype=float)
    poly = pv.PolyData(pts)
    cells = np.hstack([[pts.shape[0]], np.arange(pts.shape[0], dtype=np.int32)])
    poly.lines = cells
    return poly


def _terrain_mesh(lat_deg: np.ndarray, lon_deg: np.ndarray, elevation_km: np.ndarray):
    import pyvista as pv

    mesh = pv.StructuredGrid(lon_deg, lat_deg, elevation_km)
    mesh["terrain_km"] = elevation_km.ravel(order="F")
    return mesh


def render_pyvista_bundle(
    result: SimulationResult,
    output_dir: str | Path,
    *,
    make_animation: bool = True,
    max_rays_per_emission: int = 20,
    show_window: bool = False,
):
    try:
        import pyvista as pv
    except ImportError as exc:
        raise ImportError("pyvista is required for pyvista backend") from exc

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    generated = {}

    # Static 3D screenshot
    plotter = pv.Plotter(off_screen=not show_window, window_size=(1400, 900))
    flight_xyz = np.array(
        [
            [e.aircraft_lon_deg, e.aircraft_lat_deg, e.aircraft_alt_m / 1000.0]
            for e in result.emissions
        ],
        dtype=float,
    )
    terrain_data = None
    terrain_grid = terrain_grid_from_result(result)
    if terrain_grid is not None:
        terrain_data = downsample_terrain_grid(*terrain_grid, max_points_per_axis=180)

    terrain_mesh = None
    if terrain_data is not None:
        t_lat, t_lon, t_elev = terrain_data
        terrain_mesh = _terrain_mesh(t_lat, t_lon, t_elev / 1000.0)
        plotter.add_mesh(
            terrain_mesh,
            scalars="terrain_km",
            cmap="terrain",
            opacity=0.65,
            show_scalar_bar=False,
            smooth_shading=True,
        )

    plotter.add_mesh(_polyline(flight_xyz), color="black", line_width=4.0)

    for emission in result.emissions:
        for ray in emission.rays[:max_rays_per_emission]:
            g = ray.trajectory_geodetic
            pts = np.column_stack([g[:, 1], g[:, 0], g[:, 2] / 1000.0])
            plotter.add_mesh(_polyline(pts), color="#1f77b4", opacity=0.28, line_width=1.2)

    plotter.add_axes()
    plotter.set_background("white")
    ps = out_dir / "pyvista_3d.png"
    plotter.screenshot(ps)
    if show_window:
        plotter.show(title="Mach Cutoff 3D View", auto_close=False)
    plotter.close()
    generated["3d_screenshot_png"] = str(ps)

    # Ground-hit top view screenshot
    lat_hits, lon_hits, _ = result.all_ground_hits()
    if lat_hits.size:
        p2 = pv.Plotter(off_screen=not show_window, window_size=(1200, 800))
        overlay_z = 0.0
        if terrain_data is not None:
            t_lat, t_lon, t_elev = terrain_data
            t_km = t_elev / 1000.0
            overlay_z = float(np.max(t_km)) + 0.05
            p2.add_mesh(
                _terrain_mesh(t_lat, t_lon, t_km),
                scalars="terrain_km",
                cmap="terrain",
                opacity=0.85,
                show_scalar_bar=False,
                smooth_shading=True,
            )

        ground_pts = np.column_stack([lon_hits, lat_hits, np.full_like(lat_hits, overlay_z)])
        p2.add_points(ground_pts, color="crimson", point_size=6.0, render_points_as_spheres=True)
        track_pts = np.column_stack([flight_xyz[:, 0], flight_xyz[:, 1], np.full(flight_xyz.shape[0], overlay_z)])
        p2.add_mesh(_polyline(track_pts), color="black", line_width=2.0)
        p2.view_xy()
        p2.set_background("white")
        pg = out_dir / "pyvista_ground_hits.png"
        p2.screenshot(pg)
        if show_window:
            p2.show(title="Mach Cutoff Ground Footprint", auto_close=False)
        p2.close()
        generated["ground_hits_png"] = str(pg)

    # Animated GIF over emissions
    if make_animation and result.emissions:
        if show_window:
            import time

            p3 = pv.Plotter(off_screen=False, window_size=(1200, 800))
            p3.show(title="Mach Cutoff Animation", auto_close=False, interactive_update=True)
            for i, emission in enumerate(result.emissions):
                p3.clear()
                if terrain_data is not None:
                    t_lat, t_lon, t_elev = terrain_data
                    p3.add_mesh(
                        _terrain_mesh(t_lat, t_lon, t_elev / 1000.0),
                        scalars="terrain_km",
                        cmap="terrain",
                        opacity=0.65,
                        show_scalar_bar=False,
                        smooth_shading=True,
                    )
                partial_track = flight_xyz[: i + 1]
                p3.add_mesh(_polyline(partial_track), color="black", line_width=4.0)
                for ray in emission.rays[:max_rays_per_emission]:
                    g = ray.trajectory_geodetic
                    pts = np.column_stack([g[:, 1], g[:, 0], g[:, 2] / 1000.0])
                    p3.add_mesh(_polyline(pts), color="#1f77b4", opacity=0.35, line_width=1.4)
                p3.add_axes()
                p3.set_background("white")
                p3.render()
                p3.update()
                time.sleep(0.12)
            p3.close()
            generated["animation_window"] = "shown"
        else:
            pa = out_dir / "pyvista_3d_animation.gif"
            p3 = pv.Plotter(off_screen=True, window_size=(1200, 800))
            p3.open_gif(str(pa), fps=8)

            for i, emission in enumerate(result.emissions):
                p3.clear()
                if terrain_data is not None:
                    t_lat, t_lon, t_elev = terrain_data
                    p3.add_mesh(
                        _terrain_mesh(t_lat, t_lon, t_elev / 1000.0),
                        scalars="terrain_km",
                        cmap="terrain",
                        opacity=0.65,
                        show_scalar_bar=False,
                        smooth_shading=True,
                    )
                partial_track = flight_xyz[: i + 1]
                p3.add_mesh(_polyline(partial_track), color="black", line_width=4.0)
                for ray in emission.rays[:max_rays_per_emission]:
                    g = ray.trajectory_geodetic
                    pts = np.column_stack([g[:, 1], g[:, 0], g[:, 2] / 1000.0])
                    p3.add_mesh(_polyline(pts), color="#1f77b4", opacity=0.35, line_width=1.4)
                p3.add_axes()
                p3.set_background("white")
                p3.write_frame()

            p3.close()
            generated["animation_gif"] = str(pa)

    return generated
