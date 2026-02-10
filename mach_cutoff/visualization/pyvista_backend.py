"""PyVista visualizations."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..simulation.outputs import SimulationResult


def _polyline(points: np.ndarray):
    import pyvista as pv

    pts = np.asarray(points, dtype=float)
    poly = pv.PolyData(pts)
    cells = np.hstack([[pts.shape[0]], np.arange(pts.shape[0], dtype=np.int32)])
    poly.lines = cells
    return poly


def render_pyvista_bundle(
    result: SimulationResult,
    output_dir: str | Path,
    *,
    make_animation: bool = True,
    max_rays_per_emission: int = 20,
):
    try:
        import pyvista as pv
    except ImportError as exc:
        raise ImportError("pyvista is required for pyvista backend") from exc

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    generated = {}

    # Static 3D screenshot
    plotter = pv.Plotter(off_screen=True, window_size=(1400, 900))
    flight_xyz = np.array(
        [
            [e.aircraft_lon_deg, e.aircraft_lat_deg, e.aircraft_alt_m / 1000.0]
            for e in result.emissions
        ],
        dtype=float,
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
    plotter.close()
    generated["3d_screenshot_png"] = str(ps)

    # Ground-hit top view screenshot
    lat_hits, lon_hits, _ = result.all_ground_hits()
    if lat_hits.size:
        p2 = pv.Plotter(off_screen=True, window_size=(1200, 800))
        ground_pts = np.column_stack([lon_hits, lat_hits, np.zeros_like(lat_hits)])
        p2.add_points(ground_pts, color="crimson", point_size=6.0, render_points_as_spheres=True)
        track_pts = np.column_stack([flight_xyz[:, 0], flight_xyz[:, 1], np.zeros(flight_xyz.shape[0])])
        p2.add_mesh(_polyline(track_pts), color="black", line_width=2.0)
        p2.view_xy()
        p2.set_background("white")
        pg = out_dir / "pyvista_ground_hits.png"
        p2.screenshot(pg)
        p2.close()
        generated["ground_hits_png"] = str(pg)

    # Animated GIF over emissions
    if make_animation and result.emissions:
        pa = out_dir / "pyvista_3d_animation.gif"
        p3 = pv.Plotter(off_screen=True, window_size=(1200, 800))
        p3.open_gif(str(pa), fps=8)

        for i, emission in enumerate(result.emissions):
            p3.clear()
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
