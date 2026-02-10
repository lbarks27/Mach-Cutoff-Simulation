"""Matplotlib visualizations."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..simulation.outputs import SimulationResult


def render_matplotlib_bundle(
    result: SimulationResult,
    output_dir: str | Path,
    *,
    make_animation: bool = True,
    max_rays_per_emission: int = 24,
):
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter
    except ImportError as exc:
        raise ImportError("matplotlib is required for matplotlib backend") from exc

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    generated = {}

    # 3D static view
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    flight_lats = [e.aircraft_lat_deg for e in result.emissions]
    flight_lons = [e.aircraft_lon_deg for e in result.emissions]
    flight_alt_km = [e.aircraft_alt_m / 1000.0 for e in result.emissions]

    ax.plot(flight_lons, flight_lats, flight_alt_km, color="black", linewidth=2.0, label="Flight")

    for emission in result.emissions:
        for ray in emission.rays[:max_rays_per_emission]:
            geo = ray.trajectory_geodetic
            ax.plot(geo[:, 1], geo[:, 0], geo[:, 2] / 1000.0, alpha=0.25, linewidth=0.6, color="#1f77b4")

    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.set_zlabel("Altitude (km)")
    ax.set_title("Mach Cutoff: 3D Flight + Ray Propagation")
    ax.legend(loc="upper left")

    p3d = out_dir / "matplotlib_3d.png"
    fig.tight_layout()
    fig.savefig(p3d, dpi=180)
    plt.close(fig)
    generated["3d_static_png"] = str(p3d)

    # Ground-hit map
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    lat_hits, lon_hits, _ = result.all_ground_hits()
    if lat_hits.size:
        ax2.scatter(lon_hits, lat_hits, s=8, alpha=0.6, c="crimson", label="Sonic boom ground hits")
    ax2.plot(flight_lons, flight_lats, color="black", linewidth=1.5, label="Flight track")
    ax2.set_xlabel("Longitude (deg)")
    ax2.set_ylabel("Latitude (deg)")
    ax2.set_title("Ground Intersection Footprint")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="best")

    pg = out_dir / "matplotlib_ground_hits.png"
    fig2.tight_layout()
    fig2.savefig(pg, dpi=180)
    plt.close(fig2)
    generated["ground_hits_png"] = str(pg)

    # Vertical ray section (first emission)
    if result.emissions and result.emissions[0].rays:
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        emission = result.emissions[0]
        for ray in emission.rays[:max_rays_per_emission]:
            local = ray.trajectory_local_m
            horizontal_km = np.linalg.norm(local[:, :2], axis=1) / 1000.0
            altitude_km = (local[:, 2] + emission.aircraft_alt_m) / 1000.0
            ax3.plot(horizontal_km, altitude_km, alpha=0.35, linewidth=0.8, color="#2ca02c")
        ax3.set_xlabel("Horizontal distance from source (km)")
        ax3.set_ylabel("Altitude (km MSL)")
        ax3.set_title("Vertical Movement of Rays (Emission 0)")
        ax3.grid(True, alpha=0.25)

        pv = out_dir / "matplotlib_vertical_section.png"
        fig3.tight_layout()
        fig3.savefig(pv, dpi=180)
        plt.close(fig3)
        generated["vertical_section_png"] = str(pv)

    # 3D animation (emission-by-emission)
    if make_animation and result.emissions:
        fig4 = plt.figure(figsize=(12, 8))
        ax4 = fig4.add_subplot(111, projection="3d")

        lon_min = min(flight_lons)
        lon_max = max(flight_lons)
        lat_min = min(flight_lats)
        lat_max = max(flight_lats)
        alt_max = max(flight_alt_km)

        ax4.set_xlim(lon_min - 0.5, lon_max + 0.5)
        ax4.set_ylim(lat_min - 0.5, lat_max + 0.5)
        ax4.set_zlim(0.0, alt_max + 5.0)

        flight_line, = ax4.plot([], [], [], color="black", linewidth=2.0)
        ray_lines = [ax4.plot([], [], [], color="#1f77b4", alpha=0.4, linewidth=0.8)[0] for _ in range(max_rays_per_emission)]

        ax4.set_xlabel("Longitude (deg)")
        ax4.set_ylabel("Latitude (deg)")
        ax4.set_zlabel("Altitude (km)")
        ax4.set_title("Mach Cutoff Animation")

        def init():
            flight_line.set_data([], [])
            flight_line.set_3d_properties([])
            for line in ray_lines:
                line.set_data([], [])
                line.set_3d_properties([])
            return [flight_line, *ray_lines]

        def update(frame_idx):
            e = result.emissions[frame_idx]
            flight_line.set_data(flight_lons[: frame_idx + 1], flight_lats[: frame_idx + 1])
            flight_line.set_3d_properties(flight_alt_km[: frame_idx + 1])

            for i, line in enumerate(ray_lines):
                if i < len(e.rays):
                    geo = e.rays[i].trajectory_geodetic
                    line.set_data(geo[:, 1], geo[:, 0])
                    line.set_3d_properties(geo[:, 2] / 1000.0)
                else:
                    line.set_data([], [])
                    line.set_3d_properties([])
            return [flight_line, *ray_lines]

        anim = FuncAnimation(fig4, update, frames=len(result.emissions), init_func=init, blit=False)
        pa = out_dir / "matplotlib_3d_animation.gif"
        try:
            anim.save(pa, writer=PillowWriter(fps=8))
            generated["animation_gif"] = str(pa)
        finally:
            plt.close(fig4)

    return generated
