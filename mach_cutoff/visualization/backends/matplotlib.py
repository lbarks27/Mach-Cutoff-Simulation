"""Matplotlib visualizations."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ...simulation.outputs import SimulationResult
from ..terrain import downsample_terrain_grid, terrain_grid_from_result


def render_matplotlib_bundle(
    result: SimulationResult,
    output_dir: str | Path,
    *,
    make_animation: bool = True,
    max_rays_per_emission: int = 24,
    include_atmosphere: bool = True,
    show_window: bool = False,
):
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter
    except ImportError as exc:
        raise ImportError("matplotlib is required for matplotlib backend") from exc

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    generated = {}
    figures_to_show = []
    animations_to_show = []

    # 3D static view
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    flight_lats = [e.aircraft_lat_deg for e in result.emissions]
    flight_lons = [e.aircraft_lon_deg for e in result.emissions]
    flight_alt_km = [e.aircraft_alt_m / 1000.0 for e in result.emissions]

    terrain_3d = None
    terrain_2d = None
    terrain_grid = terrain_grid_from_result(result)
    if terrain_grid is not None:
        terrain_3d = downsample_terrain_grid(*terrain_grid, max_points_per_axis=120)
        terrain_2d = downsample_terrain_grid(*terrain_grid, max_points_per_axis=180)

    terrain_min_km = 0.0
    terrain_max_km = 0.0
    if terrain_3d is not None:
        t_lat, t_lon, t_elev = terrain_3d
        terrain_km = t_elev / 1000.0
        terrain_min_km = float(np.min(terrain_km))
        terrain_max_km = float(np.max(terrain_km))
        ax.plot_surface(t_lon, t_lat, terrain_km, cmap="terrain", linewidth=0, antialiased=False, alpha=0.5)

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
    if flight_alt_km:
        ax.set_zlim(min(terrain_min_km, 0.0) - 0.2, max(max(flight_alt_km), terrain_max_km) + 5.0)

    p3d = out_dir / "matplotlib_3d.png"
    fig.tight_layout()
    fig.savefig(p3d, dpi=180)
    if show_window:
        figures_to_show.append(fig)
    else:
        plt.close(fig)
    generated["3d_static_png"] = str(p3d)

    # Ground-hit map
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    lat_hits, lon_hits, _ = result.all_ground_hits()
    if terrain_2d is not None:
        t_lat2, t_lon2, t_elev2 = terrain_2d
        terrain_plot = ax2.contourf(t_lon2, t_lat2, t_elev2, levels=28, cmap="terrain", alpha=0.5)
        cbar = fig2.colorbar(terrain_plot, ax=ax2, pad=0.02, fraction=0.05)
        cbar.set_label("Terrain elevation (m MSL)")
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
    if show_window:
        figures_to_show.append(fig2)
    else:
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
        if show_window:
            figures_to_show.append(fig3)
        else:
            plt.close(fig3)
        generated["vertical_section_png"] = str(pv)

    # Atmospheric time-series diagnostics
    if include_atmosphere and result.atmospheric_time_series is not None:
        ts = result.atmospheric_time_series
        figa, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

        x = np.arange(len(ts.emission_times_utc), dtype=float)
        x_label = "Emission index"
        if ts.emission_times_utc:
            try:
                import matplotlib.dates as mdates

                x = mdates.date2num(ts.emission_times_utc)
                axes[2].xaxis_date()
                axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
                figa.autofmt_xdate(rotation=25)
                x_label = "Emission time (UTC)"
            except Exception:
                pass

        axes[0].plot(x, ts.temperature_k - 273.15, color="#d62728", linewidth=1.8, label="Temperature (C)")
        ax0b = axes[0].twinx()
        ax0b.plot(x, ts.relative_humidity_pct, color="#1f77b4", linewidth=1.4, alpha=0.9, label="RH (%)")
        h0, l0 = axes[0].get_legend_handles_labels()
        h0b, l0b = ax0b.get_legend_handles_labels()
        axes[0].legend(h0 + h0b, l0 + l0b, loc="best")
        axes[0].set_ylabel("Temperature (C)")
        ax0b.set_ylabel("Relative humidity (%)")
        axes[0].grid(True, alpha=0.22)

        axes[1].plot(x, ts.u_wind_mps, color="#2ca02c", linewidth=1.4, label="U wind (m/s)")
        axes[1].plot(x, ts.v_wind_mps, color="#17becf", linewidth=1.4, label="V wind (m/s)")
        axes[1].plot(x, ts.wind_projection_mps, color="#9467bd", linewidth=1.4, linestyle="--", label="Wind projection (m/s)")
        axes[1].axhline(0.0, color="0.5", linewidth=0.8, alpha=0.8)
        axes[1].set_ylabel("Wind speed (m/s)")
        axes[1].grid(True, alpha=0.22)
        axes[1].legend(loc="best")

        axes[2].plot(x, ts.sound_speed_mps, color="#8c564b", linewidth=1.3, linestyle="--", label="Sound speed (m/s)")
        axes[2].plot(
            x,
            ts.effective_sound_speed_mps,
            color="#ff7f0e",
            linewidth=1.8,
            label="Effective sound speed (m/s)",
        )
        ax2b = axes[2].twinx()
        ax2b.plot(x, ts.pressure_hpa, color="#7f7f7f", linewidth=1.2, alpha=0.9, label="Pressure (hPa)")
        h2, l2 = axes[2].get_legend_handles_labels()
        h2b, l2b = ax2b.get_legend_handles_labels()
        axes[2].legend(h2 + h2b, l2 + l2b, loc="best")
        axes[2].set_ylabel("Speed (m/s)")
        ax2b.set_ylabel("Pressure (hPa)")
        axes[2].set_xlabel(x_label)
        axes[2].grid(True, alpha=0.22)

        axes[0].set_title("Atmospheric Conditions at Aircraft Position")

        pa = out_dir / "matplotlib_atmospheric_timeseries.png"
        figa.tight_layout()
        figa.savefig(pa, dpi=180)
        if show_window:
            figures_to_show.append(figa)
        else:
            plt.close(figa)
        generated["atmospheric_timeseries_png"] = str(pa)

    # Atmospheric vertical profile diagnostics
    if include_atmosphere and result.atmospheric_vertical_profile is not None:
        profile = result.atmospheric_vertical_profile
        altitude_km = profile.altitude_m / 1000.0

        figp, axp = plt.subplots(2, 3, figsize=(15, 10), sharey=True)
        ax_temp = axp[0, 0]
        ax_rh = axp[0, 1]
        ax_pressure = axp[0, 2]
        ax_wind = axp[1, 0]
        ax_speed = axp[1, 1]
        ax_info = axp[1, 2]

        ax_temp.plot(profile.temperature_k - 273.15, altitude_km, color="#d62728", linewidth=1.8)
        ax_temp.set_xlabel("Temperature (C)")
        ax_temp.set_ylabel("Altitude (km MSL)")
        ax_temp.grid(True, alpha=0.22)

        ax_rh.plot(profile.relative_humidity_pct, altitude_km, color="#1f77b4", linewidth=1.8)
        ax_rh.set_xlabel("Relative humidity (%)")
        ax_rh.grid(True, alpha=0.22)

        ax_pressure.plot(profile.pressure_hpa, altitude_km, color="#7f7f7f", linewidth=1.8)
        ax_pressure.set_xlabel("Pressure (hPa)")
        ax_pressure.grid(True, alpha=0.22)

        ax_wind.plot(profile.u_wind_mps, altitude_km, color="#2ca02c", linewidth=1.4, label="U wind (m/s)")
        ax_wind.plot(profile.v_wind_mps, altitude_km, color="#17becf", linewidth=1.4, label="V wind (m/s)")
        ax_wind.plot(
            profile.wind_projection_mps,
            altitude_km,
            color="#9467bd",
            linewidth=1.4,
            linestyle="--",
            label="Wind projection (m/s)",
        )
        ax_wind.axvline(0.0, color="0.5", linewidth=0.8, alpha=0.8)
        ax_wind.set_xlabel("Wind speed (m/s)")
        ax_wind.set_ylabel("Altitude (km MSL)")
        ax_wind.grid(True, alpha=0.22)
        ax_wind.legend(loc="best")

        ax_speed.plot(profile.sound_speed_mps, altitude_km, color="#8c564b", linewidth=1.4, linestyle="--", label="Sound speed (m/s)")
        ax_speed.plot(
            profile.effective_sound_speed_mps,
            altitude_km,
            color="#ff7f0e",
            linewidth=1.8,
            label="Effective sound speed (m/s)",
        )
        ax_speed.set_xlabel("Speed (m/s)")
        ax_speed.grid(True, alpha=0.22)
        ax_speed.legend(loc="best")

        ax_info.axis("off")
        ax_info.text(0.0, 0.92, "Profile metadata", fontsize=11, fontweight="bold", transform=ax_info.transAxes)
        ax_info.text(0.0, 0.76, f"Emission: {profile.emission_time_utc.isoformat()}", fontsize=9, transform=ax_info.transAxes)
        ax_info.text(0.0, 0.62, f"Lat: {profile.aircraft_lat_deg:.4f} deg", fontsize=9, transform=ax_info.transAxes)
        ax_info.text(0.0, 0.52, f"Lon: {profile.aircraft_lon_deg:.4f} deg", fontsize=9, transform=ax_info.transAxes)
        ax_info.text(0.0, 0.42, f"Alt: {profile.aircraft_alt_m:.0f} m", fontsize=9, transform=ax_info.transAxes)

        figp.suptitle("Atmospheric Vertical Profile at First Emission", fontsize=14, y=0.98)

        pap = out_dir / "matplotlib_atmospheric_profile.png"
        figp.tight_layout()
        figp.savefig(pap, dpi=180)
        if show_window:
            figures_to_show.append(figp)
        else:
            plt.close(figp)
        generated["atmospheric_profile_png"] = str(pap)

    # 3D animation (emission-by-emission)
    if make_animation and result.emissions:
        fig4 = plt.figure(figsize=(12, 8))
        ax4 = fig4.add_subplot(111, projection="3d")

        lon_min = min(flight_lons)
        lon_max = max(flight_lons)
        lat_min = min(flight_lats)
        lat_max = max(flight_lats)
        alt_max = max(flight_alt_km)

        terrain_anim_min_km = 0.0
        if terrain_3d is not None:
            t_lat, t_lon, t_elev = terrain_3d
            terrain_km = t_elev / 1000.0
            terrain_anim_min_km = float(np.min(terrain_km))
            lon_min = min(lon_min, float(np.min(t_lon)))
            lon_max = max(lon_max, float(np.max(t_lon)))
            lat_min = min(lat_min, float(np.min(t_lat)))
            lat_max = max(lat_max, float(np.max(t_lat)))
            alt_max = max(alt_max, float(np.max(terrain_km)))
            ax4.plot_surface(t_lon, t_lat, terrain_km, cmap="terrain", linewidth=0, antialiased=False, alpha=0.5)

        ax4.set_xlim(lon_min - 0.5, lon_max + 0.5)
        ax4.set_ylim(lat_min - 0.5, lat_max + 0.5)
        ax4.set_zlim(min(terrain_anim_min_km, 0.0) - 0.2, alt_max + 5.0)

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
        if show_window:
            animations_to_show.append(anim)
            figures_to_show.append(fig4)
            generated["animation_window"] = "shown"
        else:
            try:
                anim.save(pa, writer=PillowWriter(fps=8))
                generated["animation_gif"] = str(pa)
            finally:
                plt.close(fig4)

    if show_window and figures_to_show:
        try:
            plt.show()
        finally:
            for fig_ in figures_to_show:
                plt.close(fig_)

    return generated
