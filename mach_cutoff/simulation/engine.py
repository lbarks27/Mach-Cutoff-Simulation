"""End-to-end simulation engine."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..flight.aircraft import PointMassAircraft, generate_shock_directions
from ..atmosphere.acoustics import build_acoustic_grid_field, compute_effective_sound_speed_mps
from ..atmosphere.hrrr import HRRRDatasetManager
from ..atmosphere.interpolation import HRRRInterpolator
from ..config import ExperimentConfig
from ..core.geodesy import ecef_to_geodetic, ecef_to_enu, enu_to_ecef, normalize_lon_deg
from ..core.raytrace import integrate_ray
from ..flight.waypoints import FlightPath, load_waypoints_json
from .outputs import (
    AtmosphericGrid3D,
    AtmosphericTimeSeries,
    AtmosphericVerticalProfile,
    EmissionResult,
    RayResult,
    SimulationResult,
)


def _parse_time_iso(value: str | None) -> datetime | None:
    if value is None:
        return None
    value = value.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _waypoint_bbox(path: FlightPath, pad_deg: float = 2.0):
    lats = np.array([w.lat_deg for w in path.waypoints], dtype=float)
    lons = normalize_lon_deg(np.array([w.lon_deg for w in path.waypoints], dtype=float))
    return (
        float(np.min(lats) - pad_deg),
        float(np.max(lats) + pad_deg),
        float(np.min(lons) - pad_deg),
        float(np.max(lons) + pad_deg),
    )


def _nearest_snapshot_time(snapshot_times: list[datetime], t: datetime) -> datetime:
    tt = t.timestamp()
    idx = min(range(len(snapshot_times)), key=lambda i: abs(snapshot_times[i].timestamp() - tt))
    return snapshot_times[idx]


def _terrain_from_snapshot(snapshot):
    lat = np.asarray(snapshot.lat_deg, dtype=np.float32)
    lon = np.asarray(snapshot.lon_deg, dtype=np.float32)
    gh = np.asarray(snapshot.geopotential_height_m, dtype=np.float32)

    if gh.ndim != 3 or lat.shape != lon.shape or gh.shape[1:] != lat.shape:
        return None

    terrain = np.nanmin(gh, axis=0).astype(np.float32)
    finite = np.isfinite(terrain)
    if not np.any(finite):
        return None
    if not np.all(finite):
        terrain = np.where(finite, terrain, np.nanmedian(terrain[finite])).astype(np.float32)
    return lat, lon, terrain


class MachCutoffSimulator:
    def __init__(self, config: ExperimentConfig):
        self.config = config

    def run_from_waypoint_file(self, waypoint_json_path: str | Path) -> SimulationResult:
        waypoints = load_waypoints_json(waypoint_json_path)
        path = FlightPath(waypoints)
        return self.run(path)

    def run(self, path: FlightPath) -> SimulationResult:
        aircraft = PointMassAircraft(path, self.config.aircraft)

        start_time = _parse_time_iso(self.config.runtime.start_time_iso) or aircraft.start_time
        end_time = _parse_time_iso(self.config.runtime.end_time_iso) or aircraft.end_time

        emission_times = path.sample_times(
            self.config.shock.emission_interval_s,
            start=start_time,
            end=end_time,
            clamp_to_path_bounds=False,
        )
        if self.config.runtime.max_emissions is not None:
            emission_times = emission_times[: self.config.runtime.max_emissions]
        if not emission_times:
            return SimulationResult(emissions=[], config_dict=self.config.to_dict())

        total_emissions = len(emission_times)
        first_emit_epoch = emission_times[0].timestamp()
        last_emit_epoch = emission_times[-1].timestamp()
        emission_window_duration_s = max(0.0, last_emit_epoch - first_emit_epoch)
        flight_start_epoch = aircraft.start_time.timestamp()
        flight_duration_s = max(0.0, aircraft.duration_s)

        bbox = _waypoint_bbox(path)
        hrrr_manager = HRRRDatasetManager(self.config.hrrr, bbox=bbox)
        snapshots_by_time = hrrr_manager.snapshots_for_times(emission_times)
        if not snapshots_by_time:
            raise RuntimeError("No HRRR snapshots were loaded for the requested emission times")
        snapshot_times = list(snapshots_by_time.keys())

        terrain_lat = None
        terrain_lon = None
        terrain_elev = None
        terrain_tuple = _terrain_from_snapshot(next(iter(snapshots_by_time.values())))
        if terrain_tuple is not None:
            terrain_lat, terrain_lon, terrain_elev = terrain_tuple

        interp_cache: dict[datetime, HRRRInterpolator] = {}
        emissions: list[EmissionResult] = []
        sample_times: list[datetime] = []
        sample_lats: list[float] = []
        sample_lons: list[float] = []
        sample_alts: list[float] = []
        sample_temp: list[float] = []
        sample_rh: list[float] = []
        sample_pressure: list[float] = []
        sample_u: list[float] = []
        sample_v: list[float] = []
        sample_c: list[float] = []
        sample_w_proj: list[float] = []
        sample_c_eff: list[float] = []
        vertical_profile: AtmosphericVerticalProfile | None = None
        atmospheric_grid_3d: AtmosphericGrid3D | None = None

        for emit_idx, emit_time in enumerate(emission_times):
            state = aircraft.state_at(emit_time)
            dirs_ecef = generate_shock_directions(state, self.config.shock)
            emit_epoch = emit_time.timestamp()
            elapsed_window_s = max(0.0, emit_epoch - first_emit_epoch)
            remaining_window_s = max(0.0, last_emit_epoch - emit_epoch)
            if emission_window_duration_s > 0.0:
                window_progress_pct = 100.0 * elapsed_window_s / emission_window_duration_s
            else:
                window_progress_pct = 100.0
            if flight_duration_s > 0.0:
                flight_progress_pct = 100.0 * np.clip((emit_epoch - flight_start_epoch) / flight_duration_s, 0.0, 1.0)
            else:
                flight_progress_pct = 100.0
            count_progress_pct = 100.0 * (emit_idx + 1) / total_emissions

            snap_time = _nearest_snapshot_time(snapshot_times, emit_time)
            if snap_time not in interp_cache:
                interp_cache[snap_time] = HRRRInterpolator(snapshots_by_time[snap_time])
            interp = interp_cache[snap_time]

            origin_ecef = np.asarray(state.position_ecef_m, dtype=float)
            velocity_enu = ecef_to_enu(
                state.lat_deg,
                state.lon_deg,
                state.alt_m,
                origin_ecef + np.asarray(state.velocity_ecef_mps, dtype=float),
            )
            velocity_enu = np.asarray(velocity_enu, dtype=np.float32).reshape(-1)

            point_sample = interp.sample_points(
                lat_deg=np.array([state.lat_deg], dtype=np.float32),
                lon_deg=np.array([state.lon_deg], dtype=np.float32),
                alt_m=np.array([state.alt_m], dtype=np.float32),
            )
            t0 = point_sample["temperature_k"].reshape(-1)
            rh0 = point_sample["relative_humidity_pct"].reshape(-1)
            p0 = point_sample["pressure_hpa"].reshape(-1)
            u0 = point_sample["u_wind_mps"].reshape(-1)
            v0 = point_sample["v_wind_mps"].reshape(-1)
            c0, w0, ce0 = compute_effective_sound_speed_mps(
                t0,
                p0,
                rh0,
                u0,
                v0,
                velocity_enu,
                self.config.grid.wind_projection_mode,
            )
            sample_times.append(emit_time)
            sample_lats.append(float(state.lat_deg))
            sample_lons.append(float(state.lon_deg))
            sample_alts.append(float(state.alt_m))
            sample_temp.append(float(t0[0]))
            sample_rh.append(float(rh0[0]))
            sample_pressure.append(float(p0[0]))
            sample_u.append(float(u0[0]))
            sample_v.append(float(v0[0]))
            sample_c.append(float(c0[0]))
            sample_w_proj.append(float(w0[0]))
            sample_c_eff.append(float(ce0[0]))
            ce0_scalar = float(ce0[0])
            effective_mach = float(state.speed_mps / max(ce0_scalar, 1e-6))
            source_mach_cutoff = bool(effective_mach <= 1.0)

            if vertical_profile is None:
                alt_profile = np.linspace(
                    self.config.grid.min_altitude_m,
                    self.config.grid.max_altitude_m,
                    self.config.grid.nz,
                    dtype=np.float32,
                )
                profile_sample = interp.sample_columns(
                    lat_deg=np.array([state.lat_deg], dtype=np.float32),
                    lon_deg=np.array([state.lon_deg], dtype=np.float32),
                    altitudes_m=alt_profile,
                )
                t_prof = profile_sample["temperature_k"][0, :]
                rh_prof = profile_sample["relative_humidity_pct"][0, :]
                p_prof = profile_sample["pressure_hpa"][0, :]
                u_prof = profile_sample["u_wind_mps"][0, :]
                v_prof = profile_sample["v_wind_mps"][0, :]
                c_prof, w_prof, ce_prof = compute_effective_sound_speed_mps(
                    t_prof,
                    p_prof,
                    rh_prof,
                    u_prof,
                    v_prof,
                    velocity_enu,
                    self.config.grid.wind_projection_mode,
                )

                vertical_profile = AtmosphericVerticalProfile(
                    emission_time_utc=emit_time,
                    aircraft_lat_deg=float(state.lat_deg),
                    aircraft_lon_deg=float(state.lon_deg),
                    aircraft_alt_m=float(state.alt_m),
                    altitude_m=alt_profile.astype(np.float32),
                    temperature_k=t_prof.astype(np.float32),
                    relative_humidity_pct=rh_prof.astype(np.float32),
                    pressure_hpa=p_prof.astype(np.float32),
                    u_wind_mps=u_prof.astype(np.float32),
                    v_wind_mps=v_prof.astype(np.float32),
                    sound_speed_mps=np.asarray(c_prof, dtype=np.float32),
                    wind_projection_mps=np.asarray(w_prof, dtype=np.float32),
                    effective_sound_speed_mps=np.asarray(ce_prof, dtype=np.float32),
                )

            field = None
            if (not source_mach_cutoff) or atmospheric_grid_3d is None:
                field, field_meta = build_acoustic_grid_field(
                    interp,
                    aircraft_lat_deg=state.lat_deg,
                    aircraft_lon_deg=state.lon_deg,
                    aircraft_alt_m=state.alt_m,
                    aircraft_velocity_ecef_mps=state.velocity_ecef_mps,
                    grid_config=self.config.grid,
                    reference_speed_mps=self.config.aircraft.reference_sound_speed_mps,
                )
                if atmospheric_grid_3d is None:
                    atmospheric_grid_3d = AtmosphericGrid3D(
                        emission_time_utc=emit_time,
                        aircraft_lat_deg=float(state.lat_deg),
                        aircraft_lon_deg=float(state.lon_deg),
                        aircraft_alt_m=float(state.alt_m),
                        lat_grid_deg=np.asarray(field_meta["lat_grid_deg"], dtype=np.float32),
                        lon_grid_deg=np.asarray(field_meta["lon_grid_deg"], dtype=np.float32),
                        altitude_m=np.asarray(field_meta["altitude_abs_m"], dtype=np.float32),
                        temperature_k=np.asarray(field_meta["temperature_k"], dtype=np.float32),
                        relative_humidity_pct=np.asarray(field_meta["relative_humidity_pct"], dtype=np.float32),
                        pressure_hpa=np.asarray(field_meta["pressure_hpa"], dtype=np.float32),
                        u_wind_mps=np.asarray(field_meta["u_wind_mps"], dtype=np.float32),
                        v_wind_mps=np.asarray(field_meta["v_wind_mps"], dtype=np.float32),
                        sound_speed_mps=np.asarray(field_meta["sound_speed_mps"], dtype=np.float32),
                        wind_projection_mps=np.asarray(field_meta["wind_projection_mps"], dtype=np.float32),
                        effective_sound_speed_mps=np.asarray(field_meta["c_eff_mps"], dtype=np.float32),
                    )

            def _ground_surface(local_pos):
                p_ecef = enu_to_ecef(state.lat_deg, state.lon_deg, state.alt_m, np.asarray(local_pos, dtype=float))
                _, _, alt = ecef_to_geodetic(p_ecef[0], p_ecef[1], p_ecef[2])
                return alt <= 0.0

            emission_result = EmissionResult(
                emission_time_utc=emit_time,
                aircraft_lat_deg=state.lat_deg,
                aircraft_lon_deg=state.lon_deg,
                aircraft_alt_m=state.alt_m,
                aircraft_position_ecef_m=origin_ecef,
                effective_mach=effective_mach,
                source_mach_cutoff=source_mach_cutoff,
                mach_cutoff=source_mach_cutoff,
                rays=[],
            )
            snapshot_offset_min = (emit_time - snap_time).total_seconds() / 60.0
            if source_mach_cutoff:
                emissions.append(emission_result)
                print(
                    "[sim] emission "
                    f"{emit_idx + 1}/{total_emissions} "
                    f"(count={count_progress_pct:5.1f}%, window={window_progress_pct:5.1f}%, route={flight_progress_pct:5.1f}%) "
                    f"time={emit_time.isoformat()} "
                    f"elapsed={elapsed_window_s / 60.0:7.2f}m remaining={remaining_window_s / 60.0:7.2f}m | "
                    f"aircraft lat={state.lat_deg:+8.4f} lon={state.lon_deg:+9.4f} alt={state.alt_m:7.1f}m "
                    f"({state.alt_m * 3.28084:8.1f}ft) speed={state.speed_mps:6.1f}m/s "
                    f"({state.speed_mps * 1.943844:6.1f}kt) mach={state.mach:.2f} "
                    f"eff_mach={effective_mach:.3f} | "
                    f"atmo T={float(t0[0]):6.2f}K RH={float(rh0[0]):5.1f}% p={float(p0[0]):7.2f}hPa "
                    f"c={float(c0[0]):6.2f}m/s w_proj={float(w0[0]):+6.2f}m/s c_eff={ce0_scalar:6.2f}m/s | "
                    f"hrrr={snap_time.isoformat()} dt={snapshot_offset_min:+6.2f}m | "
                    "SOURCE CUT OFF (effective mach <= 1.0), rays=0"
                )
                continue

            ground_hit_count = 0
            ray_point_count = 0
            terminal_alt_min_m = np.inf
            terminal_alt_max_m = -np.inf

            if field is None:
                raise RuntimeError("Acoustic field was not built for supersonic emission")

            for ray_idx, d_ecef in enumerate(dirs_ecef):
                d_local = ecef_to_enu(
                    state.lat_deg,
                    state.lon_deg,
                    state.alt_m,
                    origin_ecef + np.asarray(d_ecef, dtype=float),
                )
                d_local = np.asarray(d_local, dtype=float)
                d_local = d_local / np.linalg.norm(d_local)

                traj_local = integrate_ray(
                    r0=np.array([0.0, 0.0, 0.0], dtype=float),
                    dir0=d_local,
                    field=field,
                    ds=self.config.raytrace.ds_m,
                    steps=self.config.raytrace.max_steps,
                    adaptive=self.config.raytrace.adaptive,
                    tol=self.config.raytrace.tol,
                    min_step=self.config.raytrace.min_step_m,
                    max_step=self.config.raytrace.max_step_m,
                    domain_bounds=field.domain_bounds,
                    surfaces=[_ground_surface],
                    stop_on_exit=self.config.raytrace.stop_on_exit,
                )

                traj_ecef = enu_to_ecef(state.lat_deg, state.lon_deg, state.alt_m, traj_local)
                lat, lon, alt = ecef_to_geodetic(traj_ecef[:, 0], traj_ecef[:, 1], traj_ecef[:, 2])
                traj_geo = np.column_stack([lat, lon, alt])

                ground_hit = bool(traj_geo[-1, 2] <= 0.0)
                hit_lat_lon = (float(traj_geo[-1, 0]), float(traj_geo[-1, 1])) if ground_hit else None
                if ground_hit:
                    ground_hit_count += 1
                ray_point_count += int(traj_local.shape[0])
                terminal_alt_m = float(traj_geo[-1, 2])
                terminal_alt_min_m = min(terminal_alt_min_m, terminal_alt_m)
                terminal_alt_max_m = max(terminal_alt_max_m, terminal_alt_m)

                emission_result.rays.append(
                    RayResult(
                        ray_id=ray_idx,
                        trajectory_local_m=traj_local,
                        trajectory_ecef_m=traj_ecef,
                        trajectory_geodetic=traj_geo,
                        ground_hit=ground_hit,
                        ground_hit_lat_lon=hit_lat_lon,
                    )
                )

            emissions.append(emission_result)
            n_rays = len(emission_result.rays)
            emission_result.mach_cutoff = bool(n_rays > 0 and ground_hit_count == 0)
            ground_hit_pct = (100.0 * ground_hit_count / n_rays) if n_rays else 0.0
            avg_ray_points = (ray_point_count / n_rays) if n_rays else 0.0
            if not np.isfinite(terminal_alt_min_m):
                terminal_alt_min_m = 0.0
            if not np.isfinite(terminal_alt_max_m):
                terminal_alt_max_m = 0.0
            print(
                "[sim] emission "
                f"{emit_idx + 1}/{total_emissions} "
                f"(count={count_progress_pct:5.1f}%, window={window_progress_pct:5.1f}%, route={flight_progress_pct:5.1f}%) "
                f"time={emit_time.isoformat()} "
                f"elapsed={elapsed_window_s / 60.0:7.2f}m remaining={remaining_window_s / 60.0:7.2f}m | "
                f"aircraft lat={state.lat_deg:+8.4f} lon={state.lon_deg:+9.4f} alt={state.alt_m:7.1f}m "
                f"({state.alt_m * 3.28084:8.1f}ft) speed={state.speed_mps:6.1f}m/s "
                f"({state.speed_mps * 1.943844:6.1f}kt) mach={state.mach:.2f} eff_mach={effective_mach:.3f} | "
                f"atmo T={float(t0[0]):6.2f}K RH={float(rh0[0]):5.1f}% p={float(p0[0]):7.2f}hPa "
                f"c={float(c0[0]):6.2f}m/s w_proj={float(w0[0]):+6.2f}m/s c_eff={ce0_scalar:6.2f}m/s | "
                f"hrrr={snap_time.isoformat()} dt={snapshot_offset_min:+6.2f}m | "
                f"rays={n_rays} ground_hits={ground_hit_count} ({ground_hit_pct:5.1f}%) "
                f"avg_pts={avg_ray_points:6.1f} cutoff={'yes' if emission_result.mach_cutoff else 'no'} "
                f"terminal_alt=[{terminal_alt_min_m:7.1f},{terminal_alt_max_m:7.1f}]m"
            )

        atmospheric_time_series = None
        if sample_times:
            atmospheric_time_series = AtmosphericTimeSeries(
                emission_times_utc=sample_times,
                aircraft_lat_deg=np.asarray(sample_lats, dtype=np.float32),
                aircraft_lon_deg=np.asarray(sample_lons, dtype=np.float32),
                aircraft_alt_m=np.asarray(sample_alts, dtype=np.float32),
                temperature_k=np.asarray(sample_temp, dtype=np.float32),
                relative_humidity_pct=np.asarray(sample_rh, dtype=np.float32),
                pressure_hpa=np.asarray(sample_pressure, dtype=np.float32),
                u_wind_mps=np.asarray(sample_u, dtype=np.float32),
                v_wind_mps=np.asarray(sample_v, dtype=np.float32),
                sound_speed_mps=np.asarray(sample_c, dtype=np.float32),
                wind_projection_mps=np.asarray(sample_w_proj, dtype=np.float32),
                effective_sound_speed_mps=np.asarray(sample_c_eff, dtype=np.float32),
            )

        return SimulationResult(
            emissions=emissions,
            config_dict=self.config.to_dict(),
            terrain_lat_deg=terrain_lat,
            terrain_lon_deg=terrain_lon,
            terrain_elevation_m=terrain_elev,
            atmospheric_time_series=atmospheric_time_series,
            atmospheric_vertical_profile=vertical_profile,
            atmospheric_grid_3d=atmospheric_grid_3d,
        )
