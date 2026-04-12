"""End-to-end simulation engine."""

from __future__ import annotations

from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from ..flight.aircraft import PointMassAircraft, generate_shock_directions
from ..flight.guidance import GuidanceFeedback, GuidedPointMassAircraft
from ..atmosphere.acoustics import build_acoustic_grid_field, compute_effective_sound_speed_mps
from ..atmosphere.hrrr import HRRRDatasetManager
from ..atmosphere.interpolation import HRRRInterpolator
from ..config import ExperimentConfig
from ..core.geodesy import ecef_to_geodetic, ecef_to_enu, enu_basis, enu_to_ecef, normalize_lon_deg
from ..core.raytrace import integrate_ray
from ..flight.waypoints import FlightPath, load_waypoints_json
from ..guidance_config import GuidanceConfig
from .outputs import (
    AtmosphericGrid3D,
    AtmosphericTimeSeries,
    AtmosphericVerticalProfile,
    EmissionResult,
    GuidanceTelemetry,
    RayResult,
    SimulationResult,
)
from .population import analyze_population_impact


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
    def __init__(self, config: ExperimentConfig, guidance_config: GuidanceConfig | None = None):
        self.config = config
        self.guidance_config = guidance_config or GuidanceConfig()

    def run_from_waypoint_file(self, waypoint_json_path: str | Path) -> SimulationResult:
        waypoints = load_waypoints_json(waypoint_json_path)
        path = FlightPath(waypoints)
        return self.run(path)

    def run(self, path: FlightPath) -> SimulationResult:
        if self.guidance_config.enabled:
            aircraft = GuidedPointMassAircraft(path, self.config.aircraft, self.guidance_config)
            guidance_feedback = GuidanceFeedback()
            wind_for_propagation_enu = np.zeros(3, dtype=float)
        else:
            aircraft = PointMassAircraft(path, self.config.aircraft)
            guidance_feedback = None
            wind_for_propagation_enu = None

        start_time = _parse_time_iso(self.config.runtime.start_time_iso) or aircraft.start_time
        emission_interval_s = float(self.config.shock.emission_interval_s)
        if emission_interval_s <= 0.0:
            raise ValueError("shock.emission_interval_s must be positive")

        max_emissions: int | None = None
        if self.config.runtime.max_emissions is not None:
            max_emissions = int(self.config.runtime.max_emissions)
            if max_emissions <= 0:
                return SimulationResult(emissions=[], config_dict=self.config.to_dict())

        first_emit_epoch = start_time.timestamp()
        if not np.isfinite(first_emit_epoch):
            return SimulationResult(emissions=[], config_dict=self.config.to_dict())

        bbox = _waypoint_bbox(path)
        hrrr_manager = HRRRDatasetManager(self.config.hrrr, bbox=bbox)
        snapshots_by_time = hrrr_manager.snapshots_for_times([start_time])
        if not snapshots_by_time:
            raise RuntimeError("No HRRR snapshots were loaded for the requested emission times")
        snapshot_times = sorted(snapshots_by_time.keys())

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

        ray_workers = int(self.config.runtime.ray_batch_size)
        ray_pool_context = (
            ThreadPoolExecutor(max_workers=ray_workers) if ray_workers > 1 else nullcontext()
        )

        emit_idx = 0
        emit_time = start_time
        route_length_m = max(float(path.total_length_m), 1e-6)
        best_remaining_m = float(path.total_length_m)
        last_distance_to_destination_m = best_remaining_m
        stalled_emissions = 0
        stall_limit_emissions = 2_400  # Prevent unbounded loops if guidance never converges.
        termination_reason = "destination_reached"

        with ray_pool_context as ray_pool:
            while True:
                if max_emissions is not None and emit_idx >= max_emissions:
                    termination_reason = "max_emissions_reached"
                    break
                emit_idx += 1

                if self.guidance_config.enabled:
                    aircraft.ingest_feedback(guidance_feedback or GuidanceFeedback())
                    state, guidance_command = aircraft.state_at(emit_time, wind_enu_mps=wind_for_propagation_enu)
                else:
                    state = aircraft.state_at(emit_time)
                    guidance_command = None

                projected = path.project_ecef(np.asarray(state.position_ecef_m, dtype=float))
                projected_remaining_m = max(0.0, float(path.total_length_m) - float(projected["distance_m"]))
                if guidance_command is not None:
                    distance_to_destination_m = max(0.0, float(guidance_command.distance_to_destination_m))
                else:
                    distance_to_destination_m = projected_remaining_m
                last_distance_to_destination_m = float(distance_to_destination_m)
                route_progress_pct = 100.0 * np.clip(1.0 - projected_remaining_m / route_length_m, 0.0, 1.0)

                dirs_ecef = generate_shock_directions(state, self.config.shock)
                emit_epoch = emit_time.timestamp()
                elapsed_window_s = max(0.0, emit_epoch - first_emit_epoch)
                if max_emissions is not None and max_emissions > 0:
                    count_progress_pct = 100.0 * float(emit_idx) / float(max_emissions)
                    remaining_window_s = max(0.0, float(max_emissions - emit_idx) * emission_interval_s)
                else:
                    count_progress_pct = 100.0
                    remaining_window_s = distance_to_destination_m / max(float(state.speed_mps), 1.0)

                new_snapshots = hrrr_manager.snapshots_for_times([emit_time])
                for snap_t, snap in new_snapshots.items():
                    snapshots_by_time[snap_t] = snap
                snapshot_times = sorted(snapshots_by_time.keys())
                snap_time = _nearest_snapshot_time(snapshot_times, emit_time)
                if snap_time not in interp_cache:
                    interp_cache[snap_time] = HRRRInterpolator(snapshots_by_time[snap_time])
                interp = interp_cache[snap_time]

                origin_ecef = np.asarray(state.position_ecef_m, dtype=float)
                east_basis, north_basis, up_basis = enu_basis(state.lat_deg, state.lon_deg)
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
                source_mach_cutoff = bool(effective_mach <= 1.0 or float(state.mach) <= 1.0)

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

                emission_result = EmissionResult(
                    emission_time_utc=emit_time,
                    aircraft_lat_deg=state.lat_deg,
                    aircraft_lon_deg=state.lon_deg,
                    aircraft_alt_m=state.alt_m,
                    aircraft_position_ecef_m=origin_ecef,
                    effective_mach=effective_mach,
                    source_mach_cutoff=source_mach_cutoff,
                    mach_cutoff=source_mach_cutoff,
                    guidance=(
                        GuidanceTelemetry(
                            mode=guidance_command.mode,
                            along_track_distance_m=guidance_command.along_track_distance_m,
                            cross_track_error_m=guidance_command.cross_track_error_m,
                            distance_to_destination_m=guidance_command.distance_to_destination_m,
                            desired_heading_deg=guidance_command.desired_heading_deg,
                            desired_bank_deg=guidance_command.desired_bank_deg,
                            desired_pitch_deg=guidance_command.desired_pitch_deg,
                            desired_load_factor=guidance_command.desired_load_factor,
                            desired_speed_mps=guidance_command.desired_speed_mps,
                            desired_mach=guidance_command.desired_mach,
                            desired_body_accel_mps2=np.asarray(guidance_command.desired_body_accel_mps2, dtype=float),
                            optimizer_cost=guidance_command.optimizer_cost,
                            predicted_ground_hit_fraction=guidance_command.predicted_ground_hit_fraction,
                            predicted_source_cutoff_risk=guidance_command.predicted_source_cutoff_risk,
                            predicted_effective_mach=guidance_command.predicted_effective_mach,
                            optimizer_altitude_adjustment_m=guidance_command.optimizer_altitude_adjustment_m,
                            optimizer_mach_adjustment=guidance_command.optimizer_mach_adjustment,
                        )
                        if guidance_command is not None
                        else None
                    ),
                    rays=[],
                )
                snapshot_offset_min = (emit_time - snap_time).total_seconds() / 60.0
                if source_mach_cutoff:
                    emissions.append(emission_result)
                    if self.guidance_config.enabled:
                        guidance_feedback = GuidanceFeedback(
                            effective_mach=effective_mach,
                            commanded_mach=float(state.mach),
                            source_mach_cutoff=True,
                            ground_hit_fraction=0.0,
                        )
                        wind_for_propagation_enu = np.array([float(u0[0]), float(v0[0]), 0.0], dtype=float)
                    print(
                        "[sim] emission "
                        f"{emit_idx}" + (f"/{max_emissions}" if max_emissions is not None else "") + " "
                        f"(count={count_progress_pct:5.1f}%, route={route_progress_pct:5.1f}%) "
                        f"time={emit_time.isoformat()} "
                        f"elapsed={elapsed_window_s / 60.0:7.2f}m eta={remaining_window_s / 60.0:7.2f}m | "
                        f"aircraft lat={state.lat_deg:+8.4f} lon={state.lon_deg:+9.4f} alt={state.alt_m:7.1f}m "
                        f"({state.alt_m * 3.28084:8.1f}ft) speed={state.speed_mps:6.1f}m/s "
                        f"({state.speed_mps * 1.943844:6.1f}kt) mach={state.mach:.2f} "
                        f"eff_mach={effective_mach:.3f} | "
                        f"mode={guidance_command.mode if guidance_command is not None else 'legacy'} | "
                        f"atmo T={float(t0[0]):6.2f}K RH={float(rh0[0]):5.1f}% p={float(p0[0]):7.2f}hPa "
                        f"c={float(c0[0]):6.2f}m/s w_proj={float(w0[0]):+6.2f}m/s c_eff={ce0_scalar:6.2f}m/s | "
                        f"hrrr={snap_time.isoformat()} dt={snapshot_offset_min:+6.2f}m | "
                        "SOURCE CUT OFF (effective mach <= 1.0 or aircraft mach <= 1.0), rays=0"
                    )
                    if distance_to_destination_m + 1.0 < best_remaining_m:
                        best_remaining_m = distance_to_destination_m
                        stalled_emissions = 0
                    else:
                        stalled_emissions += 1
                    arrival_tolerance_m = max(250.0, float(state.speed_mps) * emission_interval_s)
                    if distance_to_destination_m <= arrival_tolerance_m:
                        termination_reason = "destination_reached"
                        break
                    if max_emissions is None and stalled_emissions >= stall_limit_emissions:
                        termination_reason = "destination_not_converging"
                        break
                    emit_time = emit_time + timedelta(seconds=emission_interval_s)
                    continue

                ground_hit_count = 0
                ray_point_count = 0
                terminal_alt_min_m = np.inf
                terminal_alt_max_m = -np.inf

                if field is None:
                    raise RuntimeError("Acoustic field was not built for supersonic emission")

                def _trace_one_ray(ray_spec: tuple[int, np.ndarray]):
                    ray_idx, d_ecef = ray_spec
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
                        surfaces=None,
                        stop_on_exit=self.config.raytrace.stop_on_exit,
                        ground_origin_ecef=origin_ecef,
                        ground_east=east_basis,
                        ground_north=north_basis,
                        ground_up=up_basis,
                    )

                    traj_ecef = enu_to_ecef(state.lat_deg, state.lon_deg, state.alt_m, traj_local)
                    lat, lon, alt = ecef_to_geodetic(traj_ecef[:, 0], traj_ecef[:, 1], traj_ecef[:, 2])
                    traj_geo = np.column_stack([lat, lon, alt])

                    ground_hit = bool(traj_geo[-1, 2] <= 0.0)
                    hit_lat_lon = (float(traj_geo[-1, 0]), float(traj_geo[-1, 1])) if ground_hit else None
                    terminal_alt_m = float(traj_geo[-1, 2])
                    return (
                        RayResult(
                            ray_id=ray_idx,
                            trajectory_local_m=traj_local,
                            trajectory_ecef_m=traj_ecef,
                            trajectory_geodetic=traj_geo,
                            ground_hit=ground_hit,
                            ground_hit_lat_lon=hit_lat_lon,
                        ),
                        ground_hit,
                        int(traj_local.shape[0]),
                        terminal_alt_m,
                    )

                ray_specs = list(enumerate(dirs_ecef))
                if ray_pool is not None and len(ray_specs) > 1:
                    ray_results = list(ray_pool.map(_trace_one_ray, ray_specs))
                else:
                    ray_results = [_trace_one_ray(spec) for spec in ray_specs]

                for ray_result, ground_hit, point_count, terminal_alt_m in ray_results:
                    if ground_hit:
                        ground_hit_count += 1
                    ray_point_count += point_count
                    terminal_alt_min_m = min(terminal_alt_min_m, terminal_alt_m)
                    terminal_alt_max_m = max(terminal_alt_max_m, terminal_alt_m)
                    emission_result.rays.append(ray_result)

                emissions.append(emission_result)
                n_rays = len(emission_result.rays)
                emission_result.mach_cutoff = bool(n_rays > 0 and ground_hit_count == 0)
                ground_hit_pct = (100.0 * ground_hit_count / n_rays) if n_rays else 0.0
                ground_hit_fraction = (ground_hit_count / n_rays) if n_rays else 0.0
                avg_ray_points = (ray_point_count / n_rays) if n_rays else 0.0
                if not np.isfinite(terminal_alt_min_m):
                    terminal_alt_min_m = 0.0
                if not np.isfinite(terminal_alt_max_m):
                    terminal_alt_max_m = 0.0
                if self.guidance_config.enabled:
                    guidance_feedback = GuidanceFeedback(
                        effective_mach=effective_mach,
                        commanded_mach=float(state.mach),
                        source_mach_cutoff=False,
                        ground_hit_fraction=ground_hit_fraction,
                    )
                    wind_for_propagation_enu = np.array([float(u0[0]), float(v0[0]), 0.0], dtype=float)
                print(
                    "[sim] emission "
                    f"{emit_idx}" + (f"/{max_emissions}" if max_emissions is not None else "") + " "
                    f"(count={count_progress_pct:5.1f}%, route={route_progress_pct:5.1f}%) "
                    f"time={emit_time.isoformat()} "
                    f"elapsed={elapsed_window_s / 60.0:7.2f}m eta={remaining_window_s / 60.0:7.2f}m | "
                    f"aircraft lat={state.lat_deg:+8.4f} lon={state.lon_deg:+9.4f} alt={state.alt_m:7.1f}m "
                    f"({state.alt_m * 3.28084:8.1f}ft) speed={state.speed_mps:6.1f}m/s "
                    f"({state.speed_mps * 1.943844:6.1f}kt) mach={state.mach:.2f} eff_mach={effective_mach:.3f} | "
                    f"mode={guidance_command.mode if guidance_command is not None else 'legacy'} | "
                    f"atmo T={float(t0[0]):6.2f}K RH={float(rh0[0]):5.1f}% p={float(p0[0]):7.2f}hPa "
                    f"c={float(c0[0]):6.2f}m/s w_proj={float(w0[0]):+6.2f}m/s c_eff={ce0_scalar:6.2f}m/s | "
                    f"hrrr={snap_time.isoformat()} dt={snapshot_offset_min:+6.2f}m | "
                    f"rays={n_rays} ground_hits={ground_hit_count} ({ground_hit_pct:5.1f}%) "
                    f"avg_pts={avg_ray_points:6.1f} cutoff={'yes' if emission_result.mach_cutoff else 'no'} "
                    f"terminal_alt=[{terminal_alt_min_m:7.1f},{terminal_alt_max_m:7.1f}]m"
                )
                if distance_to_destination_m + 1.0 < best_remaining_m:
                    best_remaining_m = distance_to_destination_m
                    stalled_emissions = 0
                else:
                    stalled_emissions += 1
                arrival_tolerance_m = max(250.0, float(state.speed_mps) * emission_interval_s)
                if distance_to_destination_m <= arrival_tolerance_m:
                    termination_reason = "destination_reached"
                    break
                if max_emissions is None and stalled_emissions >= stall_limit_emissions:
                    termination_reason = "destination_not_converging"
                    break
                emit_time = emit_time + timedelta(seconds=emission_interval_s)

        print(
            "[sim] stop "
            f"reason={termination_reason} "
            f"emissions={len(emissions)} "
            f"distance_to_destination={last_distance_to_destination_m:,.1f} m"
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

        population_impact = None
        if self.config.population.enabled:
            try:
                population_impact = analyze_population_impact(emissions, self.config.population)
                if population_impact is None:
                    print("[warn] population analysis enabled, but no usable population data was found")
                else:
                    print(
                        "[sim] population "
                        f"dataset={population_impact.dataset_name!r} "
                        f"total={population_impact.total_population_in_heatmap:,.0f} "
                        f"exposed={population_impact.total_exposed_population:,.0f} "
                        f"area={population_impact.total_exposed_area_km2:,.1f} km^2 "
                        f"overflight={population_impact.total_overflight_population:,.0f}"
                    )
            except Exception as exc:
                print(f"[warn] population analysis failed: {exc}")

        return SimulationResult(
            emissions=emissions,
            config_dict=self.config.to_dict(),
            terrain_lat_deg=terrain_lat,
            terrain_lon_deg=terrain_lon,
            terrain_elevation_m=terrain_elev,
            atmospheric_time_series=atmospheric_time_series,
            atmospheric_vertical_profile=vertical_profile,
            atmospheric_grid_3d=atmospheric_grid_3d,
            population_impact=population_impact,
        )
