"""End-to-end simulation engine."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..flight.aircraft import PointMassAircraft, generate_shock_directions
from ..atmosphere.acoustics import build_acoustic_grid_field
from ..atmosphere.hrrr import HRRRDatasetManager
from ..atmosphere.interpolation import HRRRInterpolator
from ..config import ExperimentConfig
from ..core.geodesy import ecef_to_geodetic, ecef_to_enu, enu_to_ecef, normalize_lon_deg
from ..core.raytrace import integrate_ray
from ..flight.waypoints import FlightPath, load_waypoints_json
from .outputs import EmissionResult, RayResult, SimulationResult


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

        start_time = _parse_time_iso(self.config.runtime.start_time_iso) or path.start_time
        end_time = _parse_time_iso(self.config.runtime.end_time_iso) or path.end_time

        emission_times = path.sample_times(self.config.shock.emission_interval_s, start=start_time, end=end_time)
        if self.config.runtime.max_emissions is not None:
            emission_times = emission_times[: self.config.runtime.max_emissions]
        if not emission_times:
            return SimulationResult(emissions=[], config_dict=self.config.to_dict())

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

        for emit_idx, emit_time in enumerate(emission_times):
            state = aircraft.state_at(emit_time)
            dirs_ecef = generate_shock_directions(state, self.config.shock)

            snap_time = _nearest_snapshot_time(snapshot_times, emit_time)
            if snap_time not in interp_cache:
                interp_cache[snap_time] = HRRRInterpolator(snapshots_by_time[snap_time])
            interp = interp_cache[snap_time]

            field, _ = build_acoustic_grid_field(
                interp,
                aircraft_lat_deg=state.lat_deg,
                aircraft_lon_deg=state.lon_deg,
                aircraft_alt_m=state.alt_m,
                aircraft_velocity_ecef_mps=state.velocity_ecef_mps,
                grid_config=self.config.grid,
                reference_speed_mps=self.config.aircraft.reference_sound_speed_mps,
            )

            origin_ecef = np.asarray(state.position_ecef_m, dtype=float)

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
                rays=[],
            )

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

            if emit_idx % 5 == 0:
                print(f"[sim] emission {emit_idx + 1}/{len(emission_times)} at {emit_time.isoformat()}")

        return SimulationResult(
            emissions=emissions,
            config_dict=self.config.to_dict(),
            terrain_lat_deg=terrain_lat,
            terrain_lon_deg=terrain_lon,
            terrain_elevation_m=terrain_elev,
        )
