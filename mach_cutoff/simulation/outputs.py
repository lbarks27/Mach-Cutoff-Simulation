"""Simulation output containers and serialization helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class RayResult:
    ray_id: int
    trajectory_local_m: np.ndarray
    trajectory_ecef_m: np.ndarray
    trajectory_geodetic: np.ndarray
    ground_hit: bool
    ground_hit_lat_lon: tuple[float, float] | None


@dataclass(slots=True)
class EmissionResult:
    emission_time_utc: datetime
    aircraft_lat_deg: float
    aircraft_lon_deg: float
    aircraft_alt_m: float
    aircraft_position_ecef_m: np.ndarray
    effective_mach: float | None = None
    source_mach_cutoff: bool = False
    mach_cutoff: bool = False
    rays: list[RayResult] = field(default_factory=list)


@dataclass(slots=True)
class AtmosphericTimeSeries:
    emission_times_utc: list[datetime]
    aircraft_lat_deg: np.ndarray
    aircraft_lon_deg: np.ndarray
    aircraft_alt_m: np.ndarray
    temperature_k: np.ndarray
    relative_humidity_pct: np.ndarray
    pressure_hpa: np.ndarray
    u_wind_mps: np.ndarray
    v_wind_mps: np.ndarray
    sound_speed_mps: np.ndarray
    wind_projection_mps: np.ndarray
    effective_sound_speed_mps: np.ndarray


@dataclass(slots=True)
class AtmosphericVerticalProfile:
    emission_time_utc: datetime
    aircraft_lat_deg: float
    aircraft_lon_deg: float
    aircraft_alt_m: float
    altitude_m: np.ndarray
    temperature_k: np.ndarray
    relative_humidity_pct: np.ndarray
    pressure_hpa: np.ndarray
    u_wind_mps: np.ndarray
    v_wind_mps: np.ndarray
    sound_speed_mps: np.ndarray
    wind_projection_mps: np.ndarray
    effective_sound_speed_mps: np.ndarray


@dataclass(slots=True)
class AtmosphericGrid3D:
    emission_time_utc: datetime
    aircraft_lat_deg: float
    aircraft_lon_deg: float
    aircraft_alt_m: float
    lat_grid_deg: np.ndarray
    lon_grid_deg: np.ndarray
    altitude_m: np.ndarray
    temperature_k: np.ndarray
    relative_humidity_pct: np.ndarray
    pressure_hpa: np.ndarray
    u_wind_mps: np.ndarray
    v_wind_mps: np.ndarray
    sound_speed_mps: np.ndarray
    wind_projection_mps: np.ndarray
    effective_sound_speed_mps: np.ndarray


@dataclass(slots=True)
class SimulationResult:
    emissions: list[EmissionResult]
    config_dict: dict
    terrain_lat_deg: np.ndarray | None = None
    terrain_lon_deg: np.ndarray | None = None
    terrain_elevation_m: np.ndarray | None = None
    atmospheric_time_series: AtmosphericTimeSeries | None = None
    atmospheric_vertical_profile: AtmosphericVerticalProfile | None = None
    atmospheric_grid_3d: AtmosphericGrid3D | None = None

    def all_ground_hits(self):
        lats = []
        lons = []
        times = []
        for emission in self.emissions:
            for ray in emission.rays:
                if ray.ground_hit and ray.ground_hit_lat_lon is not None:
                    lat, lon = ray.ground_hit_lat_lon
                    lats.append(lat)
                    lons.append(lon)
                    times.append(emission.emission_time_utc.isoformat())
        return np.asarray(lats), np.asarray(lons), times

    def save_json_summary(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        summary = {
            "num_emissions": len(self.emissions),
            "num_rays": int(sum(len(e.rays) for e in self.emissions)),
            "num_ground_hits": int(sum(1 for e in self.emissions for r in e.rays if r.ground_hit)),
            "num_cutoff_emissions": int(sum(1 for e in self.emissions if e.mach_cutoff)),
            "num_source_cutoff_emissions": int(sum(1 for e in self.emissions if e.source_mach_cutoff)),
            "config": self.config_dict,
        }
        summary["cutoff_achieved"] = bool(summary["num_cutoff_emissions"] > 0)
        summary["full_route_cutoff"] = bool(summary["num_cutoff_emissions"] == summary["num_emissions"])
        summary["source_cutoff_achieved"] = bool(summary["num_source_cutoff_emissions"] > 0)

        effective_machs = [float(e.effective_mach) for e in self.emissions if e.effective_mach is not None]
        if effective_machs:
            summary["effective_mach_min"] = float(np.min(effective_machs))
            summary["effective_mach_max"] = float(np.max(effective_machs))
            summary["effective_mach_mean"] = float(np.mean(effective_machs))

        if summary["num_rays"] > 0:
            summary["ground_hit_fraction"] = float(summary["num_ground_hits"] / summary["num_rays"])

        if self.atmospheric_time_series is not None:
            summary["num_atmospheric_samples"] = int(len(self.atmospheric_time_series.emission_times_utc))
        if self.atmospheric_vertical_profile is not None:
            summary["num_atmospheric_profile_levels"] = int(self.atmospheric_vertical_profile.altitude_m.size)
        if self.atmospheric_grid_3d is not None:
            summary["atmospheric_grid_shape"] = [int(v) for v in self.atmospheric_grid_3d.temperature_k.shape]
        with path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    def save_npz(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        lat_hits, lon_hits, _ = self.all_ground_hits()
        payload = dict(
            hit_lat_deg=lat_hits,
            hit_lon_deg=lon_hits,
            num_emissions=len(self.emissions),
            num_rays=sum(len(e.rays) for e in self.emissions),
        )
        if self.atmospheric_time_series is not None:
            payload.update(
                atmospheric_temperature_k=self.atmospheric_time_series.temperature_k,
                atmospheric_relative_humidity_pct=self.atmospheric_time_series.relative_humidity_pct,
                atmospheric_pressure_hpa=self.atmospheric_time_series.pressure_hpa,
                atmospheric_u_wind_mps=self.atmospheric_time_series.u_wind_mps,
                atmospheric_v_wind_mps=self.atmospheric_time_series.v_wind_mps,
                atmospheric_effective_sound_speed_mps=self.atmospheric_time_series.effective_sound_speed_mps,
            )
        if self.atmospheric_vertical_profile is not None:
            payload.update(
                atmospheric_profile_altitude_m=self.atmospheric_vertical_profile.altitude_m,
                atmospheric_profile_temperature_k=self.atmospheric_vertical_profile.temperature_k,
                atmospheric_profile_relative_humidity_pct=self.atmospheric_vertical_profile.relative_humidity_pct,
                atmospheric_profile_pressure_hpa=self.atmospheric_vertical_profile.pressure_hpa,
                atmospheric_profile_u_wind_mps=self.atmospheric_vertical_profile.u_wind_mps,
                atmospheric_profile_v_wind_mps=self.atmospheric_vertical_profile.v_wind_mps,
                atmospheric_profile_effective_sound_speed_mps=self.atmospheric_vertical_profile.effective_sound_speed_mps,
            )
        np.savez_compressed(path, **payload)
