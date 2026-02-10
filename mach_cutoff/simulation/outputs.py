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
    rays: list[RayResult] = field(default_factory=list)


@dataclass(slots=True)
class SimulationResult:
    emissions: list[EmissionResult]
    config_dict: dict
    terrain_lat_deg: np.ndarray | None = None
    terrain_lon_deg: np.ndarray | None = None
    terrain_elevation_m: np.ndarray | None = None

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
            "config": self.config_dict,
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    def save_npz(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        lat_hits, lon_hits, _ = self.all_ground_hits()
        np.savez_compressed(
            path,
            hit_lat_deg=lat_hits,
            hit_lon_deg=lon_hits,
            num_emissions=len(self.emissions),
            num_rays=sum(len(e.rays) for e in self.emissions),
        )
