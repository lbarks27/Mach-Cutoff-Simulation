"""Simulation output containers and serialization helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from html import escape
from pathlib import Path

import numpy as np


def _kml_color(hex_rgb: str, *, alpha: int = 255) -> str:
    rgb = hex_rgb.lstrip("#")
    if len(rgb) != 6:
        raise ValueError(f"Expected 6-digit RGB hex color, got: {hex_rgb!r}")
    r = int(rgb[0:2], 16)
    g = int(rgb[2:4], 16)
    b = int(rgb[4:6], 16)
    a = int(np.clip(alpha, 0, 255))
    return f"{a:02x}{b:02x}{g:02x}{r:02x}"


def _kml_linestring_coordinates(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    alt_m: np.ndarray,
    *,
    max_points: int,
) -> str:
    lat = np.asarray(lat_deg, dtype=float).reshape(-1)
    lon = np.asarray(lon_deg, dtype=float).reshape(-1)
    alt = np.asarray(alt_m, dtype=float).reshape(-1)
    if lat.size == 0 or lon.size == 0 or alt.size == 0:
        return ""
    n = min(lat.size, lon.size, alt.size)
    lat = lat[:n]
    lon = lon[:n]
    alt = alt[:n]
    finite_mask = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(alt)
    if not np.any(finite_mask):
        return ""
    lat = lat[finite_mask]
    lon = lon[finite_mask]
    alt = alt[finite_mask]
    if lat.size > max_points:
        idx = np.linspace(0, lat.size - 1, num=max_points, dtype=int)
        lat = lat[idx]
        lon = lon[idx]
        alt = alt[idx]
    return " ".join(f"{lo:.6f},{la:.6f},{al:.1f}" for la, lo, al in zip(lat, lon, alt, strict=False))


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
    guidance: GuidanceTelemetry | None = None
    rays: list[RayResult] = field(default_factory=list)


@dataclass(slots=True)
class GuidanceTelemetry:
    mode: str
    along_track_distance_m: float
    cross_track_error_m: float
    distance_to_destination_m: float
    desired_heading_deg: float
    desired_bank_deg: float
    desired_pitch_deg: float
    desired_load_factor: float
    desired_speed_mps: float
    desired_mach: float
    desired_body_accel_mps2: np.ndarray
    optimizer_cost: float
    predicted_ground_hit_fraction: float
    predicted_source_cutoff_risk: float
    predicted_effective_mach: float
    optimizer_altitude_adjustment_m: float
    optimizer_mach_adjustment: float


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

        guidance_samples = [e.guidance for e in self.emissions if e.guidance is not None]
        if guidance_samples:
            cross_track_errors = np.asarray([g.cross_track_error_m for g in guidance_samples], dtype=float)
            dist_to_dest = np.asarray([g.distance_to_destination_m for g in guidance_samples], dtype=float)
            optimizer_costs = np.asarray([g.optimizer_cost for g in guidance_samples], dtype=float)
            predicted_ground = np.asarray([g.predicted_ground_hit_fraction for g in guidance_samples], dtype=float)
            predicted_cutoff = np.asarray([g.predicted_source_cutoff_risk for g in guidance_samples], dtype=float)
            predicted_eff_mach = np.asarray([g.predicted_effective_mach for g in guidance_samples], dtype=float)
            mode_counts: dict[str, int] = {}
            for g in guidance_samples:
                mode_counts[g.mode] = int(mode_counts.get(g.mode, 0) + 1)
            summary["num_guidance_samples"] = int(len(guidance_samples))
            summary["guidance_mode_counts"] = mode_counts
            summary["guidance_cross_track_abs_mean_m"] = float(np.mean(np.abs(cross_track_errors)))
            summary["guidance_cross_track_abs_max_m"] = float(np.max(np.abs(cross_track_errors)))
            summary["guidance_distance_to_destination_final_m"] = float(dist_to_dest[-1])
            summary["guidance_optimizer_cost_mean"] = float(np.mean(optimizer_costs))
            summary["guidance_predicted_ground_hit_mean"] = float(np.mean(predicted_ground))
            summary["guidance_predicted_cutoff_risk_mean"] = float(np.mean(predicted_cutoff))
            summary["guidance_predicted_effective_mach_mean"] = float(np.mean(predicted_eff_mach))

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
        guidance_samples = [e.guidance for e in self.emissions if e.guidance is not None]
        if guidance_samples:
            payload.update(
                guidance_mode=np.asarray([g.mode for g in guidance_samples], dtype="<U32"),
                guidance_along_track_distance_m=np.asarray(
                    [g.along_track_distance_m for g in guidance_samples], dtype=np.float32
                ),
                guidance_cross_track_error_m=np.asarray(
                    [g.cross_track_error_m for g in guidance_samples], dtype=np.float32
                ),
                guidance_distance_to_destination_m=np.asarray(
                    [g.distance_to_destination_m for g in guidance_samples], dtype=np.float32
                ),
                guidance_desired_heading_deg=np.asarray(
                    [g.desired_heading_deg for g in guidance_samples], dtype=np.float32
                ),
                guidance_desired_bank_deg=np.asarray(
                    [g.desired_bank_deg for g in guidance_samples], dtype=np.float32
                ),
                guidance_desired_pitch_deg=np.asarray(
                    [g.desired_pitch_deg for g in guidance_samples], dtype=np.float32
                ),
                guidance_desired_load_factor=np.asarray(
                    [g.desired_load_factor for g in guidance_samples], dtype=np.float32
                ),
                guidance_desired_speed_mps=np.asarray(
                    [g.desired_speed_mps for g in guidance_samples], dtype=np.float32
                ),
                guidance_desired_mach=np.asarray([g.desired_mach for g in guidance_samples], dtype=np.float32),
                guidance_desired_body_accel_mps2=np.asarray(
                    [g.desired_body_accel_mps2 for g in guidance_samples], dtype=np.float32
                ),
                guidance_optimizer_cost=np.asarray([g.optimizer_cost for g in guidance_samples], dtype=np.float32),
                guidance_predicted_ground_hit_fraction=np.asarray(
                    [g.predicted_ground_hit_fraction for g in guidance_samples], dtype=np.float32
                ),
                guidance_predicted_source_cutoff_risk=np.asarray(
                    [g.predicted_source_cutoff_risk for g in guidance_samples], dtype=np.float32
                ),
                guidance_predicted_effective_mach=np.asarray(
                    [g.predicted_effective_mach for g in guidance_samples], dtype=np.float32
                ),
                guidance_optimizer_altitude_adjustment_m=np.asarray(
                    [g.optimizer_altitude_adjustment_m for g in guidance_samples], dtype=np.float32
                ),
                guidance_optimizer_mach_adjustment=np.asarray(
                    [g.optimizer_mach_adjustment for g in guidance_samples], dtype=np.float32
                ),
            )
        np.savez_compressed(path, **payload)

    def save_kml(
        self,
        path: str | Path,
        *,
        max_rays_per_emission: int = 14,
        max_points_per_ray: int = 220,
        max_ground_hits: int = 2_000,
    ):
        """Write Google Earth-compatible KML overlay for the current simulation."""
        if max_rays_per_emission <= 0:
            raise ValueError("max_rays_per_emission must be > 0")
        if max_points_per_ray <= 1:
            raise ValueError("max_points_per_ray must be > 1")
        if max_ground_hits <= 0:
            raise ValueError("max_ground_hits must be > 0")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        lines: list[str] = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<kml xmlns="http://www.opengis.net/kml/2.2">',
            "  <Document>",
            "    <name>Mach Cutoff Simulation Overlay</name>",
            "    <Style id=\"flightStyle\">",
            "      <LineStyle>",
            f"        <color>{_kml_color('#111111', alpha=255)}</color>",
            "        <width>3</width>",
            "      </LineStyle>",
            "    </Style>",
            "    <Style id=\"rayStyle\">",
            "      <LineStyle>",
            f"        <color>{_kml_color('#1f5dff', alpha=110)}</color>",
            "        <width>1</width>",
            "      </LineStyle>",
            "    </Style>",
            "    <Style id=\"groundHitStyle\">",
            "      <IconStyle>",
            f"        <color>{_kml_color('#d62828', alpha=230)}</color>",
            "        <scale>0.65</scale>",
            "      </IconStyle>",
            "      <LabelStyle>",
            "        <scale>0</scale>",
            "      </LabelStyle>",
            "    </Style>",
        ]

        if self.emissions:
            flight_lat = np.asarray([e.aircraft_lat_deg for e in self.emissions], dtype=float)
            flight_lon = np.asarray([e.aircraft_lon_deg for e in self.emissions], dtype=float)
            flight_alt = np.asarray([e.aircraft_alt_m for e in self.emissions], dtype=float)
            flight_coords = _kml_linestring_coordinates(
                flight_lat,
                flight_lon,
                flight_alt,
                max_points=max(2, int(flight_lat.size)),
            )
            if flight_coords:
                lines.extend(
                    [
                        "    <Folder>",
                        "      <name>Flight</name>",
                        "      <Placemark>",
                        "        <name>Aircraft track</name>",
                        "        <styleUrl>#flightStyle</styleUrl>",
                        "        <LineString>",
                        "          <tessellate>1</tessellate>",
                        "          <altitudeMode>absolute</altitudeMode>",
                        f"          <coordinates>{flight_coords}</coordinates>",
                        "        </LineString>",
                        "      </Placemark>",
                        "    </Folder>",
                    ]
                )

        ray_count = 0
        lines.extend(["    <Folder>", "      <name>Shock Rays (sampled)</name>"])
        for emission_idx, emission in enumerate(self.emissions):
            for ray in emission.rays[:max_rays_per_emission]:
                g = np.asarray(ray.trajectory_geodetic, dtype=float)
                if g.ndim != 2 or g.shape[0] == 0 or g.shape[1] < 3:
                    continue
                coords = _kml_linestring_coordinates(
                    g[:, 0],
                    g[:, 1],
                    g[:, 2],
                    max_points=max_points_per_ray,
                )
                if not coords:
                    continue
                ray_count += 1
                ray_name = f"Emission {emission_idx:03d} / Ray {int(ray.ray_id):03d}"
                description = f"ground_hit={bool(ray.ground_hit)}"
                lines.extend(
                    [
                        "      <Placemark>",
                        f"        <name>{escape(ray_name)}</name>",
                        f"        <description>{escape(description)}</description>",
                        "        <styleUrl>#rayStyle</styleUrl>",
                        "        <LineString>",
                        "          <tessellate>1</tessellate>",
                        "          <altitudeMode>absolute</altitudeMode>",
                        f"          <coordinates>{coords}</coordinates>",
                        "        </LineString>",
                        "      </Placemark>",
                    ]
                )
        lines.extend(["    </Folder>"])

        lat_hits, lon_hits, hit_times = self.all_ground_hits()
        if lat_hits.size:
            n_hits = min(int(lat_hits.size), int(max_ground_hits))
            lines.extend(["    <Folder>", "      <name>Ground Hits</name>"])
            for i in range(n_hits):
                t = hit_times[i] if i < len(hit_times) else ""
                lines.extend(
                    [
                        "      <Placemark>",
                        f"        <name>Hit {i + 1:04d}</name>",
                        f"        <description>{escape(t)}</description>",
                        "        <styleUrl>#groundHitStyle</styleUrl>",
                        "        <Point>",
                        "          <altitudeMode>clampToGround</altitudeMode>",
                        f"          <coordinates>{float(lon_hits[i]):.6f},{float(lat_hits[i]):.6f},0</coordinates>",
                        "        </Point>",
                        "      </Placemark>",
                    ]
                )
            lines.extend(["    </Folder>"])

        lines.extend(
            [
                "    <description>",
                escape(
                    f"emissions={len(self.emissions)}, sampled_rays={ray_count}, "
                    f"ground_hits={int(lat_hits.size)}"
                ),
                "    </description>",
                "  </Document>",
                "</kml>",
                "",
            ]
        )

        path.write_text("\n".join(lines), encoding="utf-8")
