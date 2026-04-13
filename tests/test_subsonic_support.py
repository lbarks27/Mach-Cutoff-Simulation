import unittest
from datetime import datetime, timezone
from unittest.mock import patch

import numpy as np

from mach_cutoff.config import ExperimentConfig
from mach_cutoff.flight.waypoints import FlightPath, Waypoint
from mach_cutoff.guidance_config import GuidanceConfig
from mach_cutoff.simulation.engine import MachCutoffSimulator


class _FakeSnapshot:
    def __init__(self):
        lat = np.asarray([[0.0]], dtype=np.float32)
        lon = np.asarray([[0.0]], dtype=np.float32)
        gh = np.asarray([[[0.0]]], dtype=np.float32)
        self.lat_deg = lat
        self.lon_deg = lon
        self.geopotential_height_m = gh


class _FakeHRRRManager:
    def __init__(self, *_args, **_kwargs):
        self._snapshot = _FakeSnapshot()

    def snapshots_for_times(self, times):
        return {times[0]: self._snapshot}


class _FakeInterpolator:
    def __init__(self, _snapshot):
        pass

    def sample_points(self, *, lat_deg, lon_deg, alt_m):
        shape = lat_deg.shape
        return {
            "temperature_k": np.full(shape, 288.0, dtype=np.float32),
            "relative_humidity_pct": np.full(shape, 50.0, dtype=np.float32),
            "pressure_hpa": np.full(shape, 1013.25, dtype=np.float32),
            "u_wind_mps": np.zeros(shape, dtype=np.float32),
            "v_wind_mps": np.zeros(shape, dtype=np.float32),
        }

    def sample_columns(self, *, lat_deg, lon_deg, altitudes_m):
        npts = lat_deg.shape[0]
        nalt = altitudes_m.shape[0]
        shape = (npts, nalt)
        return {
            "temperature_k": np.full(shape, 288.0, dtype=np.float32),
            "relative_humidity_pct": np.full(shape, 50.0, dtype=np.float32),
            "pressure_hpa": np.full(shape, 1013.25, dtype=np.float32),
            "u_wind_mps": np.zeros(shape, dtype=np.float32),
            "v_wind_mps": np.zeros(shape, dtype=np.float32),
        }


def _fake_compute_effective_sound_speed_mps(t_k, p_hpa, rh_pct, u_wind_mps, v_wind_mps, velocity_enu, mode):
    del p_hpa, rh_pct, u_wind_mps, v_wind_mps, velocity_enu, mode
    c = np.full_like(np.asarray(t_k, dtype=np.float32), 340.0, dtype=np.float32)
    w = np.zeros_like(c, dtype=np.float32)
    # Keep c_eff above the commanded subsonic speed so source cutoff stays active.
    c_eff = np.full_like(c, 360.0, dtype=np.float32)
    return c, w, c_eff


class SubsonicSupportTests(unittest.TestCase):
    def test_subsonic_simulation_skips_shock_generation_and_builds_zero_ray_emission(self):
        start = datetime(2025, 1, 15, 12, 0, tzinfo=timezone.utc)
        end = datetime(2025, 1, 15, 12, 10, tzinfo=timezone.utc)
        path = FlightPath(
            [
                Waypoint(lat_deg=39.8561, lon_deg=-104.6737, alt_m=11_129.6672, time_utc=start),
                Waypoint(lat_deg=40.7899, lon_deg=-111.9791, alt_m=11_129.6672, time_utc=end),
            ]
        )

        cfg = ExperimentConfig()
        cfg.aircraft.mach = 0.95
        cfg.runtime.max_emissions = 1
        cfg.visualization.enable_matplotlib = False
        cfg.visualization.enable_plotly = False
        cfg.visualization.enable_pyvista = False
        cfg.population.enabled = False

        guidance_cfg = GuidanceConfig.from_dict(
            {
                "enabled": False,
            }
        )

        simulator = MachCutoffSimulator(cfg, guidance_config=guidance_cfg)

        with patch("mach_cutoff.simulation.engine.HRRRDatasetManager", _FakeHRRRManager), patch(
            "mach_cutoff.simulation.engine.HRRRInterpolator", _FakeInterpolator
        ), patch(
            "mach_cutoff.simulation.engine.compute_effective_sound_speed_mps",
            _fake_compute_effective_sound_speed_mps,
        ), patch(
            "mach_cutoff.simulation.engine.generate_shock_directions",
            side_effect=AssertionError("subsonic emissions must not generate shocks"),
        ), patch(
            "mach_cutoff.simulation.engine.build_acoustic_grid_field",
            side_effect=AssertionError("subsonic emissions must not build acoustic fields"),
        ):
            result = simulator.run(path)

        self.assertEqual(len(result.emissions), 1)
        emission = result.emissions[0]
        self.assertTrue(emission.source_mach_cutoff)
        self.assertEqual(emission.rays, [])
        self.assertLessEqual(float(emission.effective_mach), 1.0)
        self.assertIsNone(result.atmospheric_grid_3d)


if __name__ == "__main__":
    unittest.main()
