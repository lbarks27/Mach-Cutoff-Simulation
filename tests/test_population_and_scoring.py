import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from mach_cutoff.config import PopulationConfig
from mach_cutoff.simulation.outputs import EmissionResult, RayResult, SimulationResult
from mach_cutoff.simulation.population import analyze_population_impact
from mach_cutoff.simulation.route_optimizer import (
    ObjectiveMetrics,
    RouteOptimizationSettings,
    _normalizer_from_baseline,
    _score_metrics,
)


class PopulationAndScoringTests(unittest.TestCase):
    def test_gridded_population_npz_and_overflight(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = Path(tmpdir) / "grid.npz"
            np.savez_compressed(
                dataset,
                lat_edges_deg=np.asarray([44.0, 44.5, 45.0], dtype=np.float32),
                lon_edges_deg=np.asarray([-94.0, -93.5, -93.0], dtype=np.float32),
                population_grid=np.asarray([[100.0, 0.0], [0.0, 200.0]], dtype=np.float32),
            )

            emissions = [
                EmissionResult(
                    emission_time_utc=datetime.now(timezone.utc),
                    aircraft_lat_deg=44.25,
                    aircraft_lon_deg=-93.75,
                    aircraft_alt_m=16500.0,
                    aircraft_position_ecef_m=np.zeros(3),
                    rays=[],
                )
            ]
            cfg = PopulationConfig(enabled=True, dataset_path=str(dataset), overflight_half_width_km=40.0)
            result = analyze_population_impact(emissions, cfg)

            self.assertIsNotNone(result)
            assert result is not None
            self.assertGreater(result.total_population_in_heatmap, 0.0)
            self.assertEqual(result.total_exposed_population, 0.0)
            self.assertGreater(result.total_overflight_population, 0.0)
            self.assertTrue(bool(np.any(result.overflight_cell_mask)))

    def test_overflight_weight_affects_optimizer_score(self):
        metrics = ObjectiveMetrics(
            exposure_people=10.0,
            overflight_population=100.0,
            ground_hit_count=5,
            populated_ground_hit_count=2,
            cutoff_emission_fraction=0.4,
            source_cutoff_emission_fraction=0.5,
            populated_hit_population=20.0,
            populated_exposed_area_km2=4.0,
            overflight_area_km2=8.0,
            elapsed_time_s=120.0,
            time_proxy_s=130.0,
            fuel_proxy=40.0,
            mean_ground_speed_mps=350.0,
            route_distance_km=500.0,
            distance_to_destination_m=0.0,
            abort_samples=0,
        )
        norm = _normalizer_from_baseline(metrics)

        base_settings = RouteOptimizationSettings(weight_overflight_population=0.0, weight_overflight_area=0.0)
        boosted_settings = RouteOptimizationSettings(weight_overflight_population=4.0, weight_overflight_area=2.0)

        score_base = _score_metrics(metrics, norm, base_settings)
        score_boosted = _score_metrics(metrics, norm, boosted_settings)
        self.assertGreater(score_boosted, score_base)

    def test_exposure_limit_and_cutoff_shortfall_affect_optimizer_score(self):
        baseline = ObjectiveMetrics(
            exposure_people=80.0,
            overflight_population=100.0,
            ground_hit_count=2,
            populated_ground_hit_count=0,
            cutoff_emission_fraction=0.85,
            source_cutoff_emission_fraction=0.9,
            populated_hit_population=0.0,
            populated_exposed_area_km2=1.0,
            overflight_area_km2=4.0,
            elapsed_time_s=120.0,
            time_proxy_s=120.0,
            fuel_proxy=30.0,
            mean_ground_speed_mps=340.0,
            route_distance_km=480.0,
            distance_to_destination_m=0.0,
            abort_samples=0,
        )
        violating = ObjectiveMetrics(
            exposure_people=160.0,
            overflight_population=100.0,
            ground_hit_count=2,
            populated_ground_hit_count=0,
            cutoff_emission_fraction=0.45,
            source_cutoff_emission_fraction=0.5,
            populated_hit_population=0.0,
            populated_exposed_area_km2=1.0,
            overflight_area_km2=4.0,
            elapsed_time_s=120.0,
            time_proxy_s=120.0,
            fuel_proxy=30.0,
            mean_ground_speed_mps=340.0,
            route_distance_km=480.0,
            distance_to_destination_m=0.0,
            abort_samples=0,
        )
        norm = _normalizer_from_baseline(baseline)
        settings = RouteOptimizationSettings(
            boom_exposure_limit_people=100.0,
            weight_boom_exposure_limit=8.0,
            min_cutoff_emission_fraction=0.75,
            weight_cutoff_shortfall=4.0,
        )

        score_baseline = _score_metrics(baseline, norm, settings)
        score_violating = _score_metrics(violating, norm, settings)
        self.assertGreater(score_violating, score_baseline)

    def test_boom_envelope_sampling_preserves_extents(self):
        ray_a = RayResult(
            ray_id=0,
            trajectory_local_m=np.zeros((20, 3), dtype=float),
            trajectory_ecef_m=np.zeros((20, 3), dtype=float),
            trajectory_geodetic=np.column_stack(
                [
                    np.linspace(44.0, 44.6, 20),
                    np.linspace(-94.0, -93.1, 20),
                    np.linspace(16_500.0, 0.0, 20),
                ]
            ),
            ground_hit=True,
            ground_hit_lat_lon=(44.6, -93.1),
        )
        ray_b = RayResult(
            ray_id=1,
            trajectory_local_m=np.zeros((20, 3), dtype=float),
            trajectory_ecef_m=np.zeros((20, 3), dtype=float),
            trajectory_geodetic=np.column_stack(
                [
                    np.linspace(44.2, 45.4, 20),
                    np.linspace(-93.8, -92.4, 20),
                    np.linspace(16_500.0, 1_200.0, 20),
                ]
            ),
            ground_hit=False,
            ground_hit_lat_lon=None,
        )
        emission = EmissionResult(
            emission_time_utc=datetime.now(timezone.utc),
            aircraft_lat_deg=44.1,
            aircraft_lon_deg=-93.9,
            aircraft_alt_m=16_500.0,
            aircraft_position_ecef_m=np.zeros(3),
            rays=[ray_a, ray_b],
        )
        result = SimulationResult(emissions=[emission], config_dict={})

        sampled = result.boom_envelope_geodetic_points(max_points_per_ray=5, max_total_points=6)

        self.assertEqual(sampled.shape[1], 3)
        self.assertLessEqual(sampled.shape[0], 6)
        self.assertTrue(np.any(np.isclose(sampled[:, 0], 44.0)))
        self.assertTrue(np.any(np.isclose(sampled[:, 0], 45.4)))
        self.assertTrue(np.any(np.isclose(sampled[:, 1], -94.0)))
        self.assertTrue(np.any(np.isclose(sampled[:, 1], -92.4)))
        self.assertTrue(np.any(np.isclose(sampled[:, 2], 0.0)))
        self.assertTrue(np.any(np.isclose(sampled[:, 2], 16_500.0)))


if __name__ == "__main__":
    unittest.main()
