import unittest

from mach_cutoff.benchmark.analysis import _annotate_relative_metrics, _feasibility_summary
from mach_cutoff.benchmark.config import ResearchObjectiveConfig, load_benchmark_config


class BenchmarkAnalysisTests(unittest.TestCase):
    def test_example_benchmark_manifest_exposes_research_objectives(self):
        cfg = load_benchmark_config("examples/benchmark_us_archetypes.json")
        self.assertEqual(cfg.research_objectives.baseline_route_class_id, "fastest")
        self.assertEqual(cfg.research_objectives.boom_exposure_limit_people, 100000.0)
        self.assertEqual(cfg.research_objectives.max_time_penalty_pct, 20.0)
        self.assertEqual(cfg.research_objectives.min_cutoff_emission_fraction, 0.75)

    def test_feasibility_uses_thresholds_and_cutoff_target(self):
        rows = [
            {
                "scenario_id": "s1",
                "sensitivity_profile": "core",
                "route_class": "fastest",
                "elapsed_time_s": 100.0,
                "boom_exposed_population": 200.0,
                "overflight_population": 320.0,
                "cutoff_emission_fraction": 0.20,
                "distance_to_destination_m": 0.0,
                "abort_samples": 0,
            },
            {
                "scenario_id": "s1",
                "sensitivity_profile": "core",
                "route_class": "population_avoidant",
                "elapsed_time_s": 110.0,
                "boom_exposed_population": 90.0,
                "overflight_population": 120.0,
                "cutoff_emission_fraction": 0.40,
                "distance_to_destination_m": 0.0,
                "abort_samples": 0,
            },
            {
                "scenario_id": "s1",
                "sensitivity_profile": "core",
                "route_class": "cutoff_optimized",
                "elapsed_time_s": 115.0,
                "boom_exposed_population": 80.0,
                "overflight_population": 150.0,
                "cutoff_emission_fraction": 0.70,
                "distance_to_destination_m": 0.0,
                "abort_samples": 0,
            },
        ]
        objectives = ResearchObjectiveConfig(
            baseline_route_class_id="fastest",
            boom_exposure_limit_people=100.0,
            max_time_penalty_pct=20.0,
            min_cutoff_emission_fraction=0.75,
            max_abort_samples=0,
            max_distance_to_destination_m=1000.0,
        )

        _annotate_relative_metrics(rows, research_objectives=objectives)
        summary = _feasibility_summary(rows, research_objectives=objectives)

        fastest, population_avoidant, cutoff_optimized = rows
        self.assertEqual(fastest["time_penalty_vs_fastest_pct"], 0.0)
        self.assertAlmostEqual(population_avoidant["boom_exposure_reduction_vs_fastest_pct"], 55.0)
        self.assertAlmostEqual(cutoff_optimized["time_penalty_vs_fastest_pct"], 15.0)

        self.assertFalse(bool(fastest["feasible_under_research_objective"]))
        self.assertTrue(bool(population_avoidant["feasible_under_research_objective"]))
        self.assertFalse(bool(cutoff_optimized["feasible_under_research_objective"]))
        self.assertIn("cutoff_fraction_below_target", cutoff_optimized["feasibility_reasons"])

        self.assertEqual(summary["by_route_class"]["population_avoidant"]["feasible_runs"], 1)
        self.assertEqual(summary["by_route_class"]["cutoff_optimized"]["feasible_runs"], 0)


if __name__ == "__main__":
    unittest.main()
