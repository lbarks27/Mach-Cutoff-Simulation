import unittest

from mach_cutoff.benchmark.config import load_benchmark_config
from mach_cutoff.benchmark.scenarios import build_core_scenarios


class BenchmarkScenarioTests(unittest.TestCase):
    def test_core_matrix_count_and_ids(self):
        cfg = load_benchmark_config("examples/benchmark_us_archetypes.json")
        scenarios = build_core_scenarios(cfg)
        self.assertEqual(len(scenarios), 24)
        self.assertEqual(len(cfg.anchor_scenario_ids), 6)
        ids = {s.scenario_id for s in scenarios}
        self.assertIn("msp_dtw__20250115T1200Z", ids)
        self.assertIn("iad_bos__20250715T1200Z", ids)


if __name__ == "__main__":
    unittest.main()
