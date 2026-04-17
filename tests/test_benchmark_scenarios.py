import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from mach_cutoff.benchmark.config import load_benchmark_config
from mach_cutoff.benchmark.scenarios import build_core_scenarios


class BenchmarkScenarioTests(unittest.TestCase):
    def test_core_matrix_count_and_ids(self):
        cfg = load_benchmark_config("examples/benchmark_us_archetypes.json")
        self.assertEqual(cfg.gpw.source, "census_blockgroup")
        self.assertAlmostEqual(cfg.gpw.prepared_cell_deg, 0.02)
        scenarios = build_core_scenarios(cfg)
        self.assertEqual(len(scenarios), 24)
        self.assertEqual(len(cfg.anchor_scenario_ids), 6)
        ids = {s.scenario_id for s in scenarios}
        self.assertIn("msp_dtw__20250115T1200Z", ids)
        self.assertIn("iad_bos__20250715T1200Z", ids)

    def test_benchmark_config_supports_extends_and_path_resolution(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            parent_dir = root / "parent"
            child_dir = root / "child"
            examples_dir = root / "examples"
            cache_dir = root / "cache"
            parent_dir.mkdir()
            child_dir.mkdir()
            examples_dir.mkdir()
            cache_dir.mkdir()

            base_path = parent_dir / "base.json"
            child_path = child_dir / "child.json"

            base_path.write_text(
                """
{
  "name": "base_manifest",
  "output_root": "../outputs/base",
  "base_config_path": "../examples/base_config.json",
  "base_guidance_config_path": "../examples/base_guidance.json",
  "population_cache_dir": "../cache/population",
  "gpw": {
    "raw_cache_dir": "../cache/raw",
    "prepared_cache_dir": "../cache/prepared"
  },
  "research_objectives": {
    "boom_exposure_limit_people": 100000.0
  },
  "route_classes": {
    "fastest": {
      "config_overrides": {
        "route_optimization": {
          "weight_speed": 4.0
        }
      }
    }
  },
  "corridors": {
    "sample": {
      "origin_lat_deg": 1.0,
      "origin_lon_deg": 2.0,
      "destination_lat_deg": 3.0,
      "destination_lon_deg": 4.0
    }
  },
  "timestamps_utc": ["2025-01-15T12:00:00Z"]
}
""".strip(),
                encoding="utf-8",
            )
            child_path.write_text(
                """
{
  "extends": "../parent/base.json",
  "name": "child_manifest",
  "output_root": "../outputs/child",
  "research_objectives": {
    "boom_exposure_limit_people": 50000.0
  },
  "route_classes": {
    "fastest": {
      "guidance_overrides": {
        "boom_avoidance": {
          "optimizer": {
            "enabled": false
          }
        }
      }
    },
    "subsonic_fastest": {
      "config_overrides": {
        "aircraft": {
          "mach": 0.95
        }
      }
    }
  }
}
""".strip(),
                encoding="utf-8",
            )

            cfg = load_benchmark_config(child_path)
            self.assertEqual(cfg.name, "child_manifest")
            self.assertEqual(cfg.research_objectives.boom_exposure_limit_people, 50000.0)
            self.assertIn("fastest", cfg.route_classes)
            self.assertIn("subsonic_fastest", cfg.route_classes)
            self.assertEqual(cfg.route_classes["fastest"].config_overrides["route_optimization"]["weight_speed"], 4.0)
            self.assertFalse(
                cfg.route_classes["fastest"].guidance_overrides["boom_avoidance"]["optimizer"]["enabled"]
            )
            self.assertEqual(cfg.route_classes["subsonic_fastest"].config_overrides["aircraft"]["mach"], 0.95)
            self.assertTrue(str(cfg.base_config_path).endswith("/examples/base_config.json"))
            self.assertTrue(str(cfg.output_root).endswith("/outputs/child"))

    def test_xb1_threshold_manifest_loads_with_expected_matrix(self):
        cfg = load_benchmark_config("results/configs/benchmark_xb1_custom_visual_limit100k_20260413.json")
        scenarios = build_core_scenarios(cfg)
        self.assertEqual(len(scenarios), 36)
        self.assertEqual(len(cfg.route_classes), 4)
        self.assertEqual(len(cfg.sensitivity_profiles), 16)
        self.assertEqual(cfg.research_objectives.boom_exposure_limit_people, 100000.0)
        self.assertIn("subsonic_fastest", cfg.route_classes)
        self.assertIn("performance_speed_limited", cfg.sensitivity_profiles)
        self.assertIn("exposure_limit_300k", cfg.sensitivity_profiles)


if __name__ == "__main__":
    unittest.main()
