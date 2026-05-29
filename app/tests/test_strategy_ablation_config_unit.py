import sys
import unittest
from pathlib import Path


TESTS_ROOT = Path(__file__).resolve().parent
APP_ROOT = TESTS_ROOT.parent
REPO_ROOT = APP_ROOT.parent

if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.precommit_gate_config import (
    get_pipeline_precommit_dataset,
    get_recognition_precommit_dataset,
)
from tests.pipeline_ablation_experiment import _build_absolute_thresholds, _build_pipeline_comparison
from tests.recognition_finetuning_config import get_precommit_hybrid_recognition_gate_config
from recognition.line_segmentation.strategy_config import (
    get_benchmark_strategy_name,
    get_proposed_strategy_name,
    get_production_strategy_name,
    get_strategy_role_config,
    normalize_strategy_role_config_payload,
)


class StrategyAblationConfigUnitTest(unittest.TestCase):
    def test_eval_dataset_pipeline_ablation_config_records_local_polygons_benchmark(self):
        config = get_pipeline_precommit_dataset("eval_dataset")
        benchmark_strategy = get_benchmark_strategy_name()
        proposed_strategy = get_proposed_strategy_name()
        self.assertEqual(benchmark_strategy, "local_polygons_v1")
        self.assertEqual(proposed_strategy, "local_polygons_stable_unwrap_v1")
        self.assertEqual(config.name, "eval_dataset")
        self.assertEqual(config.strategy_ablation.benchmark.role, "benchmark")
        self.assertEqual(config.strategy_ablation.proposed.role, "proposed")
        self.assertEqual(config.strategy_ablation.benchmark.strategy_name, benchmark_strategy)
        self.assertEqual(config.strategy_ablation.benchmark.strategy_config, {"BINARIZE_THRESHOLD": 0.45})
        self.assertEqual(config.strategy_ablation.proposed.strategy_name, proposed_strategy)
        self.assertEqual(config.strategy_ablation.proposed.strategy_config, {"BINARIZE_THRESHOLD": 0.45})
        self.assertEqual(config.strategy_ablation.max_allowed_regression_abs, 0.01)
        self.assertFalse(config.strategy_ablation.strict_primary_improvement_required)
        self.assertEqual(config.latest_artifact_basename, "pipeline_ablation_latest")
        self.assertEqual(len(config.ordered_page_ids()), config.expected_page_count)

    def test_pipeline_gate_blocks_only_on_page_cer(self):
        config = get_pipeline_precommit_dataset("eval_dataset")
        thresholds = _build_absolute_thresholds(
            config,
            {
                "page_cer": 0.39,
                "line_cer_50": 1.0,
                "line_cer_75": 1.0,
                "line_cer_range": 1.0,
            },
        )

        self.assertEqual(list(thresholds), ["page_cer"])
        self.assertTrue(thresholds["page_cer"]["passed"])

        comparison = _build_pipeline_comparison(
            config,
            {
                "role": "benchmark",
                "passed": True,
                "failure_message": "",
                "metrics": {
                    "page_cer": 0.33,
                    "line_cer_50": 0.10,
                    "line_cer_75": 0.10,
                    "line_cer_range": 0.10,
                },
            },
            {
                "role": "proposed",
                "passed": True,
                "failure_message": "",
                "metrics": {
                    "page_cer": 0.32,
                    "line_cer_50": 0.90,
                    "line_cer_75": 0.90,
                    "line_cer_range": 0.90,
                },
            },
        )

        self.assertTrue(comparison["passed"], comparison["failure_message"])
        self.assertEqual([item["metric_name"] for item in comparison["metric_comparisons"]], ["page_cer"])

    def test_eval_dataset_recognition_config_tracks_promoted_benchmark(self):
        gate = get_recognition_precommit_dataset("eval_dataset")
        config = get_precommit_hybrid_recognition_gate_config("eval_dataset")
        self.assertEqual(gate.strategy_ablation.benchmark.strategy_name, get_benchmark_strategy_name())
        self.assertEqual(gate.strategy_ablation.benchmark.strategy_config, {"BINARIZE_THRESHOLD": 0.45})
        self.assertEqual(gate.strategy_ablation.proposed.strategy_name, get_proposed_strategy_name())
        self.assertEqual(gate.strategy_ablation.proposed.strategy_config, {"BINARIZE_THRESHOLD": 0.45})
        self.assertEqual(gate.strategy_ablation.max_allowed_regression_abs, 0.02)
        self.assertFalse(gate.strategy_ablation.strict_primary_improvement_required)
        self.assertEqual(gate.latest_artifact_basename, "recognition_finetune_ablation_latest")
        self.assertEqual(len(config.ordered_page_ids()), 15)
        self.assertEqual(gate.fine_tune_page_count, 3)
        self.assertEqual(config.fine_tune_page_ids(), ["233_0002", "233_0003", "233_0004"])
        self.assertEqual(config.line_segmentation_strategy_name, get_benchmark_strategy_name())

    def test_eval_dataset_v2_circular_recognition_config_tracks_promoted_benchmark(self):
        gate = get_recognition_precommit_dataset("eval_dataset_v2")
        config = get_precommit_hybrid_recognition_gate_config("eval_dataset_v2")
        self.assertEqual(config.name, "eval_dataset_v2")
        self.assertEqual(config.ordered_page_ids(), ["page_2", "page_3", "page_4", "page_5", "page_6"])
        self.assertEqual(config.fine_tune_page_ids(), ["page_2", "page_3", "page_4"])
        self.assertEqual(config.evaluation_page_ids(), ["page_5", "page_6"])
        self.assertEqual(gate.strategy_ablation.benchmark.role, "benchmark")
        self.assertEqual(gate.strategy_ablation.proposed.role, "proposed")
        self.assertEqual(gate.strategy_ablation.benchmark.strategy_name, get_benchmark_strategy_name())
        self.assertEqual(gate.strategy_ablation.benchmark.strategy_config, {"BINARIZE_THRESHOLD": 0.45})
        self.assertEqual(gate.strategy_ablation.proposed.strategy_name, get_proposed_strategy_name())
        self.assertEqual(gate.strategy_ablation.proposed.strategy_config, {"BINARIZE_THRESHOLD": 0.45})
        self.assertTrue(gate.strategy_ablation.strict_primary_improvement_required)
        self.assertEqual(gate.strategy_ablation.max_allowed_regression_abs, 0.0)
        self.assertEqual(gate.latest_artifact_basename, "circular_ocr_ablation_latest")
        self.assertEqual(config.line_segmentation_strategy_name, get_benchmark_strategy_name())

    def test_checked_in_strategy_role_config_records_research_promotion(self):
        payload = get_strategy_role_config()

        self.assertEqual(payload["benchmark_strategy_name"], "local_polygons_v1")
        self.assertEqual(payload["proposed_strategy_name"], "local_polygons_stable_unwrap_v1")
        self.assertEqual(payload["production_strategy_name"], "local_polygons_v1")
        self.assertEqual(get_production_strategy_name(), "local_polygons_v1")
        self.assertEqual(len(payload["research_promotion_history"]), 2)
        self.assertEqual(payload["research_promotion_history"][0]["promoted_strategy_name"], "local_tangent_band_v1")
        self.assertEqual(
            payload["research_promotion_history"][0]["previous_benchmark_strategy_name"],
            "legacy_axis_bound_v1",
        )
        self.assertEqual(payload["research_promotion_history"][1]["promoted_strategy_name"], "local_polygons_v1")
        self.assertEqual(
            payload["research_promotion_history"][1]["previous_benchmark_strategy_name"],
            "local_tangent_band_v1",
        )
        self.assertEqual(len(payload["production_adoption_history"]), 1)
        self.assertEqual(
            payload["production_adoption_history"][0]["adopted_strategy_name"],
            "local_polygons_v1",
        )
        self.assertEqual(
            payload["production_adoption_history"][0]["previous_production_strategy_name"],
            "legacy_axis_bound_v1",
        )

    def test_strategy_role_config_allows_research_and_production_to_diverge(self):
        research_promoted_payload = normalize_strategy_role_config_payload(
            {
                "benchmark_strategy_name": "local_tangent_band_v1",
                "proposed_strategy_name": None,
                "production_strategy_name": "legacy_axis_bound_v1",
                "research_promotion_history": [{"promoted_strategy_name": "local_tangent_band_v1"}],
                "production_adoption_history": [],
            }
        )
        production_adopted_payload = normalize_strategy_role_config_payload(
            {
                "benchmark_strategy_name": "legacy_axis_bound_v1",
                "proposed_strategy_name": "local_tangent_band_v1",
                "production_strategy_name": "local_tangent_band_v1",
                "research_promotion_history": [],
                "production_adoption_history": [{"adopted_strategy_name": "local_tangent_band_v1"}],
            }
        )

        self.assertEqual(research_promoted_payload["benchmark_strategy_name"], "local_tangent_band_v1")
        self.assertEqual(research_promoted_payload["production_strategy_name"], "legacy_axis_bound_v1")
        self.assertEqual(production_adopted_payload["benchmark_strategy_name"], "legacy_axis_bound_v1")
        self.assertEqual(production_adopted_payload["production_strategy_name"], "local_tangent_band_v1")

    def test_legacy_promotion_history_key_normalizes_to_research_history(self):
        payload = normalize_strategy_role_config_payload(
            {
                "benchmark_strategy_name": "legacy_axis_bound_v1",
                "proposed_strategy_name": "local_tangent_band_v1",
                "promotion_history": [{"promoted_strategy_name": "legacy_axis_bound_v1"}],
            }
        )

        self.assertEqual(payload["production_strategy_name"], "legacy_axis_bound_v1")
        self.assertEqual(payload["research_promotion_history"], [{"promoted_strategy_name": "legacy_axis_bound_v1"}])
        self.assertEqual(payload["production_adoption_history"], [])


if __name__ == "__main__":
    unittest.main()
