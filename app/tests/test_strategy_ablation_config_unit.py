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
from tests.recognition_finetuning_config import get_precommit_hybrid_recognition_gate_config


class StrategyAblationConfigUnitTest(unittest.TestCase):
    def test_eval_dataset_pipeline_ablation_config(self):
        config = get_pipeline_precommit_dataset("eval_dataset")
        self.assertEqual(config.name, "eval_dataset")
        self.assertEqual(config.strategy_ablation.benchmark.role, "benchmark")
        self.assertEqual(config.strategy_ablation.proposed.role, "proposed")
        self.assertEqual(config.strategy_ablation.benchmark.strategy_name, "legacy_axis_bound_v1")
        self.assertEqual(config.strategy_ablation.proposed.strategy_name, "legacy_axis_bound_v1")
        self.assertEqual(config.strategy_ablation.max_allowed_regression_abs, 0.01)
        self.assertFalse(config.strategy_ablation.strict_primary_improvement_required)
        self.assertEqual(config.latest_artifact_basename, "pipeline_ablation_latest")
        self.assertEqual(len(config.ordered_page_ids()), config.expected_page_count)

    def test_eval_dataset_recognition_ablation_config(self):
        gate = get_recognition_precommit_dataset("eval_dataset")
        config = get_precommit_hybrid_recognition_gate_config("eval_dataset")
        self.assertEqual(gate.strategy_ablation.benchmark.strategy_name, "legacy_axis_bound_v1")
        self.assertEqual(gate.strategy_ablation.proposed.strategy_name, "legacy_axis_bound_v1")
        self.assertEqual(gate.strategy_ablation.max_allowed_regression_abs, 0.005)
        self.assertFalse(gate.strategy_ablation.strict_primary_improvement_required)
        self.assertEqual(gate.latest_artifact_basename, "recognition_finetune_ablation_latest")
        self.assertEqual(len(config.ordered_page_ids()), 15)
        self.assertEqual(config.line_segmentation_strategy_name, "legacy_axis_bound_v1")

    def test_eval_dataset_v2_circular_recognition_config(self):
        gate = get_recognition_precommit_dataset("eval_dataset_v2")
        config = get_precommit_hybrid_recognition_gate_config("eval_dataset_v2")
        self.assertEqual(config.name, "eval_dataset_v2")
        self.assertEqual(config.ordered_page_ids(), ["page_2", "page_3", "page_4", "page_5", "page_6"])
        self.assertEqual(config.fine_tune_page_ids(), ["page_2", "page_3", "page_4"])
        self.assertEqual(config.evaluation_page_ids(), ["page_5", "page_6"])
        self.assertEqual(gate.strategy_ablation.benchmark.role, "benchmark")
        self.assertEqual(gate.strategy_ablation.proposed.role, "proposed")
        self.assertEqual(gate.strategy_ablation.benchmark.strategy_name, "legacy_axis_bound_v1")
        self.assertEqual(gate.strategy_ablation.proposed.strategy_name, "legacy_axis_bound_v1")
        self.assertTrue(gate.strategy_ablation.strict_primary_improvement_required)
        self.assertEqual(gate.strategy_ablation.max_allowed_regression_abs, 0.0)
        self.assertEqual(gate.latest_artifact_basename, "circular_ocr_ablation_latest")


if __name__ == "__main__":
    unittest.main()
