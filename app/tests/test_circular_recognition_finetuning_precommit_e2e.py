import json
import os
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

from recognition.console import configure_recognition_console_streams

configure_recognition_console_streams()

from tests.recognition_finetuning_config import get_precommit_hybrid_recognition_gate_config
from tests.recognition_finetuning_experiment import run_circular_recognition_strategy_ablation_gate


class CircularRecognitionFineTuningPrecommitEndToEndTest(unittest.TestCase):
    def test_eval_dataset_v2_circular_recognition_ablation_gate(self):
        dataset_name = os.getenv("CIRCULAR_RECOGNITION_FINETUNE_DATASET", "eval_dataset_v2")
        config = get_precommit_hybrid_recognition_gate_config(dataset_name)
        result = run_circular_recognition_strategy_ablation_gate(dataset_name=dataset_name)

        self.assertEqual(result["study_mode"], "circular_recognition_strategy_ablation_gate")
        self.assertEqual(result["dataset_name"], dataset_name)
        self.assertEqual(config.ordered_page_ids(), ["page_2", "page_3", "page_4", "page_5", "page_6"])
        self.assertEqual(config.fine_tune_page_ids(), ["page_2", "page_3", "page_4"])
        self.assertEqual(config.evaluation_page_ids(), ["page_5", "page_6"])
        self.assertIn("benchmark", result["strategy_results"])
        self.assertIn("proposed", result["strategy_results"])
        self.assertTrue(result["passed"], result["failure_message"])
        self.assertTrue(result["comparison"]["passed"], result["comparison"]["failure_message"])
        self.assertEqual(result["comparison"]["primary_metric_name"], "curve_metric_value")
        self.assertTrue(result["comparison"]["strict_primary_improvement_required"])

        latest_summary_path = TESTS_ROOT / "logs" / "circular_ocr_ablation_latest.md"
        latest_metrics_path = TESTS_ROOT / "logs" / "circular_ocr_ablation_latest.json"
        latest_txt_path = TESTS_ROOT / "logs" / "circular_ocr_ablation_latest.txt"
        self.assertTrue(latest_summary_path.exists())
        self.assertTrue(latest_metrics_path.exists())
        self.assertTrue(latest_txt_path.exists())

        latest_metrics = json.loads(latest_metrics_path.read_text(encoding="utf-8"))
        self.assertEqual(latest_metrics["study_mode"], "circular_recognition_strategy_ablation_gate")
        self.assertEqual(latest_metrics["failed_datasets"], [])
        dataset_result = latest_metrics["dataset_results"][dataset_name]
        self.assertIn("benchmark", dataset_result["strategy_results"])
        self.assertIn("proposed", dataset_result["strategy_results"])
        self.assertTrue(dataset_result["comparison"]["passed"])


if __name__ == "__main__":
    unittest.main()
