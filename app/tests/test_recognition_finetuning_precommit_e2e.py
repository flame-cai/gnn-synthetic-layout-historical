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

from tests.precommit_gate_config import get_recognition_precommit_dataset
from tests.recognition_finetuning_experiment import run_recognition_precommit_gate


class RecognitionFineTuningPrecommitEndToEndTest(unittest.TestCase):
    def test_eval_dataset_recognition_precommit_gate(self):
        dataset_name = os.getenv("RECOGNITION_FINETUNE_DATASET", "eval_dataset")
        gate_config = get_recognition_precommit_dataset(dataset_name)
        result = run_recognition_precommit_gate(dataset_name=dataset_name)

        self.assertEqual(result["study_mode"], "recognition_precommit_gate")
        self.assertEqual(result["dataset_name"], dataset_name)
        self.assertTrue(result["passed"], result["failure_message"])
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["policy"]["training_policy"], "page_plus_random_history")
        self.assertEqual(result["policy"]["history_sample_line_count"], 10)
        self.assertEqual(result["policy"]["width_policy"], "batch_max_pad")
        self.assertEqual(result["policy"]["optimizer"], "adadelta")
        self.assertEqual(round(float(result["policy"]["lr"]), 8), 0.2)
        self.assertEqual(int(result["policy"]["num_iter"]), 60)

        self.assertIn("curve_metric_value", result["threshold_results"])
        self.assertIn("final_page_cer", result["threshold_results"])
        self.assertIn("first_step_gain", result["threshold_results"])
        self.assertTrue(all(item["passed"] for item in result["threshold_results"].values()))
        self.assertEqual(result["threshold_results"]["curve_metric_value"]["threshold"], gate_config.max_curve_metric_value)
        self.assertEqual(result["threshold_results"]["final_page_cer"]["threshold"], gate_config.max_final_page_cer)
        self.assertEqual(result["threshold_results"]["first_step_gain"]["threshold"], gate_config.min_first_step_gain)
        self.assertIn("regression_guard_passed", result["curve_metrics"])
        self.assertIn("max_regression", result["curve_metrics"])

        self.assertTrue(result["summary_path"].exists())
        self.assertTrue(result["metrics_path"].exists())
        self.assertTrue(result["fine_tune_metadata_path"].exists())

        latest_summary_path = TESTS_ROOT / "logs" / "recognition_finetune_precommit_latest.md"
        latest_metrics_path = TESTS_ROOT / "logs" / "recognition_finetune_precommit_latest.json"
        latest_txt_path = TESTS_ROOT / "logs" / "recognition_finetune_precommit_latest.txt"
        self.assertTrue(latest_summary_path.exists())
        self.assertTrue(latest_metrics_path.exists())
        self.assertTrue(latest_txt_path.exists())

        latest_metrics = json.loads(latest_metrics_path.read_text(encoding="utf-8"))
        self.assertEqual(latest_metrics["study_mode"], "recognition_precommit_gate")
        self.assertEqual(latest_metrics["failed_datasets"], [])
        self.assertIn("dataset_results", latest_metrics)
        self.assertIn(dataset_name, latest_metrics["dataset_results"])
        dataset_result = latest_metrics["dataset_results"][dataset_name]
        self.assertEqual(dataset_result["status"], "passed")
        self.assertTrue(dataset_result["blocking_thresholds_passed"])
        self.assertIn("curve_metrics", dataset_result)
        self.assertIn("threshold_results", dataset_result)
        self.assertIn("fine_tune_metadata_path", dataset_result)


if __name__ == "__main__":
    unittest.main()
