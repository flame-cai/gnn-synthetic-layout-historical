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

from tests.precommit_gate_config import get_recognition_precommit_dataset
from tests.recognition_finetuning_experiment import run_recognition_strategy_ablation_gate


class RecognitionFineTuningPrecommitEndToEndTest(unittest.TestCase):
    def test_eval_dataset_recognition_precommit_gate(self):
        dataset_name = os.getenv("RECOGNITION_FINETUNE_DATASET", "eval_dataset")
        gate_config = get_recognition_precommit_dataset(dataset_name)
        if gate_config.strategy_ablation.proposed.strategy_name is None:
            self.skipTest("No proposed strategy configured; recognition strategy ablation gate is inactive.")
        result = run_recognition_strategy_ablation_gate(dataset_name=dataset_name)

        self.assertEqual(result["study_mode"], "recognition_strategy_ablation_gate")
        self.assertEqual(result["dataset_name"], dataset_name)
        self.assertTrue(result["passed"], result["failure_message"])
        self.assertEqual(result["status"], "passed")
        self.assertIn("benchmark", result["strategy_results"])
        self.assertIn("proposed", result["strategy_results"])
        self.assertTrue(result["comparison"]["passed"], result["comparison"]["failure_message"])
        proposed = result["strategy_results"]["proposed"]
        self.assertEqual(proposed["policy"]["training_policy"], "page_plus_random_history")
        self.assertEqual(proposed["policy"]["history_sample_line_count"], 10)
        self.assertEqual(proposed["policy"]["width_policy"], "batch_max_pad")
        self.assertEqual(proposed["policy"]["optimizer"], "adadelta")
        self.assertEqual(round(float(proposed["policy"]["lr"]), 8), 0.2)
        self.assertEqual(int(proposed["policy"]["num_iter"]), 60)

        self.assertIn("curve_metric_value", proposed["metrics"])
        self.assertIn("final_page_cer", proposed["metrics"])
        self.assertIn("first_step_gain", proposed["metrics"])
        self.assertIn("regression_guard_passed", proposed["metrics"])
        self.assertIn("max_regression", proposed["metrics"])
        self.assertEqual(
            result["comparison"]["allowed_regression_abs"],
            gate_config.strategy_ablation.max_allowed_regression_abs,
        )

        self.assertTrue(result["summary_path"].exists())
        self.assertTrue(result["metrics_path"].exists())
        self.assertTrue(proposed["fine_tune_metadata_path"].exists())

        latest_summary_path = TESTS_ROOT / "logs" / "recognition_finetune_ablation_latest.md"
        latest_metrics_path = TESTS_ROOT / "logs" / "recognition_finetune_ablation_latest.json"
        latest_txt_path = TESTS_ROOT / "logs" / "recognition_finetune_ablation_latest.txt"
        self.assertTrue(latest_summary_path.exists())
        self.assertTrue(latest_metrics_path.exists())
        self.assertTrue(latest_txt_path.exists())

        latest_metrics = json.loads(latest_metrics_path.read_text(encoding="utf-8"))
        self.assertEqual(latest_metrics["study_mode"], "recognition_strategy_ablation_gate")
        self.assertEqual(latest_metrics["failed_datasets"], [])
        self.assertIn("dataset_results", latest_metrics)
        self.assertIn(dataset_name, latest_metrics["dataset_results"])
        dataset_result = latest_metrics["dataset_results"][dataset_name]
        self.assertEqual(dataset_result["status"], "passed")
        self.assertTrue(dataset_result["comparison"]["passed"])
        self.assertIn("benchmark", dataset_result["strategy_results"])
        self.assertIn("proposed", dataset_result["strategy_results"])
        self.assertIn("fine_tune_metadata_path", dataset_result["strategy_results"]["proposed"])


if __name__ == "__main__":
    unittest.main()
