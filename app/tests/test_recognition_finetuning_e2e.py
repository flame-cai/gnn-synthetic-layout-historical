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

from tests.recognition_finetuning_experiment import run_recognition_finetuning_experiment


class RecognitionFineTuningEndToEndTest(unittest.TestCase):
    def test_eval_dataset_sequential_fine_tuning(self):
        dataset_name = os.getenv("RECOGNITION_FINETUNE_DATASET", "eval_dataset")
        result = run_recognition_finetuning_experiment(dataset_name=dataset_name)
        self.assertEqual(len(result["steps"]), 6, "Expected step 0 through step 5 in the sequential fine-tuning run.")
        self.assertGreaterEqual(len(result["policy_runs"]), 2, "Expected blocker-first policy comparisons to run.")
        self.assertIn("curve_metric_value", result["curve_metrics"])
        self.assertIn("winning_policy", result)


if __name__ == "__main__":
    unittest.main()
