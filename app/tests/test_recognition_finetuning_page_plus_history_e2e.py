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

from tests.recognition_finetuning_config import get_page_plus_random_history_policy_configs
from tests.recognition_finetuning_experiment import run_page_plus_random_history_experiment


class RecognitionFineTuningPagePlusHistoryEndToEndTest(unittest.TestCase):
    def test_eval_dataset_page_plus_random_history(self):
        dataset_name = os.getenv("RECOGNITION_FINETUNE_DATASET", "eval_dataset")
        result = run_page_plus_random_history_experiment(dataset_name=dataset_name)

        self.assertEqual(result["study_mode"], "page_plus_random_history_followup")
        self.assertEqual(len(result["policy_runs"]), 2)
        self.assertTrue(result["run_dir"].name.endswith(f"_ocrft_pagehist_{dataset_name}"))
        self.assertTrue(all(policy_result["policy"]["training_policy"] == "page_plus_random_history" for policy_result in result["policy_runs"]))
        self.assertTrue(all(policy_result["policy"]["history_sample_line_count"] == 10 for policy_result in result["policy_runs"]))

        observed_policy_triples = {
            (
                policy_result["policy"]["optimizer"],
                round(float(policy_result["policy"]["lr"]), 8),
                int(policy_result["policy"]["num_iter"]),
            )
            for policy_result in result["policy_runs"]
        }
        self.assertEqual(
            observed_policy_triples,
            {
                ("adam", 0.00005, 60),
                ("adadelta", 0.2, 60),
            },
        )

        fine_tune_page_ids = get_page_plus_random_history_policy_configs(dataset_name)[0].fine_tune_page_ids()
        self.assertEqual(result["steps"][1]["train_dataset_history_line_count"], 0)
        self.assertEqual(result["steps"][1]["history_source_page_ids"], [])
        self.assertEqual(result["steps"][2]["history_source_page_ids"], [fine_tune_page_ids[0]])
        self.assertGreater(result["steps"][2]["train_dataset_history_line_count"], 0)
        self.assertLessEqual(result["steps"][2]["train_dataset_history_line_count"], 10)
        self.assertTrue(
            all(
                0 <= step["train_dataset_history_line_count"] <= 10
                for step in result["steps"][1:]
            )
        )

        for policy_result in result["policy_runs"]:
            fine_tune_metadata = json.loads(Path(policy_result["fine_tune_metadata_path"]).read_text(encoding="utf-8"))
            self.assertTrue(all(run["training_policy"] == "page_plus_random_history" for run in fine_tune_metadata["runs"]))
            self.assertTrue(all(run["train_dataset_history_line_count"] <= 10 for run in fine_tune_metadata["runs"]))
            if fine_tune_metadata["runs"]:
                self.assertEqual(fine_tune_metadata["runs"][0]["train_dataset_history_line_count"], 0)

        latest_summary_path = TESTS_ROOT / "logs" / "recognition_finetune_page_plus_history_latest.md"
        latest_metrics_path = TESTS_ROOT / "logs" / "recognition_finetune_page_plus_history_latest.json"
        self.assertTrue(latest_summary_path.exists())
        self.assertTrue(latest_metrics_path.exists())

        latest_metrics = json.loads(latest_metrics_path.read_text(encoding="utf-8"))
        self.assertEqual(latest_metrics["study_mode"], "page_plus_random_history_followup")
        self.assertEqual(latest_metrics["policy_run_count"], 2)


if __name__ == "__main__":
    unittest.main()
