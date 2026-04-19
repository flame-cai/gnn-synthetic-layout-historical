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

from tests.recognition_finetuning_config import get_page_only_followup_policy_configs
from tests.recognition_finetuning_experiment import run_page_only_continuation_experiment


class RecognitionFineTuningPageOnlyEndToEndTest(unittest.TestCase):
    def test_eval_dataset_page_only_continuation(self):
        dataset_name = os.getenv("RECOGNITION_FINETUNE_DATASET", "eval_dataset")
        result = run_page_only_continuation_experiment(dataset_name=dataset_name)

        self.assertEqual(result["study_mode"], "page_only_policy_continuation")
        self.assertEqual(len(result["policy_runs"]), 4)
        self.assertEqual(len(result["steps"]), 10, "Expected pretrained baseline plus 9 page-only continuation steps.")
        self.assertTrue(result["run_dir"].name.endswith(f"_ocrft_pageonly_{dataset_name}"))
        self.assertTrue(all(step["train_dataset_page_count"] == 1 for step in result["steps"][1:]))
        self.assertTrue(all(policy_result["policy"]["training_policy"] == "page_only" for policy_result in result["policy_runs"]))

        fine_tune_page_ids = get_page_only_followup_policy_configs(dataset_name)[0].fine_tune_page_ids()
        self.assertEqual(result["steps"][6]["training_page_ids"], [fine_tune_page_ids[5]])
        self.assertEqual(result["steps"][6]["base_checkpoint"], result["steps"][5]["output_checkpoint"])

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
                ("adam", 0.00001, 200),
                ("adadelta", 0.2, 60),
                ("adadelta", 0.05, 200),
            },
        )

        winning_policy_result = next(
            policy_result
            for policy_result in result["policy_runs"]
            if policy_result["policy_slug"] == result["winning_policy"]
        )
        self.assertEqual(winning_policy_result["status"], "passed")
        self.assertEqual(len(winning_policy_result["steps"]), 10)

        for policy_result in result["policy_runs"]:
            fine_tune_metadata = json.loads(Path(policy_result["fine_tune_metadata_path"]).read_text(encoding="utf-8"))
            self.assertTrue(all(run["training_policy"] == "page_only" for run in fine_tune_metadata["runs"]))
            self.assertTrue(all(run["train_dataset_page_count"] == 1 for run in fine_tune_metadata["runs"]))

        latest_summary_path = TESTS_ROOT / "logs" / "recognition_finetune_page_only_latest.md"
        latest_metrics_path = TESTS_ROOT / "logs" / "recognition_finetune_page_only_latest.json"
        self.assertTrue(latest_summary_path.exists())
        self.assertTrue(latest_metrics_path.exists())

        latest_metrics = json.loads(latest_metrics_path.read_text(encoding="utf-8"))
        self.assertEqual(latest_metrics["study_mode"], "page_only_policy_continuation")
        self.assertEqual(latest_metrics["policy_run_count"], 4)
        self.assertEqual(latest_metrics["winning_policy"], result["winning_policy"])


if __name__ == "__main__":
    unittest.main()
