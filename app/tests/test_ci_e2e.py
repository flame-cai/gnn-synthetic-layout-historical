import json
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

from tests.pipeline_ablation_experiment import run_pipeline_strategy_ablation_gate
from tests.precommit_gate_config import get_pipeline_precommit_datasets


class EndToEndEvalDatasetTest(unittest.TestCase):
    def test_precommit_pipeline_datasets_end_to_end(self):
        for dataset_config in get_pipeline_precommit_datasets():
            with self.subTest(dataset=dataset_config.name):
                result = run_pipeline_strategy_ablation_gate(dataset_config.name)

                self.assertEqual(result["study_mode"], "pipeline_strategy_ablation_gate")
                self.assertEqual(result["dataset_name"], dataset_config.name)
                self.assertTrue(result["passed"], result["failure_message"])
                self.assertEqual(result["status"], "passed")
                self.assertIn("benchmark", result["strategy_results"])
                self.assertIn("proposed", result["strategy_results"])
                self.assertTrue(result["comparison"]["passed"], result["comparison"]["failure_message"])

                latest_metrics_path = TESTS_ROOT / "logs" / "pipeline_ablation_latest.json"
                latest_summary_path = TESTS_ROOT / "logs" / "pipeline_ablation_latest.md"
                self.assertTrue(latest_metrics_path.exists())
                self.assertTrue(latest_summary_path.exists())

                latest_metrics = json.loads(latest_metrics_path.read_text(encoding="utf-8"))
                self.assertEqual(latest_metrics["study_mode"], "pipeline_strategy_ablation_gate")
                self.assertEqual(latest_metrics["failed_datasets"], [])
                dataset_result = latest_metrics["dataset_results"][dataset_config.name]
                self.assertIn("benchmark", dataset_result["strategy_results"])
                self.assertIn("proposed", dataset_result["strategy_results"])


if __name__ == "__main__":
    unittest.main()
