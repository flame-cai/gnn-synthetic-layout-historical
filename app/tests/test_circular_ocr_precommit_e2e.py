from __future__ import annotations

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

from tests.circular_OCR_test.experiment import run_circular_ocr_precommit_gate


class CircularOcrPrecommitEndToEndTest(unittest.TestCase):
    def test_eval_dataset_v2_circular_ocr_gate(self):
        strategy_name = os.getenv("CIRCULAR_OCR_STRATEGY", "local_tangent_band_v1")
        result = run_circular_ocr_precommit_gate(strategy_name=strategy_name)

        self.assertEqual(result["study_mode"], "circular_ocr_precommit_gate")
        self.assertEqual(result["dataset_name"], "eval_dataset_v2")
        self.assertEqual(result["strategy_name"], strategy_name)
        self.assertTrue(result["passed"], result["failure_message"])
        self.assertIn("curve_metric_value", result["curve_metrics"])
        self.assertIn("final_page_cer", result["curve_metrics"])
        self.assertIn("first_step_gain", result["curve_metrics"])
        self.assertTrue(Path(result["summary_path"]).exists())
        self.assertTrue(Path(result["metrics_path"]).exists())
        self.assertTrue(Path(result["segmentation_metadata_path"]).exists())
        self.assertTrue(Path(result["orientation_metadata_path"]).exists())

        latest_metrics_path = TESTS_ROOT / "logs" / "circular_ocr_latest.json"
        latest_summary_path = TESTS_ROOT / "logs" / "circular_ocr_latest.md"
        latest_txt_path = TESTS_ROOT / "logs" / "circular_ocr_latest.txt"
        self.assertTrue(latest_metrics_path.exists())
        self.assertTrue(latest_summary_path.exists())
        self.assertTrue(latest_txt_path.exists())
        latest_metrics = json.loads(latest_metrics_path.read_text(encoding="utf-8"))
        self.assertEqual(latest_metrics["study_mode"], "circular_ocr_precommit_gate")


if __name__ == "__main__":
    unittest.main()
