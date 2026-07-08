from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.downstream_ocr.reporting import write_experiment_report


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


class DownstreamOcrReportingTests(unittest.TestCase):
    def test_report_summarizes_metrics_and_gemini_usage(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            _write_json(
                root / "metrics" / "vlm_e2e" / "metrics.json",
                {
                    "method": {
                        "method_id": "vlm_e2e",
                        "display_name": "VLM (End-to-End)",
                        "uses_gt_layout": False,
                        "uses_finetuning": False,
                        "finetune_page_count": 0,
                        "uses_gemini": True,
                    },
                    "aggregate": {
                        "page_count": 1,
                        "valid_output_rate": 1.0,
                        "object_g_f1_50": 0.5,
                        "object_g_f1_75": 0.25,
                        "pixel_f1": 0.75,
                        "mean_page_cer": 0.2,
                        "median_page_cer": 0.2,
                        "micro_page_cer": 0.2,
                        "mean_textedit": 0.3,
                        "median_textedit": 0.3,
                        "micro_textedit": 0.3,
                    },
                    "page_records": [
                        {
                            "manuscript_id": "m",
                            "fold_id": "fold_1",
                            "page_id": "p1",
                            "method_id": "vlm_e2e",
                            "status": "success",
                        }
                    ],
                },
            )
            _write_json(
                root / "metrics" / "annotation_tool_gt_layout" / "metrics.json",
                {
                    "method": {
                        "method_id": "annotation_tool_gt_layout",
                        "display_name": "Annotation tool GT Layout",
                        "uses_gt_layout": True,
                        "uses_finetuning": False,
                        "finetune_page_count": 0,
                        "uses_gemini": False,
                    },
                    "aggregate": {
                        "page_count": 1,
                        "valid_output_rate": 1.0,
                        "object_g_f1_50": 1.0,
                        "object_g_f1_75": 1.0,
                        "pixel_f1": 1.0,
                        "mean_page_cer": 0.1,
                        "median_page_cer": 0.1,
                        "micro_page_cer": 0.1,
                        "mean_textedit": 0.15,
                        "median_textedit": 0.15,
                        "micro_textedit": 0.15,
                    },
                    "page_records": [],
                },
            )
            _write_json(
                root / "runs" / "vlm_e2e" / "fold_1" / "gemini_usage" / "p1.json",
                {
                    "page_id": "p1",
                    "status": "success",
                    "elapsed_seconds": 1.5,
                    "usage_metadata": {
                        "prompt_token_count": 142,
                        "candidates_token_count": 58,
                        "total_token_count": 200,
                    },
                },
            )

            artifacts = write_experiment_report(
                root,
                input_usd_per_1m_tokens=1.0,
                output_usd_per_1m_tokens=2.0,
            )

            self.assertTrue(artifacts.markdown_path.exists())
            self.assertTrue(artifacts.summary_csv_path.exists())
            self.assertTrue(artifacts.gemini_usage_csv_path.exists())
            summary = json.loads(artifacts.summary_json_path.read_text(encoding="utf-8"))
            by_method = {row["method_id"]: row for row in summary}
            self.assertEqual(by_method["vlm_e2e"]["gemini_total_token_count"], 200)
            self.assertAlmostEqual(by_method["vlm_e2e"]["gemini_estimated_cost_usd"], 0.000258)
            self.assertEqual(by_method["annotation_tool_gt_layout"]["gemini_estimated_cost_usd"], 0.0)

    def test_failed_gemini_request_without_metadata_is_not_reported_as_free(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            _write_json(
                root / "metrics" / "gemini_gt_layout" / "metrics.json",
                {
                    "method": {
                        "method_id": "gemini_gt_layout",
                        "display_name": "Gemini GT Layout",
                        "uses_gt_layout": True,
                        "uses_finetuning": False,
                        "finetune_page_count": 0,
                        "uses_gemini": True,
                    },
                    "aggregate": {
                        "page_count": 1,
                        "valid_output_rate": 0.0,
                        "object_g_f1_50": 0.0,
                        "object_g_f1_75": 0.0,
                        "pixel_f1": 0.0,
                        "mean_page_cer": 1.0,
                        "median_page_cer": 1.0,
                        "micro_page_cer": 1.0,
                        "mean_textedit": 1.0,
                        "median_textedit": 1.0,
                        "micro_textedit": 1.0,
                    },
                    "page_records": [],
                },
            )
            _write_json(
                root / "runs" / "gemini_gt_layout" / "fold_1" / "gemini_usage" / "p1.json",
                {
                    "page_id": "p1",
                    "status": "api_timeout",
                    "elapsed_seconds": 45.0,
                    "usage_records": [],
                    "usage_metadata": {
                        "prompt_token_count": 0,
                        "candidates_token_count": 0,
                        "total_token_count": 0,
                    },
                },
            )

            artifacts = write_experiment_report(
                root,
                input_usd_per_1m_tokens=1.5,
                output_usd_per_1m_tokens=9.0,
            )

            usage = json.loads(artifacts.gemini_usage_json_path.read_text(encoding="utf-8"))
            self.assertEqual(usage["rows"][0]["request_count"], 1)
            self.assertFalse(usage["rows"][0]["usage_metadata_available"])
            self.assertIsNone(usage["rows"][0]["estimated_cost_usd"])

            summary = json.loads(artifacts.summary_json_path.read_text(encoding="utf-8"))
            self.assertEqual(summary[0]["gemini_request_count"], 1)
            self.assertEqual(summary[0]["gemini_missing_usage_count"], 1)
            self.assertIn("actual API cost may be higher", summary[0]["gemini_pricing_note"])


if __name__ == "__main__":
    unittest.main()
