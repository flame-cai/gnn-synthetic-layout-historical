from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from experiments.downstream_ocr.reporting import (
    EFFORT_GROUP_LABELS,
    _bootstrap_micro_metric,
    _effort_level,
    write_combined_table_report,
    write_experiment_report,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _page_record(
    *,
    method_id: str,
    page_id: str,
    page_cer_distance: int,
    page_cer_gt_chars: int,
    textedit_distance_sum: int,
    textedit_max_length_sum: int,
    fold_id: str = "fold_1",
) -> dict:
    page_cer = page_cer_distance / page_cer_gt_chars
    textedit = textedit_distance_sum / textedit_max_length_sum
    return {
        "manuscript_id": "m",
        "fold_id": fold_id,
        "page_id": page_id,
        "method_id": method_id,
        "status": "success",
        "num_gt_lines": 1,
        "num_pred_lines": 1,
        "tp_50": 1,
        "fp_50": 0,
        "fn_50": 0,
        "tp_75": 1,
        "fp_75": 0,
        "fn_75": 0,
        "pixel_tp": 10,
        "pixel_fp": 0,
        "pixel_fn": 0,
        "page_cer_distance": page_cer_distance,
        "page_cer_gt_chars": page_cer_gt_chars,
        "page_cer": page_cer,
        "textedit_distance_sum": textedit_distance_sum,
        "textedit_max_length_sum": textedit_max_length_sum,
        "textedit": textedit,
    }


class DownstreamOcrReportingTests(unittest.TestCase):
    def test_predicted_layout_finetuning_compartments_follow_off_the_shelf(self):
        ordered_method_ids = (
            "annotation_tool_e2e",
            "annotation_tool_pred_layout_ft_1",
            "annotation_tool_pred_layout_ft_2",
            "annotation_tool_pred_layout_ft_3",
            "annotation_tool_gt_layout",
            "annotation_tool_gt_layout_ft_1",
            "annotation_tool_gt_layout_ft_2",
            "annotation_tool_gt_layout_ft_3",
        )
        self.assertEqual(
            [_effort_level(method_id) for method_id in ordered_method_ids],
            list(range(8)),
        )
        self.assertEqual(EFFORT_GROUP_LABELS[0], "Off the Shelf")
        for level in (1, 2, 3):
            self.assertIn("No test layout correction", EFFORT_GROUP_LABELS[level])

    def test_page_cluster_bootstrap_keeps_repeated_fold_occurrences_together(self):
        rows = [
            _page_record(
                method_id="m",
                page_id="p1",
                fold_id="fold_1",
                page_cer_distance=1,
                page_cer_gt_chars=10,
                textedit_distance_sum=1,
                textedit_max_length_sum=10,
            ),
            _page_record(
                method_id="m",
                page_id="p1",
                fold_id="fold_2",
                page_cer_distance=3,
                page_cer_gt_chars=10,
                textedit_distance_sum=3,
                textedit_max_length_sum=10,
            ),
            _page_record(
                method_id="m",
                page_id="p2",
                fold_id="fold_1",
                page_cer_distance=8,
                page_cer_gt_chars=10,
                textedit_distance_sum=8,
                textedit_max_length_sum=10,
            ),
        ]
        result = _bootstrap_micro_metric(
            rows,
            numerator_key="page_cer_distance",
            denominator_key="page_cer_gt_chars",
            seed_label="unit-test",
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["unique_page_count"], 2)
        self.assertAlmostEqual(result["estimate"], 0.4)
        self.assertAlmostEqual(result["ci_lower"], 0.2)
        self.assertAlmostEqual(result["ci_upper"], 0.8)

    def test_report_summarizes_metrics_and_gemini_usage(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manuscript_root = root / "manuscript"
            _write_json(
                manuscript_root / "layout_analysis_output" / "layout_effort.json",
                {
                    "pages": {
                        "p1": {
                            "page_id": "p1",
                            "revision_count": 2,
                            "totals": {
                                "edit_count": 26,
                                "active_edit_time_seconds": 89.5632,
                            },
                        }
                    }
                },
            )
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
                    "manuscript_id": "m",
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
                        _page_record(
                            method_id="vlm_e2e",
                            page_id="p1",
                            page_cer_distance=2,
                            page_cer_gt_chars=10,
                            textedit_distance_sum=3,
                            textedit_max_length_sum=10,
                        )
                    ],
                },
            )
            _write_json(
                root / "metrics" / "annotation_tool_e2e" / "metrics.json",
                {
                    "method": {
                        "method_id": "annotation_tool_e2e",
                        "display_name": "Annotation tool e2e",
                        "uses_gt_layout": False,
                        "uses_finetuning": False,
                        "finetune_page_count": 0,
                        "uses_gemini": False,
                    },
                    "manuscript_id": "m",
                    "manuscript_root": str(manuscript_root),
                    "aggregate": {
                        "page_count": 1,
                        "valid_output_rate": 1.0,
                        "object_g_f1_50": 0.7,
                        "object_g_f1_75": 0.6,
                        "pixel_f1": 0.8,
                        "mean_page_cer": 0.4,
                        "median_page_cer": 0.4,
                        "micro_page_cer": 0.4,
                        "mean_textedit": 0.5,
                        "median_textedit": 0.5,
                        "micro_textedit": 0.5,
                    },
                    "page_records": [
                        _page_record(
                            method_id="annotation_tool_e2e",
                            page_id="p1",
                            page_cer_distance=4,
                            page_cer_gt_chars=10,
                            textedit_distance_sum=5,
                            textedit_max_length_sum=10,
                        )
                    ],
                },
            )
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
                    "manuscript_id": "m",
                    "manuscript_root": str(manuscript_root),
                    "aggregate": {
                        "page_count": 1,
                        "valid_output_rate": 1.0,
                        "object_g_f1_50": 1.0,
                        "object_g_f1_75": 1.0,
                        "pixel_f1": 1.0,
                        "mean_page_cer": 0.1,
                        "median_page_cer": 0.1,
                        "micro_page_cer": 0.1,
                        "mean_textedit": 0.2,
                        "median_textedit": 0.2,
                        "micro_textedit": 0.2,
                    },
                    "page_records": [
                        _page_record(
                            method_id="gemini_gt_layout",
                            page_id="p1",
                            page_cer_distance=1,
                            page_cer_gt_chars=10,
                            textedit_distance_sum=2,
                            textedit_max_length_sum=10,
                        )
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
                    "manuscript_id": "m",
                    "manuscript_root": str(manuscript_root),
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
                    "page_records": [
                        _page_record(
                            method_id="annotation_tool_gt_layout",
                            page_id="p1",
                            page_cer_distance=1,
                            page_cer_gt_chars=10,
                            textedit_distance_sum=3,
                            textedit_max_length_sum=20,
                        )
                    ],
                },
            )
            _write_json(
                root / "metrics" / "annotation_tool_pred_layout_ft_1" / "metrics.json",
                {
                    "method": {
                        "method_id": "annotation_tool_pred_layout_ft_1",
                        "display_name": "Annotation tool predicted test layout + 1 page FT",
                        "uses_gt_layout": False,
                        "uses_finetuning": True,
                        "finetune_page_count": 1,
                        "uses_gemini": False,
                    },
                    "manuscript_id": "m",
                    "ocr_active_learning_recipe": {
                        "source": "app.ocr_active_learning_runtime._runtime_recipe",
                        "sibling_checkpoint_strategy": "best_norm_ed",
                        "recipe": {"sibling_checkpoint_strategy": "best_norm_ed"},
                    },
                    "aggregate": {
                        "page_count": 1,
                        "valid_output_rate": 1.0,
                        "object_g_f1_50": 0.7,
                        "object_g_f1_75": 0.6,
                        "pixel_f1": 0.8,
                        "mean_page_cer": 0.3,
                        "median_page_cer": 0.3,
                        "micro_page_cer": 0.3,
                        "mean_textedit": 0.4,
                        "median_textedit": 0.4,
                        "micro_textedit": 0.4,
                    },
                    "page_records": [],
                },
            )
            _write_json(
                root / "metrics" / "annotation_tool_gt_layout_ft_1" / "metrics.json",
                {
                    "method": {
                        "method_id": "annotation_tool_gt_layout_ft_1",
                        "display_name": "Annotation tool GT Layout + 1 page FT",
                        "uses_gt_layout": True,
                        "uses_finetuning": True,
                        "finetune_page_count": 1,
                        "uses_gemini": False,
                    },
                    "manuscript_id": "m",
                    "ocr_active_learning_recipe": {
                        "source": "app.ocr_active_learning_runtime._runtime_recipe",
                        "sibling_checkpoint_strategy": "best_norm_ed",
                        "recipe": {"sibling_checkpoint_strategy": "best_norm_ed"},
                    },
                    "aggregate": {
                        "page_count": 1,
                        "valid_output_rate": 1.0,
                        "object_g_f1_50": 1.0,
                        "object_g_f1_75": 1.0,
                        "pixel_f1": 1.0,
                        "mean_page_cer": 0.08,
                        "median_page_cer": 0.08,
                        "micro_page_cer": 0.08,
                        "mean_textedit": 0.12,
                        "median_textedit": 0.12,
                        "micro_textedit": 0.12,
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
            _write_json(root / "report" / "layout_effort_impact.json", {"stale": True})
            stale_figure = root / "report" / "figures" / "layout_effort_vs_ocr_gain.png"
            stale_figure.parent.mkdir(parents=True, exist_ok=True)
            stale_figure.write_bytes(b"stale")

            artifacts = write_experiment_report(
                root,
                input_usd_per_1m_tokens=1.0,
                output_usd_per_1m_tokens=2.0,
            )

            self.assertTrue(artifacts.markdown_path.exists())
            self.assertTrue(artifacts.summary_csv_path.exists())
            self.assertTrue(artifacts.gemini_usage_csv_path.exists())
            self.assertTrue(artifacts.off_the_shelf_table_csv_path.exists())
            self.assertTrue(artifacts.annotation_gains_table_csv_path.exists())
            summary = json.loads(artifacts.summary_json_path.read_text(encoding="utf-8"))
            by_method = {row["method_id"]: row for row in summary}
            self.assertNotIn("gemini_gt_layout", by_method)
            self.assertEqual(by_method["vlm_e2e"]["gemini_total_token_count"], 200)
            self.assertEqual(by_method["vlm_e2e"]["gemini_attempt_count"], 1)
            self.assertEqual(by_method["vlm_e2e"]["gemini_retry_count"], 0)
            self.assertAlmostEqual(by_method["vlm_e2e"]["gemini_estimated_cost_usd"], 0.000258)
            self.assertAlmostEqual(by_method["vlm_e2e"]["micro_page_cer_ci_lower"], 0.2)
            self.assertAlmostEqual(by_method["vlm_e2e"]["micro_page_cer_ci_upper"], 0.2)
            self.assertEqual(by_method["vlm_e2e"]["micro_page_cer_bootstrap_unique_pages"], 1)
            self.assertEqual(by_method["annotation_tool_gt_layout"]["gemini_estimated_cost_usd"], 0.0)
            self.assertEqual(by_method["vlm_e2e"]["layout_condition"], "predicted_layout")
            self.assertEqual(by_method["annotation_tool_gt_layout"]["layout_condition"], "human_corrected_gt_layout")
            self.assertEqual(
                by_method["annotation_tool_pred_layout_ft_1"]["training_layout_condition"],
                "human_corrected_gt_layout",
            )
            self.assertEqual(
                by_method["annotation_tool_pred_layout_ft_1"]["test_layout_condition"],
                "predicted_layout",
            )
            self.assertTrue(by_method["annotation_tool_pred_layout_ft_1"]["human_training_layout"])
            self.assertFalse(by_method["annotation_tool_pred_layout_ft_1"]["human_test_layout"])
            self.assertIn(
                "no test-page layout correction",
                by_method["annotation_tool_pred_layout_ft_1"]["human_effort"],
            )
            self.assertIn(
                "not layout-detector performance",
                by_method["annotation_tool_gt_layout"]["layout_metric_interpretation"],
            )
            self.assertEqual(
                by_method["annotation_tool_gt_layout_ft_1"]["ocr_recipe_source"],
                "app.ocr_active_learning_runtime._runtime_recipe",
            )
            self.assertEqual(
                by_method["annotation_tool_gt_layout_ft_1"]["sibling_checkpoint_strategy"],
                "best_norm_ed",
            )
            with artifacts.layout_mode_comparisons_csv_path.open(encoding="utf-8", newline="") as handle:
                comparison_rows = list(csv.DictReader(handle))
            self.assertEqual(len(comparison_rows), 2)
            annotation_cer = next(
                row
                for row in comparison_rows
                if row["engine"] == "Annotation Tool" and row["metric_key"] == "micro_page_cer"
            )
            self.assertAlmostEqual(float(annotation_cer["layout_effort_mean_seconds_per_page"]), 89.5632)
            self.assertAlmostEqual(float(annotation_cer["layout_effort_ci_lower"]), 89.5632)
            self.assertAlmostEqual(float(annotation_cer["relative_reduction_percent"]), 75.0)
            with artifacts.fold_metrics_csv_path.open(encoding="utf-8", newline="") as handle:
                fold_rows = list(csv.DictReader(handle))
            self.assertEqual(len(fold_rows), 3)
            off_table = json.loads(artifacts.off_the_shelf_table_json_path.read_text(encoding="utf-8"))
            self.assertEqual([row["method_id"] for row in off_table["rows"]], ["vlm_e2e"])
            gains_table = json.loads(artifacts.annotation_gains_table_json_path.read_text(encoding="utf-8"))
            gains_by_step = {row["finetune_pages"]: row for row in gains_table["rows"]}
            self.assertAlmostEqual(gains_by_step[0]["page_cer_relative_reduction_percent"], 75.0)
            self.assertAlmostEqual(gains_by_step[1]["page_cer_relative_reduction_percent"], 73.33333333333334)
            self.assertIn("m: 89.6", gains_table["caption"])
            figure_names = {path.name for path in artifacts.figure_paths}
            self.assertNotIn("micro_page_cer_by_method.png", figure_names)
            self.assertNotIn("micro_textedit_by_method.png", figure_names)
            self.assertNotIn("layout_effort_vs_ocr_gain.png", figure_names)
            self.assertFalse((root / "report" / "layout_effort_impact.json").exists())
            self.assertFalse((root / "report" / "figures" / "micro_page_cer_by_method.png").exists())
            self.assertFalse((root / "report" / "figures" / "micro_textedit_by_method.png").exists())
            self.assertFalse(stale_figure.exists())

    def test_failed_gemini_request_without_metadata_is_not_reported_as_free(self):
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
                root / "runs" / "vlm_e2e" / "fold_1" / "gemini_usage" / "p1.json",
                {
                    "page_id": "p1",
                    "status": "api_timeout",
                    "elapsed_seconds": 45.0,
                    "attempt_count": 4,
                    "retry_count": 3,
                    "max_retries": 3,
                    "request_count": 4,
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
            self.assertEqual(usage["rows"][0]["attempt_count"], 4)
            self.assertEqual(usage["rows"][0]["retry_count"], 3)
            self.assertEqual(usage["rows"][0]["request_count"], 4)
            self.assertFalse(usage["rows"][0]["usage_metadata_available"])
            self.assertIsNone(usage["rows"][0]["estimated_cost_usd"])

            summary = json.loads(artifacts.summary_json_path.read_text(encoding="utf-8"))
            self.assertEqual(summary[0]["gemini_attempt_count"], 4)
            self.assertEqual(summary[0]["gemini_retry_count"], 3)
            self.assertEqual(summary[0]["gemini_request_count"], 4)
            self.assertEqual(summary[0]["gemini_missing_usage_count"], 1)
            self.assertIn("actual API cost may be higher", summary[0]["gemini_pricing_note"])

    def test_combined_table_report_keeps_manuscripts_separate(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_roots = []
            for manuscript_id, cer_offset in (("yajn", 0), ("dense", 2)):
                run_root = root / f"run_{manuscript_id}"
                input_roots.append(run_root)
                for method_id, uses_gt_layout, distance in (
                    ("vlm_e2e", False, 3 + cer_offset),
                    ("annotation_tool_e2e", False, 5 + cer_offset),
                    ("annotation_tool_gt_layout", True, 2 + cer_offset),
                ):
                    record = _page_record(
                        method_id=method_id,
                        page_id="p1",
                        page_cer_distance=distance,
                        page_cer_gt_chars=10,
                        textedit_distance_sum=distance,
                        textedit_max_length_sum=10,
                    )
                    record["manuscript_id"] = manuscript_id
                    if uses_gt_layout:
                        record.update(
                            {
                                "layout_effort_available": True,
                                "layout_effort_active_edit_time_seconds": 60.0 + cer_offset,
                            }
                        )
                    _write_json(
                        run_root / "metrics" / method_id / "metrics.json",
                        {
                            "method": {
                                "method_id": method_id,
                                "display_name": method_id,
                                "uses_gt_layout": uses_gt_layout,
                                "uses_finetuning": False,
                                "finetune_page_count": 0,
                                "uses_gemini": method_id == "vlm_e2e",
                            },
                            "manuscript_id": manuscript_id,
                            "aggregate": {
                                "page_count": 1,
                                "valid_output_rate": 1.0,
                                "object_g_f1_50": 1.0,
                                "object_g_f1_75": 1.0,
                                "pixel_f1": 1.0,
                                "mean_page_cer": distance / 10,
                                "median_page_cer": distance / 10,
                                "micro_page_cer": distance / 10,
                                "mean_textedit": distance / 10,
                                "median_textedit": distance / 10,
                                "micro_textedit": distance / 10,
                            },
                            "page_records": [record],
                        },
                    )

            artifacts = write_combined_table_report(input_roots, root / "combined")

            off_table = json.loads(artifacts.off_the_shelf_table_json_path.read_text(encoding="utf-8"))
            self.assertEqual([row["manuscript_id"] for row in off_table["rows"]], ["yajn", "dense"])
            gains_table = json.loads(artifacts.annotation_gains_table_json_path.read_text(encoding="utf-8"))
            gains_zero_rows = [row for row in gains_table["rows"] if row["finetune_pages"] == 0]
            self.assertEqual([row["manuscript_id"] for row in gains_zero_rows], ["yajn", "dense"])
            self.assertAlmostEqual(gains_zero_rows[0]["page_cer_relative_reduction_percent"], 60.0)
            self.assertIn("yajn: 60.0", gains_table["caption"])
            self.assertIn("dense: 62.0", gains_table["caption"])
            self.assertTrue(artifacts.markdown_path.exists())


if __name__ == "__main__":
    unittest.main()
