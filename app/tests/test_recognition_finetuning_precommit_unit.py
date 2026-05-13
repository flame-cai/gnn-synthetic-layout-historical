import shutil
import sys
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np


TESTS_ROOT = Path(__file__).resolve().parent
APP_ROOT = TESTS_ROOT.parent
REPO_ROOT = APP_ROOT.parent

if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.precommit_gate_config import get_recognition_precommit_dataset
from tests.recognition_finetuning_config import get_precommit_hybrid_recognition_gate_config
from tests.recognition_finetuning_experiment import (
    _build_recognition_precommit_dataset_result,
    _build_recognition_strategy_comparison,
    _config_for_strategy_role,
    _policy_descriptor,
)
from recognition.line_segmentation.strategy_config import (
    get_benchmark_strategy_name,
    get_proposed_strategy_name,
    get_production_strategy_name,
)
from recognition.pagexml_line_dataset import prepare_page_line_dataset


class RecognitionFineTuningPrecommitUnitTest(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        tmp_root = TESTS_ROOT / "_tmp_precommit_gate_unit"
        if tmp_root.exists():
            shutil.rmtree(tmp_root)

    def test_precommit_hybrid_recipe_is_exact(self):
        gate_config = get_recognition_precommit_dataset("eval_dataset")
        config = get_precommit_hybrid_recognition_gate_config("eval_dataset")
        benchmark_strategy = get_benchmark_strategy_name()
        proposed_strategy = get_proposed_strategy_name()

        self.assertEqual(config.name, gate_config.recognition_dataset_config_name)
        self.assertEqual(config.line_geometry_source, "baseline_heatmap")
        self.assertEqual(config.line_segmentation_strategy_name, benchmark_strategy)
        self.assertTrue(config.heatmaps_dir.exists())
        self.assertEqual(float(config.line_segmentation_args["BINARIZE_THRESHOLD"]), 0.5098)
        self.assertEqual(float(config.min_geometry_source_line_coverage), 0.90)
        self.assertEqual(float(config.min_geometry_heatmap_box_assignment_rate), 0.90)
        self.assertEqual(config.training_policy, "page_plus_random_history")
        self.assertEqual(int(config.history_sample_line_count), 10)
        self.assertEqual(config.width_policy, "batch_max_pad")
        self.assertEqual(config.oversampling_policy, "none")
        self.assertEqual(config.augmentation_policy, "none")
        self.assertEqual(config.lr_scheduler, "none")
        self.assertEqual(config.optimizer, "adadelta")
        self.assertEqual(config.sibling_checkpoint_strategy, "page_cer_selector")
        self.assertEqual(float(config.training_overrides["lr"]), 0.2)
        self.assertEqual(int(config.training_overrides["num_iter"]), 60)
        self.assertEqual(config.curve_metric, "early_weighted_page_cer")
        self.assertEqual(float(config.regression_guard_abs), 0.005)
        self.assertEqual(int(config.background_plus_rotation_variant_count), 10)
        self.assertTrue(config.shuffle_train_each_epoch)

        self.assertEqual(gate_config.max_curve_metric_value, 0.26)
        self.assertEqual(gate_config.max_final_page_cer, 0.18)
        self.assertEqual(gate_config.min_first_step_gain, 0.04)
        self.assertTrue(gate_config.regression_guard_warning_only)
        self.assertEqual(gate_config.strategy_ablation.benchmark.role, "benchmark")
        self.assertEqual(gate_config.strategy_ablation.proposed.role, "proposed")
        self.assertEqual(gate_config.strategy_ablation.benchmark.strategy_name, benchmark_strategy)
        self.assertEqual(gate_config.strategy_ablation.proposed.strategy_name, proposed_strategy)
        self.assertEqual(float(gate_config.strategy_ablation.max_allowed_regression_abs), 0.02)

    def test_strategy_role_config_updates_dataset_geometry(self):
        gate_config = get_recognition_precommit_dataset("eval_dataset")
        base_config = get_precommit_hybrid_recognition_gate_config("eval_dataset")
        role_config = gate_config.strategy_ablation.proposed

        role_dataset_config = _config_for_strategy_role(base_config, role_config)

        self.assertEqual(role_dataset_config.line_geometry_source, "baseline_heatmap")
        self.assertEqual(role_dataset_config.line_segmentation_strategy_name, role_config.strategy_name)
        self.assertEqual(role_dataset_config.line_segmentation_args, base_config.line_segmentation_args)

    def test_strategy_comparison_allows_configured_small_regression(self):
        gate_config = get_recognition_precommit_dataset("eval_dataset")
        benchmark_result = {
            "role": "benchmark",
            "strategy_name": "legacy_axis_bound_v1",
            "passed": True,
            "failure_message": "",
            "metrics": {
                "curve_metric_value": 0.220,
                "final_page_cer": 0.140,
                "first_step_gain": 0.050,
            },
        }
        proposed_result = {
            "role": "proposed",
            "strategy_name": "legacy_axis_bound_v1",
            "passed": True,
            "failure_message": "",
            "metrics": {
                "curve_metric_value": 0.224,
                "final_page_cer": 0.144,
                "first_step_gain": 0.046,
            },
        }

        comparison = _build_recognition_strategy_comparison(gate_config, benchmark_result, proposed_result)

        self.assertTrue(comparison["passed"], comparison["failure_message"])
        self.assertEqual(comparison["allowed_regression_abs"], 0.02)
        self.assertTrue(all(item["passed"] for item in comparison["metric_comparisons"]))

    def test_strategy_comparison_fails_beyond_tolerance(self):
        gate_config = get_recognition_precommit_dataset("eval_dataset")
        benchmark_result = {
            "role": "benchmark",
            "strategy_name": "legacy_axis_bound_v1",
            "passed": True,
            "failure_message": "",
            "metrics": {
                "curve_metric_value": 0.220,
                "final_page_cer": 0.140,
                "first_step_gain": 0.050,
            },
        }
        proposed_result = {
            "role": "proposed",
            "strategy_name": "legacy_axis_bound_v1",
            "passed": True,
            "failure_message": "",
            "metrics": {
                "curve_metric_value": 0.250,
                "final_page_cer": 0.170,
                "first_step_gain": 0.020,
            },
        }

        comparison = _build_recognition_strategy_comparison(gate_config, benchmark_result, proposed_result)

        self.assertFalse(comparison["passed"])
        self.assertIn("curve_metric_value", comparison["failure_message"])

    def test_circular_strategy_comparison_blocks_on_primary_only(self):
        gate_config = get_recognition_precommit_dataset("eval_dataset_v2")
        benchmark_result = {
            "role": "benchmark",
            "strategy_name": "legacy_axis_bound_v1",
            "passed": True,
            "failure_message": "",
            "metrics": {
                "curve_metric_value": 0.94,
                "final_page_cer": 0.91,
                "first_step_gain": 0.06,
            },
        }
        proposed_result = {
            "role": "proposed",
            "strategy_name": "local_tangent_band_v1",
            "passed": True,
            "failure_message": "",
            "metrics": {
                "curve_metric_value": 0.18,
                "final_page_cer": 0.16,
                "first_step_gain": 0.01,
            },
        }

        comparison = _build_recognition_strategy_comparison(gate_config, benchmark_result, proposed_result)

        self.assertTrue(comparison["passed"], comparison["failure_message"])
        self.assertEqual(comparison["operator"], "<")
        first_step = next(item for item in comparison["metric_comparisons"] if item["metric_name"] == "first_step_gain")
        self.assertFalse(first_step["passed"])
        self.assertFalse(first_step["blocking"])

    def test_precommit_result_treats_regression_guard_failure_as_warning_only(self):
        dataset_config = get_precommit_hybrid_recognition_gate_config("eval_dataset")
        tmp_root = TESTS_ROOT / "_tmp_precommit_gate_unit" / "warn_only"
        tmp_root.mkdir(parents=True, exist_ok=True)
        policy_result = {
            "status": "passed",
            "failure_message": "",
            "warnings": [],
            "policy_slug": "wb_on_an_hist10_sn_optd_lr200000u",
            "policy": _policy_descriptor(dataset_config),
            "curve_metrics": {
                "curve_metric_name": "early_weighted_page_cer",
                "curve_metric_value": 0.22,
                "regression_guard_abs": 0.005,
                "regression_guard_passed": False,
                "max_regression": 0.006,
                "first_step_gain": 0.05,
                "final_page_cer": 0.15,
            },
            "run_dir": tmp_root / "policy_run",
            "summary_path": tmp_root / "policy_run" / "summary.md",
            "metrics_path": tmp_root / "policy_run" / "metrics.json",
            "curve_metrics_path": tmp_root / "policy_run" / "curve_metrics.json",
            "per_page_csv_path": tmp_root / "policy_run" / "per_page.csv",
            "per_line_csv_path": tmp_root / "policy_run" / "per_line.csv",
            "fine_tune_metadata_path": tmp_root / "policy_run" / "fine_tune_metadata.json",
            "selector_metrics_path": tmp_root / "policy_run" / "selector_metrics.json",
            "plot_path": tmp_root / "policy_run" / "plot.png",
        }

        result = _build_recognition_precommit_dataset_result("eval_dataset", policy_result)

        self.assertEqual(result["status"], "passed")
        self.assertTrue(result["passed"])
        self.assertTrue(result["blocking_thresholds_passed"])
        self.assertFalse(result["curve_metrics"]["regression_guard_passed"])
        self.assertEqual(result["threshold_results"]["curve_metric_value"]["threshold"], 0.26)
        self.assertEqual(result["threshold_results"]["final_page_cer"]["threshold"], 0.18)
        self.assertEqual(result["threshold_results"]["first_step_gain"]["threshold"], 0.04)
        self.assertTrue(any("Regression guard warning only" in warning for warning in result["warnings"]))

    def test_precommit_result_fails_when_blocking_thresholds_fail(self):
        dataset_config = get_precommit_hybrid_recognition_gate_config("eval_dataset")
        tmp_root = TESTS_ROOT / "_tmp_precommit_gate_unit" / "threshold_fail"
        tmp_root.mkdir(parents=True, exist_ok=True)
        policy_result = {
            "status": "passed",
            "failure_message": "",
            "warnings": [],
            "policy_slug": "wb_on_an_hist10_sn_optd_lr200000u",
            "policy": _policy_descriptor(dataset_config),
            "curve_metrics": {
                "curve_metric_name": "early_weighted_page_cer",
                "curve_metric_value": 0.30,
                "regression_guard_abs": 0.005,
                "regression_guard_passed": True,
                "max_regression": 0.001,
                "first_step_gain": 0.03,
                "final_page_cer": 0.15,
            },
            "run_dir": tmp_root / "policy_run",
            "summary_path": tmp_root / "policy_run" / "summary.md",
            "metrics_path": tmp_root / "policy_run" / "metrics.json",
            "curve_metrics_path": tmp_root / "policy_run" / "curve_metrics.json",
            "per_page_csv_path": tmp_root / "policy_run" / "per_page.csv",
            "per_line_csv_path": tmp_root / "policy_run" / "per_line.csv",
            "fine_tune_metadata_path": tmp_root / "policy_run" / "fine_tune_metadata.json",
            "selector_metrics_path": tmp_root / "policy_run" / "selector_metrics.json",
            "plot_path": tmp_root / "policy_run" / "plot.png",
        }

        result = _build_recognition_precommit_dataset_result("eval_dataset", policy_result)

        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["passed"])
        self.assertFalse(result["blocking_thresholds_passed"])
        self.assertIn("early_weighted_page_cer=0.3", result["failure_message"])
        self.assertIn("first_step_gain=0.03", result["failure_message"])

    def test_baseline_heatmap_geometry_prepares_without_pagexml_coords(self):
        tmp_root = TESTS_ROOT / "_tmp_precommit_gate_unit" / "baseline_heatmap_geometry"
        if tmp_root.exists():
            shutil.rmtree(tmp_root)
        tmp_root.mkdir(parents=True, exist_ok=True)

        page_id = "unit_page"
        image_path = tmp_root / f"{page_id}.jpg"
        heatmap_path = tmp_root / f"{page_id}_heatmap.jpg"
        xml_path = tmp_root / f"{page_id}.xml"
        output_root = tmp_root / "prepared"

        image = np.full((64, 96), 240, dtype=np.uint8)
        image[28:36, 24:72] = 20
        heatmap = np.zeros((32, 48), dtype=np.uint8)
        heatmap[14:18, 12:36] = 255
        cv2.imwrite(str(image_path), image)
        cv2.imwrite(str(heatmap_path), heatmap)

        ns = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
        ET.register_namespace("", ns)
        xml_path.write_text(
            f"""<?xml version="1.0" encoding="UTF-8"?>
<PcGts xmlns="{ns}">
  <Page imageFilename="{page_id}.jpg" imageWidth="96" imageHeight="64">
    <TextRegion id="region_0" custom="textbox_label_0">
      <TextLine id="region_0_line_0" custom="structure_line_id_7">
        <TextEquiv><Unicode>test</Unicode></TextEquiv>
        <Baseline points="24,36 72,36" />
      </TextLine>
    </TextRegion>
  </Page>
</PcGts>
""",
            encoding="utf-8",
        )

        prepared = prepare_page_line_dataset(
            xml_path,
            image_path,
            output_root,
            heatmap_path=heatmap_path,
            geometry_source="baseline_heatmap",
        )

        self.assertEqual(prepared.geometry_source, "baseline_heatmap")
        self.assertEqual(prepared.line_segmentation_strategy_name, get_production_strategy_name())
        self.assertTrue(Path(prepared.line_segmentation_metadata_path).exists())
        self.assertEqual(len(prepared.records), 1)
        xs = [point[0] for point in prepared.records[0].polygon_points]
        ys = [point[1] for point in prepared.records[0].polygon_points]
        self.assertGreater(max(xs), 70)
        self.assertGreater(max(ys), 40)
        self.assertGreater(prepared.geometry_summary["assigned_box_count"], 0)
        self.assertEqual(prepared.geometry_summary["source_line_coverage"], 1.0)
        self.assertEqual(prepared.geometry_summary["heatmap_box_assignment_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
