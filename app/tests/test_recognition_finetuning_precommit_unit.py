import shutil
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
from tests.recognition_finetuning_config import get_precommit_hybrid_recognition_gate_config
from tests.recognition_finetuning_experiment import _build_recognition_precommit_dataset_result, _policy_descriptor


class RecognitionFineTuningPrecommitUnitTest(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        tmp_root = TESTS_ROOT / "_tmp_precommit_gate_unit"
        if tmp_root.exists():
            shutil.rmtree(tmp_root)

    def test_precommit_hybrid_recipe_is_exact(self):
        gate_config = get_recognition_precommit_dataset("eval_dataset")
        config = get_precommit_hybrid_recognition_gate_config("eval_dataset")

        self.assertEqual(config.name, gate_config.recognition_dataset_config_name)
        self.assertEqual(config.training_policy, "page_plus_random_history")
        self.assertEqual(int(config.history_sample_line_count), 10)
        self.assertEqual(config.width_policy, "batch_max_pad")
        self.assertEqual(config.oversampling_policy, "none")
        self.assertEqual(config.augmentation_policy, "none")
        self.assertEqual(config.lr_scheduler, "none")
        self.assertEqual(config.optimizer, "adadelta")
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


if __name__ == "__main__":
    unittest.main()
