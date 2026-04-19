import json
import shutil
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


TESTS_ROOT = Path(__file__).resolve().parent
APP_ROOT = TESTS_ROOT.parent
REPO_ROOT = APP_ROOT.parent

if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recognition.active_learning import FineTuneRunResult
from tests.recognition_finetuning_config import get_page_only_followup_policy_configs
from tests.recognition_finetuning_experiment import _page_only_policy_slug, _run_single_policy_run


def _make_workspace_tmp(name: str) -> Path:
    root = TESTS_ROOT / "_tmp_page_only_unit" / name
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    return root


class RecognitionFineTuningPageOnlyUnitTest(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        tmp_root = TESTS_ROOT / "_tmp_page_only_unit"
        if tmp_root.exists():
            shutil.rmtree(tmp_root)

    def test_page_only_followup_policy_configs_are_exact_and_only_vary_on_optimizer_lr_num_iter(self):
        configs = get_page_only_followup_policy_configs("eval_dataset")

        self.assertEqual(len(configs), 4)
        self.assertEqual(
            [
                (config.optimizer, float(config.training_overrides["lr"]), int(config.training_overrides["num_iter"]))
                for config in configs
            ],
            [
                ("adam", 0.00005, 60),
                ("adam", 0.00001, 200),
                ("adadelta", 0.2, 60),
                ("adadelta", 0.05, 200),
            ],
        )

        baseline = configs[0]
        invariant_fields = (
            "name",
            "layout_type",
            "fine_tune_page_count",
            "eval_page_start_index",
            "eval_page_end_index",
            "training_policy",
            "validation_ratio",
            "split_seed",
            "width_policy",
            "oversampling_policy",
            "augmentation_policy",
            "lr_scheduler",
            "regression_guard_abs",
            "curve_metric",
            "background_plus_rotation_variant_count",
            "shuffle_train_each_epoch",
        )
        baseline_overrides = dict(baseline.training_overrides)

        for config in configs:
            self.assertEqual(config.training_policy, "page_only")
            for field_name in invariant_fields:
                self.assertEqual(getattr(config, field_name), getattr(baseline, field_name))

            overrides = dict(config.training_overrides)
            for key, value in baseline_overrides.items():
                if key in {"lr", "num_iter", "adam"}:
                    continue
                self.assertEqual(overrides[key], value)
            self.assertEqual(overrides["adam"], config.optimizer == "adam")

    def test_page_only_runner_uses_single_page_steps_and_prior_checkpoint(self):
        root = _make_workspace_tmp("sequence")
        images_dir = root / "images"
        labels_dir = root / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        page_ids = ["page_01", "page_02", "page_03", "page_04"]
        for page_id in page_ids:
            (images_dir / f"{page_id}.jpg").write_bytes(b"img")
            (labels_dir / f"{page_id}.xml").write_text("<PcGts/>", encoding="utf-8")

        pretrained_checkpoint = root / "pretrained.pth"
        pretrained_checkpoint.write_bytes(b"checkpoint")

        dataset_config = get_page_only_followup_policy_configs("eval_dataset")[0].with_updates(
            images_dir=images_dir,
            pagexml_dir=labels_dir,
            fine_tune_page_count=3,
            eval_page_start_index=3,
            eval_page_end_index=4,
        )

        prepared_pages = {page_id: SimpleNamespace(page_id=page_id) for page_id in page_ids}
        evaluation_pages = {"page_04": prepared_pages["page_04"]}
        gt_subset_dir = root / "gt_eval_subset"
        gt_subset_dir.mkdir(parents=True, exist_ok=True)
        calls = []

        def fake_finetune(prepared_page_list, base_checkpoint, output_root, step_index, **kwargs):
            training_page_ids = [page.page_id for page in prepared_page_list]
            output_checkpoint = output_root / "training_run" / f"step_{step_index:02d}.pth"
            output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            output_checkpoint.write_bytes(f"step-{step_index}".encode("utf-8"))
            selector_metrics_path = output_root / "selector_metrics.json"
            selector_metrics_path.write_text("{}", encoding="utf-8")
            call = {
                "step_index": step_index,
                "training_page_ids": training_page_ids,
                "base_checkpoint": str(Path(base_checkpoint).resolve()),
                "output_checkpoint": str(output_checkpoint.resolve()),
            }
            calls.append(call)
            return FineTuneRunResult(
                step_index=step_index,
                training_page_id=training_page_ids[-1],
                training_page_ids=training_page_ids,
                base_checkpoint=call["base_checkpoint"],
                output_checkpoint=call["output_checkpoint"],
                experiment_dir=str((output_root / "training_run").resolve()),
                dataset_root=str((output_root / "dataset").resolve()),
                lmdb_root=str((output_root / "lmdb").resolve()),
                train_seconds=0.1,
                selected_best_model="best_accuracy.pth",
                selector_metrics_path=str(selector_metrics_path.resolve()),
                selector_metrics={"winner": "best_accuracy.pth"},
                train_sample_count=1,
                val_sample_count=0,
                validation_ratio=0.0,
                split_seed=42,
                width_policy=kwargs["width_policy"],
                oversampling_policy="none",
                augmentation_policy="none",
                lr_scheduler=kwargs["lr_scheduler"],
                optimizer_name=kwargs["optimizer_name"],
                background_plus_rotation_variant_count=kwargs["background_plus_rotation_variant_count"],
                shuffle_train_each_epoch=kwargs["shuffle_train_each_epoch"],
                logical_train_sample_count=1,
                logical_val_sample_count=0,
                train_materialized_count=1,
                val_materialized_count=0,
                train_variant_labels=["base"],
                val_variant_labels=[],
                dataset_manifest_path=str((output_root / "dataset_manifest.json").resolve()),
                dataset_manifest={"step_index": step_index},
                train_options={"lr": kwargs["lr"], "num_iter": kwargs["num_iter"]},
                training_summary={"step_index": step_index},
            )

        def fake_prediction(checkpoint_path, evaluation_pages, output_root, width_policy):
            output_root.mkdir(parents=True, exist_ok=True)
            return SimpleNamespace(
                prediction_folder=str(output_root.resolve()),
                per_page_seconds={"page_04": 0.01},
                total_seconds=0.01,
                per_line_predictions=[
                    {
                        "page_id": "page_04",
                        "line_id": "line_1",
                        "line_custom": "structure_line_id_1",
                        "gt_length": 10,
                        "predicted_text": "prediction",
                        "edit_distance": 1,
                        "line_cer": 0.1,
                        "confidence": 0.9,
                        "resized_width": 100,
                        "pad_fraction": 0.0,
                        "length_bucket": "short",
                    }
                ],
            )

        def fake_evaluate(pred_folder, gt_folder, method_name, layout_type):
            step_index = int(method_name.rsplit("-", 1)[-1])
            page_cer = 0.5 - (0.05 * step_index)
            return {
                "aggregate_metrics": {
                    "page_cer": page_cer,
                    "line_cer_50": page_cer + 0.01,
                },
                "per_page": [
                    {
                        "filename": "page_04",
                        "page_cer": page_cer,
                        "line_cer_50": page_cer + 0.01,
                        "gt_len": 10,
                        "prediction_found": True,
                    }
                ],
            }

        with patch("tests.recognition_finetuning_experiment.PRETRAINED_OCR_CHECKPOINT", pretrained_checkpoint), patch(
            "tests.recognition_finetuning_experiment.fine_tune_checkpoint_on_pages",
            side_effect=fake_finetune,
        ), patch(
            "tests.recognition_finetuning_experiment.generate_prediction_pagexmls",
            side_effect=fake_prediction,
        ), patch(
            "tests.recognition_finetuning_experiment.evaluate_dataset",
            side_effect=fake_evaluate,
        ):
            result = _run_single_policy_run(
                root / "policy_run",
                dataset_config,
                prepared_pages,
                evaluation_pages,
                gt_subset_dir,
                slug_builder=_page_only_policy_slug,
            )

        self.assertEqual(result["status"], "passed")
        self.assertEqual(len(result["steps"]), 4)
        self.assertEqual([call["training_page_ids"] for call in calls], [["page_01"], ["page_02"], ["page_03"]])
        self.assertTrue(all(step["training_policy"] == "page_only" for step in result["steps"]))
        self.assertEqual(result["steps"][1]["training_page_ids"], ["page_01"])
        self.assertEqual(result["steps"][2]["training_page_ids"], ["page_02"])
        self.assertEqual(result["steps"][3]["training_page_ids"], ["page_03"])
        self.assertEqual([step["train_dataset_page_count"] for step in result["steps"][1:]], [1, 1, 1])
        self.assertEqual(calls[1]["base_checkpoint"], calls[0]["output_checkpoint"])
        self.assertEqual(calls[2]["base_checkpoint"], calls[1]["output_checkpoint"])
        self.assertEqual(result["steps"][2]["base_checkpoint"], result["steps"][1]["output_checkpoint"])
        self.assertEqual(result["steps"][3]["base_checkpoint"], result["steps"][2]["output_checkpoint"])

        fine_tune_metadata = json.loads(result["fine_tune_metadata_path"].read_text(encoding="utf-8"))
        self.assertEqual(len(fine_tune_metadata["runs"]), 3)
        self.assertTrue(all(run["training_policy"] == "page_only" for run in fine_tune_metadata["runs"]))
        self.assertTrue(all(run["train_dataset_page_count"] == 1 for run in fine_tune_metadata["runs"]))

    def test_page_only_policy_slug_distinguishes_micro_learning_rates(self):
        configs = get_page_only_followup_policy_configs("eval_dataset")
        slugs = [_page_only_policy_slug(config) for config in configs]

        self.assertEqual(len(set(slugs)), 4)
        self.assertEqual(
            slugs,
            [
                "wb_on_an_sn_opta_lr000050u",
                "wb_on_an_sn_opta_lr000010u",
                "wb_on_an_sn_optd_lr200000u",
                "wb_on_an_sn_optd_lr050000u",
            ],
        )


if __name__ == "__main__":
    unittest.main()
