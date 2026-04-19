import json
import shutil
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image


TESTS_ROOT = Path(__file__).resolve().parent
APP_ROOT = TESTS_ROOT.parent
REPO_ROOT = APP_ROOT.parent

if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recognition.active_learning import FineTuneRunResult, prepare_incremental_finetune_dataset
from recognition.pagexml_line_dataset import PreparedLineRecord, PreparedPageDataset
from tests.recognition_finetuning_config import get_page_plus_random_history_policy_configs
from tests.recognition_finetuning_experiment import _page_plus_history_policy_slug, _run_single_policy_run


def _make_workspace_tmp(name: str) -> Path:
    root = TESTS_ROOT / "_tmp_page_history_unit" / name
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _make_prepared_page(root: Path, page_id: str, line_count: int) -> PreparedPageDataset:
    dataset_dir = root / page_id / "finetune_dataset"
    test_dir = dataset_dir / "test"
    test_dir.mkdir(parents=True, exist_ok=True)
    image = np.full((16, 32), 200, dtype=np.uint8)

    records = []
    for line_index in range(1, line_count + 1):
        filename = f"word_{line_index:04d}.png"
        Image.fromarray(image).save(test_dir / filename)
        records.append(
            PreparedLineRecord(
                page_id=page_id,
                region_id="r1",
                region_custom="textbox_label_0",
                line_id=f"line_{line_index}",
                line_custom=f"structure_line_id_{line_index}",
                line_numeric_id=line_index,
                text=f"{page_id}_{line_index}",
                polygon_points=[[0, 0], [10, 0], [10, 10]],
                y_center=float(line_index),
                x_min=0.0,
                flat_image_rel_path=f"test/{filename}",
            )
        )

    return PreparedPageDataset(
        page_id=page_id,
        image_filename=f"{page_id}.jpg",
        source_xml_path=str(root / f"{page_id}.xml"),
        source_image_path=str(root / f"{page_id}.jpg"),
        output_root=str(root / page_id),
        image_format_dir=str(root / page_id / "image-format"),
        finetune_dataset_dir=str(dataset_dir),
        gt_path=str(dataset_dir / "gt.txt"),
        manifest_path=str(root / page_id / "manifest.json"),
        records=records,
    )


class RecognitionFineTuningPagePlusHistoryUnitTest(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        tmp_root = TESTS_ROOT / "_tmp_page_history_unit"
        if tmp_root.exists():
            shutil.rmtree(tmp_root)

    def test_page_plus_history_policy_configs_are_exact(self):
        configs = get_page_plus_random_history_policy_configs("eval_dataset")

        self.assertEqual(len(configs), 2)
        self.assertEqual(
            [
                (
                    config.training_policy,
                    int(config.history_sample_line_count),
                    config.optimizer,
                    float(config.training_overrides["lr"]),
                    int(config.training_overrides["num_iter"]),
                )
                for config in configs
            ],
            [
                ("page_plus_random_history", 10, "adam", 0.00005, 60),
                ("page_plus_random_history", 10, "adadelta", 0.2, 60),
            ],
        )

    def test_prepare_incremental_dataset_uses_current_page_plus_ten_history_lines_and_shuffles(self):
        root = _make_workspace_tmp("dataset_mix")
        current_page = _make_prepared_page(root, "page_03", 4)
        history_pages = [
            _make_prepared_page(root, "page_01", 6),
            _make_prepared_page(root, "page_02", 6),
        ]

        dataset_bundle = prepare_incremental_finetune_dataset(
            [current_page],
            root / "materialized",
            base_checkpoint=root / "dummy.pth",
            validation_ratio=0.0,
            oversampling_policy="none",
            augmentation_policy="none",
            history_source_pages=history_pages,
            history_sample_line_count=10,
            split_seed=17,
            optimizer_name="adam",
            shuffle_train_each_epoch=True,
        )

        manifest = dataset_bundle["manifest"]
        self.assertEqual(manifest["current_page_ids"], ["page_03"])
        self.assertEqual(manifest["history_source_page_ids"], ["page_01", "page_02"])
        self.assertEqual(manifest["current_page_line_count"], 4)
        self.assertEqual(manifest["history_sample_requested_count"], 10)
        self.assertEqual(manifest["history_sample_line_count"], 10)
        self.assertEqual(manifest["num_samples"], 14)
        self.assertEqual(dataset_bundle["logical_train_sample_count"], 14)
        self.assertTrue(set(manifest["history_sample_page_ids"]).issubset({"page_01", "page_02"}))

        current_target_names = [
            f"{current_page.page_id}__{Path(record.flat_image_rel_path).name}"
            for record in current_page.records
        ]
        selected_order = current_target_names + [
            sample_ref["target_name"] for sample_ref in manifest["history_sample_line_refs"]
        ]
        train_gt_order = [
            Path(line.split("\t", 1)[0]).name
            for line in dataset_bundle["train_gt_path"].read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(len(train_gt_order), 14)
        self.assertNotEqual(train_gt_order, selected_order)

    def test_prepare_incremental_dataset_handles_first_page_without_history(self):
        root = _make_workspace_tmp("dataset_edge")
        current_page = _make_prepared_page(root, "page_01", 3)

        dataset_bundle = prepare_incremental_finetune_dataset(
            [current_page],
            root / "materialized",
            base_checkpoint=root / "dummy.pth",
            validation_ratio=0.0,
            oversampling_policy="none",
            augmentation_policy="none",
            history_source_pages=[],
            history_sample_line_count=10,
            split_seed=17,
            optimizer_name="adadelta",
            shuffle_train_each_epoch=True,
        )

        manifest = dataset_bundle["manifest"]
        self.assertEqual(manifest["page_ids"], ["page_01"])
        self.assertEqual(manifest["history_source_page_ids"], [])
        self.assertEqual(manifest["history_sample_line_count"], 0)
        self.assertEqual(manifest["history_sample_page_ids"], [])
        self.assertEqual(manifest["num_samples"], 3)

    def test_runner_passes_previous_pages_as_history_pool(self):
        root = _make_workspace_tmp("runner_sequence")
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

        dataset_config = get_page_plus_random_history_policy_configs("eval_dataset")[0].with_updates(
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

        def fake_finetune(
            prepared_page_list,
            base_checkpoint,
            output_root,
            step_index,
            history_source_pages=None,
            history_sample_line_count=0,
            **kwargs,
        ):
            current_page_id = prepared_page_list[-1].page_id
            history_page_ids = [page.page_id for page in (history_source_pages or [])]
            sampled_history_page_ids = history_page_ids[: min(len(history_page_ids), 2)]
            dataset_page_ids = [current_page_id] + sampled_history_page_ids
            history_line_count = 0 if step_index == 1 else min(int(history_sample_line_count), 10)

            output_checkpoint = output_root / "training_run" / f"step_{step_index:02d}.pth"
            output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            output_checkpoint.write_bytes(f"step-{step_index}".encode("utf-8"))
            selector_metrics_path = output_root / "selector_metrics.json"
            selector_metrics_path.write_text("{}", encoding="utf-8")
            calls.append(
                {
                    "step_index": step_index,
                    "current_page_id": current_page_id,
                    "history_source_page_ids": history_page_ids,
                    "history_sample_line_count": history_sample_line_count,
                }
            )

            return FineTuneRunResult(
                step_index=step_index,
                training_page_id=current_page_id,
                training_page_ids=dataset_page_ids,
                base_checkpoint=str(Path(base_checkpoint).resolve()),
                output_checkpoint=str(output_checkpoint.resolve()),
                experiment_dir=str((output_root / "training_run").resolve()),
                dataset_root=str((output_root / "dataset").resolve()),
                lmdb_root=str((output_root / "lmdb").resolve()),
                train_seconds=0.1,
                selected_best_model="best_accuracy.pth",
                selector_metrics_path=str(selector_metrics_path.resolve()),
                selector_metrics={"winner": "best_accuracy.pth"},
                train_sample_count=12,
                val_sample_count=12,
                validation_ratio=0.0,
                split_seed=42,
                width_policy=kwargs["width_policy"],
                oversampling_policy="none",
                augmentation_policy="none",
                lr_scheduler=kwargs["lr_scheduler"],
                optimizer_name=kwargs["optimizer_name"],
                background_plus_rotation_variant_count=kwargs["background_plus_rotation_variant_count"],
                shuffle_train_each_epoch=kwargs["shuffle_train_each_epoch"],
                logical_train_sample_count=12,
                logical_val_sample_count=12,
                train_materialized_count=12,
                val_materialized_count=12,
                train_variant_labels=["base"],
                val_variant_labels=["base"],
                dataset_manifest_path=str((output_root / "dataset_manifest.json").resolve()),
                dataset_manifest={
                    "page_ids": dataset_page_ids,
                    "current_page_ids": [current_page_id],
                    "current_page_line_count": 4,
                    "history_source_page_ids": history_page_ids,
                    "history_sample_line_count": history_line_count,
                    "history_sample_page_ids": sampled_history_page_ids,
                },
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
                slug_builder=_page_plus_history_policy_slug,
            )

        self.assertEqual(result["status"], "passed")
        self.assertEqual(
            [call["history_source_page_ids"] for call in calls],
            [[], ["page_01"], ["page_01", "page_02"]],
        )
        self.assertEqual(result["steps"][1]["train_dataset_history_line_count"], 0)
        self.assertEqual(result["steps"][2]["history_source_page_ids"], ["page_01"])
        self.assertEqual(result["steps"][3]["history_source_page_ids"], ["page_01", "page_02"])
        self.assertEqual(result["steps"][2]["train_dataset_history_line_count"], 10)
        self.assertEqual(result["steps"][3]["train_dataset_history_line_count"], 10)

        fine_tune_metadata = json.loads(result["fine_tune_metadata_path"].read_text(encoding="utf-8"))
        self.assertTrue(all(run["training_policy"] == "page_plus_random_history" for run in fine_tune_metadata["runs"]))
        self.assertEqual(fine_tune_metadata["runs"][0]["train_dataset_history_line_count"], 0)
        self.assertEqual(fine_tune_metadata["runs"][1]["train_dataset_history_line_count"], 10)


if __name__ == "__main__":
    unittest.main()
