import shutil
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

TESTS_ROOT = Path(__file__).resolve().parent
APP_ROOT = TESTS_ROOT.parent
REPO_ROOT = APP_ROOT.parent

if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recognition.active_learning import (
    RecognitionInferenceResult,
    _estimate_background_fill,
    _foreground_mask,
    _rotated_bbox_fits,
    apply_ocr_training_augmentation,
    prepare_incremental_finetune_dataset,
    select_best_sibling_checkpoint,
)
from recognition.dataset import AlignCollate, Batch_Balanced_Dataset
from recognition.lmdb_tools import create_lmdb_dataset
from recognition.pagexml_line_dataset import PreparedLineRecord, PreparedPageDataset
from tests.recognition_finetuning_config import get_dataset_config
from tests.recognition_finetuning_experiment import _policy_slug, run_recognition_finetuning_experiment


def _make_workspace_tmp(name: str) -> Path:
    root = TESTS_ROOT / "_tmp_unit" / name
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _make_prepared_page(root: Path, page_id: str, line_specs: list[tuple[str, str]]) -> PreparedPageDataset:
    dataset_dir = root / page_id / "finetune_dataset"
    test_dir = dataset_dir / "test"
    test_dir.mkdir(parents=True, exist_ok=True)
    image = np.full((16, 32), 200, dtype=np.uint8)

    records = []
    for line_index, (line_suffix, text) in enumerate(line_specs, start=1):
        filename = f"word_{line_suffix}.png"
        Image.fromarray(image).save(test_dir / filename)
        records.append(
            PreparedLineRecord(
                page_id=page_id,
                region_id="r1",
                region_custom="textbox_label_0",
                line_id=f"line_{line_index}",
                line_custom=f"structure_line_id_{line_index}",
                line_numeric_id=line_index,
                text=text,
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


def _build_batch_balanced_dataset(root: Path, labels: list[str]) -> Batch_Balanced_Dataset:
    source_root = root / "source"
    source_root.mkdir(parents=True, exist_ok=True)
    gt_lines = []
    for index, label in enumerate(labels, start=1):
        image_path = source_root / f"sample_{index:02d}.png"
        Image.fromarray(np.full((16, 32), 200 - index, dtype=np.uint8)).save(image_path)
        gt_lines.append(f"{image_path.name}\t{label}")

    gt_path = source_root / "gt.txt"
    gt_path.write_text("\n".join(gt_lines) + "\n", encoding="utf-8")

    lmdb_root = root / "lmdb"
    create_lmdb_dataset(source_root, gt_path, lmdb_root, check_valid=False)

    opt = SimpleNamespace(
        experiment_dir=str(root / "experiment"),
        exp_name="unit_shuffle",
        train_data=str(lmdb_root),
        select_data=["/"],
        batch_ratio=["1"],
        batch_size=1,
        total_data_usage_ratio="1.0",
        workers=0,
        imgH=32,
        imgW=2000,
        PAD=True,
        width_policy="batch_max_pad",
        manualSeed=1111,
        shuffle_train_each_epoch=True,
        data_filtering_off=True,
        rgb=False,
        sensitive=False,
        character="abcdefghijklmnopqrstuvwxyz",
        batch_max_length=25,
    )
    return Batch_Balanced_Dataset(opt)


class RecognitionActiveLearningUnitTest(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        tmp_root = TESTS_ROOT / "_tmp_unit"
        if tmp_root.exists():
            shutil.rmtree(tmp_root)

    def test_selector_prefers_lower_page_cer_checkpoint(self):
        cases = [
            ("best_accuracy.pth", 0.21, 0.34),
            ("best_norm_ED.pth", 0.34, 0.21),
        ]
        for expected_name, accuracy_cer, norm_ed_cer in cases:
            with self.subTest(expected_name=expected_name):
                experiment_dir = _make_workspace_tmp(f"selector_{expected_name.replace('.', '_')}")
                (experiment_dir / "best_accuracy.pth").write_bytes(b"acc")
                (experiment_dir / "best_norm_ED.pth").write_bytes(b"ned")

                def fake_inference(checkpoint_path, prepared_pages, output_root=None, **kwargs):
                    filename = Path(checkpoint_path).name
                    page_cer = accuracy_cer if filename == "best_accuracy.pth" else norm_ed_cer
                    return RecognitionInferenceResult(
                        checkpoint_path=str(checkpoint_path),
                        prediction_folder=None,
                        per_page_seconds={},
                        total_seconds=0.0,
                        aggregate_metrics={"page_cer": page_cer, "line_cer": page_cer},
                        per_page_metrics=[{"page_id": "p1", "page_cer": page_cer, "line_cer": page_cer}],
                        per_line_predictions=[],
                    )

                with patch("recognition.active_learning.run_checkpoint_on_prepared_pages", side_effect=fake_inference):
                    selected_checkpoint, selected_name, selector_metrics = select_best_sibling_checkpoint(
                        experiment_dir,
                        prepared_pages=[],
                    )

                self.assertEqual(selected_name, expected_name)
                self.assertEqual(selected_checkpoint.name, expected_name)
                self.assertIn("lower than", selector_metrics["reason"])

    def test_align_collate_width_policy_uses_global_or_batch_max_padding(self):
        image_a = Image.fromarray(np.full((20, 40), 200, dtype=np.uint8), mode="L")
        image_b = Image.fromarray(np.full((20, 80), 200, dtype=np.uint8), mode="L")
        batch = [(image_a, "a"), (image_b, "b")]

        global_collate = AlignCollate(
            imgH=50,
            imgW=2000,
            keep_ratio_with_pad=True,
            width_policy="global_2000_pad",
            return_metadata=True,
        )
        batch_collate = AlignCollate(
            imgH=50,
            imgW=2000,
            keep_ratio_with_pad=True,
            width_policy="batch_max_pad",
            return_metadata=True,
        )

        global_tensors, _, global_meta = global_collate(batch)
        batch_tensors, _, batch_meta = batch_collate(batch)

        self.assertEqual(global_tensors.shape[-1], 2000)
        self.assertEqual(batch_tensors.shape[-1], 200)
        self.assertEqual([item["resized_width"] for item in global_meta], [100, 200])
        self.assertEqual([item["padded_width"] for item in batch_meta], [200, 200])

    def test_cer_weighted_oversampling_replication_is_capped(self):
        root = _make_workspace_tmp("oversampling")
        prepared_page = _make_prepared_page(root, "p1", [("0001", "abc"), ("0002", "def")])

        with patch(
            "recognition.active_learning.score_page_difficulty",
            return_value={
                "structure_line_id_1": {"line_id": "line_1", "line_cer": 0.0, "replication": 1},
                "structure_line_id_2": {"line_id": "line_2", "line_cer": 1.8, "replication": 4},
            },
        ):
            dataset_bundle = prepare_incremental_finetune_dataset(
                [prepared_page],
                root / "materialized",
                base_checkpoint=root / "dummy.pth",
                validation_ratio=0.0,
                oversampling_policy="cer_weighted",
                augmentation_policy="none",
            )

        self.assertEqual(dataset_bundle["manifest"]["replication_lookup"]["p1__word_0001.png"], 1)
        self.assertEqual(dataset_bundle["manifest"]["replication_lookup"]["p1__word_0002.png"], 4)
        self.assertEqual(dataset_bundle["logical_train_sample_count"], 2)
        self.assertEqual(dataset_bundle["train_sample_count"], 5)
        self.assertEqual(dataset_bundle["train_materialized_count"], 5)

    def test_background_plus_rotation_materializes_ten_extra_train_variants_only(self):
        root = _make_workspace_tmp("bgrot_variants")
        prepared_page = _make_prepared_page(root, "p1", [("0001", "abc")])

        dataset_bundle = prepare_incremental_finetune_dataset(
            [prepared_page],
            root / "materialized",
            base_checkpoint=root / "dummy.pth",
            validation_ratio=0.0,
            oversampling_policy="none",
            augmentation_policy="background_plus_rotation",
            background_plus_rotation_variant_count=10,
            optimizer_name="adam",
            shuffle_train_each_epoch=True,
        )

        expected_variant_labels = ["base"] + [f"bgrot{index:02d}" for index in range(1, 11)]
        self.assertEqual(dataset_bundle["manifest"]["background_plus_rotation_variant_count"], 10)
        self.assertEqual(dataset_bundle["manifest"]["optimizer_name"], "adam")
        self.assertTrue(dataset_bundle["manifest"]["shuffle_train_each_epoch"])
        self.assertEqual(dataset_bundle["train_variant_labels"], expected_variant_labels)
        self.assertEqual(dataset_bundle["val_variant_labels"], ["base"])
        self.assertEqual(dataset_bundle["train_materialized_count"], 11)
        self.assertEqual(dataset_bundle["val_materialized_count"], 1)

        train_rows = dataset_bundle["manifest"]["train_variant_counts"]
        self.assertEqual(train_rows["base"], 1)
        self.assertEqual(train_rows["bgrot10"], 1)

    def test_focused_policy_slug_distinguishes_optimizer_and_lr0001(self):
        dataset_config = get_dataset_config("eval_dataset")
        structural_config = dataset_config.with_structural_policy(dataset_config.focused_structural_policies[0])

        slugs = {
            _policy_slug(structural_config.with_optimizer(optimizer).with_learning_rate(lr))
            for optimizer in ("adadelta", "adam")
            for lr in (0.001, 0.01, 0.2, 0.5)
        }

        self.assertEqual(len(slugs), 8)
        self.assertIn("wb_oc_ar_sn_optd_lr0001", slugs)
        self.assertIn("wb_oc_ar_sn_opta_lr0001", slugs)
        self.assertIn("wb_oc_ar_sn_optd_lr0010", slugs)

    def test_focused_runner_returns_matrix_results_even_when_some_policies_fail(self):
        root = _make_workspace_tmp("focused_runner")
        dataset_config = get_dataset_config("eval_dataset")
        passed_config = dataset_config.with_structural_policy(dataset_config.focused_structural_policies[2]).with_optimizer(
            "adadelta"
        ).with_learning_rate(0.2)
        failed_config = dataset_config.with_structural_policy(dataset_config.focused_structural_policies[0]).with_optimizer(
            "adam"
        ).with_learning_rate(0.001)
        policy_matrix = [passed_config, failed_config]

        def fake_prepare(_dataset_config):
            run_dir = root / "study"
            run_dir.mkdir(parents=True, exist_ok=True)
            return run_dir, {}, {}, run_dir

        def fake_run(run_dir, policy_config, prepared_pages, evaluation_pages, gt_subset_dir, slug_builder):
            policy_slug = slug_builder(policy_config)
            status = "passed" if policy_config.optimizer == "adadelta" else "failed"
            plot_path = run_dir / "plot.png"
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            plot_path.write_text("plot", encoding="utf-8")
            return {
                "status": status,
                "failure_message": "" if status == "passed" else "guard failed",
                "run_dir": run_dir,
                "summary_path": run_dir / "summary.md",
                "metrics_path": run_dir / "metrics.json",
                "curve_metrics_path": run_dir / "curve_metrics.json",
                "per_page_csv_path": run_dir / "per_page.csv",
                "per_line_csv_path": run_dir / "per_line.csv",
                "fine_tune_metadata_path": run_dir / "fine_tune_metadata.json",
                "selector_metrics_path": run_dir / "selector_metrics.json",
                "plot_path": plot_path,
                "steps": [{}] * 10,
                "curve_metrics": {
                    "curve_metric_name": "early_weighted_page_cer",
                    "curve_metric_value": 0.2 if status == "passed" else 0.5,
                    "regression_guard_abs": 0.005,
                    "regression_guard_passed": status == "passed",
                    "max_regression": 0.0 if status == "passed" else 0.1,
                    "first_step_gain": 0.05 if status == "passed" else -0.2,
                    "final_page_cer": 0.15 if status == "passed" else 0.6,
                },
                "policy": {
                    "training_policy": policy_config.training_policy,
                    "history_sample_line_count": int(policy_config.history_sample_line_count),
                    "width_policy": policy_config.width_policy,
                    "oversampling_policy": policy_config.oversampling_policy,
                    "augmentation_policy": policy_config.augmentation_policy,
                    "lr_scheduler": policy_config.lr_scheduler,
                    "optimizer": policy_config.optimizer,
                    "lr": float(policy_config.training_overrides["lr"]),
                    "num_iter": int(policy_config.training_overrides["num_iter"]),
                    "curve_metric": policy_config.curve_metric,
                    "regression_guard_abs": policy_config.regression_guard_abs,
                    "background_plus_rotation_variant_count": policy_config.background_plus_rotation_variant_count,
                    "shuffle_train_each_epoch": policy_config.shuffle_train_each_epoch,
                },
                "policy_slug": policy_slug,
            }

        with patch("tests.recognition_finetuning_experiment._prepare_study_inputs", side_effect=fake_prepare), patch(
            "tests.recognition_finetuning_experiment._build_focused_policy_matrix",
            return_value=policy_matrix,
        ), patch(
            "tests.recognition_finetuning_experiment._run_single_policy_run",
            side_effect=fake_run,
        ), patch("tests.recognition_finetuning_experiment._copy_latest_artifacts", return_value=None):
            result = run_recognition_finetuning_experiment(dataset_name="eval_dataset")

        self.assertEqual(len(result["policy_runs"]), 2)
        self.assertEqual(result["winning_policy"], _policy_slug(passed_config))
        self.assertEqual(result["failed_policy_runs"], [_policy_slug(failed_config)])
        self.assertEqual(
            result["winning_policies_by_metric"]["primary_curve_metric"]["policy_slug"],
            _policy_slug(passed_config),
        )

    def test_loader_shuffle_contract_recreates_iterator_with_seeded_generator(self):
        root = _make_workspace_tmp("loader_shuffle")
        dataset = _build_batch_balanced_dataset(root, ["a", "b", "c", "d"])

        first_pass = []
        for _ in range(4):
            _, labels = dataset.get_batch()
            first_pass.extend(labels)

        second_pass = []
        for _ in range(4):
            _, labels = dataset.get_batch()
            second_pass.extend(labels)

        self.assertEqual(sorted(first_pass), sorted(second_pass))
        self.assertNotEqual(first_pass, second_pass)
        self.assertTrue(dataset.train_loader_metadata[0]["shuffle"])
        self.assertEqual(dataset.train_loader_metadata[0]["generator_seed"], 1111)
        self.assertIsInstance(dataset.train_loader_generators[0], torch.Generator)
        self.assertGreaterEqual(dataset.iterator_recreation_counts[0], 1)

    def test_augmentation_keeps_shape_and_non_empty_foreground(self):
        image = np.full((40, 80), 200, dtype=np.uint8)
        image[10:30, 20:60] = 30

        augmented, metadata = apply_ocr_training_augmentation(
            image,
            augmentation_policy="background_plus_rotation",
            seed=7,
        )

        background_fill = _estimate_background_fill(image)
        original_foreground = _foreground_mask(image, background_fill)
        augmented_background = int(np.clip(background_fill + metadata["background_delta"], 0, 255))

        self.assertEqual(augmented.shape, image.shape)
        self.assertTrue(_foreground_mask(augmented, augmented_background).any())
        self.assertLessEqual(abs(metadata["applied_rotation_degrees"]), 5.0)
        self.assertTrue(_rotated_bbox_fits(original_foreground, metadata["applied_rotation_degrees"]))


if __name__ == "__main__":
    unittest.main()
