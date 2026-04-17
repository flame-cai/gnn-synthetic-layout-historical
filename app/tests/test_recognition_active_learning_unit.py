import unittest
import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

TESTS_ROOT = Path(__file__).resolve().parent
APP_ROOT = TESTS_ROOT.parent
REPO_ROOT = APP_ROOT.parent

import sys

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
from recognition.dataset import AlignCollate
from recognition.pagexml_line_dataset import PreparedLineRecord, PreparedPageDataset


def _make_workspace_tmp(name: str) -> Path:
    root = TESTS_ROOT / "_tmp_unit" / name
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    return root


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
        test_dir = root / "finetune_dataset" / "test"
        test_dir.mkdir(parents=True, exist_ok=True)

        image = np.full((16, 32), 200, dtype=np.uint8)
        Image.fromarray(image).save(test_dir / "word_0001.png")
        Image.fromarray(image).save(test_dir / "word_0002.png")

        prepared_page = PreparedPageDataset(
            page_id="p1",
            image_filename="p1.jpg",
            source_xml_path=str(root / "p1.xml"),
            source_image_path=str(root / "p1.jpg"),
            output_root=str(root),
            image_format_dir=str(root / "image-format"),
            finetune_dataset_dir=str(root / "finetune_dataset"),
            gt_path=str(root / "finetune_dataset" / "gt.txt"),
            manifest_path=str(root / "manifest.json"),
            records=[
                PreparedLineRecord(
                    page_id="p1",
                    region_id="r1",
                    region_custom="textbox_label_0",
                    line_id="line_1",
                    line_custom="structure_line_id_1",
                    line_numeric_id=1,
                    text="abc",
                    polygon_points=[[0, 0], [10, 0], [10, 10]],
                    y_center=0.0,
                    x_min=0.0,
                    flat_image_rel_path="test/word_0001.png",
                ),
                PreparedLineRecord(
                    page_id="p1",
                    region_id="r1",
                    region_custom="textbox_label_0",
                    line_id="line_2",
                    line_custom="structure_line_id_2",
                    line_numeric_id=2,
                    text="def",
                    polygon_points=[[0, 0], [10, 0], [10, 10]],
                    y_center=1.0,
                    x_min=0.0,
                    flat_image_rel_path="test/word_0002.png",
                ),
            ],
        )

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
        self.assertEqual(dataset_bundle["train_sample_count"], 5)

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
