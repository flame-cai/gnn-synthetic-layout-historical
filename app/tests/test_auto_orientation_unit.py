from __future__ import annotations

import shutil
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import torch
from lxml import etree as ET
from PIL import Image


TESTS_ROOT = Path(__file__).resolve().parent
APP_ROOT = TESTS_ROOT.parent
REPO_ROOT = APP_ROOT.parent

if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recognition.active_learning import run_checkpoint_on_prepared_pages
from recognition.auto_orientation import (
    ROTATE_180_TRANSFORM,
    auto_orientation_custom_metadata,
    auto_orientation_transform_from_custom,
    has_explicit_reading_direction_annotation,
    select_orientation_from_predictions,
    should_auto_orient_from_predictions,
)
from recognition.pagexml_line_dataset import (
    PreparedLineRecord,
    PreparedPageDataset,
    prepare_page_line_dataset,
)
from recognition.recognize_manuscript_text_v2_pretrained import process_page_xml


WRONG_ORIENTATION_TEXT = "।le2tiseekeuk2utelr2aLंषg१२I2R2lr2titlःtLgए२२१८२"
CORRECT_ORIENTATION_TEXT = "?ईउऊऋऋलृलयपेओऔअअःकखगघऊचखजकीटठडषणतथदधनयफबभमयरसवशषसःअआ"


def _crop_metadata(line_kind="curved_open", reading_direction=None):
    annotation = (
        {"reading_direction": reading_direction}
        if reading_direction is not None
        else None
    )
    return {
        "used_unwrap": True,
        "crop_model": "local_polygon_stable_unwrap",
        "topology": {
            "line_kind": line_kind,
            "reading_direction": reading_direction,
        },
        "orientation": {
            "candidate_transforms": ["identity", "rotate_180"],
            "reading_direction": reading_direction,
        },
        "strategy_line_metadata": {
            "line_kind": line_kind,
            "reading_direction_annotation": annotation,
        },
    }


class _FakeModel:
    def __call__(self, image, text_for_pred, is_train=False):
        return torch.zeros((image.size(0), 1, 2), dtype=torch.float32, device=image.device)


class _FakeConverter:
    def decode(self, preds_index, preds_size):
        values = [WRONG_ORIENTATION_TEXT, CORRECT_ORIENTATION_TEXT]
        return values[: preds_index.size(0)]


class AutoOrientationUnitTest(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        root = TESTS_ROOT / "_tmp_auto_orientation_unit"
        if root.exists():
            shutil.rmtree(root)

    def test_prediction_text_selects_devanagari_orientation_without_confidence(self):
        selection = select_orientation_from_predictions(
            {
                "identity": WRONG_ORIENTATION_TEXT,
                "rotate_180": CORRECT_ORIENTATION_TEXT,
            }
        )

        self.assertEqual(selection.selected_transform, ROTATE_180_TRANSFORM)
        self.assertEqual(selection.selected_text, CORRECT_ORIENTATION_TEXT)
        metadata = selection.to_metadata()
        self.assertFalse(metadata["uses_model_confidence"])
        self.assertGreater(
            metadata["candidates"]["rotate_180"]["net_devanagari_evidence"],
            metadata["candidates"]["identity"]["net_devanagari_evidence"],
        )

    def test_pagexml_custom_metadata_parses_and_preserves_only_orientation_fields(self):
        custom = (
            "confidences:0.9;auto_orientation_model:model_v1;"
            "auto_orientation_transform:rotate_180;auto_orientation_reason:test"
        )
        self.assertEqual(
            auto_orientation_transform_from_custom(custom),
            ROTATE_180_TRANSFORM,
        )
        preserved = auto_orientation_custom_metadata(custom)
        self.assertNotIn("confidences", preserved)
        self.assertIn("auto_orientation_transform:rotate_180", preserved)

    def test_equal_predictions_preserve_identity(self):
        selection = select_orientation_from_predictions(
            {"identity": "कर्म", "rotate_180": "कर्म"}
        )
        self.assertEqual(selection.selected_transform, "identity")

    def test_empty_rotated_prediction_cannot_replace_identity(self):
        selection = select_orientation_from_predictions(
            {"identity": WRONG_ORIENTATION_TEXT, "rotate_180": ""}
        )
        self.assertEqual(selection.selected_transform, "identity")

    def test_only_unannotated_open_or_closed_curves_are_eligible(self):
        self.assertTrue(should_auto_orient_from_predictions(_crop_metadata("curved_open")))
        self.assertTrue(should_auto_orient_from_predictions(_crop_metadata("closed_circular")))
        self.assertFalse(should_auto_orient_from_predictions(_crop_metadata("horizontal_straight")))
        self.assertFalse(
            should_auto_orient_from_predictions(
                _crop_metadata("curved_open", reading_direction=[1.0, 0.0])
            )
        )
        masked = _crop_metadata("curved_open")
        masked["used_unwrap"] = False
        self.assertFalse(should_auto_orient_from_predictions(masked))

    def test_production_page_ocr_selects_rotated_decoded_text(self):
        root = TESTS_ROOT / "_tmp_auto_orientation_unit" / "production"
        root.mkdir(parents=True, exist_ok=True)
        xml_path = root / "page.xml"
        page_root = ET.fromstring(
            b"<PcGts><Page><TextRegion><TextLine id='line_1'/></TextRegion></Page></PcGts>"
        )
        tree = ET.ElementTree(page_root)
        line_elem = page_root.find(".//TextLine")
        crop_metadata = _crop_metadata("curved_open")
        batch_data = [
            (
                Image.fromarray(np.arange(800, dtype=np.uint8).reshape(20, 40)),
                (line_elem, "region_1", "line_1", crop_metadata),
            )
        ]
        config = SimpleNamespace(
            imgH=20,
            imgW=40,
            PAD=True,
            batch_size=2,
            workers=0,
            batch_max_length=100,
            rgb=False,
        )

        with patch(
            "recognition.recognize_manuscript_text_v2_pretrained._extract_ocr_line_crops_with_tree",
            return_value=(tree, page_root, "", batch_data),
        ):
            result = process_page_xml(
                xml_path,
                [],
                _FakeModel(),
                _FakeConverter(),
                config,
                torch.device("cpu"),
            )

        self.assertEqual(result["auto_orientation_line_count"], 1)
        self.assertEqual(result["auto_orientation_rotated_line_count"], 1)
        self.assertEqual(line_elem.findtext("TextEquiv/Unicode"), CORRECT_ORIENTATION_TEXT)
        self.assertIn("auto_orientation_transform:rotate_180", line_elem.find("TextEquiv").get("custom"))

    def test_explicit_annotation_keeps_single_orientation_ocr_path(self):
        root = TESTS_ROOT / "_tmp_auto_orientation_unit" / "annotated_production"
        root.mkdir(parents=True, exist_ok=True)
        xml_path = root / "page.xml"
        page_root = ET.fromstring(
            b"<PcGts><Page><TextRegion><TextLine id='line_1'/></TextRegion></Page></PcGts>"
        )
        tree = ET.ElementTree(page_root)
        line_elem = page_root.find(".//TextLine")
        crop_metadata = _crop_metadata("curved_open", reading_direction=[1.0, 0.0])
        batch_data = [
            (
                Image.fromarray(np.arange(800, dtype=np.uint8).reshape(20, 40)),
                (line_elem, "region_1", "line_1", crop_metadata),
            )
        ]
        config = SimpleNamespace(
            imgH=20,
            imgW=40,
            PAD=True,
            batch_size=2,
            workers=0,
            batch_max_length=100,
            rgb=False,
        )

        with patch(
            "recognition.recognize_manuscript_text_v2_pretrained._extract_ocr_line_crops_with_tree",
            return_value=(tree, page_root, "", batch_data),
        ):
            result = process_page_xml(
                xml_path,
                [],
                _FakeModel(),
                _FakeConverter(),
                config,
                torch.device("cpu"),
            )

        self.assertEqual(result["auto_orientation_line_count"], 0)
        self.assertEqual(result["auto_orientation_rotated_line_count"], 0)
        self.assertEqual(line_elem.findtext("TextEquiv/Unicode"), WRONG_ORIENTATION_TEXT)
        self.assertIsNone(line_elem.find("TextEquiv").get("custom"))

    def test_experiment_inference_uses_the_same_selection_rule(self):
        root = TESTS_ROOT / "_tmp_auto_orientation_unit" / "experiment"
        test_root = root / "prepared" / "finetune_dataset" / "test"
        test_root.mkdir(parents=True, exist_ok=True)
        image_path = test_root / "word_0001.png"
        cv2.imwrite(str(image_path), np.arange(800, dtype=np.uint8).reshape(20, 40))

        record = PreparedLineRecord(
            page_id="page",
            region_id="region",
            region_custom="textbox_label_0",
            line_id="line_1",
            line_custom="structure_line_id_1",
            line_numeric_id=1,
            text=CORRECT_ORIENTATION_TEXT,
            polygon_points=[[0, 0], [39, 0], [39, 19], [0, 19]],
            y_center=10.0,
            x_min=0.0,
            flat_image_rel_path="test/word_0001.png",
            crop_metadata=_crop_metadata("closed_circular"),
        )
        prepared_page = PreparedPageDataset(
            page_id="page",
            image_filename="page.jpg",
            source_xml_path=str(root / "page.xml"),
            source_image_path=str(root / "page.jpg"),
            output_root=str(root / "prepared"),
            image_format_dir=str(root / "prepared" / "image-format"),
            finetune_dataset_dir=str(root / "prepared" / "finetune_dataset"),
            gt_path=str(root / "prepared" / "finetune_dataset" / "gt.txt"),
            manifest_path=str(root / "prepared" / "manifest.json"),
            records=[record],
        )
        inference_results = [
            [{"image_path": str(image_path), "predicted_label": WRONG_ORIENTATION_TEXT}],
            [{"image_path": str(image_path), "predicted_label": CORRECT_ORIENTATION_TEXT}],
        ]

        with patch(
            "recognition.active_learning.load_inference_model",
            return_value=(object(), object(), SimpleNamespace(), torch.device("cpu")),
        ), patch(
            "recognition.active_learning.run_line_image_inference_from_loaded_model",
            side_effect=inference_results,
        ) as inference_mock, patch("recognition.active_learning.write_prediction_pagexml"):
            result = run_checkpoint_on_prepared_pages(
                root / "checkpoint.pth",
                [prepared_page],
                output_root=root / "predictions",
            )

        self.assertEqual(inference_mock.call_count, 2)
        line_result = result.per_line_predictions[0]
        self.assertEqual(line_result["predicted_text"], CORRECT_ORIENTATION_TEXT)
        self.assertEqual(
            line_result["auto_orientation"]["selected_transform"],
            ROTATE_180_TRANSFORM,
        )

    def test_finetune_preparation_applies_persisted_transform_to_training_image(self):
        root = TESTS_ROOT / "_tmp_auto_orientation_unit" / "finetune_orientation"
        root.mkdir(parents=True, exist_ok=True)
        xml_path = root / "page.xml"
        image_path = root / "page.jpg"
        output_root = root / "prepared"
        cv2.imwrite(str(image_path), np.full((30, 50), 240, dtype=np.uint8))
        xml_path.write_text(
            """<?xml version="1.0" encoding="UTF-8"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">
  <Page imageFilename="page.jpg" imageWidth="50" imageHeight="30">
    <TextRegion id="region_0" custom="textbox_label_0">
      <TextLine id="line_1" custom="structure_line_id_1">
        <Coords points="0,0 39,0 39,19 0,19" />
        <TextEquiv custom="auto_orientation_transform:rotate_180"><Unicode>राम</Unicode></TextEquiv>
      </TextLine>
    </TextRegion>
  </Page>
</PcGts>
""",
            encoding="utf-8",
        )
        raw_crop = np.full((20, 40), 230, dtype=np.uint8)
        raw_crop[:, :12] = 20

        with patch(
            "recognition.pagexml_line_dataset.crop_line_record_for_ocr",
            return_value=SimpleNamespace(
                image=raw_crop,
                metadata={"used_unwrap": True},
            ),
        ):
            prepared = prepare_page_line_dataset(xml_path, image_path, output_root)

        training_image = cv2.imread(
            str(Path(prepared.finetune_dataset_dir) / "test" / "word_0001.png"),
            cv2.IMREAD_GRAYSCALE,
        )
        self.assertGreater(training_image[:, :10].mean(), training_image[:, -10:].mean())
        crop_metadata = prepared.records[0].crop_metadata
        self.assertEqual(
            crop_metadata["applied_auto_orientation_transform"],
            ROTATE_180_TRANSFORM,
        )
        self.assertFalse(should_auto_orient_from_predictions(crop_metadata))

    def test_explicit_annotation_overrides_stale_transform_in_finetune_preparation(self):
        root = TESTS_ROOT / "_tmp_auto_orientation_unit" / "annotated_finetune_orientation"
        root.mkdir(parents=True, exist_ok=True)
        xml_path = root / "page.xml"
        image_path = root / "page.jpg"
        output_root = root / "prepared"
        cv2.imwrite(str(image_path), np.full((30, 50), 240, dtype=np.uint8))
        xml_path.write_text(
            """<?xml version="1.0" encoding="UTF-8"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">
  <Page imageFilename="page.jpg" imageWidth="50" imageHeight="30">
    <TextRegion id="region_0" custom="textbox_label_0">
      <TextLine id="line_1" custom="structure_line_id_1">
        <Coords points="0,0 39,0 39,19 0,19" />
        <TextEquiv custom="auto_orientation_transform:rotate_180"><Unicode>राम</Unicode></TextEquiv>
      </TextLine>
    </TextRegion>
  </Page>
</PcGts>
""",
            encoding="utf-8",
        )
        raw_crop = np.full((20, 40), 230, dtype=np.uint8)
        raw_crop[:, :12] = 20
        annotated_metadata = _crop_metadata("curved_open", reading_direction=[1.0, 0.0])

        with patch(
            "recognition.pagexml_line_dataset.crop_line_record_for_ocr",
            return_value=SimpleNamespace(image=raw_crop, metadata=annotated_metadata),
        ):
            prepared = prepare_page_line_dataset(xml_path, image_path, output_root)

        training_image = cv2.imread(
            str(Path(prepared.finetune_dataset_dir) / "test" / "word_0001.png"),
            cv2.IMREAD_GRAYSCALE,
        )
        self.assertLess(training_image[:, :10].mean(), training_image[:, -10:].mean())
        crop_metadata = prepared.records[0].crop_metadata
        self.assertTrue(has_explicit_reading_direction_annotation(crop_metadata))
        self.assertNotIn("applied_auto_orientation_transform", crop_metadata)
        self.assertEqual(crop_metadata["ignored_auto_orientation_transform"], ROTATE_180_TRANSFORM)
        self.assertEqual(
            crop_metadata["auto_orientation_ignore_reason"],
            "explicit_reading_direction_annotation",
        )


if __name__ == "__main__":
    unittest.main()
