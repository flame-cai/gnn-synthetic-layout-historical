from __future__ import annotations

import json
import shutil
import sys
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np


TESTS_ROOT = Path(__file__).resolve().parent
APP_ROOT = TESTS_ROOT.parent
REPO_ROOT = APP_ROOT.parent

if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recognition.line_segmentation.ocr_crops import (
    crop_line_record_for_ocr,
    load_line_segmentation_metadata_by_numeric_id,
    masked_line_crop,
)
from recognition.pagexml_line_dataset import prepare_page_line_dataset
from recognition.recognize_manuscript_text_v2_pretrained import extract_ocr_line_crops_from_page_xml


class StrategyAwareOcrCropsUnitTest(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        tmp_root = TESTS_ROOT / "_tmp_strategy_aware_ocr_crops_unit"
        if tmp_root.exists():
            shutil.rmtree(tmp_root)

    def _tmp_root(self, name: str) -> Path:
        tmp_root = TESTS_ROOT / "_tmp_strategy_aware_ocr_crops_unit" / name
        if tmp_root.exists():
            shutil.rmtree(tmp_root)
        tmp_root.mkdir(parents=True, exist_ok=True)
        return tmp_root

    def _image_and_record(self):
        image = np.full((96, 96), 240, dtype=np.uint8)
        image[18:78, 42:54] = 20
        record = SimpleNamespace(
            page_id="unit_page",
            region_custom="textbox_label_0",
            line_id="region_0_line_0",
            line_custom="structure_line_id_7",
            line_numeric_id=7,
            text="test",
            polygon_points=[[42, 16], [54, 16], [54, 80], [42, 80]],
            baseline_points=[[48, 18], [48, 78]],
        )
        return image, record

    def _write_page(self, tmp_root: Path, metadata: dict | None = None):
        page_id = "unit_page"
        image_path = tmp_root / f"{page_id}.jpg"
        xml_path = tmp_root / f"{page_id}.xml"
        image, _ = self._image_and_record()
        cv2.imwrite(str(image_path), image)

        ns = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
        ET.register_namespace("", ns)
        xml_path.write_text(
            f"""<?xml version="1.0" encoding="UTF-8"?>
<PcGts xmlns="{ns}">
  <Page imageFilename="{page_id}.jpg" imageWidth="96" imageHeight="96">
    <TextRegion id="region_0" custom="textbox_label_0">
      <TextLine id="region_0_line_0" custom="structure_line_id_7">
        <Coords points="42,16 54,16 54,80 42,80" />
        <Baseline points="48,18 48,78" />
        <TextEquiv><Unicode>test</Unicode></TextEquiv>
      </TextLine>
    </TextRegion>
  </Page>
</PcGts>
""",
            encoding="utf-8",
        )
        if metadata is not None:
            (tmp_root / f"{page_id}_line_segmentation_metadata.json").write_text(
                json.dumps(metadata, indent=2),
                encoding="utf-8",
            )
        return page_id, xml_path, image_path

    def test_missing_metadata_uses_masked_crop(self):
        image, record = self._image_and_record()

        result = crop_line_record_for_ocr(image, record)

        np.testing.assert_array_equal(result.image, masked_line_crop(image, record.polygon_points))
        self.assertFalse(result.metadata["used_unwrap"])
        self.assertEqual(result.metadata["crop_model"], "axis_aligned_masked_crop")
        self.assertEqual(result.metadata["fallback_reason"], "missing_metadata")

    def test_malformed_metadata_loader_returns_empty_mapping(self):
        tmp_root = self._tmp_root("malformed")
        metadata_path = tmp_root / "bad_metadata.json"
        metadata_path.write_text("{", encoding="utf-8")

        with self.assertLogs("recognition.line_segmentation.ocr_crops", level="WARNING"):
            loaded = load_line_segmentation_metadata_by_numeric_id(metadata_path)

        self.assertEqual(loaded, {})

    def test_legacy_metadata_uses_masked_crop(self):
        image, record = self._image_and_record()

        result = crop_line_record_for_ocr(
            image,
            record,
            strategy_name="legacy_axis_bound_v1",
            strategy_line_metadata={"line_numeric_id": 7},
        )

        np.testing.assert_array_equal(result.image, masked_line_crop(image, record.polygon_points))
        self.assertFalse(result.metadata["used_unwrap"])
        self.assertIsNone(result.metadata["fallback_reason"])

    def test_local_tangent_metadata_unwraps(self):
        image, record = self._image_and_record()

        result = crop_line_record_for_ocr(
            image,
            record,
            strategy_name="local_tangent_band_v1",
            strategy_line_metadata={"line_numeric_id": 7, "crop_model": "local_tangent_band"},
        )

        self.assertTrue(result.metadata["used_unwrap"])
        self.assertEqual(result.metadata["crop_model"], "local_tangent_band")
        self.assertEqual(result.metadata["unwrap_strategy"], "baseline_local_tangent")
        self.assertGreater(result.image.shape[1], result.image.shape[0])

    def test_local_tangent_delegate_uses_masked_crop(self):
        image, record = self._image_and_record()

        result = crop_line_record_for_ocr(
            image,
            record,
            strategy_name="local_tangent_band_v1",
            strategy_line_metadata={"line_numeric_id": 7, "crop_model": "legacy_axis_bound_delegate"},
        )

        np.testing.assert_array_equal(result.image, masked_line_crop(image, record.polygon_points))
        self.assertFalse(result.metadata["used_unwrap"])
        self.assertEqual(result.metadata["fallback_reason"], "non_unwrapped_crop_model:legacy_axis_bound_delegate")

    def test_pagexml_coords_preparation_uses_sibling_metadata_without_heatmap_or_xml_mutation(self):
        tmp_root = self._tmp_root("pagexml_coords_metadata")
        metadata = {
            "strategy_name": "local_tangent_band_v1",
            "line_metadata": [
                {"line_numeric_id": 7, "crop_model": "local_tangent_band"},
            ],
        }
        _, xml_path, image_path = self._write_page(tmp_root, metadata=metadata)
        original_xml = xml_path.read_text(encoding="utf-8")

        prepared = prepare_page_line_dataset(
            xml_path,
            image_path,
            tmp_root / "prepared",
            geometry_source="pagexml_coords",
        )

        self.assertEqual(xml_path.read_text(encoding="utf-8"), original_xml)
        self.assertEqual(prepared.line_segmentation_strategy_name, "local_tangent_band_v1")
        self.assertTrue(Path(prepared.line_segmentation_metadata_path).exists())
        self.assertTrue(prepared.records[0].crop_metadata["used_unwrap"])
        self.assertEqual(prepared.records[0].crop_metadata["unwrap_strategy"], "baseline_local_tangent")

    def test_local_ocr_crop_extraction_discovers_sibling_metadata_and_falls_back_when_absent(self):
        with_meta_root = self._tmp_root("local_ocr_with_metadata")
        metadata = {
            "strategy_name": "local_tangent_band_v1",
            "line_metadata": [
                {"line_numeric_id": 7, "crop_model": "local_tangent_band"},
            ],
        }
        _, xml_path, image_path = self._write_page(with_meta_root, metadata=metadata)

        crops = extract_ocr_line_crops_from_page_xml(xml_path, [str(image_path.parent)])

        self.assertEqual(len(crops), 1)
        self.assertTrue(crops[0][1][3]["used_unwrap"])

        no_meta_root = self._tmp_root("local_ocr_without_metadata")
        _, xml_path, image_path = self._write_page(no_meta_root)

        crops = extract_ocr_line_crops_from_page_xml(xml_path, [str(image_path.parent)])

        self.assertEqual(len(crops), 1)
        self.assertFalse(crops[0][1][3]["used_unwrap"])
        self.assertEqual(crops[0][1][3]["crop_model"], "axis_aligned_masked_crop")


if __name__ == "__main__":
    unittest.main()
