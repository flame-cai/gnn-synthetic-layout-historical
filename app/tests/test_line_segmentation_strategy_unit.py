from __future__ import annotations

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

from recognition.line_segmentation import (
    apply_text_line_segmentation_strategy,
    get_text_line_segmentation_strategy,
    list_text_line_segmentation_strategies,
)
from recognition.pagexml_line_dataset import prepare_page_line_dataset


class LineSegmentationStrategyUnitTest(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        tmp_root = TESTS_ROOT / "_tmp_line_segmentation_strategy_unit"
        if tmp_root.exists():
            shutil.rmtree(tmp_root)

    def _make_synthetic_page(self, name: str):
        tmp_root = TESTS_ROOT / "_tmp_line_segmentation_strategy_unit" / name
        if tmp_root.exists():
            shutil.rmtree(tmp_root)
        tmp_root.mkdir(parents=True, exist_ok=True)

        page_id = "unit_page"
        image_path = tmp_root / f"{page_id}.jpg"
        heatmap_path = tmp_root / f"{page_id}_heatmap.jpg"
        xml_path = tmp_root / f"{page_id}.xml"

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
      <Coords points="10,10 80,10 80,50 10,50" />
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
        return tmp_root, page_id, xml_path, image_path, heatmap_path

    def test_registry_returns_legacy_strategy(self):
        strategy = get_text_line_segmentation_strategy("legacy_axis_bound_v1")

        self.assertEqual(strategy.name, "legacy_axis_bound_v1")
        self.assertIn("legacy_axis_bound_v1", list_text_line_segmentation_strategies())

    def test_unknown_strategy_error_names_request(self):
        with self.assertRaisesRegex(ValueError, "does_not_exist"):
            get_text_line_segmentation_strategy("does_not_exist")

    def test_legacy_strategy_writes_coords_without_reading_pagexml_coords(self):
        tmp_root, _, xml_path, image_path, heatmap_path = self._make_synthetic_page("direct_apply")
        output_xml_path = tmp_root / "out" / "unit_page.xml"
        metadata_path = tmp_root / "out" / "metadata.json"

        result = apply_text_line_segmentation_strategy(
            page_image_path=image_path,
            heatmap_path=heatmap_path,
            source_pagexml_path=xml_path,
            output_pagexml_path=output_xml_path,
            strategy_name="legacy_axis_bound_v1",
            metadata_path=metadata_path,
        )

        self.assertEqual(result.strategy_name, "legacy_axis_bound_v1")
        self.assertEqual(result.line_count, 1)
        self.assertEqual(result.prepared_line_count, 1)
        self.assertTrue(output_xml_path.exists())
        self.assertTrue(metadata_path.exists())
        self.assertEqual(result.geometry_summary["source_line_coverage"], 1.0)
        self.assertEqual(result.geometry_summary["heatmap_box_assignment_rate"], 1.0)

        ns = {"p": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}
        root = ET.parse(output_xml_path).getroot()
        page = root.find(".//p:Page", ns)
        region = root.find(".//p:TextRegion", ns)
        line = root.find(".//p:TextLine", ns)
        coords = line.find("./p:Coords", ns)
        baseline = line.find("./p:Baseline", ns)
        unicode_elem = line.find("./p:TextEquiv/p:Unicode", ns)

        self.assertEqual(page.get("imageFilename"), "unit_page.jpg")
        self.assertEqual(region.get("id"), "region_0")
        self.assertEqual(line.get("id"), "region_0_line_0")
        self.assertEqual(line.get("custom"), "structure_line_id_7")
        self.assertEqual(baseline.get("points"), "24,36 72,36")
        self.assertEqual(unicode_elem.text, "test")
        self.assertIsNotNone(coords)
        self.assertNotEqual(coords.get("points"), "10,10 80,10 80,50 10,50")

    def test_baseline_heatmap_alias_and_explicit_strategy_match(self):
        tmp_root, _, xml_path, image_path, heatmap_path = self._make_synthetic_page("dataset_alias")
        implicit = prepare_page_line_dataset(
            xml_path,
            image_path,
            tmp_root / "implicit",
            heatmap_path=heatmap_path,
            geometry_source="baseline_heatmap",
        )
        explicit = prepare_page_line_dataset(
            xml_path,
            image_path,
            tmp_root / "explicit",
            heatmap_path=heatmap_path,
            geometry_source="baseline_heatmap",
            line_segmentation_strategy_name="legacy_axis_bound_v1",
        )

        self.assertEqual(implicit.line_segmentation_strategy_name, "legacy_axis_bound_v1")
        self.assertEqual(explicit.line_segmentation_strategy_name, "legacy_axis_bound_v1")
        self.assertEqual(len(implicit.records), len(explicit.records))
        self.assertEqual(
            implicit.geometry_summary["heatmap_box_assignment_rate"],
            explicit.geometry_summary["heatmap_box_assignment_rate"],
        )
        self.assertEqual(
            implicit.geometry_summary["source_line_coverage"],
            explicit.geometry_summary["source_line_coverage"],
        )

    def test_app_save_module_uses_named_strategy_registry(self):
        source = (APP_ROOT / "gnn_inference.py").read_text(encoding="utf-8")

        self.assertIn('DEFAULT_TEXT_LINE_SEGMENTATION_STRATEGY = "legacy_axis_bound_v1"', source)
        self.assertIn("apply_text_line_segmentation_strategy", source)


if __name__ == "__main__":
    unittest.main()
