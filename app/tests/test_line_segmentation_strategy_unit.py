from __future__ import annotations

import json
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
    get_production_strategy_name,
    get_text_line_segmentation_strategy,
    list_text_line_segmentation_strategies,
)
from recognition.line_segmentation.runtime_config import get_strategy_runtime_config
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
        self.assertIn("local_tangent_band_v1", list_text_line_segmentation_strategies())
        self.assertIn("local_polygons_v1", list_text_line_segmentation_strategies())
        self.assertIn("local_polygons_hstraight_smooth_unwrap_v1", list_text_line_segmentation_strategies())
        self.assertIn("local_polygons_stable_unwrap_v1", list_text_line_segmentation_strategies())

    def test_registry_returns_local_tangent_strategy(self):
        strategy = get_text_line_segmentation_strategy("local_tangent_band_v1")

        self.assertEqual(strategy.name, "local_tangent_band_v1")

    def test_registry_returns_local_polygons_strategy(self):
        strategy = get_text_line_segmentation_strategy("local_polygons_v1")

        self.assertEqual(strategy.name, "local_polygons_v1")

    def test_registry_returns_horizontal_straight_smooth_unwrap_strategy(self):
        strategy = get_text_line_segmentation_strategy("local_polygons_hstraight_smooth_unwrap_v1")

        self.assertEqual(strategy.name, "local_polygons_hstraight_smooth_unwrap_v1")

    def test_registry_returns_stable_unwrap_strategy(self):
        strategy = get_text_line_segmentation_strategy("local_polygons_stable_unwrap_v1")

        self.assertEqual(strategy.name, "local_polygons_stable_unwrap_v1")

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
            line_segmentation_strategy_name="local_polygons_v1",
        )

        self.assertEqual(implicit.line_segmentation_strategy_name, "local_polygons_v1")
        self.assertEqual(explicit.line_segmentation_strategy_name, "local_polygons_v1")
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
        pagexml_source = (APP_ROOT / "recognition" / "pagexml_line_dataset.py").read_text(encoding="utf-8")

        self.assertIn("get_production_strategy_name", source)
        self.assertNotIn("get_benchmark_strategy_name", source)
        self.assertIn("get_production_strategy_name", pagexml_source)
        self.assertNotIn("get_benchmark_strategy_name", pagexml_source)
        self.assertIn("apply_text_line_segmentation_strategy", source)
        self.assertIn("get_strategy_runtime_config", source)
        self.assertEqual(get_production_strategy_name(), "local_polygons_v1")

    def test_production_local_polygons_runtime_config_uses_benchmark_threshold(self):
        config = get_strategy_runtime_config(get_production_strategy_name(), include_empty_text_lines=True)

        self.assertEqual(get_production_strategy_name(), "local_polygons_v1")
        self.assertEqual(float(config["BINARIZE_THRESHOLD"]), 0.45)
        self.assertTrue(config["include_empty_text_lines"])
        self.assertEqual(float(config["final_mask_normal_pad_px"]), 0.0)

    def _make_single_line_page(self, name: str, baseline_points: str, ink_rects: list[tuple[int, int, int, int]]):
        tmp_root = TESTS_ROOT / "_tmp_line_segmentation_strategy_unit" / name
        if tmp_root.exists():
            shutil.rmtree(tmp_root)
        tmp_root.mkdir(parents=True, exist_ok=True)

        page_id = "unit_page"
        image_path = tmp_root / f"{page_id}.jpg"
        heatmap_path = tmp_root / f"{page_id}_heatmap.jpg"
        xml_path = tmp_root / f"{page_id}.xml"

        image = np.full((96, 96), 240, dtype=np.uint8)
        heatmap = np.zeros((96, 96), dtype=np.uint8)
        for x_val, y_val, width, height in ink_rects:
            image[y_val : y_val + height, x_val : x_val + width] = 20
            heatmap[y_val : y_val + height, x_val : x_val + width] = 255
        cv2.imwrite(str(image_path), image)
        cv2.imwrite(str(heatmap_path), heatmap)

        ns = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
        ET.register_namespace("", ns)
        xml_path.write_text(
            f"""<?xml version="1.0" encoding="UTF-8"?>
<PcGts xmlns="{ns}">
  <Page imageFilename="{page_id}.jpg" imageWidth="96" imageHeight="96">
    <TextRegion id="region_0" custom="textbox_label_0">
      <TextLine id="region_0_line_0" custom="structure_line_id_7">
        <TextEquiv><Unicode>test</Unicode></TextEquiv>
        <Baseline points="{baseline_points}" />
      </TextLine>
    </TextRegion>
  </Page>
</PcGts>
""",
            encoding="utf-8",
        )
        return tmp_root, xml_path, image_path, heatmap_path

    def test_local_tangent_vertical_line_unwraps_to_horizontal_crop(self):
        tmp_root, xml_path, image_path, heatmap_path = self._make_single_line_page(
            "local_vertical",
            "48,18 48,78",
            [(42, 18, 12, 60)],
        )

        prepared = prepare_page_line_dataset(
            xml_path,
            image_path,
            tmp_root / "prepared",
            heatmap_path=heatmap_path,
            geometry_source="baseline_heatmap",
            line_segmentation_strategy_name="local_tangent_band_v1",
        )

        self.assertEqual(prepared.line_segmentation_strategy_name, "local_tangent_band_v1")
        self.assertEqual(len(prepared.records), 1)
        crop = cv2.imread(str(Path(prepared.finetune_dataset_dir) / prepared.records[0].flat_image_rel_path), cv2.IMREAD_GRAYSCALE)
        self.assertGreater(crop.shape[1], crop.shape[0])
        self.assertEqual(prepared.records[0].crop_metadata["topology"]["line_kind"], "vertical_straight")
        self.assertEqual(prepared.records[0].crop_metadata["orientation"]["selected_transform"], "identity")

    def test_local_tangent_curved_line_writes_band_metadata(self):
        tmp_root, xml_path, image_path, heatmap_path = self._make_single_line_page(
            "local_curved",
            "18,70 32,44 48,30 64,44 78,70",
            [(18, 64, 60, 12), (30, 42, 36, 12)],
        )
        metadata_path = tmp_root / "out" / "metadata.json"

        result = apply_text_line_segmentation_strategy(
            page_image_path=image_path,
            heatmap_path=heatmap_path,
            source_pagexml_path=xml_path,
            output_pagexml_path=tmp_root / "out" / "unit_page.xml",
            strategy_name="local_tangent_band_v1",
            metadata_path=metadata_path,
        )

        self.assertEqual(result.strategy_name, "local_tangent_band_v1")
        self.assertEqual(result.prepared_line_count, 1)
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        line = payload["line_metadata"][0]
        self.assertEqual(line["line_kind"], "curved_open")
        self.assertEqual(line["crop_model"], "local_tangent_band")
        self.assertGreater(len(line["coords_points"]), 4)

    def test_local_tangent_circular_out_and_back_normalizes_topology(self):
        tmp_root, xml_path, image_path, heatmap_path = self._make_single_line_page(
            "local_circular",
            "48,16 78,48 48,80 18,48 48,16 18,48 48,80 78,48",
            [(16, 16, 64, 64)],
        )
        metadata_path = tmp_root / "out" / "metadata.json"

        apply_text_line_segmentation_strategy(
            page_image_path=image_path,
            heatmap_path=heatmap_path,
            source_pagexml_path=xml_path,
            output_pagexml_path=tmp_root / "out" / "unit_page.xml",
            strategy_name="local_tangent_band_v1",
            metadata_path=metadata_path,
        )

        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        topology = payload["line_metadata"][0]["topology"]
        self.assertTrue(topology["was_out_and_back"])
        self.assertTrue(topology["is_closed"])
        self.assertEqual(payload["line_metadata"][0]["line_kind"], "closed_circular")
        self.assertEqual(payload["line_metadata"][0]["crop_model"], "local_tangent_band")

    def test_local_polygons_page_preparation_and_ocr_unwrap_are_separate(self):
        tmp_root, xml_path, image_path, heatmap_path = self._make_single_line_page(
            "local_polygons_horizontal",
            "18,48 78,48",
            [(18, 42, 60, 12)],
        )
        metadata_path = tmp_root / "out" / "metadata.json"

        result = apply_text_line_segmentation_strategy(
            page_image_path=image_path,
            heatmap_path=heatmap_path,
            source_pagexml_path=xml_path,
            output_pagexml_path=tmp_root / "out" / "unit_page.xml",
            strategy_name="local_polygons_v1",
            metadata_path=metadata_path,
        )

        self.assertEqual(result.strategy_name, "local_polygons_v1")
        self.assertEqual(result.prepared_line_count, 1)
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        line = payload["line_metadata"][0]
        self.assertEqual(line["crop_model"], "local_polygon_unwrap")
        self.assertEqual(line["component_projection_model"], "heatmap_component_contour_mask")
        self.assertEqual(line["local_cleanup_model"], "legacy_remap_top_bottom_cc")
        self.assertEqual(line["final_mask_normal_pad_px"], 0.0)
        self.assertEqual(line["final_mask_station_pad_px"], 1.0)
        self.assertFalse(payload["geometry_summary"]["used_legacy_axis_bound_delegate"])
        self.assertEqual(payload["geometry_summary"]["local_cleanup_model"], "legacy_remap_top_bottom_cc")
        self.assertNotIn("unwrap_strategy", line)

        prepared = prepare_page_line_dataset(
            xml_path,
            image_path,
            tmp_root / "prepared",
            heatmap_path=heatmap_path,
            geometry_source="baseline_heatmap",
            line_segmentation_strategy_name="local_polygons_v1",
        )

        self.assertEqual(prepared.line_segmentation_strategy_name, "local_polygons_v1")
        self.assertEqual(len(prepared.records), 1)
        crop_metadata = prepared.records[0].crop_metadata
        self.assertTrue(crop_metadata["used_unwrap"])
        self.assertEqual(crop_metadata["crop_model"], "local_polygon_unwrap")
        self.assertEqual(crop_metadata["unwrap_strategy"], "baseline_local_tangent")
        self.assertEqual(
            crop_metadata["strategy_line_metadata"]["component_projection_model"],
            "heatmap_component_contour_mask",
        )
        self.assertEqual(
            crop_metadata["strategy_line_metadata"]["local_cleanup_model"],
            "legacy_remap_top_bottom_cc",
        )

    def test_horizontal_straight_smooth_unwrap_strategy_changes_only_horizontal_crop_model(self):
        tmp_root, xml_path, image_path, heatmap_path = self._make_single_line_page(
            "hstraight_smooth_unwrap",
            "18,48 48,47 78,48",
            [(18, 42, 60, 12)],
        )
        benchmark_metadata_path = tmp_root / "benchmark" / "metadata.json"
        proposed_metadata_path = tmp_root / "proposed" / "metadata.json"
        benchmark = apply_text_line_segmentation_strategy(
            page_image_path=image_path,
            heatmap_path=heatmap_path,
            source_pagexml_path=xml_path,
            output_pagexml_path=tmp_root / "benchmark" / "unit_page.xml",
            strategy_name="local_polygons_v1",
            metadata_path=benchmark_metadata_path,
        )
        proposed = apply_text_line_segmentation_strategy(
            page_image_path=image_path,
            heatmap_path=heatmap_path,
            source_pagexml_path=xml_path,
            output_pagexml_path=tmp_root / "proposed" / "unit_page.xml",
            strategy_name="local_polygons_hstraight_smooth_unwrap_v1",
            metadata_path=proposed_metadata_path,
        )

        self.assertEqual(proposed.strategy_name, "local_polygons_hstraight_smooth_unwrap_v1")
        self.assertEqual(
            proposed.geometry_summary["geometry_delegate_strategy_name"],
            "local_polygons_v1",
        )
        self.assertEqual(
            benchmark.line_metadata[0]["coords_points"],
            proposed.line_metadata[0]["coords_points"],
        )
        self.assertEqual(
            proposed.line_metadata[0]["crop_model"],
            "local_polygon_horizontal_straight_fit_unwrap",
        )

        prepared = prepare_page_line_dataset(
            xml_path,
            image_path,
            tmp_root / "prepared",
            heatmap_path=heatmap_path,
            geometry_source="baseline_heatmap",
            line_segmentation_strategy_name="local_polygons_hstraight_smooth_unwrap_v1",
        )
        crop_metadata = prepared.records[0].crop_metadata
        self.assertTrue(crop_metadata["used_unwrap"])
        self.assertEqual(crop_metadata["crop_model"], "local_polygon_horizontal_straight_fit_unwrap")
        self.assertEqual(crop_metadata["unwrap_strategy"], "horizontal_straight_fit_tangent")
        self.assertTrue(crop_metadata["horizontal_straight_fit"]["eligible"])

    def test_stable_unwrap_strategy_changes_only_crop_model_for_all_local_polygon_lines(self):
        tmp_root, xml_path, image_path, heatmap_path = self._make_single_line_page(
            "stable_unwrap_circular",
            "48,16 78,48 48,80 18,48 48,16 18,48 48,80 78,48",
            [(16, 16, 64, 64)],
        )
        benchmark = apply_text_line_segmentation_strategy(
            page_image_path=image_path,
            heatmap_path=heatmap_path,
            source_pagexml_path=xml_path,
            output_pagexml_path=tmp_root / "benchmark" / "unit_page.xml",
            strategy_name="local_polygons_v1",
            metadata_path=tmp_root / "benchmark" / "metadata.json",
        )
        proposed = apply_text_line_segmentation_strategy(
            page_image_path=image_path,
            heatmap_path=heatmap_path,
            source_pagexml_path=xml_path,
            output_pagexml_path=tmp_root / "proposed" / "unit_page.xml",
            strategy_name="local_polygons_stable_unwrap_v1",
            metadata_path=tmp_root / "proposed" / "metadata.json",
        )

        self.assertEqual(proposed.strategy_name, "local_polygons_stable_unwrap_v1")
        self.assertEqual(
            benchmark.line_metadata[0]["coords_points"],
            proposed.line_metadata[0]["coords_points"],
        )
        self.assertEqual(proposed.line_metadata[0]["crop_model"], "local_polygon_stable_unwrap")
        self.assertEqual(
            proposed.geometry_summary["geometry_delegate_strategy_name"],
            "local_polygons_v1",
        )

        prepared = prepare_page_line_dataset(
            xml_path,
            image_path,
            tmp_root / "prepared",
            heatmap_path=heatmap_path,
            geometry_source="baseline_heatmap",
            line_segmentation_strategy_name="local_polygons_stable_unwrap_v1",
        )
        crop_metadata = prepared.records[0].crop_metadata
        self.assertTrue(crop_metadata["used_unwrap"])
        self.assertEqual(crop_metadata["crop_model"], "local_polygon_stable_unwrap")
        self.assertEqual(crop_metadata["unwrap_strategy"], "stable_arclength_tangent")
        self.assertTrue(crop_metadata["stable_unwrap"]["used_stable_path"])
        self.assertTrue(crop_metadata["stable_unwrap"]["vectorized_station_sampling"])
        self.assertEqual(crop_metadata["stable_unwrap"]["station_sampling"], "endpoint_exclusive_closed")

    def test_local_polygons_vertical_line_keeps_open_final_mask_tight(self):
        tmp_root, xml_path, image_path, heatmap_path = self._make_single_line_page(
            "local_polygons_vertical",
            "48,18 48,78",
            [(42, 18, 12, 60)],
        )
        metadata_path = tmp_root / "out" / "metadata.json"

        apply_text_line_segmentation_strategy(
            page_image_path=image_path,
            heatmap_path=heatmap_path,
            source_pagexml_path=xml_path,
            output_pagexml_path=tmp_root / "out" / "unit_page.xml",
            strategy_name="local_polygons_v1",
            metadata_path=metadata_path,
        )

        line = json.loads(metadata_path.read_text(encoding="utf-8"))["line_metadata"][0]
        self.assertEqual(line["line_kind"], "vertical_straight")
        self.assertEqual(line["crop_model"], "local_polygon_unwrap")
        self.assertEqual(line["final_mask_normal_pad_px"], 0.0)

    def test_local_polygons_remapped_local_cleanup_trims_top_boundary_noise(self):
        tmp_root, xml_path, image_path, heatmap_path = self._make_single_line_page(
            "local_polygons_cleanup",
            "18,48 78,48",
            [(18, 42, 60, 12)],
        )
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        image[32:36, 18:78] = 20
        cv2.imwrite(str(image_path), image)
        metadata_path = tmp_root / "out" / "metadata.json"

        apply_text_line_segmentation_strategy(
            page_image_path=image_path,
            heatmap_path=heatmap_path,
            source_pagexml_path=xml_path,
            output_pagexml_path=tmp_root / "out" / "unit_page.xml",
            strategy_name="local_polygons_v1",
            metadata_path=metadata_path,
        )

        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        line = payload["line_metadata"][0]
        self.assertEqual(line["local_cleanup_model"], "legacy_remap_top_bottom_cc")
        self.assertGreater(line["local_cleanup_removed_boundary_component_count"], 0)
        self.assertGreater(line["local_cleanup_top_trim_px_total"], 0)
        self.assertEqual(line["crop_model"], "local_polygon_unwrap")

    def test_local_polygons_open_baseline_projects_beyond_short_endpoints(self):
        tmp_root, xml_path, image_path, heatmap_path = self._make_single_line_page(
            "local_polygons_short_baseline",
            "40,48 60,48",
            [(18, 42, 14, 12), (68, 42, 16, 12)],
        )
        metadata_path = tmp_root / "out" / "metadata.json"

        apply_text_line_segmentation_strategy(
            page_image_path=image_path,
            heatmap_path=heatmap_path,
            source_pagexml_path=xml_path,
            output_pagexml_path=tmp_root / "out" / "unit_page.xml",
            strategy_name="local_polygons_v1",
            metadata_path=metadata_path,
        )

        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        coords = payload["line_metadata"][0]["coords_points"]
        x_values = [point[0] for point in coords]
        self.assertLessEqual(min(x_values), 20)
        self.assertGreaterEqual(max(x_values), 80)
        self.assertEqual(payload["line_metadata"][0]["crop_model"], "local_polygon_unwrap")

        prepared = prepare_page_line_dataset(
            xml_path,
            image_path,
            tmp_root / "prepared",
            heatmap_path=heatmap_path,
            geometry_source="baseline_heatmap",
            line_segmentation_strategy_name="local_polygons_v1",
        )
        crop_metadata = prepared.records[0].crop_metadata
        self.assertLess(crop_metadata["station_min_px"], 0.0)
        self.assertGreater(crop_metadata["station_max_px"], 20.0)
        self.assertGreater(crop_metadata["output_width_px"], 40)

    def test_local_polygons_point_baseline_preserves_component_width(self):
        tmp_root, xml_path, image_path, heatmap_path = self._make_single_line_page(
            "local_polygons_point_baseline",
            "48,48",
            [(30, 40, 36, 16)],
        )
        metadata_path = tmp_root / "out" / "metadata.json"

        apply_text_line_segmentation_strategy(
            page_image_path=image_path,
            heatmap_path=heatmap_path,
            source_pagexml_path=xml_path,
            output_pagexml_path=tmp_root / "out" / "unit_page.xml",
            strategy_name="local_polygons_v1",
            metadata_path=metadata_path,
        )

        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        line_metadata = payload["line_metadata"][0]
        coords = line_metadata["coords_points"]
        x_values = [point[0] for point in coords]
        y_values = [point[1] for point in coords]
        self.assertEqual(line_metadata["line_kind"], "point")
        self.assertGreater(max(x_values) - min(x_values), 25)
        self.assertGreater(max(y_values) - min(y_values), 12)
        self.assertLess(line_metadata["local_s_min"], 0.0)
        self.assertGreater(line_metadata["local_s_max"], 0.0)

        prepared = prepare_page_line_dataset(
            xml_path,
            image_path,
            tmp_root / "prepared",
            heatmap_path=heatmap_path,
            geometry_source="baseline_heatmap",
            line_segmentation_strategy_name="local_polygons_v1",
        )
        crop_metadata = prepared.records[0].crop_metadata
        crop = cv2.imread(
            str(Path(prepared.finetune_dataset_dir) / prepared.records[0].flat_image_rel_path),
            cv2.IMREAD_GRAYSCALE,
        )
        self.assertTrue(crop_metadata["used_unwrap"])
        self.assertIsNone(crop_metadata["fallback_reason"])
        self.assertEqual(crop_metadata["topology"]["line_kind"], "point")
        self.assertGreater(crop_metadata["output_width_px"], 25)
        self.assertGreater(crop.shape[1], 25)
        self.assertGreater(crop.shape[0], 12)

    def test_local_polygons_circular_line_uses_normalized_closed_baseline(self):
        tmp_root, xml_path, image_path, heatmap_path = self._make_single_line_page(
            "local_polygons_circular",
            "48,16 78,48 48,80 18,48 48,16 18,48 48,80 78,48",
            [(16, 16, 64, 64)],
        )

        prepared = prepare_page_line_dataset(
            xml_path,
            image_path,
            tmp_root / "prepared",
            heatmap_path=heatmap_path,
            geometry_source="baseline_heatmap",
            line_segmentation_strategy_name="local_polygons_v1",
        )

        self.assertEqual(prepared.line_segmentation_strategy_name, "local_polygons_v1")
        self.assertEqual(len(prepared.records), 1)
        crop_metadata = prepared.records[0].crop_metadata
        self.assertTrue(crop_metadata["used_unwrap"])
        self.assertEqual(crop_metadata["crop_model"], "local_polygon_unwrap")
        self.assertEqual(crop_metadata["topology"]["line_kind"], "closed_circular")
        self.assertTrue(crop_metadata["strategy_line_metadata"]["topology"]["is_closed"])
        self.assertGreater(crop_metadata["output_width_px"], crop_metadata["output_height_px"])

    def test_local_polygons_circular_line_uses_heatmap_contour_not_axis_bbox_height(self):
        tmp_root, xml_path, image_path, heatmap_path = self._make_single_line_page(
            "local_polygons_circular_contour_bounds",
            "64,24 104,64 64,104 24,64 64,24 24,64 64,104 104,64",
            [],
        )
        image = np.full((128, 128), 240, dtype=np.uint8)
        heatmap = np.zeros((128, 128), dtype=np.uint8)
        cv2.line(image, (78, 22), (110, 54), 20, thickness=5)
        cv2.line(heatmap, (78, 22), (110, 54), 255, thickness=5)
        cv2.imwrite(str(image_path), image)
        cv2.imwrite(str(heatmap_path), heatmap)

        metadata_path = tmp_root / "out" / "metadata.json"
        apply_text_line_segmentation_strategy(
            page_image_path=image_path,
            heatmap_path=heatmap_path,
            source_pagexml_path=xml_path,
            output_pagexml_path=tmp_root / "out" / "unit_page.xml",
            strategy_name="local_polygons_v1",
            metadata_path=metadata_path,
        )

        line = json.loads(metadata_path.read_text(encoding="utf-8"))["line_metadata"][0]
        self.assertEqual(line["component_projection_model"], "heatmap_component_contour_mask")
        self.assertEqual(line["line_kind"], "closed_circular")
        self.assertEqual(line["final_mask_normal_pad_px"], 10.0)
        self.assertLess(line["local_n_max"] - line["local_n_min"], 55.0)


if __name__ == "__main__":
    unittest.main()
