from __future__ import annotations

import json
import sys
import unittest
import uuid
from pathlib import Path

TESTS_ROOT = Path(__file__).resolve().parent
APP_ROOT = TESTS_ROOT.parent
REPO_ROOT = APP_ROOT.parent
TEMP_ROOT = TESTS_ROOT / "logs" / "circular_ocr_unit_tmp"
TEMP_ROOT.mkdir(parents=True, exist_ok=True)

if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.circular_OCR_test.baseline import BaselineComparisonError, compare_metrics_to_baseline
from tests.circular_OCR_test.geometry import (
    cut_circular_curve_at_topmost,
    normalize_open_curve,
    parse_gnn_format_page,
    parse_pagexml_lines,
)
from tests.circular_OCR_test.ocr_unwrap import (
    choose_orientation_by_confidence,
    derive_half_height_from_page_coords,
    generate_orientation_candidates,
    prepare_unwrapped_line_record,
)


def _make_test_dir(name: str) -> Path:
    path = TEMP_ROOT / f"{name}_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path


class CircularOcrGeometryUnitTest(unittest.TestCase):
    def test_gnn_format_parser_loads_points_dims_and_labels(self):
        root = _make_test_dir("gnn_parser_loads")
        (root / "page_1_dims.txt").write_text("100.0 200.0\n", encoding="utf-8")
        (root / "page_1_inputs_unnormalized.txt").write_text(
            "10.0 20.0 0.5\n30.0 40.0 0.7\n",
            encoding="utf-8",
        )
        (root / "page_1_inputs_normalized.txt").write_text(
            "0.1 0.1 0.5\n0.3 0.2 0.7\n",
            encoding="utf-8",
        )
        (root / "page_1_labels_textline.txt").write_text("5\n8\n", encoding="utf-8")

        page = parse_gnn_format_page(root, "page_1")

        self.assertEqual(page.width, 100.0)
        self.assertEqual(page.height, 200.0)
        self.assertEqual([(point.x, point.y, point.font_size) for point in page.points], [(10.0, 20.0, 0.5), (30.0, 40.0, 0.7)])
        self.assertEqual(page.textline_labels, [5, 8])

    def test_gnn_format_parser_detects_label_mismatch(self):
        root = _make_test_dir("gnn_parser_mismatch")
        (root / "page_1_dims.txt").write_text("100.0 200.0\n", encoding="utf-8")
        (root / "page_1_inputs_unnormalized.txt").write_text("10.0 20.0 0.5\n", encoding="utf-8")
        (root / "page_1_labels_textline.txt").write_text("5\n8\n", encoding="utf-8")

        with self.assertRaises(ValueError):
            parse_gnn_format_page(root, "page_1")

    def test_pagexml_parser_extracts_text_baseline_and_coords(self):
        xml_text = """<?xml version='1.0' encoding='UTF-8'?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">
  <Page imageFilename="page_1.jpg" imageWidth="100" imageHeight="200">
    <TextRegion id="r1" custom="textbox_label_0">
      <TextLine id="l1" custom="structure_line_id_7">
        <Baseline points="10,20 30,40" />
        <Coords points="8,18 32,18 32,42 8,42" />
        <TextEquiv><Unicode>abc</Unicode></TextEquiv>
      </TextLine>
    </TextRegion>
  </Page>
</PcGts>
"""
        root = _make_test_dir("pagexml_parser")
        xml_path = root / "page_1.xml"
        xml_path.write_text(xml_text, encoding="utf-8")
        lines = parse_pagexml_lines(xml_path)

        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0].text, "abc")
        self.assertEqual(lines[0].baseline_points, [(10, 20), (30, 40)])
        self.assertEqual(lines[0].coords_points, [(8, 18), (32, 18), (32, 42), (8, 42)])

    def test_curve_normalization_handles_straight_vertical_curved_and_circular(self):
        straight = normalize_open_curve([(0, 0), (10, 0), (20, 0)])
        vertical = normalize_open_curve([(0, 0), (0, 10), (0, 20)])
        curved = normalize_open_curve([(0, 0), (10, 5), (20, 0)])
        circular = normalize_open_curve([(10, 0), (0, 10), (10, 20), (20, 10), (10, 0)])

        self.assertEqual(straight.points, [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)])
        self.assertEqual(vertical.points, [(0.0, 0.0), (0.0, 10.0), (0.0, 20.0)])
        self.assertEqual(curved.points, [(0.0, 0.0), (10.0, 5.0), (20.0, 0.0)])
        self.assertTrue(circular.was_closed)
        self.assertGreaterEqual(len(circular.points), 4)

    def test_circular_cut_chooses_topmost_point_and_records_it(self):
        result = cut_circular_curve_at_topmost([(5, 5), (0, 10), (5, 15), (10, 10), (5, 5)])

        self.assertEqual(result.cut_point, (5.0, 5.0))
        self.assertEqual(result.cut_index, 0)
        self.assertEqual(result.points[0], (5.0, 5.0))

    def test_retraced_baseline_normalizes_to_single_traversal(self):
        baseline = [(0, 0), (10, 0), (20, 0), (10, 0)]

        result = normalize_open_curve(baseline)

        self.assertTrue(result.removed_mirrored_tail)
        self.assertEqual(result.points, [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)])

    def test_retraced_closed_baseline_cuts_single_loop_at_topmost(self):
        baseline = [(5, 0), (0, 5), (5, 10), (10, 5), (5, 0), (10, 5), (5, 10), (0, 5)]

        result = normalize_open_curve(baseline)

        self.assertTrue(result.removed_mirrored_tail)
        self.assertTrue(result.was_closed)
        self.assertEqual(result.points, [(5.0, 0.0), (0.0, 5.0), (5.0, 10.0), (10.0, 5.0)])


class CircularOcrUnwrapUnitTest(unittest.TestCase):
    def test_unwrapped_crop_metadata_is_separate_from_page_space_coords(self):
        line_record = prepare_unwrapped_line_record(
            page_id="page_1",
            line_id="line_1",
            line_custom="structure_line_id_1",
            text="abc",
            page_space_coords=[(0, 0), (10, 0), (10, 4), (0, 4)],
            baseline_points=[(1, 2), (9, 2)],
            selected_crop_rel_path="test/word_0001.png",
            candidate_metadata=[],
        )

        self.assertEqual(line_record.polygon_points, [[0, 0], [10, 0], [10, 4], [0, 4]])
        self.assertEqual(line_record.flat_image_rel_path, "test/word_0001.png")
        self.assertNotEqual(line_record.polygon_points, [[0, 0], [8, 0], [8, 4], [0, 4]])

    def test_page_space_coords_drive_unwrapped_band_height(self):
        baseline_points = [(0, 10), (20, 10)]
        narrow_coords = [(0, 6), (20, 6), (20, 14), (0, 14)]
        wide_coords = [(0, 0), (20, 0), (20, 40), (0, 40)]

        narrow = derive_half_height_from_page_coords(narrow_coords, baseline_points, default_half_height=20)
        wide = derive_half_height_from_page_coords(wide_coords, baseline_points, default_half_height=20)

        self.assertLess(narrow, wide)
        self.assertGreaterEqual(narrow, 4)

    def test_orientation_candidate_generator_produces_four_variants(self):
        names = [candidate.name for candidate in generate_orientation_candidates("line.png")]

        self.assertEqual(
            names,
            ["forward", "reversed", "forward_vertical_flip", "reversed_vertical_flip"],
        )

    def test_orientation_selector_uses_confidence_not_ground_truth(self):
        selected = choose_orientation_by_confidence(
            [
                {"candidate_name": "forward", "predicted_text": "wrong", "confidence_score": 0.2},
                {"candidate_name": "reversed", "predicted_text": "abc", "confidence_score": 0.6},
            ],
            ground_truth_text="wrong",
        )

        self.assertEqual(selected["selected_candidate_name"], "reversed")
        self.assertEqual(selected["selector_reason"], "max_confidence")


class CircularOcrBaselineUnitTest(unittest.TestCase):
    def test_baseline_comparator_handles_metric_directions_and_delta(self):
        baseline = {
            "primary_blocking_metric_name": "curve_metric_value",
            "metric_direction": "lower_is_better",
            "minimum_improvement_delta": 0.01,
            "metrics": {"curve_metric_value": 0.5},
        }

        passing = compare_metrics_to_baseline({"curve_metric_value": 0.48}, baseline)
        failing = compare_metrics_to_baseline({"curve_metric_value": 0.495}, baseline)

        self.assertTrue(passing["passed"])
        self.assertFalse(failing["passed"])

    def test_baseline_comparator_handles_higher_is_better(self):
        baseline = {
            "primary_blocking_metric_name": "first_step_gain",
            "metric_direction": "higher_is_better",
            "minimum_improvement_delta": 0.05,
            "metrics": {"first_step_gain": 0.1},
        }

        result = compare_metrics_to_baseline({"first_step_gain": 0.16}, baseline)

        self.assertTrue(result["passed"])

    def test_baseline_comparator_rejects_missing_nan_and_malformed(self):
        baseline = {
            "primary_blocking_metric_name": "curve_metric_value",
            "metric_direction": "lower_is_better",
            "minimum_improvement_delta": 0.01,
            "metrics": {"curve_metric_value": 0.5},
        }

        self.assertFalse(compare_metrics_to_baseline({}, baseline)["passed"])
        self.assertFalse(compare_metrics_to_baseline({"curve_metric_value": float("nan")}, baseline)["passed"])
        with self.assertRaises(BaselineComparisonError):
            compare_metrics_to_baseline({"curve_metric_value": 0.4}, {"metrics": {}})

    def test_baseline_json_file_must_be_well_formed(self):
        from tests.circular_OCR_test.config import BASELINE_JSON_PATH

        payload = json.loads(BASELINE_JSON_PATH.read_text(encoding="utf-8"))

        self.assertIn("dataset_name", payload)
        self.assertIn("metrics", payload)
        self.assertIn("primary_blocking_metric_name", payload)


if __name__ == "__main__":
    unittest.main()
