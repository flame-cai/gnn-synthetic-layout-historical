from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from shapely.geometry import box

from experiments.downstream_ocr.adapter import AdapterError, vlm_json_to_page
from experiments.downstream_ocr.metrics import (
    aggregate_page_records,
    build_textedit_groups,
    compute_iou_matrix,
    evaluate_page,
    match_objects,
    page_text,
    polygons_to_mask,
)
from experiments.downstream_ocr.pagexml import PageXmlPage, TextLine, load_pagexml, write_pagexml
from experiments.downstream_ocr.splits import make_three_folds
from experiments.downstream_ocr.text import normalize_text


def line(line_id: str, bounds, text: str = "x") -> TextLine:
    polygon = box(*bounds)
    points = tuple((float(x), float(y)) for x, y in polygon.exterior.coords[:-1])
    return TextLine(
        page_id="page_1",
        line_id=line_id,
        points=points,
        polygon=polygon,
        text=text,
        region_id=f"region_{line_id}",
    )


def page(lines) -> PageXmlPage:
    return PageXmlPage(
        page_id="page_1",
        image_filename="page_1.jpg",
        width=100,
        height=100,
        lines=tuple(lines),
        namespace="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15",
    )


class DownstreamOcrMetricTests(unittest.TestCase):
    def test_identical_gt_and_prediction(self):
        gt = page([line("a", (10, 10, 40, 20), "राम")])
        pred = page([line("a", (10, 10, 40, 20), "राम")])
        record = evaluate_page(
            manuscript_id="m",
            fold_id="fold_1",
            page_id="page_1",
            gt_page=gt,
            pred_page=pred,
        )
        self.assertEqual(record["tp_50"], 1)
        self.assertEqual(record["object_g_f1_75"], 1.0)
        self.assertEqual(record["page_cer"], 0.0)
        self.assertEqual(record["textedit"], 0.0)

    def test_empty_prediction(self):
        gt = page([line("a", (10, 10, 40, 20), "abcd")])
        pred = page([])
        record = evaluate_page(
            manuscript_id="m",
            fold_id="fold_1",
            page_id="page_1",
            gt_page=gt,
            pred_page=pred,
            status="api_error",
        )
        self.assertEqual(record["tp_50"], 0)
        self.assertEqual(record["fn_50"], 1)
        self.assertEqual(record["pixel_f1"], 0.0)
        self.assertEqual(record["page_cer"], 1.0)
        self.assertEqual(record["textedit"], 1.0)

    def test_one_hallucinated_line(self):
        gt = page([])
        pred = page([line("h", (10, 10, 40, 20), "noise")])
        record = evaluate_page(
            manuscript_id="m",
            fold_id="fold_1",
            page_id="page_1",
            gt_page=gt,
            pred_page=pred,
        )
        self.assertEqual(record["fp_50"], 1)
        self.assertEqual(record["object_precision_50"], 0.0)
        self.assertEqual(record["textedit"], 1.0)

    def test_iou_below_and_above_50(self):
        gt = [line("g", (0, 0, 10, 10))]
        pred_below = [line("p", (6, 0, 16, 10))]
        pred_above = [line("p", (3, 0, 13, 10))]
        self.assertLess(compute_iou_matrix(gt, pred_below)[0, 0], 0.50)
        self.assertGreater(compute_iou_matrix(gt, pred_above)[0, 0], 0.50)

    def test_iou_below_and_above_75(self):
        gt = [line("g", (0, 0, 10, 10))]
        pred_below = [line("p", (1.5, 0, 11.5, 10))]
        pred_above = [line("p", (0.5, 0, 10.5, 10))]
        self.assertLess(compute_iou_matrix(gt, pred_below)[0, 0], 0.75)
        self.assertGreater(compute_iou_matrix(gt, pred_above)[0, 0], 0.75)

    def test_one_gt_split_into_two_predictions_textedit_groups(self):
        gt = [line("g", (0, 0, 100, 10), "abcdef")]
        pred = [
            line("p1", (0, 0, 50, 10), "abc"),
            line("p2", (50, 0, 100, 10), "def"),
        ]
        groups = build_textedit_groups(gt, pred)
        self.assertEqual(groups, [("abcdef", "abc def")])

    def test_two_gt_lines_merged_into_one_prediction(self):
        gt = [
            line("g1", (0, 0, 100, 10), "abc"),
            line("g2", (0, 10, 100, 20), "def"),
        ]
        pred = [line("p", (0, 0, 100, 20), "abc def")]
        groups = build_textedit_groups(gt, pred)
        self.assertEqual(groups, [("abc def", "abc def")])

    def test_optimal_matching_beats_greedy_failure_case(self):
        matrix = np.asarray(
            [
                [0.90, 0.80],
                [0.85, 0.00],
            ],
            dtype=float,
        )
        matches = match_objects(matrix, 0.50)
        self.assertEqual(len(matches), 2)
        self.assertEqual({(gt, pred) for gt, pred, _ in matches}, {(0, 1), (1, 0)})

    def test_overlapping_same_side_mask_union_semantics(self):
        polygons = [box(0, 0, 10, 10), box(0, 0, 10, 10)]
        mask = polygons_to_mask(polygons, 20, 20)
        self.assertEqual(int(mask.sum()), 121)

    def test_correct_text_with_reversed_line_coordinates(self):
        top = line("top", (0, 0, 10, 10), "A")
        bottom = line("bottom", (0, 20, 10, 30), "B")
        self.assertEqual(page_text([bottom, top]), "A B")

    def test_unicode_nfc_equivalence(self):
        self.assertEqual(normalize_text("e\u0301"), normalize_text("\u00e9"))

    def test_json_failure_represented_as_empty_prediction(self):
        template = page([line("g", (0, 0, 10, 10), "x")])
        with self.assertRaises(AdapterError):
            vlm_json_to_page("", template_page=template)

    def test_vlm_json_polygon_and_box_adapter(self):
        template = page([])
        payload = {
            "status": "success",
            "regions": [
                {
                    "id": "r",
                    "lines": [
                        {"id": "poly", "polygon_2d": [[0, 0], [0, 500], [100, 500], [100, 0]], "text": "a"},
                        {"id": "box", "box_2d": [200, 0, 300, 500], "text": "b"},
                    ],
                }
            ],
        }
        adapted = vlm_json_to_page(json.dumps(payload), template_page=template)
        self.assertEqual(len(adapted.lines), 2)
        self.assertEqual(adapted.lines[0].text, "a")

    def test_pagexml_parser_selects_smallest_text_equiv_index(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            xml_path = Path(tmp_dir) / "p.xml"
            xml_path.write_text(
                """<?xml version='1.0' encoding='UTF-8'?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">
  <Page imageFilename="p.jpg" imageWidth="100" imageHeight="100">
    <TextRegion id="r">
      <TextLine id="l">
        <Coords points="0,0 10,0 10,10 0,10"/>
        <TextEquiv index="2"><Unicode>later</Unicode></TextEquiv>
        <TextEquiv index="1"><Unicode>earlier</Unicode></TextEquiv>
      </TextLine>
    </TextRegion>
  </Page>
</PcGts>
""",
                encoding="utf-8",
            )
            parsed = load_pagexml(xml_path)
            self.assertEqual(parsed.lines[0].text, "earlier")

    def test_pagexml_repair_mode_accepts_self_intersecting_coords(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            xml_path = Path(tmp_dir) / "p.xml"
            xml_path.write_text(
                """<?xml version='1.0' encoding='UTF-8'?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">
  <Page imageFilename="p.jpg" imageWidth="100" imageHeight="100">
    <TextRegion id="r">
      <TextLine id="l">
        <Coords points="10,10 40,40 10,40 40,10"/>
        <TextEquiv><Unicode>text</Unicode></TextEquiv>
      </TextLine>
    </TextRegion>
  </Page>
</PcGts>
""",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_pagexml(xml_path)
            parsed = load_pagexml(xml_path, repair_geometry=True)
            self.assertEqual(len(parsed.lines), 1)
            self.assertTrue(parsed.lines[0].polygon.is_valid)
            self.assertGreater(parsed.lines[0].polygon.area, 0)

    def test_write_and_load_empty_prediction_pagexml(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            target = Path(tmp_dir) / "empty.xml"
            write_pagexml(page([]), target)
            parsed = load_pagexml(target)
            self.assertEqual(parsed.lines, ())

    def test_correct_aggregation_over_three_folds(self):
        folds = make_three_folds(tuple(f"p{i}" for i in range(12)))
        self.assertEqual(len(folds), 3)
        records = []
        for fold in folds:
            records.append(
                {
                    "status": "success",
                    "tp_50": 1,
                    "fp_50": 0,
                    "fn_50": 1,
                    "tp_75": 1,
                    "fp_75": 0,
                    "fn_75": 1,
                    "pixel_tp": 10,
                    "pixel_fp": 0,
                    "pixel_fn": 10,
                    "page_cer_distance": 2,
                    "page_cer_gt_chars": 10,
                    "page_cer": 0.2,
                    "textedit_distance_sum": 1,
                    "textedit_max_length_sum": 5,
                    "textedit": 0.2,
                }
            )
        aggregate = aggregate_page_records(records)
        self.assertEqual(aggregate["page_count"], 3)
        self.assertAlmostEqual(aggregate["object_recall_50"], 0.5)
        self.assertAlmostEqual(aggregate["micro_page_cer"], 0.2)


if __name__ == "__main__":
    unittest.main()
