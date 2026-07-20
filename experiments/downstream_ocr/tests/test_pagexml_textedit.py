from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiments.downstream_ocr.dataset.pagexml2pagexml_dataset import (
    DATASET_NAME,
    PageXml2PageXmlDataset,
    PageXmlPair,
    evaluate_pagexml_pairs,
    match_pagexml,
    parse_pagexml,
)
from experiments.downstream_ocr.omnidocbench_v1_5.registry.registry import (
    DATASET_REGISTRY,
)
from experiments.downstream_ocr.omnidocbench_v1_5.utils.match import (
    match_gt2pred_simple,
)


PAGE_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"


def _pagexml(
    regions: list[dict],
    *,
    image_filename: str = "page.jpg",
    width: int = 100,
    height: int = 100,
) -> str:
    region_xml = []
    for region_index, region in enumerate(regions):
        line_xml = []
        for line_index, line in enumerate(region.get("lines", [])):
            text_equivs = line.get("text_equivs")
            if text_equivs is None:
                text_equivs = [(None, line.get("text", ""))]
            text_equiv_xml = []
            for index, text in text_equivs:
                index_attr = "" if index is None else f' index="{index}"'
                unicode_xml = "" if text is None else f"<Unicode>{text}</Unicode>"
                text_equiv_xml.append(
                    f"<TextEquiv{index_attr}>{unicode_xml}</TextEquiv>"
                )
            coords = line.get("coords", "0,0 10,0 10,10 0,10")
            baseline = line.get("baseline", "0,5 10,5")
            line_xml.append(
                f"""
      <TextLine id="{line.get('id', f'line_{line_index}')}" custom="{line.get('custom', 'line-label')}">
        <Coords points="{coords}"/>
        <Baseline points="{baseline}"/>
        {''.join(text_equiv_xml)}
      </TextLine>"""
            )
        region_xml.append(
            f"""
    <TextRegion id="{region.get('id', f'region_{region_index}')}" custom="{region.get('custom', 'region-label')}">
      <Coords points="{region.get('coords', '0,0 20,0 20,20 0,20')}"/>
      {''.join(line_xml)}
    </TextRegion>"""
        )
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<PcGts xmlns="{PAGE_NS}">
  <Page imageFilename="{image_filename}" imageWidth="{width}" imageHeight="{height}">
    {''.join(region_xml)}
  </Page>
</PcGts>
"""


class PageXmlTextEditTests(unittest.TestCase):
    def _evaluate(self, gt_xml: str, pred_xml: str | None):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        gt_path = root / "gt.xml"
        gt_path.write_text(gt_xml, encoding="utf-8")
        pred_path = None
        if pred_xml is not None:
            pred_path = root / "pred.xml"
            pred_path.write_text(pred_xml, encoding="utf-8")
        result = evaluate_pagexml_pairs(
            [
                PageXmlPair(
                    key="page",
                    gt_xml_path=gt_path,
                    pred_xml_path=pred_path,
                )
            ]
        )
        return result, gt_path, pred_path

    def test_perfect_match(self):
        xml = _pagexml(
            [{"lines": [{"id": "a", "text": "राम"}, {"id": "b", "text": "सीता"}]}]
        )
        result, _, _ = self._evaluate(xml, xml)
        self.assertEqual(result.page_metrics["page"]["textedit_all_page_avg"], 0.0)
        self.assertEqual(
            result.official_result["Edit_dist"]["ALL_page_avg"],
            0.0,
        )

    def test_xml_order_invariance(self):
        gt = _pagexml(
            [{"lines": [{"text": "A"}, {"text": "B"}, {"text": "C"}]}]
        )
        pred = _pagexml(
            [{"lines": [{"text": "C"}, {"text": "A"}, {"text": "B"}]}]
        )
        result, _, _ = self._evaluate(gt, pred)
        self.assertEqual(result.page_metrics["page"]["textedit"], 0.0)

    def test_coordinate_and_baseline_invariance(self):
        gt = _pagexml(
            [{"lines": [{"text": "same", "coords": "0,0 10,0 10,10 0,10"}]}]
        )
        pred = _pagexml(
            [
                {
                    "coords": "80,80 99,80 99,99 80,99",
                    "lines": [
                        {
                            "text": "same",
                            "coords": "90,90 99,90 99,99 90,99",
                            "baseline": "99,99 90,90",
                        }
                    ],
                }
            ],
            width=9999,
            height=7777,
        )
        result, _, _ = self._evaluate(gt, pred)
        self.assertEqual(result.page_metrics["page"]["textedit"], 0.0)

    def test_region_membership_ids_and_labels_are_invariant(self):
        gt = _pagexml(
            [
                {"id": "left", "custom": "label-a", "lines": [{"text": "A"}]},
                {"id": "right", "custom": "label-b", "lines": [{"text": "B"}]},
            ]
        )
        pred = _pagexml(
            [
                {
                    "id": "everything_changed",
                    "custom": "unrelated",
                    "lines": [{"text": "B"}, {"text": "A"}],
                }
            ]
        )
        result, _, _ = self._evaluate(gt, pred)
        self.assertEqual(result.page_metrics["page"]["textedit"], 0.0)

    def test_atomic_line_split_is_penalized(self):
        gt = _pagexml([{"lines": [{"text": "ABC DEF"}]}])
        pred = _pagexml([{"lines": [{"text": "ABC"}, {"text": "DEF"}]}])
        result, _, _ = self._evaluate(gt, pred)
        self.assertGreater(result.page_metrics["page"]["textedit"], 0.0)
        self.assertEqual(result.page_metrics["page"]["textedit_match_count"], 2)

    def test_missing_gt_match_is_a_deletion_record(self):
        gt = _pagexml([{"lines": [{"text": "A"}, {"text": "B"}]}])
        pred = _pagexml([{"lines": [{"text": "A"}]}])
        result, _, _ = self._evaluate(gt, pred)
        matches = result.matches_by_key["page"]
        self.assertTrue(any(match["pred"] == "" for match in matches))
        self.assertGreater(result.page_metrics["page"]["textedit"], 0.0)

    def test_extra_prediction_is_an_insertion_record(self):
        gt = _pagexml([{"lines": [{"text": "A"}]}])
        pred = _pagexml([{"lines": [{"text": "A"}, {"text": "EXTRA"}]}])
        result, _, _ = self._evaluate(gt, pred)
        matches = result.matches_by_key["page"]
        self.assertTrue(any(match["gt"] == "" for match in matches))
        self.assertGreater(result.page_metrics["page"]["textedit"], 0.0)

    def test_missing_prediction_page_is_not_skipped(self):
        gt = _pagexml([{"lines": [{"text": "A"}, {"text": "B"}]}])
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        gt_root = root / "gt"
        pred_root = root / "pred"
        gt_root.mkdir()
        pred_root.mkdir()
        (gt_root / "page.xml").write_text(gt, encoding="utf-8")

        with self.assertLogs(
            "experiments.downstream_ocr.dataset.pagexml2pagexml_dataset",
            level="WARNING",
        ):
            dataset = PageXml2PageXmlDataset(
                {
                    "dataset": {
                        "ground_truth": {"data_path": str(gt_root)},
                        "prediction": {"data_path": str(pred_root)},
                        "match_method": "simple_match",
                    }
                }
            )
        self.assertEqual(len(dataset.samples["text_block"]), 2)
        self.assertTrue(
            all(match["pred"] == "" for match in dataset.samples["text_block"])
        )

    def test_duplicate_text_is_matched_one_to_one(self):
        gt = _pagexml([{"lines": [{"text": "A"}, {"text": "A"}]}])
        pred = _pagexml([{"lines": [{"text": "A"}]}])
        result, _, _ = self._evaluate(gt, pred)
        matches = result.matches_by_key["page"]
        self.assertEqual(sum(match["pred"] == "A" for match in matches), 1)
        self.assertEqual(sum(match["pred"] == "" for match in matches), 1)

    def test_multiple_text_equiv_selection_policy(self):
        xml = _pagexml(
            [
                {
                    "lines": [
                        {
                            "id": "prefer_zero",
                            "text_equivs": [("2", "two"), ("0", "zero"), ("-1", "minus")],
                        },
                        {
                            "id": "smallest_numeric",
                            "text_equivs": [("5", "five"), ("2", "two")],
                        },
                        {
                            "id": "document_order",
                            "text_equivs": [("bad", "first"), (None, "second")],
                        },
                    ]
                }
            ]
        )
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "page.xml"
        path.write_text(xml, encoding="utf-8")
        _, items = parse_pagexml(path, ground_truth=True)
        self.assertEqual(
            [item["text"] for item in items],
            ["zero", "two", "first"],
        )

    def test_empty_text_is_skipped_and_logged(self):
        xml = _pagexml(
            [
                {
                    "lines": [
                        {"id": "missing", "text_equivs": [(None, None)]},
                        {"id": "spaces", "text": "   "},
                        {"id": "kept", "text": "A"},
                    ]
                }
            ]
        )
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "page.xml"
        path.write_text(xml, encoding="utf-8")
        with self.assertLogs(
            "experiments.downstream_ocr.dataset.pagexml2pagexml_dataset",
            level="WARNING",
        ) as logs:
            _, items = parse_pagexml(path, ground_truth=True)
        self.assertEqual([item["source_id"] for item in items], ["kept"])
        self.assertIn("missing", "\n".join(logs.output))
        self.assertIn("spaces", "\n".join(logs.output))

    def test_adapter_matches_direct_official_call(self):
        gt = _pagexml([{"lines": [{"id": "g1", "text": "A"}, {"id": "g2", "text": "B"}]}])
        pred = _pagexml([{"lines": [{"id": "p1", "text": "B"}, {"id": "p2", "text": "A"}]}])
        result, gt_path, pred_path = self._evaluate(gt, pred)
        image_name, gt_items = parse_pagexml(gt_path, ground_truth=True)
        _, pred_items = parse_pagexml(pred_path, ground_truth=False)
        direct_matches, unmatched = match_gt2pred_simple(
            gt_items,
            pred_items,
            "text",
            image_name,
        )
        adapter_matches = match_pagexml(
            gt_path,
            pred_path,
            image_name=image_name,
        )
        self.assertIsNone(unmatched)
        self.assertEqual(adapter_matches, direct_matches)
        self.assertEqual(
            [match["norm_gt"] for match in result.matches_by_key["page"]],
            [match["norm_gt"] for match in direct_matches],
        )

    def test_prediction_index_zero_keeps_diagnostic_metadata(self):
        xml = _pagexml([{"lines": [{"text": "A"}]}])
        result, _, _ = self._evaluate(xml, xml)
        match = result.matches_by_key["page"][0]
        self.assertEqual(match["pred_idx"], [0])
        self.assertEqual(match["pred_category_type"], "text_all")
        self.assertEqual(match["pred_position"], 0)

    def test_dataset_is_registered_and_rejects_quick_match(self):
        self.assertIs(
            DATASET_REGISTRY.get(DATASET_NAME),
            PageXml2PageXmlDataset,
        )
        with self.assertRaisesRegex(ValueError, "simple_match"):
            PageXml2PageXmlDataset(
                {
                    "dataset": {
                        "ground_truth": {"data_path": "unused"},
                        "prediction": {"data_path": "unused"},
                        "match_method": "quick_match",
                    }
                }
            )


if __name__ == "__main__":
    unittest.main()

