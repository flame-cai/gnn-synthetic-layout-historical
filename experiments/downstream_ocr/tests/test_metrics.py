from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

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
from experiments.downstream_ocr.runners import (
    MethodSpec,
    _write_layout_grounded_gemini_prediction_pagexml,
    method_by_id,
    run_local_finetuning_ladder,
    run_local_gt_layout_finetuning_ladder,
)
from experiments.downstream_ocr.splits import load_or_create_folds_json, make_three_folds
from experiments.downstream_ocr.splits import Fold, ManuscriptPaths
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
    def test_gemini_gt_layout_method_is_disabled(self):
        with self.assertRaisesRegex(ValueError, "disabled"):
            method_by_id("gemini_gt_layout")

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

    def test_layout_grounded_gemini_writer_removes_unpredicted_gt_text(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            layout_xml = Path(tmp_dir) / "layout.xml"
            prediction_xml = Path(tmp_dir) / "prediction.xml"
            layout_xml.write_text(
                """<?xml version='1.0' encoding='UTF-8'?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">
  <Page imageFilename="p.jpg" imageWidth="100" imageHeight="100">
    <TextRegion id="r">
      <TextLine id="line_a" custom="structure_line_id_10">
        <Coords points="0,0 10,0 10,10 0,10"/>
        <TextEquiv><Unicode>gt-a</Unicode></TextEquiv>
      </TextLine>
      <TextLine id="line_b" custom="structure_line_id_11">
        <Coords points="20,0 30,0 30,10 20,10"/>
        <TextEquiv><Unicode>gt-b</Unicode></TextEquiv>
      </TextLine>
    </TextRegion>
  </Page>
</PcGts>
""",
                encoding="utf-8",
            )
            _write_layout_grounded_gemini_prediction_pagexml(
                layout_xml_path=layout_xml,
                predictions_by_structure_line_id={"10": "pred-a"},
                output_path=prediction_xml,
            )
            parsed = load_pagexml(prediction_xml)
            texts = {line.line_id: line.text for line in parsed.lines}
            self.assertEqual(texts["line_a"], "pred-a")
            self.assertEqual(texts["line_b"], "")

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

    def test_seeded_random_folds_are_deterministic_and_cover_pages(self):
        page_ids = tuple(f"p{i}" for i in range(12))
        folds_a = make_three_folds(page_ids, seed=7)
        folds_b = make_three_folds(page_ids, seed=7)
        folds_c = make_three_folds(page_ids, seed=8)
        self.assertEqual(folds_a, folds_b)
        self.assertNotEqual(folds_a, folds_c)

        page_set = set(page_ids)
        for fold in folds_a:
            train = set(fold.train_page_ids)
            test = set(fold.test_page_ids)
            self.assertEqual(len(fold.train_page_ids), 3)
            self.assertFalse(train & test)
            self.assertEqual(train | test, page_set)

    def test_folds_json_is_reused_after_creation(self):
        page_ids = tuple(f"p{i}" for i in range(8))
        with tempfile.TemporaryDirectory() as tmp_dir:
            folds_path = Path(tmp_dir) / "folds.json"
            folds_a = load_or_create_folds_json(
                folds_path,
                manuscript_id="m",
                page_ids=page_ids,
                split_seed=7,
            )
            folds_b = load_or_create_folds_json(
                folds_path,
                manuscript_id="m",
                page_ids=page_ids,
                split_seed=999,
            )
            self.assertEqual(folds_a, folds_b)
            payload = json.loads(folds_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["split_seed"], 7)
            self.assertEqual(len(payload["folds"]), 3)

    def test_finetuning_ladder_trains_once_and_reuses_step_checkpoints(self):
        class FakeRecipe:
            oversampling_policy = "none"
            augmentation_policy = "none"
            history_sample_line_count = 10
            sibling_checkpoint_strategy = "page_cer_selector"
            width_policy = "batch_max_pad"
            lr_scheduler = "none"
            optimizer = "adadelta"
            background_plus_rotation_variant_count = 10
            shuffle_train_each_epoch = True
            lr = 0.2
            num_iter = 60

            def to_dict(self):
                return {
                    "width_policy": self.width_policy,
                    "sibling_checkpoint_strategy": self.sibling_checkpoint_strategy,
                }

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            paths = ManuscriptPaths(
                manuscript_id="m",
                root=root / "manuscript",
                images_dir=root / "images",
                pagexml_dir=root / "pagexml",
                line_images_dir=root / "lines",
                heatmaps_dir=root / "heatmaps",
            )
            fold = Fold(
                fold_id="fold_1",
                train_page_ids=("train_a", "train_b", "train_c"),
                test_page_ids=("test_a", "test_b"),
            )
            prepared_pages = {
                page_id: SimpleNamespace(page_id=page_id)
                for page_id in (*fold.train_page_ids, *fold.test_page_ids)
            }
            train_calls = []
            predict_calls = []

            def fake_fine_tune(prepared, base_checkpoint, output_root, **kwargs):
                step_index = int(kwargs["step_index"])
                train_calls.append(
                    {
                        "page_id": prepared[0].page_id,
                        "base_checkpoint": str(base_checkpoint),
                        "step_index": step_index,
                        "history_count": len(kwargs["history_source_pages"]),
                    }
                )
                return SimpleNamespace(output_checkpoint=str(root / f"checkpoint_{step_index}.pth"))

            def fake_predict(checkpoint, test_pages, output_root, **kwargs):
                output = Path(output_root)
                output.mkdir(parents=True, exist_ok=True)
                predict_calls.append(
                    {
                        "checkpoint": Path(checkpoint).name,
                        "page_ids": tuple(test_pages),
                        "output_root": output,
                        "width_policy": kwargs["width_policy"],
                    }
                )
                return SimpleNamespace(prediction_folder=str(output))

            with patch(
                "experiments.downstream_ocr.runners._prepare_gt_layout_pages",
                return_value=prepared_pages,
            ), patch(
                "experiments.downstream_ocr.runners._load_gui_runtime_ocr_recipe",
                return_value=FakeRecipe(),
            ):
                prediction_dirs = run_local_gt_layout_finetuning_ladder(
                    paths=paths,
                    fold=fold,
                    methods=(
                        MethodSpec(
                            "annotation_tool_gt_layout_ft_1",
                            "ft1",
                            uses_gt_layout=True,
                            uses_finetuning=True,
                            finetune_page_count=1,
                        ),
                        MethodSpec(
                            "annotation_tool_gt_layout_ft_3",
                            "ft3",
                            uses_gt_layout=True,
                            uses_finetuning=True,
                            finetune_page_count=3,
                        ),
                    ),
                    output_root=root / "out",
                    fine_tune_fn=fake_fine_tune,
                    predict_fn=fake_predict,
                )

            self.assertEqual([call["page_id"] for call in train_calls], ["train_a", "train_b", "train_c"])
            self.assertEqual([call["history_count"] for call in train_calls], [0, 1, 2])
            self.assertEqual(Path(train_calls[1]["base_checkpoint"]).name, "checkpoint_1.pth")
            self.assertEqual(Path(train_calls[2]["base_checkpoint"]).name, "checkpoint_2.pth")
            self.assertEqual([call["checkpoint"] for call in predict_calls], ["checkpoint_1.pth", "checkpoint_3.pth"])
            self.assertEqual(set(prediction_dirs), {"annotation_tool_gt_layout_ft_1", "annotation_tool_gt_layout_ft_3"})

    def test_finetuning_ladder_reuses_checkpoints_across_test_layout_conditions(self):
        class FakeRecipe:
            oversampling_policy = "none"
            augmentation_policy = "none"
            history_sample_line_count = 10
            sibling_checkpoint_strategy = "page_cer_selector"
            width_policy = "batch_max_pad"
            lr_scheduler = "none"
            optimizer = "adadelta"
            background_plus_rotation_variant_count = 10
            shuffle_train_each_epoch = True
            lr = 0.2
            num_iter = 60

            def to_dict(self):
                return {"width_policy": self.width_policy}

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            paths = ManuscriptPaths(
                manuscript_id="m",
                root=root / "manuscript",
                images_dir=root / "images",
                pagexml_dir=root / "pagexml",
                line_images_dir=root / "lines",
                heatmaps_dir=root / "heatmaps",
            )
            fold = Fold(
                fold_id="fold_1",
                train_page_ids=("train_a", "train_b", "train_c"),
                test_page_ids=("test_a", "test_b"),
            )
            gt_pages = {
                page_id: SimpleNamespace(page_id=page_id, layout_condition="gt")
                for page_id in (*fold.train_page_ids, *fold.test_page_ids)
            }
            predicted_pages = {
                page_id: SimpleNamespace(page_id=page_id, layout_condition="predicted")
                for page_id in fold.test_page_ids
            }
            train_calls = []
            predict_calls = []

            def fake_fine_tune(prepared, base_checkpoint, output_root, **kwargs):
                step_index = int(kwargs["step_index"])
                train_calls.append((prepared[0].page_id, prepared[0].layout_condition))
                return SimpleNamespace(output_checkpoint=str(root / f"checkpoint_{step_index}.pth"))

            def fake_predict(checkpoint, test_pages, output_root, **kwargs):
                output = Path(output_root)
                output.mkdir(parents=True, exist_ok=True)
                predict_calls.append(
                    {
                        "method_id": output.parent.parent.name,
                        "checkpoint": Path(checkpoint).name,
                        "layouts": {page.layout_condition for page in test_pages.values()},
                    }
                )
                return SimpleNamespace(prediction_folder=str(output))

            methods = (
                MethodSpec(
                    "annotation_tool_pred_layout_ft_1",
                    "pred ft1",
                    uses_gt_layout=False,
                    uses_finetuning=True,
                    finetune_page_count=1,
                ),
                MethodSpec(
                    "annotation_tool_gt_layout_ft_1",
                    "gt ft1",
                    uses_gt_layout=True,
                    uses_finetuning=True,
                    finetune_page_count=1,
                ),
                MethodSpec(
                    "annotation_tool_pred_layout_ft_3",
                    "pred ft3",
                    uses_gt_layout=False,
                    uses_finetuning=True,
                    finetune_page_count=3,
                ),
                MethodSpec(
                    "annotation_tool_gt_layout_ft_3",
                    "gt ft3",
                    uses_gt_layout=True,
                    uses_finetuning=True,
                    finetune_page_count=3,
                ),
            )

            with patch(
                "experiments.downstream_ocr.runners._prepare_gt_layout_pages",
                return_value=gt_pages,
            ), patch(
                "experiments.downstream_ocr.runners._load_gui_runtime_ocr_recipe",
                return_value=FakeRecipe(),
            ):
                prediction_dirs = run_local_finetuning_ladder(
                    paths=paths,
                    fold=fold,
                    methods=methods,
                    output_root=root / "out",
                    fine_tune_fn=fake_fine_tune,
                    predict_fn=fake_predict,
                    predicted_layout_test_pages=predicted_pages,
                )

            self.assertEqual(
                train_calls,
                [("train_a", "gt"), ("train_b", "gt"), ("train_c", "gt")],
            )
            by_method = {call["method_id"]: call for call in predict_calls}
            self.assertEqual(by_method["annotation_tool_pred_layout_ft_1"]["checkpoint"], "checkpoint_1.pth")
            self.assertEqual(by_method["annotation_tool_gt_layout_ft_1"]["checkpoint"], "checkpoint_1.pth")
            self.assertEqual(by_method["annotation_tool_pred_layout_ft_3"]["checkpoint"], "checkpoint_3.pth")
            self.assertEqual(by_method["annotation_tool_gt_layout_ft_3"]["checkpoint"], "checkpoint_3.pth")
            self.assertEqual(by_method["annotation_tool_pred_layout_ft_1"]["layouts"], {"predicted"})
            self.assertEqual(by_method["annotation_tool_gt_layout_ft_1"]["layouts"], {"gt"})
            self.assertEqual(set(prediction_dirs), {method.method_id for method in methods})


if __name__ == "__main__":
    unittest.main()
