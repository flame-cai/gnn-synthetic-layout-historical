from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Iterable

import cv2
import networkx as nx
import numpy as np
from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree

from .dataset.pagexml2pagexml_dataset import (
    evaluate_text_line_items,
    items_from_text_lines,
)
from .pagexml import PageXmlPage, TextLine, iter_polygon_parts
from .text import levenshtein_distance, normalize_text


FAILURE_STATUSES = {
    "api_timeout",
    "api_error",
    "empty_response",
    "json_parse_error",
    "json_schema_error",
    "html_parse_error",
    "adapter_error",
    "other_output_error",
}


def safe_divide(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def polygon_iou(gt_polygon: BaseGeometry, pred_polygon: BaseGeometry) -> float:
    intersection = gt_polygon.intersection(pred_polygon).area
    union = gt_polygon.union(pred_polygon).area
    return float(intersection / union) if union > 0 else 0.0


def _query_indices(tree: STRtree, geometries: list[BaseGeometry], query_geometry: BaseGeometry) -> list[int]:
    result = tree.query(query_geometry)
    if len(result) == 0:
        return []
    first = result[0]
    if isinstance(first, (int, np.integer)):
        return [int(item) for item in result]
    by_id = {id(geometry): index for index, geometry in enumerate(geometries)}
    return [by_id[id(geometry)] for geometry in result]


def compute_iou_matrix(gt_lines: list[TextLine], pred_lines: list[TextLine]) -> np.ndarray:
    matrix = np.zeros((len(gt_lines), len(pred_lines)), dtype=float)
    if not gt_lines or not pred_lines:
        return matrix
    pred_polygons = [line.polygon for line in pred_lines]
    tree = STRtree(pred_polygons)
    for gt_index, gt_line in enumerate(gt_lines):
        for pred_index in _query_indices(tree, pred_polygons, gt_line.polygon):
            pred_polygon = pred_polygons[pred_index]
            if not gt_line.polygon.bounds or not pred_polygon.bounds:
                continue
            matrix[gt_index, pred_index] = polygon_iou(gt_line.polygon, pred_polygon)
    return matrix


def match_objects(iou_matrix: np.ndarray, threshold: float) -> list[tuple[int, int, float]]:
    num_gt, num_pred = iou_matrix.shape
    graph = nx.Graph()
    for gt_idx in range(num_gt):
        graph.add_node(("gt", gt_idx), bipartite=0)
    for pred_idx in range(num_pred):
        graph.add_node(("pred", pred_idx), bipartite=1)
    for gt_idx in range(num_gt):
        for pred_idx in range(num_pred):
            iou = float(iou_matrix[gt_idx, pred_idx])
            if iou >= threshold:
                graph.add_edge(
                    ("gt", gt_idx),
                    ("pred", pred_idx),
                    weight=int(round(iou * 1_000_000)),
                )

    matching = nx.algorithms.matching.max_weight_matching(
        graph,
        maxcardinality=True,
        weight="weight",
    )
    results: list[tuple[int, int, float]] = []
    for node_a, node_b in matching:
        if node_a[0] == "gt":
            gt_node, pred_node = node_a, node_b
        else:
            gt_node, pred_node = node_b, node_a
        gt_idx = gt_node[1]
        pred_idx = pred_node[1]
        results.append((gt_idx, pred_idx, float(iou_matrix[gt_idx, pred_idx])))
    return sorted(results)


def object_metrics(iou_matrix: np.ndarray, num_gt: int, num_pred: int, threshold: float) -> dict:
    matches = match_objects(iou_matrix, threshold)
    true_positive = len(matches)
    false_positive = num_pred - true_positive
    false_negative = num_gt - true_positive
    suffix = "50" if abs(threshold - 0.50) < 1e-9 else "75"
    return {
        f"tp_{suffix}": true_positive,
        f"fp_{suffix}": false_positive,
        f"fn_{suffix}": false_negative,
        f"object_precision_{suffix}": safe_divide(true_positive, true_positive + false_positive),
        f"object_recall_{suffix}": safe_divide(true_positive, true_positive + false_negative),
        f"object_g_f1_{suffix}": safe_divide(
            2 * true_positive,
            2 * true_positive + false_positive + false_negative,
        ),
        f"matches_{suffix}": matches,
    }


def polygons_to_mask(polygons: Iterable[BaseGeometry], width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for geometry in polygons:
        for polygon in iter_polygon_parts(geometry):
            coords = np.asarray(
                [[int(round(x_val)), int(round(y_val))] for x_val, y_val in polygon.exterior.coords[:-1]],
                dtype=np.int32,
            )
            if len(coords) >= 3:
                cv2.fillPoly(mask, [coords], color=1)
    return mask.astype(bool)


def pixel_metrics(gt_lines: list[TextLine], pred_lines: list[TextLine], width: int, height: int) -> dict:
    gt_mask = polygons_to_mask((line.polygon for line in gt_lines), width, height)
    pred_mask = polygons_to_mask((line.polygon for line in pred_lines), width, height)
    pixel_tp = int(np.logical_and(gt_mask, pred_mask).sum())
    pixel_fp = int(np.logical_and(~gt_mask, pred_mask).sum())
    pixel_fn = int(np.logical_and(gt_mask, ~pred_mask).sum())
    return {
        "pixel_tp": pixel_tp,
        "pixel_fp": pixel_fp,
        "pixel_fn": pixel_fn,
        "pixel_precision": safe_divide(pixel_tp, pixel_tp + pixel_fp),
        "pixel_recall": safe_divide(pixel_tp, pixel_tp + pixel_fn),
        "pixel_f1": safe_divide(2 * pixel_tp, 2 * pixel_tp + pixel_fp + pixel_fn),
    }


def order_lines_geometrically(lines: Iterable[TextLine]) -> list[TextLine]:
    return sorted(
        lines,
        key=lambda line: (
            line.polygon.centroid.y,
            line.polygon.centroid.x,
            line.line_id,
        ),
    )


def page_text(
    lines: Iterable[TextLine],
    *,
    preserve_input_order: bool = False,
) -> str:
    ordered_lines = (
        list(lines) if preserve_input_order else order_lines_geometrically(lines)
    )
    texts = [
        normalized
        for normalized in (normalize_text(line.text) for line in ordered_lines)
        if normalized
    ]
    return " ".join(texts)


def page_cer(
    gt_lines: list[TextLine],
    pred_lines: list[TextLine],
    *,
    predicted_lines_in_output_order: bool = False,
) -> dict:
    gt_page_text = page_text(gt_lines)
    pred_page_text = page_text(
        pred_lines,
        preserve_input_order=predicted_lines_in_output_order,
    )
    distance = levenshtein_distance(gt_page_text, pred_page_text)
    gt_chars = len(gt_page_text)
    return {
        "page_cer_distance": distance,
        "page_cer_gt_chars": gt_chars,
        "page_cer": distance / gt_chars if gt_chars > 0 else 0.0,
    }


def unordered_textline_textedit(
    gt_lines: list[TextLine],
    pred_lines: list[TextLine],
    *,
    image_name: str,
) -> dict:
    """Evaluate in-memory PAGE TextLines through the official text-only path."""
    return evaluate_text_line_items(
        items_from_text_lines(gt_lines, ground_truth=True),
        items_from_text_lines(pred_lines, ground_truth=False),
        image_name=image_name,
    )


def evaluate_page(
    *,
    manuscript_id: str,
    fold_id: str,
    page_id: str,
    gt_page: PageXmlPage,
    pred_page: PageXmlPage,
    status: str = "success",
    method_id: str | None = None,
    calculate_textedit: bool = True,
    calculate_layout_metrics: bool = True,
    calculate_page_cer: bool = True,
    page_cer_predicted_lines_in_output_order: bool = False,
) -> dict:
    if gt_page.width != pred_page.width or gt_page.height != pred_page.height:
        raise ValueError(
            f"PAGE dimensions differ for {page_id}: "
            f"GT {gt_page.width}x{gt_page.height}, prediction {pred_page.width}x{pred_page.height}."
        )
    gt_lines = list(gt_page.lines)
    pred_lines = [] if status in FAILURE_STATUSES else list(pred_page.lines)
    record = {
        "manuscript_id": manuscript_id,
        "fold_id": fold_id,
        "page_id": page_id,
        "status": status,
        "num_gt_lines": len(gt_lines),
        "num_pred_lines": len(pred_lines),
        "layout_metrics_available": bool(calculate_layout_metrics),
        "page_cer_available": bool(calculate_page_cer),
    }
    if method_id is not None:
        record["method_id"] = method_id
    payloads = []
    if calculate_layout_metrics:
        iou_matrix = compute_iou_matrix(gt_lines, pred_lines)
        object_50 = object_metrics(iou_matrix, len(gt_lines), len(pred_lines), 0.50)
        object_75 = object_metrics(iou_matrix, len(gt_lines), len(pred_lines), 0.75)
        payloads.extend(
            [
                {
                    key: value
                    for key, value in object_50.items()
                    if not key.startswith("matches_")
                },
                {
                    key: value
                    for key, value in object_75.items()
                    if not key.startswith("matches_")
                },
                pixel_metrics(gt_lines, pred_lines, gt_page.width, gt_page.height),
            ]
        )
    else:
        payloads.append(
            {
                key: None
                for key in (
                    "tp_50",
                    "fp_50",
                    "fn_50",
                    "object_precision_50",
                    "object_recall_50",
                    "object_g_f1_50",
                    "tp_75",
                    "fp_75",
                    "fn_75",
                    "object_precision_75",
                    "object_recall_75",
                    "object_g_f1_75",
                    "pixel_tp",
                    "pixel_fp",
                    "pixel_fn",
                    "pixel_precision",
                    "pixel_recall",
                    "pixel_f1",
                )
            }
        )
    if calculate_page_cer:
        payloads.append(
            page_cer(
                gt_lines,
                pred_lines,
                predicted_lines_in_output_order=page_cer_predicted_lines_in_output_order,
            )
        )
    else:
        payloads.append(
            {
                "page_cer_distance": None,
                "page_cer_gt_chars": None,
                "page_cer": None,
            }
        )
    if calculate_textedit:
        payloads.append(
            unordered_textline_textedit(
                gt_lines,
                pred_lines,
                image_name=gt_page.image_filename,
            )
        )
    for payload in payloads:
        record.update(payload)
    return record


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def aggregate_page_records(records: Iterable[dict]) -> dict:
    rows = list(records)
    count = len(rows)
    layout_rows = [
        row
        for row in rows
        if row.get("layout_metrics_available", True)
        and row.get("tp_50") is not None
    ]
    cer_rows = [
        row
        for row in rows
        if row.get("page_cer_available", True)
        and row.get("page_cer") is not None
    ]
    layout_metrics_available = bool(layout_rows) or not rows
    page_cer_available = bool(cer_rows) or not rows
    tp_50 = sum(int(row["tp_50"]) for row in layout_rows)
    fp_50 = sum(int(row["fp_50"]) for row in layout_rows)
    fn_50 = sum(int(row["fn_50"]) for row in layout_rows)
    tp_75 = sum(int(row["tp_75"]) for row in layout_rows)
    fp_75 = sum(int(row["fp_75"]) for row in layout_rows)
    fn_75 = sum(int(row["fn_75"]) for row in layout_rows)
    pixel_tp = sum(int(row["pixel_tp"]) for row in layout_rows)
    pixel_fp = sum(int(row["pixel_fp"]) for row in layout_rows)
    pixel_fn = sum(int(row["pixel_fn"]) for row in layout_rows)
    cer_distance = sum(int(row["page_cer_distance"]) for row in cer_rows)
    cer_chars = sum(int(row["page_cer_gt_chars"]) for row in cer_rows)
    textedit_distance = sum(int(row["textedit_distance_sum"]) for row in rows)
    textedit_denominator = sum(int(row["textedit_max_length_sum"]) for row in rows)
    textedit_sample_ratio_sum = sum(
        float(row.get("textedit_sample_ratio_sum", row["textedit"]))
        for row in rows
    )
    textedit_sample_count = sum(
        int(row.get("textedit_sample_count", 1))
        for row in rows
    )
    page_cers = [float(row["page_cer"]) for row in cer_rows]
    textedits = [
        float(row.get("textedit_all_page_avg", row["textedit"]))
        for row in rows
    ]
    successful = sum(1 for row in rows if row.get("status") == "success")
    textedit_all_page_avg = (
        float(statistics.mean(textedits)) if textedits else 0.0
    )
    textedit_edit_whole = safe_divide(
        textedit_distance,
        textedit_denominator,
    )
    textedit_edit_sample_avg = safe_divide(
        textedit_sample_ratio_sum,
        textedit_sample_count,
    )
    devanagari_textedit_rows = [
        row
        for row in rows
        if row.get("devanagari_textedit_distance_sum") is not None
        and row.get("devanagari_textedit_max_length_sum") is not None
        and row.get(
            "devanagari_textedit_all_page_avg",
            row.get("devanagari_textedit"),
        )
        is not None
    ]
    devanagari_textedit_available = bool(devanagari_textedit_rows) or not rows
    devanagari_textedit_distance = sum(
        int(row["devanagari_textedit_distance_sum"])
        for row in devanagari_textedit_rows
    )
    devanagari_textedit_denominator = sum(
        int(row["devanagari_textedit_max_length_sum"])
        for row in devanagari_textedit_rows
    )
    devanagari_textedit_sample_ratio_sum = sum(
        float(
            row.get(
                "devanagari_textedit_sample_ratio_sum",
                row.get("devanagari_textedit", 0.0),
            )
        )
        for row in devanagari_textedit_rows
    )
    devanagari_textedit_sample_count = sum(
        int(row.get("devanagari_textedit_sample_count", 1))
        for row in devanagari_textedit_rows
    )
    devanagari_textedits = [
        float(
            row["devanagari_textedit_all_page_avg"]
            if row.get("devanagari_textedit_all_page_avg") is not None
            else row.get("devanagari_textedit", 0.0)
        )
        for row in devanagari_textedit_rows
    ]
    devanagari_textedit_all_page_avg = (
        float(statistics.mean(devanagari_textedits))
        if devanagari_textedits
        else (0.0 if not rows else None)
    )
    devanagari_textedit_edit_whole = (
        safe_divide(
            devanagari_textedit_distance,
            devanagari_textedit_denominator,
        )
        if devanagari_textedit_available
        else None
    )
    devanagari_textedit_edit_sample_avg = (
        safe_divide(
            devanagari_textedit_sample_ratio_sum,
            devanagari_textedit_sample_count,
        )
        if devanagari_textedit_available
        else None
    )
    return {
        "page_count": count,
        "valid_output_rate": safe_divide(successful, count),
        "layout_metric_page_count": len(layout_rows),
        "page_cer_page_count": len(cer_rows),
        "tp_50": tp_50 if layout_metrics_available else None,
        "fp_50": fp_50 if layout_metrics_available else None,
        "fn_50": fn_50 if layout_metrics_available else None,
        "object_precision_50": (
            safe_divide(tp_50, tp_50 + fp_50)
            if layout_metrics_available
            else None
        ),
        "object_recall_50": (
            safe_divide(tp_50, tp_50 + fn_50)
            if layout_metrics_available
            else None
        ),
        "object_g_f1_50": (
            safe_divide(2 * tp_50, 2 * tp_50 + fp_50 + fn_50)
            if layout_metrics_available
            else None
        ),
        "tp_75": tp_75 if layout_metrics_available else None,
        "fp_75": fp_75 if layout_metrics_available else None,
        "fn_75": fn_75 if layout_metrics_available else None,
        "object_precision_75": (
            safe_divide(tp_75, tp_75 + fp_75)
            if layout_metrics_available
            else None
        ),
        "object_recall_75": (
            safe_divide(tp_75, tp_75 + fn_75)
            if layout_metrics_available
            else None
        ),
        "object_g_f1_75": (
            safe_divide(2 * tp_75, 2 * tp_75 + fp_75 + fn_75)
            if layout_metrics_available
            else None
        ),
        "pixel_tp": pixel_tp if layout_metrics_available else None,
        "pixel_fp": pixel_fp if layout_metrics_available else None,
        "pixel_fn": pixel_fn if layout_metrics_available else None,
        "pixel_precision": (
            safe_divide(pixel_tp, pixel_tp + pixel_fp)
            if layout_metrics_available
            else None
        ),
        "pixel_recall": (
            safe_divide(pixel_tp, pixel_tp + pixel_fn)
            if layout_metrics_available
            else None
        ),
        "pixel_f1": (
            safe_divide(2 * pixel_tp, 2 * pixel_tp + pixel_fp + pixel_fn)
            if layout_metrics_available
            else None
        ),
        "mean_page_cer": (
            float(statistics.mean(page_cers))
            if page_cers
            else (0.0 if not rows else None)
        ),
        "median_page_cer": (
            _median(page_cers) if page_cers else (0.0 if not rows else None)
        ),
        "micro_page_cer": (
            safe_divide(cer_distance, cer_chars) if page_cer_available else None
        ),
        "mean_textedit": textedit_all_page_avg,
        "median_textedit": _median(textedits),
        "micro_textedit": textedit_edit_whole,
        "textedit_all_page_avg": textedit_all_page_avg,
        "textedit_edit_whole": textedit_edit_whole,
        "textedit_edit_sample_avg": textedit_edit_sample_avg,
        "devanagari_textedit_page_count": len(devanagari_textedit_rows),
        "mean_devanagari_textedit": devanagari_textedit_all_page_avg,
        "median_devanagari_textedit": (
            _median(devanagari_textedits)
            if devanagari_textedits
            else (0.0 if not rows else None)
        ),
        "micro_devanagari_textedit": devanagari_textedit_edit_whole,
        "devanagari_textedit_all_page_avg": (
            devanagari_textedit_all_page_avg
        ),
        "devanagari_textedit_edit_whole": (
            devanagari_textedit_edit_whole
        ),
        "devanagari_textedit_edit_sample_avg": (
            devanagari_textedit_edit_sample_avg
        ),
    }


@dataclass(frozen=True)
class EvaluationResult:
    page_records: tuple[dict, ...]
    aggregate: dict


def evaluate_pages(page_records: Iterable[dict]) -> EvaluationResult:
    rows = tuple(page_records)
    return EvaluationResult(page_records=rows, aggregate=aggregate_page_records(rows))
