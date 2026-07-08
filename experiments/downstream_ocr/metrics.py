from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Iterable

import cv2
import networkx as nx
import numpy as np
from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree

from .pagexml import PageXmlPage, TextLine, iter_polygon_parts
from .text import levenshtein_distance, normalize_text


FAILURE_STATUSES = {
    "api_timeout",
    "api_error",
    "empty_response",
    "json_parse_error",
    "json_schema_error",
    "adapter_error",
    "other_output_error",
}


def safe_divide(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def polygon_iou(gt_polygon: BaseGeometry, pred_polygon: BaseGeometry) -> float:
    intersection = gt_polygon.intersection(pred_polygon).area
    union = gt_polygon.union(pred_polygon).area
    return float(intersection / union) if union > 0 else 0.0


def intersection_over_min_area(gt_polygon: BaseGeometry, pred_polygon: BaseGeometry) -> float:
    intersection = gt_polygon.intersection(pred_polygon).area
    denom = min(gt_polygon.area, pred_polygon.area)
    return float(intersection / denom) if denom > 0 else 0.0


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


def page_text(lines: Iterable[TextLine]) -> str:
    texts = [
        normalized
        for normalized in (normalize_text(line.text) for line in order_lines_geometrically(lines))
        if normalized
    ]
    return " ".join(texts)


def page_cer(gt_lines: list[TextLine], pred_lines: list[TextLine]) -> dict:
    gt_page_text = page_text(gt_lines)
    pred_page_text = page_text(pred_lines)
    distance = levenshtein_distance(gt_page_text, pred_page_text)
    gt_chars = len(gt_page_text)
    return {
        "page_cer_distance": distance,
        "page_cer_gt_chars": gt_chars,
        "page_cer": distance / gt_chars if gt_chars > 0 else 0.0,
    }


class _DisjointSet:
    def __init__(self):
        self.parent: dict[tuple[str, int], tuple[str, int]] = {}

    def add(self, item: tuple[str, int]) -> None:
        self.parent.setdefault(item, item)

    def find(self, item: tuple[str, int]) -> tuple[str, int]:
        self.add(item)
        parent = self.parent[item]
        if parent != item:
            parent = self.find(parent)
            self.parent[item] = parent
        return parent

    def union(self, left: tuple[str, int], right: tuple[str, int]) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root

    def components(self) -> list[set[tuple[str, int]]]:
        grouped: dict[tuple[str, int], set[tuple[str, int]]] = {}
        for item in list(self.parent):
            grouped.setdefault(self.find(item), set()).add(item)
        return list(grouped.values())


def build_textedit_groups(
    gt_lines: list[TextLine],
    pred_lines: list[TextLine],
    *,
    threshold: float = 0.50,
) -> list[tuple[str, str]]:
    dsu = _DisjointSet()
    for gt_idx in range(len(gt_lines)):
        dsu.add(("gt", gt_idx))
    for pred_idx in range(len(pred_lines)):
        dsu.add(("pred", pred_idx))
    for gt_idx, gt_line in enumerate(gt_lines):
        for pred_idx, pred_line in enumerate(pred_lines):
            if intersection_over_min_area(gt_line.polygon, pred_line.polygon) >= threshold:
                dsu.union(("gt", gt_idx), ("pred", pred_idx))

    groups: list[tuple[str, str]] = []
    for component in dsu.components():
        gt_group = [gt_lines[index] for side, index in component if side == "gt"]
        pred_group = [pred_lines[index] for side, index in component if side == "pred"]
        gt_text = " ".join(
            text for text in (normalize_text(line.text) for line in order_lines_geometrically(gt_group)) if text
        )
        pred_text = " ".join(
            text for text in (normalize_text(line.text) for line in order_lines_geometrically(pred_group)) if text
        )
        groups.append((gt_text, pred_text))
    return groups


def line_group_textedit(gt_lines: list[TextLine], pred_lines: list[TextLine]) -> dict:
    edit_sum = 0
    max_length_sum = 0
    for gt_text, pred_text in build_textedit_groups(gt_lines, pred_lines):
        edit_sum += levenshtein_distance(gt_text, pred_text)
        max_length_sum += max(len(gt_text), len(pred_text))
    return {
        "textedit_distance_sum": edit_sum,
        "textedit_max_length_sum": max_length_sum,
        "textedit": edit_sum / max_length_sum if max_length_sum > 0 else 0.0,
    }


def evaluate_page(
    *,
    manuscript_id: str,
    fold_id: str,
    page_id: str,
    gt_page: PageXmlPage,
    pred_page: PageXmlPage,
    status: str = "success",
    method_id: str | None = None,
) -> dict:
    if gt_page.width != pred_page.width or gt_page.height != pred_page.height:
        raise ValueError(
            f"PAGE dimensions differ for {page_id}: "
            f"GT {gt_page.width}x{gt_page.height}, prediction {pred_page.width}x{pred_page.height}."
        )
    gt_lines = list(gt_page.lines)
    pred_lines = [] if status in FAILURE_STATUSES else list(pred_page.lines)
    iou_matrix = compute_iou_matrix(gt_lines, pred_lines)
    object_50 = object_metrics(iou_matrix, len(gt_lines), len(pred_lines), 0.50)
    object_75 = object_metrics(iou_matrix, len(gt_lines), len(pred_lines), 0.75)
    record = {
        "manuscript_id": manuscript_id,
        "fold_id": fold_id,
        "page_id": page_id,
        "status": status,
        "num_gt_lines": len(gt_lines),
        "num_pred_lines": len(pred_lines),
    }
    if method_id is not None:
        record["method_id"] = method_id
    for payload in (
        {key: value for key, value in object_50.items() if not key.startswith("matches_")},
        {key: value for key, value in object_75.items() if not key.startswith("matches_")},
        pixel_metrics(gt_lines, pred_lines, gt_page.width, gt_page.height),
        page_cer(gt_lines, pred_lines),
        line_group_textedit(gt_lines, pred_lines),
    ):
        record.update(payload)
    return record


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def aggregate_page_records(records: Iterable[dict]) -> dict:
    rows = list(records)
    count = len(rows)
    tp_50 = sum(int(row["tp_50"]) for row in rows)
    fp_50 = sum(int(row["fp_50"]) for row in rows)
    fn_50 = sum(int(row["fn_50"]) for row in rows)
    tp_75 = sum(int(row["tp_75"]) for row in rows)
    fp_75 = sum(int(row["fp_75"]) for row in rows)
    fn_75 = sum(int(row["fn_75"]) for row in rows)
    pixel_tp = sum(int(row["pixel_tp"]) for row in rows)
    pixel_fp = sum(int(row["pixel_fp"]) for row in rows)
    pixel_fn = sum(int(row["pixel_fn"]) for row in rows)
    cer_distance = sum(int(row["page_cer_distance"]) for row in rows)
    cer_chars = sum(int(row["page_cer_gt_chars"]) for row in rows)
    textedit_distance = sum(int(row["textedit_distance_sum"]) for row in rows)
    textedit_denominator = sum(int(row["textedit_max_length_sum"]) for row in rows)
    page_cers = [float(row["page_cer"]) for row in rows]
    textedits = [float(row["textedit"]) for row in rows]
    successful = sum(1 for row in rows if row.get("status") == "success")
    return {
        "page_count": count,
        "valid_output_rate": safe_divide(successful, count),
        "tp_50": tp_50,
        "fp_50": fp_50,
        "fn_50": fn_50,
        "object_precision_50": safe_divide(tp_50, tp_50 + fp_50),
        "object_recall_50": safe_divide(tp_50, tp_50 + fn_50),
        "object_g_f1_50": safe_divide(2 * tp_50, 2 * tp_50 + fp_50 + fn_50),
        "tp_75": tp_75,
        "fp_75": fp_75,
        "fn_75": fn_75,
        "object_precision_75": safe_divide(tp_75, tp_75 + fp_75),
        "object_recall_75": safe_divide(tp_75, tp_75 + fn_75),
        "object_g_f1_75": safe_divide(2 * tp_75, 2 * tp_75 + fp_75 + fn_75),
        "pixel_tp": pixel_tp,
        "pixel_fp": pixel_fp,
        "pixel_fn": pixel_fn,
        "pixel_precision": safe_divide(pixel_tp, pixel_tp + pixel_fp),
        "pixel_recall": safe_divide(pixel_tp, pixel_tp + pixel_fn),
        "pixel_f1": safe_divide(2 * pixel_tp, 2 * pixel_tp + pixel_fp + pixel_fn),
        "mean_page_cer": float(statistics.mean(page_cers)) if page_cers else 0.0,
        "median_page_cer": _median(page_cers),
        "micro_page_cer": safe_divide(cer_distance, cer_chars),
        "mean_textedit": float(statistics.mean(textedits)) if textedits else 0.0,
        "median_textedit": _median(textedits),
        "micro_textedit": safe_divide(textedit_distance, textedit_denominator),
    }


@dataclass(frozen=True)
class EvaluationResult:
    page_records: tuple[dict, ...]
    aggregate: dict


def evaluate_pages(page_records: Iterable[dict]) -> EvaluationResult:
    rows = tuple(page_records)
    return EvaluationResult(page_records=rows, aggregate=aggregate_page_records(rows))
