from __future__ import annotations

import importlib.util
import json
import math
import shutil
from dataclasses import asdict
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from .pagexml import (
    PAGE_XML_NAMESPACE,
    count_text_lines_with_text_and_baseline,
    load_baseline_records,
    remove_textline_coords,
    set_textline_coords_by_numeric_id,
)
from .types import TextLineSegmentationRequest, TextLineSegmentationResult


DEFAULT_LEGACY_AXIS_BOUND_CONFIG = {
    "BINARIZE_THRESHOLD": 0.5098,
    "BBOX_PAD_V": 0.7,
    "BBOX_PAD_H": 0.5,
    "CC_SIZE_THRESHOLD_RATIO": 0.4,
}


def _load_canonical_segment_module():
    repo_root = Path(__file__).resolve().parents[3]
    module_path = repo_root / "src" / "gnn_inference" / "segment_from_point_clusters.py"
    spec = importlib.util.spec_from_file_location("_legacy_axis_bound_segment_from_point_clusters", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load canonical segment_from_point_clusters from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_SEGMENT_MODULE = None


def _segment_module():
    global _SEGMENT_MODULE
    if _SEGMENT_MODULE is None:
        _SEGMENT_MODULE = _load_canonical_segment_module()
    return _SEGMENT_MODULE


def _point_segment_distance(point, start, end):
    px_val, py_val = point
    ax_val, ay_val = start
    bx_val, by_val = end
    vx_val = bx_val - ax_val
    vy_val = by_val - ay_val
    wx_val = px_val - ax_val
    wy_val = py_val - ay_val
    denom = vx_val * vx_val + vy_val * vy_val
    if denom == 0:
        return math.hypot(px_val - ax_val, py_val - ay_val)
    ratio = max(0.0, min(1.0, (wx_val * vx_val + wy_val * vy_val) / denom))
    nearest_x = ax_val + ratio * vx_val
    nearest_y = ay_val + ratio * vy_val
    return math.hypot(px_val - nearest_x, py_val - nearest_y)


def _distance_to_baseline(point, baseline_points):
    if not baseline_points:
        return float("inf")
    if len(baseline_points) == 1:
        return math.hypot(point[0] - baseline_points[0][0], point[1] - baseline_points[0][1])
    return min(
        _point_segment_distance(point, baseline_points[index], baseline_points[index + 1])
        for index in range(len(baseline_points) - 1)
    )


def _normalise_config(config):
    merged = dict(DEFAULT_LEGACY_AXIS_BOUND_CONFIG)
    merged.update(dict(config or {}))
    merged["BINARIZE_THRESHOLD"] = float(merged["BINARIZE_THRESHOLD"])
    merged["BBOX_PAD_V"] = float(merged["BBOX_PAD_V"])
    merged["BBOX_PAD_H"] = float(merged["BBOX_PAD_H"])
    merged["CC_SIZE_THRESHOLD_RATIO"] = float(merged["CC_SIZE_THRESHOLD_RATIO"])
    merged["include_empty_text_lines"] = bool(
        merged.get("include_empty_text_lines", merged.get("INCLUDE_EMPTY_TEXT_LINES", False))
    )
    return merged


def _build_baseline_component_nodes(
    xml_path: Path,
    image_path: Path,
    heatmap_path: Path,
    binarize_threshold: float,
    include_empty_text_lines: bool = False,
):
    baseline_records = load_baseline_records(xml_path, include_empty_text_lines=include_empty_text_lines)
    if not baseline_records:
        return np.empty((0, 3)), np.empty((0,), dtype=int), {
            "heatmap_box_count": 0,
            "assigned_box_count": 0,
            "heatmap_box_assignment_rate": None,
            "baseline_line_count": 0,
            "max_assignment_distance": None,
            "mean_assignment_distance": None,
        }

    module = _segment_module()
    image = module.loadImage(str(image_path))
    heatmap = module.loadImage(str(heatmap_path))
    if heatmap.ndim == 3:
        heatmap = heatmap[:, :, 0]

    image_height, image_width = image.shape[:2]
    heatmap_height, heatmap_width = heatmap.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (image_width, image_height), interpolation=cv2.INTER_LINEAR)
    bounding_boxes = module.gen_bounding_boxes(heatmap_resized, binarize_threshold)

    x_to_heatmap = heatmap_width / image_width
    y_to_heatmap = heatmap_height / image_height
    synthetic_nodes = []
    synthetic_labels = []
    assigned_distances = []

    for x_val, y_val, width, height in bounding_boxes:
        center_x = x_val + (width / 2.0)
        center_y = y_val + (height / 2.0)
        best_record = min(
            (
                (_distance_to_baseline((center_x, center_y), record["baseline_points"]), record)
                for record in baseline_records
            ),
            key=lambda item: item[0],
            default=None,
        )
        if best_record is None:
            continue

        distance, record = best_record
        max_distance = max(20.0, float(height) * 2.5)
        if distance > max_distance:
            continue

        synthetic_nodes.append(
            [
                center_x * x_to_heatmap,
                center_y * y_to_heatmap,
                max(float(width) * x_to_heatmap, float(height) * y_to_heatmap),
            ]
        )
        synthetic_labels.append(int(record["line_numeric_id"]))
        assigned_distances.append(float(distance))

    summary = {
        "heatmap_box_count": len(bounding_boxes),
        "assigned_box_count": len(synthetic_labels),
        "heatmap_box_assignment_rate": (len(synthetic_labels) / len(bounding_boxes)) if bounding_boxes else None,
        "baseline_line_count": len(baseline_records),
        "max_assignment_distance": max(assigned_distances) if assigned_distances else None,
        "mean_assignment_distance": float(np.mean(assigned_distances)) if assigned_distances else None,
    }
    return np.asarray(synthetic_nodes, dtype=float), np.asarray(synthetic_labels, dtype=int), summary


def build_legacy_axis_bound_polygons(
    xml_path: Path,
    image_path: Path,
    heatmap_path: Path,
    output_root: Path,
    segmentation_args: dict | None = None,
):
    config = _normalise_config(segmentation_args or {})
    nodes, labels, summary = _build_baseline_component_nodes(
        xml_path,
        image_path,
        heatmap_path,
        config["BINARIZE_THRESHOLD"],
        include_empty_text_lines=config["include_empty_text_lines"],
    )

    work_root = output_root / "_legacy_axis_bound_v1_geometry"
    if work_root.exists():
        shutil.rmtree(work_root)
    images_dir = work_root / "images_resized"
    heatmaps_dir = work_root / "heatmaps"
    gnn_format_dir = work_root / "layout_analysis_output" / "gnn-format"
    images_dir.mkdir(parents=True, exist_ok=True)
    heatmaps_dir.mkdir(parents=True, exist_ok=True)
    gnn_format_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(image_path, images_dir / f"{xml_path.stem}.jpg")
    shutil.copy(heatmap_path, heatmaps_dir / f"{xml_path.stem}.jpg")
    np.savetxt(gnn_format_dir / f"{xml_path.stem}_inputs_unnormalized.txt", nodes, fmt="%.6f")
    np.savetxt(gnn_format_dir / f"{xml_path.stem}_labels_textline.txt", labels, fmt="%d")

    polygons_by_label = _segment_module().segmentLinesFromPointClusters(
        str(work_root),
        xml_path.stem,
        BINARIZE_THRESHOLD=config["BINARIZE_THRESHOLD"],
        BBOX_PAD_V=config["BBOX_PAD_V"],
        BBOX_PAD_H=config["BBOX_PAD_H"],
        CC_SIZE_THRESHOLD_RATIO=config["CC_SIZE_THRESHOLD_RATIO"],
        GNN_PRED_PATH=str(work_root / "layout_analysis_output"),
    )
    return {
        int(label): [
            [int(point[0]), int(point[1])]
            for point in (value.get("points", value) if isinstance(value, dict) else value)
        ]
        for label, value in polygons_by_label.items()
    }, summary


class LegacyAxisBoundStrategy:
    name = "legacy_axis_bound_v1"

    def apply(self, request: TextLineSegmentationRequest) -> TextLineSegmentationResult:
        config = _normalise_config(request.strategy_config)
        source_xml_path = Path(request.source_pagexml_path)
        output_xml_path = Path(request.output_pagexml_path)
        metadata_path = Path(request.metadata_path) if request.metadata_path is not None else None
        output_xml_path.parent.mkdir(parents=True, exist_ok=True)

        polygons_by_line_numeric_id, generation_summary = build_legacy_axis_bound_polygons(
            source_xml_path,
            Path(request.page_image_path),
            Path(request.heatmap_path),
            output_xml_path.parent,
            segmentation_args=config,
        )

        tree = ET.parse(source_xml_path)
        root = tree.getroot()
        ET.register_namespace("", PAGE_XML_NAMESPACE)
        remove_textline_coords(root)
        line_metadata = set_textline_coords_by_numeric_id(root, polygons_by_line_numeric_id)
        prepared_line_count = sum(1 for item in line_metadata if item["coords_points"])
        source_line_count = len(
            load_baseline_records(
                source_xml_path,
                include_empty_text_lines=config["include_empty_text_lines"],
            )
        )
        geometry_summary = {
            "geometry_source": "baseline_heatmap",
            "line_segmentation_strategy_name": self.name,
            "prepared_line_count": prepared_line_count,
            "source_text_line_count": source_line_count,
            "source_line_coverage": (prepared_line_count / source_line_count) if source_line_count else None,
        }
        geometry_summary.update(generation_summary or {})

        if hasattr(ET, "indent"):
            ET.indent(tree, space="\t", level=0)
        tree.write(output_xml_path, encoding="utf-8", xml_declaration=True)

        result = TextLineSegmentationResult(
            strategy_name=self.name,
            source_pagexml_path=str(source_xml_path.resolve()),
            output_pagexml_path=str(output_xml_path.resolve()),
            page_image_path=str(Path(request.page_image_path).resolve()),
            heatmap_path=str(Path(request.heatmap_path).resolve()),
            metadata_path=str(metadata_path.resolve()) if metadata_path is not None else None,
            line_count=source_line_count,
            prepared_line_count=prepared_line_count,
            line_metadata=line_metadata,
            geometry_summary=geometry_summary,
        )
        if metadata_path is not None:
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            metadata_path.write_text(json.dumps(asdict(result), indent=2, ensure_ascii=False), encoding="utf-8")
        return result
