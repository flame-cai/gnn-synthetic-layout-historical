from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from .geometry import BaselineTopology, nearest_point_on_polyline, normalize_baseline_topology
from .legacy_axis_bound import build_legacy_axis_bound_polygons, _segment_module
from .pagexml import (
    PAGE_XML_NAMESPACE,
    load_baseline_records,
    remove_textline_coords,
    set_textline_coords_by_numeric_id,
)
from .types import TextLineSegmentationRequest, TextLineSegmentationResult


DEFAULT_LOCAL_TANGENT_BAND_CONFIG = {
    "BINARIZE_THRESHOLD": 0.5098,
    "BBOX_PAD_V": 0.7,
    "BBOX_PAD_H": 0.5,
    "CC_SIZE_THRESHOLD_RATIO": 0.4,
    "include_empty_text_lines": False,
    "mirror_match_tolerance_px": 4.0,
    "closed_path_tolerance_px": 12.0,
    "min_mirror_pairs": 3,
    "straightness_chord_ratio": 0.985,
    "horizontal_angle_degrees": 12.0,
    "component_max_distance_px": 90.0,
    "component_distance_scale": 4.0,
    "minimum_half_width_px": 18.0,
    "maximum_half_width_px": 180.0,
    "normal_pad_px": 8.0,
    "normal_pad_scale": 1.15,
    "along_pad_scale": 0.5,
    "preserve_horizontal_with_legacy": True,
    "reading_order": "left_to_right",
    "circular_direction": "clockwise",
}


def _normalise_config(config: dict | None) -> dict:
    merged = dict(DEFAULT_LOCAL_TANGENT_BAND_CONFIG)
    merged.update(dict(config or {}))
    for key in (
        "BINARIZE_THRESHOLD",
        "BBOX_PAD_V",
        "BBOX_PAD_H",
        "CC_SIZE_THRESHOLD_RATIO",
        "mirror_match_tolerance_px",
        "closed_path_tolerance_px",
        "straightness_chord_ratio",
        "horizontal_angle_degrees",
        "component_max_distance_px",
        "component_distance_scale",
        "minimum_half_width_px",
        "maximum_half_width_px",
        "normal_pad_px",
        "normal_pad_scale",
        "along_pad_scale",
    ):
        merged[key] = float(merged[key])
    merged["min_mirror_pairs"] = int(merged["min_mirror_pairs"])
    merged["include_empty_text_lines"] = bool(
        merged.get("include_empty_text_lines", merged.get("INCLUDE_EMPTY_TEXT_LINES", False))
    )
    merged["preserve_horizontal_with_legacy"] = bool(merged["preserve_horizontal_with_legacy"])
    merged["reading_order"] = str(merged["reading_order"])
    merged["circular_direction"] = str(merged["circular_direction"])
    return merged


def _load_image_shape(image_path: Path) -> tuple[int, int]:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read page image: {image_path}")
    return image.shape[:2]


def _heatmap_boxes(image_path: Path, heatmap_path: Path, threshold: float) -> tuple[list[dict], dict]:
    module = _segment_module()
    image = module.loadImage(str(image_path))
    heatmap = module.loadImage(str(heatmap_path))
    if heatmap.ndim == 3:
        heatmap = heatmap[:, :, 0]
    image_height, image_width = image.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (image_width, image_height), interpolation=cv2.INTER_LINEAR)
    boxes = []
    for x_val, y_val, width, height in module.gen_bounding_boxes(heatmap_resized, threshold):
        boxes.append(
            {
                "x": float(x_val),
                "y": float(y_val),
                "width": float(width),
                "height": float(height),
                "center": (float(x_val) + float(width) / 2.0, float(y_val) + float(height) / 2.0),
                "max_side": max(float(width), float(height)),
            }
        )
    return boxes, {"heatmap_box_count": len(boxes)}


def _topologies_for_records(records: list[dict], config: dict) -> dict[int, BaselineTopology]:
    return {
        int(record["line_numeric_id"]): normalize_baseline_topology(
            record["baseline_points"],
            mirror_match_tolerance=config["mirror_match_tolerance_px"],
            closed_path_tolerance=config["closed_path_tolerance_px"],
            min_mirror_pairs=config["min_mirror_pairs"],
            straightness_chord_ratio=config["straightness_chord_ratio"],
            horizontal_angle_degrees=config["horizontal_angle_degrees"],
            reading_order=config["reading_order"],
            circular_direction=config["circular_direction"],
        )
        for record in records
    }


def _all_simple_horizontal(topologies: dict[int, BaselineTopology]) -> bool:
    return bool(topologies) and all(
        topology.line_kind == "horizontal_straight"
        for topology in topologies.values()
    )


def _should_preserve_legacy_line(topology: BaselineTopology) -> bool:
    return topology.line_kind in {"horizontal_straight", "point"}


def _assign_components_to_lines(boxes: list[dict], topologies: dict[int, BaselineTopology], config: dict) -> tuple[dict[int, list[dict]], dict]:
    assignments = {line_numeric_id: [] for line_numeric_id in topologies}
    assigned_distances = []
    for box in boxes:
        best_line_id = None
        best_distance = float("inf")
        best_normal_half_extent = None
        for line_numeric_id, topology in topologies.items():
            nearest = nearest_point_on_polyline(box["center"], topology.normalized_points)
            if nearest.distance < best_distance:
                best_distance = nearest.distance
                best_line_id = line_numeric_id
                normal = (-nearest.tangent[1], nearest.tangent[0])
                best_normal_half_extent = 0.5 * (
                    abs(normal[0]) * box["width"] + abs(normal[1]) * box["height"]
                )
        if best_line_id is None:
            continue
        threshold = max(config["component_max_distance_px"], box["max_side"] * config["component_distance_scale"])
        if best_distance > threshold:
            continue
        assignments[best_line_id].append(
            {
                **box,
                "baseline_distance": float(best_distance),
                "normal_half_extent": float(best_normal_half_extent or 0.0),
            }
        )
        assigned_distances.append(float(best_distance))

    summary = {
        "assigned_box_count": len(assigned_distances),
        "heatmap_box_assignment_rate": (len(assigned_distances) / len(boxes)) if boxes else None,
        "max_assignment_distance": max(assigned_distances) if assigned_distances else None,
        "mean_assignment_distance": float(np.mean(assigned_distances)) if assigned_distances else None,
    }
    return assignments, summary


def _estimate_half_width(components: list[dict], config: dict) -> float:
    if not components:
        return float(config["minimum_half_width_px"])
    distances = np.asarray(
        [
            float(component["baseline_distance"]) + float(component.get("normal_half_extent", 0.5 * component["max_side"]))
            for component in components
        ],
        dtype=float,
    )
    robust_width = float(np.percentile(distances, 90))
    padded = robust_width * config["normal_pad_scale"] + config["normal_pad_px"]
    return float(np.clip(padded, config["minimum_half_width_px"], config["maximum_half_width_px"]))


def _unit(vector_x: float, vector_y: float) -> tuple[float, float]:
    length = math.hypot(vector_x, vector_y)
    if length <= 1e-6:
        return 1.0, 0.0
    return vector_x / length, vector_y / length


def _clip_point(point: tuple[float, float], image_width: int, image_height: int) -> list[int]:
    x_val = int(round(min(max(point[0], 0.0), float(image_width - 1))))
    y_val = int(round(min(max(point[1], 0.0), float(image_height - 1))))
    return [x_val, y_val]


def _dedupe_adjacent(points: list[list[int]]) -> list[list[int]]:
    deduped: list[list[int]] = []
    for point in points:
        if not deduped or point != deduped[-1]:
            deduped.append(point)
    if len(deduped) > 1 and deduped[0] == deduped[-1]:
        deduped.pop()
    return deduped


def _fallback_box(points: list[list[float]], half_width: float, image_width: int, image_height: int) -> list[list[int]]:
    xs = [point[0] for point in points] or [0.0]
    ys = [point[1] for point in points] or [0.0]
    min_x = max(0.0, min(xs) - half_width)
    max_x = min(float(image_width - 1), max(xs) + half_width)
    min_y = max(0.0, min(ys) - half_width)
    max_y = min(float(image_height - 1), max(ys) + half_width)
    return [
        [int(round(min_x)), int(round(min_y))],
        [int(round(max_x)), int(round(min_y))],
        [int(round(max_x)), int(round(max_y))],
        [int(round(min_x)), int(round(max_y))],
    ]


def _band_polygon(
    topology: BaselineTopology,
    half_width: float,
    along_pad: float,
    image_width: int,
    image_height: int,
) -> list[list[int]]:
    points = [list(point) for point in topology.normalized_points]
    if len(points) < 2:
        return _fallback_box(points, half_width, image_width, image_height)

    if not topology.is_closed:
        start_tangent = _unit(points[1][0] - points[0][0], points[1][1] - points[0][1])
        end_tangent = _unit(points[-1][0] - points[-2][0], points[-1][1] - points[-2][1])
        points[0] = [points[0][0] - start_tangent[0] * along_pad, points[0][1] - start_tangent[1] * along_pad]
        points[-1] = [points[-1][0] + end_tangent[0] * along_pad, points[-1][1] + end_tangent[1] * along_pad]

    left_points: list[list[int]] = []
    right_points: list[list[int]] = []
    for index, point in enumerate(points):
        if index == 0:
            tangent = _unit(points[1][0] - point[0], points[1][1] - point[1])
        elif index == len(points) - 1:
            tangent = _unit(point[0] - points[index - 1][0], point[1] - points[index - 1][1])
        else:
            tangent = _unit(points[index + 1][0] - points[index - 1][0], points[index + 1][1] - points[index - 1][1])
        normal = (-tangent[1], tangent[0])
        left_points.append(_clip_point((point[0] + normal[0] * half_width, point[1] + normal[1] * half_width), image_width, image_height))
        right_points.append(_clip_point((point[0] - normal[0] * half_width, point[1] - normal[1] * half_width), image_width, image_height))

    polygon = _dedupe_adjacent(left_points + list(reversed(right_points)))
    return polygon if len(polygon) >= 3 else _fallback_box(points, half_width, image_width, image_height)


def _write_result_xml(
    *,
    source_xml_path: Path,
    output_xml_path: Path,
    polygons_by_line_numeric_id: dict[int, list[list[int]]],
) -> tuple[ET.ElementTree, list[dict]]:
    tree = ET.parse(source_xml_path)
    root = tree.getroot()
    ET.register_namespace("", PAGE_XML_NAMESPACE)
    remove_textline_coords(root)
    line_metadata = set_textline_coords_by_numeric_id(root, polygons_by_line_numeric_id)
    if hasattr(ET, "indent"):
        ET.indent(tree, space="\t", level=0)
    output_xml_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_xml_path, encoding="utf-8", xml_declaration=True)
    return tree, line_metadata


class LocalTangentBandStrategy:
    name = "local_tangent_band_v1"

    def apply(self, request: TextLineSegmentationRequest) -> TextLineSegmentationResult:
        config = _normalise_config(dict(request.strategy_config or {}))
        source_xml_path = Path(request.source_pagexml_path)
        output_xml_path = Path(request.output_pagexml_path)
        metadata_path = Path(request.metadata_path) if request.metadata_path is not None else None
        output_xml_path.parent.mkdir(parents=True, exist_ok=True)

        records = load_baseline_records(source_xml_path, include_empty_text_lines=config["include_empty_text_lines"])
        topologies = _topologies_for_records(records, config)
        source_line_count = len(records)
        image_height, image_width = _load_image_shape(Path(request.page_image_path))

        used_legacy_delegate = False
        line_details: dict[int, dict] = {}
        if config["preserve_horizontal_with_legacy"] and _all_simple_horizontal(topologies):
            used_legacy_delegate = True
            polygons_by_line_numeric_id, generation_summary = build_legacy_axis_bound_polygons(
                source_xml_path,
                Path(request.page_image_path),
                Path(request.heatmap_path),
                output_xml_path.parent,
                segmentation_args=config,
            )
            for line_numeric_id, topology in topologies.items():
                line_details[line_numeric_id] = {
                    "line_kind": topology.line_kind,
                    "topology": topology.to_metadata(),
                    "crop_model": "legacy_axis_bound_delegate",
                    "assigned_component_count": None,
                    "band_half_width_px": None,
                    "along_pad_px": None,
                }
        else:
            boxes, box_summary = _heatmap_boxes(
                Path(request.page_image_path),
                Path(request.heatmap_path),
                config["BINARIZE_THRESHOLD"],
            )
            assignments, assignment_summary = _assign_components_to_lines(boxes, topologies, config)
            generation_summary = {**box_summary, **assignment_summary, "baseline_line_count": source_line_count}
            legacy_polygons_by_line_numeric_id = {}
            if config["preserve_horizontal_with_legacy"]:
                legacy_polygons_by_line_numeric_id, _ = build_legacy_axis_bound_polygons(
                    source_xml_path,
                    Path(request.page_image_path),
                    Path(request.heatmap_path),
                    output_xml_path.parent,
                    segmentation_args=config,
                )
            polygons_by_line_numeric_id = {}
            for line_numeric_id, topology in topologies.items():
                if (
                    config["preserve_horizontal_with_legacy"]
                    and _should_preserve_legacy_line(topology)
                    and line_numeric_id in legacy_polygons_by_line_numeric_id
                ):
                    polygons_by_line_numeric_id[line_numeric_id] = legacy_polygons_by_line_numeric_id[line_numeric_id]
                    line_details[line_numeric_id] = {
                        "line_kind": topology.line_kind,
                        "topology": topology.to_metadata(),
                        "crop_model": "legacy_axis_bound_delegate",
                        "assigned_component_count": len(assignments.get(line_numeric_id, [])),
                        "band_half_width_px": None,
                        "along_pad_px": None,
                    }
                    continue
                components = assignments.get(line_numeric_id, [])
                half_width = _estimate_half_width(components, config)
                along_pad = 0.0 if topology.is_closed else max(4.0, half_width * config["along_pad_scale"])
                polygon = _band_polygon(topology, half_width, along_pad, image_width, image_height)
                polygons_by_line_numeric_id[line_numeric_id] = polygon
                line_details[line_numeric_id] = {
                    "line_kind": topology.line_kind,
                    "topology": topology.to_metadata(),
                    "crop_model": "local_tangent_band",
                    "assigned_component_count": len(components),
                    "band_half_width_px": half_width,
                    "along_pad_px": along_pad,
                    "component_distance_summary": {
                        "max": max((component["baseline_distance"] for component in components), default=None),
                        "mean": float(np.mean([component["baseline_distance"] for component in components])) if components else None,
                    },
                }

        tree, line_metadata = _write_result_xml(
            source_xml_path=source_xml_path,
            output_xml_path=output_xml_path,
            polygons_by_line_numeric_id=polygons_by_line_numeric_id,
        )
        for item in line_metadata:
            item.update(line_details.get(int(item["line_numeric_id"]), {}))
        prepared_line_count = sum(1 for item in line_metadata if item["coords_points"])
        topology_counts = {}
        for topology in topologies.values():
            topology_counts[topology.line_kind] = topology_counts.get(topology.line_kind, 0) + 1
        geometry_summary = {
            "geometry_source": "baseline_heatmap",
            "line_segmentation_strategy_name": self.name,
            "prepared_line_count": prepared_line_count,
            "source_text_line_count": source_line_count,
            "source_line_coverage": (prepared_line_count / source_line_count) if source_line_count else None,
            "used_legacy_axis_bound_delegate": used_legacy_delegate,
            "topology_counts": topology_counts,
            "orientation_policy": {
                "reading_order": config["reading_order"],
                "circular_direction": config["circular_direction"],
            },
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
