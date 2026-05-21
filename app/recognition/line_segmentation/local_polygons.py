from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from .geometry import BaselineTopology, distance, nearest_point_on_polyline, normalize_baseline_topology, polyline_length
from .legacy_axis_bound import _segment_module
from .pagexml import (
    PAGE_XML_NAMESPACE,
    load_baseline_records,
    remove_textline_coords,
    set_textline_coords_by_numeric_id,
)
from .types import TextLineSegmentationRequest, TextLineSegmentationResult


LOCAL_POLYGON_CROP_MODEL = "local_polygon_unwrap"
COMPONENT_PROJECTION_MODEL = "heatmap_component_contour_mask"
COMPONENT_PROJECTION_FALLBACK_MODEL = "heatmap_component_rectangle_bounds"

DEFAULT_LOCAL_POLYGON_CONFIG = {
    "BINARIZE_THRESHOLD": 0.45,
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
    "minimum_half_width_px": 12.0,
    "maximum_half_width_px": 180.0,
    "normal_pad_px": 6.0,
    "minimum_along_pad_px": 2.0,
    "final_mask_normal_pad_px": 0.0,
    "closed_circular_final_mask_normal_pad_px": 10.0,
    "final_mask_station_pad_px": 1.0,
    "bridge_gap_px": 80.0,
    "bridge_all_component_groups": True,
    "simplify_epsilon_px": 1.5,
    "max_polygon_points": 240,
    "minimum_page_mapping_step_px": 8.0,
    "local_canvas_margin_px": 4.0,
    "reading_order": "left_to_right",
    "circular_direction": "clockwise",
}


LOCAL_CLEANUP_MODEL = "legacy_remap_top_bottom_cc"


def _normalise_config(config: dict | None) -> dict:
    merged = dict(DEFAULT_LOCAL_POLYGON_CONFIG)
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
        "minimum_along_pad_px",
        "final_mask_normal_pad_px",
        "closed_circular_final_mask_normal_pad_px",
        "final_mask_station_pad_px",
        "bridge_gap_px",
        "simplify_epsilon_px",
        "minimum_page_mapping_step_px",
        "local_canvas_margin_px",
    ):
        merged[key] = float(merged[key])
    merged["min_mirror_pairs"] = int(merged["min_mirror_pairs"])
    merged["max_polygon_points"] = int(merged["max_polygon_points"])
    merged["include_empty_text_lines"] = bool(
        merged.get("include_empty_text_lines", merged.get("INCLUDE_EMPTY_TEXT_LINES", False))
    )
    merged["bridge_all_component_groups"] = bool(merged["bridge_all_component_groups"])
    merged["reading_order"] = str(merged["reading_order"])
    merged["circular_direction"] = str(merged["circular_direction"])
    return merged


def _load_image_shape(image_path: Path) -> tuple[int, int]:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read page image: {image_path}")
    return image.shape[:2]


def _load_processing_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read page image: {image_path}")
    return image


def _heatmap_boxes(image_path: Path, heatmap_path: Path, threshold: float) -> tuple[list[dict], dict]:
    module = _segment_module()
    image = module.loadImage(str(image_path))
    heatmap = module.loadImage(str(heatmap_path))
    if heatmap.ndim == 3:
        heatmap = heatmap[:, :, 0]
    image_height, image_width = image.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (image_width, image_height), interpolation=cv2.INTER_LINEAR)
    boxes = []
    threshold_val = int(float(threshold) * 255)
    _, binary_heatmap = cv2.threshold(np.uint8(heatmap_resized), threshold_val, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_heatmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x_val, y_val, width, height = cv2.boundingRect(contour)
        width = float(width)
        height = float(height)
        contour_points = contour.reshape(-1, 2).astype(float).tolist()
        boxes.append(
            {
                "x": float(x_val),
                "y": float(y_val),
                "width": width,
                "height": height,
                "center": (float(x_val) + width / 2.0, float(y_val) + height / 2.0),
                "max_side": max(width, height),
                "contour_points": contour_points,
                "contour_point_count": len(contour_points),
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


def _clip_point(point: tuple[float, float], image_width: int, image_height: int) -> list[int]:
    x_val = int(round(min(max(point[0], 0.0), float(image_width - 1))))
    y_val = int(round(min(max(point[1], 0.0), float(image_height - 1))))
    return [x_val, y_val]


def _unit(vector_x: float, vector_y: float) -> tuple[float, float]:
    length = math.hypot(vector_x, vector_y)
    if length <= 1e-6:
        return 1.0, 0.0
    return vector_x / length, vector_y / length


def _point_at_station(points: list[list[float]], station: float, is_closed: bool) -> tuple[tuple[float, float], tuple[float, float]]:
    total_length = polyline_length(points)
    if not points:
        return (0.0, 0.0), (1.0, 0.0)
    if len(points) == 1 or total_length <= 1e-6:
        return (points[0][0] + station, points[0][1]), (1.0, 0.0)
    if is_closed:
        station = station % total_length

    first_tangent = _unit(points[1][0] - points[0][0], points[1][1] - points[0][1])
    if station <= 0.0:
        return (points[0][0] + first_tangent[0] * station, points[0][1] + first_tangent[1] * station), first_tangent

    arc_before = 0.0
    last_tangent = first_tangent
    for index in range(len(points) - 1):
        start = points[index]
        end = points[index + 1]
        segment_length = distance(start, end)
        if segment_length <= 1e-6:
            continue
        tangent = _unit(end[0] - start[0], end[1] - start[1])
        last_tangent = tangent
        if station <= arc_before + segment_length:
            ratio = (station - arc_before) / segment_length
            return (
                start[0] + ratio * (end[0] - start[0]),
                start[1] + ratio * (end[1] - start[1]),
            ), tangent
        arc_before += segment_length

    overflow = station - total_length
    end = points[-1]
    return (end[0] + last_tangent[0] * overflow, end[1] + last_tangent[1] * overflow), last_tangent


def _project_point_to_local(
    point: tuple[float, float],
    topology: BaselineTopology,
    *,
    reference_station: float | None = None,
) -> tuple[float, float]:
    if len(topology.normalized_points) == 1 or topology.baseline_length <= 1e-6:
        origin = topology.normalized_points[0] if topology.normalized_points else [0.0, 0.0]
        return float(point[0]) - float(origin[0]), float(point[1]) - float(origin[1])

    nearest = nearest_point_on_polyline(point, topology.normalized_points)
    station = float(nearest.arc_length)
    tangent = nearest.tangent
    if not topology.is_closed and len(topology.normalized_points) >= 2:
        points = topology.normalized_points
        if nearest.segment_index == 0 and nearest.ratio <= 1e-6:
            start = points[0]
            start_tangent = _unit(points[1][0] - start[0], points[1][1] - start[1])
            start_projection = (float(point[0]) - start[0]) * start_tangent[0] + (float(point[1]) - start[1]) * start_tangent[1]
            if start_projection < 0.0:
                station = float(start_projection)
                tangent = start_tangent
                normal = (-float(tangent[1]), float(tangent[0]))
                offset_x = float(point[0]) - start[0]
                offset_y = float(point[1]) - start[1]
                return station, offset_x * normal[0] + offset_y * normal[1]
        last_segment_index = len(points) - 2
        if nearest.segment_index == last_segment_index and nearest.ratio >= 1.0 - 1e-6:
            end = points[-1]
            prev = points[-2]
            end_tangent = _unit(end[0] - prev[0], end[1] - prev[1])
            end_projection = (float(point[0]) - end[0]) * end_tangent[0] + (float(point[1]) - end[1]) * end_tangent[1]
            if end_projection > 0.0:
                station = float(topology.baseline_length + end_projection)
                tangent = end_tangent
                normal = (-float(tangent[1]), float(tangent[0]))
                offset_x = float(point[0]) - end[0]
                offset_y = float(point[1]) - end[1]
                return station, offset_x * normal[0] + offset_y * normal[1]
    baseline_length = max(float(topology.baseline_length), 1e-6)
    if topology.is_closed and reference_station is not None:
        while station - reference_station > baseline_length / 2.0:
            station -= baseline_length
        while station - reference_station < -baseline_length / 2.0:
            station += baseline_length
    normal = (-float(tangent[1]), float(tangent[0]))
    offset_x = float(point[0]) - float(nearest.point[0])
    offset_y = float(point[1]) - float(nearest.point[1])
    normal_offset = offset_x * normal[0] + offset_y * normal[1]
    return station, normal_offset


def _component_local_rect(box: dict, topology: BaselineTopology, config: dict) -> dict:
    center_station, center_normal = _project_point_to_local(box["center"], topology)
    rectangle_corners = [
        (box["x"], box["y"]),
        (box["x"] + box["width"], box["y"]),
        (box["x"] + box["width"], box["y"] + box["height"]),
        (box["x"], box["y"] + box["height"]),
    ]
    contour_points = [
        (float(point[0]), float(point[1]))
        for point in box.get("contour_points", [])
        if len(point) >= 2
    ]
    projection_points = contour_points if len(contour_points) >= 3 else rectangle_corners
    projection_model = (
        COMPONENT_PROJECTION_MODEL
        if len(contour_points) >= 3
        else COMPONENT_PROJECTION_FALLBACK_MODEL
    )
    projected = [
        _project_point_to_local(corner, topology, reference_station=center_station)
        for corner in projection_points
    ]
    station_values = [item[0] for item in projected]
    normal_values = [item[1] for item in projected]
    station_half_extent = max(0.5, (max(station_values) - min(station_values)) / 2.0)
    normal_half_extent = max(0.5, (max(normal_values) - min(normal_values)) / 2.0)
    station_pad = max(float(config["minimum_along_pad_px"]), station_half_extent * float(config["BBOX_PAD_H"]))
    normal_pad = normal_half_extent * float(config["BBOX_PAD_V"]) + float(config["normal_pad_px"])
    local_outline_points = [[float(s_val), float(n_val)] for s_val, n_val in projected] if projection_model == COMPONENT_PROJECTION_MODEL else []
    return {
        **box,
        "center_station": center_station,
        "center_normal": center_normal,
        "s_min": min(station_values) - station_pad,
        "s_max": max(station_values) + station_pad,
        "n_min": min(normal_values) - normal_pad,
        "n_max": max(normal_values) + normal_pad,
        "station_half_extent": station_half_extent,
        "normal_half_extent": normal_half_extent,
        "station_pad": station_pad,
        "normal_pad": normal_pad,
        "local_outline_points": local_outline_points,
        "local_projection_point_count": len(projection_points),
        "component_projection_model": projection_model,
    }


def _split_closed_rect(rect: dict, baseline_length: float) -> list[dict]:
    if baseline_length <= 1e-6:
        return [rect]
    width = rect["s_max"] - rect["s_min"]
    if width >= baseline_length:
        return [{**rect, "s_min": 0.0, "s_max": baseline_length, "local_outline_points": []}]
    center = (rect["s_min"] + rect["s_max"]) / 2.0
    shift = math.floor(center / baseline_length) * baseline_length
    s_min = rect["s_min"] - shift
    s_max = rect["s_max"] - shift
    outline_points = [
        [float(point[0]) - shift, float(point[1])]
        for point in rect.get("local_outline_points", [])
    ]
    while s_min < 0.0:
        s_min += baseline_length
        s_max += baseline_length
        outline_points = [[point[0] + baseline_length, point[1]] for point in outline_points]
    while s_min >= baseline_length:
        s_min -= baseline_length
        s_max -= baseline_length
        outline_points = [[point[0] - baseline_length, point[1]] for point in outline_points]
    if s_max <= baseline_length:
        return [{**rect, "s_min": s_min, "s_max": s_max, "local_outline_points": outline_points}]
    return [
        {**rect, "s_min": s_min, "s_max": baseline_length, "local_outline_points": []},
        {**rect, "s_min": 0.0, "s_max": s_max - baseline_length, "local_outline_points": []},
    ]


def _normalised_rects_for_topology(rects: list[dict], topology: BaselineTopology) -> list[dict]:
    if not topology.is_closed:
        return rects
    split_rects = []
    for rect in rects:
        split_rects.extend(_split_closed_rect(rect, topology.baseline_length))
    return split_rects


def _estimate_half_width(rects: list[dict], config: dict) -> float:
    if not rects:
        return float(config["minimum_half_width_px"])
    values = [max(abs(rect["n_min"]), abs(rect["n_max"])) for rect in rects]
    robust_width = float(np.percentile(np.asarray(values, dtype=float), 90))
    return float(np.clip(robust_width, config["minimum_half_width_px"], config["maximum_half_width_px"]))


def _draw_rect(mask: np.ndarray, rect: dict, origin_s: float, origin_n: float) -> None:
    x0 = int(math.floor(rect["s_min"] - origin_s))
    x1 = int(math.ceil(rect["s_max"] - origin_s))
    y0 = int(math.floor(rect["n_min"] - origin_n))
    y1 = int(math.ceil(rect["n_max"] - origin_n))
    x0 = max(0, min(mask.shape[1] - 1, x0))
    x1 = max(0, min(mask.shape[1] - 1, x1))
    y0 = max(0, min(mask.shape[0] - 1, y0))
    y1 = max(0, min(mask.shape[0] - 1, y1))
    if x1 >= x0 and y1 >= y0:
        cv2.rectangle(mask, (x0, y0), (x1, y1), 255, thickness=-1)


def _draw_component_mask(mask: np.ndarray, rect: dict, origin_s: float, origin_n: float) -> None:
    outline_points = rect.get("local_outline_points") or []
    if len(outline_points) < 3:
        _draw_rect(mask, rect, origin_s, origin_n)
        return

    x0 = int(math.floor(rect["s_min"] - origin_s))
    x1 = int(math.ceil(rect["s_max"] - origin_s)) + 1
    y0 = int(math.floor(rect["n_min"] - origin_n))
    y1 = int(math.ceil(rect["n_max"] - origin_n)) + 1
    x0 = max(0, min(mask.shape[1], x0))
    x1 = max(0, min(mask.shape[1], x1))
    y0 = max(0, min(mask.shape[0], y0))
    y1 = max(0, min(mask.shape[0], y1))
    if x1 <= x0 or y1 <= y0:
        return

    polygon = np.asarray(
        [
            [
                int(round(float(point[0]) - origin_s)) - x0,
                int(round(float(point[1]) - origin_n)) - y0,
            ]
            for point in outline_points
        ],
        dtype=np.int32,
    )
    component_mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    cv2.fillPoly(component_mask, [polygon], 255)
    station_pad = max(0, int(round(float(rect.get("station_pad", 0.0)))))
    normal_pad = max(0, int(round(float(rect.get("normal_pad", 0.0)))))
    if station_pad > 0 or normal_pad > 0:
        kernel_width = max(1, station_pad * 2 + 1)
        kernel_height = max(1, normal_pad * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, kernel_height))
        component_mask = cv2.dilate(component_mask, kernel, iterations=1)
    mask[y0:y1, x0:x1] = np.maximum(mask[y0:y1, x0:x1], component_mask)


def _draw_bridge(
    mask: np.ndarray,
    left_rect: dict,
    right_rect: dict,
    *,
    origin_s: float,
    origin_n: float,
    half_width: float,
) -> None:
    bridge = {
        "s_min": left_rect["s_max"],
        "s_max": right_rect["s_min"],
        "n_min": min(left_rect["center_normal"], right_rect["center_normal"]) - half_width,
        "n_max": max(left_rect["center_normal"], right_rect["center_normal"]) + half_width,
    }
    _draw_rect(mask, bridge, origin_s, origin_n)


def _final_mask_padding_px(config: dict, topology: BaselineTopology) -> tuple[float, float]:
    if topology.line_kind == "closed_circular":
        normal_pad_key = "closed_circular_final_mask_normal_pad_px"
    else:
        normal_pad_key = "final_mask_normal_pad_px"
    return float(config[normal_pad_key]), float(config["final_mask_station_pad_px"])


def _apply_final_mask_padding(
    mask: np.ndarray,
    *,
    origin_s: float,
    topology: BaselineTopology,
    baseline_length: float,
    config: dict,
) -> np.ndarray:
    normal_pad_px, station_pad_px = _final_mask_padding_px(config, topology)
    normal_pad = max(0, int(round(normal_pad_px)))
    station_pad = max(0, int(round(station_pad_px)))
    if normal_pad > 0 or station_pad > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (station_pad * 2 + 1, normal_pad * 2 + 1),
        )
        mask = cv2.dilate(mask, kernel, iterations=1)
    if topology.is_closed:
        left = max(0, int(math.floor(0.0 - origin_s)))
        right = min(mask.shape[1], int(math.ceil(baseline_length - origin_s)) + 1)
        if left > 0:
            mask[:, :left] = 0
        if right < mask.shape[1]:
            mask[:, right:] = 0
    return mask


def _mask_local_bounds(
    mask: np.ndarray,
    *,
    origin_s: float,
    origin_n: float,
    fallback_s_min: float,
    fallback_s_max: float,
    fallback_n_min: float,
    fallback_n_max: float,
    topology: BaselineTopology,
    baseline_length: float,
) -> tuple[float, float, float, float]:
    rows = np.where(mask.max(axis=1) > 0)[0]
    cols = np.where(mask.max(axis=0) > 0)[0]
    if rows.size == 0 or cols.size == 0:
        return fallback_s_min, fallback_s_max, fallback_n_min, fallback_n_max
    if topology.is_closed:
        local_s_min = 0.0
        local_s_max = baseline_length
    else:
        local_s_min = float(origin_s + int(cols[0]))
        local_s_max = float(origin_s + int(cols[-1]) + 1)
    local_n_min = float(origin_n + int(rows[0]))
    local_n_max = float(origin_n + int(rows[-1]) + 1)
    return local_s_min, local_s_max, local_n_min, local_n_max


def _remap_local_crop(
    processing_image: np.ndarray,
    topology: BaselineTopology,
    rect: dict,
    page_median_color: int,
) -> np.ndarray:
    width = max(2, int(math.ceil(rect["s_max"] - rect["s_min"])))
    height = max(2, int(math.ceil(rect["n_max"] - rect["n_min"])))
    normal_offsets = (float(rect["n_min"]) + np.arange(height, dtype=np.float32)).astype(np.float32)
    map_x = np.zeros((height, width), dtype=np.float32)
    map_y = np.zeros((height, width), dtype=np.float32)
    for column_index in range(width):
        station = float(rect["s_min"]) + float(column_index)
        center, tangent = _point_at_station(topology.normalized_points, station, topology.is_closed)
        normal = (-float(tangent[1]), float(tangent[0]))
        map_x[:, column_index] = float(center[0]) + normal[0] * normal_offsets
        map_y[:, column_index] = float(center[1]) + normal[1] * normal_offsets
    return cv2.remap(
        processing_image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=page_median_color,
    )


def _clean_local_remap_crop(local_crop: np.ndarray, config: dict) -> tuple[tuple[int, int, int, int] | None, dict]:
    if local_crop.size == 0:
        return None, {
            "removed_boundary_component_count": 0,
            "top_trim_px": 0,
            "bottom_trim_px": 0,
            "foreground_component_count": 0,
        }

    _, binary_foreground = cv2.threshold(local_crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_foreground, connectivity=8)
    crop_h, crop_w = local_crop.shape[:2]
    top = 0
    bottom = crop_h
    left = 0
    right = crop_w
    removed_boundary_component_count = 0

    if num_labels > 1:
        for label_index in range(1, num_labels):
            x_val, y_val, width, height, _ = stats[label_index]
            is_touching_boundary = y_val == 0 or y_val + height == crop_h
            is_size_constrained = height <= float(config["CC_SIZE_THRESHOLD_RATIO"]) * crop_h
            if not (is_touching_boundary and is_size_constrained):
                continue
            removed_boundary_component_count += 1
            if y_val == 0:
                top = max(top, int(y_val + height))
            if y_val + height == crop_h:
                bottom = min(bottom, int(y_val))

    if top >= bottom or left >= right:
        return None, {
            "removed_boundary_component_count": removed_boundary_component_count,
            "top_trim_px": top,
            "bottom_trim_px": crop_h - bottom,
            "foreground_component_count": max(0, num_labels - 1),
        }
    return (top, bottom, left, right), {
        "removed_boundary_component_count": removed_boundary_component_count,
        "top_trim_px": top,
        "bottom_trim_px": crop_h - bottom,
        "foreground_component_count": max(0, num_labels - 1),
    }


def _clean_component_rects_in_local_space(
    processing_image: np.ndarray,
    topology: BaselineTopology,
    rects: list[dict],
    config: dict,
) -> tuple[list[dict], dict]:
    page_median_color = int(np.median(processing_image))
    cleaned_rects: list[dict] = []
    removed_boundary_component_count = 0
    empty_component_count = 0
    top_trim_px_total = 0
    bottom_trim_px_total = 0
    foreground_component_count = 0

    for rect in rects:
        local_crop = _remap_local_crop(processing_image, topology, rect, page_median_color)
        crop_coords, cleanup_summary = _clean_local_remap_crop(local_crop, config)
        removed_boundary_component_count += int(cleanup_summary["removed_boundary_component_count"])
        top_trim_px_total += int(cleanup_summary["top_trim_px"])
        bottom_trim_px_total += int(cleanup_summary["bottom_trim_px"])
        foreground_component_count += int(cleanup_summary["foreground_component_count"])
        if crop_coords is None:
            empty_component_count += 1
            continue
        top, bottom, left, right = crop_coords
        cleaned_rect = {
            **rect,
            "s_min": float(rect["s_min"]) + float(left),
            "s_max": float(rect["s_min"]) + float(right),
            "n_min": float(rect["n_min"]) + float(top),
            "n_max": float(rect["n_min"]) + float(bottom),
        }
        cleaned_rect["center_station"] = (cleaned_rect["s_min"] + cleaned_rect["s_max"]) / 2.0
        cleaned_rect["center_normal"] = (cleaned_rect["n_min"] + cleaned_rect["n_max"]) / 2.0
        cleaned_rects.append(cleaned_rect)

    return cleaned_rects, {
        "local_cleanup_model": LOCAL_CLEANUP_MODEL,
        "local_cleanup_input_component_count": len(rects),
        "local_cleanup_output_component_count": len(cleaned_rects),
        "local_cleanup_empty_component_count": empty_component_count,
        "local_cleanup_removed_boundary_component_count": removed_boundary_component_count,
        "local_cleanup_foreground_component_count": foreground_component_count,
        "local_cleanup_top_trim_px_total": top_trim_px_total,
        "local_cleanup_bottom_trim_px_total": bottom_trim_px_total,
    }


def _contour_to_local_polygon(mask: np.ndarray, config: dict) -> np.ndarray | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    epsilon = float(config["simplify_epsilon_px"])
    max_points = max(8, int(config["max_polygon_points"]))
    simplified = cv2.approxPolyDP(contour, epsilon, True)
    while len(simplified) > max_points and epsilon < 64.0:
        epsilon *= 1.5
        simplified = cv2.approxPolyDP(contour, epsilon, True)
    return simplified.reshape(-1, 2)


def _dedupe_adjacent(points: list[list[int]]) -> list[list[int]]:
    deduped: list[list[int]] = []
    for point in points:
        if not deduped or point != deduped[-1]:
            deduped.append(point)
    if len(deduped) > 1 and deduped[0] == deduped[-1]:
        deduped.pop()
    return deduped


def _densify_local_polygon(points: np.ndarray, max_step_px: float) -> list[tuple[float, float]]:
    if len(points) == 0:
        return []
    densified: list[tuple[float, float]] = []
    for index, point in enumerate(points):
        next_point = points[(index + 1) % len(points)]
        x0, y0 = float(point[0]), float(point[1])
        x1, y1 = float(next_point[0]), float(next_point[1])
        if not densified:
            densified.append((x0, y0))
        segment_length = math.hypot(x1 - x0, y1 - y0)
        step_count = max(1, int(math.ceil(segment_length / max(max_step_px, 1.0))))
        for step_index in range(1, step_count):
            ratio = step_index / step_count
            densified.append((x0 + (x1 - x0) * ratio, y0 + (y1 - y0) * ratio))
        if index < len(points) - 1:
            densified.append((x1, y1))
    return densified


def _bounds_fallback_polygon(
    topology: BaselineTopology,
    half_width: float,
    image_width: int,
    image_height: int,
) -> list[list[int]]:
    xs = [point[0] for point in topology.normalized_points] or [0.0]
    ys = [point[1] for point in topology.normalized_points] or [0.0]
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


def _fallback_band_polygon(
    topology: BaselineTopology,
    half_width: float,
    image_width: int,
    image_height: int,
) -> list[list[int]]:
    length = max(topology.baseline_length, 1.0)
    if topology.is_closed:
        return _bounds_fallback_polygon(topology, half_width, image_width, image_height)
    local_points = [(0.0, -half_width), (length, -half_width), (length, half_width), (0.0, half_width)]
    polygon = _dedupe_adjacent(
        [
            _local_to_page_point(topology, s_val, n_val, image_width, image_height)
            for s_val, n_val in local_points
        ]
    )
    return polygon if len(polygon) >= 4 else _bounds_fallback_polygon(topology, half_width, image_width, image_height)


def _local_to_page_point(
    topology: BaselineTopology,
    station: float,
    normal_offset: float,
    image_width: int,
    image_height: int,
) -> list[int]:
    center, tangent = _point_at_station(topology.normalized_points, station, topology.is_closed)
    normal = (-tangent[1], tangent[0])
    return _clip_point(
        (center[0] + normal[0] * normal_offset, center[1] + normal[1] * normal_offset),
        image_width,
        image_height,
    )


def _build_local_polygon(
    topology: BaselineTopology,
    rects: list[dict],
    config: dict,
    processing_image: np.ndarray,
    image_width: int,
    image_height: int,
) -> tuple[list[list[int]], dict]:
    baseline_length = max(float(topology.baseline_length), 1.0)
    rects = _normalised_rects_for_topology(rects, topology)
    cleanup_summary = {
        "local_cleanup_model": LOCAL_CLEANUP_MODEL,
        "local_cleanup_input_component_count": len(rects),
        "local_cleanup_output_component_count": 0,
        "local_cleanup_empty_component_count": 0,
        "local_cleanup_removed_boundary_component_count": 0,
        "local_cleanup_foreground_component_count": 0,
        "local_cleanup_top_trim_px_total": 0,
        "local_cleanup_bottom_trim_px_total": 0,
    }
    if rects:
        rects, cleanup_summary = _clean_component_rects_in_local_space(
            processing_image,
            topology,
            rects,
            config,
        )
    used_component_fallback = False
    fallback_used = False
    if not rects:
        half_width = float(config["minimum_half_width_px"])
        rects = [
            {
                "s_min": 0.0,
                "s_max": baseline_length,
                "n_min": -half_width,
                "n_max": half_width,
                "center_station": baseline_length / 2.0,
                "center_normal": 0.0,
            }
        ]
        fallback_used = True
        used_component_fallback = True

    half_width = _estimate_half_width(rects, config)
    final_normal_pad_px, final_station_pad_px = _final_mask_padding_px(config, topology)
    margin_s = float(config["local_canvas_margin_px"]) + max(0.0, final_station_pad_px)
    margin_n = float(config["local_canvas_margin_px"]) + max(0.0, final_normal_pad_px)
    if topology.is_closed:
        min_s = 0.0
        max_s = baseline_length
    else:
        min_s = min(rect["s_min"] for rect in rects)
        max_s = max(rect["s_max"] for rect in rects)
    min_n = min(rect["n_min"] for rect in rects)
    max_n = max(rect["n_max"] for rect in rects)
    origin_s = math.floor(min_s - margin_s)
    origin_n = math.floor(min_n - margin_n)
    width = max(2, int(math.ceil(max_s - origin_s + margin_s)))
    height = max(2, int(math.ceil(max_n - origin_n + margin_n)))
    mask = np.zeros((height, width), dtype=np.uint8)

    ordered_rects = sorted(rects, key=lambda item: ((item["s_min"] + item["s_max"]) / 2.0, item["center_normal"]))
    for rect in ordered_rects:
        _draw_component_mask(mask, rect, origin_s, origin_n)
    for left_rect, right_rect in zip(ordered_rects, ordered_rects[1:]):
        gap = right_rect["s_min"] - left_rect["s_max"]
        if gap <= 0.0:
            continue
        if config["bridge_all_component_groups"] or gap <= float(config["bridge_gap_px"]):
            _draw_bridge(mask, left_rect, right_rect, origin_s=origin_s, origin_n=origin_n, half_width=half_width)
    if topology.is_closed and len(ordered_rects) >= 2:
        first_rect = ordered_rects[0]
        last_rect = ordered_rects[-1]
        seam_gap = (first_rect["s_min"] + baseline_length) - last_rect["s_max"]
        if seam_gap > 0.0 and (config["bridge_all_component_groups"] or seam_gap <= float(config["bridge_gap_px"])):
            left_bridge = {
                **last_rect,
                "s_min": last_rect["s_max"],
                "s_max": baseline_length,
                "center_normal": (last_rect["center_normal"] + first_rect["center_normal"]) / 2.0,
            }
            right_bridge = {
                **first_rect,
                "s_min": 0.0,
                "s_max": first_rect["s_min"],
                "center_normal": (last_rect["center_normal"] + first_rect["center_normal"]) / 2.0,
            }
            _draw_rect(mask, left_bridge, origin_s, origin_n)
            _draw_rect(mask, right_bridge, origin_s, origin_n)

    mask = _apply_final_mask_padding(
        mask,
        origin_s=origin_s,
        topology=topology,
        baseline_length=baseline_length,
        config=config,
    )
    local_s_min, local_s_max, local_n_min, local_n_max = _mask_local_bounds(
        mask,
        origin_s=origin_s,
        origin_n=origin_n,
        fallback_s_min=min_s,
        fallback_s_max=max_s,
        fallback_n_min=min_n,
        fallback_n_max=max_n,
        topology=topology,
        baseline_length=baseline_length,
    )

    local_polygon = _contour_to_local_polygon(mask, config)
    if local_polygon is None or len(local_polygon) < 3:
        polygon = _fallback_band_polygon(topology, half_width, image_width, image_height)
        fallback_used = True
        return polygon, {
            **cleanup_summary,
            "fallback_used": fallback_used,
            "fallback_reason": "empty_local_mask",
            "local_canvas_width_px": width,
            "local_canvas_height_px": height,
            "local_polygon_point_count": 0,
            "page_polygon_point_count": len(polygon),
            "line_half_width_px": half_width,
            "final_mask_normal_pad_px": float(final_normal_pad_px),
            "final_mask_station_pad_px": float(final_station_pad_px),
        }

    max_polygon_points = max(8, int(config["max_polygon_points"]))
    adaptive_step = max(
        float(config["minimum_page_mapping_step_px"]),
        baseline_length / max(max_polygon_points / 2.0, 1.0),
    )
    mapped_local_polygon = _densify_local_polygon(local_polygon, adaptive_step)
    page_points = []
    for x_val, y_val in mapped_local_polygon:
        page_points.append(
            _local_to_page_point(
                topology,
                float(origin_s + x_val),
                float(origin_n + y_val),
                image_width,
                image_height,
            )
        )
    page_points = _dedupe_adjacent(page_points)
    if len(page_points) < 4:
        page_points = _fallback_band_polygon(topology, half_width, image_width, image_height)
        fallback_used = True

    return page_points, {
        **cleanup_summary,
        "fallback_used": fallback_used,
        "fallback_reason": "no_assigned_components" if used_component_fallback else None,
        "local_canvas_width_px": width,
        "local_canvas_height_px": height,
        "local_polygon_point_count": int(len(local_polygon)),
        "mapped_local_polygon_point_count": int(len(mapped_local_polygon)),
        "page_polygon_point_count": int(len(page_points)),
        "line_half_width_px": half_width,
        "final_mask_normal_pad_px": float(final_normal_pad_px),
        "final_mask_station_pad_px": float(final_station_pad_px),
        "local_mask_foreground_pixel_count": int(np.count_nonzero(mask)),
        "local_mask_background_fraction": float(1.0 - (np.count_nonzero(mask) / mask.size)) if mask.size else None,
        "local_s_min": float(local_s_min),
        "local_s_max": float(local_s_max),
        "local_n_min": float(local_n_min),
        "local_n_max": float(local_n_max),
    }


def _assign_components_to_lines(
    boxes: list[dict],
    topologies: dict[int, BaselineTopology],
    config: dict,
) -> tuple[dict[int, list[dict]], dict]:
    assignments = {line_numeric_id: [] for line_numeric_id in topologies}
    rejected_counts = {"too_far_from_baseline": 0, "no_baseline": 0}
    assigned_distances = []
    for box in boxes:
        best_line_id = None
        best_distance = float("inf")
        for line_numeric_id, topology in topologies.items():
            nearest = nearest_point_on_polyline(box["center"], topology.normalized_points)
            if nearest.distance < best_distance:
                best_distance = float(nearest.distance)
                best_line_id = line_numeric_id
        if best_line_id is None:
            rejected_counts["no_baseline"] += 1
            continue
        threshold = max(float(config["component_max_distance_px"]), box["max_side"] * float(config["component_distance_scale"]))
        if best_distance > threshold:
            rejected_counts["too_far_from_baseline"] += 1
            continue
        component = _component_local_rect(box, topologies[best_line_id], config)
        component["baseline_distance"] = best_distance
        assignments[best_line_id].append(component)
        assigned_distances.append(best_distance)

    summary = {
        "assigned_box_count": len(assigned_distances),
        "rejected_box_count": int(sum(rejected_counts.values())),
        "rejected_box_counts_by_reason": rejected_counts,
        "heatmap_box_assignment_rate": (len(assigned_distances) / len(boxes)) if boxes else None,
        "max_assignment_distance": max(assigned_distances) if assigned_distances else None,
        "mean_assignment_distance": float(np.mean(assigned_distances)) if assigned_distances else None,
    }
    return assignments, summary


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


class LocalPolygonsStrategy:
    name = "local_polygons_v1"

    def apply(self, request: TextLineSegmentationRequest) -> TextLineSegmentationResult:
        config = _normalise_config(dict(request.strategy_config or {}))
        source_xml_path = Path(request.source_pagexml_path)
        output_xml_path = Path(request.output_pagexml_path)
        metadata_path = Path(request.metadata_path) if request.metadata_path is not None else None
        output_xml_path.parent.mkdir(parents=True, exist_ok=True)

        records = load_baseline_records(source_xml_path, include_empty_text_lines=config["include_empty_text_lines"])
        topologies = _topologies_for_records(records, config)
        source_line_count = len(records)
        processing_image = _load_processing_image(Path(request.page_image_path))
        image_height, image_width = processing_image.shape[:2]
        boxes, box_summary = _heatmap_boxes(
            Path(request.page_image_path),
            Path(request.heatmap_path),
            config["BINARIZE_THRESHOLD"],
        )
        assignments, assignment_summary = _assign_components_to_lines(boxes, topologies, config)

        polygons_by_line_numeric_id = {}
        line_details: dict[int, dict] = {}
        for line_numeric_id, topology in topologies.items():
            components = assignments.get(line_numeric_id, [])
            polygon, polygon_summary = _build_local_polygon(
                topology,
                components,
                config,
                processing_image,
                image_width,
                image_height,
            )
            polygons_by_line_numeric_id[line_numeric_id] = polygon
            line_details[line_numeric_id] = {
                "line_kind": topology.line_kind,
                "topology": topology.to_metadata(),
                "crop_model": LOCAL_POLYGON_CROP_MODEL,
                "component_projection_model": COMPONENT_PROJECTION_MODEL,
                "local_cleanup_model": LOCAL_CLEANUP_MODEL,
                "assigned_component_count": len(components),
                "rejected_component_counts_by_reason": assignment_summary["rejected_box_counts_by_reason"],
                **polygon_summary,
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
        normalization_counts = {}
        for topology in topologies.values():
            topology_counts[topology.line_kind] = topology_counts.get(topology.line_kind, 0) + 1
            for action in topology.normalization_actions:
                normalization_counts[action] = normalization_counts.get(action, 0) + 1
        geometry_summary = {
            "geometry_source": "baseline_heatmap",
            "line_segmentation_strategy_name": self.name,
            "prepared_line_count": prepared_line_count,
            "source_text_line_count": source_line_count,
            "source_line_coverage": (prepared_line_count / source_line_count) if source_line_count else None,
            "used_legacy_axis_bound_delegate": False,
            "crop_model_counts": {LOCAL_POLYGON_CROP_MODEL: prepared_line_count},
            "component_projection_model": COMPONENT_PROJECTION_MODEL,
            "local_cleanup_model": LOCAL_CLEANUP_MODEL,
            "topology_counts": topology_counts,
            "normalization_action_counts": normalization_counts,
            "orientation_policy": {
                "reading_order": config["reading_order"],
                "circular_direction": config["circular_direction"],
            },
        }
        geometry_summary.update(box_summary)
        geometry_summary.update(assignment_summary)

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
