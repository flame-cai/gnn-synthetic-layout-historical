from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from .geometry import (
    PageXmlLine,
    line_label_points,
    normalize_open_curve,
    parse_gnn_format_page,
    parse_pagexml_lines,
)
from .pagexml_rewrite import rewrite_textline_coords


@dataclass(frozen=True)
class SegmentationResult:
    page_id: str
    strategy_name: str
    output_xml_path: Path
    coords_by_line_custom: dict[str, list[tuple[float, float]]]
    metadata: dict


def _numeric_suffix(value: str) -> int | None:
    digits = "".join(ch if ch.isdigit() else " " for ch in value).split()
    return int(digits[-1]) if digits else None


def _point_distance_to_curve(point: tuple[float, float], curve: list[tuple[float, float]]) -> float:
    if not curve:
        return float("inf")
    return min(math.hypot(point[0] - x_val, point[1] - y_val) for x_val, y_val in curve)


def _tangent_at(points: list[tuple[float, float]], index: int) -> tuple[float, float]:
    if len(points) < 2:
        return (1.0, 0.0)
    if index == 0:
        prev_point, next_point = points[0], points[1]
    elif index == len(points) - 1:
        prev_point, next_point = points[-2], points[-1]
    else:
        prev_point, next_point = points[index - 1], points[index + 1]
    dx_val = next_point[0] - prev_point[0]
    dy_val = next_point[1] - prev_point[1]
    length = math.hypot(dx_val, dy_val)
    if length <= 1e-6:
        return (1.0, 0.0)
    return (dx_val / length, dy_val / length)


def _local_band_polygon(points: list[tuple[float, float]], half_width: float, end_padding: float) -> list[tuple[float, float]]:
    if not points:
        return []
    if len(points) == 1:
        x_val, y_val = points[0]
        return [
            (x_val - half_width, y_val - half_width),
            (x_val + half_width, y_val - half_width),
            (x_val + half_width, y_val + half_width),
            (x_val - half_width, y_val + half_width),
        ]

    padded_points = list(points)
    start_tangent = _tangent_at(points, 0)
    end_tangent = _tangent_at(points, len(points) - 1)
    padded_points[0] = (
        padded_points[0][0] - start_tangent[0] * end_padding,
        padded_points[0][1] - start_tangent[1] * end_padding,
    )
    padded_points[-1] = (
        padded_points[-1][0] + end_tangent[0] * end_padding,
        padded_points[-1][1] + end_tangent[1] * end_padding,
    )

    left_side = []
    right_side = []
    for index, point in enumerate(padded_points):
        tangent = _tangent_at(padded_points, index)
        normal = (-tangent[1], tangent[0])
        left_side.append((point[0] + normal[0] * half_width, point[1] + normal[1] * half_width))
        right_side.append((point[0] - normal[0] * half_width, point[1] - normal[1] * half_width))
    return left_side + list(reversed(right_side))


def _coords_for_local_tangent_line(
    line: PageXmlLine,
    gnn_page,
    half_width: float,
    end_padding: float,
    point_snap_radius: float,
) -> tuple[list[tuple[float, float]], dict]:
    curve = normalize_open_curve(line.baseline_points)
    curve_points = list(curve.points)
    label = _numeric_suffix(line.line_custom)
    snapped_count = 0
    if label is not None:
        candidate_points = line_label_points(gnn_page, label)
        snapped_points = [
            point for point in candidate_points if _point_distance_to_curve(point, curve_points) <= point_snap_radius
        ]
        snapped_count = len(snapped_points)
        if snapped_points:
            min_x = min(point[0] for point in snapped_points)
            max_x = max(point[0] for point in snapped_points)
            min_y = min(point[1] for point in snapped_points)
            max_y = max(point[1] for point in snapped_points)
            half_width = max(half_width, min(max(max_x - min_x, max_y - min_y) * 0.08, half_width * 1.7))
    polygon = _local_band_polygon(curve_points, half_width, end_padding)
    return polygon, {
        "line_id": line.line_id,
        "line_custom": line.line_custom,
        "cut_point": curve.cut_point,
        "cut_index": curve.cut_index,
        "was_closed": curve.was_closed,
        "removed_mirrored_tail": curve.removed_mirrored_tail,
        "snapped_gnn_point_count": snapped_count,
        "half_width_px": half_width,
    }


def run_current_implementation_control(
    dataset_config,
    page_id: str,
    output_xml_path: str | Path,
) -> SegmentationResult:
    source_xml_path = dataset_config.pagexml_dir / f"{page_id}.xml"
    lines = parse_pagexml_lines(source_xml_path)
    coords_by_line_custom = {line.line_custom: line.coords_points for line in lines if line.coords_points}
    output_xml_path = rewrite_textline_coords(source_xml_path, output_xml_path, coords_by_line_custom)
    return SegmentationResult(
        page_id=page_id,
        strategy_name="current_implementation_control",
        output_xml_path=Path(output_xml_path),
        coords_by_line_custom=coords_by_line_custom,
        metadata={
            "strategy_name": "current_implementation_control",
            "source_xml_path": str(source_xml_path.resolve()),
            "line_count": len(coords_by_line_custom),
        },
    )


def run_local_tangent_band_v1(
    dataset_config,
    page_id: str,
    output_xml_path: str | Path,
) -> SegmentationResult:
    source_xml_path = dataset_config.pagexml_dir / f"{page_id}.xml"
    lines = parse_pagexml_lines(source_xml_path)
    gnn_page = parse_gnn_format_page(dataset_config.gnn_dir, page_id)
    config = dict(dataset_config.segmentation_config)
    coords_by_line_custom = {}
    line_metadata = []
    for line in lines:
        polygon, metadata = _coords_for_local_tangent_line(
            line,
            gnn_page,
            half_width=float(config.get("half_width_px", 42)),
            end_padding=float(config.get("end_padding_px", 14)),
            point_snap_radius=float(config.get("point_snap_radius_px", 65)),
        )
        if len(polygon) < 3:
            polygon = list(line.coords_points)
        coords_by_line_custom[line.line_custom] = polygon
        line_metadata.append(metadata)
    output_xml_path = rewrite_textline_coords(source_xml_path, output_xml_path, coords_by_line_custom)
    return SegmentationResult(
        page_id=page_id,
        strategy_name="local_tangent_band_v1",
        output_xml_path=Path(output_xml_path),
        coords_by_line_custom=coords_by_line_custom,
        metadata={
            "strategy_name": "local_tangent_band_v1",
            "source_xml_path": str(source_xml_path.resolve()),
            "gnn_dir": str(dataset_config.gnn_dir.resolve()),
            "segmentation_config": config,
            "line_count": len(coords_by_line_custom),
            "lines": line_metadata,
        },
    )


STRATEGY_REGISTRY = {
    "current_implementation_control": run_current_implementation_control,
    "local_tangent_band_v1": run_local_tangent_band_v1,
}


def run_segmentation_strategy(dataset_config, page_id: str, output_xml_path: str | Path) -> SegmentationResult:
    try:
        strategy = STRATEGY_REGISTRY[dataset_config.strategy_name]
    except KeyError as exc:
        raise ValueError(f"Unknown circular segmentation strategy: {dataset_config.strategy_name}") from exc
    return strategy(dataset_config, page_id, output_xml_path)

