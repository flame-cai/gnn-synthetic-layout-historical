from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

from .geometry import nearest_point_on_polyline, normalize_baseline_topology, sample_polyline


DEFAULT_UNWRAP_CONFIG = {
    "mirror_match_tolerance_px": 4.0,
    "closed_path_tolerance_px": 12.0,
    "min_mirror_pairs": 3,
    "straightness_chord_ratio": 0.985,
    "horizontal_angle_degrees": 12.0,
    "reading_order": "left_to_right",
    "circular_direction": "clockwise",
    "minimum_half_width_px": 8.0,
    "maximum_half_width_px": 180.0,
    "default_half_width_px": 24.0,
    "sample_spacing_px": 1.0,
}


@dataclass(frozen=True)
class UnwrappedLineCrop:
    image: np.ndarray
    metadata: dict


def _normalise_config(config: dict | None) -> dict:
    merged = dict(DEFAULT_UNWRAP_CONFIG)
    merged.update(dict(config or {}))
    for key in (
        "mirror_match_tolerance_px",
        "closed_path_tolerance_px",
        "straightness_chord_ratio",
        "horizontal_angle_degrees",
        "minimum_half_width_px",
        "maximum_half_width_px",
        "default_half_width_px",
        "sample_spacing_px",
    ):
        merged[key] = float(merged[key])
    merged["min_mirror_pairs"] = int(merged["min_mirror_pairs"])
    merged["reading_order"] = str(merged["reading_order"])
    merged["circular_direction"] = str(merged["circular_direction"])
    return merged


def should_unwrap_strategy(strategy_name: str | None) -> bool:
    return strategy_name == "local_tangent_band_v1"


def _estimate_half_width(polygon_points: list[list[int]], baseline_points: list[list[float]], config: dict) -> float:
    if not polygon_points or len(baseline_points) < 2:
        return float(config["default_half_width_px"])
    distances = [
        nearest_point_on_polyline(point, baseline_points).distance
        for point in polygon_points
    ]
    finite_distances = [value for value in distances if math.isfinite(value) and value > 0]
    if not finite_distances:
        return float(config["default_half_width_px"])
    estimated = float(np.median(finite_distances))
    return float(np.clip(estimated, config["minimum_half_width_px"], config["maximum_half_width_px"]))


def _polygon_mask(shape: tuple[int, int], polygon_points: list[list[int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    if len(polygon_points) >= 3:
        polygon = np.asarray(polygon_points, dtype=np.int32)
        cv2.fillPoly(mask, [polygon], 255)
    return mask


def _trim_to_mask(image: np.ndarray, mask: np.ndarray, page_median_color: int) -> tuple[np.ndarray, np.ndarray]:
    rows = np.where(mask.max(axis=1) > 0)[0]
    cols = np.where(mask.max(axis=0) > 0)[0]
    if rows.size == 0 or cols.size == 0:
        return image, mask
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    x0, x1 = int(cols[0]), int(cols[-1]) + 1
    trimmed_image = image[y0:y1, x0:x1]
    trimmed_mask = mask[y0:y1, x0:x1]
    if trimmed_image.size == 0:
        fallback = np.full((1, 1), page_median_color, dtype=np.uint8)
        return fallback, np.zeros((1, 1), dtype=np.uint8)
    return trimmed_image, trimmed_mask


def unwrap_line_crop_for_ocr(
    processing_image: np.ndarray,
    polygon_points: list[list[int]],
    baseline_points: list[list[int]],
    *,
    text: str = "",
    unwrap_config: dict | None = None,
) -> UnwrappedLineCrop:
    config = _normalise_config(unwrap_config)
    page_median_color = int(np.median(processing_image))
    topology = normalize_baseline_topology(
        baseline_points,
        mirror_match_tolerance=config["mirror_match_tolerance_px"],
        closed_path_tolerance=config["closed_path_tolerance_px"],
        min_mirror_pairs=config["min_mirror_pairs"],
        straightness_chord_ratio=config["straightness_chord_ratio"],
        horizontal_angle_degrees=config["horizontal_angle_degrees"],
        reading_order=config["reading_order"],
        circular_direction=config["circular_direction"],
    )
    normalized_points = topology.normalized_points
    baseline_length = topology.baseline_length
    if len(normalized_points) < 2 or baseline_length <= 1e-6:
        return UnwrappedLineCrop(
            image=np.full((1, 1), page_median_color, dtype=np.uint8),
            metadata={
                "unwrap_strategy": "baseline_local_tangent",
                "fallback_reason": "baseline_too_short",
                "topology": topology.to_metadata(),
            },
        )

    half_width = _estimate_half_width(polygon_points, normalized_points, config)
    output_width = max(1, int(math.ceil(baseline_length / max(config["sample_spacing_px"], 1e-6))))
    output_height = max(2, int(math.ceil(half_width * 2.0)))
    centers, tangents, _ = sample_polyline(normalized_points, output_width)
    row_offsets = np.linspace(-half_width, half_width, output_height, dtype=np.float32)

    map_x = np.zeros((output_height, output_width), dtype=np.float32)
    map_y = np.zeros((output_height, output_width), dtype=np.float32)
    for col_index, (center, tangent) in enumerate(zip(centers, tangents)):
        normal = (-float(tangent[1]), float(tangent[0]))
        for row_index, offset in enumerate(row_offsets):
            map_x[row_index, col_index] = float(center[0]) + normal[0] * float(offset)
            map_y[row_index, col_index] = float(center[1]) + normal[1] * float(offset)

    page_mask = _polygon_mask(processing_image.shape[:2], polygon_points)
    unwrapped = cv2.remap(
        processing_image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=page_median_color,
    )
    unwrapped_mask = cv2.remap(
        page_mask,
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    unwrapped[unwrapped_mask < 128] = page_median_color
    unwrapped, unwrapped_mask = _trim_to_mask(unwrapped, unwrapped_mask, page_median_color)

    orientation_selection_mode = "supervised_text_equiv_baseline_order" if text else "geometry_only_baseline_order"
    metadata = {
        "unwrap_strategy": "baseline_local_tangent",
        "topology": topology.to_metadata(),
        "baseline_length_px": baseline_length,
        "output_width_px": int(unwrapped.shape[1]),
        "output_height_px": int(unwrapped.shape[0]),
        "estimated_half_width_px": half_width,
        "page_median_color": page_median_color,
        "foreground_pixel_count": int(np.count_nonzero(unwrapped_mask >= 128)),
        "orientation": {
            "candidate_transforms": ["identity", "rotate_180"],
            "selected_transform": "identity",
            "selection_mode": orientation_selection_mode,
            "reading_order": config["reading_order"],
            "circular_direction": config["circular_direction"],
            "reason": topology.orientation_action,
        },
    }
    return UnwrappedLineCrop(image=unwrapped, metadata=metadata)
