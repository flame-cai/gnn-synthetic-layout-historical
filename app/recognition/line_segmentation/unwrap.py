from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

from .geometry import nearest_point_on_polyline, normalize_baseline_topology, polyline_length, sample_polyline


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
    "horizontal_fit_min_baseline_length_px": 16.0,
    "horizontal_fit_residual_abs_px": 2.0,
    "horizontal_fit_residual_half_width_ratio": 0.15,
    "stable_unwrap_smoothing_window_px": 9.0,
    "stable_unwrap_closed_smoothing_window_px": 7.0,
    "stable_unwrap_max_deviation_half_width_ratio": 0.5,
}


@dataclass(frozen=True)
class UnwrappedLineCrop:
    image: np.ndarray
    metadata: dict


def _page_median_color(processing_image: np.ndarray, config: dict) -> int:
    if config.get("page_median_color") is not None:
        return int(config["page_median_color"])
    page_median_color = int(np.median(processing_image))
    config["page_median_color"] = page_median_color
    return page_median_color


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
        "horizontal_fit_min_baseline_length_px",
        "horizontal_fit_residual_abs_px",
        "horizontal_fit_residual_half_width_ratio",
        "stable_unwrap_smoothing_window_px",
        "stable_unwrap_closed_smoothing_window_px",
        "stable_unwrap_max_deviation_half_width_ratio",
    ):
        merged[key] = float(merged[key])
    merged["min_mirror_pairs"] = int(merged["min_mirror_pairs"])
    merged["reading_order"] = str(merged["reading_order"])
    merged["circular_direction"] = str(merged["circular_direction"])
    if "reading_direction" in merged and merged["reading_direction"] is not None:
        merged["reading_direction"] = list(merged["reading_direction"])
    if "reading_cut_point" in merged and merged["reading_cut_point"] is not None:
        merged["reading_cut_point"] = list(merged["reading_cut_point"])
    if merged.get("page_median_color") is not None:
        merged["page_median_color"] = int(merged["page_median_color"])
    return merged


def should_unwrap_strategy(strategy_name: str | None) -> bool:
    return strategy_name in {
        "local_tangent_band_v1",
        "local_polygons_v1",
        "local_polygons_hstraight_smooth_unwrap_v1",
        "local_polygons_stable_unwrap_v1",
    }


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


def _unit(vector_x: float, vector_y: float) -> tuple[float, float]:
    length = math.hypot(vector_x, vector_y)
    if length <= 1e-6:
        return 1.0, 0.0
    return vector_x / length, vector_y / length


def _point_baseline_tangent(topology) -> tuple[float, float]:
    direction = topology.reading_direction
    if direction is None or len(direction) < 2:
        return 1.0, 0.0
    return _unit(float(direction[0]), float(direction[1]))


def _point_at_station(
    points: list[list[float]],
    station: float,
    *,
    is_closed: bool,
    baseline_length: float,
    fallback_tangent: tuple[float, float] | None = None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    if not points:
        return (0.0, 0.0), (1.0, 0.0)
    if len(points) == 1 or baseline_length <= 1e-6:
        tangent = fallback_tangent or (1.0, 0.0)
        return (points[0][0] + tangent[0] * station, points[0][1] + tangent[1] * station), tangent
    if is_closed:
        station = station % baseline_length

    first_tangent = _unit(points[1][0] - points[0][0], points[1][1] - points[0][1])
    if station <= 0.0:
        return (
            points[0][0] + first_tangent[0] * station,
            points[0][1] + first_tangent[1] * station,
        ), first_tangent

    arc_before = 0.0
    last_tangent = first_tangent
    for index in range(len(points) - 1):
        start = points[index]
        end = points[index + 1]
        dx_val = end[0] - start[0]
        dy_val = end[1] - start[1]
        segment_length = math.hypot(dx_val, dy_val)
        if segment_length <= 1e-6:
            continue
        tangent = _unit(dx_val, dy_val)
        last_tangent = tangent
        if station <= arc_before + segment_length:
            ratio = (station - arc_before) / segment_length
            return (
                start[0] + ratio * dx_val,
                start[1] + ratio * dy_val,
            ), tangent
        arc_before += segment_length

    end = points[-1]
    overflow = station - baseline_length
    return (
        end[0] + last_tangent[0] * overflow,
        end[1] + last_tangent[1] * overflow,
    ), last_tangent


def _station_range_from_config(config: dict, baseline_length: float) -> tuple[float, float]:
    station_min = 0.0
    station_max = baseline_length
    try:
        configured_min = float(config["local_s_min"])
        configured_max = float(config["local_s_max"])
    except Exception:
        return station_min, station_max
    if math.isfinite(configured_min) and math.isfinite(configured_max) and configured_max > configured_min:
        return configured_min, configured_max
    return station_min, station_max


def _half_width_from_local_bounds(config: dict) -> float | None:
    try:
        configured_min = float(config["local_n_min"])
        configured_max = float(config["local_n_max"])
    except Exception:
        return None
    if not (math.isfinite(configured_min) and math.isfinite(configured_max) and configured_max > configured_min):
        return None
    half_width = max(abs(configured_min), abs(configured_max))
    return float(np.clip(half_width, config["minimum_half_width_px"], config["maximum_half_width_px"]))


def _odd_window_size(window_px: float, sample_spacing_px: float, sample_count: int) -> int:
    if sample_count < 3:
        return 1
    window = max(1, int(round(float(window_px) / max(float(sample_spacing_px), 1e-6))))
    if window % 2 == 0:
        window += 1
    return max(1, min(window, sample_count if sample_count % 2 == 1 else sample_count - 1))


def _smooth_centers(centers: np.ndarray, *, window_size: int, is_closed: bool) -> np.ndarray:
    if window_size <= 1 or len(centers) < 3:
        return centers
    pad = window_size // 2
    kernel = np.ones(window_size, dtype=np.float64) / float(window_size)
    smoothed = np.empty_like(centers, dtype=np.float64)
    for axis_index in range(2):
        values = centers[:, axis_index].astype(np.float64)
        if is_closed:
            padded = np.pad(values, (pad, pad), mode="wrap")
        else:
            padded = np.pad(values, (pad, pad), mode="edge")
        smoothed[:, axis_index] = np.convolve(padded, kernel, mode="valid")
    return smoothed


def _tangents_from_centers(centers: np.ndarray, *, is_closed: bool) -> np.ndarray:
    sample_count = len(centers)
    if sample_count < 2:
        return np.asarray([[1.0, 0.0]], dtype=np.float64)
    if is_closed and sample_count >= 3:
        deltas = np.roll(centers, -1, axis=0) - np.roll(centers, 1, axis=0)
    else:
        deltas = np.gradient(centers, axis=0)
    norms = np.linalg.norm(deltas, axis=1)
    fallback = np.asarray([1.0, 0.0], dtype=np.float64)
    tangents = np.empty_like(deltas, dtype=np.float64)
    last_valid = fallback
    for index, norm in enumerate(norms):
        if norm > 1e-6:
            last_valid = deltas[index] / norm
        tangents[index] = last_valid
    return tangents


def _sample_centers_at_stations(
    points: list[list[float]],
    stations: np.ndarray,
    *,
    is_closed: bool,
    baseline_length: float,
    fallback_tangent: tuple[float, float] | None = None,
) -> np.ndarray:
    if not points:
        return np.zeros((len(stations), 2), dtype=np.float64)
    point_array = np.asarray(points, dtype=np.float64)
    if len(point_array) == 1 or baseline_length <= 1e-6:
        tangent = np.asarray(fallback_tangent or (1.0, 0.0), dtype=np.float64)
        centers = np.repeat(point_array[:1], len(stations), axis=0)
        centers += stations[:, None] * tangent[None, :]
        return centers

    deltas = np.diff(point_array, axis=0)
    segment_lengths = np.linalg.norm(deltas, axis=1)
    valid = segment_lengths > 1e-6
    if not np.any(valid):
        tangent = np.asarray(fallback_tangent or (1.0, 0.0), dtype=np.float64)
        centers = np.repeat(point_array[:1], len(stations), axis=0)
        centers += stations[:, None] * tangent[None, :]
        return centers

    segment_starts = point_array[:-1][valid]
    segment_deltas = deltas[valid]
    segment_lengths = segment_lengths[valid]
    segment_starts_arc = np.concatenate(([0.0], np.cumsum(segment_lengths)[:-1]))
    segment_ends_arc = segment_starts_arc + segment_lengths
    first_tangent = segment_deltas[0] / segment_lengths[0]
    last_tangent = segment_deltas[-1] / segment_lengths[-1]

    sample_stations = stations.astype(np.float64, copy=True)
    if is_closed:
        sample_stations = np.mod(sample_stations, max(float(baseline_length), 1e-6))

    centers = np.empty((len(sample_stations), 2), dtype=np.float64)
    before_mask = sample_stations < 0.0
    after_mask = sample_stations > baseline_length
    inside_mask = ~(before_mask | after_mask)
    if np.any(before_mask):
        centers[before_mask] = point_array[0] + sample_stations[before_mask, None] * first_tangent[None, :]
    if np.any(after_mask):
        overflow = sample_stations[after_mask] - baseline_length
        centers[after_mask] = point_array[-1] + overflow[:, None] * last_tangent[None, :]
    if np.any(inside_mask):
        inside_stations = sample_stations[inside_mask]
        indices = np.searchsorted(segment_ends_arc, inside_stations, side="left")
        indices = np.clip(indices, 0, len(segment_lengths) - 1)
        ratios = (inside_stations - segment_starts_arc[indices]) / segment_lengths[indices]
        centers[inside_mask] = segment_starts[indices] + ratios[:, None] * segment_deltas[indices]
    return centers


def _sample_stable_path(
    points: list[list[float]],
    *,
    station_min: float,
    station_max: float,
    output_width: int,
    topology_is_closed: bool,
    baseline_length: float,
    config: dict,
    fallback_tangent: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    station_span = station_max - station_min
    if topology_is_closed:
        stations = station_min + (np.arange(output_width, dtype=np.float64) * station_span / max(output_width, 1))
    else:
        stations = np.linspace(station_min, station_max, output_width, dtype=np.float64)

    raw_centers_array = _sample_centers_at_stations(
        points,
        stations,
        is_closed=topology_is_closed,
        baseline_length=baseline_length,
        fallback_tangent=fallback_tangent,
    )

    window_key = "stable_unwrap_closed_smoothing_window_px" if topology_is_closed else "stable_unwrap_smoothing_window_px"
    window_size = _odd_window_size(
        float(config[window_key]),
        float(config["sample_spacing_px"]),
        output_width,
    )
    smoothed_centers = _smooth_centers(
        raw_centers_array,
        window_size=window_size,
        is_closed=topology_is_closed,
    )
    tangents = _tangents_from_centers(smoothed_centers, is_closed=topology_is_closed)

    deviations = np.linalg.norm(smoothed_centers - raw_centers_array, axis=1) if len(smoothed_centers) else np.asarray([])
    steps = np.linalg.norm(
        np.roll(smoothed_centers, -1, axis=0) - smoothed_centers,
        axis=1,
    ) if topology_is_closed and len(smoothed_centers) > 1 else np.linalg.norm(np.diff(smoothed_centers, axis=0), axis=1)
    duplicate_step_count = int(np.count_nonzero(steps <= 0.25)) if steps.size else 0
    metadata = {
        "station_sampling": "endpoint_exclusive_closed" if topology_is_closed else "endpoint_inclusive_open",
        "smoothing_window_px": float(config[window_key]),
        "smoothing_window_samples": int(window_size),
        "vectorized_station_sampling": True,
        "center_deviation_p95_px": float(np.percentile(deviations, 95)) if deviations.size else 0.0,
        "center_deviation_max_px": float(np.max(deviations)) if deviations.size else 0.0,
        "duplicate_step_count": duplicate_step_count,
        "minimum_step_px": float(np.min(steps)) if steps.size else None,
    }
    return stations, smoothed_centers, tangents, metadata


def _stable_unwrap_fallback(
    processing_image: np.ndarray,
    polygon_points: list[list[int]],
    baseline_points: list[list[int]],
    *,
    text: str,
    unwrap_config: dict,
    fallback_reason: str,
    stable_metadata: dict | None = None,
) -> UnwrappedLineCrop:
    fallback = unwrap_line_crop_for_ocr(
        processing_image,
        polygon_points,
        baseline_points,
        text=text,
        unwrap_config=unwrap_config,
    )
    fallback.metadata["requested_unwrap_strategy"] = "stable_arclength_tangent"
    fallback.metadata["stable_unwrap"] = {
        "used_stable_path": False,
        "fallback_reason": fallback_reason,
        **dict(stable_metadata or {}),
    }
    return fallback


def _horizontal_fit_fallback(
    processing_image: np.ndarray,
    polygon_points: list[list[int]],
    baseline_points: list[list[int]],
    *,
    text: str,
    unwrap_config: dict,
    fallback_reason: str,
    fit_metadata: dict | None = None,
) -> UnwrappedLineCrop:
    fallback = unwrap_line_crop_for_ocr(
        processing_image,
        polygon_points,
        baseline_points,
        text=text,
        unwrap_config=unwrap_config,
    )
    fallback.metadata["requested_unwrap_strategy"] = "horizontal_straight_fit_tangent"
    fallback.metadata["horizontal_straight_fit"] = {
        "eligible": False,
        "fallback_reason": fallback_reason,
        **dict(fit_metadata or {}),
    }
    return fallback


def _horizontal_fit_line(
    points: list[list[float]],
    half_width: float,
    config: dict,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float], dict, str | None]:
    if len(points) < 2:
        return (0.0, 0.0), (1.0, 0.0), (0.0, 1.0), {}, "not_enough_points"

    baseline_length = polyline_length(points)
    if baseline_length < float(config["horizontal_fit_min_baseline_length_px"]):
        return (
            (float(points[0][0]), float(points[0][1])),
            (1.0, 0.0),
            (0.0, 1.0),
            {"baseline_length_px": baseline_length},
            "baseline_too_short_for_horizontal_fit",
        )

    first = np.asarray(points[0], dtype=float)
    last = np.asarray(points[-1], dtype=float)
    chord = last - first
    chord_length = float(np.linalg.norm(chord))
    if chord_length <= 1e-6:
        return (
            (float(first[0]), float(first[1])),
            (1.0, 0.0),
            (0.0, 1.0),
            {"chord_length_px": chord_length},
            "horizontal_fit_chord_too_short",
        )

    tangent = chord / chord_length
    normal = np.asarray([-tangent[1], tangent[0]], dtype=float)
    point_array = np.asarray(points, dtype=float)
    normal_offsets = (point_array - first[None, :]) @ normal
    median_offset = float(np.median(normal_offsets))
    residuals = np.abs(normal_offsets - median_offset)
    residual_p95 = float(np.percentile(residuals, 95)) if residuals.size else 0.0
    residual_max = float(np.max(residuals)) if residuals.size else 0.0
    threshold = max(
        float(config["horizontal_fit_residual_abs_px"]),
        float(config["horizontal_fit_residual_half_width_ratio"]) * max(float(half_width), 1.0),
    )
    metadata = {
        "baseline_length_px": baseline_length,
        "chord_length_px": chord_length,
        "normal_median_offset_px": median_offset,
        "residual_p95_px": residual_p95,
        "residual_max_px": residual_max,
        "residual_threshold_px": threshold,
        "tangent": [float(tangent[0]), float(tangent[1])],
        "normal": [float(normal[0]), float(normal[1])],
    }
    if residual_p95 > threshold:
        return (
            (float(first[0] + normal[0] * median_offset), float(first[1] + normal[1] * median_offset)),
            (float(tangent[0]), float(tangent[1])),
            (float(normal[0]), float(normal[1])),
            metadata,
            "horizontal_fit_residual_too_large",
        )

    anchor = first + normal * median_offset
    return (
        (float(anchor[0]), float(anchor[1])),
        (float(tangent[0]), float(tangent[1])),
        (float(normal[0]), float(normal[1])),
        metadata,
        None,
    )


def unwrap_horizontal_straight_fit_line_crop_for_ocr(
    processing_image: np.ndarray,
    polygon_points: list[list[int]],
    baseline_points: list[list[int]],
    *,
    text: str = "",
    unwrap_config: dict | None = None,
) -> UnwrappedLineCrop:
    config = _normalise_config(unwrap_config)
    page_median_color = _page_median_color(processing_image, config)
    topology = normalize_baseline_topology(
        baseline_points,
        mirror_match_tolerance=config["mirror_match_tolerance_px"],
        closed_path_tolerance=config["closed_path_tolerance_px"],
        min_mirror_pairs=config["min_mirror_pairs"],
        straightness_chord_ratio=config["straightness_chord_ratio"],
        horizontal_angle_degrees=config["horizontal_angle_degrees"],
        reading_order=config["reading_order"],
        circular_direction=config["circular_direction"],
        reading_direction=config.get("reading_direction"),
        reading_cut_point=config.get("reading_cut_point"),
    )
    if topology.line_kind != "horizontal_straight" or topology.is_closed:
        return _horizontal_fit_fallback(
            processing_image,
            polygon_points,
            baseline_points,
            text=text,
            unwrap_config=config,
            fallback_reason=f"line_kind:{topology.line_kind}",
            fit_metadata={"topology": topology.to_metadata()},
        )

    normalized_points = topology.normalized_points
    baseline_length = topology.baseline_length
    station_min, station_max = _station_range_from_config(config, baseline_length)
    station_span = station_max - station_min
    if len(normalized_points) < 2 or baseline_length <= 1e-6:
        return _horizontal_fit_fallback(
            processing_image,
            polygon_points,
            baseline_points,
            text=text,
            unwrap_config=config,
            fallback_reason="baseline_too_short",
            fit_metadata={"topology": topology.to_metadata()},
        )

    half_width = _half_width_from_local_bounds(config)
    if half_width is None:
        half_width = _estimate_half_width(polygon_points, normalized_points, config)
    anchor, tangent, normal, fit_metadata, fallback_reason = _horizontal_fit_line(
        normalized_points,
        half_width,
        config,
    )
    if fallback_reason:
        return _horizontal_fit_fallback(
            processing_image,
            polygon_points,
            baseline_points,
            text=text,
            unwrap_config=config,
            fallback_reason=fallback_reason,
            fit_metadata={**fit_metadata, "topology": topology.to_metadata()},
        )

    output_width = max(1, int(math.ceil(station_span / max(config["sample_spacing_px"], 1e-6))))
    output_height = max(2, int(math.ceil(half_width * 2.0)))
    stations = np.linspace(station_min, station_max, output_width, dtype=np.float32)
    row_offsets = np.linspace(-half_width, half_width, output_height, dtype=np.float32)
    centers_x = float(anchor[0]) + float(tangent[0]) * stations
    centers_y = float(anchor[1]) + float(tangent[1]) * stations
    map_x = centers_x[None, :] + float(normal[0]) * row_offsets[:, None]
    map_y = centers_y[None, :] + float(normal[1]) * row_offsets[:, None]
    map_x = np.ascontiguousarray(map_x, dtype=np.float32)
    map_y = np.ascontiguousarray(map_y, dtype=np.float32)

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
    background_pixel_count = int(np.count_nonzero(unwrapped_mask < 128))
    total_pixel_count = int(unwrapped_mask.size)

    orientation_selection_mode = "supervised_text_equiv_baseline_order" if text else "geometry_only_baseline_order"
    metadata = {
        "unwrap_strategy": "horizontal_straight_fit_tangent",
        "topology": topology.to_metadata(),
        "baseline_length_px": baseline_length,
        "station_min_px": station_min,
        "station_max_px": station_max,
        "station_span_px": station_span,
        "output_width_px": int(unwrapped.shape[1]),
        "output_height_px": int(unwrapped.shape[0]),
        "estimated_half_width_px": half_width,
        "page_median_color": page_median_color,
        "foreground_pixel_count": int(np.count_nonzero(unwrapped_mask >= 128)),
        "background_pixel_count": background_pixel_count,
        "median_background_fraction": (background_pixel_count / total_pixel_count) if total_pixel_count else None,
        "horizontal_straight_fit": {
            "eligible": True,
            "vectorized_map": True,
            **fit_metadata,
        },
        "orientation": {
            "candidate_transforms": ["identity", "rotate_180"],
            "selected_transform": "identity",
            "selection_mode": orientation_selection_mode,
            "reading_order": config["reading_order"],
            "circular_direction": config["circular_direction"],
            "reading_direction": config.get("reading_direction"),
            "reading_cut_point": config.get("reading_cut_point"),
            "reason": topology.orientation_action,
        },
    }
    return UnwrappedLineCrop(image=unwrapped, metadata=metadata)


def unwrap_stable_line_crop_for_ocr(
    processing_image: np.ndarray,
    polygon_points: list[list[int]],
    baseline_points: list[list[int]],
    *,
    text: str = "",
    unwrap_config: dict | None = None,
) -> UnwrappedLineCrop:
    config = _normalise_config(unwrap_config)
    page_median_color = _page_median_color(processing_image, config)
    topology = normalize_baseline_topology(
        baseline_points,
        mirror_match_tolerance=config["mirror_match_tolerance_px"],
        closed_path_tolerance=config["closed_path_tolerance_px"],
        min_mirror_pairs=config["min_mirror_pairs"],
        straightness_chord_ratio=config["straightness_chord_ratio"],
        horizontal_angle_degrees=config["horizontal_angle_degrees"],
        reading_order=config["reading_order"],
        circular_direction=config["circular_direction"],
        reading_direction=config.get("reading_direction"),
        reading_cut_point=config.get("reading_cut_point"),
    )
    normalized_points = topology.normalized_points
    baseline_length = topology.baseline_length
    station_min, station_max = _station_range_from_config(config, baseline_length)
    station_span = station_max - station_min
    can_unwrap_point_baseline = len(normalized_points) == 1 and station_span > 1e-6
    if (len(normalized_points) < 2 or baseline_length <= 1e-6) and not can_unwrap_point_baseline:
        return _stable_unwrap_fallback(
            processing_image,
            polygon_points,
            baseline_points,
            text=text,
            unwrap_config=config,
            fallback_reason="baseline_too_short",
            stable_metadata={"topology": topology.to_metadata()},
        )

    half_width = _half_width_from_local_bounds(config)
    if half_width is None:
        half_width = _estimate_half_width(polygon_points, normalized_points, config)
    output_width = max(1, int(math.ceil(station_span / max(config["sample_spacing_px"], 1e-6))))
    output_height = max(2, int(math.ceil(half_width * 2.0)))
    _, centers, tangents, stable_metadata = _sample_stable_path(
        normalized_points,
        station_min=station_min,
        station_max=station_max,
        output_width=output_width,
        topology_is_closed=topology.is_closed,
        baseline_length=baseline_length,
        config=config,
        fallback_tangent=_point_baseline_tangent(topology),
    )
    max_allowed_deviation = max(
        1.0,
        float(config["stable_unwrap_max_deviation_half_width_ratio"]) * max(float(half_width), 1.0),
    )
    if stable_metadata["center_deviation_p95_px"] > max_allowed_deviation:
        return _stable_unwrap_fallback(
            processing_image,
            polygon_points,
            baseline_points,
            text=text,
            unwrap_config=config,
            fallback_reason="stable_path_deviation_too_large",
            stable_metadata={
                **stable_metadata,
                "max_allowed_deviation_px": max_allowed_deviation,
                "topology": topology.to_metadata(),
            },
        )

    normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)
    row_offsets = np.linspace(-half_width, half_width, output_height, dtype=np.float32)
    map_x = centers[:, 0][None, :] + row_offsets[:, None] * normals[:, 0][None, :]
    map_y = centers[:, 1][None, :] + row_offsets[:, None] * normals[:, 1][None, :]
    map_x = np.ascontiguousarray(map_x, dtype=np.float32)
    map_y = np.ascontiguousarray(map_y, dtype=np.float32)

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
    foreground_pixel_count = int(np.count_nonzero(unwrapped_mask >= 128))
    if foreground_pixel_count <= 0:
        return _stable_unwrap_fallback(
            processing_image,
            polygon_points,
            baseline_points,
            text=text,
            unwrap_config=config,
            fallback_reason="empty_unwrapped_mask",
            stable_metadata={
                **stable_metadata,
                "max_allowed_deviation_px": max_allowed_deviation,
                "topology": topology.to_metadata(),
            },
        )
    unwrapped[unwrapped_mask < 128] = page_median_color
    unwrapped, unwrapped_mask = _trim_to_mask(unwrapped, unwrapped_mask, page_median_color)
    background_pixel_count = int(np.count_nonzero(unwrapped_mask < 128))
    total_pixel_count = int(unwrapped_mask.size)

    orientation_selection_mode = "supervised_text_equiv_baseline_order" if text else "geometry_only_baseline_order"
    metadata = {
        "unwrap_strategy": "stable_arclength_tangent",
        "topology": topology.to_metadata(),
        "baseline_length_px": baseline_length,
        "station_min_px": station_min,
        "station_max_px": station_max,
        "station_span_px": station_span,
        "output_width_px": int(unwrapped.shape[1]),
        "output_height_px": int(unwrapped.shape[0]),
        "estimated_half_width_px": half_width,
        "page_median_color": page_median_color,
        "foreground_pixel_count": int(np.count_nonzero(unwrapped_mask >= 128)),
        "background_pixel_count": background_pixel_count,
        "median_background_fraction": (background_pixel_count / total_pixel_count) if total_pixel_count else None,
        "stable_unwrap": {
            "used_stable_path": True,
            "vectorized_map": True,
            "max_allowed_deviation_px": max_allowed_deviation,
            **stable_metadata,
        },
        "orientation": {
            "candidate_transforms": ["identity", "rotate_180"],
            "selected_transform": "identity",
            "selection_mode": orientation_selection_mode,
            "reading_order": config["reading_order"],
            "circular_direction": config["circular_direction"],
            "reading_direction": config.get("reading_direction"),
            "reading_cut_point": config.get("reading_cut_point"),
            "reason": topology.orientation_action,
        },
    }
    return UnwrappedLineCrop(image=unwrapped, metadata=metadata)


def unwrap_line_crop_for_ocr(
    processing_image: np.ndarray,
    polygon_points: list[list[int]],
    baseline_points: list[list[int]],
    *,
    text: str = "",
    unwrap_config: dict | None = None,
) -> UnwrappedLineCrop:
    config = _normalise_config(unwrap_config)
    page_median_color = _page_median_color(processing_image, config)
    topology = normalize_baseline_topology(
        baseline_points,
        mirror_match_tolerance=config["mirror_match_tolerance_px"],
        closed_path_tolerance=config["closed_path_tolerance_px"],
        min_mirror_pairs=config["min_mirror_pairs"],
        straightness_chord_ratio=config["straightness_chord_ratio"],
        horizontal_angle_degrees=config["horizontal_angle_degrees"],
        reading_order=config["reading_order"],
        circular_direction=config["circular_direction"],
        reading_direction=config.get("reading_direction"),
        reading_cut_point=config.get("reading_cut_point"),
    )
    normalized_points = topology.normalized_points
    baseline_length = topology.baseline_length
    station_min, station_max = _station_range_from_config(config, baseline_length)
    station_span = station_max - station_min
    can_unwrap_point_baseline = len(normalized_points) == 1 and station_span > 1e-6
    if (len(normalized_points) < 2 or baseline_length <= 1e-6) and not can_unwrap_point_baseline:
        return UnwrappedLineCrop(
            image=np.full((1, 1), page_median_color, dtype=np.uint8),
            metadata={
                "unwrap_strategy": "baseline_local_tangent",
                "fallback_reason": "baseline_too_short",
                "topology": topology.to_metadata(),
            },
        )

    half_width = _half_width_from_local_bounds(config)
    if half_width is None:
        half_width = _estimate_half_width(polygon_points, normalized_points, config)
    output_width = max(1, int(math.ceil(station_span / max(config["sample_spacing_px"], 1e-6))))
    output_height = max(2, int(math.ceil(half_width * 2.0)))
    if baseline_length > 1e-6 and station_min == 0.0 and station_max == baseline_length:
        centers, tangents, _ = sample_polyline(normalized_points, output_width)
    else:
        centers = []
        tangents = []
        for sample_index in range(output_width):
            station = station_min + min(station_span, sample_index * station_span / max(output_width - 1, 1))
            center, tangent = _point_at_station(
                normalized_points,
                station,
                is_closed=topology.is_closed,
                baseline_length=baseline_length,
                fallback_tangent=_point_baseline_tangent(topology),
            )
            centers.append(center)
            tangents.append(tangent)
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
    background_pixel_count = int(np.count_nonzero(unwrapped_mask < 128))
    total_pixel_count = int(unwrapped_mask.size)

    orientation_selection_mode = "supervised_text_equiv_baseline_order" if text else "geometry_only_baseline_order"
    metadata = {
        "unwrap_strategy": "baseline_local_tangent",
        "topology": topology.to_metadata(),
        "baseline_length_px": baseline_length,
        "station_min_px": station_min,
        "station_max_px": station_max,
        "station_span_px": station_span,
        "output_width_px": int(unwrapped.shape[1]),
        "output_height_px": int(unwrapped.shape[0]),
        "estimated_half_width_px": half_width,
        "page_median_color": page_median_color,
        "foreground_pixel_count": int(np.count_nonzero(unwrapped_mask >= 128)),
        "background_pixel_count": background_pixel_count,
        "median_background_fraction": (background_pixel_count / total_pixel_count) if total_pixel_count else None,
        "orientation": {
            "candidate_transforms": ["identity", "rotate_180"],
            "selected_transform": "identity",
            "selection_mode": orientation_selection_mode,
            "reading_order": config["reading_order"],
            "circular_direction": config["circular_direction"],
            "reading_direction": config.get("reading_direction"),
            "reading_cut_point": config.get("reading_cut_point"),
            "reason": topology.orientation_action,
        },
    }
    return UnwrappedLineCrop(image=unwrapped, metadata=metadata)
