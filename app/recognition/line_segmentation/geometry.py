from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass(frozen=True)
class BaselineTopology:
    original_points: list[list[float]]
    normalized_points: list[list[float]]
    original_point_count: int
    normalized_point_count: int
    split_index: int | None
    mirror_match_tolerance: float
    mean_mirror_distance: float | None
    max_mirror_distance: float | None
    was_out_and_back: bool
    is_closed: bool
    closed_path_tolerance: float
    cut_index: int | None
    baseline_length: float
    line_kind: str
    orientation_action: str

    def to_metadata(self) -> dict:
        payload = asdict(self)
        payload.pop("original_points", None)
        payload.pop("normalized_points", None)
        return payload


@dataclass(frozen=True)
class NearestPoint:
    distance: float
    point: tuple[float, float]
    segment_index: int
    ratio: float
    tangent: tuple[float, float]
    arc_length: float


def as_float_points(points: Iterable[Iterable[float]]) -> list[list[float]]:
    return [[float(point[0]), float(point[1])] for point in points]


def distance(point_a: Iterable[float], point_b: Iterable[float]) -> float:
    ax_val, ay_val = point_a
    bx_val, by_val = point_b
    return math.hypot(float(ax_val) - float(bx_val), float(ay_val) - float(by_val))


def polyline_length(points: list[list[float]]) -> float:
    if len(points) < 2:
        return 0.0
    return sum(distance(points[index], points[index + 1]) for index in range(len(points) - 1))


def signed_area(points: list[list[float]]) -> float:
    if len(points) < 3:
        return 0.0
    total = 0.0
    for index, point in enumerate(points):
        next_point = points[(index + 1) % len(points)]
        total += point[0] * next_point[1] - next_point[0] * point[1]
    return total / 2.0


def _unit(vector_x: float, vector_y: float) -> tuple[float, float]:
    length = math.hypot(vector_x, vector_y)
    if length <= 1e-6:
        return 1.0, 0.0
    return vector_x / length, vector_y / length


def _detect_out_and_back_split(
    points: list[list[float]],
    mirror_match_tolerance: float,
    min_mirror_pairs: int,
) -> tuple[int | None, float | None, float | None]:
    best: tuple[int, int, float, float] | None = None
    point_count = len(points)
    for split_index in range(1, point_count - 1):
        pair_count = min(split_index, point_count - split_index - 1)
        if pair_count < min_mirror_pairs:
            continue
        distances = [
            distance(points[split_index + offset], points[split_index - offset])
            for offset in range(1, pair_count + 1)
        ]
        max_distance = max(distances)
        if max_distance > mirror_match_tolerance:
            continue
        mean_distance = sum(distances) / len(distances)
        candidate = (pair_count, split_index, mean_distance, max_distance)
        if best is None or candidate[0] > best[0] or (
            candidate[0] == best[0] and candidate[2] < best[2]
        ):
            best = candidate
    if best is None:
        return None, None, None
    _, split_index, mean_distance, max_distance = best
    return split_index, mean_distance, max_distance


def _cut_closed_path_at_top(points: list[list[float]]) -> tuple[list[list[float]], int | None]:
    if len(points) < 2:
        return points, None
    ring_points = list(points)
    if distance(ring_points[0], ring_points[-1]) <= 1e-6:
        ring_points = ring_points[:-1]
    if not ring_points:
        return points, None
    cut_index = min(range(len(ring_points)), key=lambda idx: (ring_points[idx][1], ring_points[idx][0]))
    cut_points = ring_points[cut_index:] + ring_points[:cut_index] + [ring_points[cut_index]]
    return cut_points, cut_index


def _line_kind(points: list[list[float]], closed: bool, straightness_chord_ratio: float, horizontal_angle_degrees: float) -> str:
    if closed:
        return "closed_circular"
    length = polyline_length(points)
    if len(points) < 2 or length <= 1e-6:
        return "point"
    chord = distance(points[0], points[-1])
    is_straight = chord / max(length, 1e-6) >= straightness_chord_ratio
    dx_val = points[-1][0] - points[0][0]
    dy_val = points[-1][1] - points[0][1]
    angle = abs(math.degrees(math.atan2(dy_val, dx_val)))
    horizontal_angle = min(angle, abs(180.0 - angle))
    vertical_angle = abs(90.0 - angle)
    if is_straight and horizontal_angle <= horizontal_angle_degrees:
        return "horizontal_straight"
    if is_straight and vertical_angle <= horizontal_angle_degrees:
        return "vertical_straight"
    return "curved_open"


def normalize_baseline_topology(
    points: Iterable[Iterable[float]],
    *,
    mirror_match_tolerance: float = 4.0,
    closed_path_tolerance: float = 12.0,
    min_mirror_pairs: int = 3,
    straightness_chord_ratio: float = 0.985,
    horizontal_angle_degrees: float = 12.0,
    reading_order: str = "left_to_right",
    circular_direction: str = "clockwise",
) -> BaselineTopology:
    original_points = as_float_points(points)
    split_index, mean_mirror_distance, max_mirror_distance = _detect_out_and_back_split(
        original_points,
        mirror_match_tolerance=mirror_match_tolerance,
        min_mirror_pairs=min_mirror_pairs,
    )
    if split_index is not None:
        normalized_points = original_points[: split_index + 1]
    else:
        normalized_points = list(original_points)

    is_closed = (
        len(normalized_points) >= 3
        and distance(normalized_points[0], normalized_points[-1]) <= closed_path_tolerance
    )
    cut_index = None
    orientation_action = "preserved"

    if is_closed:
        normalized_points, cut_index = _cut_closed_path_at_top(normalized_points)
        area = signed_area(normalized_points[:-1] if normalized_points and normalized_points[0] == normalized_points[-1] else normalized_points)
        wants_clockwise = circular_direction == "clockwise"
        is_clockwise_in_image_space = area > 0
        if wants_clockwise != is_clockwise_in_image_space and len(normalized_points) > 2:
            ring = normalized_points[:-1] if distance(normalized_points[0], normalized_points[-1]) <= 1e-6 else normalized_points
            reversed_ring = list(reversed(ring))
            cut_index = min(range(len(reversed_ring)), key=lambda idx: (reversed_ring[idx][1], reversed_ring[idx][0]))
            normalized_points = reversed_ring[cut_index:] + reversed_ring[:cut_index] + [reversed_ring[cut_index]]
            orientation_action = "reversed_to_clockwise" if wants_clockwise else "reversed_to_counterclockwise"
        elif cut_index is not None:
            orientation_action = "cut_at_top"
    elif reading_order == "left_to_right" and len(normalized_points) >= 2:
        dx_val = normalized_points[-1][0] - normalized_points[0][0]
        dy_val = normalized_points[-1][1] - normalized_points[0][1]
        if abs(dx_val) >= abs(dy_val) and dx_val < 0:
            normalized_points = list(reversed(normalized_points))
            orientation_action = "reversed_to_left_to_right"

    kind = _line_kind(
        normalized_points,
        is_closed,
        straightness_chord_ratio=straightness_chord_ratio,
        horizontal_angle_degrees=horizontal_angle_degrees,
    )
    return BaselineTopology(
        original_points=original_points,
        normalized_points=normalized_points,
        original_point_count=len(original_points),
        normalized_point_count=len(normalized_points),
        split_index=split_index,
        mirror_match_tolerance=float(mirror_match_tolerance),
        mean_mirror_distance=mean_mirror_distance,
        max_mirror_distance=max_mirror_distance,
        was_out_and_back=split_index is not None,
        is_closed=is_closed,
        closed_path_tolerance=float(closed_path_tolerance),
        cut_index=cut_index,
        baseline_length=polyline_length(normalized_points),
        line_kind=kind,
        orientation_action=orientation_action,
    )


def nearest_point_on_polyline(point: Iterable[float], points: list[list[float]]) -> NearestPoint:
    px_val, py_val = float(point[0]), float(point[1])
    if not points:
        return NearestPoint(float("inf"), (px_val, py_val), -1, 0.0, (1.0, 0.0), 0.0)
    if len(points) == 1:
        only = points[0]
        return NearestPoint(distance((px_val, py_val), only), (only[0], only[1]), 0, 0.0, (1.0, 0.0), 0.0)

    best: NearestPoint | None = None
    arc_before = 0.0
    for index in range(len(points) - 1):
        start = points[index]
        end = points[index + 1]
        vx_val = end[0] - start[0]
        vy_val = end[1] - start[1]
        segment_length_sq = vx_val * vx_val + vy_val * vy_val
        if segment_length_sq <= 1e-9:
            arc_before += math.sqrt(segment_length_sq)
            continue
        ratio = ((px_val - start[0]) * vx_val + (py_val - start[1]) * vy_val) / segment_length_sq
        ratio = max(0.0, min(1.0, ratio))
        nearest_x = start[0] + ratio * vx_val
        nearest_y = start[1] + ratio * vy_val
        segment_length = math.sqrt(segment_length_sq)
        tangent = _unit(vx_val, vy_val)
        item = NearestPoint(
            distance=math.hypot(px_val - nearest_x, py_val - nearest_y),
            point=(nearest_x, nearest_y),
            segment_index=index,
            ratio=ratio,
            tangent=tangent,
            arc_length=arc_before + ratio * segment_length,
        )
        if best is None or item.distance < best.distance:
            best = item
        arc_before += segment_length
    return best if best is not None else NearestPoint(float("inf"), (px_val, py_val), -1, 0.0, (1.0, 0.0), 0.0)


def sample_polyline(points: list[list[float]], sample_count: int) -> tuple[list[tuple[float, float]], list[tuple[float, float]], float]:
    total_length = polyline_length(points)
    if not points:
        return [], [], 0.0
    if len(points) == 1 or total_length <= 1e-6 or sample_count <= 1:
        return [(points[0][0], points[0][1])], [(1.0, 0.0)], total_length

    segment_lengths = [distance(points[index], points[index + 1]) for index in range(len(points) - 1)]
    centers: list[tuple[float, float]] = []
    tangents: list[tuple[float, float]] = []
    segment_index = 0
    arc_before = 0.0
    for sample_index in range(sample_count):
        target = min(total_length, sample_index * total_length / max(sample_count - 1, 1))
        while segment_index < len(segment_lengths) - 1 and arc_before + segment_lengths[segment_index] < target:
            arc_before += segment_lengths[segment_index]
            segment_index += 1
        segment_length = max(segment_lengths[segment_index], 1e-6)
        ratio = max(0.0, min(1.0, (target - arc_before) / segment_length))
        start = points[segment_index]
        end = points[segment_index + 1]
        center_x = start[0] + ratio * (end[0] - start[0])
        center_y = start[1] + ratio * (end[1] - start[1])
        centers.append((center_x, center_y))
        tangents.append(_unit(end[0] - start[0], end[1] - start[1]))
    return centers, tangents, total_length
