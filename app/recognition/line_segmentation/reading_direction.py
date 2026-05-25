from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


READING_DIRECTION_METADATA_SUFFIX = "_reading_direction_metadata.json"


def default_reading_direction_metadata_path(pagexml_path: str | Path) -> Path:
    pagexml_path = Path(pagexml_path)
    return pagexml_path.with_name(f"{pagexml_path.stem}{READING_DIRECTION_METADATA_SUFFIX}")


def _finite_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except Exception:
        return None
    return numeric if math.isfinite(numeric) else None


def _point_from_payload(value: Any) -> list[float] | None:
    if isinstance(value, dict):
        x_val = _finite_float(value.get("x"))
        y_val = _finite_float(value.get("y"))
    elif isinstance(value, (list, tuple)) and len(value) >= 2:
        x_val = _finite_float(value[0])
        y_val = _finite_float(value[1])
    else:
        return None
    if x_val is None or y_val is None:
        return None
    return [float(x_val), float(y_val)]


def _unit(vector: Iterable[float] | None) -> list[float] | None:
    if vector is None:
        return None
    values = list(vector)
    if len(values) < 2:
        return None
    x_val = _finite_float(values[0])
    y_val = _finite_float(values[1])
    if x_val is None or y_val is None:
        return None
    length = math.hypot(x_val, y_val)
    if length <= 1e-6:
        return None
    return [float(x_val / length), float(y_val / length)]


def reading_tangent_from_cross_cut(cut_start: Iterable[float], cut_end: Iterable[float]) -> list[float] | None:
    start = _point_from_payload(cut_start)
    end = _point_from_payload(cut_end)
    if start is None or end is None:
        return None
    dx_val = end[0] - start[0]
    dy_val = end[1] - start[1]
    return _unit([-dy_val, dx_val])


def _safe_node_indices(value: Any, *, max_node_count: int | None = None) -> list[int]:
    if not isinstance(value, (list, tuple, set)):
        return []
    indices = []
    seen = set()
    for item in value:
        try:
            index = int(item)
        except Exception:
            continue
        if index < 0 or (max_node_count is not None and index >= max_node_count) or index in seen:
            continue
        seen.add(index)
        indices.append(index)
    return sorted(indices)


def normalize_reading_direction_annotation(raw: Any, *, max_node_count: int | None = None) -> dict | None:
    if not isinstance(raw, dict):
        return None

    cut_start = _point_from_payload(raw.get("cut_start") or raw.get("start"))
    cut_end = _point_from_payload(raw.get("cut_end") or raw.get("end"))
    reading_direction = _unit(raw.get("reading_direction") or raw.get("readingTangent"))
    if reading_direction is None and cut_start is not None and cut_end is not None:
        reading_direction = reading_tangent_from_cross_cut(cut_start, cut_end)
    if reading_direction is None:
        return None

    cut_midpoint = _point_from_payload(raw.get("cut_midpoint") or raw.get("midpoint"))
    if cut_midpoint is None and cut_start is not None and cut_end is not None:
        cut_midpoint = [(cut_start[0] + cut_end[0]) / 2.0, (cut_start[1] + cut_end[1]) / 2.0]

    return {
        "annotation_id": str(raw.get("annotation_id") or raw.get("id") or ""),
        "frontend_line_id": str(raw.get("frontend_line_id") or raw.get("lineId") or ""),
        "component_node_indices": _safe_node_indices(
            raw.get("component_node_indices") or raw.get("componentNodeIndices"),
            max_node_count=max_node_count,
        ),
        "cut_start": cut_start,
        "cut_end": cut_end,
        "cut_midpoint": cut_midpoint,
        "reading_direction": reading_direction,
        "source": str(raw.get("source") or "user_cross_cut"),
        "updated_at": str(raw.get("updated_at") or raw.get("updatedAt") or ""),
    }


def _component_sets_from_labels(final_structural_labels: Iterable[int], num_nodes: int) -> dict[int, set[int]]:
    labels = np.asarray(list(final_structural_labels), dtype=int)
    components: dict[int, set[int]] = {}
    for node_index in range(min(num_nodes, len(labels))):
        components.setdefault(int(labels[node_index]), set()).add(node_index)
    return components


def resolve_reading_direction_annotations(
    annotations: Iterable[Any] | None,
    final_structural_labels: Iterable[int],
    num_nodes: int,
    *,
    min_overlap_ratio: float = 0.5,
) -> tuple[dict[int, dict], list[dict]]:
    components = _component_sets_from_labels(final_structural_labels, num_nodes)
    resolved: dict[int, dict] = {}
    stale: list[dict] = []

    for raw in annotations or []:
        annotation = normalize_reading_direction_annotation(raw, max_node_count=num_nodes)
        if annotation is None:
            stale.append({"reason": "malformed_annotation", "raw_annotation": raw})
            continue

        annotation_nodes = set(annotation.get("component_node_indices") or [])
        if not annotation_nodes:
            stale.append({**annotation, "status": "stale", "reason": "missing_component_nodes"})
            continue

        best_label = None
        best_overlap = 0
        best_component_size = 0
        for line_label, component_nodes in components.items():
            overlap = len(annotation_nodes & component_nodes)
            if overlap > best_overlap:
                best_label = int(line_label)
                best_overlap = overlap
                best_component_size = len(component_nodes)

        denominator = max(len(annotation_nodes), best_component_size, 1)
        overlap_ratio = best_overlap / denominator
        if best_label is None or overlap_ratio < min_overlap_ratio:
            stale.append(
                {
                    **annotation,
                    "status": "stale",
                    "reason": "component_overlap_below_threshold",
                    "best_overlap_ratio": float(overlap_ratio),
                }
            )
            continue

        resolved[best_label] = {
            **annotation,
            "status": "active",
            "resolved_line_numeric_id": int(best_label),
            "component_overlap_ratio": float(overlap_ratio),
        }

    return resolved, stale


def build_reading_direction_metadata_payload(
    *,
    page_id: str,
    annotations: Iterable[Any] | None,
    final_structural_labels: Iterable[int],
    num_nodes: int,
) -> dict:
    resolved, stale = resolve_reading_direction_annotations(
        annotations,
        final_structural_labels,
        num_nodes,
    )
    return {
        "schema_version": 1,
        "page_id": str(page_id),
        "annotation_model": "cross_cut_clockwise_90",
        "line_annotations": [resolved[key] for key in sorted(resolved)],
        "stale_annotations": stale,
    }


def load_reading_direction_metadata(path: str | Path | None) -> dict:
    if path is None:
        return {"line_annotations": [], "stale_annotations": []}
    path = Path(path)
    if not path.exists():
        return {"line_annotations": [], "stale_annotations": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"line_annotations": [], "stale_annotations": []}
    if not isinstance(payload, dict):
        return {"line_annotations": [], "stale_annotations": []}
    return payload


def load_reading_direction_annotations_by_line_id(path: str | Path | None) -> dict[int, dict]:
    payload = load_reading_direction_metadata(path)
    annotations = payload.get("line_annotations")
    if not isinstance(annotations, list):
        return {}
    result = {}
    for item in annotations:
        if not isinstance(item, dict):
            continue
        try:
            line_id = int(item.get("resolved_line_numeric_id"))
        except Exception:
            continue
        result[line_id] = dict(item)
    return result


def write_reading_direction_metadata(path: str | Path, payload: dict) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return destination


__all__ = [
    "READING_DIRECTION_METADATA_SUFFIX",
    "build_reading_direction_metadata_payload",
    "default_reading_direction_metadata_path",
    "load_reading_direction_annotations_by_line_id",
    "load_reading_direction_metadata",
    "normalize_reading_direction_annotation",
    "reading_tangent_from_cross_cut",
    "resolve_reading_direction_annotations",
    "write_reading_direction_metadata",
]
