from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping


LAYOUT_EFFORT_KEYS = ("a", "d", "e", "q")
SINGLE_EDIT_TIME_SECONDS = 2.0


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _coerce_bool(value) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on", "enabled"}:
        return True
    if normalized in {"0", "false", "no", "off", "disabled", ""}:
        return False
    return None


def layout_effort_logging_enabled(config: Mapping[str, object] | None = None) -> bool:
    config = dict(config or {})
    for key in ("layout_effort_logging_enabled", "LAYOUT_EFFORT_LOGGING_ENABLED"):
        configured = _coerce_bool(config.get(key))
        if configured is not None:
            return configured
    configured = _coerce_bool(os.getenv("LAYOUT_EFFORT_LOGGING_ENABLED"))
    return True if configured is None else configured


def _safe_slug(value: str) -> str:
    slug = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in str(value))
    return slug or "manuscript"


def layout_effort_log_path(manuscript_root: str | Path) -> Path:
    manuscript_root = Path(manuscript_root)
    explicit_dir = os.getenv("LAYOUT_EFFORT_LOG_DIR")
    if explicit_dir:
        return Path(explicit_dir) / f"{_safe_slug(manuscript_root.name)}_layout_effort.json"
    return manuscript_root / "layout_analysis_output" / "layout_effort.json"


def _finite_float(value, default: float | None = None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return number


def _nonnegative_float(value, default: float = 0.0) -> float:
    number = _finite_float(value)
    if number is None:
        return default
    return max(0.0, number)


def _nonnegative_int(value, default: int = 0) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


def _keyed_int_counts(payload: Mapping[str, object] | None) -> dict[str, int]:
    payload = dict(payload or {})
    return {key: _nonnegative_int(payload.get(key), 0) for key in LAYOUT_EFFORT_KEYS}


def _keyed_second_counts(payload: Mapping[str, object] | None) -> dict[str, float]:
    payload = dict(payload or {})
    values = {}
    for key in LAYOUT_EFFORT_KEYS:
        raw_value = payload.get(key)
        if raw_value is None and f"{key}_ms" in payload:
            raw_value = _nonnegative_float(payload.get(f"{key}_ms")) / 1000.0
        values[key] = _nonnegative_float(raw_value, 0.0)
    values["total"] = sum(values[key] for key in LAYOUT_EFFORT_KEYS)
    return values


def _keyed_millisecond_counts(payload: Mapping[str, object] | None) -> dict[str, float]:
    payload = dict(payload or {})
    return {
        key: _nonnegative_float(payload.get(key), 0.0) / 1000.0
        for key in LAYOUT_EFFORT_KEYS
    }


def _normalize_frontend_effort(layout_effort: Mapping[str, object] | None) -> dict:
    if not isinstance(layout_effort, Mapping):
        return {
            "measurement_status": "missing_frontend_effort",
            "edit_count": 0,
            "active_edit_time_seconds": 0.0,
            "raw_first_to_last_seconds": 0.0,
            "first_edit_at": None,
            "last_edit_at": None,
            "left_click_add_node_edits": 0,
            "key_hold_edit_count": 0,
            "key_hold_edit_counts": {key: 0 for key in LAYOUT_EFFORT_KEYS},
            "key_hold_duration_seconds": {**{key: 0.0 for key in LAYOUT_EFFORT_KEYS}, "total": 0.0},
            "single_edit_time_floor_seconds": SINGLE_EDIT_TIME_SECONDS,
        }

    payload = dict(layout_effort)
    edit_count = _nonnegative_int(payload.get("edit_count"), 0)
    active_seconds = _finite_float(payload.get("active_edit_time_seconds"))
    if active_seconds is None:
        active_seconds = _finite_float(payload.get("active_edit_time_ms"))
        active_seconds = active_seconds / 1000.0 if active_seconds is not None else None
    raw_seconds = _finite_float(payload.get("raw_first_to_last_seconds"))
    if raw_seconds is None:
        raw_seconds = _finite_float(payload.get("raw_first_to_last_ms"))
        raw_seconds = raw_seconds / 1000.0 if raw_seconds is not None else None
    if raw_seconds is None:
        raw_seconds = active_seconds

    if edit_count <= 0:
        normalized_active_seconds = 0.0
        status = "measured_no_layout_edits"
    elif edit_count == 1:
        normalized_active_seconds = SINGLE_EDIT_TIME_SECONDS
        status = "measured"
    else:
        normalized_active_seconds = _nonnegative_float(active_seconds, 0.0)
        status = "measured"

    key_hold_edit_counts = _keyed_int_counts(payload.get("key_hold_edit_counts"))
    key_hold_duration_seconds = _keyed_second_counts(payload.get("key_hold_duration_seconds"))
    if key_hold_duration_seconds["total"] == 0.0:
        key_hold_duration_seconds = _keyed_millisecond_counts(payload.get("key_hold_duration_ms"))
        key_hold_duration_seconds["total"] = sum(
            key_hold_duration_seconds[key] for key in LAYOUT_EFFORT_KEYS
        )

    return {
        "measurement_status": status,
        "edit_count": edit_count,
        "active_edit_time_seconds": normalized_active_seconds,
        "raw_first_to_last_seconds": _nonnegative_float(raw_seconds, 0.0),
        "first_edit_at": payload.get("first_edit_at"),
        "last_edit_at": payload.get("last_edit_at"),
        "captured_at": payload.get("captured_at"),
        "left_click_add_node_edits": _nonnegative_int(payload.get("left_click_add_node_edits"), 0),
        "key_hold_edit_count": _nonnegative_int(
            payload.get("key_hold_edit_count"),
            sum(key_hold_edit_counts.values()),
        ),
        "key_hold_edit_counts": key_hold_edit_counts,
        "key_hold_duration_seconds": key_hold_duration_seconds,
        "single_edit_time_floor_seconds": SINGLE_EDIT_TIME_SECONDS,
    }


def _normalize_layout_counts(layout_metrics: Mapping[str, object] | None) -> dict:
    metrics = dict(layout_metrics or {})
    text_region_metrics = dict(metrics.get("text_region_metrics") or {})
    reading_metrics = dict(metrics.get("reading_direction_metrics") or {})
    reading_direction_annotation_edits = (
        _nonnegative_int(reading_metrics.get("reading_direction_annotations_added"), 0)
        + _nonnegative_int(reading_metrics.get("reading_direction_annotations_changed"), 0)
        + _nonnegative_int(reading_metrics.get("reading_direction_annotations_deleted"), 0)
    )
    return {
        "original_nodes": _nonnegative_int(metrics.get("original_nodes"), 0),
        "final_nodes": _nonnegative_int(metrics.get("final_nodes"), 0),
        "nodes_added": _nonnegative_int(metrics.get("nodes_added"), 0),
        "nodes_deleted": _nonnegative_int(metrics.get("nodes_deleted"), 0),
        "original_edges": _nonnegative_int(metrics.get("original_edges"), 0),
        "final_edges": _nonnegative_int(metrics.get("final_edges"), 0),
        "edges_added": _nonnegative_int(metrics.get("edges_added"), 0),
        "edges_deleted": _nonnegative_int(metrics.get("edges_deleted"), 0),
        "text_lines_region_labeled": _nonnegative_int(
            metrics.get("text_region_annotations_changed"),
            _nonnegative_int(text_region_metrics.get("text_region_annotations_changed"), 0),
        ),
        "manual_text_line_orientations_labeled": _nonnegative_int(
            reading_metrics.get("reading_direction_annotation_count"),
            0,
        ),
        "manual_text_line_orientation_annotation_edits": reading_direction_annotation_edits,
        "text_region_metrics": text_region_metrics,
        "reading_direction_metrics": reading_metrics,
    }


def _normalize_processing_metrics(processing_metrics: Mapping[str, object] | None) -> dict:
    metrics = dict(processing_metrics or {})
    duration = _finite_float(metrics.get("duration_seconds"))
    return {
        "measurement_status": "measured" if duration is not None else "missing",
        "duration_seconds": _nonnegative_float(duration, 0.0),
        "started_at": metrics.get("started_at"),
        "finished_at": metrics.get("finished_at"),
        "status": metrics.get("status", "unknown"),
        "line_count": _nonnegative_int(metrics.get("line_count"), 0),
        "layout_artifacts_regenerated": bool(metrics.get("layout_artifacts_regenerated", True)),
    }


def _sum_numeric_field(events: list[dict], section: str, field: str) -> float:
    total = 0.0
    for event in events:
        value = ((event.get(section) or {}).get(field))
        if isinstance(value, bool):
            continue
        number = _finite_float(value)
        if number is not None:
            total += number
    return total


def _sum_count_field(events: list[dict], field: str) -> int:
    return int(sum(_nonnegative_int((event.get("layout_counts") or {}).get(field), 0) for event in events))


def _sum_key_counts(events: list[dict], section: str, field: str) -> dict:
    totals = defaultdict(float)
    for event in events:
        values = (((event.get(section) or {}).get(field)) or {})
        for key, value in values.items():
            number = _finite_float(value)
            if number is not None:
                totals[str(key)] += number
    result = {}
    for key, value in sorted(totals.items()):
        result[key] = int(value) if float(value).is_integer() else value
    return result


def _latest_processing_event(events: list[dict]) -> dict | None:
    for event in reversed(events):
        processing = dict(event.get("processing") or {})
        if processing.get("measurement_status") == "measured":
            return processing
    return None


def _recompute_page_summary(page_entry: dict) -> None:
    events = list(page_entry.get("layout_revisions") or [])
    latest_event = events[-1] if events else None
    first_counts = dict((events[0].get("layout_counts") or {}) if events else {})
    latest_counts = dict((latest_event.get("layout_counts") or {}) if latest_event else {})
    latest_processing = _latest_processing_event(events)

    page_entry["revision_count"] = len(events)
    page_entry["latest_recorded_at"] = latest_event.get("recorded_at") if latest_event else None
    page_entry["original_nodes"] = _nonnegative_int(first_counts.get("original_nodes"), 0)
    page_entry["final_nodes"] = _nonnegative_int(latest_counts.get("final_nodes"), 0)
    page_entry["original_edges"] = _nonnegative_int(first_counts.get("original_edges"), 0)
    page_entry["final_edges"] = _nonnegative_int(latest_counts.get("final_edges"), 0)
    page_entry["latest_processing"] = latest_processing
    page_entry["totals"] = {
        "edit_count": int(_sum_numeric_field(events, "edit_timing", "edit_count")),
        "active_edit_time_seconds": _sum_numeric_field(events, "edit_timing", "active_edit_time_seconds"),
        "left_click_add_node_edits": int(
            _sum_numeric_field(events, "edit_timing", "left_click_add_node_edits")
        ),
        "key_hold_edit_count": int(_sum_numeric_field(events, "edit_timing", "key_hold_edit_count")),
        "key_hold_edit_counts": _sum_key_counts(events, "edit_timing", "key_hold_edit_counts"),
        "key_hold_duration_seconds": _sum_key_counts(events, "edit_timing", "key_hold_duration_seconds"),
        "nodes_added": _sum_count_field(events, "nodes_added"),
        "nodes_deleted": _sum_count_field(events, "nodes_deleted"),
        "edges_added": _sum_count_field(events, "edges_added"),
        "edges_deleted": _sum_count_field(events, "edges_deleted"),
        "text_lines_region_labeled": _sum_count_field(events, "text_lines_region_labeled"),
        "manual_text_line_orientation_annotation_edits": _sum_count_field(
            events,
            "manual_text_line_orientation_annotation_edits",
        ),
        "manual_text_line_orientations_labeled_current": _nonnegative_int(
            latest_counts.get("manual_text_line_orientations_labeled"),
            0,
        ),
        "latest_processing_time_seconds": (
            _nonnegative_float(latest_processing.get("duration_seconds"), 0.0)
            if latest_processing
            else 0.0
        ),
    }


def _recompute_manuscript_summary(payload: dict) -> None:
    pages = dict(payload.get("pages") or {})
    page_entries = list(pages.values())
    totals = defaultdict(float)
    key_hold_edit_counts = defaultdict(float)
    key_hold_duration_seconds = defaultdict(float)
    processing_time = 0.0
    revision_count = 0
    pages_with_measured_effort = 0

    for page_entry in page_entries:
        revision_count += _nonnegative_int(page_entry.get("revision_count"), 0)
        page_totals = dict(page_entry.get("totals") or {})
        for field in (
            "edit_count",
            "active_edit_time_seconds",
            "left_click_add_node_edits",
            "key_hold_edit_count",
            "nodes_added",
            "nodes_deleted",
            "edges_added",
            "edges_deleted",
            "text_lines_region_labeled",
            "manual_text_line_orientation_annotation_edits",
            "manual_text_line_orientations_labeled_current",
        ):
            totals[field] += _nonnegative_float(page_totals.get(field), 0.0)
        for key, value in dict(page_totals.get("key_hold_edit_counts") or {}).items():
            key_hold_edit_counts[str(key)] += _nonnegative_float(value, 0.0)
        for key, value in dict(page_totals.get("key_hold_duration_seconds") or {}).items():
            key_hold_duration_seconds[str(key)] += _nonnegative_float(value, 0.0)
        processing_time += _nonnegative_float(page_totals.get("latest_processing_time_seconds"), 0.0)
        if _nonnegative_float(page_totals.get("active_edit_time_seconds"), 0.0) > 0.0:
            pages_with_measured_effort += 1

    payload["manuscript_summary"] = {
        "page_count": len(page_entries),
        "layout_revision_count": revision_count,
        "pages_with_measured_edit_time": pages_with_measured_effort,
        "edit_count": int(totals["edit_count"]),
        "active_edit_time_seconds": totals["active_edit_time_seconds"],
        "left_click_add_node_edits": int(totals["left_click_add_node_edits"]),
        "key_hold_edit_count": int(totals["key_hold_edit_count"]),
        "key_hold_edit_counts": {
            key: int(value) if float(value).is_integer() else value
            for key, value in sorted(key_hold_edit_counts.items())
        },
        "key_hold_duration_seconds": {
            key: int(value) if float(value).is_integer() else value
            for key, value in sorted(key_hold_duration_seconds.items())
        },
        "nodes_added": int(totals["nodes_added"]),
        "nodes_deleted": int(totals["nodes_deleted"]),
        "original_nodes": sum(_nonnegative_int(page.get("original_nodes"), 0) for page in page_entries),
        "final_nodes": sum(_nonnegative_int(page.get("final_nodes"), 0) for page in page_entries),
        "edges_added": int(totals["edges_added"]),
        "edges_deleted": int(totals["edges_deleted"]),
        "original_edges": sum(_nonnegative_int(page.get("original_edges"), 0) for page in page_entries),
        "final_edges": sum(_nonnegative_int(page.get("final_edges"), 0) for page in page_entries),
        "text_lines_region_labeled": int(totals["text_lines_region_labeled"]),
        "manual_text_line_orientation_annotation_edits": int(
            totals["manual_text_line_orientation_annotation_edits"]
        ),
        "manual_text_line_orientations_labeled_current": int(
            totals["manual_text_line_orientations_labeled_current"]
        ),
        "latest_layout_processing_time_seconds": processing_time,
    }


def _read_existing_payload(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_payload(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(path)


def record_layout_effort_save(
    *,
    manuscript_root: str | Path,
    page_id: str,
    save_scope: str,
    save_intent: str,
    layout_metrics: Mapping[str, object] | None,
    layout_effort: Mapping[str, object] | None,
    processing_metrics: Mapping[str, object] | None,
    active_learning_revision: Mapping[str, object] | None = None,
    config: Mapping[str, object] | None = None,
) -> Path | None:
    if not layout_effort_logging_enabled(config):
        return None

    manuscript_root = Path(manuscript_root)
    path = layout_effort_log_path(manuscript_root)
    current = _read_existing_payload(path)
    pages = current.setdefault("pages", {})
    page_entry = pages.setdefault(str(page_id), {"page_id": str(page_id), "layout_revisions": []})
    revisions = page_entry.setdefault("layout_revisions", [])
    revision = dict(active_learning_revision or {})
    event = {
        "schema_version": 1,
        "event_type": "layout_effort_revision",
        "recorded_at": utc_now_iso(),
        "page_id": str(page_id),
        "layout_revision_number": len(revisions) + 1,
        "ocr_revision_number": revision.get("revision_number"),
        "ocr_revision_is_duplicate": bool(revision.get("is_duplicate", False)),
        "save_scope": str(save_scope or "layout"),
        "save_intent": str(save_intent or "commit"),
        "edit_timing": _normalize_frontend_effort(layout_effort),
        "layout_counts": _normalize_layout_counts(layout_metrics),
        "processing": _normalize_processing_metrics(processing_metrics),
    }
    revisions.append(event)

    _recompute_page_summary(page_entry)
    for item in pages.values():
        _recompute_page_summary(item)

    current["schema_version"] = 1
    current["event_type"] = "layout_effort_log"
    current["manuscript"] = manuscript_root.name
    current["updated_at"] = utc_now_iso()
    current["logging_enabled"] = True
    _recompute_manuscript_summary(current)
    _write_payload(path, current)
    return path
