from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _edit_distance(source: str, target: str) -> int:
    if len(source) < len(target):
        return _edit_distance(target, source)
    if not target:
        return len(source)

    previous_row = list(range(len(target) + 1))
    for source_index, source_char in enumerate(source):
        current_row = [source_index + 1]
        for target_index, target_char in enumerate(target):
            current_row.append(
                min(
                    previous_row[target_index + 1] + 1,
                    current_row[target_index] + 1,
                    previous_row[target_index] + (source_char != target_char),
                )
            )
        previous_row = current_row
    return previous_row[-1]


def compute_text_edit_metrics(predicted_lines: dict[str, str], saved_lines: dict[str, str]) -> dict:
    predicted_lines = {str(key): str(value or "") for key, value in (predicted_lines or {}).items()}
    saved_lines = {str(key): str(value or "") for key, value in (saved_lines or {}).items()}
    ordered_line_ids = sorted(set(predicted_lines) | set(saved_lines), key=str)

    total_edit_distance = 0
    total_saved_characters = 0
    changed_line_count = 0
    line_cer_values = []
    per_line_diffs = []

    for line_id in ordered_line_ids:
        predicted_text = predicted_lines.get(line_id, "")
        saved_text = saved_lines.get(line_id, "")
        edit_distance = _edit_distance(predicted_text, saved_text)
        changed = predicted_text != saved_text
        saved_character_count = len(saved_text)
        line_cer = float(edit_distance) / float(max(saved_character_count, 1))
        changed_line_count += int(changed)
        total_edit_distance += edit_distance
        total_saved_characters += saved_character_count
        line_cer_values.append(line_cer)
        per_line_diffs.append(
            {
                "line_id": line_id,
                "predicted_text": predicted_text,
                "saved_text": saved_text,
                "saved_character_count": saved_character_count,
                "edit_distance": edit_distance,
                "line_cer": line_cer,
                "changed": changed,
            }
        )

    page_cer = (
        float(total_edit_distance) / float(total_saved_characters)
        if total_saved_characters
        else 0.0
    )
    return {
        "total_edit_distance": total_edit_distance,
        "changed_line_count": changed_line_count,
        "total_saved_characters": total_saved_characters,
        "normalized_edit_distance": page_cer,
        "page_cer": page_cer,
        "mean_line_cer": (
            float(sum(line_cer_values)) / float(len(line_cer_values))
            if line_cer_values
            else 0.0
        ),
        "per_line_diffs": per_line_diffs,
    }


def _coerce_int_list(values) -> list[int] | None:
    if values is None:
        return None
    if not isinstance(values, (list, tuple)):
        return None
    coerced = []
    for value in values:
        try:
            label = int(value)
        except (TypeError, ValueError):
            label = -1
        coerced.append(label if label >= 0 else -1)
    return coerced


def _components_from_graph(graph_payload: dict | None) -> list[list[int]]:
    graph_payload = graph_payload or {}
    nodes = graph_payload.get("nodes") or []
    node_count = len(nodes)
    if node_count <= 0:
        return []

    parent = list(range(node_count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for edge in graph_payload.get("edges") or []:
        try:
            source = int(edge.get("source"))
            target = int(edge.get("target"))
        except (AttributeError, TypeError, ValueError):
            continue
        if 0 <= source < node_count and 0 <= target < node_count:
            union(source, target)

    grouped = defaultdict(list)
    for node_index in range(node_count):
        grouped[find(node_index)].append(node_index)
    return sorted(grouped.values(), key=lambda component: component[0])


def _majority_label_for_component(labels: list[int], component: list[int], default_label: int) -> int:
    values = [
        labels[node_index]
        for node_index in component
        if 0 <= node_index < len(labels) and labels[node_index] >= 0
    ]
    if not values:
        return int(default_label)
    counts = Counter(values)
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _region_membership_signatures(
    labels_by_line: dict[str, int],
    *,
    ignore_singleton_labels: bool = False,
) -> dict[str, tuple]:
    grouped = defaultdict(list)
    for line_id, label in labels_by_line.items():
        grouped[int(label)].append(str(line_id))

    signatures = {}
    for label, members in grouped.items():
        ordered_members = tuple(sorted(members, key=lambda value: int(value)))
        if ignore_singleton_labels and len(ordered_members) == 1:
            signatures[ordered_members[0]] = ("line", ordered_members[0])
            continue
        signature = ("region", int(label), ordered_members)
        for line_id in ordered_members:
            signatures[line_id] = signature
    return signatures


def compute_text_region_edit_metrics(
    graph_payload: dict | None,
    textbox_labels=None,
    previous_textbox_labels=None,
) -> dict:
    components = _components_from_graph(graph_payload)
    current_labels = _coerce_int_list(textbox_labels)
    previous_labels = _coerce_int_list(previous_textbox_labels)

    if not components or current_labels is None:
        return {
            "text_line_count": len(components),
            "current_region_count": 0,
            "previous_region_count": 0,
            "text_region_annotations_changed": 0,
            "changed_text_lines": [],
        }

    current_by_line = {}
    previous_by_line = {}
    for line_index, component in enumerate(components):
        current_by_line[str(line_index)] = _majority_label_for_component(
            current_labels,
            component,
            default_label=line_index,
        )
        if previous_labels is not None and len(previous_labels) == len(current_labels):
            previous_by_line[str(line_index)] = _majority_label_for_component(
                previous_labels,
                component,
                default_label=line_index,
            )
        else:
            # The app's intended default is one text region per text line.
            previous_by_line[str(line_index)] = line_index

    ignore_singleton_labels = previous_labels is None
    current_signatures = _region_membership_signatures(
        current_by_line,
        ignore_singleton_labels=ignore_singleton_labels,
    )
    previous_signatures = _region_membership_signatures(
        previous_by_line,
        ignore_singleton_labels=ignore_singleton_labels,
    )

    changed_text_lines = []
    for line_id in sorted(current_by_line, key=lambda value: int(value)):
        previous_region_signature = previous_signatures.get(line_id)
        current_region_signature = current_signatures.get(line_id)
        if previous_region_signature != current_region_signature:
            changed_text_lines.append(
                {
                    "line_id": line_id,
                    "previous_region": previous_by_line.get(line_id),
                    "saved_region": current_by_line.get(line_id),
                }
            )

    return {
        "text_line_count": len(components),
        "current_region_count": len(set(current_by_line.values())),
        "previous_region_count": len(set(previous_by_line.values())),
        "text_region_annotations_changed": len(changed_text_lines),
        "changed_text_lines": changed_text_lines,
    }


def _annotation_list(payload) -> list[dict]:
    if payload is None:
        return []
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("line_annotations", "lineAnnotations"):
            value = payload.get(key)
            if isinstance(value, list):
                return [dict(item) for item in value if isinstance(item, dict)]
    return []


def _rounded_point(value):
    if not isinstance(value, (list, tuple)):
        return value
    rounded = []
    for item in value:
        try:
            rounded.append(round(float(item), 6))
        except (TypeError, ValueError):
            rounded.append(item)
    return rounded


def _coerce_node_indices(value) -> list[int]:
    if not isinstance(value, (list, tuple)):
        return []
    node_indices = []
    for item in value:
        try:
            node_indices.append(int(item))
        except (TypeError, ValueError):
            continue
    return sorted(node_indices)


def _canonical_reading_annotation(annotation: dict) -> dict:
    key = (
        annotation.get("resolved_line_numeric_id")
        if annotation.get("resolved_line_numeric_id") is not None
        else annotation.get("frontend_line_id", annotation.get("annotation_id"))
    )
    return {
        "key": str(key),
        "component_node_indices": _coerce_node_indices(annotation.get("component_node_indices", [])),
        "cut_start": _rounded_point(annotation.get("cut_start")),
        "cut_end": _rounded_point(annotation.get("cut_end")),
        "cut_midpoint": _rounded_point(annotation.get("cut_midpoint")),
        "reading_direction": _rounded_point(annotation.get("reading_direction")),
        "source": annotation.get("source"),
    }


def compute_reading_direction_edit_metrics(current_annotations=None, previous_annotations=None) -> dict:
    current = {
        item["key"]: item
        for item in (_canonical_reading_annotation(annotation) for annotation in _annotation_list(current_annotations))
        if item.get("key") not in {"", "None"}
    }
    previous = {
        item["key"]: item
        for item in (_canonical_reading_annotation(annotation) for annotation in _annotation_list(previous_annotations))
        if item.get("key") not in {"", "None"}
    }

    added = sorted(set(current) - set(previous))
    deleted = sorted(set(previous) - set(current))
    changed = sorted(key for key in set(current) & set(previous) if current[key] != previous[key])
    return {
        "reading_direction_annotation_count": len(current),
        "reading_direction_annotations_added": len(added),
        "reading_direction_annotations_changed": len(changed),
        "reading_direction_annotations_deleted": len(deleted),
        "changed_line_ids": sorted(set(added) | set(changed) | set(deleted), key=str),
    }


def compute_layout_edit_metrics(
    modifications: list[dict] | None,
    graph_payload: dict | None = None,
    textbox_labels=None,
    previous_textbox_labels=None,
    reading_direction_annotations=None,
    previous_reading_direction_annotations=None,
    include_annotation_deltas: bool = False,
) -> dict:
    graph_payload = graph_payload or {}
    final_nodes = len(graph_payload.get("nodes") or [])
    final_edges = len(graph_payload.get("edges") or [])
    metrics = {
        "original_nodes": final_nodes,
        "final_nodes": final_nodes,
        "original_edges": final_edges,
        "final_edges": final_edges,
        "nodes_added": 0,
        "nodes_deleted": 0,
        "edges_added": 0,
        "edges_deleted": 0,
        "reset_heuristic_count": 0,
        "reading_direction_modification_count": 0,
        "text_region_annotations_changed": 0,
        "reading_direction_annotations_added": 0,
        "reading_direction_annotations_changed": 0,
        "reading_direction_annotations_deleted": 0,
        "modification_count": 0,
        "total_layout_interventions": 0,
    }
    for modification in modifications or []:
        metrics["modification_count"] += 1
        mod_type = str(modification.get("type", ""))
        if mod_type == "node_add":
            metrics["nodes_added"] += 1
        elif mod_type == "node_delete":
            metrics["nodes_deleted"] += 1
        elif mod_type == "add":
            metrics["edges_added"] += 1
        elif mod_type == "delete":
            metrics["edges_deleted"] += 1
        elif mod_type == "reset_heuristic":
            metrics["reset_heuristic_count"] += 1
        elif mod_type == "reading_direction":
            metrics["reading_direction_modification_count"] += 1

    metrics["original_nodes"] = max(
        0,
        metrics["final_nodes"] - metrics["nodes_added"] + metrics["nodes_deleted"],
    )
    metrics["original_edges"] = max(
        0,
        metrics["final_edges"] - metrics["edges_added"] + metrics["edges_deleted"],
    )

    if include_annotation_deltas:
        region_metrics = compute_text_region_edit_metrics(
            graph_payload,
            textbox_labels=textbox_labels,
            previous_textbox_labels=previous_textbox_labels,
        )
        reading_metrics = compute_reading_direction_edit_metrics(
            reading_direction_annotations,
            previous_reading_direction_annotations,
        )
        metrics["text_region_metrics"] = region_metrics
        metrics["reading_direction_metrics"] = reading_metrics
        metrics["text_region_annotations_changed"] = int(
            region_metrics.get("text_region_annotations_changed", 0)
        )
        metrics["reading_direction_annotations_added"] = int(
            reading_metrics.get("reading_direction_annotations_added", 0)
        )
        metrics["reading_direction_annotations_changed"] = int(
            reading_metrics.get("reading_direction_annotations_changed", 0)
        )
        metrics["reading_direction_annotations_deleted"] = int(
            reading_metrics.get("reading_direction_annotations_deleted", 0)
        )

    metrics["total_layout_interventions"] = (
        metrics["nodes_added"]
        + metrics["nodes_deleted"]
        + metrics["edges_added"]
        + metrics["edges_deleted"]
        + metrics["reset_heuristic_count"]
        + metrics["text_region_annotations_changed"]
        + metrics["reading_direction_annotations_added"]
        + metrics["reading_direction_annotations_changed"]
        + metrics["reading_direction_annotations_deleted"]
    )
    return metrics


def update_summary_json(path: str | Path, key: str, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    current = {}
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
    current[str(key)] = payload
    path.write_text(json.dumps(current, indent=2, ensure_ascii=False), encoding="utf-8")


def _sum_numeric_fields(events: list[dict], section: str) -> dict:
    totals = defaultdict(float)
    for event in events:
        metrics = dict(event.get(section) or {})
        for key, value in metrics.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, int):
                totals[key] += int(value)
            elif isinstance(value, float):
                totals[key] += float(value)
    return {
        key: int(value) if float(value).is_integer() else value
        for key, value in sorted(totals.items())
    }


def _latest_supervised_text_event(events: list[dict]) -> dict | None:
    supervised_events = [
        event for event in events
        if event.get("save_scope") == "text_only" and bool(event.get("supervision_present"))
    ]
    if not supervised_events:
        return None
    return sorted(supervised_events, key=lambda event: int(event.get("revision_number", 0)))[-1]


def update_human_interventions_summary(path: str | Path, page_event: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    current = {}
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))

    page_id = str(page_event.get("page_id"))
    revision_key = f"r{int(page_event.get('revision_number', 0))}"
    pages = current.setdefault("pages", {})
    page_entry = pages.setdefault(page_id, {"revisions": {}})
    page_entry["revisions"][revision_key] = page_event

    all_page_events = []
    latest_text_effort_by_page = {}
    for current_page_id, current_page in pages.items():
        revision_events = list((current_page.get("revisions") or {}).values())
        revision_events.sort(key=lambda event: int(event.get("revision_number", 0)))
        current_page["latest_revision_number"] = (
            int(revision_events[-1].get("revision_number", 0)) if revision_events else None
        )
        current_page["layout_totals"] = _sum_numeric_fields(revision_events, "layout_metrics")
        current_page["read_totals"] = _sum_numeric_fields(revision_events, "text_metrics")
        latest_text_event = _latest_supervised_text_event(revision_events)
        current_page["latest_supervised_read_metrics"] = (
            dict(latest_text_event.get("text_metrics") or {}) if latest_text_event else None
        )
        if latest_text_event:
            latest_text_effort_by_page[str(current_page_id)] = {
                "page_id": str(current_page_id),
                "revision_number": int(latest_text_event.get("revision_number", 0)),
                "recorded_at": latest_text_event.get("recorded_at"),
                "page_cer": (latest_text_event.get("text_metrics") or {}).get("page_cer"),
                "mean_line_cer": (latest_text_event.get("text_metrics") or {}).get("mean_line_cer"),
                "prediction_source_engine": (latest_text_event.get("text_metrics") or {}).get("prediction_source_engine"),
                "prediction_source_checkpoint_id": (latest_text_event.get("text_metrics") or {}).get("prediction_source_checkpoint_id"),
            }
        all_page_events.extend(revision_events)

    latest_effort_values = list(latest_text_effort_by_page.values())
    page_cers = [item["page_cer"] for item in latest_effort_values if isinstance(item.get("page_cer"), (int, float))]
    line_cers = [item["mean_line_cer"] for item in latest_effort_values if isinstance(item.get("mean_line_cer"), (int, float))]

    current["schema_version"] = 1
    current["manuscript"] = page_event.get("manuscript")
    current["updated_at"] = utc_now_iso()
    current["manuscript_totals"] = {
        "page_count": len(pages),
        "revision_event_count": len(all_page_events),
        "layout_totals": _sum_numeric_fields(all_page_events, "layout_metrics"),
        "read_totals": _sum_numeric_fields(all_page_events, "text_metrics"),
        "latest_supervised_page_count": len(latest_effort_values),
        "mean_latest_page_cer": (
            float(sum(page_cers)) / float(len(page_cers)) if page_cers else None
        ),
        "mean_latest_line_cer": (
            float(sum(line_cers)) / float(len(line_cers)) if line_cers else None
        ),
    }
    current["read_mode_effort_curve"] = [
        latest_text_effort_by_page[key]
        for key in sorted(latest_text_effort_by_page)
    ]
    path.write_text(json.dumps(current, indent=2, ensure_ascii=False), encoding="utf-8")
