from __future__ import annotations

import json
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
    per_line_diffs = []

    for line_id in ordered_line_ids:
        predicted_text = predicted_lines.get(line_id, "")
        saved_text = saved_lines.get(line_id, "")
        edit_distance = _edit_distance(predicted_text, saved_text)
        changed = predicted_text != saved_text
        changed_line_count += int(changed)
        total_edit_distance += edit_distance
        total_saved_characters += len(saved_text)
        per_line_diffs.append(
            {
                "line_id": line_id,
                "predicted_text": predicted_text,
                "saved_text": saved_text,
                "edit_distance": edit_distance,
                "changed": changed,
            }
        )

    return {
        "total_edit_distance": total_edit_distance,
        "changed_line_count": changed_line_count,
        "total_saved_characters": total_saved_characters,
        "normalized_edit_distance": (
            float(total_edit_distance) / float(total_saved_characters)
            if total_saved_characters
            else 0.0
        ),
        "per_line_diffs": per_line_diffs,
    }


def compute_layout_edit_metrics(modifications: list[dict] | None) -> dict:
    metrics = {
        "nodes_added": 0,
        "nodes_deleted": 0,
        "edges_added": 0,
        "edges_deleted": 0,
        "reset_heuristic_count": 0,
        "modification_count": 0,
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
    return metrics


def update_summary_json(path: str | Path, key: str, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    current = {}
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
    current[str(key)] = payload
    path.write_text(json.dumps(current, indent=2, ensure_ascii=False), encoding="utf-8")
