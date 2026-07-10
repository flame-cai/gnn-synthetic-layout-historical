from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


METHOD_ORDER = (
    "vlm_e2e",
    "annotation_tool_e2e",
    "gemini_gt_layout",
    "annotation_tool_gt_layout",
    "annotation_tool_gt_layout_ft_1",
    "annotation_tool_gt_layout_ft_2",
    "annotation_tool_gt_layout_ft_3",
)

METHOD_LABELS = {
    "vlm_e2e": "Gemini e2e",
    "annotation_tool_e2e": "Annotation tool e2e",
    "gemini_gt_layout": "Gemini human-corrected GT layout",
    "annotation_tool_gt_layout": "Annotation tool human-corrected GT layout",
    "annotation_tool_gt_layout_ft_1": "Annotation tool human-corrected GT layout + 1 page FT",
    "annotation_tool_gt_layout_ft_2": "Annotation tool human-corrected GT layout + 2 page FT",
    "annotation_tool_gt_layout_ft_3": "Annotation tool human-corrected GT layout + 3 page FT",
}

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_INPUT_MANUSCRIPTS = REPO_ROOT / "app" / "input_manuscripts"


@dataclass(frozen=True)
class ReportArtifacts:
    report_dir: Path
    markdown_path: Path
    summary_csv_path: Path
    summary_json_path: Path
    per_page_csv_path: Path
    layout_effort_impact_csv_path: Path
    layout_effort_impact_json_path: Path
    gemini_usage_csv_path: Path
    gemini_usage_json_path: Path
    figure_paths: tuple[Path, ...]
    manifest_path: Path


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(path: Path, rows: Iterable[dict], *, fieldnames: list[str] | None = None) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
    if not fieldnames:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _method_sort_key(method_id: str) -> tuple[int, str]:
    try:
        return (METHOD_ORDER.index(method_id), method_id)
    except ValueError:
        return (len(METHOD_ORDER), method_id)


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _format_float(value: Any, digits: int = 4) -> str:
    number = _safe_float(value)
    if number is None:
        return ""
    return f"{number:.{digits}f}"


def _format_cost(value: Any) -> str:
    number = _safe_float(value)
    if number is None:
        return ""
    return f"{number:.6f}"


def _markdown_table(rows: list[dict], columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for _, label in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        values = []
        for key, _ in columns:
            value = row.get(key, "")
            values.append(str(value).replace("|", "\\|"))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *body])


def _metric_payload_paths(output_root: Path) -> list[Path]:
    paths = list((output_root / "metrics").glob("*/metrics.json"))
    paths.extend(output_root.glob("*/metrics.json"))
    unique = sorted({path.resolve() for path in paths if path.is_file()})
    return [Path(path) for path in unique]


def _payload_method_id(payload: dict, fallback: str) -> str:
    method = payload.get("method")
    if isinstance(method, dict) and method.get("method_id"):
        return str(method["method_id"])
    return fallback


def _safe_optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _candidate_layout_effort_paths(payload: dict) -> list[Path]:
    paths: list[Path] = []
    for key in ("manuscript_root", "input_manuscript_root"):
        raw_root = payload.get(key)
        if raw_root:
            paths.append(Path(raw_root) / "layout_analysis_output" / "layout_effort.json")
    manuscript_id = payload.get("manuscript_id")
    if manuscript_id:
        paths.append(APP_INPUT_MANUSCRIPTS / str(manuscript_id) / "layout_analysis_output" / "layout_effort.json")
    return paths


def _layout_effort_from_path(effort_path: Path) -> dict[str, dict]:
    if not effort_path.exists():
        return {}
    try:
        payload = json.loads(effort_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    pages = payload.get("pages") if isinstance(payload, dict) else None
    if not isinstance(pages, dict):
        return {}

    effort_by_page: dict[str, dict] = {}
    for page_key, page_payload in pages.items():
        if not isinstance(page_payload, dict):
            continue
        page_id = str(page_payload.get("page_id") or page_key)
        totals = page_payload.get("totals")
        if not isinstance(totals, dict):
            totals = page_payload
        revisions = page_payload.get("layout_revisions")
        revision_count = _safe_optional_int(page_payload.get("revision_count"))
        if revision_count is None and isinstance(revisions, list):
            revision_count = len(revisions)
        effort_by_page[page_id] = {
            "layout_effort_available": True,
            "layout_effort_edit_count": _safe_optional_int(totals.get("edit_count")),
            "layout_effort_active_edit_time_seconds": _safe_float(totals.get("active_edit_time_seconds")),
            "layout_effort_revision_count": revision_count,
            "layout_effort_source_path": str(effort_path),
        }
    return effort_by_page


def _layout_effort_from_payload(payload: dict) -> dict[str, dict]:
    for effort_path in _candidate_layout_effort_paths(payload):
        effort_by_page = _layout_effort_from_path(effort_path)
        if effort_by_page:
            return effort_by_page
    return {}


def _layout_effort_truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _default_layout_effort_fields() -> dict:
    return {
        "layout_effort_available": False,
        "layout_effort_edit_count": None,
        "layout_effort_active_edit_time_seconds": None,
        "layout_effort_revision_count": None,
        "layout_effort_source_path": "",
    }


def _apply_layout_effort_fallback(row: dict, effort_by_page: dict[str, dict]) -> None:
    has_effort_columns = any(key.startswith("layout_effort_") for key in row)
    if not has_effort_columns:
        row.update(_default_layout_effort_fields())
    if not _layout_effort_truthy(row.get("layout_effort_available")):
        row.update(effort_by_page.get(str(row.get("page_id", ""))) or {})


def _human_effort_label(method_id: str, method: dict) -> str:
    if method_id == "vlm_e2e":
        return "none"
    if method_id == "gemini_gt_layout":
        return "human layout correction"
    if method_id == "annotation_tool_e2e":
        return "none"
    if method.get("uses_finetuning"):
        return f"human layout correction + {int(method.get('finetune_page_count') or 0)} read-mode page(s)"
    if method.get("uses_gt_layout"):
        return "human layout correction"
    return "none"


def _engine_label(method_id: str, method: dict) -> str:
    if method.get("uses_gemini") or method_id in {"vlm_e2e", "gemini_gt_layout"}:
        return "Gemini"
    return "Annotation tool"


def _layout_condition_label(method: dict) -> str:
    return "human_corrected_gt_layout" if method.get("uses_gt_layout") else "predicted_layout"


def _layout_metric_interpretation(method: dict) -> str:
    if method.get("uses_gt_layout"):
        return "Human-corrected GT-layout condition; layout metrics are not layout-detector performance."
    return "Predicted-layout condition; layout metrics reflect the method's geometry output."


def _summary_row_from_payload(payload: dict, metrics_path: Path) -> dict:
    method = dict(payload.get("method") or {})
    method_id = _payload_method_id(payload, metrics_path.parent.name)
    aggregate = dict(payload.get("aggregate") or {})
    ocr_recipe = dict(payload.get("ocr_active_learning_recipe") or {})
    return {
        "method_id": method_id,
        "display_name": METHOD_LABELS.get(method_id) or method.get("display_name") or method_id,
        "engine": _engine_label(method_id, method),
        "human_effort": _human_effort_label(method_id, method),
        "human_layout": bool(method.get("uses_gt_layout")),
        "layout_condition": _layout_condition_label(method),
        "layout_metric_interpretation": _layout_metric_interpretation(method),
        "finetune_pages": int(method.get("finetune_page_count") or 0),
        "ocr_recipe_source": ocr_recipe.get("source", ""),
        "sibling_checkpoint_strategy": ocr_recipe.get("sibling_checkpoint_strategy", ""),
        "page_count": int(aggregate.get("page_count") or 0),
        "valid_output_rate": _safe_float(aggregate.get("valid_output_rate")),
        "object_g_f1_50": _safe_float(aggregate.get("object_g_f1_50")),
        "object_g_f1_75": _safe_float(aggregate.get("object_g_f1_75")),
        "pixel_f1": _safe_float(aggregate.get("pixel_f1")),
        "mean_page_cer": _safe_float(aggregate.get("mean_page_cer")),
        "median_page_cer": _safe_float(aggregate.get("median_page_cer")),
        "micro_page_cer": _safe_float(aggregate.get("micro_page_cer")),
        "mean_textedit": _safe_float(aggregate.get("mean_textedit")),
        "median_textedit": _safe_float(aggregate.get("median_textedit")),
        "micro_textedit": _safe_float(aggregate.get("micro_textedit")),
        "metrics_path": str(metrics_path),
    }


def _metadata_to_dict(value: Any) -> dict:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "to_json_dict"):
        try:
            converted = value.to_json_dict()
            return dict(converted) if isinstance(converted, dict) else {}
        except Exception:
            return {}
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return dict(parsed) if isinstance(parsed, dict) else {}
    return {}


def usage_metadata_to_dict(value: Any) -> dict:
    return _metadata_to_dict(value)


def summarize_usage_metadata(records: Iterable[dict]) -> dict:
    prompt_tokens = 0
    candidate_tokens = 0
    total_tokens = 0
    billable_characters = 0
    for record in records:
        metadata = _metadata_to_dict(record.get("usage_metadata"))
        prompt_tokens += _safe_int(metadata.get("prompt_token_count"))
        candidate_tokens += _safe_int(metadata.get("candidates_token_count"))
        total_tokens += _safe_int(metadata.get("total_token_count"))
        billable_characters += _safe_int(metadata.get("total_billable_characters"))
    return {
        "prompt_token_count": prompt_tokens,
        "candidates_token_count": candidate_tokens,
        "total_token_count": total_tokens,
        "total_billable_characters": billable_characters,
    }


def _usage_records_from_payload(payload: dict) -> list[dict]:
    if isinstance(payload.get("usage_records"), list):
        return [item for item in payload["usage_records"] if isinstance(item, dict)]
    if isinstance(payload.get("samples"), list):
        return [item for item in payload["samples"] if isinstance(item, dict)]
    metadata = _metadata_to_dict(payload.get("usage_metadata"))
    return [{"usage_metadata": metadata}] if metadata else []


def _has_usage_counts(usage_summary: dict) -> bool:
    return any(
        _safe_int(usage_summary.get(key)) > 0
        for key in (
            "prompt_token_count",
            "candidates_token_count",
            "total_token_count",
            "total_billable_characters",
        )
    )


def _estimate_gemini_cost_usd(
    *,
    prompt_tokens: int,
    candidate_tokens: int,
    input_usd_per_1m_tokens: float | None,
    output_usd_per_1m_tokens: float | None,
) -> float | None:
    if input_usd_per_1m_tokens is None or output_usd_per_1m_tokens is None:
        return None
    return (
        (prompt_tokens / 1_000_000.0) * input_usd_per_1m_tokens
        + (candidate_tokens / 1_000_000.0) * output_usd_per_1m_tokens
    )


def _env_float(name: str) -> float | None:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    return _safe_float(raw)


def collect_gemini_usage(
    output_root: str | Path,
    *,
    input_usd_per_1m_tokens: float | None = None,
    output_usd_per_1m_tokens: float | None = None,
) -> tuple[list[dict], dict[str, dict]]:
    root = Path(output_root)
    input_rate = input_usd_per_1m_tokens
    output_rate = output_usd_per_1m_tokens
    if input_rate is None:
        input_rate = _env_float("GEMINI_INPUT_USD_PER_1M_TOKENS")
    if output_rate is None:
        output_rate = _env_float("GEMINI_OUTPUT_USD_PER_1M_TOKENS")

    rows: list[dict] = []
    for usage_path in sorted(root.glob("runs/*/*/gemini_usage/*.json")):
        try:
            relative_parts = usage_path.relative_to(root).parts
        except ValueError:
            relative_parts = usage_path.parts
        if len(relative_parts) < 5:
            continue
        method_id = relative_parts[1]
        fold_id = relative_parts[2]
        page_id = usage_path.stem
        payload = _read_json(usage_path)
        usage_records = _usage_records_from_payload(payload)
        usage_summary = summarize_usage_metadata(usage_records)
        status = payload.get("status", "success" if usage_records else "unknown")
        request_count = _safe_int(payload.get("request_count"))
        if request_count <= 0:
            request_count = len(usage_records)
        if request_count <= 0 and status not in {"unknown", "not_attempted"}:
            request_count = 1
        attempt_count = _safe_int(payload.get("attempt_count"))
        if attempt_count <= 0:
            attempt_count = request_count
        retry_count = _safe_int(payload.get("retry_count"))
        if "retry_count" not in payload:
            retry_count = max(0, attempt_count - 1)
        usage_metadata_available = _has_usage_counts(usage_summary)
        estimated_cost = (
            _estimate_gemini_cost_usd(
                prompt_tokens=usage_summary["prompt_token_count"],
                candidate_tokens=usage_summary["candidates_token_count"],
                input_usd_per_1m_tokens=input_rate,
                output_usd_per_1m_tokens=output_rate,
            )
            if usage_metadata_available
            else None
        )
        rows.append(
            {
                "method_id": method_id,
                "fold_id": fold_id,
                "page_id": page_id,
                "status": status,
                "elapsed_seconds": _safe_float(payload.get("elapsed_seconds")),
                "attempt_count": attempt_count,
                "retry_count": retry_count,
                "max_retries": _safe_int(payload.get("max_retries")),
                "request_count": request_count,
                "usage_metadata_available": usage_metadata_available,
                "prompt_token_count": usage_summary["prompt_token_count"],
                "candidates_token_count": usage_summary["candidates_token_count"],
                "total_token_count": usage_summary["total_token_count"],
                "total_billable_characters": usage_summary["total_billable_characters"],
                "estimated_cost_usd": estimated_cost,
                "usage_path": str(usage_path),
            }
        )

    summaries: dict[str, dict] = {}
    for row in rows:
        method_id = row["method_id"]
        summary = summaries.setdefault(
            method_id,
            {
                "method_id": method_id,
                "gemini_usage_status": "api_usage_recorded",
                "gemini_page_count": 0,
                "gemini_success_count": 0,
                "gemini_attempt_count": 0,
                "gemini_retry_count": 0,
                "gemini_request_count": 0,
                "gemini_missing_usage_count": 0,
                "gemini_elapsed_seconds": 0.0,
                "gemini_prompt_token_count": 0,
                "gemini_candidates_token_count": 0,
                "gemini_total_token_count": 0,
                "gemini_total_billable_characters": 0,
                "gemini_estimated_cost_usd": 0.0 if input_rate is not None and output_rate is not None else None,
                "gemini_pricing_note": (
                    "Estimated from GEMINI_INPUT_USD_PER_1M_TOKENS and "
                    "GEMINI_OUTPUT_USD_PER_1M_TOKENS."
                    if input_rate is not None and output_rate is not None
                    else "Token counts recorded; USD estimate omitted because Gemini price env vars were not set."
                ),
            },
        )
        summary["gemini_page_count"] += 1
        if row.get("status") == "success":
            summary["gemini_success_count"] += 1
        summary["gemini_attempt_count"] += _safe_int(row.get("attempt_count"))
        summary["gemini_retry_count"] += _safe_int(row.get("retry_count"))
        summary["gemini_request_count"] += _safe_int(row.get("request_count"))
        if row.get("request_count") and not row.get("usage_metadata_available"):
            summary["gemini_missing_usage_count"] += 1
        summary["gemini_elapsed_seconds"] += _safe_float(row.get("elapsed_seconds")) or 0.0
        summary["gemini_prompt_token_count"] += _safe_int(row.get("prompt_token_count"))
        summary["gemini_candidates_token_count"] += _safe_int(row.get("candidates_token_count"))
        summary["gemini_total_token_count"] += _safe_int(row.get("total_token_count"))
        summary["gemini_total_billable_characters"] += _safe_int(row.get("total_billable_characters"))
        if summary["gemini_estimated_cost_usd"] is not None:
            summary["gemini_estimated_cost_usd"] += _safe_float(row.get("estimated_cost_usd")) or 0.0

    for summary in summaries.values():
        missing_usage_count = _safe_int(summary.get("gemini_missing_usage_count"))
        if missing_usage_count > 0:
            summary["gemini_pricing_note"] = (
                "Known-usage subtotal from Gemini usage metadata; "
                f"{missing_usage_count} attempted request(s) returned no usage metadata, "
                "so actual API cost may be higher."
            )

    return rows, summaries


def _load_summary_rows(output_root: Path) -> tuple[list[dict], list[dict]]:
    summary_rows = []
    per_page_rows = []
    for metrics_path in _metric_payload_paths(output_root):
        payload = _read_json(metrics_path)
        method_id = _payload_method_id(payload, metrics_path.parent.name)
        layout_effort_by_page = _layout_effort_from_payload(payload)
        summary_rows.append(_summary_row_from_payload(payload, metrics_path))
        for record in payload.get("page_records") or []:
            row = dict(record)
            row.setdefault("method_id", method_id)
            row["display_name"] = METHOD_LABELS.get(method_id, method_id)
            _apply_layout_effort_fallback(row, layout_effort_by_page)
            per_page_rows.append(row)

    summary_rows.sort(key=lambda row: _method_sort_key(row["method_id"]))
    per_page_rows.sort(key=lambda row: (_method_sort_key(row.get("method_id", "")), row.get("fold_id", ""), row.get("page_id", "")))
    return summary_rows, per_page_rows


def _augment_rows_with_usage(summary_rows: list[dict], usage_summaries: dict[str, dict]) -> None:
    for row in summary_rows:
        method_id = row["method_id"]
        usage = usage_summaries.get(method_id)
        if usage:
            row.update(usage)
        elif row["engine"] == "Gemini":
            row.update(
                {
                    "gemini_usage_status": "no_usage_metadata_found",
                    "gemini_page_count": 0,
                    "gemini_success_count": 0,
                    "gemini_attempt_count": 0,
                    "gemini_retry_count": 0,
                    "gemini_request_count": 0,
                    "gemini_missing_usage_count": 0,
                    "gemini_elapsed_seconds": 0.0,
                    "gemini_prompt_token_count": 0,
                    "gemini_candidates_token_count": 0,
                    "gemini_total_token_count": 0,
                    "gemini_total_billable_characters": 0,
                    "gemini_estimated_cost_usd": None,
                    "gemini_pricing_note": "No Gemini usage JSON files were found for this method.",
                }
            )
        else:
            row.update(
                {
                    "gemini_usage_status": "not_applicable_no_api_cost",
                    "gemini_page_count": 0,
                    "gemini_success_count": 0,
                    "gemini_attempt_count": 0,
                    "gemini_retry_count": 0,
                    "gemini_request_count": 0,
                    "gemini_missing_usage_count": 0,
                    "gemini_elapsed_seconds": 0.0,
                    "gemini_prompt_token_count": 0,
                    "gemini_candidates_token_count": 0,
                    "gemini_total_token_count": 0,
                    "gemini_total_billable_characters": 0,
                    "gemini_estimated_cost_usd": 0.0,
                    "gemini_pricing_note": "Annotation-tool methods use local computation; no Gemini API cost.",
                }
            )


LAYOUT_EFFORT_IMPACT_FIELDS = [
    "method_id",
    "display_name",
    "fold_id",
    "page_id",
    "baseline_method_id",
    "baseline_page_cer",
    "target_page_cer",
    "page_cer_reduction",
    "baseline_textedit",
    "target_textedit",
    "textedit_reduction",
    "layout_effort_edit_count",
    "layout_effort_active_edit_time_seconds",
    "layout_effort_revision_count",
    "layout_effort_source_path",
]


def _is_human_corrected_local_layout_method(method_id: str) -> bool:
    return method_id == "annotation_tool_gt_layout" or method_id.startswith("annotation_tool_gt_layout_ft_")


def _has_layout_effort(row: dict) -> bool:
    return _layout_effort_truthy(row.get("layout_effort_available"))


def _layout_effort_impact_rows(per_page_rows: list[dict]) -> list[dict]:
    baseline_by_page: dict[tuple[str, str], dict] = {}
    for row in per_page_rows:
        if row.get("method_id") == "annotation_tool_e2e":
            baseline_by_page[(str(row.get("fold_id", "")), str(row.get("page_id", "")))] = row

    impact_rows = []
    for row in per_page_rows:
        method_id = str(row.get("method_id") or "")
        if not _is_human_corrected_local_layout_method(method_id):
            continue
        fold_id = str(row.get("fold_id", ""))
        page_id = str(row.get("page_id", ""))
        baseline = baseline_by_page.get((fold_id, page_id))
        if baseline is None:
            continue
        baseline_cer = _safe_float(baseline.get("page_cer"))
        target_cer = _safe_float(row.get("page_cer"))
        if baseline_cer is None or target_cer is None:
            continue
        edit_count = _safe_float(row.get("layout_effort_edit_count"))
        active_seconds = _safe_float(row.get("layout_effort_active_edit_time_seconds"))
        if not _has_layout_effort(row) and edit_count is None and active_seconds is None:
            continue
        baseline_textedit = _safe_float(baseline.get("textedit"))
        target_textedit = _safe_float(row.get("textedit"))
        textedit_reduction = (
            baseline_textedit - target_textedit
            if baseline_textedit is not None and target_textedit is not None
            else None
        )
        impact_rows.append(
            {
                "method_id": method_id,
                "display_name": METHOD_LABELS.get(method_id, method_id),
                "fold_id": fold_id,
                "page_id": page_id,
                "baseline_method_id": "annotation_tool_e2e",
                "baseline_page_cer": baseline_cer,
                "target_page_cer": target_cer,
                "page_cer_reduction": baseline_cer - target_cer,
                "baseline_textedit": baseline_textedit,
                "target_textedit": target_textedit,
                "textedit_reduction": textedit_reduction,
                "layout_effort_edit_count": int(edit_count) if edit_count is not None else None,
                "layout_effort_active_edit_time_seconds": active_seconds,
                "layout_effort_revision_count": _safe_float(row.get("layout_effort_revision_count")),
                "layout_effort_source_path": row.get("layout_effort_source_path", ""),
            }
        )
    impact_rows.sort(key=lambda item: (_method_sort_key(item["method_id"]), item["fold_id"], item["page_id"]))
    return impact_rows


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    midpoint = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return sorted_values[midpoint]
    return (sorted_values[midpoint - 1] + sorted_values[midpoint]) / 2.0


def _numeric_impact_values(rows: list[dict], key: str) -> list[float]:
    values = []
    for row in rows:
        value = _safe_float(row.get(key))
        if value is not None:
            values.append(value)
    return values


def _layout_effort_summary_table_rows(impact_rows: list[dict]) -> list[dict]:
    rows_by_method: dict[str, list[dict]] = {}
    for row in impact_rows:
        rows_by_method.setdefault(row["method_id"], []).append(row)
    table_rows = []
    for method_id in sorted(rows_by_method, key=_method_sort_key):
        rows = rows_by_method[method_id]
        table_rows.append(
            {
                "Method": METHOD_LABELS.get(method_id, method_id),
                "Pages": len(rows),
                "Mean Edits": _format_float(_mean(_numeric_impact_values(rows, "layout_effort_edit_count")), 1),
                "Mean Active Time (s)": _format_float(
                    _mean(_numeric_impact_values(rows, "layout_effort_active_edit_time_seconds")),
                    1,
                ),
                "Mean CER Reduction": _format_float(_mean(_numeric_impact_values(rows, "page_cer_reduction"))),
                "Median CER Reduction": _format_float(_median(_numeric_impact_values(rows, "page_cer_reduction"))),
                "Mean TextEdit Reduction": _format_float(_mean(_numeric_impact_values(rows, "textedit_reduction"))),
            }
        )
    return table_rows


def _plotting():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    return plt


def _numeric_values(rows: list[dict], key: str) -> list[float]:
    return [(_safe_float(row.get(key)) or 0.0) for row in rows]


def _method_labels(rows: list[dict]) -> list[str]:
    return [str(row.get("display_name") or row.get("method_id")) for row in rows]


def _save_bar_figure(rows: list[dict], key: str, output_path: Path, *, title: str, ylabel: str) -> Path | None:
    plt = _plotting()
    if plt is None or not rows:
        return None
    labels = _method_labels(rows)
    values = _numeric_values(rows, key)
    fig_width = max(8.0, len(rows) * 1.25)
    fig, ax = plt.subplots(figsize=(fig_width, 4.8))
    bars = ax.bar(range(len(rows)), values, color="#4C78A8")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _save_layout_figure(rows: list[dict], output_path: Path) -> Path | None:
    plt = _plotting()
    if plt is None or not rows:
        return None
    labels = _method_labels(rows)
    keys = ("object_g_f1_50", "object_g_f1_75", "pixel_f1")
    colors = ("#4C78A8", "#F58518", "#54A24B")
    width = 0.24
    fig_width = max(8.5, len(rows) * 1.35)
    fig, ax = plt.subplots(figsize=(fig_width, 5.0))
    x_positions = list(range(len(rows)))
    for offset, (key, color) in enumerate(zip(keys, colors)):
        values = _numeric_values(rows, key)
        shifted = [x + (offset - 1) * width for x in x_positions]
        ax.bar(shifted, values, width=width, label=key, color=color)
    ax.set_title("Layout Metrics By Method")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _save_finetuning_curve(rows: list[dict], output_path: Path) -> Path | None:
    plt = _plotting()
    curve_rows = [
        row
        for row in rows
        if row["method_id"] == "annotation_tool_gt_layout"
        or row["method_id"].startswith("annotation_tool_gt_layout_ft_")
    ]
    curve_rows.sort(key=lambda row: int(row.get("finetune_pages") or 0))
    if plt is None or len(curve_rows) < 2:
        return None
    x_values = [int(row.get("finetune_pages") or 0) for row in curve_rows]
    cer_values = _numeric_values(curve_rows, "micro_page_cer")
    textedit_values = _numeric_values(curve_rows, "micro_textedit")
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.plot(x_values, cer_values, marker="o", label="Micro Page CER", color="#4C78A8")
    ax.plot(x_values, textedit_values, marker="o", label="Micro TextEdit", color="#F58518")
    ax.set_title("Read Mode Fine-Tuning Curve")
    ax.set_xlabel("Fine-tuning pages")
    ax.set_ylabel("Error")
    ax.set_xticks(x_values)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _save_gemini_token_figure(rows: list[dict], output_path: Path) -> Path | None:
    plt = _plotting()
    gemini_rows = [row for row in rows if row.get("engine") == "Gemini"]
    if plt is None or not gemini_rows:
        return None
    labels = _method_labels(gemini_rows)
    prompt = _numeric_values(gemini_rows, "gemini_prompt_token_count")
    candidates = _numeric_values(gemini_rows, "gemini_candidates_token_count")
    fig, ax = plt.subplots(figsize=(max(6.5, len(gemini_rows) * 1.7), 4.5))
    positions = list(range(len(gemini_rows)))
    ax.bar(positions, prompt, label="Prompt tokens", color="#4C78A8")
    ax.bar(positions, candidates, bottom=prompt, label="Candidate tokens", color="#F58518")
    ax.set_title("Gemini Token Usage")
    ax.set_ylabel("Tokens")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _save_layout_effort_impact_figure(rows: list[dict], output_path: Path) -> Path | None:
    plt = _plotting()
    if plt is None or not rows:
        return None
    method_ids = sorted({row["method_id"] for row in rows}, key=_method_sort_key)
    colors = ("#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2")
    figure_has_points = False
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), sharey=True)
    panels = (
        ("layout_effort_active_edit_time_seconds", "Human layout correction time (seconds)"),
        ("layout_effort_edit_count", "Human layout correction edits"),
    )
    for ax, (x_key, xlabel) in zip(axes, panels):
        for index, method_id in enumerate(method_ids):
            method_rows = [row for row in rows if row["method_id"] == method_id]
            points = [
                (_safe_float(row.get(x_key)), _safe_float(row.get("page_cer_reduction")))
                for row in method_rows
            ]
            points = [(x_value, y_value) for x_value, y_value in points if x_value is not None and y_value is not None]
            if not points:
                continue
            figure_has_points = True
            x_values = [point[0] for point in points]
            y_values = [point[1] for point in points]
            ax.scatter(
                x_values,
                y_values,
                label=METHOD_LABELS.get(method_id, method_id),
                color=colors[index % len(colors)],
                alpha=0.8,
                edgecolor="white",
                linewidth=0.7,
                s=58,
            )
        ax.axhline(0, color="#333333", linewidth=0.8, alpha=0.6)
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.25)
    if not figure_has_points:
        plt.close(fig)
        return None
    axes[0].set_ylabel("Page CER reduction vs annotation_tool_e2e")
    axes[1].legend(loc="best", fontsize=8)
    fig.suptitle("Human Layout Effort vs OCR Gain")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _write_figures(report_dir: Path, rows: list[dict], layout_effort_rows: list[dict]) -> list[Path]:
    figure_dir = report_dir / "figures"
    figure_paths = [
        _save_bar_figure(
            rows,
            "micro_page_cer",
            figure_dir / "micro_page_cer_by_method.png",
            title="Micro Page CER By Method",
            ylabel="Micro Page CER",
        ),
        _save_bar_figure(
            rows,
            "micro_textedit",
            figure_dir / "micro_textedit_by_method.png",
            title="Micro Line-group TextEdit By Method",
            ylabel="Micro TextEdit",
        ),
        _save_layout_figure(rows, figure_dir / "layout_metrics_by_method.png"),
        _save_finetuning_curve(rows, figure_dir / "annotation_tool_finetuning_curve.png"),
        _save_gemini_token_figure(rows, figure_dir / "gemini_token_usage.png"),
        _save_layout_effort_impact_figure(layout_effort_rows, figure_dir / "layout_effort_vs_ocr_gain.png"),
    ]
    return [path for path in figure_paths if path is not None]


def _summary_table_rows(rows: list[dict]) -> list[dict]:
    table_rows = []
    for row in rows:
        table_rows.append(
            {
                "Method": row["display_name"],
                "Human Effort": row["human_effort"],
                "Layout Condition": row["layout_condition"],
                "Pages": row["page_count"],
                "Valid Output": _format_float(row.get("valid_output_rate")),
                "G-F1@0.50": _format_float(row.get("object_g_f1_50")),
                "Pixel F1": _format_float(row.get("pixel_f1")),
                "Micro CER": _format_float(row.get("micro_page_cer")),
                "Micro TextEdit": _format_float(row.get("micro_textedit")),
                "Gemini Tokens": row.get("gemini_total_token_count", 0),
                "Gemini USD": _format_cost(row.get("gemini_estimated_cost_usd")),
            }
        )
    return table_rows


def _write_markdown_report(
    output_root: Path,
    report_dir: Path,
    rows: list[dict],
    layout_effort_rows: list[dict],
    usage_rows: list[dict],
    figure_paths: list[Path],
) -> Path:
    report_path = report_dir / "experiment_report.md"
    generated_at = datetime.now(timezone.utc).isoformat()
    table_rows = _summary_table_rows(rows)
    table_columns = [
        ("Method", "Method"),
        ("Human Effort", "Human Effort"),
        ("Layout Condition", "Layout Condition"),
        ("Pages", "Pages"),
        ("Valid Output", "Valid Output"),
        ("G-F1@0.50", "G-F1@0.50"),
        ("Pixel F1", "Pixel F1"),
        ("Micro CER", "Micro CER"),
        ("Micro TextEdit", "Micro TextEdit"),
        ("Gemini Tokens", "Gemini Tokens"),
        ("Gemini USD", "Gemini USD"),
    ]

    gemini_rows = [row for row in rows if row.get("engine") == "Gemini"]
    gemini_table_rows = [
        {
            "Method": row["display_name"],
            "Pages": row.get("gemini_page_count", 0),
            "Attempts": row.get("gemini_attempt_count", 0),
            "Retries": row.get("gemini_retry_count", 0),
            "Requests": row.get("gemini_request_count", 0),
            "Missing Usage": row.get("gemini_missing_usage_count", 0),
            "Prompt Tokens": row.get("gemini_prompt_token_count", 0),
            "Candidate Tokens": row.get("gemini_candidates_token_count", 0),
            "Total Tokens": row.get("gemini_total_token_count", 0),
            "Elapsed Seconds": _format_float(row.get("gemini_elapsed_seconds"), digits=2),
            "Estimated USD": _format_cost(row.get("gemini_estimated_cost_usd")),
            "Usage Note": row.get("gemini_pricing_note", ""),
        }
        for row in gemini_rows
    ]
    layout_effort_table_rows = _layout_effort_summary_table_rows(layout_effort_rows)

    figure_lines = []
    for path in figure_paths:
        relative = path.relative_to(report_dir).as_posix()
        figure_lines.append(f"- `{relative}`")

    content = [
        "# Downstream OCR Experiment Report",
        "",
        f"- Generated at UTC: `{generated_at}`",
        f"- Output root: `{output_root.resolve()}`",
        "",
        "## Headline Metrics",
        "",
        _markdown_table(table_rows, table_columns) if table_rows else "No metric rows found.",
        "",
        "## Interpretation Guide",
        "",
        "- Compare `annotation_tool_e2e` against `annotation_tool_gt_layout` to estimate the practical value of human layout correction for the local OCR pipeline.",
        "- Compare the `annotation_tool_gt_layout` fine-tuning series at 0/1/2/3 pages to estimate the value of Read Mode corrections as manuscript-local OCR supervision.",
        "- Compare `vlm_e2e` against `gemini_gt_layout` as a practical Gemini system comparison. This comparison changes both layout grounding and prompt/interface format, so it is not a perfectly isolated layout-only ablation.",
        "- Rows with `layout_condition=human_corrected_gt_layout` use layout obtained through careful human inspection and correction. Their G-F1 and pixel F1 scores describe the provided human-corrected layout condition, not automatic layout-detector performance.",
        "- Fine-tuning methods record the GUI runtime OCR active-learning recipe and sibling checkpoint selector in `summary_metrics.csv`.",
        "",
        "## Human Layout Effort And OCR Gain",
        "",
        "These rows compare each local human-corrected GT-layout condition against `annotation_tool_e2e` on the same fold and page. Positive reductions mean the human-corrected layout condition lowered the error.",
        "",
        _markdown_table(
            layout_effort_table_rows,
            [
                ("Method", "Method"),
                ("Pages", "Pages"),
                ("Mean Edits", "Mean Edits"),
                ("Mean Active Time (s)", "Mean Active Time (s)"),
                ("Mean CER Reduction", "Mean CER Reduction"),
                ("Median CER Reduction", "Median CER Reduction"),
                ("Mean TextEdit Reduction", "Mean TextEdit Reduction"),
            ],
        )
        if layout_effort_table_rows
        else "No layout-effort impact rows found. This requires `annotation_tool_e2e`, local human-corrected GT-layout page records, and a manuscript `layout_analysis_output/layout_effort.json` file.",
        "",
        f"Per-page layout effort impact rows: `{(report_dir / 'layout_effort_impact.csv').relative_to(report_dir).as_posix()}`",
        "",
        "## Gemini Usage And Cost",
        "",
        "Annotation-tool methods use local computation and are assigned zero Gemini API cost. Gemini rows report API token usage when the SDK returns usage metadata. USD estimates are only filled when `GEMINI_INPUT_USD_PER_1M_TOKENS` and `GEMINI_OUTPUT_USD_PER_1M_TOKENS` are set for the run.",
        "",
        _markdown_table(
            gemini_table_rows,
            [
                ("Method", "Method"),
                ("Pages", "Pages"),
                ("Attempts", "Attempts"),
                ("Retries", "Retries"),
                ("Requests", "Requests"),
                ("Missing Usage", "Missing Usage"),
                ("Prompt Tokens", "Prompt Tokens"),
                ("Candidate Tokens", "Candidate Tokens"),
                ("Total Tokens", "Total Tokens"),
                ("Elapsed Seconds", "Elapsed Seconds"),
                ("Estimated USD", "Estimated USD"),
                ("Usage Note", "Usage Note"),
            ],
        )
        if gemini_table_rows
        else "No Gemini usage rows found.",
        "",
        f"Per-page Gemini usage rows: `{(report_dir / 'gemini_usage.csv').relative_to(report_dir).as_posix()}`",
        f"Per-page metric rows: `{(report_dir / 'per_page_metrics.csv').relative_to(report_dir).as_posix()}`",
        "",
        "## Figures",
        "",
        "\n".join(figure_lines) if figure_lines else "No figures were generated. Install matplotlib to enable figures.",
        "",
    ]
    report_path.write_text("\n".join(content), encoding="utf-8")
    return report_path


def write_experiment_report(
    output_root: str | Path,
    *,
    input_usd_per_1m_tokens: float | None = None,
    output_usd_per_1m_tokens: float | None = None,
) -> ReportArtifacts:
    root = Path(output_root)
    report_dir = root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    summary_rows, per_page_rows = _load_summary_rows(root)
    usage_rows, usage_summaries = collect_gemini_usage(
        root,
        input_usd_per_1m_tokens=input_usd_per_1m_tokens,
        output_usd_per_1m_tokens=output_usd_per_1m_tokens,
    )
    _augment_rows_with_usage(summary_rows, usage_summaries)

    summary_csv_path = report_dir / "summary_metrics.csv"
    summary_json_path = report_dir / "summary_metrics.json"
    per_page_csv_path = report_dir / "per_page_metrics.csv"
    layout_effort_impact_csv_path = report_dir / "layout_effort_impact.csv"
    layout_effort_impact_json_path = report_dir / "layout_effort_impact.json"
    gemini_usage_csv_path = report_dir / "gemini_usage.csv"
    gemini_usage_json_path = report_dir / "gemini_usage.json"
    layout_effort_impact_rows = _layout_effort_impact_rows(per_page_rows)

    summary_fields = [
        "method_id",
        "display_name",
        "engine",
        "human_effort",
        "human_layout",
        "layout_condition",
        "layout_metric_interpretation",
        "finetune_pages",
        "ocr_recipe_source",
        "sibling_checkpoint_strategy",
        "page_count",
        "valid_output_rate",
        "object_g_f1_50",
        "object_g_f1_75",
        "pixel_f1",
        "mean_page_cer",
        "median_page_cer",
        "micro_page_cer",
        "mean_textedit",
        "median_textedit",
        "micro_textedit",
        "gemini_usage_status",
        "gemini_page_count",
        "gemini_success_count",
        "gemini_attempt_count",
        "gemini_retry_count",
        "gemini_request_count",
        "gemini_missing_usage_count",
        "gemini_elapsed_seconds",
        "gemini_prompt_token_count",
        "gemini_candidates_token_count",
        "gemini_total_token_count",
        "gemini_total_billable_characters",
        "gemini_estimated_cost_usd",
        "gemini_pricing_note",
        "metrics_path",
    ]
    _write_csv(summary_csv_path, summary_rows, fieldnames=summary_fields)
    _write_json(summary_json_path, summary_rows)
    _write_csv(per_page_csv_path, per_page_rows)
    _write_csv(
        layout_effort_impact_csv_path,
        layout_effort_impact_rows,
        fieldnames=LAYOUT_EFFORT_IMPACT_FIELDS,
    )
    _write_json(
        layout_effort_impact_json_path,
        {
            "comparison_baseline": "annotation_tool_e2e",
            "positive_reduction_means": "lower error after the human-corrected GT-layout condition",
            "rows": layout_effort_impact_rows,
        },
    )
    _write_csv(gemini_usage_csv_path, usage_rows)
    _write_json(gemini_usage_json_path, {"rows": usage_rows, "summaries": usage_summaries})

    figure_paths = _write_figures(report_dir, summary_rows, layout_effort_impact_rows)
    markdown_path = _write_markdown_report(
        root,
        report_dir,
        summary_rows,
        layout_effort_impact_rows,
        usage_rows,
        figure_paths,
    )
    manifest_path = report_dir / "report_manifest.json"
    manifest = {
        "report_dir": str(report_dir.resolve()),
        "markdown_path": str(markdown_path.resolve()),
        "summary_csv_path": str(summary_csv_path.resolve()),
        "summary_json_path": str(summary_json_path.resolve()),
        "per_page_csv_path": str(per_page_csv_path.resolve()),
        "layout_effort_impact_csv_path": str(layout_effort_impact_csv_path.resolve()),
        "layout_effort_impact_json_path": str(layout_effort_impact_json_path.resolve()),
        "gemini_usage_csv_path": str(gemini_usage_csv_path.resolve()),
        "gemini_usage_json_path": str(gemini_usage_json_path.resolve()),
        "figure_paths": [str(path.resolve()) for path in figure_paths],
        "method_count": len(summary_rows),
        "per_page_record_count": len(per_page_rows),
        "layout_effort_impact_record_count": len(layout_effort_impact_rows),
        "gemini_usage_record_count": len(usage_rows),
    }
    _write_json(manifest_path, manifest)

    return ReportArtifacts(
        report_dir=report_dir,
        markdown_path=markdown_path,
        summary_csv_path=summary_csv_path,
        summary_json_path=summary_json_path,
        per_page_csv_path=per_page_csv_path,
        layout_effort_impact_csv_path=layout_effort_impact_csv_path,
        layout_effort_impact_json_path=layout_effort_impact_json_path,
        gemini_usage_csv_path=gemini_usage_csv_path,
        gemini_usage_json_path=gemini_usage_json_path,
        figure_paths=tuple(figure_paths),
        manifest_path=manifest_path,
    )
