from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .metrics import aggregate_page_records
from .vlm_providers import VLM_PROVIDER_SPECS, is_vlm_method


METHOD_ORDER = (
    *(spec.method_id for spec in VLM_PROVIDER_SPECS),
    "annotation_tool_e2e",
    "annotation_tool_pred_layout_ft_1",
    "annotation_tool_pred_layout_ft_2",
    "annotation_tool_pred_layout_ft_3",
    "annotation_tool_gt_layout",
    "annotation_tool_gt_layout_ft_1",
    "annotation_tool_gt_layout_ft_2",
    "annotation_tool_gt_layout_ft_3",
)

METHOD_LABELS = {
    **{spec.method_id: spec.display_name for spec in VLM_PROVIDER_SPECS},
    "annotation_tool_e2e": "Annotation tool e2e",
    "annotation_tool_gt_layout": "Annotation tool human-corrected GT layout",
    "annotation_tool_pred_layout_ft_1": "Annotation tool predicted test layout + 1 page FT",
    "annotation_tool_gt_layout_ft_1": "Annotation tool human-corrected GT layout + 1 page FT",
    "annotation_tool_pred_layout_ft_2": "Annotation tool predicted test layout + 2 page FT",
    "annotation_tool_gt_layout_ft_2": "Annotation tool human-corrected GT layout + 2 page FT",
    "annotation_tool_pred_layout_ft_3": "Annotation tool predicted test layout + 3 page FT",
    "annotation_tool_gt_layout_ft_3": "Annotation tool human-corrected GT layout + 3 page FT",
}

# Important: what we refer to as annotation tool here, is refered to as the Traditional Pipeline in the paper.
FIGURE_METHOD_LABELS = {
    **{spec.method_id: spec.display_name.split(" (", 1)[0] for spec in VLM_PROVIDER_SPECS},
    "annotation_tool_e2e": "Traditional\nPipeline",
    "annotation_tool_gt_layout": "Traditional\nPipeline",
    "annotation_tool_pred_layout_ft_1": "Traditional\nPipeline",
    "annotation_tool_gt_layout_ft_1": "Traditional\nPipeline",
    "annotation_tool_pred_layout_ft_2": "Traditional\nPipeline",
    "annotation_tool_gt_layout_ft_2": "Traditional\nPipeline",
    "annotation_tool_pred_layout_ft_3": "Traditional\nPipeline",
    "annotation_tool_gt_layout_ft_3": "Traditional\nPipeline",
}

EFFORT_LEVELS = {
    **{spec.method_id: 0 for spec in VLM_PROVIDER_SPECS},
    "annotation_tool_e2e": 0,
    "annotation_tool_pred_layout_ft_1": 1,
    "annotation_tool_pred_layout_ft_2": 2,
    "annotation_tool_pred_layout_ft_3": 3,
    "annotation_tool_gt_layout": 4,
    "annotation_tool_gt_layout_ft_1": 5,
    "annotation_tool_gt_layout_ft_2": 6,
    "annotation_tool_gt_layout_ft_3": 7,
}

EFFORT_LEVEL_COLORS = {
    0: "#F7FBFF",
    1: "#DEEBF7",
    2: "#C6DBEF",
    3: "#9ECAE1",
    4: "#6BAED6",
    5: "#4292C6",
    6: "#2171B5",
    7: "#08306B",
}

EFFORT_GROUP_LABELS = {
    0: "Off the Shelf",
    1: "No test layout correction\n+ 1-page Fine-Tuning",
    2: "No test layout correction\n+ 2-page Fine-Tuning",
    3: "No test layout correction\n+ 3-page Fine-Tuning",
    4: "Test-page layout\npost-correction",
    5: "Test layout post-correction\n+ 1-page Fine-Tuning",
    6: "Test layout post-correction\n+ 2-page Fine-Tuning",
    7: "Test layout post-correction\n+ 3-page Fine-Tuning",
}

EFFORT_COMPARTMENT_ALPHA = 0.82
EFFORT_MIN_COMPARTMENT_WIDTH = 2.25
EFFORT_COMPARTMENT_GAP = 0.0
BOOTSTRAP_CONFIDENCE_LEVEL = 0.95
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 42

LAYOUT_MODE_COMPARISONS = (
    ("Annotation Tool", "annotation_tool_e2e", "annotation_tool_gt_layout"),
)

DISABLED_METHOD_IDS = {"gemini_gt_layout"}

OFF_THE_SHELF_METHODS = {
    spec.method_id: {
        "provider": spec.display_name.split(" (", 1)[0],
        "model_family": spec.model_id,
        "input_contract": "page_image_only",
        "prompt_contract": "vlm_end_to_end_prompt_v1_json_lines_or_polygons",
    }
    for spec in VLM_PROVIDER_SPECS
}

ANNOTATION_GAIN_METHOD_PAIRS = {
    0: ("annotation_tool_e2e", "annotation_tool_gt_layout"),
    1: ("annotation_tool_pred_layout_ft_1", "annotation_tool_gt_layout_ft_1"),
    2: ("annotation_tool_pred_layout_ft_2", "annotation_tool_gt_layout_ft_2"),
    3: ("annotation_tool_pred_layout_ft_3", "annotation_tool_gt_layout_ft_3"),
}

BOOTSTRAP_METRICS = {
    "micro_page_cer": ("page_cer_distance", "page_cer_gt_chars", "Page CER"),
    "micro_textedit": ("textedit_distance_sum", "textedit_max_length_sum", "TextEdit"),
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
    fold_metrics_csv_path: Path
    fold_metrics_json_path: Path
    layout_mode_comparisons_csv_path: Path
    layout_mode_comparisons_json_path: Path
    off_the_shelf_table_csv_path: Path
    off_the_shelf_table_json_path: Path
    annotation_gains_table_csv_path: Path
    annotation_gains_table_json_path: Path
    vlm_usage_csv_path: Path
    vlm_usage_json_path: Path
    figure_paths: tuple[Path, ...]
    manifest_path: Path


@dataclass(frozen=True)
class CombinedTableArtifacts:
    report_dir: Path
    markdown_path: Path
    off_the_shelf_table_csv_path: Path
    off_the_shelf_table_json_path: Path
    annotation_gains_table_csv_path: Path
    annotation_gains_table_json_path: Path
    summary_csv_path: Path
    summary_json_path: Path
    per_page_csv_path: Path
    fold_metrics_csv_path: Path
    layout_mode_comparisons_csv_path: Path
    vlm_usage_csv_path: Path
    vlm_usage_json_path: Path
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


def _format_estimate_ci(estimate: Any, lower: Any, upper: Any, digits: int = 4) -> str:
    estimate_value = _safe_float(estimate)
    lower_value = _safe_float(lower)
    upper_value = _safe_float(upper)
    if estimate_value is None:
        return ""
    if lower_value is None or upper_value is None:
        return f"{estimate_value:.{digits}f}"
    return (
        f"{estimate_value:.{digits}f} "
        f"[{lower_value:.{digits}f}, {upper_value:.{digits}f}]"
    )


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
    if is_vlm_method(method_id):
        return "none"
    if method_id == "annotation_tool_e2e":
        return "none"
    if method.get("uses_finetuning"):
        page_count = int(method.get("finetune_page_count") or 0)
        page_word = "page" if page_count == 1 else "pages"
        test_layout = (
            "human-corrected test layouts"
            if method.get("uses_gt_layout")
            else "no test-page layout correction"
        )
        return (
            f"human-corrected layout and text on {page_count} training {page_word}; "
            f"{test_layout}; Read Mode effort not quantified"
        )
    if method.get("uses_gt_layout"):
        return "human layout correction"
    return "none"


def _engine_label(method_id: str, method: dict) -> str:
    provider_id = method.get("provider_id")
    for spec in VLM_PROVIDER_SPECS:
        if method_id == spec.method_id or provider_id == spec.provider_id:
            return spec.display_name.split(" (", 1)[0]
    return "Annotation tool"


def _layout_condition_label(method: dict) -> str:
    return "human_corrected_gt_layout" if method.get("uses_gt_layout") else "predicted_layout"


def _training_layout_condition_label(method: dict) -> str:
    if method.get("uses_finetuning"):
        return "human_corrected_gt_layout"
    return "not_applicable"


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
        "manuscript_id": str(payload.get("manuscript_id") or ""),
        "method_id": method_id,
        "display_name": METHOD_LABELS.get(method_id) or method.get("display_name") or method_id,
        "engine": _engine_label(method_id, method),
        "human_effort": _human_effort_label(method_id, method),
        "human_training_layout": bool(method.get("uses_finetuning")),
        "human_test_layout": bool(method.get("uses_gt_layout")),
        "human_layout": bool(method.get("uses_gt_layout")),
        "training_layout_condition": _training_layout_condition_label(method),
        "test_layout_condition": _layout_condition_label(method),
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


def collect_vlm_usage(
    output_root: str | Path,
    *,
    input_usd_per_1m_tokens: float | None = None,
    output_usd_per_1m_tokens: float | None = None,
) -> tuple[list[dict], dict[str, dict]]:
    root = Path(output_root)
    rows: list[dict] = []
    seen_results: set[Path] = set()
    for metrics_path in _metric_payload_paths(root):
        metrics_payload = _read_json(metrics_path)
        method_id = _payload_method_id(metrics_payload, metrics_path.parent.name)
        cache = metrics_payload.get("preprediction_cache")
        if not isinstance(cache, dict):
            continue
        manifest_path = Path(str(cache.get("manifest_path") or ""))
        if not manifest_path.exists():
            continue
        manifest = _read_json(manifest_path)
        provider = dict(manifest.get("provider") or {})
        provider_id = str(provider.get("provider_id") or "")
        input_rate = (
            input_usd_per_1m_tokens
            if provider_id == "gemini" and input_usd_per_1m_tokens is not None
            else _env_float(f"{provider_id.upper()}_INPUT_USD_PER_1M_TOKENS")
        )
        output_rate = (
            output_usd_per_1m_tokens
            if provider_id == "gemini" and output_usd_per_1m_tokens is not None
            else _env_float(f"{provider_id.upper()}_OUTPUT_USD_PER_1M_TOKENS")
        )
        for page_id, page_entry in sorted((manifest.get("pages") or {}).items()):
            result_path = (manifest_path.parent / page_entry["result_path"]).resolve()
            if result_path in seen_results:
                continue
            seen_results.add(result_path)
            payload = _read_json(result_path)
            attempt_count = _safe_int(payload.get("attempt_count"))
            input_tokens = _safe_int(payload.get("input_tokens"))
            output_tokens = _safe_int(payload.get("output_tokens"))
            total_tokens = _safe_int(payload.get("total_tokens"))
            usage_metadata_available = any((input_tokens, output_tokens, total_tokens))
            estimated_cost = (
                _estimate_gemini_cost_usd(
                    prompt_tokens=input_tokens,
                    candidate_tokens=output_tokens,
                    input_usd_per_1m_tokens=input_rate,
                    output_usd_per_1m_tokens=output_rate,
                )
                if usage_metadata_available
                else None
            )
            rows.append(
                {
                    "manuscript_id": manifest.get("manuscript_id", ""),
                    "method_id": method_id,
                    "provider_id": provider_id,
                    "model_id": provider.get("model_id", ""),
                    "fold_id": "acquisition_cache",
                    "page_id": page_id,
                    "status": payload.get("status", "unknown"),
                    "elapsed_seconds": _safe_float(payload.get("elapsed_seconds")),
                    "attempt_count": attempt_count,
                    "retry_count": max(0, attempt_count - 1),
                    "max_retries": manifest.get("max_retries_after_initial_attempt", 0),
                    "request_count": attempt_count,
                    "usage_metadata_available": usage_metadata_available,
                    "prompt_token_count": input_tokens,
                    "candidates_token_count": output_tokens,
                    "total_token_count": total_tokens,
                    "total_billable_characters": 0,
                    "estimated_cost_usd": estimated_cost,
                    "usage_path": str(result_path),
                    "input_usd_per_1m_tokens": input_rate,
                    "output_usd_per_1m_tokens": output_rate,
                }
            )

    summaries: dict[str, dict] = {}
    for row in rows:
        method_id = row["method_id"]
        pricing_available = (
            row.get("input_usd_per_1m_tokens") is not None
            and row.get("output_usd_per_1m_tokens") is not None
        )
        summary = summaries.setdefault(
            method_id,
            {
                "method_id": method_id,
                "vlm_usage_status": "api_usage_recorded",
                "vlm_page_count": 0,
                "vlm_success_count": 0,
                "vlm_attempt_count": 0,
                "vlm_retry_count": 0,
                "vlm_request_count": 0,
                "vlm_missing_usage_count": 0,
                "vlm_elapsed_seconds": 0.0,
                "vlm_input_token_count": 0,
                "vlm_output_token_count": 0,
                "vlm_total_token_count": 0,
                "vlm_estimated_cost_usd": 0.0 if pricing_available else None,
                "vlm_pricing_note": (
                    "Estimated from provider-specific input/output token rates."
                    if pricing_available
                    else "Token counts recorded; USD estimate omitted because provider price env vars were not set."
                ),
            },
        )
        summary["vlm_page_count"] += 1
        if row.get("status") == "success":
            summary["vlm_success_count"] += 1
        summary["vlm_attempt_count"] += _safe_int(row.get("attempt_count"))
        summary["vlm_retry_count"] += _safe_int(row.get("retry_count"))
        summary["vlm_request_count"] += _safe_int(row.get("request_count"))
        if row.get("request_count") and not row.get("usage_metadata_available"):
            summary["vlm_missing_usage_count"] += 1
        summary["vlm_elapsed_seconds"] += _safe_float(row.get("elapsed_seconds")) or 0.0
        summary["vlm_input_token_count"] += _safe_int(row.get("prompt_token_count"))
        summary["vlm_output_token_count"] += _safe_int(row.get("candidates_token_count"))
        summary["vlm_total_token_count"] += _safe_int(row.get("total_token_count"))
        if summary["vlm_estimated_cost_usd"] is not None:
            summary["vlm_estimated_cost_usd"] += _safe_float(row.get("estimated_cost_usd")) or 0.0

    for summary in summaries.values():
        missing_usage_count = _safe_int(summary.get("vlm_missing_usage_count"))
        if missing_usage_count > 0:
            summary["vlm_pricing_note"] = (
                "Known-usage subtotal from provider usage metadata; "
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
        if method_id in DISABLED_METHOD_IDS:
            continue
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
        elif is_vlm_method(method_id):
            row.update(
                {
                    "vlm_usage_status": "no_usage_metadata_found",
                    "vlm_page_count": 0,
                    "vlm_success_count": 0,
                    "vlm_attempt_count": 0,
                    "vlm_retry_count": 0,
                    "vlm_request_count": 0,
                    "vlm_missing_usage_count": 0,
                    "vlm_elapsed_seconds": 0.0,
                    "vlm_input_token_count": 0,
                    "vlm_output_token_count": 0,
                    "vlm_total_token_count": 0,
                    "vlm_estimated_cost_usd": None,
                    "vlm_pricing_note": "No referenced VLM acquisition-cache usage records were found for this method.",
                }
            )
        else:
            row.update(
                {
                    "vlm_usage_status": "not_applicable_no_api_cost",
                    "vlm_page_count": 0,
                    "vlm_success_count": 0,
                    "vlm_attempt_count": 0,
                    "vlm_retry_count": 0,
                    "vlm_request_count": 0,
                    "vlm_missing_usage_count": 0,
                    "vlm_elapsed_seconds": 0.0,
                    "vlm_input_token_count": 0,
                    "vlm_output_token_count": 0,
                    "vlm_total_token_count": 0,
                    "vlm_estimated_cost_usd": 0.0,
                    "vlm_pricing_note": "Annotation-tool methods use local computation; no VLM API cost.",
                }
            )


def _cluster_key(row: dict) -> tuple[str, str]:
    return (str(row.get("manuscript_id") or ""), str(row.get("page_id") or ""))


def _stable_bootstrap_seed(label: str) -> int:
    digest = hashlib.sha256(f"{BOOTSTRAP_SEED}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = max(0.0, min(1.0, float(probability))) * (len(ordered) - 1)
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return float(ordered[lower_index])
    fraction = position - lower_index
    return float(ordered[lower_index] * (1.0 - fraction) + ordered[upper_index] * fraction)


def _confidence_bounds(values: list[float]) -> tuple[float | None, float | None]:
    tail = (1.0 - BOOTSTRAP_CONFIDENCE_LEVEL) / 2.0
    return _percentile(values, tail), _percentile(values, 1.0 - tail)


def _group_metric_counts(
    rows: Iterable[dict],
    *,
    numerator_key: str,
    denominator_key: str,
) -> dict[tuple[str, str], tuple[float, float]]:
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        numerator = _safe_float(row.get(numerator_key))
        denominator = _safe_float(row.get(denominator_key))
        if numerator is None or denominator is None:
            continue
        counts = grouped.setdefault(_cluster_key(row), [0.0, 0.0])
        counts[0] += numerator
        counts[1] += denominator
    return {key: (values[0], values[1]) for key, values in grouped.items()}


def _bootstrap_micro_metric(
    rows: Iterable[dict],
    *,
    numerator_key: str,
    denominator_key: str,
    seed_label: str,
) -> dict | None:
    grouped = _group_metric_counts(
        rows,
        numerator_key=numerator_key,
        denominator_key=denominator_key,
    )
    cluster_keys = sorted(grouped)
    if not cluster_keys:
        return None
    numerator = sum(grouped[key][0] for key in cluster_keys)
    denominator = sum(grouped[key][1] for key in cluster_keys)
    if denominator <= 0:
        return None
    estimate = numerator / denominator
    if len(cluster_keys) == 1:
        samples = [estimate]
    else:
        rng = random.Random(_stable_bootstrap_seed(seed_label))
        samples = []
        for _ in range(BOOTSTRAP_RESAMPLES):
            sampled_keys = [cluster_keys[rng.randrange(len(cluster_keys))] for _ in cluster_keys]
            sample_numerator = sum(grouped[key][0] for key in sampled_keys)
            sample_denominator = sum(grouped[key][1] for key in sampled_keys)
            if sample_denominator > 0:
                samples.append(sample_numerator / sample_denominator)
    lower, upper = _confidence_bounds(samples)
    return {
        "estimate": float(estimate),
        "ci_lower": lower,
        "ci_upper": upper,
        "unique_page_count": len(cluster_keys),
        "confidence_level": BOOTSTRAP_CONFIDENCE_LEVEL,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "cluster_unit": "manuscript_id+page_id",
    }


def _augment_rows_with_bootstrap_cis(summary_rows: list[dict], per_page_rows: list[dict]) -> None:
    rows_by_method: dict[str, list[dict]] = {}
    for row in per_page_rows:
        rows_by_method.setdefault(str(row.get("method_id") or ""), []).append(row)
    for summary in summary_rows:
        method_id = str(summary.get("method_id") or "")
        method_rows = rows_by_method.get(method_id, [])
        for metric_key, (numerator_key, denominator_key, _) in BOOTSTRAP_METRICS.items():
            result = _bootstrap_micro_metric(
                method_rows,
                numerator_key=numerator_key,
                denominator_key=denominator_key,
                seed_label=f"method:{method_id}:{metric_key}",
            )
            summary[f"{metric_key}_ci_lower"] = result["ci_lower"] if result else None
            summary[f"{metric_key}_ci_upper"] = result["ci_upper"] if result else None
            summary[f"{metric_key}_bootstrap_unique_pages"] = result["unique_page_count"] if result else 0


def _mean_effort_by_cluster(rows: Iterable[dict]) -> dict[tuple[str, str], float]:
    effort_by_cluster: dict[tuple[str, str], float] = {}
    for row in rows:
        effort = _safe_float(row.get("layout_effort_active_edit_time_seconds"))
        if effort is not None:
            effort_by_cluster.setdefault(_cluster_key(row), effort)
    return effort_by_cluster


def _bootstrap_mean_effort(rows: Iterable[dict], *, seed_label: str) -> dict | None:
    effort_by_cluster = _mean_effort_by_cluster(rows)
    cluster_keys = sorted(effort_by_cluster)
    if not cluster_keys:
        return None
    estimate = sum(effort_by_cluster.values()) / len(cluster_keys)
    if len(cluster_keys) == 1:
        samples = [estimate]
    else:
        rng = random.Random(_stable_bootstrap_seed(seed_label))
        samples = [
            sum(effort_by_cluster[cluster_keys[rng.randrange(len(cluster_keys))]] for _ in cluster_keys)
            / len(cluster_keys)
            for _ in range(BOOTSTRAP_RESAMPLES)
        ]
    lower, upper = _confidence_bounds(samples)
    return {
        "mean_seconds_per_page": float(estimate),
        "ci_lower": lower,
        "ci_upper": upper,
        "unique_page_count": len(cluster_keys),
        "confidence_level": BOOTSTRAP_CONFIDENCE_LEVEL,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "cluster_unit": "manuscript_id+page_id",
    }


def _layout_mode_comparison_rows(per_page_rows: list[dict]) -> list[dict]:
    rows_by_method: dict[str, list[dict]] = {}
    for row in per_page_rows:
        rows_by_method.setdefault(str(row.get("method_id") or ""), []).append(row)

    comparison_rows: list[dict] = []
    for engine, baseline_method_id, target_method_id in LAYOUT_MODE_COMPARISONS:
        baseline_rows = rows_by_method.get(baseline_method_id, [])
        target_rows = rows_by_method.get(target_method_id, [])
        for metric_key, (numerator_key, denominator_key, metric_label) in BOOTSTRAP_METRICS.items():
            baseline_groups = _group_metric_counts(
                baseline_rows,
                numerator_key=numerator_key,
                denominator_key=denominator_key,
            )
            target_groups = _group_metric_counts(
                target_rows,
                numerator_key=numerator_key,
                denominator_key=denominator_key,
            )
            common_keys = sorted(set(baseline_groups) & set(target_groups))
            if not common_keys:
                continue
            common_key_set = set(common_keys)
            effort = _bootstrap_mean_effort(
                [row for row in target_rows if _cluster_key(row) in common_key_set],
                seed_label="layout-effort",
            )

            def reduction(sampled_keys: list[tuple[str, str]]) -> tuple[float, float, float, float] | None:
                baseline_numerator = sum(baseline_groups[key][0] for key in sampled_keys)
                baseline_denominator = sum(baseline_groups[key][1] for key in sampled_keys)
                target_numerator = sum(target_groups[key][0] for key in sampled_keys)
                target_denominator = sum(target_groups[key][1] for key in sampled_keys)
                if baseline_denominator <= 0 or target_denominator <= 0:
                    return None
                baseline_value = baseline_numerator / baseline_denominator
                target_value = target_numerator / target_denominator
                absolute = baseline_value - target_value
                relative = (absolute / baseline_value) * 100.0 if baseline_value > 0 else 0.0
                return baseline_value, target_value, absolute, relative

            point = reduction(common_keys)
            if point is None:
                continue
            if len(common_keys) == 1:
                bootstrap_results = [point]
            else:
                rng = random.Random(
                    _stable_bootstrap_seed(f"comparison:{baseline_method_id}:{target_method_id}:{metric_key}")
                )
                bootstrap_results = []
                for _ in range(BOOTSTRAP_RESAMPLES):
                    sampled = [common_keys[rng.randrange(len(common_keys))] for _ in common_keys]
                    sample_result = reduction(sampled)
                    if sample_result is not None:
                        bootstrap_results.append(sample_result)
            absolute_lower, absolute_upper = _confidence_bounds([item[2] for item in bootstrap_results])
            relative_lower, relative_upper = _confidence_bounds([item[3] for item in bootstrap_results])
            comparison_rows.append(
                {
                    "engine": engine,
                    "baseline_method_id": baseline_method_id,
                    "target_method_id": target_method_id,
                    "metric_key": metric_key,
                    "metric_label": metric_label,
                    "baseline_micro": point[0],
                    "target_micro": point[1],
                    "absolute_reduction": point[2],
                    "absolute_reduction_ci_lower": absolute_lower,
                    "absolute_reduction_ci_upper": absolute_upper,
                    "relative_reduction_percent": point[3],
                    "relative_reduction_ci_lower": relative_lower,
                    "relative_reduction_ci_upper": relative_upper,
                    "layout_effort_mean_seconds_per_page": effort["mean_seconds_per_page"] if effort else None,
                    "layout_effort_ci_lower": effort["ci_lower"] if effort else None,
                    "layout_effort_ci_upper": effort["ci_upper"] if effort else None,
                    "unique_page_count": len(common_keys),
                    "confidence_level": BOOTSTRAP_CONFIDENCE_LEVEL,
                    "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                    "cluster_unit": "manuscript_id+page_id",
                }
            )
    return comparison_rows


def _fold_metric_rows(per_page_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in per_page_rows:
        key = (str(row.get("method_id") or ""), str(row.get("fold_id") or ""))
        grouped.setdefault(key, []).append(row)
    rows = []
    for (method_id, fold_id), records in sorted(grouped.items(), key=lambda item: (_method_sort_key(item[0][0]), item[0][1])):
        aggregate = aggregate_page_records(records)
        rows.append({"method_id": method_id, "fold_id": fold_id, **aggregate})
    return rows


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


def _figure_method_labels(rows: list[dict]) -> list[str]:
    return [
        FIGURE_METHOD_LABELS.get(str(row.get("method_id") or ""))
        or str(row.get("display_name") or row.get("method_id"))
        for row in rows
    ]


def _effort_level(method_id: str) -> int:
    return EFFORT_LEVELS.get(method_id, max(EFFORT_LEVELS.values()) + 1)


def _effort_sort_key(row: dict) -> tuple[int, tuple[int, str]]:
    method_id = str(row.get("method_id") or "")
    return (_effort_level(method_id), _method_sort_key(method_id))


def _effort_plot_geometry(rows: list[dict]) -> tuple[list[float], list[tuple[int, float, float, float]], tuple[float, float]]:
    x_positions: list[float] = []
    group_spans: list[tuple[int, float, float, float]] = []
    start = 0
    cursor = 0.0
    while start < len(rows):
        level = _effort_level(str(rows[start].get("method_id") or ""))
        end = start
        while end + 1 < len(rows) and _effort_level(str(rows[end + 1].get("method_id") or "")) == level:
            end += 1
        group_count = end - start + 1
        width = max(float(group_count), EFFORT_MIN_COMPARTMENT_WIDTH)
        left = cursor
        right = cursor + width
        if group_count == 1:
            x_positions.append((left + right) / 2.0)
        else:
            first_x = left + (width - (group_count - 1)) / 2.0
            x_positions.extend(first_x + offset for offset in range(group_count))
        group_spans.append((level, left, right, (left + right) / 2.0))
        cursor = right + EFFORT_COMPARTMENT_GAP
        start = end + 1

    if not group_spans:
        return x_positions, group_spans, (-0.5, 0.5)
    return x_positions, group_spans, (group_spans[0][1], group_spans[-1][2])


def _add_effort_group_guides(ax, group_spans: list[tuple[int, float, float, float]]) -> None:
    for group_index, (level, left, right, center) in enumerate(group_spans):
        ax.axvspan(left, right, color=EFFORT_LEVEL_COLORS[level], alpha=EFFORT_COMPARTMENT_ALPHA, zorder=0)
        if group_index > 0:
            ax.axvline(left, color="#333333", linewidth=0.9, alpha=0.55)
        label = EFFORT_GROUP_LABELS.get(level)
        if label:
            ax.text(
                center,
                1.035,
                label,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=8.2,
                color="#333333",
                linespacing=1.05,
                clip_on=False,
            )



def _layout_comparison_note(comparison_rows: list[dict], metric_key: str) -> str:
    selected = [row for row in comparison_rows if row.get("metric_key") == metric_key]

    effort_values = []
    performance_improvements = []

    for row in selected:
        effort = _safe_float(row.get("layout_effort_mean_seconds_per_page"))
        effort_low = _safe_float(row.get("layout_effort_ci_lower"))
        effort_high = _safe_float(row.get("layout_effort_ci_upper"))

        reduction = _safe_float(row.get("relative_reduction_percent"))
        reduction_low = _safe_float(row.get("relative_reduction_ci_lower"))
        reduction_high = _safe_float(row.get("relative_reduction_ci_upper"))

        if None in {
            effort,
            effort_low,
            effort_high,
            reduction,
            reduction_low,
            reduction_high,
        }:
            continue

        effort_margin = (effort_high - effort_low) / 2
        reduction_margin = (reduction_high - reduction_low) / 2

        effort_values.append((effort, effort_margin))
        performance_improvements.append(
            f"performance of {row['engine']} by "
            f"{reduction:.1f}% (±{reduction_margin:.1f}%)"
        )

    if not performance_improvements:
        return ""

    # Layout effort is expected to be shared across engines.
    effort, effort_margin = effort_values[0]

    if len(performance_improvements) == 1:
        improvements_text = performance_improvements[0]
    else:
        improvements_text = (
            ", ".join(performance_improvements[:-1])
            + f", and {performance_improvements[-1]}"
        )

    note = (
        "Manually post-correcting page layouts "
        f"({effort:.1f} seconds per page, ±{effort_margin:.1f} seconds) "
        f"improves {improvements_text}."
    )

    # Bar error bars: 95% page-cluster bootstrap CI. Only Layout Mode effort
    # is quantified; Read Mode fine-tuning effort is not included.

    return note
# def _layout_comparison_note(comparison_rows: list[dict], metric_key: str) -> str:
#     selected = [row for row in comparison_rows if row.get("metric_key") == metric_key]
#     lines = []
#     for row in selected:
#         effort = _safe_float(row.get("layout_effort_mean_seconds_per_page"))
#         effort_low = _safe_float(row.get("layout_effort_ci_lower"))
#         effort_high = _safe_float(row.get("layout_effort_ci_upper"))
#         reduction = _safe_float(row.get("relative_reduction_percent"))
#         reduction_low = _safe_float(row.get("relative_reduction_ci_lower"))
#         reduction_high = _safe_float(row.get("relative_reduction_ci_upper"))
#         if None in {effort, effort_low, effort_high, reduction, reduction_low, reduction_high}:
#             continue
#         lines.append(
#             f"{row['engine']} e2e → GT layout: "
#             f"{effort:.1f} s/page active layout editing (95% CI {effort_low:.1f}–{effort_high:.1f}); "
#             f"error ↓ {reduction:.1f}% (95% CI {reduction_low:.1f}–{reduction_high:.1f})"
#         )
#     if not lines:
#         return ""
#     lines.append(
#         "Bar error bars: 95% page-cluster bootstrap CI. Only Layout Mode effort is quantified; "
#         "Read Mode fine-tuning effort is not included."
#     )
#     return "\n".join(lines)


def _save_bar_figure(
    rows: list[dict],
    key: str,
    output_path: Path,
    *,
    title: str,
    ylabel: str,
    group_by_effort: bool = False,
    comparison_rows: list[dict] | None = None,
) -> Path | None:
    plt = _plotting()
    if plt is None or not rows:
        return None
    if group_by_effort:
        rows = sorted(rows, key=_effort_sort_key)
        labels = _figure_method_labels(rows)
        colors = ["#111111"] * len(rows)
        x_positions, group_spans, x_limits = _effort_plot_geometry(rows)
        bar_width = 0.72
    else:
        labels = _method_labels(rows)
        colors = ["#4C78A8"] * len(rows)
        x_positions = [float(index) for index in range(len(rows))]
        group_spans = []
        x_limits = (-0.5, len(rows) - 0.5)
        bar_width = 0.8
    values = _numeric_values(rows, key)
    ci_lower_values = [_safe_float(row.get(f"{key}_ci_lower")) for row in rows]
    ci_upper_values = [_safe_float(row.get(f"{key}_ci_upper")) for row in rows]
    lower_errors = [
        max(0.0, value - lower) if lower is not None else 0.0
        for value, lower in zip(values, ci_lower_values)
    ]
    upper_errors = [
        max(0.0, upper - value) if upper is not None else 0.0
        for value, upper in zip(values, ci_upper_values)
    ]
    plot_width = x_limits[1] - x_limits[0]
    fig_width = max(12.0, plot_width * 1.05) if group_by_effort else max(8.0, len(rows) * 1.35)
    fig, ax = plt.subplots(figsize=(fig_width, 6.0 if group_by_effort else 4.8))
    bars = ax.bar(
        x_positions,
        values,
        width=bar_width,
        color=colors,
        edgecolor="#111111",
        linewidth=0.5,
        yerr=[lower_errors, upper_errors],
        capsize=4,
        error_kw={"ecolor": "#7A1F1F", "elinewidth": 1.25, "capthick": 1.25},
    )
    if not group_by_effort:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        labels,
        rotation=0 if group_by_effort else 30,
        ha="center" if group_by_effort else "right",
    )
    if group_by_effort:
        ax.tick_params(axis="x", labelsize=9)
        ax.set_xlim(x_limits)
    ax.grid(axis="y", alpha=0.25)
    max_value = max(
        [
            upper if upper is not None else value
            for value, upper in zip(values, ci_upper_values)
        ],
        default=0.0,
    )
    ax.set_ylim(0, max(max_value * 1.18, 0.05))
    if not group_by_effort:
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    if group_by_effort:
        _add_effort_group_guides(ax, group_spans)
        note = _layout_comparison_note(comparison_rows or [], key)
        if note:
            fig.text(
                0.5,
                0.015,
                note,
                ha="center",
                va="bottom",
                fontsize=8.5,
                linespacing=1.25,
                bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "#777777", "alpha": 0.95},
            )
        fig.tight_layout(rect=(0, 0.16 if note else 0, 1, 0.84))
    else:
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
    series_specs = (
        (
            "Predicted test layout",
            "annotation_tool_e2e",
            "annotation_tool_pred_layout_ft_",
            "#4C78A8",
        ),
        (
            "Human-corrected test layout",
            "annotation_tool_gt_layout",
            "annotation_tool_gt_layout_ft_",
            "#F58518",
        ),
    )
    series = []
    for label, baseline_id, finetune_prefix, color in series_specs:
        curve_rows = [
            row
            for row in rows
            if row["method_id"] == baseline_id
            or row["method_id"].startswith(finetune_prefix)
        ]
        curve_rows.sort(key=lambda row: int(row.get("finetune_pages") or 0))
        if len(curve_rows) >= 2:
            series.append((label, color, curve_rows))
    if plt is None or not series:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharex=True)
    metric_specs = (
        ("micro_page_cer", "Micro Page CER"),
        ("micro_textedit", "Micro TextEdit"),
    )
    all_x_values: set[int] = set()
    for ax, (metric_key, metric_label) in zip(axes, metric_specs):
        for series_label, color, curve_rows in series:
            x_values = [int(row.get("finetune_pages") or 0) for row in curve_rows]
            all_x_values.update(x_values)
            ax.plot(
                x_values,
                _numeric_values(curve_rows, metric_key),
                marker="o",
                label=series_label,
                color=color,
            )
        ax.set_title(metric_label)
        ax.set_xlabel("Fine-tuning pages")
        ax.set_ylabel("Error")
        ax.grid(alpha=0.25)
    for ax in axes:
        ax.set_xticks(sorted(all_x_values))
    axes[0].legend(fontsize=8.5)
    fig.suptitle("Read Mode Fine-Tuning By Held-Out Layout Condition")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _save_vlm_token_figure(rows: list[dict], output_path: Path) -> Path | None:
    plt = _plotting()
    vlm_rows = [row for row in rows if is_vlm_method(str(row.get("method_id") or ""))]
    if plt is None or not vlm_rows:
        return None
    labels = _method_labels(vlm_rows)
    prompt = _numeric_values(vlm_rows, "vlm_input_token_count")
    candidates = _numeric_values(vlm_rows, "vlm_output_token_count")
    fig, ax = plt.subplots(figsize=(max(6.5, len(vlm_rows) * 1.7), 4.5))
    positions = list(range(len(vlm_rows)))
    ax.bar(positions, prompt, label="Prompt tokens", color="#4C78A8")
    ax.bar(positions, candidates, bottom=prompt, label="Candidate tokens", color="#F58518")
    ax.set_title("VLM Acquisition Token Usage")
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


def _write_figures(report_dir: Path, rows: list[dict], comparison_rows: list[dict]) -> list[Path]:
    figure_dir = report_dir / "figures"
    figure_paths = [
        _save_layout_figure(rows, figure_dir / "layout_metrics_by_method.png"),
        _save_finetuning_curve(rows, figure_dir / "annotation_tool_finetuning_curve.png"),
        _save_vlm_token_figure(rows, figure_dir / "vlm_token_usage.png"),
    ]
    return [path for path in figure_paths if path is not None]


def _summary_table_rows(rows: list[dict]) -> list[dict]:
    table_rows = []
    for row in rows:
        table_rows.append(
            {
                "Method": row["display_name"],
                "Human Effort": row["human_effort"],
                "Training Layout": row["training_layout_condition"],
                "Test Layout": row["test_layout_condition"],
                "Pages": row["page_count"],
                "Valid Output": _format_float(row.get("valid_output_rate")),
                "G-F1@0.50": _format_float(row.get("object_g_f1_50")),
                "Pixel F1": _format_float(row.get("pixel_f1")),
                "Micro CER (95% CI)": _format_estimate_ci(
                    row.get("micro_page_cer"),
                    row.get("micro_page_cer_ci_lower"),
                    row.get("micro_page_cer_ci_upper"),
                ),
                "Micro TextEdit (95% CI)": _format_estimate_ci(
                    row.get("micro_textedit"),
                    row.get("micro_textedit_ci_lower"),
                    row.get("micro_textedit_ci_upper"),
                ),
                "API Tokens": row.get("vlm_total_token_count", 0),
                "API USD": _format_cost(row.get("vlm_estimated_cost_usd")),
            }
        )
    return table_rows


def _layout_mode_comparison_table_rows(rows: list[dict]) -> list[dict]:
    table_rows = []
    for row in rows:
        table_rows.append(
            {
                "Comparison": f"{row['engine']} e2e → GT layout",
                "Metric": row["metric_label"],
                "Pages": row["unique_page_count"],
                "Active Layout Edit Seconds/Page (95% CI)": _format_estimate_ci(
                    row.get("layout_effort_mean_seconds_per_page"),
                    row.get("layout_effort_ci_lower"),
                    row.get("layout_effort_ci_upper"),
                    digits=1,
                ),
                "Baseline": _format_float(row.get("baseline_micro")),
                "GT Layout": _format_float(row.get("target_micro")),
                "Relative Error Reduction (95% CI)": _format_estimate_ci(
                    row.get("relative_reduction_percent"),
                    row.get("relative_reduction_ci_lower"),
                    row.get("relative_reduction_ci_upper"),
                    digits=1,
                )
                + "%",
            }
        )
    return table_rows


def _format_percent(value: Any, digits: int = 1) -> str:
    number = _safe_float(value)
    if number is None:
        return ""
    return f"{number:.{digits}f}%"


def _format_seconds_ci(value: Any, lower: Any, upper: Any) -> str:
    return _format_estimate_ci(value, lower, upper, digits=1)


def _metric_ci_text(row: dict | None, key: str, digits: int = 4) -> str:
    if row is None:
        return ""
    return _format_estimate_ci(
        row.get(key),
        row.get(f"{key}_ci_lower"),
        row.get(f"{key}_ci_upper"),
        digits=digits,
    )


def _safe_metric_reduction(baseline: Any, target: Any) -> tuple[float | None, float | None]:
    baseline_value = _safe_float(baseline)
    target_value = _safe_float(target)
    if baseline_value is None or target_value is None:
        return None, None
    absolute = baseline_value - target_value
    relative = (absolute / baseline_value) * 100.0 if baseline_value > 0 else 0.0
    return absolute, relative


def _summary_lookup(summary_rows: list[dict]) -> dict[tuple[str, str], dict]:
    lookup: dict[tuple[str, str], dict] = {}
    for row in summary_rows:
        lookup[(str(row.get("manuscript_id") or ""), str(row.get("method_id") or ""))] = row
    return lookup


def _manuscript_sort_key(manuscript_id: str) -> tuple[int, str]:
    preferred = ("yajn", "dense", "circle_10", "circle_new")
    try:
        return (preferred.index(manuscript_id), manuscript_id)
    except ValueError:
        return (len(preferred), manuscript_id)


def _build_off_the_shelf_table_rows(summary_rows: list[dict]) -> list[dict]:
    rows = []
    for row in summary_rows:
        method_id = str(row.get("method_id") or "")
        metadata = OFF_THE_SHELF_METHODS.get(method_id)
        if metadata is None:
            continue
        rows.append(
            {
                "manuscript_id": str(row.get("manuscript_id") or ""),
                "provider": metadata["provider"],
                "model_family": metadata["model_family"],
                "method_id": method_id,
                "input_contract": metadata["input_contract"],
                "prompt_contract": metadata["prompt_contract"],
                "test_set_contract": "same folds/test pages as the downstream OCR run",
                "page_count": row.get("page_count"),
                "valid_output_rate": row.get("valid_output_rate"),
                "micro_page_cer": row.get("micro_page_cer"),
                "micro_page_cer_ci_lower": row.get("micro_page_cer_ci_lower"),
                "micro_page_cer_ci_upper": row.get("micro_page_cer_ci_upper"),
                "micro_textedit": row.get("micro_textedit"),
                "micro_textedit_ci_lower": row.get("micro_textedit_ci_lower"),
                "micro_textedit_ci_upper": row.get("micro_textedit_ci_upper"),
                "vlm_request_count": row.get("vlm_request_count"),
                "vlm_total_token_count": row.get("vlm_total_token_count"),
                "metrics_path": row.get("metrics_path", ""),
            }
        )
    return sorted(rows, key=lambda item: (_manuscript_sort_key(item["manuscript_id"]), item["provider"], item["method_id"]))


def _off_the_shelf_markdown_rows(rows: list[dict]) -> list[dict]:
    return [
        {
            "Manuscript": row["manuscript_id"],
            "Provider": row["provider"],
            "Model Family": row["model_family"],
            "Input": row["input_contract"],
            "Prompt/Input Contract": row["prompt_contract"],
            "Pages": row.get("page_count", ""),
            "Valid Output": _format_float(row.get("valid_output_rate")),
            "Micro CER (95% CI)": _format_estimate_ci(
                row.get("micro_page_cer"),
                row.get("micro_page_cer_ci_lower"),
                row.get("micro_page_cer_ci_upper"),
            ),
            "Micro TextEdit (95% CI)": _format_estimate_ci(
                row.get("micro_textedit"),
                row.get("micro_textedit_ci_lower"),
                row.get("micro_textedit_ci_upper"),
            ),
        }
        for row in rows
    ]


def _per_page_rows_by_method(per_page_rows: list[dict]) -> dict[tuple[str, str], list[dict]]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in per_page_rows:
        grouped.setdefault(
            (str(row.get("manuscript_id") or ""), str(row.get("method_id") or "")),
            [],
        ).append(row)
    return grouped


def _layout_effort_for_table(
    per_page_by_method: dict[tuple[str, str], list[dict]],
    *,
    manuscript_id: str,
    gt_method_id: str,
) -> dict | None:
    rows = per_page_by_method.get((manuscript_id, gt_method_id), [])
    return _bootstrap_mean_effort(
        rows,
        seed_label=f"annotation-gain-layout-effort:{manuscript_id}:{gt_method_id}",
    )


def _build_annotation_gains_table_rows(summary_rows: list[dict], per_page_rows: list[dict]) -> list[dict]:
    lookup = _summary_lookup(summary_rows)
    per_page_by_method = _per_page_rows_by_method(per_page_rows)
    manuscript_ids = sorted(
        {
            manuscript_id
            for manuscript_id, method_id in lookup
            if any(method_id in pair for pair in ANNOTATION_GAIN_METHOD_PAIRS.values())
        },
        key=_manuscript_sort_key,
    )
    rows = []
    for manuscript_id in manuscript_ids:
        for finetune_pages, (pred_method_id, gt_method_id) in ANNOTATION_GAIN_METHOD_PAIRS.items():
            pred_row = lookup.get((manuscript_id, pred_method_id))
            gt_row = lookup.get((manuscript_id, gt_method_id))
            page_abs, page_rel = _safe_metric_reduction(
                pred_row.get("micro_page_cer") if pred_row else None,
                gt_row.get("micro_page_cer") if gt_row else None,
            )
            text_abs, text_rel = _safe_metric_reduction(
                pred_row.get("micro_textedit") if pred_row else None,
                gt_row.get("micro_textedit") if gt_row else None,
            )
            effort = _layout_effort_for_table(
                per_page_by_method,
                manuscript_id=manuscript_id,
                gt_method_id=gt_method_id,
            )
            rows.append(
                {
                    "manuscript_id": manuscript_id,
                    "finetune_pages": finetune_pages,
                    "without_gt_layout_method_id": pred_method_id,
                    "with_gt_layout_method_id": gt_method_id,
                    "without_gt_layout_pages": pred_row.get("page_count") if pred_row else None,
                    "with_gt_layout_pages": gt_row.get("page_count") if gt_row else None,
                    "without_gt_layout_valid_output_rate": pred_row.get("valid_output_rate") if pred_row else None,
                    "with_gt_layout_valid_output_rate": gt_row.get("valid_output_rate") if gt_row else None,
                    "without_gt_layout_micro_page_cer": pred_row.get("micro_page_cer") if pred_row else None,
                    "without_gt_layout_micro_page_cer_ci_lower": pred_row.get("micro_page_cer_ci_lower") if pred_row else None,
                    "without_gt_layout_micro_page_cer_ci_upper": pred_row.get("micro_page_cer_ci_upper") if pred_row else None,
                    "with_gt_layout_micro_page_cer": gt_row.get("micro_page_cer") if gt_row else None,
                    "with_gt_layout_micro_page_cer_ci_lower": gt_row.get("micro_page_cer_ci_lower") if gt_row else None,
                    "with_gt_layout_micro_page_cer_ci_upper": gt_row.get("micro_page_cer_ci_upper") if gt_row else None,
                    "page_cer_absolute_reduction": page_abs,
                    "page_cer_relative_reduction_percent": page_rel,
                    "without_gt_layout_micro_textedit": pred_row.get("micro_textedit") if pred_row else None,
                    "without_gt_layout_micro_textedit_ci_lower": pred_row.get("micro_textedit_ci_lower") if pred_row else None,
                    "without_gt_layout_micro_textedit_ci_upper": pred_row.get("micro_textedit_ci_upper") if pred_row else None,
                    "with_gt_layout_micro_textedit": gt_row.get("micro_textedit") if gt_row else None,
                    "with_gt_layout_micro_textedit_ci_lower": gt_row.get("micro_textedit_ci_lower") if gt_row else None,
                    "with_gt_layout_micro_textedit_ci_upper": gt_row.get("micro_textedit_ci_upper") if gt_row else None,
                    "textedit_absolute_reduction": text_abs,
                    "textedit_relative_reduction_percent": text_rel,
                    "layout_effort_mean_seconds_per_page": effort["mean_seconds_per_page"] if effort else None,
                    "layout_effort_ci_lower": effort["ci_lower"] if effort else None,
                    "layout_effort_ci_upper": effort["ci_upper"] if effort else None,
                    "layout_effort_unique_page_count": effort["unique_page_count"] if effort else 0,
                }
            )
    return rows


def _annotation_gains_caption(rows: list[dict]) -> str:
    effort_by_manuscript = {}
    for row in rows:
        manuscript_id = str(row.get("manuscript_id") or "")
        if not manuscript_id or manuscript_id in effort_by_manuscript:
            continue
        effort = _safe_float(row.get("layout_effort_mean_seconds_per_page"))
        if effort is None:
            continue
        effort_by_manuscript[manuscript_id] = _format_seconds_ci(
            effort,
            row.get("layout_effort_ci_lower"),
            row.get("layout_effort_ci_upper"),
        )
    if not effort_by_manuscript:
        return "GT layout-correction timing unavailable for these runs."
    parts = [
        f"{manuscript_id}: {effort_by_manuscript[manuscript_id]} s/page"
        for manuscript_id in sorted(effort_by_manuscript, key=_manuscript_sort_key)
    ]
    return "GT layout-correction time, active Layout Mode seconds per evaluated page: " + "; ".join(parts) + "."


def _annotation_gains_markdown_rows(rows: list[dict]) -> list[dict]:
    formatted = []
    for row in rows:
        pred_metric_row = {
            "micro_page_cer": row.get("without_gt_layout_micro_page_cer"),
            "micro_page_cer_ci_lower": row.get("without_gt_layout_micro_page_cer_ci_lower"),
            "micro_page_cer_ci_upper": row.get("without_gt_layout_micro_page_cer_ci_upper"),
            "micro_textedit": row.get("without_gt_layout_micro_textedit"),
            "micro_textedit_ci_lower": row.get("without_gt_layout_micro_textedit_ci_lower"),
            "micro_textedit_ci_upper": row.get("without_gt_layout_micro_textedit_ci_upper"),
        }
        gt_metric_row = {
            "micro_page_cer": row.get("with_gt_layout_micro_page_cer"),
            "micro_page_cer_ci_lower": row.get("with_gt_layout_micro_page_cer_ci_lower"),
            "micro_page_cer_ci_upper": row.get("with_gt_layout_micro_page_cer_ci_upper"),
            "micro_textedit": row.get("with_gt_layout_micro_textedit"),
            "micro_textedit_ci_lower": row.get("with_gt_layout_micro_textedit_ci_lower"),
            "micro_textedit_ci_upper": row.get("with_gt_layout_micro_textedit_ci_upper"),
        }
        formatted.append(
            {
                "Manuscript": row["manuscript_id"],
                "Fine-tuning Pages": row["finetune_pages"],
                "Without GT Layout CER": _metric_ci_text(pred_metric_row, "micro_page_cer"),
                "With GT Layout CER": _metric_ci_text(gt_metric_row, "micro_page_cer"),
                "CER Reduction": _format_percent(row.get("page_cer_relative_reduction_percent")),
                "Without GT Layout TextEdit": _metric_ci_text(pred_metric_row, "micro_textedit"),
                "With GT Layout TextEdit": _metric_ci_text(gt_metric_row, "micro_textedit"),
                "TextEdit Reduction": _format_percent(row.get("textedit_relative_reduction_percent")),
                "GT Layout Time": _format_seconds_ci(
                    row.get("layout_effort_mean_seconds_per_page"),
                    row.get("layout_effort_ci_lower"),
                    row.get("layout_effort_ci_upper"),
                ),
            }
        )
    return formatted


def _write_primary_table_artifacts(
    report_dir: Path,
    summary_rows: list[dict],
    per_page_rows: list[dict],
) -> dict:
    off_the_shelf_rows = _build_off_the_shelf_table_rows(summary_rows)
    annotation_gains_rows = _build_annotation_gains_table_rows(summary_rows, per_page_rows)
    annotation_caption = _annotation_gains_caption(annotation_gains_rows)

    off_csv = report_dir / "table_1_off_the_shelf_models.csv"
    off_json = report_dir / "table_1_off_the_shelf_models.json"
    gains_csv = report_dir / "table_2_annotation_tool_gains.csv"
    gains_json = report_dir / "table_2_annotation_tool_gains.json"

    _write_csv(off_csv, off_the_shelf_rows)
    _write_json(
        off_json,
        {
            "caption": "Off-the-shelf models evaluated on the same held-out PAGE-XML folds used by all methods in each manuscript run.",
            "rows": off_the_shelf_rows,
        },
    )
    _write_csv(gains_csv, annotation_gains_rows)
    _write_json(
        gains_json,
        {
            "caption": annotation_caption,
            "rows": annotation_gains_rows,
        },
    )
    return {
        "off_the_shelf_rows": off_the_shelf_rows,
        "annotation_gains_rows": annotation_gains_rows,
        "annotation_gains_caption": annotation_caption,
        "off_the_shelf_table_csv_path": off_csv,
        "off_the_shelf_table_json_path": off_json,
        "annotation_gains_table_csv_path": gains_csv,
        "annotation_gains_table_json_path": gains_json,
    }


def _write_markdown_report(
    output_root: Path,
    report_dir: Path,
    rows: list[dict],
    comparison_rows: list[dict],
    usage_rows: list[dict],
    figure_paths: list[Path],
    table_artifacts: dict,
) -> Path:
    report_path = report_dir / "experiment_report.md"
    generated_at = datetime.now(timezone.utc).isoformat()
    table_rows = _summary_table_rows(rows)
    table_columns = [
        ("Method", "Method"),
        ("Human Effort", "Human Effort"),
        ("Training Layout", "Training Layout"),
        ("Test Layout", "Test Layout"),
        ("Pages", "Pages"),
        ("Valid Output", "Valid Output"),
        ("G-F1@0.50", "G-F1@0.50"),
        ("Pixel F1", "Pixel F1"),
        ("Micro CER (95% CI)", "Micro CER (95% CI)"),
        ("Micro TextEdit (95% CI)", "Micro TextEdit (95% CI)"),
        ("API Tokens", "API Tokens"),
        ("API USD", "API USD"),
    ]

    vlm_rows = [row for row in rows if is_vlm_method(str(row.get("method_id") or ""))]
    vlm_table_rows = [
        {
            "Method": row["display_name"],
            "Pages": row.get("vlm_page_count", 0),
            "Attempts": row.get("vlm_attempt_count", 0),
            "Retries": row.get("vlm_retry_count", 0),
            "Requests": row.get("vlm_request_count", 0),
            "Missing Usage": row.get("vlm_missing_usage_count", 0),
            "Input Tokens": row.get("vlm_input_token_count", 0),
            "Output Tokens": row.get("vlm_output_token_count", 0),
            "Total Tokens": row.get("vlm_total_token_count", 0),
            "Elapsed Seconds": _format_float(row.get("vlm_elapsed_seconds"), digits=2),
            "Estimated USD": _format_cost(row.get("vlm_estimated_cost_usd")),
            "Usage Note": row.get("vlm_pricing_note", ""),
        }
        for row in vlm_rows
    ]
    comparison_table_rows = _layout_mode_comparison_table_rows(comparison_rows)
    off_the_shelf_table_rows = _off_the_shelf_markdown_rows(table_artifacts["off_the_shelf_rows"])
    annotation_gains_table_rows = _annotation_gains_markdown_rows(table_artifacts["annotation_gains_rows"])
    annotation_gains_caption = table_artifacts["annotation_gains_caption"]

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
        "## Table 1: Off-The-Shelf Models",
        "",
        "All off-the-shelf model rows use the same held-out folds as the annotation-tool rows for the same manuscript. The table is method-registry driven so future providers can be added with the same prompt/input/test-set contract.",
        "",
        _markdown_table(
            off_the_shelf_table_rows,
            [
                ("Manuscript", "Manuscript"),
                ("Provider", "Provider"),
                ("Model Family", "Model Family"),
                ("Input", "Input"),
                ("Prompt/Input Contract", "Prompt/Input Contract"),
                ("Pages", "Pages"),
                ("Valid Output", "Valid Output"),
                ("Micro CER (95% CI)", "Micro CER (95% CI)"),
                ("Micro TextEdit (95% CI)", "Micro TextEdit (95% CI)"),
            ],
        )
        if off_the_shelf_table_rows
        else "No off-the-shelf model rows found.",
        "",
        f"Machine-readable rows: `{(report_dir / 'table_1_off_the_shelf_models.csv').relative_to(report_dir).as_posix()}`",
        "",
        "## Table 2: Annotation Tool Gains",
        "",
        annotation_gains_caption,
        "",
        _markdown_table(
            annotation_gains_table_rows,
            [
                ("Manuscript", "Manuscript"),
                ("Fine-tuning Pages", "Fine-tuning Pages"),
                ("Without GT Layout CER", "Without GT Layout CER"),
                ("With GT Layout CER", "With GT Layout CER"),
                ("CER Reduction", "CER Reduction"),
                ("Without GT Layout TextEdit", "Without GT Layout TextEdit"),
                ("With GT Layout TextEdit", "With GT Layout TextEdit"),
                ("TextEdit Reduction", "TextEdit Reduction"),
                ("GT Layout Time", "GT Layout Time"),
            ],
        )
        if annotation_gains_table_rows
        else "No complete annotation-tool gain rows found.",
        "",
        f"Machine-readable rows: `{(report_dir / 'table_2_annotation_tool_gains.csv').relative_to(report_dir).as_posix()}`",
        "",
        "## Headline Metrics",
        "",
        _markdown_table(table_rows, table_columns) if table_rows else "No metric rows found.",
        "",
        "## Interpretation Guide",
        "",
        "- Compare `annotation_tool_e2e` against `annotation_tool_gt_layout` to estimate the practical value of human layout correction for the local OCR pipeline.",
        "- Compare `annotation_tool_e2e` with the `annotation_tool_pred_layout_ft_1/2/3` series to estimate the value of manuscript-local Read Mode supervision when held-out layouts remain fully automatic.",
        "- Compare `annotation_tool_gt_layout` with the `annotation_tool_gt_layout_ft_1/2/3` series to estimate the value of the same Read Mode supervision when held-out layouts are human-corrected.",
        "- At each fine-tuning page count, compare `annotation_tool_pred_layout_ft_N` against `annotation_tool_gt_layout_ft_N`. Both rows use the same checkpoint trained from corrected training-page layout and Unicode text; only held-out layout correction differs.",
        "- VLM rows are end-to-end off-the-shelf acquisitions; layout-corrected VLM variants remain disabled.",
        "- Rows with `test_layout_condition=human_corrected_gt_layout` use held-out layout obtained through careful human inspection and correction. Their G-F1 and pixel F1 scores describe the provided human-corrected layout condition, not automatic layout-detector performance.",
        "- Fine-tuning methods record the GUI runtime OCR active-learning recipe and sibling checkpoint selector in `summary_metrics.csv`.",
        "",
        "## Layout Mode Effort And OCR Reduction",
        "",
        (
            "These paired comparisons quantify only active Layout Mode editing effort. "
            "Read Mode fine-tuning effort is intentionally not included. Intervals are deterministic "
            "95% page-cluster bootstrap confidence intervals; repeated occurrences of a page across "
            "folds remain in the same resampled page cluster."
        ),
        "",
        _markdown_table(
            comparison_table_rows,
            [
                ("Comparison", "Comparison"),
                ("Metric", "Metric"),
                ("Pages", "Pages"),
                ("Active Layout Edit Seconds/Page (95% CI)", "Active Layout Edit Seconds/Page (95% CI)"),
                ("Baseline", "Baseline"),
                ("GT Layout", "GT Layout"),
                ("Relative Error Reduction (95% CI)", "Relative Error Reduction (95% CI)"),
            ],
        )
        if comparison_table_rows
        else "No complete paired e2e-versus-GT-layout comparisons with layout-effort data were found.",
        "",
        f"Paired layout comparison rows: `{(report_dir / 'layout_mode_comparisons.csv').relative_to(report_dir).as_posix()}`",
        f"Per-fold metrics: `{(report_dir / 'fold_metrics.csv').relative_to(report_dir).as_posix()}`",
        "",
        "## VLM Acquisition Usage And Cost",
        "",
        "Each cached provider/page acquisition is counted exactly once, independent of how many folds reuse it. Annotation-tool methods use local computation and have zero VLM API cost. USD estimates require `<PROVIDER>_INPUT_USD_PER_1M_TOKENS` and `<PROVIDER>_OUTPUT_USD_PER_1M_TOKENS`.",
        "",
        _markdown_table(
            vlm_table_rows,
            [
                ("Method", "Method"),
                ("Pages", "Pages"),
                ("Attempts", "Attempts"),
                ("Retries", "Retries"),
                ("Requests", "Requests"),
                ("Missing Usage", "Missing Usage"),
                ("Input Tokens", "Input Tokens"),
                ("Output Tokens", "Output Tokens"),
                ("Total Tokens", "Total Tokens"),
                ("Elapsed Seconds", "Elapsed Seconds"),
                ("Estimated USD", "Estimated USD"),
                ("Usage Note", "Usage Note"),
            ],
        )
        if vlm_table_rows
        else "No VLM acquisition usage rows found.",
        "",
        f"Per-page VLM acquisition rows: `{(report_dir / 'vlm_usage.csv').relative_to(report_dir).as_posix()}`",
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
    _augment_rows_with_bootstrap_cis(summary_rows, per_page_rows)
    fold_metric_rows = _fold_metric_rows(per_page_rows)
    layout_mode_comparison_rows = _layout_mode_comparison_rows(per_page_rows)
    usage_rows, usage_summaries = collect_vlm_usage(
        root,
        input_usd_per_1m_tokens=input_usd_per_1m_tokens,
        output_usd_per_1m_tokens=output_usd_per_1m_tokens,
    )
    _augment_rows_with_usage(summary_rows, usage_summaries)

    summary_csv_path = report_dir / "summary_metrics.csv"
    summary_json_path = report_dir / "summary_metrics.json"
    per_page_csv_path = report_dir / "per_page_metrics.csv"
    fold_metrics_csv_path = report_dir / "fold_metrics.csv"
    fold_metrics_json_path = report_dir / "fold_metrics.json"
    layout_mode_comparisons_csv_path = report_dir / "layout_mode_comparisons.csv"
    layout_mode_comparisons_json_path = report_dir / "layout_mode_comparisons.json"
    vlm_usage_csv_path = report_dir / "vlm_usage.csv"
    vlm_usage_json_path = report_dir / "vlm_usage.json"

    summary_fields = [
        "manuscript_id",
        "method_id",
        "display_name",
        "engine",
        "human_effort",
        "human_training_layout",
        "human_test_layout",
        "human_layout",
        "training_layout_condition",
        "test_layout_condition",
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
        "micro_page_cer_ci_lower",
        "micro_page_cer_ci_upper",
        "micro_page_cer_bootstrap_unique_pages",
        "mean_textedit",
        "median_textedit",
        "micro_textedit",
        "micro_textedit_ci_lower",
        "micro_textedit_ci_upper",
        "micro_textedit_bootstrap_unique_pages",
        "vlm_usage_status",
        "vlm_page_count",
        "vlm_success_count",
        "vlm_attempt_count",
        "vlm_retry_count",
        "vlm_request_count",
        "vlm_missing_usage_count",
        "vlm_elapsed_seconds",
        "vlm_input_token_count",
        "vlm_output_token_count",
        "vlm_total_token_count",
        "vlm_estimated_cost_usd",
        "vlm_pricing_note",
        "metrics_path",
    ]
    _write_csv(summary_csv_path, summary_rows, fieldnames=summary_fields)
    _write_json(summary_json_path, summary_rows)
    _write_csv(per_page_csv_path, per_page_rows)
    _write_csv(fold_metrics_csv_path, fold_metric_rows)
    _write_json(
        fold_metrics_json_path,
        {
            "aggregation": "pooled independently within each method and fold",
            "rows": fold_metric_rows,
        },
    )
    _write_csv(layout_mode_comparisons_csv_path, layout_mode_comparison_rows)
    _write_json(
        layout_mode_comparisons_json_path,
        {
            "confidence_level": BOOTSTRAP_CONFIDENCE_LEVEL,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "cluster_unit": "manuscript_id+page_id",
            "effort_definition": "mean active Layout Mode edit seconds per unique evaluated page",
            "read_mode_effort_included": False,
            "rows": layout_mode_comparison_rows,
        },
    )
    _write_csv(vlm_usage_csv_path, usage_rows)
    _write_json(vlm_usage_json_path, {"rows": usage_rows, "summaries": usage_summaries})

    stale_paths = (
        report_dir / "layout_effort_impact.csv",
        report_dir / "layout_effort_impact.json",
        report_dir / "figures" / "layout_effort_vs_ocr_gain.png",
        report_dir / "figures" / "micro_page_cer_by_method.png",
        report_dir / "figures" / "micro_textedit_by_method.png",
    )
    for stale_path in stale_paths:
        stale_path.unlink(missing_ok=True)

    table_artifacts = _write_primary_table_artifacts(report_dir, summary_rows, per_page_rows)
    figure_paths = _write_figures(report_dir, summary_rows, layout_mode_comparison_rows)
    markdown_path = _write_markdown_report(
        root,
        report_dir,
        summary_rows,
        layout_mode_comparison_rows,
        usage_rows,
        figure_paths,
        table_artifacts,
    )
    manifest_path = report_dir / "report_manifest.json"
    manifest = {
        "report_dir": str(report_dir.resolve()),
        "markdown_path": str(markdown_path.resolve()),
        "summary_csv_path": str(summary_csv_path.resolve()),
        "summary_json_path": str(summary_json_path.resolve()),
        "per_page_csv_path": str(per_page_csv_path.resolve()),
        "fold_metrics_csv_path": str(fold_metrics_csv_path.resolve()),
        "fold_metrics_json_path": str(fold_metrics_json_path.resolve()),
        "layout_mode_comparisons_csv_path": str(layout_mode_comparisons_csv_path.resolve()),
        "layout_mode_comparisons_json_path": str(layout_mode_comparisons_json_path.resolve()),
        "off_the_shelf_table_csv_path": str(table_artifacts["off_the_shelf_table_csv_path"].resolve()),
        "off_the_shelf_table_json_path": str(table_artifacts["off_the_shelf_table_json_path"].resolve()),
        "annotation_gains_table_csv_path": str(table_artifacts["annotation_gains_table_csv_path"].resolve()),
        "annotation_gains_table_json_path": str(table_artifacts["annotation_gains_table_json_path"].resolve()),
        "vlm_usage_csv_path": str(vlm_usage_csv_path.resolve()),
        "vlm_usage_json_path": str(vlm_usage_json_path.resolve()),
        "figure_paths": [str(path.resolve()) for path in figure_paths],
        "method_count": len(summary_rows),
        "per_page_record_count": len(per_page_rows),
        "fold_metric_record_count": len(fold_metric_rows),
        "layout_mode_comparison_record_count": len(layout_mode_comparison_rows),
        "vlm_usage_record_count": len(usage_rows),
    }
    _write_json(manifest_path, manifest)

    return ReportArtifacts(
        report_dir=report_dir,
        markdown_path=markdown_path,
        summary_csv_path=summary_csv_path,
        summary_json_path=summary_json_path,
        per_page_csv_path=per_page_csv_path,
        fold_metrics_csv_path=fold_metrics_csv_path,
        fold_metrics_json_path=fold_metrics_json_path,
        layout_mode_comparisons_csv_path=layout_mode_comparisons_csv_path,
        layout_mode_comparisons_json_path=layout_mode_comparisons_json_path,
        off_the_shelf_table_csv_path=table_artifacts["off_the_shelf_table_csv_path"],
        off_the_shelf_table_json_path=table_artifacts["off_the_shelf_table_json_path"],
        annotation_gains_table_csv_path=table_artifacts["annotation_gains_table_csv_path"],
        annotation_gains_table_json_path=table_artifacts["annotation_gains_table_json_path"],
        vlm_usage_csv_path=vlm_usage_csv_path,
        vlm_usage_json_path=vlm_usage_json_path,
        figure_paths=tuple(figure_paths),
        manifest_path=manifest_path,
    )


def write_combined_table_report(
    input_roots: Iterable[str | Path],
    output_root: str | Path,
    *,
    input_usd_per_1m_tokens: float | None = None,
    output_usd_per_1m_tokens: float | None = None,
) -> CombinedTableArtifacts:
    roots = [Path(root) for root in input_roots]
    if not roots:
        raise ValueError("At least one input root is required for a combined table report.")

    root = Path(output_root)
    report_dir = root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    per_page_rows: list[dict] = []
    usage_rows: list[dict] = []
    input_root_payloads = []
    for input_root in roots:
        run_summary_rows, run_per_page_rows = _load_summary_rows(input_root)
        _augment_rows_with_bootstrap_cis(run_summary_rows, run_per_page_rows)
        run_usage_rows, run_usage_summaries = collect_vlm_usage(
            input_root,
            input_usd_per_1m_tokens=input_usd_per_1m_tokens,
            output_usd_per_1m_tokens=output_usd_per_1m_tokens,
        )
        _augment_rows_with_usage(run_summary_rows, run_usage_summaries)
        for row in run_summary_rows:
            row["source_output_root"] = str(input_root.resolve())
        for row in run_per_page_rows:
            row["source_output_root"] = str(input_root.resolve())
        for row in run_usage_rows:
            row["source_output_root"] = str(input_root.resolve())
        summary_rows.extend(run_summary_rows)
        per_page_rows.extend(run_per_page_rows)
        usage_rows.extend(run_usage_rows)
        input_root_payloads.append(
            {
                "input_root": str(input_root.resolve()),
                "method_count": len(run_summary_rows),
                "per_page_record_count": len(run_per_page_rows),
                "vlm_usage_record_count": len(run_usage_rows),
            }
        )

    summary_rows.sort(key=lambda row: (_manuscript_sort_key(str(row.get("manuscript_id") or "")), _method_sort_key(row["method_id"])))
    per_page_rows.sort(
        key=lambda row: (
            _manuscript_sort_key(str(row.get("manuscript_id") or "")),
            _method_sort_key(str(row.get("method_id") or "")),
            row.get("fold_id", ""),
            row.get("page_id", ""),
        )
    )

    summary_csv_path = report_dir / "summary_metrics.csv"
    summary_json_path = report_dir / "summary_metrics.json"
    per_page_csv_path = report_dir / "per_page_metrics.csv"
    fold_metrics_csv_path = report_dir / "fold_metrics.csv"
    fold_metrics_json_path = report_dir / "fold_metrics.json"
    layout_mode_comparisons_csv_path = report_dir / "layout_mode_comparisons.csv"
    layout_mode_comparisons_json_path = report_dir / "layout_mode_comparisons.json"
    vlm_usage_csv_path = report_dir / "vlm_usage.csv"
    vlm_usage_json_path = report_dir / "vlm_usage.json"
    _write_csv(summary_csv_path, summary_rows)
    _write_json(summary_json_path, summary_rows)
    _write_csv(per_page_csv_path, per_page_rows)
    fold_metric_rows = _fold_metric_rows(per_page_rows)
    comparison_rows = _layout_mode_comparison_rows(per_page_rows)
    _write_csv(fold_metrics_csv_path, fold_metric_rows)
    _write_json(
        fold_metrics_json_path,
        {
            "aggregation": "pooled independently within each method and fold across combined input roots",
            "rows": fold_metric_rows,
        },
    )
    _write_csv(layout_mode_comparisons_csv_path, comparison_rows)
    _write_json(
        layout_mode_comparisons_json_path,
        {
            "confidence_level": BOOTSTRAP_CONFIDENCE_LEVEL,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "cluster_unit": "manuscript_id+page_id",
            "effort_definition": "mean active Layout Mode edit seconds per unique evaluated page",
            "read_mode_effort_included": False,
            "rows": comparison_rows,
        },
    )
    _write_csv(vlm_usage_csv_path, usage_rows)
    _write_json(vlm_usage_json_path, {"rows": usage_rows})

    table_artifacts = _write_primary_table_artifacts(report_dir, summary_rows, per_page_rows)
    markdown_path = _write_markdown_report(
        root,
        report_dir,
        summary_rows,
        comparison_rows,
        usage_rows,
        [],
        table_artifacts,
    )

    manifest_path = report_dir / "combined_table_report_manifest.json"
    _write_json(
        manifest_path,
        {
            "report_dir": str(report_dir.resolve()),
            "markdown_path": str(markdown_path.resolve()),
            "summary_csv_path": str(summary_csv_path.resolve()),
            "summary_json_path": str(summary_json_path.resolve()),
            "per_page_csv_path": str(per_page_csv_path.resolve()),
            "fold_metrics_csv_path": str(fold_metrics_csv_path.resolve()),
            "fold_metrics_json_path": str(fold_metrics_json_path.resolve()),
            "layout_mode_comparisons_csv_path": str(layout_mode_comparisons_csv_path.resolve()),
            "layout_mode_comparisons_json_path": str(layout_mode_comparisons_json_path.resolve()),
            "vlm_usage_csv_path": str(vlm_usage_csv_path.resolve()),
            "vlm_usage_json_path": str(vlm_usage_json_path.resolve()),
            "off_the_shelf_table_csv_path": str(table_artifacts["off_the_shelf_table_csv_path"].resolve()),
            "off_the_shelf_table_json_path": str(table_artifacts["off_the_shelf_table_json_path"].resolve()),
            "annotation_gains_table_csv_path": str(table_artifacts["annotation_gains_table_csv_path"].resolve()),
            "annotation_gains_table_json_path": str(table_artifacts["annotation_gains_table_json_path"].resolve()),
            "input_roots": input_root_payloads,
            "method_count": len(summary_rows),
            "per_page_record_count": len(per_page_rows),
        },
    )

    return CombinedTableArtifacts(
        report_dir=report_dir,
        markdown_path=markdown_path,
        off_the_shelf_table_csv_path=table_artifacts["off_the_shelf_table_csv_path"],
        off_the_shelf_table_json_path=table_artifacts["off_the_shelf_table_json_path"],
        annotation_gains_table_csv_path=table_artifacts["annotation_gains_table_csv_path"],
        annotation_gains_table_json_path=table_artifacts["annotation_gains_table_json_path"],
        summary_csv_path=summary_csv_path,
        summary_json_path=summary_json_path,
        per_page_csv_path=per_page_csv_path,
        fold_metrics_csv_path=fold_metrics_csv_path,
        layout_mode_comparisons_csv_path=layout_mode_comparisons_csv_path,
        vlm_usage_csv_path=vlm_usage_csv_path,
        vlm_usage_json_path=vlm_usage_json_path,
        manifest_path=manifest_path,
    )
