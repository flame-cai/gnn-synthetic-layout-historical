from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

STRATEGY_ROLE_CONFIG_JSON = r"""{
  "benchmark_strategy_name": "local_polygons_stable_unwrap_v1",
  "proposed_strategy_name": null,
  "production_strategy_name": "local_polygons_stable_unwrap_v1",
  "research_promotion_history": [
    {
      "promoted_strategy_name": "local_tangent_band_v1",
      "previous_benchmark_strategy_name": "legacy_axis_bound_v1",
      "evidence_metrics_path": "C:\\Users\\intro\\OneDrive\\Documents\\MEGA\\CAI-FLAME\\gnn-synthetic-layout-historical\\app\\tests\\logs\\strategy_promotion_latest.json",
      "evidence_generated_at_utc": "2026-05-16T07:14:45Z",
      "gate_artifact_paths": {
        "pipeline_eval_dataset": {
          "latest_metrics_json": "C:\\Users\\intro\\OneDrive\\Documents\\MEGA\\CAI-FLAME\\gnn-synthetic-layout-historical\\app\\tests\\logs\\pipeline_ablation_latest.json",
          "latest_summary_md": "C:\\Users\\intro\\OneDrive\\Documents\\MEGA\\CAI-FLAME\\gnn-synthetic-layout-historical\\app\\tests\\logs\\pipeline_ablation_latest.md",
          "run_metrics_json": "C:\\Users\\intro\\OneDrive\\Documents\\MEGA\\CAI-FLAME\\gnn-synthetic-layout-historical\\app\\tests\\logs\\20260515_145658_pipeline_ablation_eval_dataset_summary\\metrics.json",
          "run_summary_md": "C:\\Users\\intro\\OneDrive\\Documents\\MEGA\\CAI-FLAME\\gnn-synthetic-layout-historical\\app\\tests\\logs\\20260515_145658_pipeline_ablation_eval_dataset_summary\\summary.md"
        },
        "ocr_eval_dataset": {
          "latest_metrics_json": "C:\\Users\\intro\\OneDrive\\Documents\\MEGA\\CAI-FLAME\\gnn-synthetic-layout-historical\\app\\tests\\logs\\recognition_finetune_ablation_latest.json",
          "latest_summary_md": "C:\\Users\\intro\\OneDrive\\Documents\\MEGA\\CAI-FLAME\\gnn-synthetic-layout-historical\\app\\tests\\logs\\recognition_finetune_ablation_latest.md",
          "run_metrics_json": "C:\\Users\\intro\\OneDrive\\Documents\\MEGA\\CAI-FLAME\\gnn-synthetic-layout-historical\\app\\tests\\logs\\20260512_135604_ocr_ablation_eval_dataset_summary\\metrics.json",
          "run_summary_md": "C:\\Users\\intro\\OneDrive\\Documents\\MEGA\\CAI-FLAME\\gnn-synthetic-layout-historical\\app\\tests\\logs\\20260512_135604_ocr_ablation_eval_dataset_summary\\summary.md"
        },
        "circular_ocr_eval_dataset_v2": {
          "latest_metrics_json": "C:\\Users\\intro\\OneDrive\\Documents\\MEGA\\CAI-FLAME\\gnn-synthetic-layout-historical\\app\\tests\\logs\\circular_ocr_ablation_latest.json",
          "latest_summary_md": "C:\\Users\\intro\\OneDrive\\Documents\\MEGA\\CAI-FLAME\\gnn-synthetic-layout-historical\\app\\tests\\logs\\circular_ocr_ablation_latest.md",
          "run_metrics_json": "C:\\Users\\intro\\OneDrive\\Documents\\MEGA\\CAI-FLAME\\gnn-synthetic-layout-historical\\app\\tests\\logs\\20260512_135937_circular_ocr_ablation_eval_dataset_v2_summary\\metrics.json",
          "run_summary_md": "C:\\Users\\intro\\OneDrive\\Documents\\MEGA\\CAI-FLAME\\gnn-synthetic-layout-historical\\app\\tests\\logs\\20260512_135937_circular_ocr_ablation_eval_dataset_v2_summary\\summary.md"
        }
      },
      "gate_metric_summary": {
        "pipeline_eval_dataset": {
          "primary_metric_name": "page_cer",
          "benchmark_value": 0.3309031044214487,
          "proposed_value": 0.33827218563813105,
          "operator": "<=",
          "passed": true
        },
        "ocr_eval_dataset": {
          "primary_metric_name": "curve_metric_value",
          "benchmark_value": 0.23991308761022648,
          "proposed_value": 0.24910867220706717,
          "operator": "<=",
          "passed": true
        },
        "circular_ocr_eval_dataset_v2": {
          "primary_metric_name": "curve_metric_value",
          "benchmark_value": 0.9447010869565217,
          "proposed_value": 0.18057065217391305,
          "operator": "<",
          "passed": true
        }
      },
      "promotion_timestamp_utc": "2026-05-16T07:14:57Z",
      "author_or_tool": "scripts/promote_text_line_strategy.py"
    },
    {
      "promoted_strategy_name": "local_polygons_v1",
      "previous_benchmark_strategy_name": "local_tangent_band_v1",
      "evidence_metrics_path": "C:\\Users\\intro\\OneDrive\\Documents\\MEGA\\CAI-FLAME\\gnn-synthetic-layout-historical\\app\\tests\\logs\\strategy_promotion_latest.json",
      "evidence_generated_at_utc": "2026-05-22T07:18:29Z",
      "gate_artifact_paths": {
        "pipeline_eval_dataset": {
          "latest_metrics_json": "C:\\Users\\intro\\OneDrive\\Documents\\MEGA\\CAI-FLAME\\gnn-synthetic-layout-historical\\app\\tests\\logs\\pipeline_ablation_latest.json",
          "latest_summary_md": "C:\\Users\\intro\\OneDrive\\Documents\\MEGA\\CAI-FLAME\\gnn-synthetic-layout-historical\\app\\tests\\logs\\pipeline_ablation_latest.md",
          "run_metrics_json": "C:\\Users\\intro\\OneDrive\\Documents\\MEGA\\CAI-FLAME\\gnn-synthetic-layout-historical\\app\\tests\\logs\\20260522_123345_pipeline_ablation_eval_dataset_summary\\metrics.json",
          "run_summary_md": "C:\\Users\\intro\\OneDrive\\Documents\\MEGA\\CAI-FLAME\\gnn-synthetic-layout-historical\\app\\tests\\logs\\20260522_123345_pipeline_ablation_eval_dataset_summary\\summary.md"
        },
        "ocr_eval_dataset": {
          "latest_metrics_json": "C:\\Users\\intro\\OneDrive\\Documents\\MEGA\\CAI-FLAME\\gnn-synthetic-layout-historical\\app\\tests\\logs\\recognition_finetune_ablation_latest.json",
          "latest_summary_md": "C:\\Users\\intro\\OneDrive\\Documents\\MEGA\\CAI-FLAME\\gnn-synthetic-layout-historical\\app\\tests\\logs\\recognition_finetune_ablation_latest.md",
          "run_metrics_json": "C:\\Users\\intro\\OneDrive\\Documents\\MEGA\\CAI-FLAME\\gnn-synthetic-layout-historical\\app\\tests\\logs\\20260522_124343_ocr_ablation_eval_dataset_summary\\metrics.json",
          "run_summary_md": "C:\\Users\\intro\\OneDrive\\Documents\\MEGA\\CAI-FLAME\\gnn-synthetic-layout-historical\\app\\tests\\logs\\20260522_124343_ocr_ablation_eval_dataset_summary\\summary.md"
        },
        "circular_ocr_eval_dataset_v2": {
          "latest_metrics_json": "C:\\Users\\intro\\OneDrive\\Documents\\MEGA\\CAI-FLAME\\gnn-synthetic-layout-historical\\app\\tests\\logs\\circular_ocr_ablation_latest.json",
          "latest_summary_md": "C:\\Users\\intro\\OneDrive\\Documents\\MEGA\\CAI-FLAME\\gnn-synthetic-layout-historical\\app\\tests\\logs\\circular_ocr_ablation_latest.md",
          "run_metrics_json": "C:\\Users\\intro\\OneDrive\\Documents\\MEGA\\CAI-FLAME\\gnn-synthetic-layout-historical\\app\\tests\\logs\\20260522_124828_circular_ocr_ablation_eval_dataset_v2_summary\\metrics.json",
          "run_summary_md": "C:\\Users\\intro\\OneDrive\\Documents\\MEGA\\CAI-FLAME\\gnn-synthetic-layout-historical\\app\\tests\\logs\\20260522_124828_circular_ocr_ablation_eval_dataset_v2_summary\\summary.md"
        }
      },
      "gate_metric_summary": {
        "pipeline_eval_dataset": {
          "primary_metric_name": "page_cer",
          "benchmark_value": 0.3327845719661336,
          "proposed_value": 0.321574161179053,
          "operator": "<=",
          "passed": true
        },
        "ocr_eval_dataset": {
          "primary_metric_name": "curve_metric_value",
          "benchmark_value": 0.2867759944173064,
          "proposed_value": 0.27201674808094906,
          "operator": "<=",
          "passed": true
        },
        "circular_ocr_eval_dataset_v2": {
          "primary_metric_name": "curve_metric_value",
          "benchmark_value": 0.17744565217391303,
          "proposed_value": 0.16032608695652176,
          "operator": "<",
          "passed": true
        }
      },
      "promotion_timestamp_utc": "2026-05-22T07:31:01Z",
      "author_or_tool": "scripts/promote_text_line_strategy.py"
    },
    {
      "promoted_strategy_name": "local_polygons_stable_unwrap_v1",
      "previous_benchmark_strategy_name": "local_polygons_v1",
      "evidence_metrics_path": "C:\\Users\\intro\\OneDrive\\Documents\\MEGA\\CAI-FLAME\\gnn-synthetic-layout-historical\\app\\tests\\logs\\strategy_promotion_latest.json",
      "evidence_generated_at_utc": "2026-05-29T11:14:24Z",
      "gate_artifact_paths": {
        "pipeline_eval_dataset": {
          "latest_metrics_json": "C:\\Users\\intro\\OneDrive\\Documents\\MEGA\\CAI-FLAME\\gnn-synthetic-layout-historical\\app\\tests\\logs\\pipeline_ablation_latest.json",
          "latest_summary_md": "C:\\Users\\intro\\OneDrive\\Documents\\MEGA\\CAI-FLAME\\gnn-synthetic-layout-historical\\app\\tests\\logs\\pipeline_ablation_latest.md",
          "run_metrics_json": "C:\\Users\\intro\\OneDrive\\Documents\\MEGA\\CAI-FLAME\\gnn-synthetic-layout-historical\\app\\tests\\logs\\20260529_162925_pipeline_ablation_eval_dataset_summary\\metrics.json",
          "run_summary_md": "C:\\Users\\intro\\OneDrive\\Documents\\MEGA\\CAI-FLAME\\gnn-synthetic-layout-historical\\app\\tests\\logs\\20260529_162925_pipeline_ablation_eval_dataset_summary\\summary.md"
        },
        "ocr_eval_dataset": {
          "latest_metrics_json": "C:\\Users\\intro\\OneDrive\\Documents\\MEGA\\CAI-FLAME\\gnn-synthetic-layout-historical\\app\\tests\\logs\\recognition_finetune_ablation_latest.json",
          "latest_summary_md": "C:\\Users\\intro\\OneDrive\\Documents\\MEGA\\CAI-FLAME\\gnn-synthetic-layout-historical\\app\\tests\\logs\\recognition_finetune_ablation_latest.md",
          "run_metrics_json": "C:\\Users\\intro\\OneDrive\\Documents\\MEGA\\CAI-FLAME\\gnn-synthetic-layout-historical\\app\\tests\\logs\\20260529_163930_ocr_ablation_eval_dataset_summary\\metrics.json",
          "run_summary_md": "C:\\Users\\intro\\OneDrive\\Documents\\MEGA\\CAI-FLAME\\gnn-synthetic-layout-historical\\app\\tests\\logs\\20260529_163930_ocr_ablation_eval_dataset_summary\\summary.md"
        },
        "circular_ocr_eval_dataset_v2": {
          "latest_metrics_json": "C:\\Users\\intro\\OneDrive\\Documents\\MEGA\\CAI-FLAME\\gnn-synthetic-layout-historical\\app\\tests\\logs\\circular_ocr_ablation_latest.json",
          "latest_summary_md": "C:\\Users\\intro\\OneDrive\\Documents\\MEGA\\CAI-FLAME\\gnn-synthetic-layout-historical\\app\\tests\\logs\\circular_ocr_ablation_latest.md",
          "run_metrics_json": "C:\\Users\\intro\\OneDrive\\Documents\\MEGA\\CAI-FLAME\\gnn-synthetic-layout-historical\\app\\tests\\logs\\20260529_164421_circular_ocr_ablation_eval_dataset_v2_summary\\metrics.json",
          "run_summary_md": "C:\\Users\\intro\\OneDrive\\Documents\\MEGA\\CAI-FLAME\\gnn-synthetic-layout-historical\\app\\tests\\logs\\20260529_164421_circular_ocr_ablation_eval_dataset_v2_summary\\summary.md"
        }
      },
      "gate_metric_summary": {
        "pipeline_eval_dataset": {
          "primary_metric_name": "page_cer",
          "benchmark_value": 0.321574161179053,
          "proposed_value": 0.3180464095327689,
          "operator": "<=",
          "passed": true
        },
        "ocr_eval_dataset": {
          "primary_metric_name": "curve_metric_value",
          "benchmark_value": 0.26966154919748775,
          "proposed_value": 0.26174110258199584,
          "operator": "<=",
          "passed": true
        },
        "circular_ocr_eval_dataset_v2": {
          "primary_metric_name": "curve_metric_value",
          "benchmark_value": 0.16358695652173913,
          "proposed_value": 0.15625,
          "operator": "<",
          "passed": true
        }
      },
      "promotion_timestamp_utc": "2026-05-30T08:59:54Z",
      "author_or_tool": "scripts/promote_text_line_strategy.py"
    }
  ],
  "production_adoption_history": [
    {
      "adopted_strategy_name": "local_polygons_v1",
      "previous_production_strategy_name": "legacy_axis_bound_v1",
      "adoption_timestamp_utc": "2026-05-23T07:30:13Z",
      "author_or_tool": "scripts/adopt_text_line_strategy_for_app.py",
      "reason": "Adopt current research benchmark with strategy-owned runtime config and reading-direction metadata support."
    },
    {
      "adopted_strategy_name": "local_polygons_stable_unwrap_v1",
      "previous_production_strategy_name": "local_polygons_v1",
      "adoption_timestamp_utc": "2026-05-30T09:37:04Z",
      "author_or_tool": "scripts/adopt_text_line_strategy_for_app.py",
      "reason": "Adopt current research benchmark for production after stable unwrap runtime validation."
    }
  ]
}
"""

STRATEGY_ROLE_CONFIG = json.loads(STRATEGY_ROLE_CONFIG_JSON)
STRATEGY_ROLE_CONFIG_PATH = Path(__file__).resolve()
DEFAULT_PRODUCTION_STRATEGY_NAME = "legacy_axis_bound_v1"


def _clone_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(payload))


def _normalize_optional_strategy_name(value: Any) -> str | None:
    if value is None:
        return None
    return str(value).strip() or None


def _normalize_required_strategy_name(payload: dict[str, Any], key: str) -> str:
    strategy_name = str(payload[key]).strip()
    if not strategy_name:
        raise ValueError(f"{key} must be a non-empty string.")
    return strategy_name


def _normalize_history_list(payload: dict[str, Any], key: str, *, legacy_key: str | None = None) -> list[Any]:
    if key in payload:
        history = payload[key]
    elif legacy_key is not None and legacy_key in payload:
        history = payload[legacy_key]
    else:
        history = []
    if not isinstance(history, list):
        raise ValueError(f"{key} must be a list.")
    return _clone_payload({"items": history})["items"]


def normalize_strategy_role_config_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("strategy role config payload must be a dictionary.")

    benchmark_strategy_name = _normalize_required_strategy_name(payload, "benchmark_strategy_name")
    proposed_strategy_name = _normalize_optional_strategy_name(payload.get("proposed_strategy_name"))
    production_strategy_name = _normalize_optional_strategy_name(
        payload.get("production_strategy_name", DEFAULT_PRODUCTION_STRATEGY_NAME)
    )
    if production_strategy_name is None:
        raise ValueError("production_strategy_name must be a non-empty string.")

    return {
        "benchmark_strategy_name": benchmark_strategy_name,
        "proposed_strategy_name": proposed_strategy_name,
        "production_strategy_name": production_strategy_name,
        "research_promotion_history": _normalize_history_list(
            payload,
            "research_promotion_history",
            legacy_key="promotion_history",
        ),
        "production_adoption_history": _normalize_history_list(payload, "production_adoption_history"),
    }


def validate_strategy_role_config_payload_roles(payload: dict[str, Any]) -> None:
    normalized = normalize_strategy_role_config_payload(payload)

    from .registry import validate_production_role_strategy, validate_research_role_strategy
    from .runtime_config import PRODUCTION_STRATEGY_RUNTIME_CONFIGS

    validate_research_role_strategy(normalized["benchmark_strategy_name"], role_label="Research benchmark")
    proposed_strategy_name = normalized.get("proposed_strategy_name")
    if proposed_strategy_name is not None:
        validate_research_role_strategy(proposed_strategy_name, role_label="Research proposed")

    production_strategy_name = normalized["production_strategy_name"]
    validate_production_role_strategy(production_strategy_name, role_label="Production app")
    if production_strategy_name not in PRODUCTION_STRATEGY_RUNTIME_CONFIGS:
        raise ValueError(
            f"Production app strategy {production_strategy_name!r} does not have a production runtime config. "
            "Add an explicit config before using it for app saves."
        )


def get_strategy_role_config() -> dict[str, Any]:
    normalized = normalize_strategy_role_config_payload(STRATEGY_ROLE_CONFIG)
    validate_strategy_role_config_payload_roles(normalized)
    return normalized


def get_benchmark_strategy_name() -> str:
    return get_strategy_role_config()["benchmark_strategy_name"]


def get_proposed_strategy_name() -> str | None:
    return get_strategy_role_config()["proposed_strategy_name"]


def get_production_strategy_name() -> str:
    return get_strategy_role_config()["production_strategy_name"]


def load_strategy_role_config_from_path(path: str | Path) -> dict[str, Any]:
    source = Path(path).read_text(encoding="utf-8")
    match = re.search(r'STRATEGY_ROLE_CONFIG_JSON = r?"""([\s\S]*?)"""', source)
    if match is None:
        match = re.search(r"STRATEGY_ROLE_CONFIG_JSON = r?'''([\s\S]*?)'''", source)
    if match is None:
        raise ValueError(f"Could not locate STRATEGY_ROLE_CONFIG_JSON in {path}")
    return normalize_strategy_role_config_payload(json.loads(match.group(1)))


def render_strategy_role_config(payload: dict[str, Any]) -> str:
    normalized = normalize_strategy_role_config_payload(payload)
    source = STRATEGY_ROLE_CONFIG_PATH.read_text(encoding="utf-8")
    replacement = 'STRATEGY_ROLE_CONFIG_JSON = r"""' + json.dumps(
        normalized,
        indent=2,
        ensure_ascii=False,
    ) + '\n"""'
    updated = re.sub(
        r'STRATEGY_ROLE_CONFIG_JSON = r?"""[\s\S]*?"""',
        lambda _: replacement,
        source,
        count=1,
    )
    if re.search(r'STRATEGY_ROLE_CONFIG_JSON = r?"""', updated) is None:
        raise ValueError("Could not locate STRATEGY_ROLE_CONFIG_JSON block for rewrite.")
    return updated


def write_strategy_role_config(path: str | Path, payload: dict[str, Any]) -> Path:
    destination = Path(path)
    destination.write_text(render_strategy_role_config(payload), encoding="utf-8", newline="\n")
    return destination


__all__ = [
    "DEFAULT_PRODUCTION_STRATEGY_NAME",
    "STRATEGY_ROLE_CONFIG",
    "STRATEGY_ROLE_CONFIG_PATH",
    "get_benchmark_strategy_name",
    "get_proposed_strategy_name",
    "get_production_strategy_name",
    "get_strategy_role_config",
    "load_strategy_role_config_from_path",
    "normalize_strategy_role_config_payload",
    "render_strategy_role_config",
    "validate_strategy_role_config_payload_roles",
    "write_strategy_role_config",
]
