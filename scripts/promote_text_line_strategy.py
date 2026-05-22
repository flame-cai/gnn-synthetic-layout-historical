from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = REPO_ROOT / "app"
LOGS_DIR = APP_ROOT / "tests" / "logs"
DEFAULT_PROMOTION_EVIDENCE_JSON_PATH = LOGS_DIR / "strategy_promotion_latest.json"
DEFAULT_PROMOTION_EVIDENCE_MD_PATH = LOGS_DIR / "strategy_promotion_latest.md"
DEFAULT_CHECKED_IN_PROMOTION_RECORD_MD_PATH = (
    REPO_ROOT
    / "docs"
    / "pipeline-improvement"
    / "text-line-segmentation"
    / "strategy-promotion-record.md"
)

if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recognition.line_segmentation.strategy_config import (
    STRATEGY_ROLE_CONFIG_PATH,
    load_strategy_role_config_from_path,
    write_strategy_role_config,
)

LOGGER = logging.getLogger(__name__)

LATEST_GATE_SPECS = (
    {
        "gate_key": "pipeline_eval_dataset",
        "label": "Full Pipeline Strategy Ablation Gate",
        "dataset_name": "eval_dataset",
        "expected_study_mode": "pipeline_strategy_ablation_gate",
        "metrics_path": LOGS_DIR / "pipeline_ablation_latest.json",
        "summary_path": LOGS_DIR / "pipeline_ablation_latest.md",
    },
    {
        "gate_key": "ocr_eval_dataset",
        "label": "Recognition Fine-Tune Strategy Ablation Gate",
        "dataset_name": "eval_dataset",
        "expected_study_mode": "recognition_strategy_ablation_gate",
        "metrics_path": LOGS_DIR / "recognition_finetune_ablation_latest.json",
        "summary_path": LOGS_DIR / "recognition_finetune_ablation_latest.md",
    },
    {
        "gate_key": "circular_ocr_eval_dataset_v2",
        "label": "Circular Recognition Fine-Tune Strategy Ablation Gate",
        "dataset_name": "eval_dataset_v2",
        "expected_study_mode": "circular_recognition_strategy_ablation_gate",
        "metrics_path": LOGS_DIR / "circular_ocr_ablation_latest.json",
        "summary_path": LOGS_DIR / "circular_ocr_ablation_latest.md",
    },
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _mtime_iso(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00",
        "Z",
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _require_path(path: Path, *, label: str) -> Path:
    if not path.exists():
        raise ValueError(f"Missing {label}: {path}")
    return path


def _extract_gate_result(spec: dict[str, Any]) -> dict[str, Any]:
    metrics_path = _require_path(Path(spec["metrics_path"]), label=f"{spec['label']} metrics")
    summary_path = _require_path(Path(spec["summary_path"]), label=f"{spec['label']} summary")
    payload = _load_json(metrics_path)
    if payload.get("study_mode") != spec["expected_study_mode"]:
        raise ValueError(
            f"{spec['label']} metrics at {metrics_path} had study_mode={payload.get('study_mode')!r}, "
            f"expected {spec['expected_study_mode']!r}."
        )

    dataset_result = payload.get("dataset_results", {}).get(spec["dataset_name"])
    if not isinstance(dataset_result, dict):
        raise ValueError(f"{spec['label']} metrics at {metrics_path} did not contain dataset {spec['dataset_name']!r}.")

    comparison = dataset_result.get("comparison", {})
    benchmark_result = dataset_result.get("strategy_results", {}).get("benchmark", {})
    proposed_result = dataset_result.get("strategy_results", {}).get("proposed", {})
    run_summary_path = Path(dataset_result["summary_path"])
    run_metrics_path = Path(dataset_result["metrics_path"])

    return {
        "gate_key": spec["gate_key"],
        "label": spec["label"],
        "dataset_name": spec["dataset_name"],
        "study_mode": spec["expected_study_mode"],
        "passed": bool(dataset_result.get("passed")),
        "failure_message": dataset_result.get("failure_message", ""),
        "primary_metric_name": comparison.get("primary_metric_name"),
        "benchmark_value": comparison.get("benchmark_value"),
        "proposed_value": comparison.get("proposed_value"),
        "operator": comparison.get("operator"),
        "allowed_regression_abs": comparison.get("allowed_regression_abs"),
        "strict_primary_improvement_required": bool(comparison.get("strict_primary_improvement_required", False)),
        "benchmark_strategy_name": benchmark_result.get("strategy_name"),
        "proposed_strategy_name": proposed_result.get("strategy_name"),
        "artifact_paths": {
            "latest_metrics_json": str(metrics_path.resolve()),
            "latest_summary_md": str(summary_path.resolve()),
            "run_metrics_json": str(run_metrics_path.resolve()),
            "run_summary_md": str(run_summary_path.resolve()),
        },
        "artifact_mtime_utc": {
            "latest_metrics_json": _mtime_iso(metrics_path),
            "latest_summary_md": _mtime_iso(summary_path),
            "run_metrics_json": _mtime_iso(run_metrics_path),
            "run_summary_md": _mtime_iso(run_summary_path),
        },
    }


def build_strategy_promotion_evidence() -> dict[str, Any]:
    gate_results = {}
    benchmark_names = set()
    proposed_names = set()
    blockers = []

    for spec in LATEST_GATE_SPECS:
        gate_result = _extract_gate_result(spec)
        gate_results[spec["gate_key"]] = gate_result
        benchmark_names.add(str(gate_result["benchmark_strategy_name"]))
        proposed_names.add(str(gate_result["proposed_strategy_name"]))
        if not gate_result["passed"]:
            blockers.append(f"{spec['gate_key']}: {gate_result['failure_message'] or 'gate failed'}")

    if len(benchmark_names) != 1:
        raise ValueError(f"Inconsistent benchmark strategies across gate artifacts: {sorted(benchmark_names)}")
    if len(proposed_names) != 1:
        raise ValueError(f"Inconsistent proposed strategies across gate artifacts: {sorted(proposed_names)}")

    return {
        "study_mode": "strategy_promotion_evidence",
        "generated_at_utc": _utc_now_iso(),
        "benchmark_strategy_name": next(iter(benchmark_names)),
        "proposed_strategy_name": next(iter(proposed_names)),
        "promotion_recommended": not blockers,
        "promotion_blockers": blockers,
        "gate_results": gate_results,
    }


def render_strategy_promotion_markdown(evidence: dict[str, Any]) -> str:
    lines = [
        "# Text-Line Strategy Research Promotion Evidence",
        "",
        f"Generated at: `{evidence['generated_at_utc']}`",
        f"Research benchmark strategy: `{evidence['benchmark_strategy_name']}`",
        f"Research proposed strategy: `{evidence['proposed_strategy_name']}`",
        f"Research harness promotion recommended: `{evidence['promotion_recommended']}`",
        "",
        "## Gates",
        "",
    ]
    for gate_key in ("pipeline_eval_dataset", "ocr_eval_dataset", "circular_ocr_eval_dataset_v2"):
        gate = evidence["gate_results"][gate_key]
        lines.append(
            f"- {gate_key}: passed={gate['passed']}, "
            f"primary_metric={gate['primary_metric_name']}, "
            f"benchmark={gate['benchmark_value']}, "
            f"proposed={gate['proposed_value']}, "
            f"operator={gate['operator']}"
        )

    if evidence.get("promotion_blockers"):
        lines.extend(["", "## Blockers", ""])
        lines.extend(f"- {item}" for item in evidence["promotion_blockers"])

    lines.extend(["", "## Artifacts", ""])
    for gate_key in ("pipeline_eval_dataset", "ocr_eval_dataset", "circular_ocr_eval_dataset_v2"):
        gate = evidence["gate_results"][gate_key]
        lines.append(f"- {gate_key}_metrics={gate['artifact_paths']['latest_metrics_json']}")
        lines.append(f"- {gate_key}_summary={gate['artifact_paths']['latest_summary_md']}")

    return "\n".join(lines) + "\n"


def _display_path(path_value: str | Path) -> str:
    path = Path(path_value)
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except Exception:
        return str(path_value)


def render_checked_in_strategy_promotion_record(evidence: dict[str, Any]) -> str:
    lines = [
        "# Text-Line Strategy Promotion Record",
        "",
        "This file is generated by `scripts/run_precommit_eval.py` through "
        "`scripts/promote_text_line_strategy.py`. It is checked in because `app/tests/logs/` "
        "is ignored and local gate artifacts are not durable in a fresh clone.",
        "",
        f"Generated at: `{evidence['generated_at_utc']}`",
        f"Research benchmark strategy: `{evidence['benchmark_strategy_name']}`",
        f"Research proposed strategy: `{evidence['proposed_strategy_name']}`",
        f"Research promotion recommended: `{evidence['promotion_recommended']}`",
        "",
        "## Gate Summary",
        "",
        "| Gate | Dataset | Passed | Primary metric | Benchmark | Proposed | Operator |",
        "| --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    for gate_key in ("pipeline_eval_dataset", "ocr_eval_dataset", "circular_ocr_eval_dataset_v2"):
        gate = evidence["gate_results"][gate_key]
        dataset_name = gate.get("dataset_name", "")
        lines.append(
            "| "
            f"`{gate_key}` | "
            f"`{dataset_name}` | "
            f"`{gate['passed']}` | "
            f"`{gate['primary_metric_name']}` | "
            f"{gate['benchmark_value']} | "
            f"{gate['proposed_value']} | "
            f"`{gate['operator']}` |"
        )

    if evidence.get("promotion_blockers"):
        lines.extend(["", "## Blockers", ""])
        lines.extend(f"- {item}" for item in evidence["promotion_blockers"])

    lines.extend(["", "## Local Artifact References", ""])
    lines.append("The paths below are relative where possible. They may not exist in a fresh checkout.")
    lines.append("")
    for gate_key in ("pipeline_eval_dataset", "ocr_eval_dataset", "circular_ocr_eval_dataset_v2"):
        gate = evidence["gate_results"][gate_key]
        lines.append(f"### `{gate_key}`")
        lines.append("")
        for label, path_value in gate.get("artifact_paths", {}).items():
            mtime = gate.get("artifact_mtime_utc", {}).get(label)
            suffix = f" (mtime `{mtime}`)" if mtime else ""
            lines.append(f"- `{label}`: `{_display_path(path_value)}`{suffix}")
        lines.append("")

    lines.extend(
        [
            "## Promotion Command",
            "",
            "Run the promotion command only after reviewing this record and the local artifacts:",
            "",
            "```powershell",
            "$env:CONDA_NO_PLUGINS='true'",
            "conda run -n gnn_layout python scripts/promote_text_line_strategy.py "
            f"--candidate {evidence['proposed_strategy_name']} "
            f"--previous-benchmark {evidence['benchmark_strategy_name']} "
            "--metrics app/tests/logs/strategy_promotion_latest.json --apply",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def write_checked_in_strategy_promotion_record(
    evidence: dict[str, Any],
    record_path: Path = DEFAULT_CHECKED_IN_PROMOTION_RECORD_MD_PATH,
) -> Path:
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(
        render_checked_in_strategy_promotion_record(evidence),
        encoding="utf-8",
        newline="\n",
    )
    return record_path


def write_strategy_promotion_evidence(
    *,
    json_path: Path = DEFAULT_PROMOTION_EVIDENCE_JSON_PATH,
    markdown_path: Path = DEFAULT_PROMOTION_EVIDENCE_MD_PATH,
    checked_in_record_path: Path | None = DEFAULT_CHECKED_IN_PROMOTION_RECORD_MD_PATH,
) -> dict[str, Any]:
    evidence = build_strategy_promotion_evidence()
    _write_json(json_path, evidence)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_strategy_promotion_markdown(evidence), encoding="utf-8")
    if checked_in_record_path is not None:
        write_checked_in_strategy_promotion_record(evidence, checked_in_record_path)
    return evidence


def _validate_registered_strategy(strategy_name: str, *, role_label: str) -> None:
    from recognition.line_segmentation.registry import list_text_line_segmentation_strategies

    available = set(list_text_line_segmentation_strategies())
    if strategy_name not in available:
        raise ValueError(
            f"{role_label} strategy {strategy_name!r} is not registered. "
            f"Available strategies: {', '.join(sorted(available))}"
        )


def _load_and_validate_evidence(metrics_path: Path, *, candidate: str, previous_benchmark: str) -> dict[str, Any]:
    metrics_path = _require_path(metrics_path, label="promotion evidence")
    LOGGER.info(
        "Validating research promotion evidence metrics_path=%s candidate=%s previous_benchmark=%s",
        metrics_path,
        candidate,
        previous_benchmark,
    )
    evidence = _load_json(metrics_path)
    if evidence.get("study_mode") != "strategy_promotion_evidence":
        raise ValueError(
            f"Promotion evidence at {metrics_path} had study_mode={evidence.get('study_mode')!r}, "
            "expected 'strategy_promotion_evidence'."
        )

    benchmark_strategy_name = evidence.get("benchmark_strategy_name")
    proposed_strategy_name = evidence.get("proposed_strategy_name")
    if benchmark_strategy_name != previous_benchmark:
        raise ValueError(
            f"Evidence benchmark strategy {benchmark_strategy_name!r} did not match "
            f"--previous-benchmark {previous_benchmark!r}."
        )
    if proposed_strategy_name != candidate:
        raise ValueError(f"Evidence proposed strategy {proposed_strategy_name!r} did not match --candidate {candidate!r}.")
    if not evidence.get("promotion_recommended"):
        raise ValueError(f"Promotion evidence did not recommend promotion: {evidence.get('promotion_blockers', [])}")

    evidence_mtime = metrics_path.stat().st_mtime
    gate_results = evidence.get("gate_results", {})
    missing_gates = {spec["gate_key"] for spec in LATEST_GATE_SPECS} - set(gate_results)
    if missing_gates:
        raise ValueError(f"Promotion evidence was missing gate results for: {sorted(missing_gates)}")

    for spec in LATEST_GATE_SPECS:
        gate = gate_results[spec["gate_key"]]
        if not gate.get("passed"):
            raise ValueError(f"Gate {spec['gate_key']} did not pass: {gate.get('failure_message', '')}")
        artifact_paths = gate.get("artifact_paths", {})
        artifact_mtimes = gate.get("artifact_mtime_utc", {})
        for artifact_label, artifact_path in artifact_paths.items():
            resolved = _require_path(Path(artifact_path), label=f"{spec['gate_key']} artifact {artifact_label}")
            actual_mtime = _mtime_iso(resolved)
            expected_mtime = artifact_mtimes.get(artifact_label)
            if expected_mtime and actual_mtime != expected_mtime:
                raise ValueError(
                    f"Promotion evidence was stale for {spec['gate_key']} artifact {artifact_label}: "
                    f"expected mtime {expected_mtime}, found {actual_mtime}."
                )
            if resolved.stat().st_mtime > evidence_mtime:
                raise ValueError(
                    f"Promotion evidence at {metrics_path} is stale because {resolved} is newer than the evidence file."
                )

    return evidence


def _build_history_entry(
    *,
    candidate: str,
    previous_benchmark: str,
    metrics_path: Path,
    evidence: dict[str, Any],
    author_or_tool: str,
) -> dict[str, Any]:
    return {
        "promoted_strategy_name": candidate,
        "previous_benchmark_strategy_name": previous_benchmark,
        "evidence_metrics_path": str(metrics_path.resolve()),
        "evidence_generated_at_utc": evidence["generated_at_utc"],
        "gate_artifact_paths": {
            gate_key: gate_result["artifact_paths"]
            for gate_key, gate_result in evidence["gate_results"].items()
        },
        "gate_metric_summary": {
            gate_key: {
                "primary_metric_name": gate_result["primary_metric_name"],
                "benchmark_value": gate_result["benchmark_value"],
                "proposed_value": gate_result["proposed_value"],
                "operator": gate_result["operator"],
                "passed": gate_result["passed"],
            }
            for gate_key, gate_result in evidence["gate_results"].items()
        },
        "promotion_timestamp_utc": _utc_now_iso(),
        "author_or_tool": author_or_tool,
    }


def _find_matching_history_entry(history: list[dict[str, Any]], *, candidate: str, previous_benchmark: str, metrics_path: Path, evidence: dict[str, Any]) -> dict[str, Any] | None:
    expected_metrics_path = str(metrics_path.resolve())
    expected_generated_at = evidence["generated_at_utc"]
    for entry in history:
        if (
            entry.get("promoted_strategy_name") == candidate
            and entry.get("previous_benchmark_strategy_name") == previous_benchmark
            and entry.get("evidence_metrics_path") == expected_metrics_path
            and entry.get("evidence_generated_at_utc") == expected_generated_at
        ):
            return entry
    return None


def promote_text_line_strategy(
    *,
    candidate: str,
    previous_benchmark: str,
    metrics_path: Path | None = None,
    apply: bool = False,
    strategy_config_path: Path = STRATEGY_ROLE_CONFIG_PATH,
    author_or_tool: str = "scripts/promote_text_line_strategy.py",
) -> dict[str, Any]:
    candidate = str(candidate).strip()
    previous_benchmark = str(previous_benchmark).strip()
    if not candidate:
        raise ValueError("candidate must be a non-empty string.")
    if not previous_benchmark:
        raise ValueError("previous_benchmark must be a non-empty string.")
    if candidate == previous_benchmark:
        raise ValueError("candidate must differ from previous_benchmark.")

    _validate_registered_strategy(candidate, role_label="Candidate")
    _validate_registered_strategy(previous_benchmark, role_label="Previous benchmark")

    resolved_metrics_path = Path(metrics_path or DEFAULT_PROMOTION_EVIDENCE_JSON_PATH)
    evidence = _load_and_validate_evidence(
        resolved_metrics_path,
        candidate=candidate,
        previous_benchmark=previous_benchmark,
    )

    current_config = load_strategy_role_config_from_path(strategy_config_path)
    current_benchmark = current_config["benchmark_strategy_name"]
    current_proposed = current_config.get("proposed_strategy_name")
    promotion_history = list(current_config.get("research_promotion_history", []))
    matching_history = _find_matching_history_entry(
        promotion_history,
        candidate=candidate,
        previous_benchmark=previous_benchmark,
        metrics_path=resolved_metrics_path,
        evidence=evidence,
    )

    if current_benchmark == candidate and matching_history is not None:
        return {
            "changed": False,
            "applied": False,
            "idempotent": True,
            "message": (
                f"{candidate} is already the research benchmark strategy and matching "
                "research promotion history is already recorded."
            ),
            "metrics_path": str(resolved_metrics_path.resolve()),
            "config_path": str(strategy_config_path.resolve()),
            "config_before": current_config,
            "config_after": current_config,
        }

    if current_benchmark != previous_benchmark:
        raise ValueError(
            f"Current benchmark strategy is {current_benchmark!r}, expected {previous_benchmark!r} before promotion."
        )
    if current_proposed != candidate:
        raise ValueError(
            f"Current proposed strategy is {current_proposed!r}, expected {candidate!r} before promotion."
        )

    history_entry = _build_history_entry(
        candidate=candidate,
        previous_benchmark=previous_benchmark,
        metrics_path=resolved_metrics_path,
        evidence=evidence,
        author_or_tool=author_or_tool,
    )
    updated_history = promotion_history if matching_history is not None else promotion_history + [history_entry]
    updated_config = {
        **current_config,
        "benchmark_strategy_name": candidate,
        "proposed_strategy_name": None,
        "research_promotion_history": updated_history,
    }

    result = {
        "changed": current_config != updated_config,
        "applied": False,
        "idempotent": False,
        "message": f"Promote {candidate} to research benchmark and clear the proposed strategy slot.",
        "metrics_path": str(resolved_metrics_path.resolve()),
        "config_path": str(strategy_config_path.resolve()),
        "config_before": current_config,
        "config_after": updated_config,
    }
    if apply and result["changed"]:
        LOGGER.info(
            "Applying research promotion config_path=%s previous_benchmark=%s promoted_strategy=%s production_strategy_preserved=%s",
            strategy_config_path,
            previous_benchmark,
            candidate,
            current_config.get("production_strategy_name"),
        )
        write_strategy_role_config(strategy_config_path, updated_config)
        result["applied"] = True
        result["message"] = f"Promoted {candidate} to research benchmark in {strategy_config_path}."
    return result


def _build_promotion_command(benchmark_strategy_name: str, proposed_strategy_name: str | None) -> str | None:
    if not proposed_strategy_name:
        return None
    return (
        "python scripts/promote_text_line_strategy.py "
        f"--candidate {proposed_strategy_name} "
        f"--previous-benchmark {benchmark_strategy_name} "
        f"--metrics {DEFAULT_PROMOTION_EVIDENCE_JSON_PATH.as_posix()} --apply"
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Promote a proposed text-line segmentation strategy inside the research harness."
    )
    parser.add_argument("--candidate", required=True, help="Registered strategy name to promote in the research harness.")
    parser.add_argument("--previous-benchmark", required=True, help="Current research benchmark strategy name.")
    parser.add_argument(
        "--metrics",
        default=str(DEFAULT_PROMOTION_EVIDENCE_JSON_PATH),
        help="Path to strategy promotion evidence JSON. Defaults to app/tests/logs/strategy_promotion_latest.json.",
    )
    parser.add_argument("--apply", action="store_true", help="Write the checked-in strategy role config.")
    parser.add_argument(
        "--strategy-config-path",
        default=str(STRATEGY_ROLE_CONFIG_PATH),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--author-or-tool", default="scripts/promote_text_line_strategy.py", help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = _build_arg_parser().parse_args(argv)
    result = promote_text_line_strategy(
        candidate=args.candidate,
        previous_benchmark=args.previous_benchmark,
        metrics_path=Path(args.metrics),
        apply=bool(args.apply),
        strategy_config_path=Path(args.strategy_config_path),
        author_or_tool=args.author_or_tool,
    )
    print(f"[research-promotion] Evidence: {result['metrics_path']}")
    print(f"[research-promotion] Config: {result['config_path']}")
    print(f"[research-promotion] {result['message']}")
    if not args.apply:
        before = result["config_before"]
        after = result["config_after"]
        print(
            "[research-promotion] Dry run only. research benchmark: "
            f"{before['benchmark_strategy_name']} -> {after['benchmark_strategy_name']}"
        )
        print(
            "[research-promotion] Dry run only. proposed: "
            f"{before.get('proposed_strategy_name')} -> {after.get('proposed_strategy_name')}"
        )
        print(
            "[research-promotion] Dry run only. production app strategy: "
            f"{before.get('production_strategy_name')} -> {after.get('production_strategy_name')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
