from __future__ import annotations

import csv
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

from tests.recognition_finetuning_config import RecognitionEvalDatasetConfig
from tests.recognition_finetuning_experiment import (
    PRETRAINED_OCR_CHECKPOINT,
    _page_plus_history_policy_slug,
    _policy_descriptor,
    _run_single_policy_run,
)

from .baseline import compare_metrics_to_baseline, load_baseline
from .config import BASELINE_JSON_PATH, LATEST_BASENAME, LOGS_ROOT, CircularDatasetConfig, get_circular_dataset_config
from .ocr_unwrap import apply_ocr_confidence_orientation_selection_to_pages, prepare_circular_page_line_dataset
from .segmentation_strategies import run_segmentation_strategy


def _timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _write_json(path: str | Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _write_csv(path: str | Path, fieldnames: list[str], rows: list[dict]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _copy_eval_ground_truth_subset(gt_dir: Path, page_ids: list[str], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    for page_id in page_ids:
        shutil.copy2(gt_dir / f"{page_id}.xml", output_dir / f"{page_id}.xml")
    return output_dir


def _recognition_config_for_circular(dataset_config: CircularDatasetConfig, rewritten_pagexml_dir: Path) -> RecognitionEvalDatasetConfig:
    training_overrides = {
        "num_iter": int(dataset_config.num_iter),
        "valInterval": 5,
        "lr": float(dataset_config.lr),
        "adam": dataset_config.optimizer == "adam",
        "batch_size": 1,
        "workers": 0,
    }
    return RecognitionEvalDatasetConfig(
        name=dataset_config.name,
        images_dir=dataset_config.images_dir,
        pagexml_dir=rewritten_pagexml_dir,
        layout_type="simple",
        fine_tune_page_count=dataset_config.fine_tune_page_count,
        eval_page_start_index=dataset_config.fine_tune_page_count,
        eval_page_end_index=dataset_config.fine_tune_page_count + dataset_config.eval_page_count,
        training_policy=dataset_config.training_policy,
        history_sample_line_count=dataset_config.history_sample_line_count,
        width_policy=dataset_config.width_policy,
        oversampling_policy=dataset_config.oversampling_policy,
        augmentation_policy=dataset_config.augmentation_policy,
        lr_scheduler=dataset_config.lr_scheduler,
        optimizer=dataset_config.optimizer,
        sibling_checkpoint_strategy=dataset_config.sibling_checkpoint_strategy,
        regression_guard_abs=dataset_config.regression_guard_abs,
        curve_metric=dataset_config.curve_metric,
        shuffle_train_each_epoch=dataset_config.shuffle_train_each_epoch,
        training_overrides=training_overrides,
    )


def _prepare_circular_inputs(
    dataset_config: CircularDatasetConfig,
    run_dir: Path,
) -> tuple[RecognitionEvalDatasetConfig, dict, dict, Path, Path, Path]:
    ordered_page_ids = dataset_config.ordered_page_ids()
    rewritten_pagexml_dir = run_dir / "pagexml" / dataset_config.strategy_name
    segmentation_metadata = []
    for page_id in ordered_page_ids:
        result = run_segmentation_strategy(
            dataset_config,
            page_id,
            rewritten_pagexml_dir / f"{page_id}.xml",
        )
        segmentation_metadata.append(result.metadata)
    segmentation_metadata_path = _write_json(
        run_dir / "segmentation_metadata.json",
        {
            "dataset": dataset_config.to_dict(),
            "pages": segmentation_metadata,
        },
    )

    recognition_config = _recognition_config_for_circular(dataset_config, rewritten_pagexml_dir)
    prepared_pages = {}
    for page_id in ordered_page_ids:
        image_path = dataset_config.images_dir / f"{page_id}.jpg"
        if not image_path.exists():
            matches = list(dataset_config.images_dir.glob(f"{page_id}.*"))
            if not matches:
                raise FileNotFoundError(f"Image not found for {page_id} in {dataset_config.images_dir}")
            image_path = matches[0]
        prepared_page = prepare_circular_page_line_dataset(
            rewritten_pagexml_dir / f"{page_id}.xml",
            image_path,
            run_dir / "prepared_pages" / page_id,
            dataset_config.unwrapping_config,
        )
        prepared_pages[page_id] = prepared_page

    selected_pages = apply_ocr_confidence_orientation_selection_to_pages(
        list(prepared_pages.values()),
        PRETRAINED_OCR_CHECKPOINT,
        width_policy=dataset_config.width_policy,
    )
    prepared_pages = {prepared_page.page_id: prepared_page for prepared_page in selected_pages}

    orientation_metadata = []
    for page_id in ordered_page_ids:
        prepared_page = prepared_pages[page_id]
        manifest = json.loads(Path(prepared_page.manifest_path).read_text(encoding="utf-8"))
        orientation_metadata.extend(manifest.get("orientation_metadata", []))

    orientation_metadata_path = _write_json(
        run_dir / "orientation_metadata.json",
        {
            "selector": "max_confidence",
            "uses_ground_truth_text": False,
            "checkpoint_path": str(PRETRAINED_OCR_CHECKPOINT.resolve()),
            "width_policy": dataset_config.width_policy,
            "pages": orientation_metadata,
        },
    )
    gt_subset_dir = _copy_eval_ground_truth_subset(
        rewritten_pagexml_dir,
        recognition_config.evaluation_page_ids(),
        run_dir / "gt_eval_subset",
    )
    evaluation_pages = {page_id: prepared_pages[page_id] for page_id in recognition_config.evaluation_page_ids()}
    return (
        recognition_config,
        prepared_pages,
        evaluation_pages,
        gt_subset_dir,
        segmentation_metadata_path,
        orientation_metadata_path,
    )


def _write_circular_summary(path: Path, result: dict) -> None:
    comparison = result["baseline_comparison"]
    curve_metrics = result["curve_metrics"]
    lines = [
        f"# Circular OCR Pre-Commit Gate: {result['dataset_name']}",
        "",
        f"Status: **{result['status'].upper()}**",
        "",
        f"Strategy: `{result['strategy_name']}`",
        "",
        "## Metrics",
        "",
        f"- curve_metric_value={curve_metrics.get('curve_metric_value')}",
        f"- final_page_cer={curve_metrics.get('final_page_cer')}",
        f"- first_step_gain={curve_metrics.get('first_step_gain')}",
        f"- regression_guard_passed={curve_metrics.get('regression_guard_passed')}",
        f"- max_regression={curve_metrics.get('max_regression')}",
        "",
        "## Baseline Comparison",
        "",
        f"- metric={comparison.get('metric_name')}",
        f"- direction={comparison.get('direction')}",
        f"- baseline_value={comparison.get('baseline_value')}",
        f"- current_value={comparison.get('current_value')}",
        f"- minimum_improvement_delta={comparison.get('minimum_improvement_delta')}",
        f"- delta={comparison.get('delta')}",
        "",
        "## Artifacts",
        "",
        f"- run_dir={Path(result['run_dir']).resolve()}",
        f"- metrics={Path(result['metrics_path']).resolve()}",
        f"- policy_metrics={Path(result['policy_metrics_path']).resolve()}",
        f"- per_page_csv={Path(result['per_page_csv_path']).resolve()}",
        f"- per_line_csv={Path(result['per_line_csv_path']).resolve()}",
        f"- segmentation_metadata={Path(result['segmentation_metadata_path']).resolve()}",
        f"- orientation_metadata={Path(result['orientation_metadata_path']).resolve()}",
    ]
    if result.get("failure_message"):
        lines.extend(["", "## Failure", "", result["failure_message"]])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _copy_latest_artifacts(result: dict) -> None:
    LOGS_ROOT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(result["summary_path"], LOGS_ROOT / f"{LATEST_BASENAME}.md")
    shutil.copy2(result["metrics_path"], LOGS_ROOT / f"{LATEST_BASENAME}.json")
    (LOGS_ROOT / f"{LATEST_BASENAME}.txt").write_text(
        f"Latest run: {Path(result['run_dir']).resolve()}\n",
        encoding="utf-8",
    )


def run_circular_ocr_experiment(
    strategy_name: str | None = None,
    baseline_path: str | Path = BASELINE_JSON_PATH,
) -> dict:
    dataset_config = get_circular_dataset_config(strategy_name=strategy_name)
    if not PRETRAINED_OCR_CHECKPOINT.exists():
        raise FileNotFoundError(f"Missing OCR checkpoint: {PRETRAINED_OCR_CHECKPOINT}")
    LOGS_ROOT.mkdir(parents=True, exist_ok=True)
    run_dir = LOGS_ROOT / f"{_timestamp_slug()}_circular_ocr_{dataset_config.name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    (
        recognition_config,
        prepared_pages,
        evaluation_pages,
        gt_subset_dir,
        segmentation_metadata_path,
        orientation_metadata_path,
    ) = _prepare_circular_inputs(dataset_config, run_dir)

    policy_slug = _page_plus_history_policy_slug(recognition_config)
    policy_result = _run_single_policy_run(
        run_dir / "policy" / policy_slug,
        recognition_config,
        prepared_pages,
        evaluation_pages,
        gt_subset_dir,
        slug_builder=_page_plus_history_policy_slug,
        regression_guard_mode="warn",
    )

    baseline = load_baseline(baseline_path)
    comparison = compare_metrics_to_baseline(policy_result["curve_metrics"], baseline)
    status = "passed" if policy_result["status"] == "passed" and comparison["passed"] else "failed"
    failure_message = policy_result.get("failure_message", "") or comparison.get("failure_message", "")

    result = {
        "study_mode": "circular_ocr_precommit_gate",
        "dataset_name": dataset_config.name,
        "status": status,
        "passed": status == "passed",
        "failure_message": failure_message,
        "strategy_name": dataset_config.strategy_name,
        "dataset": dataset_config.to_dict(),
        "policy": _policy_descriptor(recognition_config),
        "policy_slug": policy_slug,
        "curve_metrics": policy_result["curve_metrics"],
        "baseline_path": str(Path(baseline_path).resolve()),
        "baseline_comparison": comparison,
        "run_dir": str(run_dir.resolve()),
        "policy_run_dir": str(Path(policy_result["run_dir"]).resolve()),
        "policy_summary_path": str(Path(policy_result["summary_path"]).resolve()),
        "policy_metrics_path": str(Path(policy_result["metrics_path"]).resolve()),
        "per_page_csv_path": str(Path(policy_result["per_page_csv_path"]).resolve()),
        "per_line_csv_path": str(Path(policy_result["per_line_csv_path"]).resolve()),
        "fine_tune_metadata_path": str(Path(policy_result["fine_tune_metadata_path"]).resolve()),
        "selector_metrics_path": str(Path(policy_result["selector_metrics_path"]).resolve()),
        "segmentation_metadata_path": str(Path(segmentation_metadata_path).resolve()),
        "orientation_metadata_path": str(Path(orientation_metadata_path).resolve()),
        "python_executable": sys.executable,
    }
    summary_path = run_dir / "summary.md"
    metrics_path = run_dir / "metrics.json"
    result["summary_path"] = str(summary_path.resolve())
    result["metrics_path"] = str(metrics_path.resolve())
    _write_circular_summary(summary_path, result)
    _write_json(metrics_path, result)
    _copy_latest_artifacts(result)
    return result


def run_circular_ocr_precommit_gate(strategy_name: str | None = None) -> dict:
    result = run_circular_ocr_experiment(strategy_name=strategy_name)
    if not result["passed"]:
        raise AssertionError(
            f"Circular OCR gate failed: {result['failure_message']}\n"
            f"Artifacts: {result['summary_path']} {result['metrics_path']}"
        )
    return result
