from __future__ import annotations

import csv
import json
import shutil
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch

TESTS_ROOT = Path(__file__).resolve().parent
APP_ROOT = TESTS_ROOT.parent
REPO_ROOT = APP_ROOT.parent
LOGS_ROOT = TESTS_ROOT / "logs"

if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recognition.active_learning import fine_tune_checkpoint_on_pages, generate_prediction_pagexmls, prepare_page_datasets
from tests.evaluate import evaluate_dataset
from tests.precommit_gate_config import get_recognition_precommit_dataset
from tests.recognition_finetuning_config import (
    RecognitionEvalDatasetConfig,
    get_page_plus_random_history_policy_configs,
    get_precommit_hybrid_recognition_gate_config,
)


PRETRAINED_OCR_CHECKPOINT = APP_ROOT / "recognition" / "pretrained_model" / "vadakautuhala.pth"
RECOGNITION_PRECOMMIT_LATEST_BASENAME = "recognition_finetune_precommit_latest"


def _timestamp_slug():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _copy_eval_ground_truth_subset(gt_dir, page_ids, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    for page_id in page_ids:
        shutil.copy2(gt_dir / f"{page_id}.xml", output_dir / f"{page_id}.xml")
    return output_dir


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _format_lr(value: float) -> str:
    return format(float(value), "g")


def _micro_lr_token(value: float) -> str:
    return f"lr{int(round(float(value) * 1_000_000)):06d}u"


def _page_plus_history_policy_slug(config: RecognitionEvalDatasetConfig) -> str:
    width_token = {"global_2000_pad": "wg", "batch_max_pad": "wb"}[config.width_policy]
    oversampling_token = {"none": "on", "cer_weighted": "oc"}[config.oversampling_policy]
    augmentation_token = {
        "none": "an",
        "background_only": "ab",
        "background_plus_rotation": "ar",
    }[config.augmentation_policy]
    scheduler_token = {"none": "sn", "step": "ss", "cosine": "sc"}[config.lr_scheduler]
    optimizer_token = {"adadelta": "optd", "adam": "opta"}[config.optimizer]
    history_token = f"hist{int(config.history_sample_line_count):02d}"
    return (
        f"{width_token}_{oversampling_token}_{augmentation_token}_{history_token}_{scheduler_token}_"
        f"{optimizer_token}_{_micro_lr_token(float(config.training_overrides['lr']))}"
    )


def _policy_descriptor(config: RecognitionEvalDatasetConfig):
    return {
        "training_policy": config.training_policy,
        "history_sample_line_count": int(config.history_sample_line_count),
        "width_policy": config.width_policy,
        "oversampling_policy": config.oversampling_policy,
        "augmentation_policy": config.augmentation_policy,
        "lr_scheduler": config.lr_scheduler,
        "optimizer": config.optimizer,
        "lr": float(config.training_overrides["lr"]),
        "num_iter": int(config.training_overrides["num_iter"]),
        "curve_metric": config.curve_metric,
        "regression_guard_abs": config.regression_guard_abs,
        "background_plus_rotation_variant_count": config.background_plus_rotation_variant_count,
        "shuffle_train_each_epoch": config.shuffle_train_each_epoch,
    }


def _aggregate_length_bucket_metrics(per_line_rows):
    aggregate = {
        bucket: {
            "edit_distance": 0,
            "gt_length": 0,
            "line_count": 0,
        }
        for bucket in ("short", "medium", "long")
    }

    for row in per_line_rows:
        bucket = row["length_bucket"]
        if bucket not in aggregate:
            continue
        aggregate[bucket]["edit_distance"] += int(row["edit_distance"])
        aggregate[bucket]["gt_length"] += int(row["gt_length"])
        aggregate[bucket]["line_count"] += 1

    for bucket_metrics in aggregate.values():
        gt_length = bucket_metrics["gt_length"]
        bucket_metrics["line_cer"] = bucket_metrics["edit_distance"] / max(gt_length, 1)

    return aggregate


def _compute_curve_metrics(steps, dataset_config: RecognitionEvalDatasetConfig):
    if not steps:
        return {
            "curve_metric_name": dataset_config.curve_metric,
            "curve_metric_value": None,
            "regression_guard_abs": dataset_config.regression_guard_abs,
            "regression_guard_passed": False,
            "max_regression": None,
            "first_step_gain": None,
            "final_page_cer": None,
        }

    k_value = len(steps) - 1
    weights = [k_value - step["step_index"] + 1 for step in steps]
    page_cers = [step["metrics"]["aggregate_metrics"]["page_cer"] for step in steps]
    weighted_sum = sum(weight * page_cer for weight, page_cer in zip(weights, page_cers))
    weight_total = sum(weights)
    max_regression = max((max(step.get("delta_page_cer") or 0.0, 0.0) for step in steps[1:]), default=0.0)
    first_step_gain = None
    if len(steps) > 1:
        first_step_gain = steps[0]["metrics"]["aggregate_metrics"]["page_cer"] - steps[1]["metrics"]["aggregate_metrics"]["page_cer"]

    return {
        "curve_metric_name": dataset_config.curve_metric,
        "curve_metric_value": weighted_sum / max(weight_total, 1),
        "regression_guard_abs": dataset_config.regression_guard_abs,
        "regression_guard_passed": max_regression <= dataset_config.regression_guard_abs,
        "max_regression": max_regression,
        "first_step_gain": first_step_gain,
        "final_page_cer": steps[-1]["metrics"]["aggregate_metrics"]["page_cer"],
        "per_step_train_seconds": [step["train_seconds"] for step in steps],
        "per_step_page_cer": page_cers,
    }


def _plot_metric_curve(path, policy_slug, steps):
    x_vals = [step["page_count_finetuned"] for step in steps]
    page_cer = [step["metrics"]["aggregate_metrics"]["page_cer"] for step in steps]
    line_cer = [step["metrics"]["aggregate_metrics"]["line_cer_50"] for step in steps]

    plt.figure(figsize=(8, 5))
    plt.plot(x_vals, page_cer, marker="o", linewidth=2, label="Page CER")
    plt.plot(x_vals, line_cer, marker="s", linewidth=2, label="Line CER @ IoU 0.50")
    plt.xlabel("Sequential Fine-Tuning Pages")
    plt.ylabel("CER")
    plt.title(f"Recognition CER vs Sequential Fine-Tuning Data ({policy_slug})")
    plt.grid(True, alpha=0.3)
    plt.xticks(x_vals)
    plt.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _summarize_checkpoint_path(path, run_dir: Path) -> str:
    resolved_path = Path(path).resolve()
    try:
        return str(resolved_path.relative_to(run_dir.resolve()))
    except ValueError:
        return str(resolved_path)


def _write_policy_summary(path, status, policy_slug, dataset_config, steps, curve_metrics, failure_message, warnings=None):
    run_dir = path.parent
    warnings = list(warnings or [])
    lines = [
        f"# Recognition Fine-Tuning Policy Run: {policy_slug}",
        "",
        f"Status: **{status.upper()}**",
        "",
        "## Policy",
        "",
        f"- training_policy={dataset_config.training_policy}",
        f"- history_sample_line_count={int(dataset_config.history_sample_line_count)}",
        f"- width_policy={dataset_config.width_policy}",
        f"- oversampling_policy={dataset_config.oversampling_policy}",
        f"- augmentation_policy={dataset_config.augmentation_policy}",
        f"- lr_scheduler={dataset_config.lr_scheduler}",
        f"- optimizer={dataset_config.optimizer}",
        f"- lr={_format_lr(float(dataset_config.training_overrides['lr']))}",
        f"- num_iter={int(dataset_config.training_overrides['num_iter'])}",
        f"- curve_metric={dataset_config.curve_metric}",
        f"- regression_guard_abs={dataset_config.regression_guard_abs:.3f}",
        f"- background_plus_rotation_variant_count={dataset_config.background_plus_rotation_variant_count}",
        f"- shuffle_train_each_epoch={dataset_config.shuffle_train_each_epoch}",
        "",
        "## Curve",
        "",
        f"- curve_metric_value={curve_metrics['curve_metric_value']}",
        f"- regression_guard_passed={curve_metrics['regression_guard_passed']}",
        f"- max_regression={curve_metrics['max_regression']}",
        f"- first_step_gain={curve_metrics['first_step_gain']}",
        f"- final_page_cer={curve_metrics['final_page_cer']}",
        "",
        "## Steps",
        "",
    ]

    for step in steps:
        metrics = step["metrics"]["aggregate_metrics"]
        length_buckets = step["length_bucket_metrics"]
        lines.append(
            f"- Step {step['step_index']} ({step['step_label']}): "
            f"page_cer={metrics['page_cer']:.4f}, "
            f"line_cer_50={metrics['line_cer_50']:.4f}, "
            f"delta_page_cer={step['delta_page_cer']}, "
            f"guard_passed={step['regression_guard_passed']}, "
            f"training_policy={step['training_policy']}, "
            f"train_pages={step['training_page_ids']}, "
            f"train_dataset_page_count={step['train_dataset_page_count']}, "
            f"history_source_pages={step.get('history_source_page_ids', [])}, "
            f"history_lines={step.get('train_dataset_history_line_count', 0)}, "
            f"history_pages={step.get('train_dataset_history_page_ids', [])}, "
            f"optimizer={step['optimizer_name']}, "
            f"shuffle={step['shuffle_train_each_epoch']}, "
            f"train_materialized={step['train_materialized_count']}, "
            f"short={length_buckets['short']['line_cer']:.4f}, "
            f"medium={length_buckets['medium']['line_cer']:.4f}, "
            f"long={length_buckets['long']['line_cer']:.4f}, "
            f"selector={step.get('selected_best_model') or '-'}, "
            f"base_checkpoint={_summarize_checkpoint_path(step['base_checkpoint'], run_dir)}, "
            f"output_checkpoint={_summarize_checkpoint_path(step['output_checkpoint'], run_dir)}"
        )

    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)

    if failure_message:
        lines.extend(["", "## Failure", "", failure_message])

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _prepare_study_inputs(dataset_config: RecognitionEvalDatasetConfig, study_slug: str | None = None):
    ordered_page_ids = dataset_config.ordered_page_ids()
    evaluation_page_ids = dataset_config.evaluation_page_ids()

    if not PRETRAINED_OCR_CHECKPOINT.exists():
        raise FileNotFoundError(f"Missing OCR checkpoint: {PRETRAINED_OCR_CHECKPOINT}")

    LOGS_ROOT.mkdir(parents=True, exist_ok=True)
    study_prefix = f"{study_slug}_" if study_slug else ""
    run_dir = LOGS_ROOT / f"{_timestamp_slug()}_ocrft_{study_prefix}{dataset_config.name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    prepared_pages = prepare_page_datasets(
        dataset_config.images_dir,
        dataset_config.pagexml_dir,
        ordered_page_ids,
        run_dir / "prepared_pages",
    )
    evaluation_pages = {page_id: prepared_pages[page_id] for page_id in evaluation_page_ids}
    gt_subset_dir = _copy_eval_ground_truth_subset(dataset_config.pagexml_dir, evaluation_page_ids, run_dir / "gt_eval_subset")

    return run_dir, prepared_pages, evaluation_pages, gt_subset_dir


def _run_single_policy_run(
    run_dir: Path,
    dataset_config: RecognitionEvalDatasetConfig,
    prepared_pages,
    evaluation_pages,
    gt_subset_dir: Path,
    slug_builder=_page_plus_history_policy_slug,
    regression_guard_mode: str = "fail",
):
    if regression_guard_mode not in {"fail", "warn"}:
        raise ValueError(f"Unsupported regression_guard_mode: {regression_guard_mode}")

    run_dir.mkdir(parents=True, exist_ok=True)
    policy_slug = slug_builder(dataset_config)
    config_payload = {
        "dataset": dataset_config.to_dict(),
        "policy_slug": policy_slug,
        "policy": _policy_descriptor(dataset_config),
        "pretrained_checkpoint": str(PRETRAINED_OCR_CHECKPOINT.resolve()),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "python_executable": sys.executable,
    }
    _write_json(run_dir / "config.json", config_payload)

    fine_tune_page_ids = dataset_config.fine_tune_page_ids()
    steps = []
    per_page_rows = []
    per_line_rows = []
    fine_tune_metadata = []
    selector_metrics = []
    status = "passed"
    failure_message = ""
    warnings = []
    current_checkpoint = PRETRAINED_OCR_CHECKPOINT
    previous_page_cer = None

    try:
        for step_index in range(0, len(fine_tune_page_ids) + 1):
            if step_index == 0:
                step_label = "pretrained"
                train_page_id = None
                training_page_ids = []
                history_source_page_ids = []
                train_dataset_page_count = 0
                train_dataset_current_page_line_count = 0
                train_dataset_history_line_count = 0
                train_dataset_history_page_ids = []
                base_checkpoint = str(Path(PRETRAINED_OCR_CHECKPOINT).resolve())
                output_checkpoint = base_checkpoint
                train_seconds = 0.0
                train_sample_count = 0
                val_sample_count = 0
                logical_train_sample_count = 0
                logical_val_sample_count = 0
                train_materialized_count = 0
                val_materialized_count = 0
                train_variant_labels = []
                val_variant_labels = []
                selected_best_model = None
                step_selector_metrics = None
                optimizer_name = dataset_config.optimizer
                background_plus_rotation_variant_count = dataset_config.background_plus_rotation_variant_count
                shuffle_train_each_epoch = dataset_config.shuffle_train_each_epoch
                training_summary = None
            else:
                train_page_id = fine_tune_page_ids[step_index - 1]
                step_label = f"page_{train_page_id}"
                if dataset_config.training_policy != "page_plus_random_history":
                    raise ValueError(f"Unsupported training policy: {dataset_config.training_policy}")
                training_page_ids = [train_page_id]
                history_source_page_ids = fine_tune_page_ids[: step_index - 1]
                history_source_pages = [prepared_pages[page_id] for page_id in history_source_page_ids]
                history_sample_line_count = dataset_config.history_sample_line_count

                train_dataset_page_count = len(training_page_ids)
                train_dataset_current_page_line_count = 0
                train_dataset_history_line_count = 0
                train_dataset_history_page_ids = []
                training_overrides = dict(dataset_config.training_overrides)
                training_overrides["width_policy"] = dataset_config.width_policy
                training_overrides["lr_scheduler"] = dataset_config.lr_scheduler
                training_overrides["optimizer_name"] = dataset_config.optimizer
                training_overrides["background_plus_rotation_variant_count"] = (
                    dataset_config.background_plus_rotation_variant_count
                )
                training_overrides["shuffle_train_each_epoch"] = dataset_config.shuffle_train_each_epoch

                base_checkpoint = str(Path(current_checkpoint).resolve())
                finetune_result = fine_tune_checkpoint_on_pages(
                    [prepared_pages[page_id] for page_id in training_page_ids],
                    current_checkpoint,
                    run_dir / "models" / f"step_{step_index:02d}_{train_page_id}",
                    step_index=step_index,
                    validation_ratio=dataset_config.validation_ratio,
                    split_seed=dataset_config.split_seed,
                    oversampling_policy=dataset_config.oversampling_policy,
                    augmentation_policy=dataset_config.augmentation_policy,
                    history_source_pages=history_source_pages,
                    history_sample_line_count=history_sample_line_count,
                    **training_overrides,
                )
                selector_metrics.append(
                    {
                        "step_index": step_index,
                        "step_label": step_label,
                        "selector_metrics": finetune_result.selector_metrics,
                    }
                )
                output_checkpoint = finetune_result.output_checkpoint
                current_checkpoint = Path(output_checkpoint)
                train_seconds = finetune_result.train_seconds
                train_sample_count = finetune_result.train_sample_count
                val_sample_count = finetune_result.val_sample_count
                logical_train_sample_count = finetune_result.logical_train_sample_count
                logical_val_sample_count = finetune_result.logical_val_sample_count
                train_materialized_count = finetune_result.train_materialized_count
                val_materialized_count = finetune_result.val_materialized_count
                train_variant_labels = finetune_result.train_variant_labels
                val_variant_labels = finetune_result.val_variant_labels
                selected_best_model = finetune_result.selected_best_model
                step_selector_metrics = finetune_result.selector_metrics
                optimizer_name = finetune_result.optimizer_name
                background_plus_rotation_variant_count = finetune_result.background_plus_rotation_variant_count
                shuffle_train_each_epoch = finetune_result.shuffle_train_each_epoch
                training_summary = finetune_result.training_summary
                dataset_manifest = finetune_result.dataset_manifest or {}
                training_page_ids = list(dataset_manifest.get("page_ids", training_page_ids))
                history_source_page_ids = list(dataset_manifest.get("history_source_page_ids", history_source_page_ids))
                train_dataset_page_count = len(training_page_ids)
                train_dataset_current_page_line_count = int(dataset_manifest.get("current_page_line_count", 0))
                train_dataset_history_line_count = int(dataset_manifest.get("history_sample_line_count", 0))
                train_dataset_history_page_ids = list(dataset_manifest.get("history_sample_page_ids", []))
                fine_tune_metadata.append(
                    {
                        **asdict(finetune_result),
                        "training_policy": dataset_config.training_policy,
                        "train_dataset_page_count": train_dataset_page_count,
                        "history_source_page_ids": history_source_page_ids,
                        "train_dataset_current_page_line_count": train_dataset_current_page_line_count,
                        "train_dataset_history_line_count": train_dataset_history_line_count,
                        "train_dataset_history_page_ids": train_dataset_history_page_ids,
                    }
                )

            prediction_output = generate_prediction_pagexmls(
                current_checkpoint,
                evaluation_pages,
                run_dir / "predicted_page_xml" / f"step_{step_index:02d}_{step_label}",
                width_policy=dataset_config.width_policy,
            )
            metrics = evaluate_dataset(
                pred_folder=prediction_output.prediction_folder,
                gt_folder=gt_subset_dir,
                method_name=f"recognition-finetune-step-{step_index}",
                layout_type=dataset_config.layout_type,
            )

            aggregate_page_cer = metrics["aggregate_metrics"]["page_cer"]
            if previous_page_cer is None:
                improved = None
                delta_page_cer = None
                regression_guard_passed = True
            else:
                improved = aggregate_page_cer < previous_page_cer
                delta_page_cer = aggregate_page_cer - previous_page_cer
                regression_guard_passed = delta_page_cer <= dataset_config.regression_guard_abs

            step_line_rows = []
            for line_row in prediction_output.per_line_predictions:
                step_line_rows.append(
                    {
                        "step_index": step_index,
                        "step_label": step_label,
                        "train_page_id": train_page_id or "",
                        "page_id": line_row["page_id"],
                        "line_id": line_row["line_id"],
                        "line_custom": line_row["line_custom"],
                        "gt_length": line_row["gt_length"],
                        "predicted_text": line_row["predicted_text"],
                        "edit_distance": line_row["edit_distance"],
                        "line_cer": line_row["line_cer"],
                        "confidence": line_row["confidence"],
                        "resized_width": line_row["resized_width"],
                        "pad_fraction": line_row["pad_fraction"],
                        "length_bucket": line_row["length_bucket"],
                    }
                )
            per_line_rows.extend(step_line_rows)
            length_bucket_metrics = _aggregate_length_bucket_metrics(step_line_rows)

            step_record = {
                "step_index": step_index,
                "step_label": step_label,
                "page_count_finetuned": step_index,
                "train_page_id": train_page_id,
                "training_policy": dataset_config.training_policy,
                "training_page_ids": training_page_ids,
                "history_source_page_ids": history_source_page_ids,
                "train_dataset_page_count": train_dataset_page_count,
                "train_dataset_current_page_line_count": train_dataset_current_page_line_count,
                "train_dataset_history_line_count": train_dataset_history_line_count,
                "train_dataset_history_page_ids": train_dataset_history_page_ids,
                "base_checkpoint": base_checkpoint,
                "checkpoint_path": str(Path(output_checkpoint).resolve()),
                "output_checkpoint": str(Path(output_checkpoint).resolve()),
                "train_seconds": train_seconds,
                "train_sample_count": train_sample_count,
                "val_sample_count": val_sample_count,
                "logical_train_sample_count": logical_train_sample_count,
                "logical_val_sample_count": logical_val_sample_count,
                "train_materialized_count": train_materialized_count,
                "val_materialized_count": val_materialized_count,
                "train_variant_labels": train_variant_labels,
                "val_variant_labels": val_variant_labels,
                "optimizer_name": optimizer_name,
                "shuffle_train_each_epoch": shuffle_train_each_epoch,
                "background_plus_rotation_variant_count": background_plus_rotation_variant_count,
                "training_summary": training_summary,
                "inference_seconds": prediction_output.total_seconds,
                "prediction_folder": prediction_output.prediction_folder,
                "metrics": metrics,
                "improved_over_previous": improved,
                "delta_page_cer": delta_page_cer,
                "regression_guard_passed": regression_guard_passed,
                "length_bucket_metrics": length_bucket_metrics,
                "selected_best_model": selected_best_model,
                "selector_metrics": step_selector_metrics,
            }
            steps.append(step_record)

            inference_seconds_by_page = prediction_output.per_page_seconds
            for page_result in metrics["per_page"]:
                per_page_rows.append(
                    {
                        "step_index": step_index,
                        "step_label": step_label,
                        "train_page_id": train_page_id or "",
                        "eval_page_id": page_result["filename"],
                        "page_cer": page_result["page_cer"],
                        "line_cer_50": page_result["line_cer_50"],
                        "gt_len": page_result["gt_len"],
                        "prediction_found": page_result["prediction_found"],
                        "inference_seconds": inference_seconds_by_page.get(page_result["filename"], 0.0),
                    }
                )

            if not regression_guard_passed:
                warning_message = (
                    f"Step {step_index} ({step_label}) regressed aggregate page CER by "
                    f"{delta_page_cer:.6f}, exceeding {dataset_config.regression_guard_abs:.6f}."
                )
                if regression_guard_mode == "fail":
                    raise AssertionError(warning_message)
                warnings.append(
                    f"Regression guard warning only: {warning_message}"
                )

            previous_page_cer = aggregate_page_cer
    except Exception as exc:
        status = "failed"
        failure_message = str(exc)

    curve_metrics = _compute_curve_metrics(steps, dataset_config)

    summary_path = run_dir / "summary.md"
    metrics_path = run_dir / "metrics.json"
    curve_metrics_path = run_dir / "curve_metrics.json"
    per_page_csv_path = run_dir / "per_page.csv"
    per_line_csv_path = run_dir / "per_line.csv"
    fine_tune_metadata_path = run_dir / "fine_tune_metadata.json"
    selector_metrics_path = run_dir / "selector_metrics.json"
    plot_path = run_dir / "plots" / "page_cer_vs_finetune_pages.png"

    _write_policy_summary(
        summary_path,
        status,
        policy_slug,
        dataset_config,
        steps,
        curve_metrics,
        failure_message,
        warnings=warnings,
    )
    _write_json(
        metrics_path,
        {
            "status": status,
            "failure_message": failure_message,
            "warnings": warnings,
            "run_dir": str(run_dir.resolve()),
            "config": config_payload,
            "policy": _policy_descriptor(dataset_config),
            "steps": steps,
        },
    )
    _write_json(curve_metrics_path, curve_metrics)
    _write_json(
        fine_tune_metadata_path,
        {
            "policy_slug": policy_slug,
            "policy": _policy_descriptor(dataset_config),
            "runs": fine_tune_metadata,
        },
    )
    _write_json(selector_metrics_path, {"policy_slug": policy_slug, "steps": selector_metrics})
    _write_csv(
        per_page_csv_path,
        [
            "step_index",
            "step_label",
            "train_page_id",
            "eval_page_id",
            "page_cer",
            "line_cer_50",
            "gt_len",
            "prediction_found",
            "inference_seconds",
        ],
        per_page_rows,
    )
    _write_csv(
        per_line_csv_path,
        [
            "step_index",
            "step_label",
            "train_page_id",
            "page_id",
            "line_id",
            "line_custom",
            "gt_length",
            "predicted_text",
            "edit_distance",
            "line_cer",
            "confidence",
            "resized_width",
            "pad_fraction",
            "length_bucket",
        ],
        per_line_rows,
    )
    if steps:
        _plot_metric_curve(plot_path, policy_slug, steps)

    return {
        "status": status,
        "failure_message": failure_message,
        "warnings": warnings,
        "run_dir": run_dir,
        "summary_path": summary_path,
        "metrics_path": metrics_path,
        "curve_metrics_path": curve_metrics_path,
        "per_page_csv_path": per_page_csv_path,
        "per_line_csv_path": per_line_csv_path,
        "fine_tune_metadata_path": fine_tune_metadata_path,
        "selector_metrics_path": selector_metrics_path,
        "plot_path": plot_path,
        "steps": steps,
        "curve_metrics": curve_metrics,
        "policy": _policy_descriptor(dataset_config),
        "policy_slug": policy_slug,
    }


def _winning_policy_entry(policy_result, metric_key, metric_name):
    return {
        "policy_slug": policy_result["policy_slug"],
        "metric_name": metric_name,
        "metric_value": policy_result["curve_metrics"][metric_key],
        "curve_metrics": policy_result["curve_metrics"],
    }


def _curve_metric_sort_key(policy_result):
    curve_metrics = policy_result["curve_metrics"]
    return (
        curve_metrics["curve_metric_value"],
        curve_metrics["final_page_cer"],
        curve_metrics["max_regression"],
        policy_result["policy_slug"],
    )


def _final_page_cer_sort_key(policy_result):
    curve_metrics = policy_result["curve_metrics"]
    return (
        curve_metrics["final_page_cer"],
        curve_metrics["max_regression"],
        policy_result["policy_slug"],
    )


def _first_step_gain_sort_key(policy_result):
    curve_metrics = policy_result["curve_metrics"]
    first_step_gain = curve_metrics["first_step_gain"]
    if first_step_gain is None:
        return (float("inf"), curve_metrics["final_page_cer"], curve_metrics["max_regression"], policy_result["policy_slug"])
    return (
        -first_step_gain,
        curve_metrics["final_page_cer"],
        curve_metrics["max_regression"],
        policy_result["policy_slug"],
    )


def _select_winning_policies_by_metric(policy_results, dataset_config: RecognitionEvalDatasetConfig):
    passed_results = [policy_result for policy_result in policy_results if policy_result["status"] == "passed"]
    if not passed_results:
        return {}

    primary_winner = min(passed_results, key=_curve_metric_sort_key)
    final_page_winner = min(passed_results, key=_final_page_cer_sort_key)
    first_step_gain_winner = min(passed_results, key=_first_step_gain_sort_key)

    return {
        "primary_curve_metric": _winning_policy_entry(
            primary_winner,
            "curve_metric_value",
            dataset_config.curve_metric,
        ),
        "final_page_cer": _winning_policy_entry(
            final_page_winner,
            "final_page_cer",
            "final_page_cer",
        ),
        "first_step_gain": _winning_policy_entry(
            first_step_gain_winner,
            "first_step_gain",
            "first_step_gain",
        ),
    }


def _write_policy_study_summary(path, dataset_config, policy_results, winning_policies_by_metric, study_mode):
    primary_winner = winning_policies_by_metric.get("primary_curve_metric", {})
    lines = [
        f"# Recognition Fine-Tuning Study: {dataset_config.name}",
        "",
        f"Study mode: `{study_mode}`",
        "",
        f"Policy run count: {len(policy_results)}",
        f"Winning policy: `{primary_winner.get('policy_slug', '-')}`",
        "",
        "## Winners",
        "",
    ]

    for metric_name in ("primary_curve_metric", "final_page_cer", "first_step_gain"):
        winner = winning_policies_by_metric.get(metric_name)
        if not winner:
            lines.append(f"- {metric_name}: no passed policy runs")
            continue
        lines.append(
            f"- {metric_name}: `{winner['policy_slug']}` "
            f"({winner['metric_name']}={winner['metric_value']})"
        )

    lines.extend(["", "## Policy Runs", ""])
    for policy_result in policy_results:
        curve_metrics = policy_result["curve_metrics"]
        policy = policy_result["policy"]
        lines.append(
            f"- `{policy_result['policy_slug']}`: status={policy_result['status']}, "
            f"optimizer={policy['optimizer']}, "
            f"lr={_format_lr(policy['lr'])}, "
            f"num_iter={policy['num_iter']}, "
            f"history_sample_line_count={policy.get('history_sample_line_count', 0)}, "
            f"curve_metric={curve_metrics['curve_metric_value']}, "
            f"final_page_cer={curve_metrics['final_page_cer']}, "
            f"first_step_gain={curve_metrics['first_step_gain']}"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _copy_latest_artifacts(
    run_dir,
    summary_path,
    metrics_path,
    plot_path,
    latest_basename: str = "recognition_finetune_results_latest",
):
    shutil.copy2(summary_path, LOGS_ROOT / f"{latest_basename}.md")
    shutil.copy2(metrics_path, LOGS_ROOT / f"{latest_basename}.json")
    if plot_path.exists():
        shutil.copy2(plot_path, LOGS_ROOT / f"{latest_basename}.png")
    (LOGS_ROOT / f"{latest_basename}.txt").write_text(
        f"Latest run: {run_dir.resolve()}\n",
        encoding="utf-8",
    )


def _threshold_result(metric_name: str, observed, operator: str, threshold: float) -> dict:
    if operator == "<=":
        passed = observed is not None and observed <= threshold
    elif operator == ">=":
        passed = observed is not None and observed >= threshold
    else:
        raise ValueError(f"Unsupported threshold operator: {operator}")

    return {
        "metric_name": metric_name,
        "observed": observed,
        "operator": operator,
        "threshold": threshold,
        "passed": passed,
    }


def _build_recognition_precommit_threshold_results(curve_metrics: dict, gate_config) -> dict:
    return {
        "curve_metric_value": _threshold_result(
            gate_config.recipe.curve_metric,
            curve_metrics.get("curve_metric_value"),
            "<=",
            gate_config.max_curve_metric_value,
        ),
        "final_page_cer": _threshold_result(
            "final_page_cer",
            curve_metrics.get("final_page_cer"),
            "<=",
            gate_config.max_final_page_cer,
        ),
        "first_step_gain": _threshold_result(
            "first_step_gain",
            curve_metrics.get("first_step_gain"),
            ">=",
            gate_config.min_first_step_gain,
        ),
    }


def _recognition_precommit_threshold_failure_message(threshold_results: dict) -> str:
    failed_checks = []
    for result in threshold_results.values():
        if result["passed"]:
            continue
        failed_checks.append(
            f"{result['metric_name']}={result['observed']} must be {result['operator']} {result['threshold']}"
        )
    if not failed_checks:
        return ""
    return "Recognition fine-tuning gate threshold failure: " + "; ".join(failed_checks)


def _build_recognition_precommit_dataset_result(dataset_name: str, policy_result: dict) -> dict:
    gate_config = get_recognition_precommit_dataset(dataset_name)
    curve_metrics = policy_result["curve_metrics"]
    threshold_results = _build_recognition_precommit_threshold_results(curve_metrics, gate_config)
    blocking_thresholds_passed = all(result["passed"] for result in threshold_results.values())
    warnings = list(policy_result.get("warnings", []))

    if gate_config.regression_guard_warning_only and not curve_metrics.get("regression_guard_passed", False):
        warning_summary = (
            f"Regression guard warning only: max_regression={curve_metrics.get('max_regression')} "
            f"exceeded {gate_config.recipe.regression_guard_abs}."
        )
        if warning_summary not in warnings:
            warnings.append(warning_summary)

    if policy_result["status"] != "passed":
        status = "failed"
        failure_message = policy_result["failure_message"] or "Recognition fine-tuning policy run failed."
    elif blocking_thresholds_passed:
        status = "passed"
        failure_message = ""
    else:
        status = "failed"
        failure_message = _recognition_precommit_threshold_failure_message(threshold_results)

    return {
        "study_mode": "recognition_precommit_gate",
        "dataset_name": dataset_name,
        "status": status,
        "passed": status == "passed",
        "failure_message": failure_message,
        "policy_slug": policy_result["policy_slug"],
        "policy": policy_result["policy"],
        "curve_metrics": curve_metrics,
        "threshold_results": threshold_results,
        "blocking_thresholds_passed": blocking_thresholds_passed,
        "regression_guard_warning_only": gate_config.regression_guard_warning_only,
        "warnings": warnings,
        "policy_run_dir": policy_result["run_dir"],
        "policy_summary_path": policy_result["summary_path"],
        "policy_metrics_path": policy_result["metrics_path"],
        "curve_metrics_path": policy_result["curve_metrics_path"],
        "per_page_csv_path": policy_result["per_page_csv_path"],
        "per_line_csv_path": policy_result["per_line_csv_path"],
        "fine_tune_metadata_path": policy_result["fine_tune_metadata_path"],
        "selector_metrics_path": policy_result["selector_metrics_path"],
        "plot_path": policy_result["plot_path"],
    }


def _write_recognition_precommit_summary(path: Path, dataset_result: dict) -> None:
    policy = dataset_result["policy"]
    curve_metrics = dataset_result["curve_metrics"]
    threshold_results = dataset_result["threshold_results"]

    lines = [
        f"# Recognition Fine-Tuning Pre-Commit Gate: {dataset_result['dataset_name']}",
        "",
        f"Status: **{dataset_result['status'].upper()}**",
        "",
        f"Policy slug: `{dataset_result['policy_slug']}`",
        "",
        "## Policy",
        "",
        f"- training_policy={policy['training_policy']}",
        f"- history_sample_line_count={policy['history_sample_line_count']}",
        f"- width_policy={policy['width_policy']}",
        f"- oversampling_policy={policy['oversampling_policy']}",
        f"- augmentation_policy={policy['augmentation_policy']}",
        f"- lr_scheduler={policy['lr_scheduler']}",
        f"- optimizer={policy['optimizer']}",
        f"- lr={_format_lr(policy['lr'])}",
        f"- num_iter={policy['num_iter']}",
        f"- curve_metric={policy['curve_metric']}",
        f"- regression_guard_abs={policy['regression_guard_abs']}",
        f"- regression_guard_warning_only={dataset_result['regression_guard_warning_only']}",
        "",
        "## Metrics",
        "",
        f"- curve_metric_value={curve_metrics['curve_metric_value']}",
        f"- final_page_cer={curve_metrics['final_page_cer']}",
        f"- first_step_gain={curve_metrics['first_step_gain']}",
        f"- regression_guard_passed={curve_metrics['regression_guard_passed']}",
        f"- max_regression={curve_metrics['max_regression']}",
        "",
        "## Thresholds",
        "",
    ]

    for threshold_name, result in threshold_results.items():
        lines.append(
            f"- {threshold_name}: observed={result['observed']}, "
            f"required {result['operator']} {result['threshold']}, "
            f"passed={result['passed']}"
        )

    if dataset_result["warnings"]:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in dataset_result["warnings"])

    if dataset_result["failure_message"]:
        lines.extend(["", "## Failure", "", dataset_result["failure_message"]])

    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- policy_summary={dataset_result['policy_summary_path'].resolve()}",
            f"- policy_metrics={dataset_result['policy_metrics_path'].resolve()}",
            f"- fine_tune_metadata={dataset_result['fine_tune_metadata_path'].resolve()}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _jsonable_recognition_precommit_dataset_result(dataset_result: dict) -> dict:
    return {
        "study_mode": dataset_result["study_mode"],
        "dataset_name": dataset_result["dataset_name"],
        "status": dataset_result["status"],
        "passed": dataset_result["passed"],
        "failure_message": dataset_result["failure_message"],
        "policy_slug": dataset_result["policy_slug"],
        "policy": dataset_result["policy"],
        "curve_metrics": dataset_result["curve_metrics"],
        "threshold_results": dataset_result["threshold_results"],
        "blocking_thresholds_passed": dataset_result["blocking_thresholds_passed"],
        "regression_guard_warning_only": dataset_result["regression_guard_warning_only"],
        "warnings": dataset_result["warnings"],
        "policy_run_dir": str(dataset_result["policy_run_dir"].resolve()),
        "policy_summary_path": str(dataset_result["policy_summary_path"].resolve()),
        "policy_metrics_path": str(dataset_result["policy_metrics_path"].resolve()),
        "curve_metrics_path": str(dataset_result["curve_metrics_path"].resolve()),
        "per_page_csv_path": str(dataset_result["per_page_csv_path"].resolve()),
        "per_line_csv_path": str(dataset_result["per_line_csv_path"].resolve()),
        "fine_tune_metadata_path": str(dataset_result["fine_tune_metadata_path"].resolve()),
        "selector_metrics_path": str(dataset_result["selector_metrics_path"].resolve()),
        "plot_path": str(dataset_result["plot_path"].resolve()),
    }


def run_recognition_finetuning_experiment(dataset_name="eval_dataset"):
    return run_page_plus_random_history_experiment(dataset_name=dataset_name)


def run_page_plus_random_history_experiment(dataset_name="eval_dataset"):
    policy_configs = get_page_plus_random_history_policy_configs(dataset_name)
    if not policy_configs:
        raise AssertionError("Hybrid page-plus-history study requires at least one explicit policy config.")

    dataset_config = policy_configs[0]
    run_dir, prepared_pages, evaluation_pages, gt_subset_dir = _prepare_study_inputs(
        dataset_config,
        study_slug="pagehist",
    )

    policy_results = []
    for policy_config in policy_configs:
        policy_slug = _page_plus_history_policy_slug(policy_config)
        policy_results.append(
            _run_single_policy_run(
                run_dir / "policies" / policy_slug,
                policy_config,
                prepared_pages,
                evaluation_pages,
                gt_subset_dir,
                slug_builder=_page_plus_history_policy_slug,
            )
        )

    winning_policies_by_metric = _select_winning_policies_by_metric(policy_results, dataset_config)
    primary_winner_slug = winning_policies_by_metric.get("primary_curve_metric", {}).get("policy_slug")
    primary_result = next(
        (policy_result for policy_result in policy_results if policy_result["policy_slug"] == primary_winner_slug),
        None,
    )

    study_summary_path = run_dir / "summary.md"
    study_metrics_path = run_dir / "metrics.json"
    _write_policy_study_summary(
        study_summary_path,
        dataset_config,
        policy_results,
        winning_policies_by_metric,
        study_mode="page_plus_random_history_followup",
    )

    metrics_payload = {
        "study_mode": "page_plus_random_history_followup",
        "dataset_name": dataset_name,
        "run_dir": str(run_dir.resolve()),
        "winning_policy": primary_winner_slug,
        "winning_curve_metrics": primary_result["curve_metrics"] if primary_result else None,
        "winning_policies_by_metric": winning_policies_by_metric,
        "policy_run_count": len(policy_results),
        "failed_policy_runs": [
            policy_result["policy_slug"]
            for policy_result in policy_results
            if policy_result["status"] != "passed"
        ],
        "policy_runs": [
            {
                "policy_slug": policy_result["policy_slug"],
                "status": policy_result["status"],
                "policy": policy_result["policy"],
                "curve_metrics": policy_result["curve_metrics"],
                "metrics_path": str(policy_result["metrics_path"].resolve()),
                "summary_path": str(policy_result["summary_path"].resolve()),
                "fine_tune_metadata_path": str(policy_result["fine_tune_metadata_path"].resolve()),
            }
            for policy_result in policy_results
        ],
    }
    _write_json(study_metrics_path, metrics_payload)
    if primary_result is not None:
        _copy_latest_artifacts(
            run_dir,
            study_summary_path,
            study_metrics_path,
            primary_result["plot_path"],
            latest_basename="recognition_finetune_results_latest",
        )

    if primary_result is None:
        raise AssertionError("Page-plus-history study did not produce any passed policy runs.")

    return {
        "study_mode": "page_plus_random_history_followup",
        "run_dir": run_dir,
        "summary_path": study_summary_path,
        "metrics_path": study_metrics_path,
        "policy_runs": policy_results,
        "winning_policy": primary_winner_slug,
        "winning_policies_by_metric": winning_policies_by_metric,
        "curve_metrics": primary_result["curve_metrics"],
        "steps": primary_result["steps"],
        "failed_policy_runs": metrics_payload["failed_policy_runs"],
    }


def run_recognition_precommit_gate(dataset_name="eval_dataset"):
    gate_config = get_recognition_precommit_dataset(dataset_name)
    dataset_config = get_precommit_hybrid_recognition_gate_config(dataset_name)
    run_dir, prepared_pages, evaluation_pages, gt_subset_dir = _prepare_study_inputs(
        dataset_config,
        study_slug="precommit",
    )

    policy_slug = _page_plus_history_policy_slug(dataset_config)
    policy_result = _run_single_policy_run(
        run_dir / "policy" / policy_slug,
        dataset_config,
        prepared_pages,
        evaluation_pages,
        gt_subset_dir,
        slug_builder=_page_plus_history_policy_slug,
        regression_guard_mode="warn" if gate_config.regression_guard_warning_only else "fail",
    )
    dataset_result = _build_recognition_precommit_dataset_result(dataset_name, policy_result)

    summary_path = run_dir / "summary.md"
    metrics_path = run_dir / "metrics.json"
    dataset_result["run_dir"] = run_dir
    dataset_result["summary_path"] = summary_path
    dataset_result["metrics_path"] = metrics_path

    _write_recognition_precommit_summary(summary_path, dataset_result)
    jsonable_dataset_result = _jsonable_recognition_precommit_dataset_result(dataset_result)
    jsonable_dataset_result["run_dir"] = str(run_dir.resolve())
    jsonable_dataset_result["summary_path"] = str(summary_path.resolve())
    jsonable_dataset_result["metrics_path"] = str(metrics_path.resolve())
    _write_json(
        metrics_path,
        {
            "study_mode": "recognition_precommit_gate",
            "run_dir": str(run_dir.resolve()),
            "dataset_results": {dataset_name: jsonable_dataset_result},
            "failed_datasets": [] if dataset_result["passed"] else [dataset_name],
            "passed_dataset_count": 1 if dataset_result["passed"] else 0,
        },
    )
    _copy_latest_artifacts(
        run_dir,
        summary_path,
        metrics_path,
        policy_result["plot_path"],
        latest_basename=RECOGNITION_PRECOMMIT_LATEST_BASENAME,
    )
    return dataset_result
