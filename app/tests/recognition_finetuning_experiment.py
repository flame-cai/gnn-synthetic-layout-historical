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
from tests.recognition_finetuning_config import RecognitionEvalDatasetConfig, get_dataset_config


PRETRAINED_OCR_CHECKPOINT = APP_ROOT / "recognition" / "pretrained_model" / "vadakautuhala.pth"


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


def _short_float_token(value: float) -> str:
    return f"{int(round(value * 100)):03d}"


def _policy_slug(config: RecognitionEvalDatasetConfig) -> str:
    width_token = {"global_2000_pad": "wg", "batch_max_pad": "wb"}[config.width_policy]
    oversampling_token = {"none": "on", "cer_weighted": "oc"}[config.oversampling_policy]
    augmentation_token = {
        "none": "an",
        "background_only": "ab",
        "background_plus_rotation": "ar",
    }[config.augmentation_policy]
    scheduler_token = {"none": "sn", "step": "ss", "cosine": "sc"}[config.lr_scheduler]
    lr_token = _short_float_token(float(config.training_overrides["lr"]))
    return f"{width_token}_{oversampling_token}_{augmentation_token}_{scheduler_token}{lr_token}"


def _policy_descriptor(config: RecognitionEvalDatasetConfig):
    return {
        "width_policy": config.width_policy,
        "oversampling_policy": config.oversampling_policy,
        "augmentation_policy": config.augmentation_policy,
        "lr_scheduler": config.lr_scheduler,
        "lr": float(config.training_overrides["lr"]),
        "curve_metric": config.curve_metric,
        "regression_guard_abs": config.regression_guard_abs,
    }


def _with_training_lr(config: RecognitionEvalDatasetConfig, lr: float, lr_scheduler: str | None = None):
    updated_overrides = dict(config.training_overrides)
    updated_overrides["lr"] = lr
    changes = {"training_overrides": updated_overrides}
    if lr_scheduler is not None:
        changes["lr_scheduler"] = lr_scheduler
    return config.with_updates(**changes)


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


def _write_policy_summary(path, status, dataset_config, steps, curve_metrics, failure_message):
    lines = [
        f"# Recognition Fine-Tuning Policy Run: {_policy_slug(dataset_config)}",
        "",
        f"Status: **{status.upper()}**",
        "",
        "## Policy",
        "",
        f"- width_policy={dataset_config.width_policy}",
        f"- oversampling_policy={dataset_config.oversampling_policy}",
        f"- augmentation_policy={dataset_config.augmentation_policy}",
        f"- lr_scheduler={dataset_config.lr_scheduler}",
        f"- lr={float(dataset_config.training_overrides['lr']):.3f}",
        f"- curve_metric={dataset_config.curve_metric}",
        f"- regression_guard_abs={dataset_config.regression_guard_abs:.3f}",
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
            f"short={length_buckets['short']['line_cer']:.4f}, "
            f"medium={length_buckets['medium']['line_cer']:.4f}, "
            f"long={length_buckets['long']['line_cer']:.4f}, "
            f"selector={step.get('selected_best_model') or '-'}"
        )

    if failure_message:
        lines.extend(["", "## Failure", "", failure_message])

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_single_policy_run(
    run_dir: Path,
    dataset_config: RecognitionEvalDatasetConfig,
    prepared_pages,
    evaluation_pages,
    gt_subset_dir: Path,
):
    run_dir.mkdir(parents=True, exist_ok=True)
    config_payload = {
        "dataset": dataset_config.to_dict(),
        "policy": _policy_descriptor(dataset_config),
        "pretrained_checkpoint": str(PRETRAINED_OCR_CHECKPOINT.resolve()),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "python_executable": sys.executable,
    }
    _write_json(run_dir / "config.json", config_payload)

    fine_tune_page_ids = dataset_config.fine_tune_page_ids()
    evaluation_page_ids = dataset_config.evaluation_page_ids()
    steps = []
    per_page_rows = []
    per_line_rows = []
    fine_tune_metadata = []
    selector_metrics = []
    status = "passed"
    failure_message = ""
    current_checkpoint = PRETRAINED_OCR_CHECKPOINT
    previous_page_cer = None

    try:
        for step_index in range(0, len(fine_tune_page_ids) + 1):
            if step_index == 0:
                step_label = "pretrained"
                train_page_id = None
                training_page_ids = []
                train_seconds = 0.0
                train_sample_count = 0
                val_sample_count = 0
                selected_best_model = None
                step_selector_metrics = None
            else:
                train_page_id = fine_tune_page_ids[step_index - 1]
                step_label = f"page_{train_page_id}"
                if dataset_config.training_policy == "cumulative":
                    training_page_ids = fine_tune_page_ids[:step_index]
                elif dataset_config.training_policy == "page_only":
                    training_page_ids = [train_page_id]
                else:
                    raise ValueError(f"Unsupported training policy: {dataset_config.training_policy}")

                training_overrides = dict(dataset_config.training_overrides)
                training_overrides["width_policy"] = dataset_config.width_policy
                training_overrides["lr_scheduler"] = dataset_config.lr_scheduler

                finetune_result = fine_tune_checkpoint_on_pages(
                    [prepared_pages[page_id] for page_id in training_page_ids],
                    current_checkpoint,
                    run_dir / "models" / f"step_{step_index:02d}_{train_page_id}",
                    step_index=step_index,
                    validation_ratio=dataset_config.validation_ratio,
                    split_seed=dataset_config.split_seed,
                    oversampling_policy=dataset_config.oversampling_policy,
                    augmentation_policy=dataset_config.augmentation_policy,
                    **training_overrides,
                )
                fine_tune_metadata.append(asdict(finetune_result))
                selector_metrics.append(
                    {
                        "step_index": step_index,
                        "step_label": step_label,
                        "selector_metrics": finetune_result.selector_metrics,
                    }
                )
                current_checkpoint = Path(finetune_result.output_checkpoint)
                train_seconds = finetune_result.train_seconds
                train_sample_count = finetune_result.train_sample_count
                val_sample_count = finetune_result.val_sample_count
                selected_best_model = finetune_result.selected_best_model
                step_selector_metrics = finetune_result.selector_metrics

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
                "training_page_ids": training_page_ids,
                "checkpoint_path": str(Path(current_checkpoint).resolve()),
                "train_seconds": train_seconds,
                "train_sample_count": train_sample_count,
                "val_sample_count": val_sample_count,
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
                raise AssertionError(
                    f"Step {step_index} ({step_label}) regressed aggregate page CER by "
                    f"{delta_page_cer:.6f}, exceeding {dataset_config.regression_guard_abs:.6f}."
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

    _write_policy_summary(summary_path, status, dataset_config, steps, curve_metrics, failure_message)
    _write_json(
        metrics_path,
        {
            "status": status,
            "failure_message": failure_message,
            "run_dir": str(run_dir.resolve()),
            "config": config_payload,
            "policy": _policy_descriptor(dataset_config),
            "steps": steps,
        },
    )
    _write_json(curve_metrics_path, curve_metrics)
    _write_json(fine_tune_metadata_path, {"runs": fine_tune_metadata})
    _write_json(selector_metrics_path, {"steps": selector_metrics})
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
        _plot_metric_curve(plot_path, _policy_slug(dataset_config), steps)

    return {
        "status": status,
        "failure_message": failure_message,
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
        "policy_slug": _policy_slug(dataset_config),
    }


def _choose_better_run(current_best, candidate_result):
    if candidate_result["status"] != "passed":
        return current_best, False, "candidate failed"

    candidate_curve = candidate_result["curve_metrics"]["curve_metric_value"]
    current_curve = current_best["curve_metrics"]["curve_metric_value"]
    candidate_guard = candidate_result["curve_metrics"]["regression_guard_passed"]
    current_guard = current_best["curve_metrics"]["regression_guard_passed"]

    if candidate_guard and (not current_guard or candidate_curve < current_curve):
        return candidate_result, True, "candidate improved curve metric and passed regression guard"

    if not candidate_guard:
        return current_best, False, "candidate failed the regression guard"
    return current_best, False, "baseline remained better on the primary curve metric"


def _write_study_summary(path, dataset_name, policy_results, decisions, winning_result):
    lines = [
        f"# Recognition Fine-Tuning Study: {dataset_name}",
        "",
        f"Winning policy: `{winning_result['policy_slug']}`",
        "",
        "## Decisions",
        "",
    ]
    for decision in decisions:
        lines.append(
            f"- {decision['axis']}: baseline `{decision['baseline_policy']}` vs challenger "
            f"`{decision['challenger_policy']}` ({decision['reason']})."
        )

    lines.extend(["", "## Policy Runs", ""])
    for policy_result in policy_results:
        curve_metrics = policy_result["curve_metrics"]
        lines.append(
            f"- `{policy_result['policy_slug']}`: status={policy_result['status']}, "
            f"curve_metric={curve_metrics['curve_metric_value']}, "
            f"guard={curve_metrics['regression_guard_passed']}, "
            f"final_page_cer={curve_metrics['final_page_cer']}"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _copy_latest_artifacts(run_dir, summary_path, metrics_path, plot_path):
    shutil.copy2(summary_path, LOGS_ROOT / "recognition_finetune_results_latest.md")
    shutil.copy2(metrics_path, LOGS_ROOT / "recognition_finetune_results_latest.json")
    if plot_path.exists():
        shutil.copy2(plot_path, LOGS_ROOT / "recognition_finetune_results_latest.png")
    (LOGS_ROOT / "recognition_finetune_results_latest.txt").write_text(
        f"Latest run: {run_dir.resolve()}\n",
        encoding="utf-8",
    )


def run_recognition_finetuning_experiment(dataset_name="eval_dataset"):
    dataset_config = get_dataset_config(dataset_name)
    ordered_page_ids = dataset_config.ordered_page_ids()
    evaluation_page_ids = dataset_config.evaluation_page_ids()

    if not PRETRAINED_OCR_CHECKPOINT.exists():
        raise FileNotFoundError(f"Missing OCR checkpoint: {PRETRAINED_OCR_CHECKPOINT}")

    LOGS_ROOT.mkdir(parents=True, exist_ok=True)
    run_dir = LOGS_ROOT / f"{_timestamp_slug()}_ocrft_{dataset_config.name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    prepared_pages = prepare_page_datasets(
        dataset_config.images_dir,
        dataset_config.pagexml_dir,
        ordered_page_ids,
        run_dir / "prepared_pages",
    )
    evaluation_pages = {page_id: prepared_pages[page_id] for page_id in evaluation_page_ids}
    gt_subset_dir = _copy_eval_ground_truth_subset(dataset_config.pagexml_dir, evaluation_page_ids, run_dir / "gt_eval_subset")

    policy_results = []
    decisions = []

    baseline_result = _run_single_policy_run(
        run_dir / "policies" / _policy_slug(dataset_config),
        dataset_config,
        prepared_pages,
        evaluation_pages,
        gt_subset_dir,
    )
    policy_results.append(baseline_result)
    if baseline_result["status"] != "passed":
        raise AssertionError(baseline_result["failure_message"])

    best_config = dataset_config
    best_result = baseline_result

    width_candidate = best_config.with_updates(width_policy="batch_max_pad")
    width_result = _run_single_policy_run(
        run_dir / "policies" / _policy_slug(width_candidate),
        width_candidate,
        prepared_pages,
        evaluation_pages,
        gt_subset_dir,
    )
    policy_results.append(width_result)
    previous_best_slug = best_result["policy_slug"]
    best_result, promoted, reason = _choose_better_run(best_result, width_result)
    if promoted:
        best_config = width_candidate
    decisions.append(
        {
            "axis": "width_policy",
            "baseline_policy": previous_best_slug,
            "kept_policy": best_result["policy_slug"],
            "challenger_policy": width_result["policy_slug"],
            "reason": reason,
        }
    )

    oversampling_candidate = best_config.with_updates(oversampling_policy="cer_weighted")
    oversampling_result = _run_single_policy_run(
        run_dir / "policies" / _policy_slug(oversampling_candidate),
        oversampling_candidate,
        prepared_pages,
        evaluation_pages,
        gt_subset_dir,
    )
    policy_results.append(oversampling_result)
    previous_best_slug = best_result["policy_slug"]
    best_result, promoted, reason = _choose_better_run(best_result, oversampling_result)
    if promoted:
        best_config = oversampling_candidate
    decisions.append(
        {
            "axis": "oversampling_policy",
            "baseline_policy": previous_best_slug,
            "kept_policy": best_result["policy_slug"],
            "challenger_policy": oversampling_result["policy_slug"],
            "reason": reason,
        }
    )

    for augmentation_policy in ("background_only", "background_plus_rotation"):
        augmentation_candidate = best_config.with_updates(augmentation_policy=augmentation_policy)
        augmentation_result = _run_single_policy_run(
            run_dir / "policies" / _policy_slug(augmentation_candidate),
            augmentation_candidate,
            prepared_pages,
            evaluation_pages,
            gt_subset_dir,
        )
        policy_results.append(augmentation_result)
        previous_best_slug = best_result["policy_slug"]
        best_result, promoted, reason = _choose_better_run(best_result, augmentation_result)
        if promoted:
            best_config = augmentation_candidate
        decisions.append(
            {
                "axis": f"augmentation_policy:{augmentation_policy}",
                "baseline_policy": previous_best_slug,
                "kept_policy": best_result["policy_slug"],
                "challenger_policy": augmentation_result["policy_slug"],
                "reason": reason,
            }
        )

    scheduler_candidates = [
        _with_training_lr(best_config.with_updates(lr_scheduler="none"), 0.05),
        _with_training_lr(best_config.with_updates(lr_scheduler="none"), 0.1),
        _with_training_lr(best_config.with_updates(lr_scheduler="step"), 0.1),
        _with_training_lr(best_config.with_updates(lr_scheduler="step"), 0.2),
        _with_training_lr(best_config.with_updates(lr_scheduler="cosine"), 0.1),
        _with_training_lr(best_config.with_updates(lr_scheduler="cosine"), 0.2),
    ]

    for scheduler_candidate in scheduler_candidates:
        scheduler_result = _run_single_policy_run(
            run_dir / "policies" / _policy_slug(scheduler_candidate),
            scheduler_candidate,
            prepared_pages,
            evaluation_pages,
            gt_subset_dir,
        )
        policy_results.append(scheduler_result)
        previous_best_slug = best_result["policy_slug"]
        best_result, promoted, reason = _choose_better_run(best_result, scheduler_result)
        if promoted:
            best_config = scheduler_candidate
        decisions.append(
            {
                "axis": f"lr_scheduler:{scheduler_candidate.lr_scheduler}:{scheduler_candidate.training_overrides['lr']}",
                "baseline_policy": previous_best_slug,
                "kept_policy": best_result["policy_slug"],
                "challenger_policy": scheduler_result["policy_slug"],
                "reason": reason,
            }
        )

    study_summary_path = run_dir / "summary.md"
    study_metrics_path = run_dir / "metrics.json"
    _write_study_summary(study_summary_path, dataset_name, policy_results, decisions, best_result)
    _write_json(
        study_metrics_path,
        {
            "dataset_name": dataset_name,
            "run_dir": str(run_dir.resolve()),
            "winning_policy": best_result["policy_slug"],
            "winning_curve_metrics": best_result["curve_metrics"],
            "policy_runs": [
                {
                    "policy_slug": policy_result["policy_slug"],
                    "status": policy_result["status"],
                    "curve_metrics": policy_result["curve_metrics"],
                    "metrics_path": str(policy_result["metrics_path"].resolve()),
                }
                for policy_result in policy_results
            ],
            "decisions": decisions,
        },
    )
    _copy_latest_artifacts(run_dir, study_summary_path, study_metrics_path, best_result["plot_path"])

    return {
        "run_dir": run_dir,
        "summary_path": study_summary_path,
        "metrics_path": study_metrics_path,
        "policy_runs": policy_results,
        "winning_policy": best_result["policy_slug"],
        "curve_metrics": best_result["curve_metrics"],
        "steps": best_result["steps"],
    }
