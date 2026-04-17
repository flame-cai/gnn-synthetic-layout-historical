from __future__ import annotations

import csv
import json
import os
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
from tests.recognition_finetuning_config import get_dataset_config


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


def _write_per_page_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "step_index",
        "step_label",
        "train_page_id",
        "eval_page_id",
        "page_cer",
        "line_cer_50",
        "gt_len",
        "prediction_found",
        "inference_seconds",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_summary(path, status, dataset_name, train_pages, eval_pages, steps, failure_message):
    lines = [
        f"# Recognition Fine-Tuning Evaluation: {dataset_name}",
        "",
        f"Status: **{status.upper()}**",
        "",
        f"Train pages: {', '.join(train_pages)}",
        f"Evaluation pages: {', '.join(eval_pages)}",
        "",
        "## Step Summary",
        "",
    ]
    for step in steps:
        metrics = step["metrics"]["aggregate_metrics"]
        trained_pages = ", ".join(step.get("training_page_ids") or ([] if not step["train_page_id"] else [step["train_page_id"]]))
        if not trained_pages:
            trained_pages = "-"
        lines.extend(
            [
                f"- Step {step['step_index']} ({step['step_label']}): page CER={metrics['page_cer']:.4f}, "
                f"train_pages=[{trained_pages}], "
                f"line CER@0.50={metrics['line_cer_50']:.4f}, "
                f"train_seconds={step['train_seconds']:.2f}, "
                f"inference_seconds={step['inference_seconds']:.2f}, "
                f"improved={step['improved_over_previous']}",
            ]
        )
    if failure_message:
        lines.extend(["", "## Failure", "", failure_message])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_metric_curve(path, steps):
    x_vals = [step["page_count_finetuned"] for step in steps]
    page_cer = [step["metrics"]["aggregate_metrics"]["page_cer"] for step in steps]
    line_cer = [step["metrics"]["aggregate_metrics"]["line_cer_50"] for step in steps]

    plt.figure(figsize=(8, 5))
    plt.plot(x_vals, page_cer, marker="o", linewidth=2, label="Page CER")
    plt.plot(x_vals, line_cer, marker="s", linewidth=2, label="Line CER @ IoU 0.50")
    plt.xlabel("Sequential Fine-Tuning Pages")
    plt.ylabel("CER")
    plt.title("Recognition CER vs Sequential Fine-Tuning Data")
    plt.grid(True, alpha=0.3)
    plt.xticks(x_vals)
    plt.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _copy_latest_artifacts(run_dir, summary_path, metrics_path, plot_path):
    shutil.copy2(summary_path, LOGS_ROOT / "recognition_finetune_results_latest.md")
    shutil.copy2(metrics_path, LOGS_ROOT / "recognition_finetune_results_latest.json")
    shutil.copy2(plot_path, LOGS_ROOT / "recognition_finetune_results_latest.png")
    (LOGS_ROOT / "recognition_finetune_results_latest.txt").write_text(
        f"Latest run: {run_dir.resolve()}\n",
        encoding="utf-8",
    )


def run_recognition_finetuning_experiment(dataset_name="eval_dataset"):
    dataset_config = get_dataset_config(dataset_name)
    ordered_page_ids = dataset_config.ordered_page_ids()
    fine_tune_page_ids = dataset_config.fine_tune_page_ids()
    evaluation_page_ids = dataset_config.evaluation_page_ids()

    if not PRETRAINED_OCR_CHECKPOINT.exists():
        raise FileNotFoundError(f"Missing OCR checkpoint: {PRETRAINED_OCR_CHECKPOINT}")

    LOGS_ROOT.mkdir(parents=True, exist_ok=True)
    run_dir = LOGS_ROOT / f"{_timestamp_slug()}_recognition_finetune_eval_{dataset_config.name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    prepared_pages = prepare_page_datasets(
        dataset_config.images_dir,
        dataset_config.pagexml_dir,
        ordered_page_ids,
        run_dir / "prepared_pages",
    )
    evaluation_pages = {page_id: prepared_pages[page_id] for page_id in evaluation_page_ids}
    gt_subset_dir = _copy_eval_ground_truth_subset(dataset_config.pagexml_dir, evaluation_page_ids, run_dir / "gt_eval_subset")

    config_payload = {
        "dataset": dataset_config.to_dict(),
        "pretrained_checkpoint": str(PRETRAINED_OCR_CHECKPOINT.resolve()),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "python_executable": sys.executable,
    }
    _write_json(run_dir / "config.json", config_payload)

    steps = []
    per_page_rows = []
    fine_tune_metadata = []
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
            else:
                train_page_id = fine_tune_page_ids[step_index - 1]
                step_label = f"page_{train_page_id}"
                if dataset_config.training_policy == "cumulative":
                    training_page_ids = fine_tune_page_ids[:step_index]
                elif dataset_config.training_policy == "page_only":
                    training_page_ids = [train_page_id]
                else:
                    raise ValueError(f"Unsupported training policy: {dataset_config.training_policy}")

                finetune_result = fine_tune_checkpoint_on_pages(
                    [prepared_pages[page_id] for page_id in training_page_ids],
                    current_checkpoint,
                    run_dir / "models" / f"step_{step_index:02d}_{train_page_id}",
                    step_index=step_index,
                    validation_ratio=dataset_config.validation_ratio,
                    split_seed=dataset_config.split_seed,
                    **dataset_config.training_overrides,
                )
                fine_tune_metadata.append(asdict(finetune_result))
                current_checkpoint = Path(finetune_result.output_checkpoint)
                train_seconds = finetune_result.train_seconds
                train_sample_count = finetune_result.train_sample_count
                val_sample_count = finetune_result.val_sample_count

            prediction_output = generate_prediction_pagexmls(
                current_checkpoint,
                evaluation_pages,
                run_dir / "predicted_page_xml" / f"step_{step_index:02d}_{step_label}",
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
            else:
                improved = aggregate_page_cer < previous_page_cer
                delta_page_cer = aggregate_page_cer - previous_page_cer

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

            if previous_page_cer is not None and not improved:
                raise AssertionError(
                    f"Step {step_index} ({step_label}) did not improve aggregate page CER: "
                    f"{aggregate_page_cer:.6f} >= {previous_page_cer:.6f}"
                )

            previous_page_cer = aggregate_page_cer
    except Exception as exc:
        status = "failed"
        failure_message = str(exc)

    summary_path = run_dir / "summary.md"
    metrics_path = run_dir / "metrics.json"
    per_page_csv_path = run_dir / "per_page.csv"
    fine_tune_metadata_path = run_dir / "fine_tune_metadata.json"
    plot_path = run_dir / "plots" / "page_cer_vs_finetune_pages.png"

    _write_summary(summary_path, status, dataset_config.name, fine_tune_page_ids, evaluation_page_ids, steps, failure_message)
    _write_json(
        metrics_path,
        {
            "status": status,
            "failure_message": failure_message,
            "run_dir": str(run_dir.resolve()),
            "config": config_payload,
            "steps": steps,
        },
    )
    _write_per_page_csv(per_page_csv_path, per_page_rows)
    _write_json(fine_tune_metadata_path, {"runs": fine_tune_metadata})
    if steps:
        _plot_metric_curve(plot_path, steps)
        _copy_latest_artifacts(run_dir, summary_path, metrics_path, plot_path)

    if status != "passed":
        raise AssertionError(failure_message)

    return {
        "run_dir": run_dir,
        "summary_path": summary_path,
        "metrics_path": metrics_path,
        "per_page_csv_path": per_page_csv_path,
        "fine_tune_metadata_path": fine_tune_metadata_path,
        "plot_path": plot_path,
        "steps": steps,
    }
