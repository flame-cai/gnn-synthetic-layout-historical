from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

try:
    from .active_learning_recipe import normalize_sibling_checkpoint_strategy
    from .lmdb_tools import create_lmdb_dataset
    from .ocr_defaults import (
        SANSKRIT_OCR_CHARACTER_SET,
        get_device,
        load_inference_model,
        run_line_image_inference_from_loaded_model,
    )
    from .pagexml_line_dataset import (
        PreparedPageDataset,
        load_prepared_page_dataset,
        prepare_page_line_dataset,
        write_prediction_pagexml,
    )
    from .train import train
except ImportError:  # pragma: no cover - script execution fallback
    from active_learning_recipe import normalize_sibling_checkpoint_strategy
    from lmdb_tools import create_lmdb_dataset
    from ocr_defaults import (
        SANSKRIT_OCR_CHARACTER_SET,
        get_device,
        load_inference_model,
        run_line_image_inference_from_loaded_model,
    )
    from pagexml_line_dataset import (
        PreparedPageDataset,
        load_prepared_page_dataset,
        prepare_page_line_dataset,
        write_prediction_pagexml,
    )
    from train import train


BACKGROUND_JITTER_DELTAS = (-8, -4, 0, 4, 8)
LENGTH_BUCKETS = {
    "short": lambda length: length <= 10,
    "medium": lambda length: 11 <= length <= 30,
    "long": lambda length: length > 30,
}
SUPPORTED_OPTIMIZERS = {"adadelta", "adam"}


def _normalize_optimizer_name(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in SUPPORTED_OPTIMIZERS:
        raise ValueError(f"Unsupported optimizer: {value}")
    return normalized


@dataclass
class FineTuneRunResult:
    step_index: int
    training_page_id: str
    training_page_ids: list[str]
    base_checkpoint: str
    output_checkpoint: str
    experiment_dir: str
    dataset_root: str
    lmdb_root: str
    train_seconds: float
    selected_best_model: str
    sibling_checkpoint_strategy: str
    selector_metrics_path: str
    selector_metrics: dict
    train_sample_count: int
    val_sample_count: int
    validation_ratio: float
    split_seed: int
    width_policy: str
    oversampling_policy: str
    augmentation_policy: str
    lr_scheduler: str
    optimizer_name: str
    background_plus_rotation_variant_count: int
    shuffle_train_each_epoch: bool
    logical_train_sample_count: int
    logical_val_sample_count: int
    train_materialized_count: int
    val_materialized_count: int
    train_variant_labels: list[str]
    val_variant_labels: list[str]
    dataset_manifest_path: str
    dataset_manifest: dict
    train_options: dict
    training_summary: dict


@dataclass
class RecognitionInferenceResult:
    checkpoint_path: str
    prediction_folder: str | None
    per_page_seconds: dict[str, float]
    total_seconds: float
    aggregate_metrics: dict = field(default_factory=dict)
    per_page_metrics: list[dict] = field(default_factory=list)
    per_line_predictions: list[dict] = field(default_factory=list)


def prepare_page_datasets(images_dir: str | Path, pagexml_dir: str | Path, page_ids, output_root: str | Path):
    images_dir = Path(images_dir)
    pagexml_dir = Path(pagexml_dir)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    prepared = {}
    for page_id in page_ids:
        xml_path = pagexml_dir / f"{page_id}.xml"
        image_path = images_dir / f"{page_id}.jpg"
        if not image_path.exists():
            matches = list(images_dir.glob(f"{page_id}.*"))
            if not matches:
                raise FileNotFoundError(f"Image not found for page {page_id} in {images_dir}")
            image_path = matches[0]

        page_output_root = output_root / page_id
        prepared[page_id] = prepare_page_line_dataset(xml_path, image_path, page_output_root)

    return prepared


def load_page_datasets_from_output(output_root: str | Path, page_ids):
    output_root = Path(output_root)
    prepared = {}
    for page_id in page_ids:
        prepared[page_id] = load_prepared_page_dataset(output_root / page_id / "manifest.json")
    return prepared


def _normalize_prepared_pages(prepared_pages) -> list[PreparedPageDataset]:
    if isinstance(prepared_pages, dict):
        return list(prepared_pages.values())
    return list(prepared_pages)


def _build_finetune_options(base_checkpoint, lmdb_root, experiment_dir, **overrides):
    options = {
        "exp_name": experiment_dir.name,
        "experiment_dir": str(experiment_dir.resolve()),
        "train_data": str(lmdb_root.resolve()),
        "valid_data": str((lmdb_root / "val").resolve()),
        "manualSeed": 1111,
        "workers": 0,
        "batch_size": 1,
        "num_iter": 100,
        "valInterval": 5,
        "saved_model": str(Path(base_checkpoint).resolve()),
        "FT": True,
        "adam": False,
        "lr": 0.8,
        "beta1": 0.9,
        "rho": 0.95,
        "eps": 1e-8,
        "grad_clip": 5,
        "baiduCTC": False,
        "select_data": "train",
        "batch_ratio": "1",
        "total_data_usage_ratio": "1.0",
        "batch_max_length": 250,
        "imgH": 50,
        "imgW": 2000,
        "rgb": False,
        "character": SANSKRIT_OCR_CHARACTER_SET,
        "sensitive": False,
        "PAD": True,
        "width_policy": "global_2000_pad",
        "data_filtering_off": True,
        "lr_scheduler": "none",
        "optimizer_name": "adadelta",
        "background_plus_rotation_variant_count": 1,
        "shuffle_train_each_epoch": True,
        "Transformation": None,
        "FeatureExtraction": "ResNet",
        "SequenceModeling": "BiLSTM",
        "Prediction": "CTC",
        "num_fiducial": 20,
        "input_channel": 1,
        "output_channel": 512,
        "hidden_size": 512,
    }
    options.update(overrides)
    return SimpleNamespace(**options)


def _edit_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return _edit_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def _normalized_line_cer(predicted_text: str, ground_truth: str) -> float:
    return _edit_distance(predicted_text, ground_truth) / max(len(ground_truth), 1)


def compute_replication_factor(line_cer: float) -> int:
    bounded = min(max(float(line_cer), 0.0), 1.0)
    return min(4, 1 + math.floor(3 * bounded))


def _length_bucket(gt_length: int) -> str:
    for bucket_name, predicate in LENGTH_BUCKETS.items():
        if predicate(gt_length):
            return bucket_name
    return "long"


def _stable_seed(*parts) -> int:
    joined = "||".join(str(part) for part in parts)
    digest = hashlib.sha1(joined.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _estimate_background_fill(image: np.ndarray) -> int:
    if image.ndim != 2:
        raise ValueError("Expected a single-channel line image.")

    top = image[0, :]
    bottom = image[-1, :]
    left = image[:, 0]
    right = image[:, -1]
    border = np.concatenate([top, bottom, left, right])
    return int(np.clip(np.median(border), 0, 255))


def _foreground_mask(image: np.ndarray, background_fill: int) -> np.ndarray:
    return np.abs(image.astype(np.int16) - int(background_fill)) > 2


def _foreground_bbox_corners(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x_min = float(xs.min())
    x_max = float(xs.max())
    y_min = float(ys.min())
    y_max = float(ys.max())
    return np.array(
        [
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max],
        ],
        dtype=np.float32,
    )


def _rotated_bbox_fits(mask: np.ndarray, angle_degrees: float) -> bool:
    corners = _foreground_bbox_corners(mask)
    if corners is None:
        return True

    height, width = mask.shape[:2]
    center = ((width - 1) / 2.0, (height - 1) / 2.0)
    rotation = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    homogeneous = np.concatenate([corners, np.ones((4, 1), dtype=np.float32)], axis=1)
    rotated = homogeneous @ rotation.T

    within_x = rotated[:, 0].min() >= 0.0 and rotated[:, 0].max() <= (width - 1)
    within_y = rotated[:, 1].min() >= 0.0 and rotated[:, 1].max() <= (height - 1)
    return bool(within_x and within_y)


def _choose_safe_rotation_angle(mask: np.ndarray, requested_angle: float) -> float:
    if abs(requested_angle) < 1e-6 or not mask.any():
        return 0.0

    for scale in (1.0, 0.75, 0.5, 0.25):
        candidate = requested_angle * scale
        if _rotated_bbox_fits(mask, candidate):
            return float(candidate)

    return 0.0


def apply_ocr_training_augmentation(
    image: np.ndarray,
    augmentation_policy: str,
    seed: int,
) -> tuple[np.ndarray, dict]:
    if augmentation_policy not in {"none", "background_only", "background_plus_rotation"}:
        raise ValueError(f"Unsupported augmentation policy: {augmentation_policy}")

    if augmentation_policy == "none":
        return image.astype(np.uint8).copy(), {
            "background_fill": _estimate_background_fill(image),
            "background_delta": 0,
            "requested_rotation_degrees": 0.0,
            "applied_rotation_degrees": 0.0,
        }

    rng = random.Random(seed)
    augmented = image.astype(np.uint8).copy()
    background_fill = _estimate_background_fill(augmented)
    background_delta = int(rng.choice(BACKGROUND_JITTER_DELTAS))
    adjusted_fill = int(np.clip(background_fill + background_delta, 0, 255))

    foreground_mask = _foreground_mask(augmented, background_fill)
    augmented[~foreground_mask] = adjusted_fill

    requested_rotation = 0.0
    applied_rotation = 0.0
    if augmentation_policy == "background_plus_rotation" and foreground_mask.any():
        requested_rotation = float(rng.uniform(-5.0, 5.0))
        applied_rotation = _choose_safe_rotation_angle(foreground_mask, requested_rotation)
        if abs(applied_rotation) >= 1e-6:
            height, width = augmented.shape[:2]
            center = ((width - 1) / 2.0, (height - 1) / 2.0)
            rotation = cv2.getRotationMatrix2D(center, applied_rotation, 1.0)
            rotated = cv2.warpAffine(
                augmented,
                rotation,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=adjusted_fill,
            )
            if _foreground_mask(rotated, adjusted_fill).any():
                augmented = rotated
            else:
                applied_rotation = 0.0

    return augmented.astype(np.uint8), {
        "background_fill": background_fill,
        "background_delta": background_delta,
        "requested_rotation_degrees": requested_rotation,
        "applied_rotation_degrees": applied_rotation,
    }


def _collect_corpus_samples(prepared_pages: list[PreparedPageDataset]):
    samples = []
    for prepared_page in prepared_pages:
        dataset_root = Path(prepared_page.finetune_dataset_dir)
        for record in prepared_page.records:
            if not record.flat_image_rel_path:
                continue
            source_path = dataset_root / record.flat_image_rel_path
            if not source_path.exists():
                continue
            samples.append(
                {
                    "page_id": prepared_page.page_id,
                    "line_id": record.line_id,
                    "line_custom": record.line_custom,
                    "source_path": source_path,
                    "label": record.text,
                    "target_name": f"{prepared_page.page_id}__{Path(record.flat_image_rel_path).name}",
                }
            )
    if not samples:
        raise ValueError("No prepared line samples were found for OCR fine-tuning.")
    return samples


def _select_incremental_training_samples(
    prepared_pages: list[PreparedPageDataset],
    history_source_pages: list[PreparedPageDataset] | None = None,
    history_sample_line_count: int = 0,
    split_seed: int = 42,
):
    current_pages = _normalize_prepared_pages(prepared_pages)
    current_samples = [
        {**sample, "sample_origin": "current_page"}
        for sample in _collect_corpus_samples(current_pages)
    ]

    history_pages = _normalize_prepared_pages(history_source_pages or [])
    history_source_samples = []
    if history_pages:
        history_source_samples = _collect_corpus_samples(history_pages)

    requested_history_count = max(0, int(history_sample_line_count))
    history_sample_seed = None
    sampled_history_samples = []
    if requested_history_count > 0 and history_source_samples:
        history_sample_seed = _stable_seed(
            split_seed,
            "history_sample",
            current_pages[-1].page_id,
            requested_history_count,
            "|".join(page.page_id for page in history_pages),
        )
        sample_rng = random.Random(history_sample_seed)
        sampled_history_samples = sample_rng.sample(
            history_source_samples,
            min(requested_history_count, len(history_source_samples)),
        )

    selected_history_samples = [
        {**sample, "sample_origin": "history_sample"}
        for sample in sampled_history_samples
    ]
    selected_samples = current_samples + selected_history_samples
    if not selected_samples:
        raise ValueError("No prepared line samples were found for OCR fine-tuning.")

    selected_page_ids = list(dict.fromkeys(sample["page_id"] for sample in selected_samples))
    sampled_history_page_ids = list(dict.fromkeys(sample["page_id"] for sample in selected_history_samples))

    return selected_samples, {
        "page_ids": selected_page_ids,
        "current_page_ids": [page.page_id for page in current_pages],
        "current_page_line_count": len(current_samples),
        "history_source_page_ids": [page.page_id for page in history_pages],
        "history_source_line_count": len(history_source_samples),
        "history_sample_requested_count": requested_history_count,
        "history_sample_line_count": len(selected_history_samples),
        "history_sample_seed": history_sample_seed,
        "history_sample_page_ids": sampled_history_page_ids,
        "history_sample_line_refs": [
            {
                "page_id": sample["page_id"],
                "line_id": sample["line_id"],
                "line_custom": sample["line_custom"],
                "target_name": sample["target_name"],
            }
            for sample in selected_history_samples
        ],
    }


def _split_corpus_samples(samples, validation_ratio, split_seed):
    shuffled = list(samples)
    random.Random(split_seed).shuffle(shuffled)

    if validation_ratio <= 0:
        return shuffled, shuffled

    if len(shuffled) == 1:
        return shuffled, shuffled

    val_count = max(1, int(round(len(shuffled) * validation_ratio)))
    val_count = min(val_count, len(shuffled) - 1)
    val_samples = shuffled[:val_count]
    train_samples = shuffled[val_count:]
    return train_samples, val_samples


def _load_grayscale_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Failed to load line image: {image_path}")
    return image


def _build_materialized_name(sample, replication_index: int, variant_label: str) -> str:
    original = Path(sample["target_name"])
    suffix = original.suffix or ".png"
    stem = original.stem
    name_parts = []
    if replication_index > 0:
        name_parts.append(f"r{replication_index + 1}")
    if variant_label != "base":
        name_parts.append(variant_label)

    if not name_parts:
        return original.name
    return f"{stem}__{'__'.join(name_parts)}{suffix}"


def _materialize_split_dataset(
    dataset_root: Path,
    split_name: str,
    samples,
    replication_lookup: dict[str, int] | None = None,
    augmentation_policy: str = "none",
    augmentation_seed: int = 0,
    apply_augmentation: bool = False,
    background_plus_rotation_variant_count: int = 1,
):
    split_dir = dataset_root / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    gt_lines = []
    materialized_rows = []
    variant_labels = []
    variant_counts = {}

    for sample in samples:
        replications = int((replication_lookup or {}).get(sample["target_name"], 1))
        replications = max(1, replications)

        for replication_index in range(replications):
            variant_specs = [("base", None)]
            if apply_augmentation and augmentation_policy == "background_only":
                variant_specs.append(("bg", "background_only"))
            elif apply_augmentation and augmentation_policy == "background_plus_rotation":
                extra_variant_count = max(0, int(background_plus_rotation_variant_count))
                variant_specs.extend(
                    (f"bgrot{variant_index:02d}", "background_plus_rotation")
                    for variant_index in range(1, extra_variant_count + 1)
                )

            for variant_label, variant_policy in variant_specs:
                target_name = _build_materialized_name(sample, replication_index, variant_label)
                target_rel_path = Path(split_name) / target_name
                target_abs_path = dataset_root / target_rel_path
                target_abs_path.parent.mkdir(parents=True, exist_ok=True)
                if variant_label not in variant_labels:
                    variant_labels.append(variant_label)
                variant_counts[variant_label] = variant_counts.get(variant_label, 0) + 1

                augmentation_metadata = None
                if variant_policy is None:
                    shutil.copy2(sample["source_path"], target_abs_path)
                else:
                    image = _load_grayscale_image(sample["source_path"])
                    variant_seed = _stable_seed(
                        augmentation_seed,
                        split_name,
                        sample["target_name"],
                        replication_index,
                        variant_label,
                    )
                    augmented_image, augmentation_metadata = apply_ocr_training_augmentation(
                        image,
                        variant_policy,
                        seed=variant_seed,
                    )
                    cv2.imwrite(str(target_abs_path), augmented_image)

                gt_lines.append(f"{target_rel_path.as_posix()}\t{sample['label']}")
                materialized_rows.append(
                    {
                        "page_id": sample["page_id"],
                        "line_id": sample["line_id"],
                        "line_custom": sample["line_custom"],
                        "sample_origin": sample.get("sample_origin", "current_page"),
                        "label": sample["label"],
                        "target_rel_path": target_rel_path.as_posix(),
                        "replication_index": replication_index,
                        "variant_label": variant_label,
                        "augmentation_metadata": augmentation_metadata,
                    }
                )

    gt_path = dataset_root / f"gt_{split_name}.txt"
    gt_path.write_text("\n".join(gt_lines) + ("\n" if gt_lines else ""), encoding="utf-8")
    return {
        "gt_path": gt_path,
        "rows": materialized_rows,
        "count": len(materialized_rows),
        "variant_labels": variant_labels,
        "variant_counts": variant_counts,
    }


def run_checkpoint_on_prepared_pages(
    checkpoint_path: str | Path,
    prepared_pages,
    output_root: str | Path | None = None,
    **inference_overrides,
):
    ordered_pages = _normalize_prepared_pages(prepared_pages)
    output_path = Path(output_root) if output_root is not None else None
    if output_path is not None:
        if output_path.exists():
            shutil.rmtree(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

    model, converter, opt, device = load_inference_model(checkpoint_path, device=get_device(), **inference_overrides)
    per_page_seconds = {}
    total_start = time.perf_counter()
    per_line_predictions = []
    per_page_metrics = []

    sum_page_distance = 0
    sum_page_length = 0
    sum_line_distance = 0
    sum_line_gt_length = 0

    for prepared_page in ordered_pages:
        page_start = time.perf_counter()
        test_root = Path(prepared_page.finetune_dataset_dir) / "test"
        predictions = run_line_image_inference_from_loaded_model(test_root, model, converter, opt, device)
        prediction_lookup = {Path(prediction["image_path"]).name: prediction for prediction in predictions}

        predictions_by_line_custom = {}
        page_gt_segments = []
        page_pred_segments = []
        page_line_distance = 0
        page_line_gt_length = 0

        for record in prepared_page.records:
            if not record.flat_image_rel_path:
                continue

            line_name = Path(record.flat_image_rel_path).name
            prediction = prediction_lookup.get(line_name, {})
            predicted_text = prediction.get("predicted_label", "")
            confidence_score = float(prediction.get("confidence_score", 0.0) or 0.0)
            resized_width = int(prediction.get("resized_width", 0) or 0)
            pad_fraction = float(prediction.get("pad_fraction", 0.0) or 0.0)
            gt_length = len(record.text)
            line_distance = _edit_distance(predicted_text, record.text)
            line_cer = _normalized_line_cer(predicted_text, record.text)
            length_bucket = _length_bucket(gt_length)

            predictions_by_line_custom[record.line_custom] = predicted_text
            page_gt_segments.append(record.text)
            page_pred_segments.append(predicted_text)
            page_line_distance += line_distance
            page_line_gt_length += gt_length

            per_line_predictions.append(
                {
                    "page_id": record.page_id,
                    "line_id": record.line_id,
                    "line_custom": record.line_custom,
                    "gt_text": record.text,
                    "gt_length": gt_length,
                    "predicted_text": predicted_text,
                    "edit_distance": line_distance,
                    "line_cer": line_cer,
                    "confidence": confidence_score,
                    "resized_width": resized_width,
                    "pad_fraction": pad_fraction,
                    "length_bucket": length_bucket,
                }
            )

        page_gt_blob = "\n".join(page_gt_segments)
        page_pred_blob = "\n".join(page_pred_segments)
        page_distance = _edit_distance(page_pred_blob, page_gt_blob)
        page_length = len(page_gt_blob)
        sum_page_distance += page_distance
        sum_page_length += page_length
        sum_line_distance += page_line_distance
        sum_line_gt_length += page_line_gt_length

        per_page_metrics.append(
            {
                "page_id": prepared_page.page_id,
                "page_cer": page_distance / max(page_length, 1),
                "line_cer": page_line_distance / max(page_line_gt_length, 1),
                "page_distance": page_distance,
                "page_gt_length": page_length,
            }
        )

        if output_path is not None:
            write_prediction_pagexml(
                prepared_page.source_xml_path,
                predictions_by_line_custom,
                output_path / f"{prepared_page.page_id}.xml",
            )

        per_page_seconds[prepared_page.page_id] = time.perf_counter() - page_start

    return RecognitionInferenceResult(
        checkpoint_path=str(Path(checkpoint_path).resolve()),
        prediction_folder=str(output_path.resolve()) if output_path is not None else None,
        per_page_seconds=per_page_seconds,
        total_seconds=time.perf_counter() - total_start,
        aggregate_metrics={
            "page_cer": sum_page_distance / max(sum_page_length, 1),
            "line_cer": sum_line_distance / max(sum_line_gt_length, 1),
        },
        per_page_metrics=per_page_metrics,
        per_line_predictions=per_line_predictions,
    )


def score_page_difficulty(
    prepared_page: PreparedPageDataset,
    checkpoint_path: str | Path,
    **inference_overrides,
) -> dict[str, dict]:
    inference_result = run_checkpoint_on_prepared_pages(
        checkpoint_path,
        [prepared_page],
        output_root=None,
        **inference_overrides,
    )
    difficulty_by_line = {}
    for row in inference_result.per_line_predictions:
        difficulty_by_line[row["line_custom"]] = {
            "line_id": row["line_id"],
            "line_cer": row["line_cer"],
            "replication": compute_replication_factor(row["line_cer"]),
        }
    return difficulty_by_line


def _list_sibling_checkpoint_candidates(experiment_dir: str | Path) -> list[tuple[str, Path]]:
    experiment_dir = Path(experiment_dir)
    candidate_files = []
    for filename in ("best_accuracy.pth", "best_norm_ED.pth"):
        checkpoint_path = experiment_dir / filename
        if checkpoint_path.exists():
            candidate_files.append((filename, checkpoint_path))
    if not candidate_files:
        raise FileNotFoundError(f"No fine-tuned checkpoint was written to {experiment_dir}")
    return candidate_files


def _write_selector_metrics(selector_output_path: str | Path | None, payload: dict) -> None:
    if selector_output_path is None:
        return
    selector_output_path = Path(selector_output_path)
    selector_output_path.parent.mkdir(parents=True, exist_ok=True)
    selector_output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def select_best_sibling_checkpoint(
    experiment_dir: str | Path,
    prepared_pages,
    selector_output_path: str | Path | None = None,
    **inference_overrides,
):
    experiment_dir = Path(experiment_dir)
    ordered_pages = _normalize_prepared_pages(prepared_pages)
    candidate_files = _list_sibling_checkpoint_candidates(experiment_dir)

    payload = {
        "selection_strategy": "page_cer_selector",
        "selector_metric": "page_cer",
        "candidates": {},
    }

    for filename, checkpoint_path in candidate_files:
        inference_result = run_checkpoint_on_prepared_pages(
            checkpoint_path,
            ordered_pages,
            output_root=None,
            **inference_overrides,
        )
        payload["candidates"][filename] = {
            "checkpoint_path": str(checkpoint_path.resolve()),
            "page_cer": inference_result.aggregate_metrics["page_cer"],
            "line_cer": inference_result.aggregate_metrics["line_cer"],
            "per_page": inference_result.per_page_metrics,
        }

    chosen_filename = min(
        payload["candidates"],
        key=lambda filename: (
            payload["candidates"][filename]["page_cer"],
            payload["candidates"][filename]["line_cer"],
            filename,
        ),
    )
    chosen_checkpoint = experiment_dir / chosen_filename

    if len(candidate_files) == 1:
        reason = f"Selected {chosen_filename} because it was the only sibling checkpoint present."
    else:
        chosen_metrics = payload["candidates"][chosen_filename]
        sibling_name = next(name for name in payload["candidates"] if name != chosen_filename)
        sibling_metrics = payload["candidates"][sibling_name]
        reason = (
            f"Selected {chosen_filename} because page CER {chosen_metrics['page_cer']:.6f} "
            f"is lower than {sibling_name} at {sibling_metrics['page_cer']:.6f}."
        )

    payload["chosen_file"] = chosen_filename
    payload["chosen_checkpoint"] = str(chosen_checkpoint.resolve())
    payload["reason"] = reason

    _write_selector_metrics(selector_output_path, payload)
    return chosen_checkpoint, chosen_filename, payload


def choose_sibling_checkpoint(
    experiment_dir: str | Path,
    prepared_pages,
    strategy: str = "page_cer_selector",
    selector_output_path: str | Path | None = None,
    **inference_overrides,
):
    strategy = normalize_sibling_checkpoint_strategy(strategy)
    if strategy == "page_cer_selector":
        return select_best_sibling_checkpoint(
            experiment_dir,
            prepared_pages,
            selector_output_path=selector_output_path,
            **inference_overrides,
        )

    experiment_dir = Path(experiment_dir)
    candidate_files = _list_sibling_checkpoint_candidates(experiment_dir)
    candidate_lookup = {filename: checkpoint_path for filename, checkpoint_path in candidate_files}
    chosen_filename = "best_norm_ED.pth" if "best_norm_ED.pth" in candidate_lookup else candidate_files[0][0]
    chosen_checkpoint = candidate_lookup[chosen_filename]
    if chosen_filename == "best_norm_ED.pth":
        reason = (
            "Configured sibling checkpoint strategy prefers best_norm_ED.pth "
            "without running page-CER selection."
        )
    else:
        reason = (
            f"Configured sibling checkpoint strategy prefers best_norm_ED.pth, but that file was not present. "
            f"Fell back to the only available sibling checkpoint {chosen_filename}."
        )
    payload = {
        "selection_strategy": strategy,
        "selector_metric": None,
        "candidates": {
            filename: {"checkpoint_path": str(checkpoint_path.resolve())}
            for filename, checkpoint_path in candidate_files
        },
        "chosen_file": chosen_filename,
        "chosen_checkpoint": str(chosen_checkpoint.resolve()),
        "reason": reason,
    }
    _write_selector_metrics(selector_output_path, payload)
    return chosen_checkpoint, chosen_filename, payload


def prepare_incremental_finetune_dataset(
    prepared_pages: list[PreparedPageDataset],
    output_root: str | Path,
    base_checkpoint: str | Path,
    validation_ratio: float = 0.0,
    split_seed: int = 42,
    oversampling_policy: str = "none",
    augmentation_policy: str = "none",
    background_plus_rotation_variant_count: int = 1,
    shuffle_train_each_epoch: bool = True,
    optimizer_name: str = "adadelta",
    history_source_pages: list[PreparedPageDataset] | None = None,
    history_sample_line_count: int = 0,
    **inference_overrides,
):
    if oversampling_policy not in {"none", "cer_weighted"}:
        raise ValueError(f"Unsupported oversampling policy: {oversampling_policy}")
    if augmentation_policy not in {"none", "background_only", "background_plus_rotation"}:
        raise ValueError(f"Unsupported augmentation policy: {augmentation_policy}")
    optimizer_name = _normalize_optimizer_name(optimizer_name)
    background_plus_rotation_variant_count = max(0, int(background_plus_rotation_variant_count))

    output_root = Path(output_root)
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    samples, selection_metadata = _select_incremental_training_samples(
        prepared_pages,
        history_source_pages=history_source_pages,
        history_sample_line_count=history_sample_line_count,
        split_seed=split_seed,
    )
    replication_lookup = {sample["target_name"]: 1 for sample in samples}
    difficulty_scores = {}

    if oversampling_policy == "cer_weighted":
        newest_page = prepared_pages[-1]
        difficulty_scores = score_page_difficulty(newest_page, base_checkpoint, **inference_overrides)
        for sample in samples:
            if sample["page_id"] != newest_page.page_id:
                continue
            line_metrics = difficulty_scores.get(sample["line_custom"])
            if line_metrics is not None:
                replication_lookup[sample["target_name"]] = int(line_metrics["replication"])

    train_samples, val_samples = _split_corpus_samples(samples, validation_ratio, split_seed)
    train_materialized = _materialize_split_dataset(
        output_root,
        "train",
        train_samples,
        replication_lookup=replication_lookup,
        augmentation_policy=augmentation_policy,
        augmentation_seed=split_seed,
        apply_augmentation=True,
        background_plus_rotation_variant_count=background_plus_rotation_variant_count,
    )
    val_materialized = _materialize_split_dataset(
        output_root,
        "val",
        val_samples,
        replication_lookup={sample["target_name"]: 1 for sample in val_samples},
        augmentation_policy="none",
        augmentation_seed=split_seed,
        apply_augmentation=False,
        background_plus_rotation_variant_count=background_plus_rotation_variant_count,
    )

    manifest = {
        "page_ids": selection_metadata["page_ids"],
        "current_page_ids": selection_metadata["current_page_ids"],
        "current_page_line_count": selection_metadata["current_page_line_count"],
        "history_source_page_ids": selection_metadata["history_source_page_ids"],
        "history_source_line_count": selection_metadata["history_source_line_count"],
        "history_sample_requested_count": selection_metadata["history_sample_requested_count"],
        "history_sample_line_count": selection_metadata["history_sample_line_count"],
        "history_sample_seed": selection_metadata["history_sample_seed"],
        "history_sample_page_ids": selection_metadata["history_sample_page_ids"],
        "history_sample_line_refs": selection_metadata["history_sample_line_refs"],
        "num_samples": len(samples),
        "logical_train_sample_count": len(train_samples),
        "logical_val_sample_count": len(val_samples),
        "train_sample_count": len(train_samples),
        "val_sample_count": len(val_samples),
        "train_materialized_count": train_materialized["count"],
        "val_materialized_count": val_materialized["count"],
        "effective_train_materialized_count": train_materialized["count"],
        "effective_val_materialized_count": val_materialized["count"],
        "validation_ratio": validation_ratio,
        "split_seed": split_seed,
        "oversampling_policy": oversampling_policy,
        "augmentation_policy": augmentation_policy,
        "background_plus_rotation_variant_count": background_plus_rotation_variant_count,
        "shuffle_train_each_epoch": bool(shuffle_train_each_epoch),
        "optimizer_name": optimizer_name,
        "train_gt_path": str(train_materialized["gt_path"].resolve()),
        "val_gt_path": str(val_materialized["gt_path"].resolve()),
        "train_variant_labels": train_materialized["variant_labels"],
        "val_variant_labels": val_materialized["variant_labels"],
        "train_variant_counts": train_materialized["variant_counts"],
        "val_variant_counts": val_materialized["variant_counts"],
        "difficulty_scores": difficulty_scores,
        "replication_lookup": replication_lookup,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "dataset_root": output_root,
        "manifest_path": manifest_path,
        "train_gt_path": train_materialized["gt_path"],
        "val_gt_path": val_materialized["gt_path"],
        "train_sample_count": train_materialized["count"],
        "val_sample_count": val_materialized["count"],
        "logical_train_sample_count": len(train_samples),
        "logical_val_sample_count": len(val_samples),
        "train_materialized_count": train_materialized["count"],
        "val_materialized_count": val_materialized["count"],
        "train_variant_labels": train_materialized["variant_labels"],
        "val_variant_labels": val_materialized["variant_labels"],
        "manifest": manifest,
    }


def fine_tune_checkpoint_on_pages(
    prepared_pages: list[PreparedPageDataset],
    base_checkpoint: str | Path,
    output_root: str | Path,
    step_index: int,
    validation_ratio: float = 0.0,
    split_seed: int = 42,
    oversampling_policy: str = "none",
    augmentation_policy: str = "none",
    history_source_pages: list[PreparedPageDataset] | None = None,
    history_sample_line_count: int = 0,
    sibling_checkpoint_strategy: str = "page_cer_selector",
    **training_overrides,
):
    output_root = Path(output_root)
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    training_overrides = dict(training_overrides)
    width_policy = training_overrides.get("width_policy", "global_2000_pad")
    lr_scheduler = training_overrides.get("lr_scheduler", "none")
    optimizer_name = _normalize_optimizer_name(
        training_overrides.get("optimizer_name", "adam" if training_overrides.get("adam") else "adadelta")
    )
    background_plus_rotation_variant_count = max(
        0,
        int(training_overrides.get("background_plus_rotation_variant_count", 1)),
    )
    shuffle_train_each_epoch = bool(training_overrides.get("shuffle_train_each_epoch", True))
    training_overrides["optimizer_name"] = optimizer_name
    training_overrides["adam"] = optimizer_name == "adam"
    training_overrides["background_plus_rotation_variant_count"] = background_plus_rotation_variant_count
    training_overrides["shuffle_train_each_epoch"] = shuffle_train_each_epoch

    dataset_bundle = prepare_incremental_finetune_dataset(
        prepared_pages,
        output_root / "dataset",
        base_checkpoint=base_checkpoint,
        validation_ratio=validation_ratio,
        split_seed=split_seed,
        oversampling_policy=oversampling_policy,
        augmentation_policy=augmentation_policy,
        background_plus_rotation_variant_count=background_plus_rotation_variant_count,
        shuffle_train_each_epoch=shuffle_train_each_epoch,
        optimizer_name=optimizer_name,
        history_source_pages=history_source_pages,
        history_sample_line_count=history_sample_line_count,
        width_policy=width_policy,
    )
    dataset_root = Path(dataset_bundle["dataset_root"])

    lmdb_root = output_root / "lmdb"
    create_lmdb_dataset(dataset_root, dataset_bundle["train_gt_path"], lmdb_root / "train", check_valid=False)
    create_lmdb_dataset(dataset_root, dataset_bundle["val_gt_path"], lmdb_root / "val", check_valid=False)

    experiment_dir = output_root / "training_run"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    options = _build_finetune_options(base_checkpoint, lmdb_root, experiment_dir, **training_overrides)

    start_time = time.perf_counter()
    training_summary = train(options)
    train_seconds = time.perf_counter() - start_time

    selector_metrics_path = output_root / "selector_metrics.json"
    sibling_checkpoint_strategy = normalize_sibling_checkpoint_strategy(sibling_checkpoint_strategy)
    selected_checkpoint, selected_best_model, selector_metrics = choose_sibling_checkpoint(
        experiment_dir,
        prepared_pages,
        strategy=sibling_checkpoint_strategy,
        selector_output_path=selector_metrics_path,
        width_policy=width_policy,
    )

    training_page_ids = list(dataset_bundle["manifest"]["page_ids"])
    metadata = FineTuneRunResult(
        step_index=step_index,
        training_page_id=prepared_pages[-1].page_id,
        training_page_ids=training_page_ids,
        base_checkpoint=str(Path(base_checkpoint).resolve()),
        output_checkpoint=str(selected_checkpoint.resolve()),
        experiment_dir=str(experiment_dir.resolve()),
        dataset_root=str(dataset_root.resolve()),
        lmdb_root=str(lmdb_root.resolve()),
        train_seconds=train_seconds,
        selected_best_model=selected_best_model,
        sibling_checkpoint_strategy=sibling_checkpoint_strategy,
        selector_metrics_path=str(selector_metrics_path.resolve()),
        selector_metrics=selector_metrics,
        train_sample_count=dataset_bundle["train_sample_count"],
        val_sample_count=dataset_bundle["val_sample_count"],
        validation_ratio=validation_ratio,
        split_seed=split_seed,
        width_policy=width_policy,
        oversampling_policy=oversampling_policy,
        augmentation_policy=augmentation_policy,
        lr_scheduler=lr_scheduler,
        optimizer_name=optimizer_name,
        background_plus_rotation_variant_count=background_plus_rotation_variant_count,
        shuffle_train_each_epoch=shuffle_train_each_epoch,
        logical_train_sample_count=dataset_bundle["logical_train_sample_count"],
        logical_val_sample_count=dataset_bundle["logical_val_sample_count"],
        train_materialized_count=dataset_bundle["train_materialized_count"],
        val_materialized_count=dataset_bundle["val_materialized_count"],
        train_variant_labels=dataset_bundle["train_variant_labels"],
        val_variant_labels=dataset_bundle["val_variant_labels"],
        dataset_manifest_path=str(Path(dataset_bundle["manifest_path"]).resolve()),
        dataset_manifest=dataset_bundle["manifest"],
        train_options={key: value for key, value in vars(options).items()},
        training_summary=training_summary,
    )

    metadata_path = output_root / "fine_tune_metadata.json"
    metadata_path.write_text(json.dumps(asdict(metadata), indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata


def fine_tune_checkpoint_on_page(
    prepared_page: PreparedPageDataset,
    base_checkpoint: str | Path,
    output_root: str | Path,
    step_index: int,
    validation_ratio: float = 0.0,
    split_seed: int = 42,
    oversampling_policy: str = "none",
    augmentation_policy: str = "none",
    **training_overrides,
):
    return fine_tune_checkpoint_on_pages(
        [prepared_page],
        base_checkpoint,
        output_root,
        step_index=step_index,
        validation_ratio=validation_ratio,
        split_seed=split_seed,
        oversampling_policy=oversampling_policy,
        augmentation_policy=augmentation_policy,
        **training_overrides,
    )


def generate_prediction_pagexmls(
    checkpoint_path: str | Path,
    prepared_pages: dict[str, PreparedPageDataset],
    output_root: str | Path,
    **inference_overrides,
):
    return run_checkpoint_on_prepared_pages(
        checkpoint_path,
        prepared_pages,
        output_root=output_root,
        **inference_overrides,
    )
