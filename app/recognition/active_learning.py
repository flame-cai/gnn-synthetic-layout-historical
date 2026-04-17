from __future__ import annotations

import json
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

try:
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
    train_sample_count: int
    val_sample_count: int
    validation_ratio: float
    split_seed: int
    train_options: dict
    training_summary: dict


@dataclass
class RecognitionInferenceResult:
    checkpoint_path: str
    prediction_folder: str
    per_page_seconds: dict[str, float]
    total_seconds: float


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
        "data_filtering_off": True,
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
                    "line_custom": record.line_custom,
                    "source_path": source_path,
                    "label": record.text,
                    "target_name": f"{prepared_page.page_id}__{Path(record.flat_image_rel_path).name}",
                }
            )
    if not samples:
        raise ValueError("No prepared line samples were found for OCR fine-tuning.")
    return samples


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


def _materialize_split_dataset(dataset_root: Path, split_name: str, samples):
    split_dir = dataset_root / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    gt_lines = []

    for sample in samples:
        target_rel_path = Path(split_name) / sample["target_name"]
        target_abs_path = dataset_root / target_rel_path
        target_abs_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sample["source_path"], target_abs_path)
        gt_lines.append(f"{target_rel_path.as_posix()}\t{sample['label']}")

    gt_path = dataset_root / f"gt_{split_name}.txt"
    gt_path.write_text("\n".join(gt_lines) + ("\n" if gt_lines else ""), encoding="utf-8")
    return gt_path


def prepare_incremental_finetune_dataset(
    prepared_pages: list[PreparedPageDataset],
    output_root: str | Path,
    validation_ratio: float = 0.2,
    split_seed: int = 42,
):
    output_root = Path(output_root)
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    samples = _collect_corpus_samples(prepared_pages)
    train_samples, val_samples = _split_corpus_samples(samples, validation_ratio, split_seed)
    train_gt_path = _materialize_split_dataset(output_root, "train", train_samples)
    val_gt_path = _materialize_split_dataset(output_root, "val", val_samples)

    manifest = {
        "page_ids": [page.page_id for page in prepared_pages],
        "num_samples": len(samples),
        "train_sample_count": len(train_samples),
        "val_sample_count": len(val_samples),
        "validation_ratio": validation_ratio,
        "split_seed": split_seed,
        "train_gt_path": str(train_gt_path.resolve()),
        "val_gt_path": str(val_gt_path.resolve()),
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "dataset_root": output_root,
        "train_gt_path": train_gt_path,
        "val_gt_path": val_gt_path,
        "train_sample_count": len(train_samples),
        "val_sample_count": len(val_samples),
        "manifest": manifest,
    }


def fine_tune_checkpoint_on_pages(
    prepared_pages: list[PreparedPageDataset],
    base_checkpoint: str | Path,
    output_root: str | Path,
    step_index: int,
    validation_ratio: float = 0.2,
    split_seed: int = 42,
    **training_overrides,
):
    output_root = Path(output_root)
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    dataset_bundle = prepare_incremental_finetune_dataset(
        prepared_pages,
        output_root / "dataset",
        validation_ratio=validation_ratio,
        split_seed=split_seed,
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

    best_norm_ed = experiment_dir / "best_norm_ED.pth"
    best_accuracy = experiment_dir / "best_accuracy.pth"
    if best_norm_ed.exists():
        selected_checkpoint = best_norm_ed
        selected_best_model = "best_norm_ED.pth"
    elif best_accuracy.exists():
        selected_checkpoint = best_accuracy
        selected_best_model = "best_accuracy.pth"
    else:
        raise FileNotFoundError(f"No fine-tuned checkpoint was written to {experiment_dir}")

    training_page_ids = [prepared_page.page_id for prepared_page in prepared_pages]
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
        train_sample_count=dataset_bundle["train_sample_count"],
        val_sample_count=dataset_bundle["val_sample_count"],
        validation_ratio=validation_ratio,
        split_seed=split_seed,
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
    validation_ratio: float = 0.2,
    split_seed: int = 42,
    **training_overrides,
):
    return fine_tune_checkpoint_on_pages(
        [prepared_page],
        base_checkpoint,
        output_root,
        step_index=step_index,
        validation_ratio=validation_ratio,
        split_seed=split_seed,
        **training_overrides,
    )


def generate_prediction_pagexmls(
    checkpoint_path: str | Path,
    prepared_pages: dict[str, PreparedPageDataset],
    output_root: str | Path,
):
    output_root = Path(output_root)
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    model, converter, opt, device = load_inference_model(checkpoint_path, device=get_device())
    per_page_seconds = {}
    total_start = time.perf_counter()

    for page_id, prepared_page in prepared_pages.items():
        page_start = time.perf_counter()
        test_root = Path(prepared_page.finetune_dataset_dir) / "test"
        predictions = run_line_image_inference_from_loaded_model(test_root, model, converter, opt, device)

        line_lookup = {
            Path(record.flat_image_rel_path).name: record.line_custom
            for record in prepared_page.records
            if record.flat_image_rel_path
        }
        predictions_by_line_custom = {}
        for prediction in predictions:
            line_name = Path(prediction["image_path"]).name
            line_custom = line_lookup.get(line_name)
            if line_custom:
                predictions_by_line_custom[line_custom] = prediction["predicted_label"]

        write_prediction_pagexml(
            prepared_page.source_xml_path,
            predictions_by_line_custom,
            output_root / f"{page_id}.xml",
        )
        per_page_seconds[page_id] = time.perf_counter() - page_start

    return RecognitionInferenceResult(
        checkpoint_path=str(Path(checkpoint_path).resolve()),
        prediction_folder=str(output_root.resolve()),
        per_page_seconds=per_page_seconds,
        total_seconds=time.perf_counter() - total_start,
    )
