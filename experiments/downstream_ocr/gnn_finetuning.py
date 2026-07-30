from __future__ import annotations

import hashlib
import json
import logging
import math
import multiprocessing
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
import yaml
from torch.optim.lr_scheduler import LambdaLR
from torch_geometric.loader import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
SYNTHETIC_DATA_ROOT = SRC_ROOT / "synthetic_data_gen"
GNN_DATA_PREPARATION_ROOT = SRC_ROOT / "gnn_training" / "gnn_data_preparation"
for import_root in (SRC_ROOT, SYNTHETIC_DATA_ROOT, GNN_DATA_PREPARATION_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from gnn_training.gnn_data_preparation.config_models import DatasetCreationConfig
from gnn_training.gnn_data_preparation.main_create_dataset import process_page
from gnn_training.training.engine import evaluate, train_one_epoch
from gnn_training.training.utils import FocalLoss, get_device, set_seed
from synthetic_data_gen.augment import _augment_single_instance
from synthetic_data_gen.manuscript_generator.configs.augmentation_config import (
    AugmentationConfig,
)


LOGGER = logging.getLogger(__name__)
GNN_FINETUNING_SCHEMA_VERSION = 2
TRAINABLE_SCOPE_FULL_MODEL = "full_model"
TRAINABLE_SCOPE_HEAD_AND_FINAL_GNN_LAYER = (
    "edge_classifier_head_and_final_gnn_layer"
)


@dataclass(frozen=True)
class GNNFineTuningRecipe:
    config_path: Path
    augmentation_config_path: Path
    preprocessing_config_path: Path
    history_replay_ratio: float
    expected_model_class: str
    expected_backbone_class: str
    trainable_scope: str
    deleted_node_supervision: dict
    augmentation_config: AugmentationConfig
    preprocessing_config: DatasetCreationConfig
    training_config: dict

    @property
    def augmentations_per_page(self) -> int:
        return int(self.augmentation_config.general.num_augmentations_per_sample)

    def metadata(self) -> dict:
        return {
            "schema_version": GNN_FINETUNING_SCHEMA_VERSION,
            "config_path": str(self.config_path.resolve()),
            "config_sha256": _sha256(self.config_path),
            "augmentation_config_path": str(self.augmentation_config_path.resolve()),
            "augmentation_config_sha256": _sha256(self.augmentation_config_path),
            "preprocessing_config_path": str(self.preprocessing_config_path.resolve()),
            "preprocessing_config_sha256": _sha256(self.preprocessing_config_path),
            "augmentations_per_page": self.augmentations_per_page,
            "history_replay_ratio": self.history_replay_ratio,
            "expected_model_class": self.expected_model_class,
            "expected_backbone_class": self.expected_backbone_class,
            "trainable_scope": self.trainable_scope,
            "deleted_node_supervision": dict(self.deleted_node_supervision),
            "training_params": dict(self.training_config["training_params"]),
            "checkpoint_metric": self.training_config["checkpoint_metric"],
        }


@dataclass(frozen=True)
class GNNStepResult:
    output_checkpoint: Path
    metadata_path: Path
    selected_epoch: int
    selected_metric: float


@dataclass(frozen=True)
class GNNLadderResult:
    checkpoint_by_count: dict[int, Path]
    step_metadata_by_count: dict[int, Path]
    summary_path: Path
    recipe: GNNFineTuningRecipe


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def _load_yaml_mapping(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return payload


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_gnn_finetuning_recipe(config_path: str | Path) -> GNNFineTuningRecipe:
    path = _resolve_repo_path(config_path)
    payload = _load_yaml_mapping(path)
    schema_version = int(payload.get("schema_version", 0))
    if schema_version != GNN_FINETUNING_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported GNN fine-tuning config schema {schema_version}: {path}"
        )

    augmentation_path = _resolve_repo_path(payload["augmentation_config"])
    preprocessing_path = _resolve_repo_path(payload["preprocessing_config"])
    for referenced_path in (augmentation_path, preprocessing_path):
        if not referenced_path.is_file():
            raise FileNotFoundError(f"GNN fine-tuning config file not found: {referenced_path}")

    augmentation_config = AugmentationConfig(
        **_load_yaml_mapping(augmentation_path)
    )
    preprocessing_config = DatasetCreationConfig(
        **_load_yaml_mapping(preprocessing_path)
    )
    training_config = {
        "device": payload.get("device", "auto"),
        "random_seed": int(payload["random_seed"]),
        "checkpoint_metric": str(payload["checkpoint_metric"]),
        "training_params": dict(payload["training_params"]),
    }
    history_replay_ratio = float(payload["history_replay_ratio"])
    expected_model_class = str(payload["expected_model_class"])
    expected_backbone_class = str(payload["expected_backbone_class"])
    trainable_scope = str(payload["trainable_scope"])
    deleted_node_supervision = dict(payload["deleted_node_supervision"])

    if augmentation_config.general.num_augmentations_per_sample != 50:
        raise ValueError(
            "The downstream GNN fine-tuning experiment requires exactly 50 "
            f"augmentations per page; {augmentation_path} specifies "
            f"{augmentation_config.general.num_augmentations_per_sample}."
        )
    if not 0.0 <= history_replay_ratio <= 1.0:
        raise ValueError("history_replay_ratio must be between 0.0 and 1.0.")
    if trainable_scope not in {
        TRAINABLE_SCOPE_FULL_MODEL,
        TRAINABLE_SCOPE_HEAD_AND_FINAL_GNN_LAYER,
    }:
        raise ValueError(
            f"Unsupported GNN trainable_scope {trainable_scope!r}."
        )
    deleted_node_supervision_enabled = bool(
        deleted_node_supervision.get("enabled", False)
    )
    if deleted_node_supervision_enabled:
        match_tolerance = float(
            deleted_node_supervision["match_tolerance_image_pixels"]
        )
        if match_tolerance < 0.0:
            raise ValueError(
                "deleted_node_supervision.match_tolerance_image_pixels must "
                "be nonnegative."
            )
        noise_label = int(deleted_node_supervision["noise_textline_label"])
        if noise_label != -1:
            raise ValueError(
                "Deleted CRAFT nodes must use noise_textline_label -1 so the "
                "ground-truth edge builder gives them no positive edges."
            )
        deleted_node_supervision = {
            "enabled": True,
            "match_tolerance_image_pixels": match_tolerance,
            "noise_textline_label": noise_label,
        }
    else:
        deleted_node_supervision = {"enabled": False}
    required_training_keys = {
        "epochs",
        "batch_size",
        "learning_rate",
        "optimizer",
        "imbalance_handler",
    }
    missing_training_keys = required_training_keys - set(
        training_config.get("training_params") or {}
    )
    if missing_training_keys:
        raise ValueError(
            f"Missing GNN training parameters in {path}: "
            f"{sorted(missing_training_keys)}"
        )

    return GNNFineTuningRecipe(
        config_path=path,
        augmentation_config_path=augmentation_path,
        preprocessing_config_path=preprocessing_path,
        history_replay_ratio=history_replay_ratio,
        expected_model_class=expected_model_class,
        expected_backbone_class=expected_backbone_class,
        trainable_scope=trainable_scope,
        deleted_node_supervision=deleted_node_supervision,
        augmentation_config=augmentation_config,
        preprocessing_config=preprocessing_config,
        training_config=training_config,
    )


def _corrected_graph_source_paths(source_dir: Path, page_id: str) -> dict[str, Path]:
    paths = {
        "dims": source_dir / f"{page_id}_dims.txt",
        "inputs_normalized": source_dir / f"{page_id}_inputs_normalized.txt",
        "inputs_unnormalized": source_dir / f"{page_id}_inputs_unnormalized.txt",
        "labels_textline": source_dir / f"{page_id}_labels_textline.txt",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Corrected GNN supervision is incomplete for page {page_id}: {missing}"
        )
    return paths


def _raw_graph_source_paths(source_dir: Path, page_id: str) -> dict[str, Path]:
    paths = {
        "dims": source_dir / f"{page_id}_dims.txt",
        "inputs_normalized": source_dir / f"{page_id}_inputs_normalized.txt",
        "inputs_unnormalized": source_dir / f"{page_id}_inputs_unnormalized.txt",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Raw CRAFT GNN inputs are incomplete for page {page_id}: {missing}"
        )
    return paths


def _load_points(path: Path) -> np.ndarray:
    points = np.loadtxt(path, ndmin=2)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"Expected an N x 3 point matrix in {path}, got {points.shape}."
        )
    return points


def _match_raw_to_corrected_nodes(
    raw_points_normalized: np.ndarray,
    corrected_points_normalized: np.ndarray,
    *,
    image_scale: float,
    tolerance_image_pixels: float,
) -> tuple[dict[int, int], dict[int, float]]:
    """Greedily match original CRAFT points to unchanged corrected points."""
    unmatched_corrected = set(range(len(corrected_points_normalized)))
    matches: dict[int, int] = {}
    match_distances: dict[int, float] = {}
    for raw_index, raw_point in enumerate(raw_points_normalized):
        nearest = min(
            (
                (
                    float(
                        np.linalg.norm(
                            (
                                corrected_points_normalized[corrected_index, :2]
                                - raw_point[:2]
                            )
                            * image_scale
                        )
                    ),
                    corrected_index,
                )
                for corrected_index in unmatched_corrected
            ),
            default=(float("inf"), None),
        )
        distance, corrected_index = nearest
        if (
            corrected_index is not None
            and distance <= float(tolerance_image_pixels)
        ):
            matches[raw_index] = corrected_index
            match_distances[raw_index] = distance
            unmatched_corrected.remove(corrected_index)
    return matches, match_distances


def prepare_deleted_node_supervision_page(
    *,
    page_id: str,
    raw_source_dir: Path,
    corrected_source_dir: Path,
    output_dir: Path,
    recipe: GNNFineTuningRecipe,
) -> dict:
    """Append only deleted original CRAFT nodes as noise-labelled supervision."""
    raw_paths = _raw_graph_source_paths(raw_source_dir, page_id)
    corrected_paths = _corrected_graph_source_paths(
        corrected_source_dir,
        page_id,
    )
    source_hashes = {
        "raw": {
            name: _sha256(path) for name, path in sorted(raw_paths.items())
        },
        "corrected": {
            name: _sha256(path)
            for name, path in sorted(corrected_paths.items())
        },
    }
    manifest_path = output_dir / f"{page_id}_deleted_node_manifest.json"
    output_paths = {
        name: output_dir / f"{page_id}_{suffix}"
        for name, suffix in {
            "dims": "dims.txt",
            "inputs_normalized": "inputs_normalized.txt",
            "inputs_unnormalized": "inputs_unnormalized.txt",
            "labels_textline": "labels_textline.txt",
        }.items()
    }
    supervision_config = dict(recipe.deleted_node_supervision)
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        outputs_valid = all(
            path.is_file()
            and manifest.get("output_sha256", {}).get(name) == _sha256(path)
            for name, path in output_paths.items()
        )
        if (
            manifest.get("source_sha256") == source_hashes
            and manifest.get("deleted_node_supervision") == supervision_config
            and outputs_valid
        ):
            return manifest
        raise RuntimeError(
            "Refusing to reuse stale deleted-node supervision for "
            f"{page_id}: {output_dir}"
        )
    if any(path.exists() for path in output_paths.values()):
        raise RuntimeError(
            "Deleted-node supervision outputs exist without a valid manifest "
            f"for {page_id}: {output_dir}"
        )

    raw_normalized = _load_points(raw_paths["inputs_normalized"])
    raw_unnormalized = _load_points(raw_paths["inputs_unnormalized"])
    corrected_normalized = _load_points(
        corrected_paths["inputs_normalized"]
    )
    corrected_unnormalized = _load_points(
        corrected_paths["inputs_unnormalized"]
    )
    corrected_labels = np.atleast_1d(
        np.loadtxt(corrected_paths["labels_textline"], dtype=int)
    )
    if len(raw_normalized) != len(raw_unnormalized):
        raise ValueError(f"Raw point-count mismatch for page {page_id}.")
    if not (
        len(corrected_normalized)
        == len(corrected_unnormalized)
        == len(corrected_labels)
    ):
        raise ValueError(f"Corrected point/label-count mismatch for page {page_id}.")

    raw_dims = np.atleast_1d(np.loadtxt(raw_paths["dims"]))
    corrected_dims = np.atleast_1d(np.loadtxt(corrected_paths["dims"]))
    if raw_dims.size < 2 or corrected_dims.size < 2:
        raise ValueError(f"Malformed dimensions for page {page_id}.")
    if not np.allclose(raw_dims[:2], corrected_dims[:2]):
        raise ValueError(
            f"Raw and corrected dimensions differ for page {page_id}: "
            f"{raw_dims[:2]} vs {corrected_dims[:2]}."
        )

    image_scale = 2.0 * float(max(raw_dims[0], raw_dims[1]))
    matches, match_distances = _match_raw_to_corrected_nodes(
        raw_normalized,
        corrected_normalized,
        image_scale=image_scale,
        tolerance_image_pixels=float(
            supervision_config["match_tolerance_image_pixels"]
        ),
    )
    deleted_raw_indices = tuple(
        sorted(set(range(len(raw_normalized))) - set(matches))
    )
    if deleted_raw_indices:
        deleted_indices_array = np.asarray(deleted_raw_indices, dtype=int)
        output_normalized = np.concatenate(
            (corrected_normalized, raw_normalized[deleted_indices_array]),
            axis=0,
        )
        output_unnormalized = np.concatenate(
            (corrected_unnormalized, raw_unnormalized[deleted_indices_array]),
            axis=0,
        )
        output_labels = np.concatenate(
            (
                corrected_labels,
                np.full(
                    len(deleted_raw_indices),
                    int(supervision_config["noise_textline_label"]),
                    dtype=int,
                ),
            )
        )
    else:
        output_normalized = corrected_normalized.copy()
        output_unnormalized = corrected_unnormalized.copy()
        output_labels = corrected_labels.copy()

    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(corrected_paths["dims"], output_paths["dims"])
    np.savetxt(
        output_paths["inputs_normalized"],
        output_normalized,
        fmt="%.6f %.6f %.6f",
    )
    np.savetxt(
        output_paths["inputs_unnormalized"],
        output_unnormalized,
        fmt="%.6f %.6f %.6f",
    )
    np.savetxt(output_paths["labels_textline"], output_labels, fmt="%d")

    corrected_unmatched_count = (
        len(corrected_normalized) - len(set(matches.values()))
    )
    manifest = {
        "schema_version": 1,
        "kind": "fine_tuning_deleted_craft_node_supervision",
        "page_id": page_id,
        "raw_source_dir": str(raw_source_dir.resolve()),
        "corrected_source_dir": str(corrected_source_dir.resolve()),
        "source_sha256": source_hashes,
        "deleted_node_supervision": supervision_config,
        "raw_node_count": len(raw_normalized),
        "corrected_node_count": len(corrected_normalized),
        "matched_raw_node_count": len(matches),
        "recovered_deleted_raw_node_count": len(deleted_raw_indices),
        "recovered_deleted_raw_node_indices": list(deleted_raw_indices),
        "corrected_unmatched_node_count_left_untouched": (
            corrected_unmatched_count
        ),
        "maximum_match_distance_image_pixels": max(
            match_distances.values(),
            default=0.0,
        ),
        "output_node_count": len(output_normalized),
        "deleted_nodes_have_no_positive_ground_truth_edges": True,
        "output_sha256": {
            name: _sha256(path) for name, path in output_paths.items()
        },
    }
    _write_json(manifest_path, manifest)
    return manifest


def _augmentation_output_is_valid(
    *,
    output_dir: Path,
    manifest_path: Path,
    page_id: str,
    expected_ids: tuple[str, ...],
    recipe: GNNFineTuningRecipe,
    source_hashes: dict[str, str],
) -> bool:
    if not manifest_path.is_file():
        return False
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if payload.get("page_id") != page_id:
        return False
    if payload.get("augmentation_config_sha256") != _sha256(
        recipe.augmentation_config_path
    ):
        return False
    if payload.get("source_sha256") != source_hashes:
        return False
    if tuple(payload.get("augmented_page_ids") or ()) != expected_ids:
        return False
    suffixes = (
        "_dims.txt",
        "_inputs_normalized.txt",
        "_inputs_unnormalized.txt",
        "_labels_textline.txt",
    )
    return all(
        (output_dir / f"{augmented_id}{suffix}").is_file()
        for augmented_id in expected_ids
        for suffix in suffixes
    )


def augment_corrected_graph_page(
    *,
    page_id: str,
    page_index: int,
    source_dir: Path,
    output_dir: Path,
    recipe: GNNFineTuningRecipe,
) -> tuple[str, ...]:
    """Generate the exact 50 configured variants for one corrected training page."""
    source_paths = _corrected_graph_source_paths(source_dir, page_id)
    source_hashes = {
        name: _sha256(path) for name, path in sorted(source_paths.items())
    }
    count = recipe.augmentations_per_page
    expected_ids = tuple(f"{page_id}_{index}" for index in range(count))
    manifest_path = output_dir / "augmentation_manifest.json"
    if output_dir.exists():
        if _augmentation_output_is_valid(
            output_dir=output_dir,
            manifest_path=manifest_path,
            page_id=page_id,
            expected_ids=expected_ids,
            recipe=recipe,
            source_hashes=source_hashes,
        ):
            return expected_ids
        if any(output_dir.iterdir()):
            raise RuntimeError(
                "Refusing to mix stale GNN augmentation artifacts. Use a new "
                f"experiment output root or inspect: {output_dir}"
            )
    output_dir.mkdir(parents=True, exist_ok=True)

    base_seed = int(recipe.augmentation_config.general.base_seed)
    tasks = [
        (
            page_id,
            page_index,
            augmentation_index,
            base_seed + page_index * count + augmentation_index,
            recipe.augmentation_config,
            source_dir,
            output_dir,
        )
        for augmentation_index in range(count)
    ]
    configured_workers = int(recipe.augmentation_config.general.num_workers)
    worker_count = (
        (os.cpu_count() or 1)
        if configured_workers == -1
        else configured_workers
    )
    worker_count = max(1, min(int(worker_count), len(tasks)))
    if worker_count > 1:
        context = multiprocessing.get_context("spawn" if os.name == "nt" else None)
        with context.Pool(processes=worker_count) as pool:
            results = pool.map(_augment_single_instance, tasks)
    else:
        results = [_augment_single_instance(task) for task in tasks]
    errors = [result for result in results if result is not None]
    if errors:
        raise RuntimeError(
            f"Failed to augment corrected GNN page {page_id}: {errors[:3]}"
        )

    _write_json(
        manifest_path,
        {
            "schema_version": 1,
            "page_id": page_id,
            "page_index": page_index,
            "source_dir": str(source_dir.resolve()),
            "source_sha256": source_hashes,
            "augmentation_config_path": str(
                recipe.augmentation_config_path.resolve()
            ),
            "augmentation_config_sha256": _sha256(
                recipe.augmentation_config_path
            ),
            "base_seed": base_seed,
            "augmentations_per_page": count,
            "augmented_page_ids": list(expected_ids),
        },
    )
    return expected_ids


def select_history_replay_page_ids(
    augmented_page_ids: Iterable[str],
    *,
    replay_ratio: float,
    seed: int,
) -> tuple[str, ...]:
    candidates = tuple(sorted(str(page_id) for page_id in augmented_page_ids))
    if not candidates or replay_ratio <= 0.0:
        return ()
    replay_count = min(
        len(candidates),
        max(1, int(round(len(candidates) * float(replay_ratio)))),
    )
    rng = random.Random(int(seed))
    return tuple(sorted(rng.sample(candidates, replay_count)))


def _process_graph_or_fail(
    page_id: str,
    source_dir: Path,
    preprocessing_config: DatasetCreationConfig,
):
    data = process_page(page_id, source_dir, preprocessing_config)
    if data is None:
        raise ValueError(
            f"Could not create GNN edge-classification data for {page_id} "
            f"from {source_dir}."
        )
    return data


def _loss_function(training_params: dict, training_data: list):
    imbalance_handler = str(training_params["imbalance_handler"])
    if imbalance_handler == "weighted_loss":
        labels = torch.cat([data.edge_y for data in training_data])
        positive_count = int(labels.sum().item())
        negative_count = int(labels.numel() - positive_count)
        if positive_count and negative_count:
            weights = torch.tensor(
                [
                    labels.numel() / (2 * negative_count),
                    labels.numel() / (2 * positive_count),
                ],
                dtype=torch.float32,
            )
        else:
            weights = torch.tensor([1.0, 1.0], dtype=torch.float32)
        return torch.nn.CrossEntropyLoss(weight=weights)
    if imbalance_handler == "focal_loss":
        return FocalLoss(
            alpha=float(training_params["focal_loss_alpha"]),
            gamma=float(training_params["focal_loss_gamma"]),
        )
    if imbalance_handler == "none":
        return torch.nn.CrossEntropyLoss()
    raise ValueError(f"Unknown GNN imbalance handler: {imbalance_handler}")


def _checkpoint_selection_score(value: float, mode: str) -> float:
    if mode == "max":
        return value
    if mode == "min":
        return -value
    raise ValueError(f"Unsupported early stopping mode: {mode}")


def configure_gnn_trainable_parameters(model, trainable_scope: str) -> dict:
    """Freeze the checkpoint except for the requested, architecture-checked scope."""
    if trainable_scope not in {
        TRAINABLE_SCOPE_FULL_MODEL,
        TRAINABLE_SCOPE_HEAD_AND_FINAL_GNN_LAYER,
    }:
        raise ValueError(f"Unsupported GNN trainable_scope: {trainable_scope!r}")

    predictor = getattr(model, "predictor", None)
    backbone = getattr(model, "backbone", None)
    convs = getattr(backbone, "convs", None)
    if not isinstance(predictor, torch.nn.Module):
        raise ValueError("The pretrained GNN checkpoint has no predictor module.")
    if not isinstance(convs, torch.nn.ModuleList) or not convs:
        raise ValueError(
            "The pretrained GNN backbone has no non-empty ModuleList named convs."
        )

    if trainable_scope == TRAINABLE_SCOPE_FULL_MODEL:
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        allowed_prefixes = None
        trainable_final_gnn_layer_index = None
    else:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in predictor.parameters():
            parameter.requires_grad_(True)
        for parameter in convs[-1].parameters():
            parameter.requires_grad_(True)
        allowed_prefixes = (
            "predictor.",
            f"backbone.convs.{len(convs) - 1}.",
        )
        trainable_final_gnn_layer_index = len(convs) - 1

    trainable_names = tuple(
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    )
    unexpected_names = tuple(
        name
        for name in trainable_names
        if allowed_prefixes is not None
        and not name.startswith(allowed_prefixes)
    )
    if not trainable_names or unexpected_names:
        raise RuntimeError(
            "The requested GNN freeze scope produced an invalid trainable set: "
            f"{unexpected_names or trainable_names}."
        )
    if (
        trainable_scope == TRAINABLE_SCOPE_HEAD_AND_FINAL_GNN_LAYER
        and not any(name.startswith("predictor.") for name in trainable_names)
    ):
        raise RuntimeError("The GNN predictor has no trainable parameters.")
    if (
        trainable_scope == TRAINABLE_SCOPE_HEAD_AND_FINAL_GNN_LAYER
        and not any(
            name.startswith(allowed_prefixes[1]) for name in trainable_names
        )
    ):
        raise RuntimeError("The final GNN layer has no trainable parameters.")

    total_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return {
        "trainable_scope": trainable_scope,
        "message_passing_layer_count": len(convs),
        "trainable_final_gnn_layer_index": trainable_final_gnn_layer_index,
        "total_parameter_count": total_parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "frozen_parameter_count": total_parameter_count - trainable_parameter_count,
        "trainable_parameter_names": list(trainable_names),
    }


def _save_finetuned_checkpoint(
    *,
    path: Path,
    model,
    optimizer,
    epoch: int,
    metrics: dict,
    fine_tuning_metadata: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "model": model,
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "fine_tuning": fine_tuning_metadata,
        },
        path,
    )


def fine_tune_gnn_checkpoint(
    *,
    base_checkpoint: Path,
    training_data: list,
    validation_data: list,
    output_dir: Path,
    recipe: GNNFineTuningRecipe,
    step_index: int,
    current_page_id: str,
    history_page_ids: tuple[str, ...],
    training_sample_ids: tuple[str, ...],
    validation_page_ids: tuple[str, ...],
) -> GNNStepResult:
    """Continue the unchanged checkpoint model using the checked-in GNN recipe."""
    if not training_data:
        raise ValueError("GNN fine-tuning requires at least one training graph.")
    if not validation_data:
        raise ValueError("GNN fine-tuning requires fold-local validation graphs.")

    config = recipe.training_config
    training_params = config["training_params"]
    set_seed(int(config["random_seed"]) + int(step_index))
    device = get_device(config.get("device", "auto"))
    checkpoint = torch.load(
        base_checkpoint,
        map_location=device,
        weights_only=False,
    )
    model = checkpoint.get("model")
    if not isinstance(model, torch.nn.Module):
        raise ValueError(f"GNN checkpoint has no loadable model object: {base_checkpoint}")
    model_class = f"{model.__class__.__module__}.{model.__class__.__qualname__}"
    if model_class != recipe.expected_model_class:
        raise ValueError(
            "The pretrained GNN checkpoint model class does not match the "
            f"fine-tuning contract: expected {recipe.expected_model_class}, "
            f"got {model_class}."
        )
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        raise ValueError("The pretrained GNN checkpoint has no backbone.")
    backbone_class = (
        f"{backbone.__class__.__module__}.{backbone.__class__.__qualname__}"
    )
    if backbone_class != recipe.expected_backbone_class:
        raise ValueError(
            "The pretrained GNN checkpoint backbone does not match the "
            f"fine-tuning contract: expected {recipe.expected_backbone_class}, "
            f"got {backbone_class}."
        )
    model = model.to(device)
    trainable_parameter_metadata = configure_gnn_trainable_parameters(
        model,
        recipe.trainable_scope,
    )

    sample = training_data[0].to(device)
    try:
        with torch.no_grad():
            logits = model(sample.x, sample.edge_index, sample.edge_attr)
        if logits.ndim != 2 or logits.shape[0] != sample.edge_y.shape[0] or logits.shape[1] != 2:
            raise ValueError(
                f"Unexpected GNN output shape {tuple(logits.shape)} for "
                f"{sample.edge_y.shape[0]} binary edge labels."
            )
    finally:
        sample = sample.cpu()

    batch_size = int(training_params["batch_size"])
    train_loader = DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    optimizer_name = str(training_params["optimizer"])
    try:
        optimizer_class = getattr(torch.optim, optimizer_name)
    except AttributeError as exc:
        raise ValueError(f"Unknown GNN optimizer: {optimizer_name}") from exc
    optimizer = optimizer_class(
        [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ],
        lr=float(training_params["learning_rate"]),
    )
    loss_fn = _loss_function(training_params, training_data).to(device)

    warmup_epochs = int(training_params.get("lr_warmup_epochs", 0))
    scheduler = None
    if warmup_epochs > 0:
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: (
                float(epoch + 1) / float(warmup_epochs)
                if epoch < warmup_epochs
                else 1.0
            ),
        )

    checkpoint_metric = str(config["checkpoint_metric"])
    validation_metric_key = checkpoint_metric.removeprefix("val_")
    patience = int(training_params.get("early_stopping_patience", 10))
    min_delta = float(training_params.get("early_stopping_min_delta", 0.001))
    selection_mode = str(training_params.get("early_stopping_mode", "max"))
    epochs = int(training_params["epochs"])
    output_checkpoint = output_dir / "best_model.pt"
    epoch_records: list[dict] = []
    best_score = None
    best_value = None
    best_epoch = None
    epochs_without_improvement = 0

    fixed_metadata = {
        "schema_version": 1,
        "kind": "downstream_ocr_fold_local_gnn_finetune",
        "step_index": int(step_index),
        "current_page_id": current_page_id,
        "history_page_ids": list(history_page_ids),
        "training_sample_ids": list(training_sample_ids),
        "validation_page_ids": list(validation_page_ids),
        "parent_checkpoint": str(base_checkpoint.resolve()),
        "parent_checkpoint_sha256": _sha256(base_checkpoint),
        "optimizer_state_reused": False,
        "model_class": model_class,
        "backbone_class": backbone_class,
        "model_parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        **trainable_parameter_metadata,
        "recipe": recipe.metadata(),
    }

    # Match the OCR ladder's ``train_seconds`` contract: measure the model
    # continuation itself, while excluding graph augmentation/preparation and
    # metadata serialization. CUDA synchronization keeps the wall-clock value
    # accurate when kernels are queued asynchronously.
    if getattr(device, "type", None) == "cuda":
        torch.cuda.synchronize(device)
    training_started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
        )
        validation_metrics, _, _ = evaluate(
            model,
            validation_loader,
            loss_fn,
            device,
        )
        if validation_metric_key not in validation_metrics:
            raise KeyError(
                f"Configured checkpoint metric {checkpoint_metric} is unavailable; "
                f"validation produced {sorted(validation_metrics)}."
            )
        metric_value = float(validation_metrics[validation_metric_key])
        if not math.isfinite(metric_value):
            raise ValueError(
                f"Non-finite GNN checkpoint metric at epoch {epoch}: {metric_value}"
            )
        learning_rate = float(optimizer.param_groups[0]["lr"])
        epoch_records.append(
            {
                "epoch": epoch,
                "learning_rate": learning_rate,
                "train": train_metrics,
                "validation": validation_metrics,
            }
        )

        score = _checkpoint_selection_score(metric_value, selection_mode)
        improved = best_score is None or score >= best_score + min_delta
        if improved:
            best_score = score
            best_value = metric_value
            best_epoch = epoch
            epochs_without_improvement = 0
            _save_finetuned_checkpoint(
                path=output_checkpoint,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=validation_metrics,
                fine_tuning_metadata={
                    **fixed_metadata,
                    "selected_metric_name": checkpoint_metric,
                    "selected_metric_value": metric_value,
                },
            )
        else:
            epochs_without_improvement += 1

        if scheduler is not None:
            scheduler.step()
        if epochs_without_improvement >= patience:
            break

    if getattr(device, "type", None) == "cuda":
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter() - training_started

    if best_epoch is None or best_value is None or not output_checkpoint.is_file():
        raise RuntimeError(f"GNN fine-tuning did not produce a checkpoint: {output_dir}")
    metadata_path = _write_json(
        output_dir / "fine_tune_metadata.json",
        {
            **fixed_metadata,
            "selected_checkpoint": str(output_checkpoint.resolve()),
            "selected_checkpoint_sha256": _sha256(output_checkpoint),
            "selected_epoch": best_epoch,
            "selected_metric_name": checkpoint_metric,
            "selected_metric_value": best_value,
            "epochs_completed": len(epoch_records),
            "train_seconds": train_seconds,
            "epoch_records": epoch_records,
        },
    )
    return GNNStepResult(
        output_checkpoint=output_checkpoint,
        metadata_path=metadata_path,
        selected_epoch=best_epoch,
        selected_metric=best_value,
    )


def run_gnn_finetuning_ladder(
    *,
    manuscript_root: str | Path,
    train_page_ids: Iterable[str],
    base_checkpoint: str | Path,
    output_root: str | Path,
    config_path: str | Path,
    fine_tune_step_fn: Callable[..., GNNStepResult] | None = None,
) -> GNNLadderResult:
    """Build the sequential 1/2/3-page GNN checkpoint ladder for one fold."""
    selected_train = tuple(str(page_id) for page_id in train_page_ids)
    if not selected_train:
        raise ValueError("The GNN fine-tuning ladder requires training pages.")
    manuscript_root = Path(manuscript_root)
    base_checkpoint = Path(base_checkpoint)
    output_root = Path(output_root)
    raw_graph_dir = manuscript_root / "gnn-dataset"
    corrected_graph_dir = (
        manuscript_root / "layout_analysis_output" / "gnn-format"
    )
    recipe = load_gnn_finetuning_recipe(config_path)
    if not base_checkpoint.is_file():
        raise FileNotFoundError(f"Pretrained GNN checkpoint not found: {base_checkpoint}")
    if not corrected_graph_dir.is_dir():
        raise FileNotFoundError(
            f"Corrected graph-format labels not found: {corrected_graph_dir}"
        )
    if recipe.deleted_node_supervision["enabled"] and not raw_graph_dir.is_dir():
        raise FileNotFoundError(
            f"Raw CRAFT graph-format inputs not found: {raw_graph_dir}"
        )

    deleted_node_supervision_enabled = bool(
        recipe.deleted_node_supervision["enabled"]
    )
    if deleted_node_supervision_enabled:
        training_graph_dir = output_root / "deleted_node_supervision"
        supervision_by_page = {
            page_id: prepare_deleted_node_supervision_page(
                page_id=page_id,
                raw_source_dir=raw_graph_dir,
                corrected_source_dir=corrected_graph_dir,
                output_dir=training_graph_dir,
                recipe=recipe,
            )
            for page_id in selected_train
        }
    else:
        training_graph_dir = corrected_graph_dir
        supervision_by_page = {}
    augmentations_by_page: dict[str, tuple[str, ...]] = {}
    augmentation_dirs_by_page: dict[str, Path] = {}
    for page_index, page_id in enumerate(selected_train):
        augmentation_dir = output_root / "aug" / f"{page_index + 1:02d}"
        augmentation_dirs_by_page[page_id] = augmentation_dir
        augmentations_by_page[page_id] = augment_corrected_graph_page(
            page_id=page_id,
            page_index=page_index,
            source_dir=training_graph_dir,
            output_dir=augmentation_dir,
            recipe=recipe,
        )

    processed_augmented_by_page = {
        page_id: {
            augmented_id: _process_graph_or_fail(
                augmented_id,
                augmentation_dirs_by_page[page_id],
                recipe.preprocessing_config,
            )
            for augmented_id in augmented_ids
        }
        for page_id, augmented_ids in augmentations_by_page.items()
    }
    processed_original_by_page = {
        page_id: _process_graph_or_fail(
            page_id,
            training_graph_dir,
            recipe.preprocessing_config,
        )
        for page_id in selected_train
    }

    if fine_tune_step_fn is None:
        fine_tune_step_fn = fine_tune_gnn_checkpoint
    checkpoint_by_count: dict[int, Path] = {}
    metadata_by_count: dict[int, Path] = {}
    current_checkpoint = base_checkpoint
    base_seed = int(recipe.augmentation_config.general.base_seed)
    step_summaries = []

    for step_index, current_page_id in enumerate(selected_train, start=1):
        history_page_ids = selected_train[: step_index - 1]
        current_augmented_ids = augmentations_by_page[current_page_id]
        training_sample_ids = list(current_augmented_ids)
        training_data = [
            processed_augmented_by_page[current_page_id][page_id]
            for page_id in current_augmented_ids
        ]
        history_replay_ids: dict[str, tuple[str, ...]] = {}
        for history_index, history_page_id in enumerate(history_page_ids):
            replay_ids = select_history_replay_page_ids(
                augmentations_by_page[history_page_id],
                replay_ratio=recipe.history_replay_ratio,
                seed=base_seed + step_index * 10_000 + history_index,
            )
            history_replay_ids[history_page_id] = replay_ids
            training_sample_ids.extend(replay_ids)
            training_data.extend(
                processed_augmented_by_page[history_page_id][page_id]
                for page_id in replay_ids
            )

        validation_page_ids = selected_train[:step_index]
        validation_data = [
            processed_original_by_page[page_id]
            for page_id in validation_page_ids
        ]
        result = fine_tune_step_fn(
            base_checkpoint=current_checkpoint,
            training_data=training_data,
            validation_data=validation_data,
            output_dir=output_root / "steps" / f"{step_index:02d}_{current_page_id}",
            recipe=recipe,
            step_index=step_index,
            current_page_id=current_page_id,
            history_page_ids=history_page_ids,
            training_sample_ids=tuple(training_sample_ids),
            validation_page_ids=validation_page_ids,
        )
        current_checkpoint = Path(result.output_checkpoint)
        checkpoint_by_count[step_index] = current_checkpoint
        metadata_by_count[step_index] = Path(result.metadata_path)
        step_summaries.append(
            {
                "step_index": step_index,
                "current_page_id": current_page_id,
                "history_page_ids": list(history_page_ids),
                "history_replay_ids": {
                    page_id: list(replay_ids)
                    for page_id, replay_ids in history_replay_ids.items()
                },
                "training_sample_count": len(training_sample_ids),
                "validation_page_ids": list(validation_page_ids),
                "checkpoint_path": str(current_checkpoint.resolve()),
                "checkpoint_sha256": _sha256(current_checkpoint),
                "fine_tune_metadata_path": str(Path(result.metadata_path).resolve()),
                "selected_epoch": result.selected_epoch,
                "selected_metric": result.selected_metric,
            }
        )

    summary_path = _write_json(
        output_root / "ladder_summary.json",
        {
            "schema_version": 1,
            "kind": "downstream_ocr_fold_local_gnn_finetuning_ladder",
            "manuscript_root": str(manuscript_root.resolve()),
            "raw_graph_source_dir": str(raw_graph_dir.resolve()),
            "corrected_graph_source_dir": str(corrected_graph_dir.resolve()),
            "training_graph_source_dir": str(training_graph_dir.resolve()),
            "deleted_node_supervision_dir": (
                str(training_graph_dir.resolve())
                if deleted_node_supervision_enabled
                else None
            ),
            "deleted_node_supervision_by_page": supervision_by_page,
            "train_page_ids": list(selected_train),
            "base_checkpoint": str(base_checkpoint.resolve()),
            "base_checkpoint_sha256": _sha256(base_checkpoint),
            "recipe": recipe.metadata(),
            "steps": step_summaries,
        },
    )
    return GNNLadderResult(
        checkpoint_by_count=checkpoint_by_count,
        step_metadata_by_count=metadata_by_count,
        summary_path=summary_path,
        recipe=recipe,
    )
