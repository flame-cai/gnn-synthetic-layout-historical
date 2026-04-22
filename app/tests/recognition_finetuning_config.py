from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from recognition.active_learning_recipe import DEFAULT_OCR_ACTIVE_LEARNING_RECIPE
from tests.precommit_gate_config import get_recognition_precommit_dataset


TESTS_ROOT = Path(__file__).resolve().parent


def _normalize_optimizer_name(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in {"adadelta", "adam"}:
        raise ValueError(f"Unsupported optimizer: {value}")
    return normalized


DEFAULT_TRAINING_OVERRIDES = {
    "num_iter": int(DEFAULT_OCR_ACTIVE_LEARNING_RECIPE.num_iter),
    "valInterval": 5,
    "lr": float(DEFAULT_OCR_ACTIVE_LEARNING_RECIPE.lr),
    "adam": False,
    "batch_size": 1,
    "workers": 0,
}


PAGE_PLUS_RANDOM_HISTORY_POLICIES = (
    {"optimizer": "adam", "lr": 0.00005, "num_iter": 60},
    {"optimizer": "adadelta", "lr": 0.2, "num_iter": 60},
)


@dataclass(frozen=True)
class RecognitionEvalDatasetConfig:
    name: str
    images_dir: Path
    pagexml_dir: Path
    layout_type: str = "simple"
    fine_tune_page_count: int = 9
    eval_page_start_index: int = 9
    eval_page_end_index: int = 15
    training_policy: str = DEFAULT_OCR_ACTIVE_LEARNING_RECIPE.training_policy
    history_sample_line_count: int = DEFAULT_OCR_ACTIVE_LEARNING_RECIPE.history_sample_line_count
    validation_ratio: float = 0.0
    split_seed: int = 42
    width_policy: str = DEFAULT_OCR_ACTIVE_LEARNING_RECIPE.width_policy
    oversampling_policy: str = DEFAULT_OCR_ACTIVE_LEARNING_RECIPE.oversampling_policy
    augmentation_policy: str = DEFAULT_OCR_ACTIVE_LEARNING_RECIPE.augmentation_policy
    lr_scheduler: str = DEFAULT_OCR_ACTIVE_LEARNING_RECIPE.lr_scheduler
    optimizer: str = DEFAULT_OCR_ACTIVE_LEARNING_RECIPE.optimizer
    sibling_checkpoint_strategy: str = DEFAULT_OCR_ACTIVE_LEARNING_RECIPE.sibling_checkpoint_strategy
    promotion_guard_strategy: str = DEFAULT_OCR_ACTIVE_LEARNING_RECIPE.promotion_guard_strategy
    regression_guard_abs: float = DEFAULT_OCR_ACTIVE_LEARNING_RECIPE.regression_guard_abs
    curve_metric: str = DEFAULT_OCR_ACTIVE_LEARNING_RECIPE.curve_metric
    background_plus_rotation_variant_count: int = (
        DEFAULT_OCR_ACTIVE_LEARNING_RECIPE.background_plus_rotation_variant_count
    )
    shuffle_train_each_epoch: bool = DEFAULT_OCR_ACTIVE_LEARNING_RECIPE.shuffle_train_each_epoch
    training_overrides: dict = field(default_factory=lambda: dict(DEFAULT_TRAINING_OVERRIDES))

    def __post_init__(self):
        normalized_optimizer = _normalize_optimizer_name(self.optimizer)
        normalized_overrides = dict(self.training_overrides)
        normalized_overrides["adam"] = normalized_optimizer == "adam"
        object.__setattr__(self, "optimizer", normalized_optimizer)
        object.__setattr__(self, "training_overrides", normalized_overrides)

    def ordered_page_ids(self) -> list[str]:
        return sorted(path.stem for path in self.images_dir.glob("*.jpg"))

    def fine_tune_page_ids(self) -> list[str]:
        ordered = self.ordered_page_ids()
        return ordered[: self.fine_tune_page_count]

    def evaluation_page_ids(self) -> list[str]:
        ordered = self.ordered_page_ids()
        return ordered[self.eval_page_start_index : self.eval_page_end_index]

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["images_dir"] = str(self.images_dir.resolve())
        payload["pagexml_dir"] = str(self.pagexml_dir.resolve())
        payload["ordered_page_ids"] = self.ordered_page_ids()
        payload["fine_tune_page_ids"] = self.fine_tune_page_ids()
        payload["evaluation_page_ids"] = self.evaluation_page_ids()
        return payload

    def with_updates(self, **changes) -> "RecognitionEvalDatasetConfig":
        normalized_changes = dict(changes)
        updated_overrides = dict(normalized_changes.get("training_overrides", self.training_overrides))
        if "optimizer" in normalized_changes:
            optimizer = _normalize_optimizer_name(normalized_changes["optimizer"])
        elif "training_overrides" in normalized_changes and "adam" in updated_overrides:
            optimizer = "adam" if bool(updated_overrides["adam"]) else "adadelta"
        else:
            optimizer = self.optimizer

        updated_overrides["adam"] = optimizer == "adam"
        normalized_changes["optimizer"] = optimizer
        normalized_changes["training_overrides"] = updated_overrides
        return replace(self, **normalized_changes)

    def with_learning_rate(self, lr: float) -> "RecognitionEvalDatasetConfig":
        updated_overrides = dict(self.training_overrides)
        updated_overrides["lr"] = float(lr)
        return self.with_updates(training_overrides=updated_overrides)

    def with_optimizer(self, optimizer: str) -> "RecognitionEvalDatasetConfig":
        return self.with_updates(optimizer=optimizer)


DATASET_CONFIGS = {
    "eval_dataset": RecognitionEvalDatasetConfig(
        name="eval_dataset",
        images_dir=TESTS_ROOT / "eval_dataset" / "images",
        pagexml_dir=TESTS_ROOT / "eval_dataset" / "labels" / "PAGE-XML",
    )
}


def get_dataset_config(name: str = "eval_dataset") -> RecognitionEvalDatasetConfig:
    if name not in DATASET_CONFIGS:
        raise KeyError(f"Unknown recognition evaluation dataset config: {name}")
    return DATASET_CONFIGS[name]


def get_page_plus_random_history_policy_configs(name: str = "eval_dataset") -> tuple[RecognitionEvalDatasetConfig, ...]:
    base_config = get_dataset_config(name).with_updates(
        training_policy="page_plus_random_history",
        history_sample_line_count=10,
        width_policy="batch_max_pad",
        oversampling_policy="none",
        augmentation_policy="none",
        lr_scheduler="none",
    )
    configs = []

    for policy in PAGE_PLUS_RANDOM_HISTORY_POLICIES:
        policy_config = base_config.with_optimizer(policy["optimizer"])
        training_overrides = dict(policy_config.training_overrides)
        training_overrides["lr"] = float(policy["lr"])
        training_overrides["num_iter"] = int(policy["num_iter"])
        configs.append(policy_config.with_updates(training_overrides=training_overrides))

    return tuple(configs)


def get_precommit_hybrid_recognition_gate_config(name: str = "eval_dataset") -> RecognitionEvalDatasetConfig:
    gate_config = get_recognition_precommit_dataset(name)
    recipe = gate_config.recipe
    base_config = get_dataset_config(gate_config.recognition_dataset_config_name).with_updates(
        training_policy=recipe.training_policy,
        history_sample_line_count=int(recipe.history_sample_line_count),
        width_policy=recipe.width_policy,
        oversampling_policy=recipe.oversampling_policy,
        augmentation_policy=recipe.augmentation_policy,
        lr_scheduler=recipe.lr_scheduler,
        optimizer=recipe.optimizer,
        sibling_checkpoint_strategy=recipe.sibling_checkpoint_strategy,
        promotion_guard_strategy=recipe.promotion_guard_strategy,
        curve_metric=recipe.curve_metric,
        regression_guard_abs=float(recipe.regression_guard_abs),
        background_plus_rotation_variant_count=int(recipe.background_plus_rotation_variant_count),
        shuffle_train_each_epoch=bool(recipe.shuffle_train_each_epoch),
    )
    training_overrides = dict(base_config.training_overrides)
    training_overrides["lr"] = float(recipe.lr)
    training_overrides["num_iter"] = int(recipe.num_iter)
    return base_config.with_updates(training_overrides=training_overrides)
