from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path


TESTS_ROOT = Path(__file__).resolve().parent


def _normalize_optimizer_name(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in {"adadelta", "adam"}:
        raise ValueError(f"Unsupported optimizer: {value}")
    return normalized


@dataclass(frozen=True)
class FocusedStructuralPolicy:
    slug: str
    width_policy: str
    oversampling_policy: str
    augmentation_policy: str

    def apply(self, config: "RecognitionEvalDatasetConfig") -> "RecognitionEvalDatasetConfig":
        return config.with_updates(
            width_policy=self.width_policy,
            oversampling_policy=self.oversampling_policy,
            augmentation_policy=self.augmentation_policy,
        )


DEFAULT_FOCUSED_STRUCTURAL_POLICIES = (
    FocusedStructuralPolicy(
        slug="wb_oc_ar",
        width_policy="batch_max_pad",
        oversampling_policy="cer_weighted",
        augmentation_policy="background_plus_rotation",
    ),
    FocusedStructuralPolicy(
        slug="wb_oc_an",
        width_policy="batch_max_pad",
        oversampling_policy="cer_weighted",
        augmentation_policy="none",
    ),
    FocusedStructuralPolicy(
        slug="wb_on_an",
        width_policy="batch_max_pad",
        oversampling_policy="none",
        augmentation_policy="none",
    ),
)


DEFAULT_TRAINING_OVERRIDES = {
    "num_iter": 60,
    "valInterval": 5,
    "lr": 0.2,
    "adam": False,
    "batch_size": 1,
    "workers": 0,
}


PAGE_ONLY_FOLLOWUP_POLICIES = (
    {"optimizer": "adam", "lr": 0.00005, "num_iter": 60},
    {"optimizer": "adam", "lr": 0.00001, "num_iter": 200},
    {"optimizer": "adadelta", "lr": 0.2, "num_iter": 60},
    {"optimizer": "adadelta", "lr": 0.05, "num_iter": 200},
)


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
    training_policy: str = "cumulative"
    history_sample_line_count: int = 0
    validation_ratio: float = 0.0
    split_seed: int = 42
    width_policy: str = "batch_max_pad"
    oversampling_policy: str = "none"
    augmentation_policy: str = "none"
    lr_scheduler: str = "none"
    optimizer: str = "adadelta"
    regression_guard_abs: float = 0.005
    curve_metric: str = "early_weighted_page_cer"
    focused_learning_rates: tuple[float, ...] = (0.001, 0.01, 0.2, 0.5)
    focused_optimizers: tuple[str, ...] = ("adadelta", "adam")
    focused_structural_policies: tuple[FocusedStructuralPolicy, ...] = field(
        default_factory=lambda: DEFAULT_FOCUSED_STRUCTURAL_POLICIES
    )
    background_plus_rotation_variant_count: int = 10
    shuffle_train_each_epoch: bool = True
    training_overrides: dict = field(default_factory=lambda: dict(DEFAULT_TRAINING_OVERRIDES))

    def __post_init__(self):
        normalized_optimizer = _normalize_optimizer_name(self.optimizer)
        normalized_overrides = dict(self.training_overrides)
        normalized_overrides["adam"] = normalized_optimizer == "adam"
        object.__setattr__(self, "optimizer", normalized_optimizer)
        object.__setattr__(self, "training_overrides", normalized_overrides)

    def ordered_page_ids(self):
        return sorted(path.stem for path in self.images_dir.glob("*.jpg"))

    def fine_tune_page_ids(self):
        ordered = self.ordered_page_ids()
        return ordered[: self.fine_tune_page_count]

    def evaluation_page_ids(self):
        ordered = self.ordered_page_ids()
        return ordered[self.eval_page_start_index : self.eval_page_end_index]

    def to_dict(self):
        payload = asdict(self)
        payload["images_dir"] = str(self.images_dir.resolve())
        payload["pagexml_dir"] = str(self.pagexml_dir.resolve())
        payload["ordered_page_ids"] = self.ordered_page_ids()
        payload["fine_tune_page_ids"] = self.fine_tune_page_ids()
        payload["evaluation_page_ids"] = self.evaluation_page_ids()
        return payload

    def with_updates(self, **changes):
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

    def with_learning_rate(self, lr: float):
        updated_overrides = dict(self.training_overrides)
        updated_overrides["lr"] = float(lr)
        return self.with_updates(training_overrides=updated_overrides)

    def with_optimizer(self, optimizer: str):
        return self.with_updates(optimizer=optimizer)

    def with_structural_policy(self, policy: FocusedStructuralPolicy):
        return policy.apply(self)


DATASET_CONFIGS = {
    "eval_dataset": RecognitionEvalDatasetConfig(
        name="eval_dataset",
        images_dir=TESTS_ROOT / "eval_dataset" / "images",
        pagexml_dir=TESTS_ROOT / "eval_dataset" / "labels" / "PAGE-XML",
    )
}


def get_dataset_config(name="eval_dataset"):
    if name not in DATASET_CONFIGS:
        raise KeyError(f"Unknown recognition evaluation dataset config: {name}")
    return DATASET_CONFIGS[name]


def get_historical_broad_search_config(name="eval_dataset"):
    return get_dataset_config(name).with_updates(
        fine_tune_page_count=5,
        width_policy="global_2000_pad",
        oversampling_policy="none",
        augmentation_policy="none",
        optimizer="adadelta",
        background_plus_rotation_variant_count=1,
    )


def get_page_only_followup_policy_configs(name: str = "eval_dataset") -> tuple[RecognitionEvalDatasetConfig, ...]:
    base_config = get_dataset_config(name).with_updates(training_policy="page_only")
    configs = []

    for policy in PAGE_ONLY_FOLLOWUP_POLICIES:
        policy_config = base_config.with_optimizer(policy["optimizer"])
        training_overrides = dict(policy_config.training_overrides)
        training_overrides["lr"] = float(policy["lr"])
        training_overrides["num_iter"] = int(policy["num_iter"])
        configs.append(policy_config.with_updates(training_overrides=training_overrides))

    return tuple(configs)


def get_page_plus_random_history_policy_configs(name: str = "eval_dataset") -> tuple[RecognitionEvalDatasetConfig, ...]:
    base_config = get_dataset_config(name).with_updates(
        training_policy="page_plus_random_history",
        history_sample_line_count=10,
    )
    configs = []

    for policy in PAGE_PLUS_RANDOM_HISTORY_POLICIES:
        policy_config = base_config.with_optimizer(policy["optimizer"])
        training_overrides = dict(policy_config.training_overrides)
        training_overrides["lr"] = float(policy["lr"])
        training_overrides["num_iter"] = int(policy["num_iter"])
        configs.append(policy_config.with_updates(training_overrides=training_overrides))

    return tuple(configs)
