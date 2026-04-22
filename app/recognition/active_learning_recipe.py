from __future__ import annotations

from dataclasses import asdict, dataclass


SUPPORTED_SIBLING_CHECKPOINT_STRATEGIES = {
    "page_cer_selector",
    "best_norm_ed",
}


def normalize_sibling_checkpoint_strategy(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in SUPPORTED_SIBLING_CHECKPOINT_STRATEGIES:
        raise ValueError(f"Unsupported sibling checkpoint strategy: {value}")
    return normalized


@dataclass(frozen=True)
class OcrActiveLearningRecipe:
    training_policy: str = "page_plus_random_history"
    history_sample_line_count: int = 10
    width_policy: str = "batch_max_pad"
    oversampling_policy: str = "none"
    augmentation_policy: str = "none"
    lr_scheduler: str = "none"
    optimizer: str = "adadelta"
    lr: float = 0.2
    num_iter: int = 60
    curve_metric: str = "early_weighted_page_cer"
    regression_guard_abs: float = 0.005
    background_plus_rotation_variant_count: int = 10
    shuffle_train_each_epoch: bool = True
    sibling_checkpoint_strategy: str = "page_cer_selector"

    def __post_init__(self):
        object.__setattr__(
            self,
            "sibling_checkpoint_strategy",
            normalize_sibling_checkpoint_strategy(self.sibling_checkpoint_strategy),
        )

    def to_training_overrides(self) -> dict:
        return {
            "lr": float(self.lr),
            "num_iter": int(self.num_iter),
            "adam": str(self.optimizer).strip().lower() == "adam",
        }

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_OCR_ACTIVE_LEARNING_RECIPE = OcrActiveLearningRecipe()


def get_default_ocr_active_learning_recipe() -> OcrActiveLearningRecipe:
    return DEFAULT_OCR_ACTIVE_LEARNING_RECIPE
