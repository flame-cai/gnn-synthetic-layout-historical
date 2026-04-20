from __future__ import annotations

from dataclasses import asdict, dataclass


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
