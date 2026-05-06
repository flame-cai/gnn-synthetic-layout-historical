from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from recognition.active_learning_recipe import DEFAULT_OCR_ACTIVE_LEARNING_RECIPE


TESTS_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = TESTS_ROOT.parent
REPO_ROOT = APP_ROOT.parent
EXPERIMENT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = TESTS_ROOT / "eval_dataset_v2"
LOGS_ROOT = TESTS_ROOT / "logs"
BASELINE_JSON_PATH = EXPERIMENT_ROOT / "baseline.json"
LATEST_BASENAME = "circular_ocr_latest"


@dataclass(frozen=True)
class CircularDatasetConfig:
    name: str = "eval_dataset_v2"
    root: Path = DATASET_ROOT
    fine_tune_page_count: int = 3
    eval_page_count: int = 2
    strategy_name: str = "local_tangent_band_v1"
    segmentation_config: dict = field(
        default_factory=lambda: {
            "half_width_px": 42,
            "end_padding_px": 14,
            "point_snap_radius_px": 65,
            "min_polygon_area_px": 16,
        }
    )
    unwrapping_config: dict = field(
        default_factory=lambda: {
            "half_height_px": 42,
            "sample_step_px": 2,
            "min_width_px": 12,
            "candidate_names": [
                "forward",
                "reversed",
                "forward_vertical_flip",
                "reversed_vertical_flip",
            ],
        }
    )
    training_policy: str = "page_plus_random_history"
    history_sample_line_count: int = 10
    width_policy: str = "batch_max_pad"
    oversampling_policy: str = "none"
    augmentation_policy: str = "none"
    lr_scheduler: str = "none"
    optimizer: str = "adadelta"
    lr: float = 0.2
    num_iter: int = 60
    sibling_checkpoint_strategy: str = DEFAULT_OCR_ACTIVE_LEARNING_RECIPE.sibling_checkpoint_strategy
    regression_guard_abs: float = DEFAULT_OCR_ACTIVE_LEARNING_RECIPE.regression_guard_abs
    curve_metric: str = DEFAULT_OCR_ACTIVE_LEARNING_RECIPE.curve_metric
    shuffle_train_each_epoch: bool = DEFAULT_OCR_ACTIVE_LEARNING_RECIPE.shuffle_train_each_epoch

    @property
    def images_dir(self) -> Path:
        return self.root / "images_resized"

    @property
    def heatmaps_dir(self) -> Path:
        return self.root / "heatmaps"

    @property
    def gnn_dir(self) -> Path:
        return self.root / "layout_analysis_output" / "gnn-format"

    @property
    def pagexml_dir(self) -> Path:
        return self.root / "layout_analysis_output" / "page-xml-format"

    def ordered_page_ids(self) -> list[str]:
        return sorted(path.stem for path in self.images_dir.glob("*.jpg"))

    def fine_tune_page_ids(self) -> list[str]:
        return self.ordered_page_ids()[: self.fine_tune_page_count]

    def evaluation_page_ids(self) -> list[str]:
        return self.ordered_page_ids()[self.fine_tune_page_count : self.fine_tune_page_count + self.eval_page_count]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "root": str(self.root.resolve()),
            "images_dir": str(self.images_dir.resolve()),
            "heatmaps_dir": str(self.heatmaps_dir.resolve()),
            "gnn_dir": str(self.gnn_dir.resolve()),
            "pagexml_dir": str(self.pagexml_dir.resolve()),
            "ordered_page_ids": self.ordered_page_ids(),
            "fine_tune_page_ids": self.fine_tune_page_ids(),
            "evaluation_page_ids": self.evaluation_page_ids(),
            "strategy_name": self.strategy_name,
            "segmentation_config": self.segmentation_config,
            "unwrapping_config": self.unwrapping_config,
            "training_policy": self.training_policy,
            "history_sample_line_count": self.history_sample_line_count,
            "width_policy": self.width_policy,
            "oversampling_policy": self.oversampling_policy,
            "augmentation_policy": self.augmentation_policy,
            "lr_scheduler": self.lr_scheduler,
            "optimizer": self.optimizer,
            "lr": self.lr,
            "num_iter": self.num_iter,
            "sibling_checkpoint_strategy": self.sibling_checkpoint_strategy,
            "regression_guard_abs": self.regression_guard_abs,
            "curve_metric": self.curve_metric,
            "shuffle_train_each_epoch": self.shuffle_train_each_epoch,
        }


def get_circular_dataset_config(strategy_name: str | None = None) -> CircularDatasetConfig:
    config = CircularDatasetConfig()
    if strategy_name is None:
        return config
    return CircularDatasetConfig(strategy_name=strategy_name)

