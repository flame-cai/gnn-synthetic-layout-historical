from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from recognition.active_learning_recipe import OcrActiveLearningRecipe

TESTS_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class PipelinePrecommitDatasetConfig:
    name: str
    manuscript_name: str
    images_dir: Path
    pagexml_dir: Path
    layout_type: str = "simple"
    longest_side: int = 2500
    min_distance: int = 20
    expected_page_count: int = 15
    max_page_cer: float = 0.40
    max_line_cer_50: float = 0.40
    max_line_cer_75: float = 0.45
    max_line_cer_range: float = 0.48
    max_worst_page_line_cer_50: float = 0.55

    def ordered_page_ids(self) -> list[str]:
        return sorted(path.stem for path in self.images_dir.glob("*.jpg"))

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["images_dir"] = str(self.images_dir.resolve())
        payload["pagexml_dir"] = str(self.pagexml_dir.resolve())
        payload["ordered_page_ids"] = self.ordered_page_ids()
        return payload


RecognitionPrecommitRecipe = OcrActiveLearningRecipe


@dataclass(frozen=True)
class RecognitionPrecommitDatasetConfig:
    name: str
    recognition_dataset_config_name: str
    max_curve_metric_value: float
    max_final_page_cer: float
    min_first_step_gain: float
    regression_guard_warning_only: bool = True
    recipe: RecognitionPrecommitRecipe = field(default_factory=RecognitionPrecommitRecipe)

    def to_dict(self) -> dict:
        return asdict(self)


PIPELINE_PRECOMMIT_DATASETS = {
    "eval_dataset": PipelinePrecommitDatasetConfig(
        name="eval_dataset",
        manuscript_name="ci_eval_dataset",
        images_dir=TESTS_ROOT / "eval_dataset" / "images",
        pagexml_dir=TESTS_ROOT / "eval_dataset" / "labels" / "PAGE-XML",
    )
}


RECOGNITION_PRECOMMIT_DATASETS = {
    "eval_dataset": RecognitionPrecommitDatasetConfig(
        name="eval_dataset",
        recognition_dataset_config_name="eval_dataset",
        max_curve_metric_value=0.26,
        max_final_page_cer=0.18,
        min_first_step_gain=0.04,
    )
}


def _ordered_configs(registry: dict[str, object]) -> tuple[object, ...]:
    return tuple(registry[name] for name in sorted(registry))


def get_pipeline_precommit_dataset(name: str = "eval_dataset") -> PipelinePrecommitDatasetConfig:
    if name not in PIPELINE_PRECOMMIT_DATASETS:
        raise KeyError(f"Unknown pipeline pre-commit dataset config: {name}")
    return PIPELINE_PRECOMMIT_DATASETS[name]


def get_pipeline_precommit_datasets() -> tuple[PipelinePrecommitDatasetConfig, ...]:
    return _ordered_configs(PIPELINE_PRECOMMIT_DATASETS)


def get_recognition_precommit_dataset(name: str = "eval_dataset") -> RecognitionPrecommitDatasetConfig:
    if name not in RECOGNITION_PRECOMMIT_DATASETS:
        raise KeyError(f"Unknown recognition pre-commit dataset config: {name}")
    return RECOGNITION_PRECOMMIT_DATASETS[name]


def get_recognition_precommit_datasets() -> tuple[RecognitionPrecommitDatasetConfig, ...]:
    return _ordered_configs(RECOGNITION_PRECOMMIT_DATASETS)
