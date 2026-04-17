from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path


TESTS_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class RecognitionEvalDatasetConfig:
    name: str
    images_dir: Path
    pagexml_dir: Path
    layout_type: str = "simple"
    fine_tune_page_count: int = 5
    eval_page_start_index: int = 9
    eval_page_end_index: int = 15
    training_policy: str = "cumulative"
    validation_ratio: float = 0.0
    split_seed: int = 42
    training_overrides: dict = field(
        default_factory=lambda: {
            "num_iter": 60,
            "valInterval": 5,
            "lr": 0.2,
            "adam": False,
            "batch_size": 1,
            "workers": 0,
        }
    )

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
