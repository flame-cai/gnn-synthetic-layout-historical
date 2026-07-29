from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG", ".TIF", ".TIFF", ".jp2")
DEFAULT_SPLIT_SEED = 42
FOLDS_JSON_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Fold:
    fold_id: str
    train_page_ids: tuple[str, ...]
    test_page_ids: tuple[str, ...]


@dataclass(frozen=True)
class ManuscriptPaths:
    manuscript_id: str
    root: Path
    images_dir: Path
    pagexml_dir: Path
    line_images_dir: Path
    heatmaps_dir: Path | None


def default_manuscript_paths(root: str | Path) -> ManuscriptPaths:
    manuscript_root = Path(root)
    return ManuscriptPaths(
        manuscript_id=manuscript_root.name,
        root=manuscript_root,
        images_dir=manuscript_root / "images_resized",
        pagexml_dir=manuscript_root / "layout_analysis_output" / "page-xml-format",
        line_images_dir=manuscript_root / "layout_analysis_output" / "image-format",
        heatmaps_dir=manuscript_root / "heatmaps",
    )


def discover_page_ids(paths: ManuscriptPaths) -> tuple[str, ...]:
    image_ids = {
        path.stem
        for path in paths.images_dir.iterdir()
        if path.is_file() and path.suffix in IMAGE_EXTENSIONS
    }
    xml_ids = {
        path.stem
        for path in paths.pagexml_dir.glob("*.xml")
        if not path.name.endswith("_metadata.xml")
    }
    return tuple(sorted(image_ids & xml_ids))


def make_three_folds(
    page_ids: list[str] | tuple[str, ...],
    *,
    train_size: int = 3,
    fold_count: int = 5,
    seed: int = DEFAULT_SPLIT_SEED,
) -> tuple[Fold, ...]:
    ordered = tuple(sorted(page_ids))
    if len(ordered) <= train_size:
        raise ValueError("Need more pages than the train size to create held-out folds.")
    rng = random.Random(int(seed))
    folds: list[Fold] = []
    for fold_index in range(fold_count):
        train = tuple(rng.sample(ordered, train_size))
        test = tuple(page_id for page_id in ordered if page_id not in set(train))
        folds.append(Fold(fold_id=f"fold_{fold_index + 1}", train_page_ids=train, test_page_ids=test))
    return tuple(folds)


def _fold_from_payload(payload: dict) -> Fold:
    return Fold(
        fold_id=str(payload["fold_id"]),
        train_page_ids=tuple(str(page_id) for page_id in payload["train_page_ids"]),
        test_page_ids=tuple(str(page_id) for page_id in payload["test_page_ids"]),
    )


def _validate_folds(folds: tuple[Fold, ...], page_ids: tuple[str, ...]) -> None:
    known = set(page_ids)
    for fold in folds:
        train = set(fold.train_page_ids)
        test = set(fold.test_page_ids)
        if len(train) != len(fold.train_page_ids):
            raise ValueError(f"{fold.fold_id}: train_page_ids contains duplicates.")
        if train & test:
            raise ValueError(f"{fold.fold_id}: train and test pages overlap.")
        unknown = (train | test) - known
        if unknown:
            raise ValueError(f"{fold.fold_id}: unknown page ids in folds.json: {sorted(unknown)}")
        if train | test != known:
            raise ValueError(f"{fold.fold_id}: train/test pages do not cover the discovered page set.")


def write_folds_json(
    folds_path: str | Path,
    *,
    manuscript_id: str,
    page_ids: list[str] | tuple[str, ...],
    folds: tuple[Fold, ...],
    split_seed: int = DEFAULT_SPLIT_SEED,
    train_size: int = 3,
    fold_count: int = 5,
) -> Path:
    output = Path(folds_path)
    ordered = tuple(sorted(str(page_id) for page_id in page_ids))
    _validate_folds(folds, ordered)
    payload = {
        "schema_version": FOLDS_JSON_SCHEMA_VERSION,
        "manuscript_id": str(manuscript_id),
        "split_seed": int(split_seed),
        "train_size": int(train_size),
        "fold_count": int(fold_count),
        "page_ids": list(ordered),
        "folds": [asdict(fold) for fold in folds],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return output


def load_folds_json(
    folds_path: str | Path,
    *,
    page_ids: list[str] | tuple[str, ...],
) -> tuple[Fold, ...]:
    path = Path(folds_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    ordered = tuple(sorted(str(page_id) for page_id in page_ids))
    persisted_page_ids = tuple(sorted(str(page_id) for page_id in payload.get("page_ids", ())))
    if persisted_page_ids != ordered:
        raise ValueError(
            f"Persisted folds page set does not match discovered pages: {path}"
        )
    folds = tuple(_fold_from_payload(item) for item in payload.get("folds", ()))
    _validate_folds(folds, ordered)
    return folds


def load_or_create_folds_json(
    folds_path: str | Path,
    *,
    manuscript_id: str,
    page_ids: list[str] | tuple[str, ...],
    split_seed: int = DEFAULT_SPLIT_SEED,
    train_size: int = 3,
    fold_count: int = 5,
) -> tuple[Fold, ...]:
    path = Path(folds_path)
    ordered = tuple(sorted(str(page_id) for page_id in page_ids))
    if path.exists():
        return load_folds_json(path, page_ids=ordered)
    folds = make_three_folds(
        ordered,
        train_size=train_size,
        fold_count=fold_count,
        seed=split_seed,
    )
    write_folds_json(
        path,
        manuscript_id=manuscript_id,
        page_ids=ordered,
        folds=folds,
        split_seed=split_seed,
        train_size=train_size,
        fold_count=fold_count,
    )
    return folds
