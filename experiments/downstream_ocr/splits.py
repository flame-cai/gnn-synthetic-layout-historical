from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG", ".TIF", ".TIFF", ".jp2")


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
    fold_count: int = 3,
) -> tuple[Fold, ...]:
    ordered = tuple(sorted(page_ids))
    if len(ordered) <= train_size:
        raise ValueError("Need more pages than the train size to create held-out folds.")
    folds: list[Fold] = []
    for fold_index in range(fold_count):
        start = fold_index * train_size
        if start + train_size <= len(ordered):
            train = ordered[start : start + train_size]
        else:
            train = tuple(ordered[(start + offset) % len(ordered)] for offset in range(train_size))
        test = tuple(page_id for page_id in ordered if page_id not in set(train))
        folds.append(Fold(fold_id=f"fold_{fold_index + 1}", train_page_ids=train, test_page_ids=test))
    return tuple(folds)

