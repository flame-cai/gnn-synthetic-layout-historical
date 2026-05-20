from __future__ import annotations

from pathlib import Path
from typing import Mapping

from .legacy_axis_bound import LegacyAxisBoundStrategy
from .local_polygons import LocalPolygonsStrategy
from .local_tangent_band import LocalTangentBandStrategy
from .types import TextLineSegmentationRequest, TextLineSegmentationResult, TextLineSegmentationStrategy


_STRATEGIES = {
    LegacyAxisBoundStrategy.name: LegacyAxisBoundStrategy(),
    LocalPolygonsStrategy.name: LocalPolygonsStrategy(),
    LocalTangentBandStrategy.name: LocalTangentBandStrategy(),
}


def get_text_line_segmentation_strategy(strategy_name: str) -> TextLineSegmentationStrategy:
    try:
        return _STRATEGIES[strategy_name]
    except KeyError as exc:
        available = ", ".join(sorted(_STRATEGIES))
        raise ValueError(f"Unknown text-line segmentation strategy '{strategy_name}'. Available: {available}") from exc


def list_text_line_segmentation_strategies() -> tuple[str, ...]:
    return tuple(sorted(_STRATEGIES))


def apply_text_line_segmentation_strategy(
    page_image_path: Path,
    heatmap_path: Path,
    source_pagexml_path: Path,
    output_pagexml_path: Path,
    strategy_name: str,
    strategy_config: Mapping[str, object] | None = None,
    metadata_path: Path | None = None,
) -> TextLineSegmentationResult:
    strategy = get_text_line_segmentation_strategy(strategy_name)
    request = TextLineSegmentationRequest(
        page_image_path=Path(page_image_path),
        heatmap_path=Path(heatmap_path),
        source_pagexml_path=Path(source_pagexml_path),
        output_pagexml_path=Path(output_pagexml_path),
        strategy_name=strategy_name,
        strategy_config=dict(strategy_config or {}),
        metadata_path=Path(metadata_path) if metadata_path is not None else None,
    )
    return strategy.apply(request)
