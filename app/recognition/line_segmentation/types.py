from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Protocol


@dataclass(frozen=True)
class TextLineSegmentationRequest:
    page_image_path: Path
    heatmap_path: Path
    source_pagexml_path: Path
    output_pagexml_path: Path
    strategy_name: str
    strategy_config: Mapping[str, object] = field(default_factory=dict)
    metadata_path: Path | None = None


@dataclass
class TextLineSegmentationResult:
    strategy_name: str
    source_pagexml_path: str
    output_pagexml_path: str
    page_image_path: str
    heatmap_path: str
    metadata_path: str | None
    line_count: int
    prepared_line_count: int
    line_metadata: list[dict]
    geometry_summary: dict


class TextLineSegmentationStrategy(Protocol):
    name: str

    def apply(self, request: TextLineSegmentationRequest) -> TextLineSegmentationResult:
        ...


StrategyFactory = Callable[[], TextLineSegmentationStrategy]
