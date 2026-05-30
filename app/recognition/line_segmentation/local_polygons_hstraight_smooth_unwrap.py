from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .local_polygons import LOCAL_POLYGON_CROP_MODEL, LocalPolygonsStrategy
from .ocr_crops import LOCAL_POLYGON_HORIZONTAL_STRAIGHT_FIT_CROP_MODEL
from .types import TextLineSegmentationRequest, TextLineSegmentationResult


class LocalPolygonsHorizontalStraightSmoothUnwrapStrategy:
    name = "local_polygons_hstraight_smooth_unwrap_v1"
    research_role_independent = False
    production_role_independent = False
    geometry_delegate_strategy_name = LocalPolygonsStrategy.name

    def __init__(self) -> None:
        self._delegate = LocalPolygonsStrategy()

    def apply(self, request: TextLineSegmentationRequest) -> TextLineSegmentationResult:
        result = self._delegate.apply(request)
        result.strategy_name = self.name

        crop_model_counts: dict[str, int] = {}
        for item in result.line_metadata:
            item["line_segmentation_strategy_name"] = self.name
            item["geometry_delegate_strategy_name"] = self.geometry_delegate_strategy_name
            if (
                item.get("crop_model") == LOCAL_POLYGON_CROP_MODEL
                and item.get("line_kind") == "horizontal_straight"
            ):
                item["delegated_crop_model"] = LOCAL_POLYGON_CROP_MODEL
                item["crop_model"] = LOCAL_POLYGON_HORIZONTAL_STRAIGHT_FIT_CROP_MODEL
                item["crop_ablation_model"] = "horizontal_straight_fit_tangent"
            crop_model = str(item.get("crop_model") or "")
            if crop_model:
                crop_model_counts[crop_model] = crop_model_counts.get(crop_model, 0) + 1

        result.geometry_summary = {
            **dict(result.geometry_summary),
            "line_segmentation_strategy_name": self.name,
            "geometry_delegate_strategy_name": self.geometry_delegate_strategy_name,
            "crop_only_ablation": True,
            "crop_ablation_model": "horizontal_straight_fit_tangent",
            "crop_model_counts": crop_model_counts,
        }

        if result.metadata_path:
            metadata_path = Path(result.metadata_path)
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            metadata_path.write_text(
                json.dumps(asdict(result), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        return result


__all__ = ["LocalPolygonsHorizontalStraightSmoothUnwrapStrategy"]
