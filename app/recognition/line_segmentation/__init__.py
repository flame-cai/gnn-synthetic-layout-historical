from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .strategy_config import (
    get_benchmark_strategy_name,
    get_proposed_strategy_name,
    get_production_strategy_name,
    get_strategy_role_config,
)
from .types import TextLineSegmentationRequest, TextLineSegmentationResult, TextLineSegmentationStrategy

if TYPE_CHECKING:
    from .legacy_axis_bound import DEFAULT_LEGACY_AXIS_BOUND_CONFIG, LegacyAxisBoundStrategy
    from .local_polygons import DEFAULT_LOCAL_POLYGON_CONFIG, LocalPolygonsStrategy
    from .local_polygons_hstraight_smooth_unwrap import LocalPolygonsHorizontalStraightSmoothUnwrapStrategy
    from .local_polygons_stable_unwrap import LocalPolygonsStableUnwrapStrategy
    from .local_tangent_band import DEFAULT_LOCAL_TANGENT_BAND_CONFIG, LocalTangentBandStrategy


def apply_text_line_segmentation_strategy(*args: Any, **kwargs: Any):
    from .registry import apply_text_line_segmentation_strategy as _apply

    return _apply(*args, **kwargs)


def get_text_line_segmentation_strategy(*args: Any, **kwargs: Any):
    from .registry import get_text_line_segmentation_strategy as _get

    return _get(*args, **kwargs)


def list_text_line_segmentation_strategies(*args: Any, **kwargs: Any):
    from .registry import list_text_line_segmentation_strategies as _list

    return _list(*args, **kwargs)


def __getattr__(name: str):
    if name in {"DEFAULT_LEGACY_AXIS_BOUND_CONFIG", "LegacyAxisBoundStrategy"}:
        from .legacy_axis_bound import DEFAULT_LEGACY_AXIS_BOUND_CONFIG, LegacyAxisBoundStrategy

        return {
            "DEFAULT_LEGACY_AXIS_BOUND_CONFIG": DEFAULT_LEGACY_AXIS_BOUND_CONFIG,
            "LegacyAxisBoundStrategy": LegacyAxisBoundStrategy,
        }[name]
    if name in {"DEFAULT_LOCAL_TANGENT_BAND_CONFIG", "LocalTangentBandStrategy"}:
        from .local_tangent_band import DEFAULT_LOCAL_TANGENT_BAND_CONFIG, LocalTangentBandStrategy

        return {
            "DEFAULT_LOCAL_TANGENT_BAND_CONFIG": DEFAULT_LOCAL_TANGENT_BAND_CONFIG,
            "LocalTangentBandStrategy": LocalTangentBandStrategy,
        }[name]
    if name in {"DEFAULT_LOCAL_POLYGON_CONFIG", "LocalPolygonsStrategy"}:
        from .local_polygons import DEFAULT_LOCAL_POLYGON_CONFIG, LocalPolygonsStrategy

        return {
            "DEFAULT_LOCAL_POLYGON_CONFIG": DEFAULT_LOCAL_POLYGON_CONFIG,
            "LocalPolygonsStrategy": LocalPolygonsStrategy,
        }[name]
    if name == "LocalPolygonsHorizontalStraightSmoothUnwrapStrategy":
        from .local_polygons_hstraight_smooth_unwrap import LocalPolygonsHorizontalStraightSmoothUnwrapStrategy

        return LocalPolygonsHorizontalStraightSmoothUnwrapStrategy
    if name == "LocalPolygonsStableUnwrapStrategy":
        from .local_polygons_stable_unwrap import LocalPolygonsStableUnwrapStrategy

        return LocalPolygonsStableUnwrapStrategy
    raise AttributeError(name)


__all__ = [
    "DEFAULT_LEGACY_AXIS_BOUND_CONFIG",
    "DEFAULT_LOCAL_POLYGON_CONFIG",
    "DEFAULT_LOCAL_TANGENT_BAND_CONFIG",
    "LegacyAxisBoundStrategy",
    "LocalPolygonsStrategy",
    "LocalPolygonsHorizontalStraightSmoothUnwrapStrategy",
    "LocalPolygonsStableUnwrapStrategy",
    "LocalTangentBandStrategy",
    "TextLineSegmentationRequest",
    "TextLineSegmentationResult",
    "TextLineSegmentationStrategy",
    "apply_text_line_segmentation_strategy",
    "get_benchmark_strategy_name",
    "get_proposed_strategy_name",
    "get_production_strategy_name",
    "get_strategy_role_config",
    "get_text_line_segmentation_strategy",
    "list_text_line_segmentation_strategies",
]
