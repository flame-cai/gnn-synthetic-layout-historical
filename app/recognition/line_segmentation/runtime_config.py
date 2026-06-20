from __future__ import annotations

from typing import Mapping

from .legacy_axis_bound import DEFAULT_LEGACY_AXIS_BOUND_CONFIG
from .local_polygons import DEFAULT_LOCAL_POLYGON_CONFIG
from .local_polygons_stable_unwrap import DEFAULT_LOCAL_POLYGON_CONFIG as DEFAULT_STABLE_UNWRAP_CONFIG
from .local_tangent_band import DEFAULT_LOCAL_TANGENT_BAND_CONFIG


PRODUCTION_STRATEGY_RUNTIME_CONFIGS: dict[str, dict] = {
    "legacy_axis_bound_v1": dict(DEFAULT_LEGACY_AXIS_BOUND_CONFIG),
    "local_tangent_band_v1": dict(DEFAULT_LOCAL_TANGENT_BAND_CONFIG),
    "local_polygons_v1": {
        **dict(DEFAULT_LOCAL_POLYGON_CONFIG),
        "BINARIZE_THRESHOLD": 0.45,
    },
    "local_polygons_stable_unwrap_v1": {
        **dict(DEFAULT_STABLE_UNWRAP_CONFIG),
        "BINARIZE_THRESHOLD": 0.45,
        "image_fallback_when_no_heatmap_components": True,
        "anchor_window_clip_enabled": True,
    },
}


def get_strategy_runtime_config(
    strategy_name: str,
    *,
    overrides: Mapping[str, object] | None = None,
    include_empty_text_lines: bool | None = None,
) -> dict:
    config = dict(PRODUCTION_STRATEGY_RUNTIME_CONFIGS.get(str(strategy_name), {}))
    for key, value in dict(overrides or {}).items():
        if value is not None:
            config[key] = value
    if include_empty_text_lines is not None:
        config["include_empty_text_lines"] = bool(include_empty_text_lines)
    return config


__all__ = ["PRODUCTION_STRATEGY_RUNTIME_CONFIGS", "get_strategy_runtime_config"]
