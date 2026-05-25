from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .unwrap import should_unwrap_strategy, unwrap_line_crop_for_ocr


LOGGER = logging.getLogger(__name__)
LOCAL_TANGENT_CROP_MODEL = "local_tangent_band"
LOCAL_POLYGON_CROP_MODEL = "local_polygon_unwrap"
LEGACY_DELEGATE_CROP_MODEL = "legacy_axis_bound_delegate"
AXIS_ALIGNED_CROP_MODEL = "axis_aligned_masked_crop"
UNWRAPPED_CROP_MODELS = {LOCAL_TANGENT_CROP_MODEL, LOCAL_POLYGON_CROP_MODEL}


@dataclass(frozen=True)
class OcrCropResult:
    image: np.ndarray
    metadata: dict


def masked_line_crop(processing_image: np.ndarray, polygon_points) -> np.ndarray:
    polygon = np.array(polygon_points, dtype=np.int32)
    x_val, y_val, width, height = cv2.boundingRect(polygon)
    cropped_line_image = processing_image[y_val : y_val + height, x_val : x_val + width]
    page_median_color = int(np.median(processing_image))
    new_img = np.ones(cropped_line_image.shape, dtype=np.uint8) * page_median_color
    mask_polygon = np.zeros(cropped_line_image.shape[:2], dtype=np.uint8)
    polygon_shifted = polygon - [x_val, y_val]
    cv2.drawContours(mask_polygon, [polygon_shifted], -1, 255, -1)
    new_img[mask_polygon == 255] = cropped_line_image[mask_polygon == 255]
    return new_img


def default_line_segmentation_metadata_path(pagexml_path: str | Path) -> Path:
    pagexml_path = Path(pagexml_path)
    return pagexml_path.with_name(f"{pagexml_path.stem}_line_segmentation_metadata.json")


def _safe_metadata_payload(metadata_path: str | Path | None) -> dict[str, Any]:
    if metadata_path is None:
        return {}
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        return {}
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as exc:
        LOGGER.warning("Could not read line segmentation metadata %s: %s", metadata_path, exc)
        return {}
    if not isinstance(payload, dict):
        LOGGER.warning("Line segmentation metadata %s is not a JSON object.", metadata_path)
        return {}
    return payload


def load_line_segmentation_strategy_name(metadata_path: str | Path | None) -> str | None:
    payload = _safe_metadata_payload(metadata_path)
    if not payload:
        return None
    strategy_name = payload.get("strategy_name")
    if strategy_name:
        return str(strategy_name)
    geometry_summary = payload.get("geometry_summary")
    if isinstance(geometry_summary, dict) and geometry_summary.get("line_segmentation_strategy_name"):
        return str(geometry_summary["line_segmentation_strategy_name"])
    return None


def load_line_segmentation_metadata_by_numeric_id(metadata_path: str | Path | None) -> dict[int, dict]:
    payload = _safe_metadata_payload(metadata_path)
    if not payload:
        return {}

    strategy_name = load_line_segmentation_strategy_name(metadata_path)
    raw_items = payload.get("line_metadata")
    if raw_items is None:
        raw_items = payload.get("lines")
    if not isinstance(raw_items, list):
        LOGGER.warning("Line segmentation metadata %s does not contain a line_metadata list.", metadata_path)
        return {}

    metadata_by_numeric_id: dict[int, dict] = {}
    for item in raw_items:
        if not isinstance(item, dict):
            LOGGER.warning("Skipping malformed line segmentation metadata item in %s.", metadata_path)
            continue
        try:
            line_numeric_id = int(item["line_numeric_id"])
        except Exception:
            LOGGER.warning("Skipping line segmentation metadata item without integer line_numeric_id in %s.", metadata_path)
            continue
        line_payload = dict(item)
        if strategy_name and "line_segmentation_strategy_name" not in line_payload:
            line_payload["line_segmentation_strategy_name"] = strategy_name
        metadata_by_numeric_id[line_numeric_id] = line_payload
    return metadata_by_numeric_id


def _record_value(record: Any, key: str, default=None):
    if isinstance(record, dict):
        return record.get(key, default)
    return getattr(record, key, default)


def _fallback_crop_metadata(
    *,
    record: Any,
    strategy_name: str | None,
    strategy_line_metadata: dict | None,
    fallback_reason: str | None,
) -> dict:
    line_numeric_id = _record_value(record, "line_numeric_id")
    return {
        "crop_model": AXIS_ALIGNED_CROP_MODEL,
        "crop_source": "pagexml_coords",
        "line_segmentation_strategy_name": strategy_name,
        "line_numeric_id": int(line_numeric_id) if line_numeric_id is not None else None,
        "used_unwrap": False,
        "fallback_reason": fallback_reason,
        "strategy_line_metadata": dict(strategy_line_metadata or {}),
    }


def _normalise_line_metadata(strategy_line_metadata: Any) -> tuple[dict, str | None]:
    if strategy_line_metadata is None:
        return {}, "missing_metadata"
    if not isinstance(strategy_line_metadata, dict):
        return {}, "malformed_line_metadata"
    return dict(strategy_line_metadata), None


def crop_line_record_for_ocr(
    processing_image: np.ndarray,
    record: Any,
    strategy_name: str | None = None,
    strategy_line_metadata: dict | None = None,
    crop_config: dict | None = None,
) -> OcrCropResult:
    line_metadata, metadata_fallback = _normalise_line_metadata(strategy_line_metadata)
    effective_strategy_name = strategy_name or line_metadata.get("line_segmentation_strategy_name")
    crop_model = line_metadata.get("crop_model")

    should_unwrap_record = (
        should_unwrap_strategy(effective_strategy_name)
        and crop_model in UNWRAPPED_CROP_MODELS
    )
    if should_unwrap_record:
        try:
            unwrap_config = dict(crop_config or {})
            if crop_model == LOCAL_POLYGON_CROP_MODEL:
                for key in ("local_s_min", "local_s_max", "local_n_min", "local_n_max"):
                    if key in line_metadata:
                        unwrap_config.setdefault(key, line_metadata[key])
            reading_annotation = line_metadata.get("reading_direction_annotation")
            if isinstance(reading_annotation, dict):
                if reading_annotation.get("reading_direction") is not None:
                    unwrap_config.setdefault("reading_direction", reading_annotation.get("reading_direction"))
                if reading_annotation.get("cut_midpoint") is not None:
                    unwrap_config.setdefault("reading_cut_point", reading_annotation.get("cut_midpoint"))
            crop_result = unwrap_line_crop_for_ocr(
                processing_image,
                _record_value(record, "polygon_points") or [],
                _record_value(record, "baseline_points") or [],
                text=_record_value(record, "text", "") or "",
                unwrap_config=unwrap_config,
            )
            if crop_result.metadata.get("fallback_reason"):
                fallback_reason = f"unwrap_{crop_result.metadata['fallback_reason']}"
            else:
                metadata = {
                    **crop_result.metadata,
                    "crop_model": crop_model,
                    "crop_source": "pagexml_coords_and_baseline",
                    "line_segmentation_strategy_name": effective_strategy_name,
                    "line_numeric_id": int(_record_value(record, "line_numeric_id")),
                    "used_unwrap": True,
                    "fallback_reason": crop_result.metadata.get("fallback_reason"),
                    "strategy_line_metadata": line_metadata,
                }
                return OcrCropResult(image=crop_result.image, metadata=metadata)
        except Exception as exc:
            LOGGER.warning(
                "Falling back to masked OCR crop after baseline-local unwrap failed line=%s strategy=%s crop_model=%s: %s",
                _record_value(record, "line_numeric_id"),
                effective_strategy_name,
                crop_model,
                exc,
            )
            fallback_reason = "unwrap_failed"
    elif crop_model and crop_model not in {LEGACY_DELEGATE_CROP_MODEL, *UNWRAPPED_CROP_MODELS}:
        LOGGER.warning(
            "Unsupported OCR crop model '%s' for line=%s strategy=%s; using masked PAGE Coords crop.",
            crop_model,
            _record_value(record, "line_numeric_id"),
            effective_strategy_name,
        )
        fallback_reason = f"unsupported_crop_model:{crop_model}"
    elif metadata_fallback:
        fallback_reason = metadata_fallback
    elif should_unwrap_strategy(effective_strategy_name) and crop_model not in UNWRAPPED_CROP_MODELS:
        fallback_reason = f"non_unwrapped_crop_model:{crop_model or 'missing'}"
    else:
        fallback_reason = None

    return OcrCropResult(
        image=masked_line_crop(processing_image, _record_value(record, "polygon_points") or []),
        metadata=_fallback_crop_metadata(
            record=record,
            strategy_name=effective_strategy_name,
            strategy_line_metadata=line_metadata,
            fallback_reason=fallback_reason,
        ),
    )
