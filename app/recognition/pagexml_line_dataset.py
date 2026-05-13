from __future__ import annotations

import json
import logging
import math
import re
import shutil
import sys
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
import skimage.io as io
from shapely.geometry import Polygon

try:
    from .line_segmentation import apply_text_line_segmentation_strategy
    from .line_segmentation.strategy_config import get_production_strategy_name
    from .line_segmentation.legacy_axis_bound import (
        build_legacy_axis_bound_polygons,
        _build_baseline_component_nodes as _strategy_build_baseline_component_nodes,
    )
    from .line_segmentation.unwrap import should_unwrap_strategy, unwrap_line_crop_for_ocr
    from .line_segmentation.pagexml import (
        count_text_lines_with_text_and_baseline as _strategy_count_text_lines_with_text_and_baseline,
        load_baseline_records as _strategy_load_baseline_records,
    )
except ImportError:  # pragma: no cover - script execution fallback
    from line_segmentation import apply_text_line_segmentation_strategy
    from line_segmentation.strategy_config import get_production_strategy_name
    from line_segmentation.legacy_axis_bound import (
        build_legacy_axis_bound_polygons,
        _build_baseline_component_nodes as _strategy_build_baseline_component_nodes,
    )
    from line_segmentation.unwrap import should_unwrap_strategy, unwrap_line_crop_for_ocr
    from line_segmentation.pagexml import (
        count_text_lines_with_text_and_baseline as _strategy_count_text_lines_with_text_and_baseline,
        load_baseline_records as _strategy_load_baseline_records,
    )


PAGE_XML_NAMESPACE = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
PAGE_XML_NS = {"p": PAGE_XML_NAMESPACE}
GEOMETRY_SOURCE_PAGEXML_COORDS = "pagexml_coords"
GEOMETRY_SOURCE_BASELINE_HEATMAP = "baseline_heatmap"
SUPPORTED_GEOMETRY_SOURCES = {GEOMETRY_SOURCE_PAGEXML_COORDS, GEOMETRY_SOURCE_BASELINE_HEATMAP}
DEFAULT_LINE_SEGMENTATION_STRATEGY = get_production_strategy_name()
LOGGER = logging.getLogger(__name__)


@dataclass
class PreparedLineRecord:
    page_id: str
    region_id: str
    region_custom: str
    line_id: str
    line_custom: str
    line_numeric_id: int
    text: str
    polygon_points: list[list[int]]
    y_center: float
    x_min: float
    baseline_points: list[list[int]] = field(default_factory=list)
    app_image_rel_path: str | None = None
    flat_image_rel_path: str | None = None
    crop_metadata: dict | None = None


@dataclass
class PreparedPageDataset:
    page_id: str
    image_filename: str
    source_xml_path: str
    source_image_path: str
    output_root: str
    image_format_dir: str
    finetune_dataset_dir: str
    gt_path: str
    manifest_path: str
    records: list[PreparedLineRecord]
    geometry_source: str = GEOMETRY_SOURCE_PAGEXML_COORDS
    geometry_summary: dict | None = None
    line_segmentation_strategy_name: str | None = None
    line_segmentation_metadata_path: str | None = None


def _normalize_text(text):
    if text is None:
        return ""
    return unicodedata.normalize("NFC", text).strip()


def _parse_polygon(points_str):
    points = []
    for point in points_str.strip().split():
        x_val, y_val = point.split(",")
        points.append([int(x_val), int(y_val)])
    return points


def _parse_points(points_str):
    return _parse_polygon(points_str)


def _line_order_key_from_points(points):
    if not points:
        return (float("inf"), float("inf"))
    y_values = [point[1] for point in points]
    x_values = [point[0] for point in points]
    return ((min(y_values) + max(y_values)) / 2.0, min(x_values))


def _polygon_for_metrics(points):
    polygon = Polygon(points)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    return polygon


def _parse_numeric_suffix(value, prefix, fallback):
    if value:
        match = re.search(rf"{re.escape(prefix)}(\d+)", value)
        if match:
            return int(match.group(1))
        digits = re.findall(r"\d+", value)
        if digits:
            return int(digits[-1])
    return fallback


def _point_segment_distance(point, start, end):
    px_val, py_val = point
    ax_val, ay_val = start
    bx_val, by_val = end
    vx_val = bx_val - ax_val
    vy_val = by_val - ay_val
    wx_val = px_val - ax_val
    wy_val = py_val - ay_val
    denom = vx_val * vx_val + vy_val * vy_val
    if denom == 0:
        return math.hypot(px_val - ax_val, py_val - ay_val)
    ratio = max(0.0, min(1.0, (wx_val * vx_val + wy_val * vy_val) / denom))
    nearest_x = ax_val + ratio * vx_val
    nearest_y = ay_val + ratio * vy_val
    return math.hypot(px_val - nearest_x, py_val - nearest_y)


def _distance_to_baseline(point, baseline_points):
    if not baseline_points:
        return float("inf")
    if len(baseline_points) == 1:
        return math.hypot(point[0] - baseline_points[0][0], point[1] - baseline_points[0][1])
    return min(
        _point_segment_distance(point, baseline_points[index], baseline_points[index + 1])
        for index in range(len(baseline_points) - 1)
    )


def _load_pagexml_baseline_records(xml_path: Path):
    return [
        {
            "line_numeric_id": record["line_numeric_id"],
            "baseline_points": record["baseline_points"],
        }
        for record in _strategy_load_baseline_records(xml_path, include_empty_text_lines=False)
    ]


def _count_text_lines_with_text_and_baseline(xml_path: Path):
    return _strategy_count_text_lines_with_text_and_baseline(xml_path)


def _build_baseline_component_nodes(xml_path: Path, image_path: Path, heatmap_path: Path, binarize_threshold: float):
    return _strategy_build_baseline_component_nodes(
        xml_path,
        image_path,
        heatmap_path,
        binarize_threshold,
        include_empty_text_lines=False,
    )


def _generate_polygons_from_baselines(
    xml_path: Path,
    image_path: Path,
    heatmap_path: Path,
    output_root: Path,
    segmentation_args: dict | None = None,
):
    return build_legacy_axis_bound_polygons(
        xml_path,
        image_path,
        heatmap_path,
        output_root,
        segmentation_args=segmentation_args,
    )


def load_pagexml_lines(
    xml_path: str | Path,
    polygons_by_line_numeric_id: dict[int, list[list[int]]] | None = None,
    include_empty_text_lines: bool = False,
):
    xml_path = Path(xml_path)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    page_elem = root.find(".//p:Page", PAGE_XML_NS)
    if page_elem is None:
        raise ValueError(f"No Page element found in {xml_path}")

    image_filename = page_elem.get("imageFilename", f"{xml_path.stem}.jpg")
    records = []
    line_fallback_index = 0

    for region_index, region in enumerate(root.findall(".//p:TextRegion", PAGE_XML_NS)):
        region_id = region.get("id", f"region_{region_index}")
        region_custom = region.get("custom") or f"textbox_label_{region_index}"

        for line in region.findall("./p:TextLine", PAGE_XML_NS):
            text_equiv = line.find("./p:TextEquiv", PAGE_XML_NS)
            unicode_elem = text_equiv.find("./p:Unicode", PAGE_XML_NS) if text_equiv is not None else None
            text = _normalize_text(unicode_elem.text if unicode_elem is not None else "")
            if not text and not include_empty_text_lines:
                continue

            line_id = line.get("id", f"{region_id}_line_{line_fallback_index}")
            line_custom = line.get("custom") or f"structure_line_id_{line_fallback_index}"
            line_numeric_id = _parse_numeric_suffix(line_custom, "structure_line_id_", line_fallback_index)
            baseline_elem = line.find("./p:Baseline", PAGE_XML_NS)
            baseline_points = (
                _parse_points(baseline_elem.get("points"))
                if baseline_elem is not None and baseline_elem.get("points")
                else []
            )

            if polygons_by_line_numeric_id is not None:
                polygon_points = polygons_by_line_numeric_id.get(line_numeric_id)
                if not polygon_points:
                    continue
            else:
                coords_elem = line.find("./p:Coords", PAGE_XML_NS)
                if coords_elem is None or not coords_elem.get("points"):
                    continue
                polygon_points = _parse_polygon(coords_elem.get("points"))

            polygon = _polygon_for_metrics(polygon_points)
            centroid = polygon.centroid
            min_x, _, _, _ = polygon.bounds
            order_y, order_x = (
                _line_order_key_from_points(baseline_points)
                if baseline_points
                else (float(centroid.y), float(min_x))
            )

            records.append(
                PreparedLineRecord(
                    page_id=xml_path.stem,
                    region_id=region_id,
                    region_custom=region_custom,
                    line_id=line_id,
                    line_custom=line_custom,
                    line_numeric_id=line_numeric_id,
                    text=text,
                    polygon_points=polygon_points,
                    y_center=float(order_y),
                    x_min=float(order_x),
                    baseline_points=baseline_points,
                )
            )
            line_fallback_index += 1

    return image_filename, records


def sort_lines_for_page_level_cer(records: list[PreparedLineRecord]):
    return sorted(records, key=lambda item: (item.y_center, item.x_min))


def _load_processing_image(image_path: str | Path):
    image = io.imread(str(image_path))
    if getattr(image, "shape", (0,))[0] == 2:
        image = image[0]
    if len(image.shape) == 2:
        return image.astype(np.uint8)
    if image.shape[2] == 4:
        image = image[:, :, :3]
    image = np.array(image)
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def _masked_line_crop(processing_image, polygon_points):
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


def _encode_like_app_jpg(image):
    success, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not success:
        raise ValueError("Failed to JPEG encode line image.")
    jpg_bytes = bytes(buffer)
    decoded = cv2.imdecode(np.frombuffer(jpg_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    return jpg_bytes, decoded


def _build_geometry_summary(records, generation_summary, geometry_source, source_text_line_count):
    summary = {
        "geometry_source": geometry_source,
        "prepared_line_count": len(records),
        "source_text_line_count": source_text_line_count,
        "source_line_coverage": (len(records) / source_text_line_count) if source_text_line_count else None,
    }
    summary.update(generation_summary or {})
    return summary


def prepare_page_line_dataset(
    xml_path: str | Path,
    image_path: str | Path,
    output_root: str | Path,
    heatmap_path: str | Path | None = None,
    geometry_source: str = GEOMETRY_SOURCE_PAGEXML_COORDS,
    segmentation_args: dict | None = None,
    line_segmentation_strategy_name: str | None = None,
    strategy_name: str | None = None,
):
    xml_path = Path(xml_path)
    image_path = Path(image_path)
    output_root = Path(output_root)
    requested_strategy_name = strategy_name or line_segmentation_strategy_name
    if geometry_source not in SUPPORTED_GEOMETRY_SOURCES:
        raise ValueError(f"Unsupported PAGE line geometry source: {geometry_source}")
    effective_geometry_source = (
        GEOMETRY_SOURCE_BASELINE_HEATMAP if requested_strategy_name is not None else geometry_source
    )

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    effective_strategy_name = requested_strategy_name
    if effective_geometry_source == GEOMETRY_SOURCE_BASELINE_HEATMAP and effective_strategy_name is None:
        effective_strategy_name = DEFAULT_LINE_SEGMENTATION_STRATEGY
        LOGGER.info(
            "Using production text-line segmentation strategy for baseline_heatmap preparation strategy=%s page=%s",
            effective_strategy_name,
            xml_path.stem,
        )

    if effective_strategy_name is not None:
        source_text_line_count = _count_text_lines_with_text_and_baseline(xml_path)
    else:
        _, source_records = load_pagexml_lines(xml_path)
        source_text_line_count = len(source_records)

    effective_xml_path = xml_path
    generation_summary = {}
    strategy_metadata_path = None
    strategy_line_metadata_by_numeric_id = {}
    if effective_strategy_name is not None:
        if heatmap_path is None:
            raise ValueError("heatmap_path is required when line segmentation strategy is requested.")
        strategy_root = output_root / "_line_segmentation" / effective_strategy_name
        effective_xml_path = strategy_root / f"{xml_path.stem}.xml"
        strategy_metadata_path = strategy_root / f"{xml_path.stem}_metadata.json"
        strategy_result = apply_text_line_segmentation_strategy(
            page_image_path=image_path,
            heatmap_path=Path(heatmap_path),
            source_pagexml_path=xml_path,
            output_pagexml_path=effective_xml_path,
            strategy_name=effective_strategy_name,
            strategy_config=segmentation_args or {},
            metadata_path=strategy_metadata_path,
        )
        generation_summary = dict(strategy_result.geometry_summary)
        strategy_line_metadata_by_numeric_id = {
            int(item["line_numeric_id"]): dict(item)
            for item in strategy_result.line_metadata
        }

    image_filename, records = load_pagexml_lines(effective_xml_path)
    ordered_records = sort_lines_for_page_level_cer(records)
    processing_image = _load_processing_image(image_path)
    if effective_strategy_name is not None:
        geometry_summary = dict(generation_summary)
        geometry_summary["prepared_line_count"] = len(ordered_records)
        geometry_summary["source_text_line_count"] = source_text_line_count
        geometry_summary["source_line_coverage"] = (
            len(ordered_records) / source_text_line_count if source_text_line_count else None
        )
    else:
        geometry_summary = _build_geometry_summary(
            ordered_records,
            generation_summary,
            effective_geometry_source,
            source_text_line_count,
        )

    image_format_root = output_root / "image-format" / xml_path.stem
    finetune_dataset_root = output_root / "finetune_dataset"
    test_dir = finetune_dataset_root / "test"
    test_dir.mkdir(parents=True, exist_ok=True)

    gt_lines = []
    prepared_records = []

    for index, record in enumerate(ordered_records, start=1):
        crop_metadata = None
        strategy_line_metadata = strategy_line_metadata_by_numeric_id.get(int(record.line_numeric_id), {})
        should_unwrap_record = (
            should_unwrap_strategy(effective_strategy_name)
            and strategy_line_metadata.get("crop_model") == "local_tangent_band"
        )
        if should_unwrap_record:
            crop_result = unwrap_line_crop_for_ocr(
                processing_image,
                record.polygon_points,
                record.baseline_points,
                text=record.text,
                unwrap_config=segmentation_args or {},
            )
            raw_crop = crop_result.image
            crop_metadata = {
                **crop_result.metadata,
                "strategy_line_metadata": strategy_line_metadata,
            }
        else:
            raw_crop = _masked_line_crop(processing_image, record.polygon_points)
            if should_unwrap_strategy(effective_strategy_name):
                crop_metadata = {
                    "unwrap_strategy": "axis_aligned_masked_crop",
                    "strategy_line_metadata": strategy_line_metadata,
                }
        jpg_bytes, decoded_jpg = _encode_like_app_jpg(raw_crop)

        app_rel_path = Path("image-format") / record.page_id / record.region_custom / f"line_{record.line_numeric_id}.jpg"
        app_abs_path = output_root / app_rel_path
        app_abs_path.parent.mkdir(parents=True, exist_ok=True)
        app_abs_path.write_bytes(jpg_bytes)

        flat_rel_path = Path("test") / f"word_{index:04d}.png"
        flat_abs_path = finetune_dataset_root / flat_rel_path
        cv2.imwrite(str(flat_abs_path), decoded_jpg)

        gt_lines.append(f"{flat_rel_path.as_posix()}\t{record.text}")
        prepared_records.append(
            PreparedLineRecord(
                **{
                    **asdict(record),
                    "app_image_rel_path": app_rel_path.as_posix(),
                    "flat_image_rel_path": flat_rel_path.as_posix(),
                    "crop_metadata": crop_metadata,
                }
            )
        )

    gt_path = finetune_dataset_root / "gt.txt"
    gt_path.write_text("\n".join(gt_lines) + ("\n" if gt_lines else ""), encoding="utf-8")

    manifest_path = output_root / "manifest.json"
    manifest_payload = {
        "page_id": xml_path.stem,
        "image_filename": image_filename,
        "source_xml_path": str(effective_xml_path.resolve()),
        "original_source_xml_path": str(xml_path.resolve()),
        "source_image_path": str(image_path.resolve()),
        "geometry_source": effective_geometry_source,
        "geometry_summary": geometry_summary,
        "line_segmentation_strategy_name": effective_strategy_name,
        "line_segmentation_metadata_path": str(strategy_metadata_path.resolve()) if strategy_metadata_path else None,
        "records": [asdict(record) for record in prepared_records],
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    return PreparedPageDataset(
        page_id=xml_path.stem,
        image_filename=image_filename,
        source_xml_path=str(effective_xml_path.resolve()),
        source_image_path=str(image_path.resolve()),
        output_root=str(output_root.resolve()),
        image_format_dir=str(image_format_root.resolve()),
        finetune_dataset_dir=str(finetune_dataset_root.resolve()),
        gt_path=str(gt_path.resolve()),
        manifest_path=str(manifest_path.resolve()),
        records=prepared_records,
        geometry_source=effective_geometry_source,
        geometry_summary=geometry_summary,
        line_segmentation_strategy_name=effective_strategy_name,
        line_segmentation_metadata_path=str(strategy_metadata_path.resolve()) if strategy_metadata_path else None,
    )


def load_prepared_page_dataset(manifest_path: str | Path):
    manifest_path = Path(manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_root = manifest_path.parent
    records = [PreparedLineRecord(**record) for record in payload["records"]]
    return PreparedPageDataset(
        page_id=payload["page_id"],
        image_filename=payload["image_filename"],
        source_xml_path=payload["source_xml_path"],
        source_image_path=payload["source_image_path"],
        output_root=str(output_root.resolve()),
        image_format_dir=str((output_root / "image-format" / payload["page_id"]).resolve()),
        finetune_dataset_dir=str((output_root / "finetune_dataset").resolve()),
        gt_path=str((output_root / "finetune_dataset" / "gt.txt").resolve()),
        manifest_path=str(manifest_path.resolve()),
        records=records,
        geometry_source=payload.get("geometry_source", GEOMETRY_SOURCE_PAGEXML_COORDS),
        geometry_summary=payload.get("geometry_summary"),
        line_segmentation_strategy_name=payload.get("line_segmentation_strategy_name"),
        line_segmentation_metadata_path=payload.get("line_segmentation_metadata_path"),
    )


def write_prediction_pagexml(gt_xml_path: str | Path, predictions_by_line_custom, output_path: str | Path):
    gt_xml_path = Path(gt_xml_path)
    output_path = Path(output_path)

    tree = ET.parse(gt_xml_path)
    root = tree.getroot()
    ET.register_namespace("", PAGE_XML_NAMESPACE)

    for textline in root.findall(".//p:TextLine", PAGE_XML_NS):
        for text_equiv in textline.findall("./p:TextEquiv", PAGE_XML_NS):
            textline.remove(text_equiv)

        line_custom = textline.get("custom", "")
        predicted_text = _normalize_text(predictions_by_line_custom.get(line_custom, ""))
        if not predicted_text:
            continue

        text_equiv = ET.SubElement(textline, f"{{{PAGE_XML_NAMESPACE}}}TextEquiv")
        unicode_elem = ET.SubElement(text_equiv, f"{{{PAGE_XML_NAMESPACE}}}Unicode")
        unicode_elem.text = predicted_text

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(ET, "indent"):
        ET.indent(tree, space="\t", level=0)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
