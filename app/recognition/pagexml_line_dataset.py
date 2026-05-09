from __future__ import annotations

import json
import math
import re
import shutil
import sys
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import skimage.io as io
from shapely.geometry import Polygon

SRC_GNN_INFERENCE_DIR = Path(__file__).resolve().parents[2] / "src" / "gnn_inference"
if str(SRC_GNN_INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_GNN_INFERENCE_DIR))

from segment_from_point_clusters import gen_bounding_boxes, loadImage, segmentLinesFromPointClusters


PAGE_XML_NAMESPACE = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
PAGE_XML_NS = {"p": PAGE_XML_NAMESPACE}
GEOMETRY_SOURCE_PAGEXML_COORDS = "pagexml_coords"
GEOMETRY_SOURCE_BASELINE_HEATMAP = "baseline_heatmap"
SUPPORTED_GEOMETRY_SOURCES = {GEOMETRY_SOURCE_PAGEXML_COORDS, GEOMETRY_SOURCE_BASELINE_HEATMAP}


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
    app_image_rel_path: str | None = None
    flat_image_rel_path: str | None = None


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
    tree = ET.parse(xml_path)
    root = tree.getroot()
    records = []
    line_fallback_index = 0
    for line in root.findall(".//p:TextLine", PAGE_XML_NS):
        baseline_elem = line.find("./p:Baseline", PAGE_XML_NS)
        if baseline_elem is None or not baseline_elem.get("points"):
            continue
        text_equiv = line.find("./p:TextEquiv", PAGE_XML_NS)
        unicode_elem = text_equiv.find("./p:Unicode", PAGE_XML_NS) if text_equiv is not None else None
        if not _normalize_text(unicode_elem.text if unicode_elem is not None else ""):
            continue
        line_custom = line.get("custom") or f"structure_line_id_{line_fallback_index}"
        line_numeric_id = _parse_numeric_suffix(line_custom, "structure_line_id_", line_fallback_index)
        records.append(
            {
                "line_numeric_id": line_numeric_id,
                "baseline_points": _parse_points(baseline_elem.get("points")),
            }
        )
        line_fallback_index += 1
    return records


def _count_text_lines_with_text_and_baseline(xml_path: Path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    count = 0
    for line in root.findall(".//p:TextLine", PAGE_XML_NS):
        baseline_elem = line.find("./p:Baseline", PAGE_XML_NS)
        if baseline_elem is None or not baseline_elem.get("points"):
            continue
        text_equiv = line.find("./p:TextEquiv", PAGE_XML_NS)
        unicode_elem = text_equiv.find("./p:Unicode", PAGE_XML_NS) if text_equiv is not None else None
        if _normalize_text(unicode_elem.text if unicode_elem is not None else ""):
            count += 1
    return count


def _build_baseline_component_nodes(xml_path: Path, image_path: Path, heatmap_path: Path, binarize_threshold: float):
    baseline_records = _load_pagexml_baseline_records(xml_path)
    if not baseline_records:
        return np.empty((0, 3)), np.empty((0,), dtype=int), {"heatmap_box_count": 0, "assigned_box_count": 0}

    image = loadImage(str(image_path))
    heatmap = loadImage(str(heatmap_path))
    if heatmap.ndim == 3:
        heatmap = heatmap[:, :, 0]

    image_height, image_width = image.shape[:2]
    heatmap_height, heatmap_width = heatmap.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (image_width, image_height), interpolation=cv2.INTER_LINEAR)
    bounding_boxes = gen_bounding_boxes(heatmap_resized, binarize_threshold)

    x_to_heatmap = heatmap_width / image_width
    y_to_heatmap = heatmap_height / image_height
    synthetic_nodes = []
    synthetic_labels = []
    assigned_distances = []

    for x_val, y_val, width, height in bounding_boxes:
        center_x = x_val + (width / 2.0)
        center_y = y_val + (height / 2.0)
        best_record = min(
            (
                (_distance_to_baseline((center_x, center_y), record["baseline_points"]), record)
                for record in baseline_records
            ),
            key=lambda item: item[0],
            default=None,
        )
        if best_record is None:
            continue

        distance, record = best_record
        max_distance = max(20.0, float(height) * 2.5)
        if distance > max_distance:
            continue

        synthetic_nodes.append(
            [
                center_x * x_to_heatmap,
                center_y * y_to_heatmap,
                max(float(width) * x_to_heatmap, float(height) * y_to_heatmap),
            ]
        )
        synthetic_labels.append(int(record["line_numeric_id"]))
        assigned_distances.append(float(distance))

    summary = {
        "heatmap_box_count": len(bounding_boxes),
        "assigned_box_count": len(synthetic_labels),
        "heatmap_box_assignment_rate": (len(synthetic_labels) / len(bounding_boxes)) if bounding_boxes else None,
        "baseline_line_count": len(baseline_records),
        "max_assignment_distance": max(assigned_distances) if assigned_distances else None,
        "mean_assignment_distance": float(np.mean(assigned_distances)) if assigned_distances else None,
    }
    return np.asarray(synthetic_nodes, dtype=float), np.asarray(synthetic_labels, dtype=int), summary


def _generate_polygons_from_baselines(
    xml_path: Path,
    image_path: Path,
    heatmap_path: Path,
    output_root: Path,
    segmentation_args: dict | None = None,
):
    segmentation_args = dict(segmentation_args or {})
    binarize_threshold = float(segmentation_args.get("BINARIZE_THRESHOLD", 0.5098))
    nodes, labels, summary = _build_baseline_component_nodes(xml_path, image_path, heatmap_path, binarize_threshold)

    work_root = output_root / "_baseline_heatmap_geometry"
    if work_root.exists():
        shutil.rmtree(work_root)
    images_dir = work_root / "images_resized"
    heatmaps_dir = work_root / "heatmaps"
    gnn_format_dir = work_root / "layout_analysis_output" / "gnn-format"
    images_dir.mkdir(parents=True, exist_ok=True)
    heatmaps_dir.mkdir(parents=True, exist_ok=True)
    gnn_format_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(image_path, images_dir / f"{xml_path.stem}.jpg")
    shutil.copy(heatmap_path, heatmaps_dir / f"{xml_path.stem}.jpg")
    np.savetxt(gnn_format_dir / f"{xml_path.stem}_inputs_unnormalized.txt", nodes, fmt="%.6f")
    np.savetxt(gnn_format_dir / f"{xml_path.stem}_labels_textline.txt", labels, fmt="%d")

    polygons_by_label = segmentLinesFromPointClusters(
        str(work_root),
        xml_path.stem,
        BINARIZE_THRESHOLD=binarize_threshold,
        BBOX_PAD_V=float(segmentation_args.get("BBOX_PAD_V", 0.7)),
        BBOX_PAD_H=float(segmentation_args.get("BBOX_PAD_H", 0.5)),
        CC_SIZE_THRESHOLD_RATIO=float(segmentation_args.get("CC_SIZE_THRESHOLD_RATIO", 0.4)),
        GNN_PRED_PATH=str(work_root / "layout_analysis_output"),
    )
    return {
        int(label): [[int(point[0]), int(point[1])] for point in points]
        for label, points in polygons_by_label.items()
    }, summary


def load_pagexml_lines(xml_path: str | Path, polygons_by_line_numeric_id: dict[int, list[list[int]]] | None = None):
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
            if not text:
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
):
    xml_path = Path(xml_path)
    image_path = Path(image_path)
    output_root = Path(output_root)
    if geometry_source not in SUPPORTED_GEOMETRY_SOURCES:
        raise ValueError(f"Unsupported PAGE line geometry source: {geometry_source}")

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if geometry_source == GEOMETRY_SOURCE_BASELINE_HEATMAP:
        source_text_line_count = _count_text_lines_with_text_and_baseline(xml_path)
    else:
        _, source_records = load_pagexml_lines(xml_path)
        source_text_line_count = len(source_records)

    polygons_by_line_numeric_id = None
    generation_summary = {}
    if geometry_source == GEOMETRY_SOURCE_BASELINE_HEATMAP:
        if heatmap_path is None:
            raise ValueError("heatmap_path is required when geometry_source='baseline_heatmap'.")
        polygons_by_line_numeric_id, generation_summary = _generate_polygons_from_baselines(
            xml_path,
            image_path,
            Path(heatmap_path),
            output_root,
            segmentation_args=segmentation_args,
        )

    image_filename, records = load_pagexml_lines(xml_path, polygons_by_line_numeric_id=polygons_by_line_numeric_id)
    ordered_records = sort_lines_for_page_level_cer(records)
    processing_image = _load_processing_image(image_path)
    geometry_summary = _build_geometry_summary(
        ordered_records,
        generation_summary,
        geometry_source,
        source_text_line_count,
    )

    image_format_root = output_root / "image-format" / xml_path.stem
    finetune_dataset_root = output_root / "finetune_dataset"
    test_dir = finetune_dataset_root / "test"
    test_dir.mkdir(parents=True, exist_ok=True)

    gt_lines = []
    prepared_records = []

    for index, record in enumerate(ordered_records, start=1):
        raw_crop = _masked_line_crop(processing_image, record.polygon_points)
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
                }
            )
        )

    gt_path = finetune_dataset_root / "gt.txt"
    gt_path.write_text("\n".join(gt_lines) + ("\n" if gt_lines else ""), encoding="utf-8")

    manifest_path = output_root / "manifest.json"
    manifest_payload = {
        "page_id": xml_path.stem,
        "image_filename": image_filename,
        "source_xml_path": str(xml_path.resolve()),
        "source_image_path": str(image_path.resolve()),
        "geometry_source": geometry_source,
        "geometry_summary": geometry_summary,
        "records": [asdict(record) for record in prepared_records],
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    return PreparedPageDataset(
        page_id=xml_path.stem,
        image_filename=image_filename,
        source_xml_path=str(xml_path.resolve()),
        source_image_path=str(image_path.resolve()),
        output_root=str(output_root.resolve()),
        image_format_dir=str(image_format_root.resolve()),
        finetune_dataset_dir=str(finetune_dataset_root.resolve()),
        gt_path=str(gt_path.resolve()),
        manifest_path=str(manifest_path.resolve()),
        records=prepared_records,
        geometry_source=geometry_source,
        geometry_summary=geometry_summary,
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
