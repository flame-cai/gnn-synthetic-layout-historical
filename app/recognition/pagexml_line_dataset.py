from __future__ import annotations

import json
import re
import shutil
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import skimage.io as io
from shapely.geometry import Polygon


PAGE_XML_NAMESPACE = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
PAGE_XML_NS = {"p": PAGE_XML_NAMESPACE}


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


def load_pagexml_lines(xml_path: str | Path):
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
            coords_elem = line.find("./p:Coords", PAGE_XML_NS)
            if coords_elem is None or not coords_elem.get("points"):
                continue

            text_equiv = line.find("./p:TextEquiv", PAGE_XML_NS)
            unicode_elem = text_equiv.find("./p:Unicode", PAGE_XML_NS) if text_equiv is not None else None
            text = _normalize_text(unicode_elem.text if unicode_elem is not None else "")
            if not text:
                continue

            polygon_points = _parse_polygon(coords_elem.get("points"))
            polygon = _polygon_for_metrics(polygon_points)
            centroid = polygon.centroid
            min_x, _, _, _ = polygon.bounds

            line_id = line.get("id", f"{region_id}_line_{line_fallback_index}")
            line_custom = line.get("custom") or f"structure_line_id_{line_fallback_index}"
            line_numeric_id = _parse_numeric_suffix(line_custom, "structure_line_id_", line_fallback_index)

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
                    y_center=float(centroid.y),
                    x_min=float(min_x),
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


def prepare_page_line_dataset(xml_path: str | Path, image_path: str | Path, output_root: str | Path):
    xml_path = Path(xml_path)
    image_path = Path(image_path)
    output_root = Path(output_root)

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    image_filename, records = load_pagexml_lines(xml_path)
    ordered_records = sort_lines_for_page_level_cer(records)
    processing_image = _load_processing_image(image_path)

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
