from __future__ import annotations

import re
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path


PAGE_XML_NAMESPACE = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
PAGE_XML_NS = {"p": PAGE_XML_NAMESPACE}


def normalize_text(text: str | None) -> str:
    if text is None:
        return ""
    return unicodedata.normalize("NFC", text).strip()


def local_name(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def element_namespace(tag: str) -> str:
    return tag.split("}", 1)[0].strip("{") if tag.startswith("{") else ""


def qualified(tag: str, namespace: str) -> str:
    return f"{{{namespace}}}{tag}" if namespace else tag


def find_child(element: ET.Element, name: str) -> ET.Element | None:
    for child in element:
        if local_name(child.tag) == name:
            return child
    return None


def find_children(element: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in element if local_name(child.tag) == name]


def iter_descendants(element: ET.Element, name: str):
    for descendant in element.iter():
        if local_name(descendant.tag) == name:
            yield descendant


def parse_points(points_str: str | None) -> list[list[int]]:
    points = []
    for point in (points_str or "").strip().split():
        x_val, y_val = point.split(",")
        points.append([int(round(float(x_val))), int(round(float(y_val)))])
    return points


def format_points(points: list[list[int]] | tuple[tuple[int, int], ...]) -> str:
    return " ".join(f"{int(point[0])},{int(point[1])}" for point in points)


def parse_numeric_suffix(value: str | None, prefix: str, fallback: int) -> int:
    if value:
        match = re.search(rf"{re.escape(prefix)}(\d+)", value)
        if match:
            return int(match.group(1))
        digits = re.findall(r"\d+", value)
        if digits:
            return int(digits[-1])
    return fallback


def page_element(root: ET.Element, xml_path: Path) -> ET.Element:
    for element in root.iter():
        if local_name(element.tag) == "Page":
            return element
    raise ValueError(f"No Page element found in {xml_path}")


def textline_text(line: ET.Element) -> str:
    text_equiv = find_child(line, "TextEquiv")
    unicode_elem = find_child(text_equiv, "Unicode") if text_equiv is not None else None
    return normalize_text(unicode_elem.text if unicode_elem is not None else "")


def load_baseline_records(xml_path: str | Path, include_empty_text_lines: bool = False) -> list[dict]:
    xml_path = Path(xml_path)
    root = ET.parse(xml_path).getroot()
    records = []
    line_fallback_index = 0
    for line in iter_descendants(root, "TextLine"):
        baseline_elem = find_child(line, "Baseline")
        if baseline_elem is None or not baseline_elem.get("points"):
            continue
        text = textline_text(line)
        if not include_empty_text_lines and not text:
            continue
        line_id = line.get("id", f"line_{line_fallback_index}")
        line_custom = line.get("custom") or f"structure_line_id_{line_fallback_index}"
        line_numeric_id = parse_numeric_suffix(line_custom, "structure_line_id_", line_fallback_index)
        records.append(
            {
                "line_id": line_id,
                "line_custom": line_custom,
                "line_numeric_id": line_numeric_id,
                "baseline_points": parse_points(baseline_elem.get("points")),
                "text": text,
            }
        )
        line_fallback_index += 1
    return records


def count_text_lines_with_text_and_baseline(xml_path: str | Path) -> int:
    return len(load_baseline_records(xml_path, include_empty_text_lines=False))


def remove_textline_coords(root: ET.Element) -> None:
    for line in iter_descendants(root, "TextLine"):
        for coords in find_children(line, "Coords"):
            line.remove(coords)


def set_textline_coords_by_numeric_id(
    root: ET.Element,
    polygons_by_line_numeric_id: dict[int, list[list[int]]],
) -> list[dict]:
    metadata = []
    line_fallback_index = 0
    for line in iter_descendants(root, "TextLine"):
        line_custom = line.get("custom") or f"structure_line_id_{line_fallback_index}"
        line_numeric_id = parse_numeric_suffix(line_custom, "structure_line_id_", line_fallback_index)
        baseline_elem = find_child(line, "Baseline")
        baseline_points = (
            parse_points(baseline_elem.get("points"))
            if baseline_elem is not None and baseline_elem.get("points")
            else []
        )
        coords_points = polygons_by_line_numeric_id.get(line_numeric_id)
        if coords_points:
            namespace = element_namespace(line.tag)
            coords_elem = ET.Element(qualified("Coords", namespace), points=format_points(coords_points))
            baseline_index = list(line).index(baseline_elem) if baseline_elem is not None else 0
            line.insert(baseline_index, coords_elem)
        metadata.append(
            {
                "line_id": line.get("id", f"line_{line_fallback_index}"),
                "line_custom": line_custom,
                "line_numeric_id": line_numeric_id,
                "baseline_points": baseline_points,
                "coords_points": coords_points or [],
            }
        )
        line_fallback_index += 1
    return metadata
