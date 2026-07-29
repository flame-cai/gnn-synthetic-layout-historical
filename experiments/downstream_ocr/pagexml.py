from __future__ import annotations

import copy
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union


PAGE_XML_2013_NAMESPACE = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"


@dataclass(frozen=True)
class TextLine:
    page_id: str
    line_id: str
    points: tuple[tuple[float, float], ...]
    polygon: BaseGeometry
    text: str
    region_id: str = ""


@dataclass(frozen=True)
class PageXmlPage:
    page_id: str
    image_filename: str
    width: int
    height: int
    lines: tuple[TextLine, ...]
    namespace: str | None = None
    source_path: Path | None = None


def local_name(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def tag_namespace(tag: str) -> str | None:
    if tag.startswith("{") and "}" in tag:
        return tag.split("}", 1)[0].strip("{")
    return None


def qualified(tag: str, namespace: str | None) -> str:
    return f"{{{namespace}}}{tag}" if namespace else tag


def iter_children(element: ET.Element, name: str) -> Iterable[ET.Element]:
    for child in list(element):
        if local_name(child.tag) == name:
            yield child


def iter_descendants(element: ET.Element, name: str) -> Iterable[ET.Element]:
    for child in element.iter():
        if local_name(child.tag) == name:
            yield child


def first_child(element: ET.Element, name: str) -> ET.Element | None:
    return next(iter_children(element, name), None)


def parse_points(points_str: str | None) -> tuple[tuple[float, float], ...]:
    points: list[tuple[float, float]] = []
    for raw_point in str(points_str or "").strip().split():
        if "," not in raw_point:
            raise ValueError(f"Invalid PAGE points token: {raw_point!r}")
        raw_x, raw_y = raw_point.split(",", 1)
        points.append((float(raw_x), float(raw_y)))
    return tuple(points)


def format_points(points: Iterable[tuple[float, float]]) -> str:
    formatted = []
    for x_val, y_val in points:
        x_out = int(round(float(x_val)))
        y_out = int(round(float(y_val)))
        formatted.append(f"{x_out},{y_out}")
    return " ".join(formatted)


def _distinct_point_count(points: tuple[tuple[float, float], ...]) -> int:
    return len({(round(x_val, 6), round(y_val, 6)) for x_val, y_val in points})


def iter_polygon_parts(geometry: BaseGeometry) -> Iterable[Polygon]:
    if isinstance(geometry, Polygon):
        yield geometry
    elif isinstance(geometry, MultiPolygon):
        yield from geometry.geoms
    elif isinstance(geometry, GeometryCollection):
        for child in geometry.geoms:
            yield from iter_polygon_parts(child)


def exterior_points_for_xml(geometry: BaseGeometry) -> tuple[tuple[float, float], ...]:
    parts = sorted(iter_polygon_parts(geometry), key=lambda item: item.area, reverse=True)
    if not parts:
        return ()
    return tuple((float(x_val), float(y_val)) for x_val, y_val in parts[0].exterior.coords[:-1])


def _normalise_valid_area_geometry(
    geometry: BaseGeometry,
    *,
    width: int,
    height: int,
) -> BaseGeometry | None:
    if geometry.is_empty:
        return None
    try:
        if not geometry.is_valid:
            geometry = geometry.buffer(0)
    except Exception:
        return None
    if geometry.is_empty:
        return None
    try:
        clipped = geometry.intersection(box(0, 0, float(width), float(height)))
    except Exception:
        return None
    if clipped.is_empty:
        return None
    if not clipped.is_valid:
        try:
            clipped = clipped.buffer(0)
        except Exception:
            return None
    parts = [
        part
        for part in iter_polygon_parts(clipped)
        if part.is_valid and not part.is_empty and part.area > 0
    ]
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    unioned = unary_union(parts)
    if unioned.is_empty or unioned.area <= 0 or not unioned.is_valid:
        return None
    return unioned


def _repair_polygon_with_mask(
    points: tuple[tuple[float, float], ...],
    *,
    width: int,
    height: int,
) -> BaseGeometry | None:
    rounded = np.asarray(
        [
            [
                int(np.clip(round(x_val), 0, max(width - 1, 0))),
                int(np.clip(round(y_val), 0, max(height - 1, 0))),
            ]
            for x_val, y_val in points
        ],
        dtype=np.int32,
    )
    if len({(int(x_val), int(y_val)) for x_val, y_val in rounded}) < 3:
        return None

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.drawContours(mask, [rounded], -1, 1, -1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    repaired_parts: list[BaseGeometry] = []
    for contour in contours:
        contour_points = contour.reshape(-1, 2)
        if len(contour_points) < 3:
            continue
        polygon = Polygon([(float(x_val), float(y_val)) for x_val, y_val in contour_points])
        normalised = _normalise_valid_area_geometry(polygon, width=width, height=height)
        if normalised is not None:
            repaired_parts.append(normalised)
    if not repaired_parts:
        return None
    return _normalise_valid_area_geometry(unary_union(repaired_parts), width=width, height=height)


def validate_polygon(
    points: tuple[tuple[float, float], ...],
    *,
    width: int,
    height: int,
    context: str,
    repair: bool = False,
) -> BaseGeometry:
    if _distinct_point_count(points) < 3:
        raise ValueError(f"{context}: TextLine/Coords must contain at least three distinct points.")
    polygon = Polygon(points)
    strict_geometry = _normalise_valid_area_geometry(polygon, width=width, height=height)
    if polygon.is_valid and strict_geometry is not None:
        return strict_geometry
    if repair:
        repaired = _repair_polygon_with_mask(points, width=width, height=height) or strict_geometry
        if repaired is not None:
            return repaired
    if not polygon.is_valid:
        raise ValueError(f"{context}: invalid polygon geometry.")
    raise ValueError(f"{context}: polygon has no positive area after clipping to the page.")


def _text_equiv_sort_key(indexed: tuple[int, ET.Element]) -> tuple[int, int, int]:
    xml_order, text_equiv = indexed
    raw_index = text_equiv.get("index")
    if raw_index is None:
        return (1, 0, xml_order)
    try:
        return (0, int(raw_index), xml_order)
    except ValueError:
        return (1, 0, xml_order)


def extract_unicode_text(textline: ET.Element) -> str:
    text_equivs = list(iter_children(textline, "TextEquiv"))
    if not text_equivs:
        return ""
    selected = min(enumerate(text_equivs), key=_text_equiv_sort_key)[1]
    unicode_elem = first_child(selected, "Unicode")
    return "" if unicode_elem is None or unicode_elem.text is None else unicode_elem.text


def load_pagexml(
    path: str | Path,
    *,
    strict: bool = True,
    repair_geometry: bool = False,
    allow_empty_geometry: bool = False,
) -> PageXmlPage:
    xml_path = Path(path)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    namespace = tag_namespace(root.tag)
    page_elem = next(iter_descendants(root, "Page"), None)
    if page_elem is None:
        raise ValueError(f"No PAGE Page element found in {xml_path}.")

    try:
        width = int(float(page_elem.get("imageWidth", "")))
        height = int(float(page_elem.get("imageHeight", "")))
    except ValueError as exc:
        raise ValueError(f"Invalid PAGE dimensions in {xml_path}.") from exc
    if width <= 0 or height <= 0:
        raise ValueError(f"PAGE dimensions must be positive in {xml_path}.")

    page_id = xml_path.stem
    image_filename = page_elem.get("imageFilename") or f"{page_id}.jpg"
    lines: list[TextLine] = []
    fallback_index = 0

    for region_index, region in enumerate(iter_descendants(page_elem, "TextRegion")):
        region_id = region.get("id") or f"region_{region_index}"
        for textline in iter_children(region, "TextLine"):
            line_id = textline.get("id") or f"{region_id}_line_{fallback_index}"
            coords_elem = first_child(textline, "Coords")
            if coords_elem is None or not coords_elem.get("points"):
                if allow_empty_geometry:
                    points: tuple[tuple[float, float], ...] = ()
                    polygon: BaseGeometry = GeometryCollection()
                elif strict:
                    raise ValueError(f"{xml_path}:{line_id}: missing TextLine/Coords points.")
                else:
                    continue
            else:
                points = parse_points(coords_elem.get("points"))
                polygon = validate_polygon(
                    points,
                    width=width,
                    height=height,
                    context=f"{xml_path}:{line_id}",
                    repair=repair_geometry,
                )
            lines.append(
                TextLine(
                    page_id=page_id,
                    line_id=line_id,
                    points=exterior_points_for_xml(polygon) if not polygon.is_empty else (),
                    polygon=polygon,
                    text=extract_unicode_text(textline),
                    region_id=region_id,
                )
            )
            fallback_index += 1

    return PageXmlPage(
        page_id=page_id,
        image_filename=image_filename,
        width=width,
        height=height,
        lines=tuple(lines),
        namespace=namespace,
        source_path=xml_path,
    )


def empty_page_like(page: PageXmlPage, *, source_path: Path | None = None) -> PageXmlPage:
    return PageXmlPage(
        page_id=page.page_id,
        image_filename=page.image_filename,
        width=page.width,
        height=page.height,
        lines=(),
        namespace=page.namespace,
        source_path=source_path,
    )


def _copy_page_without_regions(template_root: ET.Element, namespace: str | None) -> ET.ElementTree:
    root = copy.deepcopy(template_root)
    page_elem = next(iter_descendants(root, "Page"), None)
    if page_elem is None:
        raise ValueError("Template PAGE XML has no Page element.")
    for region in list(iter_children(page_elem, "TextRegion")):
        page_elem.remove(region)
    return ET.ElementTree(root)


def write_pagexml(
    page: PageXmlPage,
    output_path: str | Path,
    *,
    lines: Iterable[TextLine] | None = None,
    template_xml_path: str | Path | None = None,
) -> Path:
    output = Path(output_path)
    namespace = page.namespace or PAGE_XML_2013_NAMESPACE
    ET.register_namespace("", namespace)

    if template_xml_path is not None:
        template_root = ET.parse(template_xml_path).getroot()
        tree = _copy_page_without_regions(template_root, namespace)
        root = tree.getroot()
        page_elem = next(iter_descendants(root, "Page"), None)
        assert page_elem is not None
    else:
        root = ET.Element(qualified("PcGts", namespace))
        page_elem = ET.SubElement(root, qualified("Page", namespace))
        tree = ET.ElementTree(root)

    page_elem.set("imageFilename", page.image_filename)
    page_elem.set("imageWidth", str(int(page.width)))
    page_elem.set("imageHeight", str(int(page.height)))

    for region_index, line in enumerate(lines if lines is not None else page.lines):
        region_id = line.region_id or f"region_{region_index}"
        region_elem = ET.SubElement(page_elem, qualified("TextRegion", namespace), {"id": region_id})
        region_elem.set("custom", region_id)
        ET.SubElement(
            region_elem,
            qualified("Coords", namespace),
            {"points": format_points(line.points)},
        )
        line_elem = ET.SubElement(
            region_elem,
            qualified("TextLine", namespace),
            {"id": line.line_id},
        )
        ET.SubElement(
            line_elem,
            qualified("Coords", namespace),
            {"points": format_points(line.points)},
        )
        text_equiv = ET.SubElement(line_elem, qualified("TextEquiv", namespace))
        unicode_elem = ET.SubElement(text_equiv, qualified("Unicode", namespace))
        unicode_elem.text = line.text or ""

    output.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(ET, "indent"):
        ET.indent(tree, space="\t", level=0)
    tree.write(output, encoding="UTF-8", xml_declaration=True)
    return output


def write_text_only_pagexml(
    page: PageXmlPage,
    output_path: str | Path,
    *,
    lines: Iterable[TextLine] | None = None,
    region_id: str = "region_0",
) -> Path:
    """Write text-line transcriptions without inventing PAGE geometry."""
    output = Path(output_path)
    namespace = page.namespace or PAGE_XML_2013_NAMESPACE
    ET.register_namespace("", namespace)

    root = ET.Element(qualified("PcGts", namespace))
    page_elem = ET.SubElement(
        root,
        qualified("Page", namespace),
        {
            "imageFilename": page.image_filename,
            "imageWidth": str(int(page.width)),
            "imageHeight": str(int(page.height)),
        },
    )
    region_elem = ET.SubElement(
        page_elem,
        qualified("TextRegion", namespace),
        {
            "id": region_id,
            "custom": "text_only_no_geometry",
        },
    )
    ET.SubElement(region_elem, qualified("Coords", namespace), {"points": ""})

    for line_index, line in enumerate(lines if lines is not None else page.lines):
        line_elem = ET.SubElement(
            region_elem,
            qualified("TextLine", namespace),
            {"id": line.line_id or f"line_{line_index}"},
        )
        ET.SubElement(line_elem, qualified("Coords", namespace), {"points": ""})
        ET.SubElement(line_elem, qualified("Baseline", namespace), {"points": ""})
        text_equiv = ET.SubElement(
            line_elem,
            qualified("TextEquiv", namespace),
            {"index": "0"},
        )
        unicode_elem = ET.SubElement(text_equiv, qualified("Unicode", namespace))
        unicode_elem.text = line.text or ""

    tree = ET.ElementTree(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(ET, "indent"):
        ET.indent(tree, space="\t", level=0)
    tree.write(output, encoding="UTF-8", xml_declaration=True)
    return output


def extract_structure_line_id(custom_value: str | None) -> str | None:
    match = re.search(r"structure_line_id_([^;\s]+)", str(custom_value or ""))
    return match.group(1) if match else None
