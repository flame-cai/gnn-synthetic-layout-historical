from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from .geometry import PAGE_XML_NAMESPACE, PAGE_XML_NS, format_points


def rewrite_textline_coords(
    source_xml_path: str | Path,
    output_xml_path: str | Path,
    coords_by_line_custom: dict[str, list[tuple[float, float]]],
) -> Path:
    source_xml_path = Path(source_xml_path)
    output_xml_path = Path(output_xml_path)
    output_xml_path.parent.mkdir(parents=True, exist_ok=True)

    if not coords_by_line_custom:
        shutil.copy2(source_xml_path, output_xml_path)
        return output_xml_path

    tree = ET.parse(source_xml_path)
    root = tree.getroot()
    ET.register_namespace("", PAGE_XML_NAMESPACE)

    for textline in root.findall(".//p:TextLine", PAGE_XML_NS):
        line_custom = textline.get("custom", "")
        if line_custom not in coords_by_line_custom:
            continue
        coords_elem = textline.find("./p:Coords", PAGE_XML_NS)
        if coords_elem is None:
            coords_elem = ET.SubElement(textline, f"{{{PAGE_XML_NAMESPACE}}}Coords")
        coords_elem.set("points", format_points(coords_by_line_custom[line_custom]))

    if hasattr(ET, "indent"):
        ET.indent(tree, space="\t", level=0)
    tree.write(output_xml_path, encoding="utf-8", xml_declaration=True)
    return output_xml_path

