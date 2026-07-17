from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .pagexml import PageXmlPage, TextLine, exterior_points_for_xml, validate_polygon, write_pagexml


VLM_END_TO_END_PROMPT = """
You are an expert Indologist and Paleographer specializing in handwritten Sanskrit manuscripts.
Your Task: Perform a diplomatic transcription (OCR) of the manuscript image and provide text-line geometry.
CRITICAL INSTRUCTIONS:
1. Output Format: Output ONLY raw valid JSON. No Markdown.
2. Coordinates: Coordinates are normalized from 0 to 1000, where [0,0] is top-left and [1000,1000] is bottom-right.
3. Geometry: For every visual text line, output polygon_2d as [[y,x], ...]. Use a tight polygon following the visible line. If the line is straight and rectangular, box_2d [ymin,xmin,ymax,xmax] is also acceptable. For curved or circular lines, polygon_2d is mandatory.
4. Granularity: Transcribe at the visual text-line level.
5. Script: Unicode Devanagari.
JSON SCHEMA:
{
  "status": "success",
  "regions": [
    {
      "id": "region_0",
      "type": "main_text",
      "polygon_2d": [[y,x], [y,x], [y,x]],
      "box_2d": [ymin, xmin, ymax, xmax],
      "lines": [
        {
          "id": "line_0",
          "polygon_2d": [[y,x], [y,x], [y,x]],
          "box_2d": [ymin, xmin, ymax, xmax],
          "text": "Transcribed text here"
        }
      ]
    }
  ]
}
""".strip()


class AdapterError(ValueError):
    pass


@dataclass(frozen=True)
class AdapterResult:
    page: PageXmlPage
    output_path: Path | None = None


def _strip_single_json_code_fence(text: str) -> str:
    lines = text.splitlines()
    if len(lines) < 3:
        return text
    opening = lines[0].strip().lower()
    closing = lines[-1].strip()
    if opening not in {"```", "```json"} or closing != "```":
        return text
    return "\n".join(lines[1:-1]).strip()


def parse_json_payload(raw_payload: str | bytes | dict[str, Any]) -> dict[str, Any]:
    if isinstance(raw_payload, dict):
        return raw_payload
    text = raw_payload.decode("utf-8") if isinstance(raw_payload, bytes) else str(raw_payload or "")
    text = text.strip()
    if not text:
        raise AdapterError("empty_response")
    text = _strip_single_json_code_fence(text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise AdapterError("json_parse_error") from exc
    if not isinstance(payload, dict):
        raise AdapterError("json_schema_error")
    return payload


def _scale_yx_pair(pair: Any, page: PageXmlPage) -> tuple[float, float]:
    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
        raise AdapterError("json_schema_error")
    y_norm = float(pair[0])
    x_norm = float(pair[1])
    x_val = max(0.0, min(1000.0, x_norm)) * page.width / 1000.0
    y_val = max(0.0, min(1000.0, y_norm)) * page.height / 1000.0
    return (x_val, y_val)


def _points_from_box(box_2d: Any, page: PageXmlPage) -> tuple[tuple[float, float], ...]:
    if not isinstance(box_2d, (list, tuple)) or len(box_2d) != 4:
        raise AdapterError("json_schema_error")
    ymin, xmin, ymax, xmax = [float(value) for value in box_2d]
    return (
        _scale_yx_pair((ymin, xmin), page),
        _scale_yx_pair((ymin, xmax), page),
        _scale_yx_pair((ymax, xmax), page),
        _scale_yx_pair((ymax, xmin), page),
    )


def _points_from_polygon(raw_polygon: Any, page: PageXmlPage) -> tuple[tuple[float, float], ...]:
    if not isinstance(raw_polygon, (list, tuple)):
        raise AdapterError("json_schema_error")
    return tuple(_scale_yx_pair(point, page) for point in raw_polygon)


def _line_points(line_payload: dict[str, Any], page: PageXmlPage) -> tuple[tuple[float, float], ...]:
    for key in ("polygon_2d", "polygon", "points", "coords"):
        if key in line_payload:
            return _points_from_polygon(line_payload[key], page)
    if "box_2d" in line_payload:
        return _points_from_box(line_payload["box_2d"], page)
    raise AdapterError("json_schema_error")


def vlm_json_to_page(
    raw_payload: str | bytes | dict[str, Any],
    *,
    template_page: PageXmlPage,
) -> PageXmlPage:
    payload = parse_json_payload(raw_payload)
    if payload.get("status", "success") != "success":
        raise AdapterError("other_output_error")
    regions = payload.get("regions")
    if not isinstance(regions, list):
        raise AdapterError("json_schema_error")

    lines: list[TextLine] = []
    for region_index, region in enumerate(regions):
        if not isinstance(region, dict):
            raise AdapterError("json_schema_error")
        region_id = str(region.get("id") or f"region_{region_index}")
        region_lines = region.get("lines")
        if not isinstance(region_lines, list):
            raise AdapterError("json_schema_error")
        for line_index, line_payload in enumerate(region_lines):
            if not isinstance(line_payload, dict):
                raise AdapterError("json_schema_error")
            text = "" if line_payload.get("text") is None else str(line_payload.get("text"))
            points = _line_points(line_payload, template_page)
            line_id = str(line_payload.get("id") or f"{region_id}_line_{line_index}")
            polygon = validate_polygon(
                points,
                width=template_page.width,
                height=template_page.height,
                context=f"vlm_json:{template_page.page_id}:{line_id}",
                repair=True,
            )
            lines.append(
                TextLine(
                    page_id=template_page.page_id,
                    line_id=line_id,
                    points=exterior_points_for_xml(polygon),
                    polygon=polygon,
                    text=text,
                    region_id=region_id,
                )
            )

    return PageXmlPage(
        page_id=template_page.page_id,
        image_filename=template_page.image_filename,
        width=template_page.width,
        height=template_page.height,
        lines=tuple(lines),
        namespace=template_page.namespace,
    )


def vlm_json_to_pagexml(
    raw_payload: str | bytes | dict[str, Any],
    *,
    template_page: PageXmlPage,
    output_path: str | Path,
) -> AdapterResult:
    page = vlm_json_to_page(raw_payload, template_page=template_page)
    written = write_pagexml(page, output_path, lines=page.lines)
    return AdapterResult(page=page, output_path=written)
