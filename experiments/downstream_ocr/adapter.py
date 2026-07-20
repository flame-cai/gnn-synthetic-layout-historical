from __future__ import annotations

import json
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from shapely.geometry import GeometryCollection

from .pagexml import (
    PageXmlPage,
    TextLine,
    exterior_points_for_xml,
    validate_polygon,
    write_pagexml,
    write_text_only_pagexml,
)


VLM_JSON_OUTPUT_ADAPTER_ID = "vlm_json_geometry_v2"
SARVAM_HTML_OUTPUT_ADAPTER_ID = "sarvam_html_text_lines_v1"


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


_SARVAM_TEXT_BLOCK_TAGS = frozenset(
    {
        "p",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "li",
        "dt",
        "dd",
        "blockquote",
        "figcaption",
        "caption",
        "pre",
        "header",
        "footer",
        "aside",
        "address",
        "td",
        "th",
        "body",
        "main",
        "article",
        "section",
        "nav",
        "div",
        "ol",
        "ul",
        "menu",
        "dl",
        "figure",
        "table",
        "thead",
        "tbody",
        "tfoot",
        "tr",
        "form",
        "fieldset",
        "legend",
        "details",
        "summary",
        "dialog",
    }
)
_SARVAM_TEXT_BLOCK_CLASSES = frozenset(
    {
        "paragraph",
        "block",
        "quote",
        "ordered-list",
        "unordered-list",
        "sub-ordered-list",
        "sub-unordered-list",
        "subsub-ordered-list",
        "subsub-unordered-list",
        "advertisement",
        "answer",
        "contact-info",
        "index",
        "options",
        "reference",
        "unknown",
        "sidebar",
        "footnote",
        "image-caption",
        "header",
        "footer",
        "author",
        "dateline",
        "flag",
        "formula",
        "jumpline",
        "page-number",
        "folio",
        "website-link",
        "first-level-question",
        "second-level-question",
        "third-level-question",
        "chapter-title",
        "section-title",
        "headline",
        "sub-section-title",
        "sub-headline",
        "subsub-section-title",
        "subsub-headline",
    }
)
_SARVAM_IGNORED_TAGS = frozenset(
    {"head", "style", "script", "noscript", "template", "svg"}
)
_HTML_VOID_TAGS = frozenset(
    {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }
)


def _normalize_html_line(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


class _SarvamHtmlLineParser(HTMLParser):
    """Extract semantic HTML blocks and split each block at explicit BR tags."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.lines: list[str] = []
        self._chunks: list[str] = []
        self._block_depth = 0
        self._ignored_depth = 0
        self._element_stack: list[tuple[str, bool, bool]] = []

    def _flush_line(self) -> None:
        line = _normalize_html_line("".join(self._chunks))
        self._chunks.clear()
        if line:
            self.lines.append(line)

    @staticmethod
    def _is_text_block(tag: str, attrs) -> bool:
        if tag in _SARVAM_TEXT_BLOCK_TAGS:
            return True
        classes = {
            class_name
            for attr_name, attr_value in attrs
            if attr_name.lower() == "class" and attr_value
            for class_name in str(attr_value).lower().split()
        }
        return bool(classes & _SARVAM_TEXT_BLOCK_CLASSES)

    def _close_frame(self, frame: tuple[str, bool, bool]) -> None:
        _, starts_block, starts_ignored = frame
        if starts_block:
            self._flush_line()
            self._block_depth = max(0, self._block_depth - 1)
        if starts_ignored:
            self._ignored_depth = max(0, self._ignored_depth - 1)

    def handle_starttag(self, tag: str, attrs) -> None:
        normalized_tag = tag.lower()
        if normalized_tag in {"br", "hr"}:
            if self._block_depth and not self._ignored_depth:
                self._flush_line()
            return

        starts_ignored = normalized_tag in _SARVAM_IGNORED_TAGS
        if starts_ignored:
            self._ignored_depth += 1
        starts_block = (
            not self._ignored_depth and self._is_text_block(normalized_tag, attrs)
        )
        if starts_block:
            if self._block_depth:
                self._flush_line()
            self._block_depth += 1

        if normalized_tag not in _HTML_VOID_TAGS:
            self._element_stack.append(
                (normalized_tag, starts_block, starts_ignored)
            )

    def handle_startendtag(self, tag: str, attrs) -> None:
        normalized_tag = tag.lower()
        self.handle_starttag(tag, attrs)
        if normalized_tag not in _HTML_VOID_TAGS:
            self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        normalized_tag = tag.lower()
        if normalized_tag in {"br", "hr"}:
            if self._block_depth and not self._ignored_depth:
                self._flush_line()
            return

        matching_index = next(
            (
                index
                for index in range(len(self._element_stack) - 1, -1, -1)
                if self._element_stack[index][0] == normalized_tag
            ),
            None,
        )
        if matching_index is None:
            return
        frames = self._element_stack[matching_index:]
        del self._element_stack[matching_index:]
        for frame in reversed(frames):
            self._close_frame(frame)

    def handle_data(self, data: str) -> None:
        if self._block_depth and not self._ignored_depth:
            self._chunks.append(data)

    def close(self) -> None:
        super().close()
        for frame in reversed(self._element_stack):
            self._close_frame(frame)
        self._element_stack.clear()
        self._flush_line()


def extract_sarvam_html_text_lines(raw_html: str | bytes) -> tuple[str, ...]:
    if isinstance(raw_html, bytes):
        try:
            html_text = raw_html.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise AdapterError("html_parse_error") from exc
    else:
        html_text = str(raw_html or "")
    if not html_text.strip():
        raise AdapterError("empty_response")

    parser = _SarvamHtmlLineParser()
    try:
        parser.feed(html_text)
        parser.close()
    except Exception as exc:
        raise AdapterError("html_parse_error") from exc
    if not parser.lines:
        raise AdapterError("html_parse_error")
    return tuple(parser.lines)


def sarvam_html_to_page(
    raw_html: str | bytes,
    *,
    template_page: PageXmlPage,
) -> PageXmlPage:
    text_lines = extract_sarvam_html_text_lines(raw_html)
    region_id = "region_0"
    lines = tuple(
        TextLine(
            page_id=template_page.page_id,
            line_id=f"line_{line_index}",
            points=(),
            polygon=GeometryCollection(),
            text=text,
            region_id=region_id,
        )
        for line_index, text in enumerate(text_lines)
    )
    return PageXmlPage(
        page_id=template_page.page_id,
        image_filename=template_page.image_filename,
        width=template_page.width,
        height=template_page.height,
        lines=lines,
        namespace=template_page.namespace,
    )


def sarvam_html_to_pagexml(
    raw_html: str | bytes,
    *,
    template_page: PageXmlPage,
    output_path: str | Path,
) -> AdapterResult:
    page = sarvam_html_to_page(raw_html, template_page=template_page)
    written = write_text_only_pagexml(
        page,
        output_path,
        lines=page.lines,
        region_id="region_0",
    )
    return AdapterResult(page=page, output_path=written)


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


def provider_output_to_pagexml(
    raw_payload: str | bytes,
    *,
    output_adapter_id: str,
    template_page: PageXmlPage,
    output_path: str | Path,
) -> AdapterResult:
    if output_adapter_id == VLM_JSON_OUTPUT_ADAPTER_ID:
        return vlm_json_to_pagexml(
            raw_payload,
            template_page=template_page,
            output_path=output_path,
        )
    if output_adapter_id == SARVAM_HTML_OUTPUT_ADAPTER_ID:
        return sarvam_html_to_pagexml(
            raw_payload,
            template_page=template_page,
            output_path=output_path,
        )
    raise ValueError(f"Unsupported provider output adapter: {output_adapter_id!r}.")
