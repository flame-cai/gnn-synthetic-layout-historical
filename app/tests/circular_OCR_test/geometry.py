from __future__ import annotations

import math
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


PAGE_XML_NAMESPACE = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
PAGE_XML_NS = {"p": PAGE_XML_NAMESPACE}


@dataclass(frozen=True)
class GnnPoint:
    x: float
    y: float
    font_size: float


@dataclass(frozen=True)
class GnnFormatPage:
    page_id: str
    width: float
    height: float
    points: list[GnnPoint]
    normalized_points: list[GnnPoint]
    textline_labels: list[int]


@dataclass(frozen=True)
class PageXmlLine:
    page_id: str
    region_id: str
    region_custom: str
    line_id: str
    line_custom: str
    text: str
    baseline_points: list[tuple[int, int]]
    coords_points: list[tuple[int, int]]


@dataclass(frozen=True)
class OpenCurve:
    points: list[tuple[float, float]]
    was_closed: bool
    removed_mirrored_tail: bool
    cut_index: int | None = None
    cut_point: tuple[float, float] | None = None


def _read_float_rows(path: Path, expected_columns: int) -> list[list[float]]:
    rows = []
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        values = [float(item) for item in stripped.split()]
        if len(values) != expected_columns:
            raise ValueError(f"{path} line {line_no} has {len(values)} columns, expected {expected_columns}.")
        rows.append(values)
    return rows


def _read_int_rows(path: Path) -> list[int]:
    return [int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_gnn_format_page(gnn_dir: str | Path, page_id: str) -> GnnFormatPage:
    gnn_dir = Path(gnn_dir)
    dims = _read_float_rows(gnn_dir / f"{page_id}_dims.txt", 2)
    if len(dims) != 1:
        raise ValueError(f"Expected exactly one dims row for {page_id}.")
    width, height = dims[0]
    unnormalized = [GnnPoint(*row) for row in _read_float_rows(gnn_dir / f"{page_id}_inputs_unnormalized.txt", 3)]
    normalized_path = gnn_dir / f"{page_id}_inputs_normalized.txt"
    normalized = [GnnPoint(*row) for row in _read_float_rows(normalized_path, 3)] if normalized_path.exists() else []
    labels = _read_int_rows(gnn_dir / f"{page_id}_labels_textline.txt")
    if len(labels) != len(unnormalized):
        raise ValueError(
            f"GNN point/label mismatch for {page_id}: {len(unnormalized)} points and {len(labels)} labels."
        )
    if normalized and len(normalized) != len(unnormalized):
        raise ValueError(
            f"GNN normalized/unnormalized mismatch for {page_id}: {len(normalized)} vs {len(unnormalized)}."
        )
    return GnnFormatPage(
        page_id=page_id,
        width=width,
        height=height,
        points=unnormalized,
        normalized_points=normalized,
        textline_labels=labels,
    )


def parse_points(points_str: str | None) -> list[tuple[int, int]]:
    if not points_str:
        return []
    points = []
    for raw_point in points_str.strip().split():
        x_val, y_val = raw_point.split(",", 1)
        points.append((int(round(float(x_val))), int(round(float(y_val)))))
    return points


def format_points(points: list[tuple[float, float]] | list[tuple[int, int]]) -> str:
    return " ".join(f"{int(round(x_val))},{int(round(y_val))}" for x_val, y_val in points)


def _normalize_text(text: str | None) -> str:
    return unicodedata.normalize("NFC", text or "").strip()


def parse_pagexml_lines(xml_path: str | Path) -> list[PageXmlLine]:
    xml_path = Path(xml_path)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    lines: list[PageXmlLine] = []
    for region_index, region in enumerate(root.findall(".//p:TextRegion", PAGE_XML_NS)):
        region_id = region.get("id", f"region_{region_index}")
        region_custom = region.get("custom", f"textbox_label_{region_index}")
        for line_index, line in enumerate(region.findall("./p:TextLine", PAGE_XML_NS)):
            baseline = line.find("./p:Baseline", PAGE_XML_NS)
            coords = line.find("./p:Coords", PAGE_XML_NS)
            text_equiv = line.find("./p:TextEquiv", PAGE_XML_NS)
            unicode_elem = text_equiv.find("./p:Unicode", PAGE_XML_NS) if text_equiv is not None else None
            lines.append(
                PageXmlLine(
                    page_id=xml_path.stem,
                    region_id=region_id,
                    region_custom=region_custom or "",
                    line_id=line.get("id", f"{region_id}_line_{line_index}"),
                    line_custom=line.get("custom", f"structure_line_id_{line_index}"),
                    text=_normalize_text(unicode_elem.text if unicode_elem is not None else ""),
                    baseline_points=parse_points(baseline.get("points") if baseline is not None else ""),
                    coords_points=parse_points(coords.get("points") if coords is not None else ""),
                )
            )
    return lines


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _dedupe_consecutive(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    deduped: list[tuple[float, float]] = []
    for point in points:
        if not deduped or _distance(deduped[-1], point) > 1e-6:
            deduped.append(point)
    return deduped


def _remove_mirrored_tail(points: list[tuple[float, float]]) -> tuple[list[tuple[float, float]], bool]:
    best_end_index = None
    best_count = 0
    n_points = len(points)
    for pivot_index in range(1, n_points - 1):
        # Pattern A: [..., A, pivot, A, ...]. The pivot appears once and the
        # suffix immediately retraces the prefix.
        count = 0
        while (
            pivot_index - 1 - count >= 0
            and pivot_index + 1 + count < n_points
            and _distance(points[pivot_index - 1 - count], points[pivot_index + 1 + count]) <= 2.0
        ):
            count += 1
        if count > best_count:
            best_count = count
            best_end_index = pivot_index + 1

        # Pattern B: [..., A, pivot, pivot, A, ...]. Some circular PAGE-XML
        # baselines duplicate the cut/start point before retracing.
        if _distance(points[pivot_index], points[pivot_index + 1]) <= 2.0:
            count = 1
            while (
                pivot_index - count >= 0
                and pivot_index + 1 + count < n_points
                and _distance(points[pivot_index - count], points[pivot_index + 1 + count]) <= 2.0
            ):
                count += 1
            if count > best_count:
                best_count = count
                best_end_index = pivot_index + 1

        # Pattern C: closed curve retrace where the cut point is repeated after
        # one full loop: [start, ..., start, next, ...].
        if _distance(points[0], points[pivot_index]) <= 2.0:
            count = 0
            while (
                pivot_index - 1 - count >= 1
                and pivot_index + 1 + count < n_points
                and _distance(points[pivot_index - 1 - count], points[pivot_index + 1 + count]) <= 2.0
            ):
                count += 1
            if count > best_count:
                best_count = count
                best_end_index = pivot_index + 1

    if best_end_index is not None and best_count >= 1:
        return points[:best_end_index], True
    return points, False


def cut_circular_curve_at_topmost(points: list[tuple[float, float]] | list[tuple[int, int]]) -> OpenCurve:
    normalized = [(float(x_val), float(y_val)) for x_val, y_val in points]
    normalized = _dedupe_consecutive(normalized)
    if len(normalized) > 1 and _distance(normalized[0], normalized[-1]) <= 2.0:
        normalized = normalized[:-1]
    if not normalized:
        return OpenCurve(points=[], was_closed=True, removed_mirrored_tail=False, cut_index=None, cut_point=None)
    cut_index, cut_point = min(enumerate(normalized), key=lambda item: (item[1][1], item[1][0]))
    cut_points = normalized[cut_index:] + normalized[:cut_index]
    return OpenCurve(points=cut_points, was_closed=True, removed_mirrored_tail=False, cut_index=cut_index, cut_point=cut_point)


def normalize_open_curve(points: list[tuple[float, float]] | list[tuple[int, int]]) -> OpenCurve:
    normalized = [(float(x_val), float(y_val)) for x_val, y_val in points]
    normalized = _dedupe_consecutive(normalized)
    normalized, removed_tail = _remove_mirrored_tail(normalized)
    was_closed = len(normalized) > 2 and _distance(normalized[0], normalized[-1]) <= 2.0
    if was_closed:
        cut = cut_circular_curve_at_topmost(normalized)
        return OpenCurve(
            points=cut.points,
            was_closed=True,
            removed_mirrored_tail=removed_tail,
            cut_index=cut.cut_index,
            cut_point=cut.cut_point,
        )
    return OpenCurve(points=normalized, was_closed=False, removed_mirrored_tail=removed_tail)


def line_label_points(gnn_page: GnnFormatPage, label: int) -> list[tuple[float, float]]:
    return [(point.x, point.y) for point, point_label in zip(gnn_page.points, gnn_page.textline_labels) if point_label == label]
