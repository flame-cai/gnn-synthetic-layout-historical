import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
import xml.etree.ElementTree as ET


TEXT_RECOVERY_BACKUP_DIR = "text_recovery_backups"
PAGE_XML_NAMESPACE = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
MIN_TEXT_SIMILARITY_WITH_COORDS = 0.70
MIN_COMBINED_SCORE_WITH_COORDS = 0.78
MIN_TEXT_SIMILARITY_WITHOUT_COORDS = 0.88
MIN_SHORT_TEXT_SIMILARITY = 0.98
AMBIGUITY_MARGIN = 0.04


@dataclass(frozen=True)
class PageXmlLine:
    line_id: str
    text: str
    coords: tuple[tuple[float, float], ...]
    bbox: tuple[float, float, float, float] | None


def _utc_backup_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _local_name(tag: str) -> str:
    return str(tag).rsplit("}", 1)[-1]


def _iter_children_named(elem, name: str):
    return [child for child in list(elem) if _local_name(child.tag) == name]


def _first_child_named(elem, name: str):
    for child in list(elem):
        if _local_name(child.tag) == name:
            return child
    return None


def _extract_structure_line_id(custom_attr: str | None) -> str | None:
    match = re.search(r"structure_line_id_([^;\s]+)", str(custom_attr or ""))
    return match.group(1) if match else None


def _parse_points(points_str: str | None) -> tuple[tuple[float, float], ...]:
    points = []
    for raw_point in str(points_str or "").split():
        if "," not in raw_point:
            continue
        x_raw, y_raw = raw_point.split(",", 1)
        try:
            points.append((float(x_raw), float(y_raw)))
        except ValueError:
            continue
    return tuple(points)


def _bbox_for_points(points: tuple[tuple[float, float], ...]):
    if not points:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return (min(xs), min(ys), max(xs), max(ys))


def _bbox_iou(a, b) -> float | None:
    if a is None or b is None:
        return None
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def normalize_text_for_recovery(text: str | None) -> str:
    value = "" if text is None else str(text)
    value = value.replace("\u200c", "").replace("\u200d", "")
    return re.sub(r"\s+", "", value.strip())


def text_similarity_for_recovery(a: str | None, b: str | None) -> float:
    left = normalize_text_for_recovery(a)
    right = normalize_text_for_recovery(b)
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def extract_page_xml_lines(xml_path: str | Path) -> list[PageXmlLine]:
    xml_path = Path(xml_path)
    if not xml_path.exists():
        return []
    tree = ET.parse(xml_path)
    root = tree.getroot()
    lines: list[PageXmlLine] = []
    for elem in root.iter():
        if _local_name(elem.tag) != "TextLine":
            continue
        line_id = _extract_structure_line_id(elem.get("custom"))
        if line_id is None:
            continue
        coords_elem = _first_child_named(elem, "Coords")
        coords = _parse_points(coords_elem.get("points", "") if coords_elem is not None else "")
        text_equiv = _first_child_named(elem, "TextEquiv")
        unicode_elem = _first_child_named(text_equiv, "Unicode") if text_equiv is not None else None
        text = unicode_elem.text if unicode_elem is not None and unicode_elem.text else ""
        lines.append(PageXmlLine(line_id=str(line_id), text=str(text), coords=coords, bbox=_bbox_for_points(coords)))
    return lines


def _non_empty_text_lines(lines: list[PageXmlLine]) -> list[PageXmlLine]:
    return [line for line in lines if normalize_text_for_recovery(line.text)]


def text_recovery_backup_root(manuscript_root: str | Path) -> Path:
    return Path(manuscript_root) / "layout_analysis_output" / TEXT_RECOVERY_BACKUP_DIR


def text_recovery_page_backup_dir(manuscript_root: str | Path, page: str) -> Path:
    return text_recovery_backup_root(manuscript_root) / str(page)


def backup_page_xml_for_text_recovery(
    manuscript_root: str | Path,
    page: str,
    *,
    xml_path: str | Path | None = None,
    layout_fingerprint: str | None = None,
) -> dict | None:
    manuscript_root = Path(manuscript_root)
    xml_path = Path(xml_path or manuscript_root / "layout_analysis_output" / "page-xml-format" / f"{page}.xml")
    if not xml_path.exists():
        return None

    lines = extract_page_xml_lines(xml_path)
    non_empty_count = len(_non_empty_text_lines(lines))
    backup_dir = text_recovery_page_backup_dir(manuscript_root, page)
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_id = _utc_backup_id()
    backup_xml_path = backup_dir / f"{backup_id}.xml"
    metadata_path = backup_dir / f"{backup_id}.json"
    shutil.copy2(xml_path, backup_xml_path)

    metadata = {
        "backup_id": backup_id,
        "page": str(page),
        "source_xml": str(xml_path),
        "backup_xml": str(backup_xml_path),
        "backed_up_at": datetime.now(timezone.utc).isoformat(),
        "layout_fingerprint": layout_fingerprint,
        "text_line_count": len(lines),
        "non_empty_text_line_count": non_empty_count,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata


def latest_text_recovery_backup(manuscript_root: str | Path, page: str) -> dict | None:
    backup_dir = text_recovery_page_backup_dir(manuscript_root, page)
    if not backup_dir.exists():
        return None
    metadata_paths = sorted(backup_dir.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    for metadata_path in metadata_paths:
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        backup_xml = Path(metadata.get("backup_xml") or metadata_path.with_suffix(".xml"))
        if not backup_xml.is_absolute():
            backup_xml = metadata_path.parent / backup_xml
        if backup_xml.exists():
            metadata["backup_xml"] = str(backup_xml)
            metadata["metadata_path"] = str(metadata_path)
            return metadata
    xml_paths = sorted(backup_dir.glob("*.xml"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not xml_paths:
        return None
    backup_xml = xml_paths[0]
    return {
        "backup_id": backup_xml.stem,
        "page": str(page),
        "backup_xml": str(backup_xml),
        "metadata_path": None,
    }


def build_text_recovery_state(
    manuscript_root: str | Path,
    page: str,
    *,
    current_xml_path: str | Path | None = None,
) -> dict:
    manuscript_root = Path(manuscript_root)
    current_xml_path = Path(
        current_xml_path or manuscript_root / "layout_analysis_output" / "page-xml-format" / f"{page}.xml"
    )
    backup = latest_text_recovery_backup(manuscript_root, page)
    if not backup:
        return {"available": False, "reason": "no_backup"}

    try:
        backup_lines = extract_page_xml_lines(backup["backup_xml"])
        backup_non_empty = len(_non_empty_text_lines(backup_lines))
    except Exception:
        return {"available": False, "reason": "invalid_backup", "backup_id": backup.get("backup_id")}

    current_non_empty = 0
    if current_xml_path.exists():
        try:
            current_non_empty = len(_non_empty_text_lines(extract_page_xml_lines(current_xml_path)))
        except Exception:
            current_non_empty = 0

    available = backup_non_empty > 0 and current_xml_path.exists() and current_non_empty > 0
    return {
        "available": bool(available),
        "reason": "ready" if available else "missing_current_text",
        "backup_id": backup.get("backup_id"),
        "backed_up_at": backup.get("backed_up_at"),
        "backup_text_line_count": int(backup_non_empty),
        "current_text_line_count": int(current_non_empty),
    }


def _candidate_score(backup_line: PageXmlLine, current_line: PageXmlLine) -> dict | None:
    text_similarity = text_similarity_for_recovery(backup_line.text, current_line.text)
    overlap = _bbox_iou(backup_line.bbox, current_line.bbox)
    normalized_length = max(
        len(normalize_text_for_recovery(backup_line.text)),
        len(normalize_text_for_recovery(current_line.text)),
    )
    if normalized_length < 4:
        if text_similarity < MIN_SHORT_TEXT_SIMILARITY:
            return None
        if overlap is not None and overlap < 0.10:
            return None

    if overlap is None:
        if text_similarity < MIN_TEXT_SIMILARITY_WITHOUT_COORDS:
            return None
        score = text_similarity
    else:
        score = (0.75 * text_similarity) + (0.25 * overlap)
        if text_similarity < MIN_TEXT_SIMILARITY_WITH_COORDS and not (text_similarity >= 0.92 and overlap >= 0.10):
            return None
        if score < MIN_COMBINED_SCORE_WITH_COORDS and text_similarity < 0.93:
            return None
        if overlap < 0.05 and text_similarity < 0.93:
            return None

    return {
        "backup_line_id": backup_line.line_id,
        "current_line_id": current_line.line_id,
        "score": round(float(score), 6),
        "text_similarity": round(float(text_similarity), 6),
        "coords_overlap": None if overlap is None else round(float(overlap), 6),
        "recovered_text": backup_line.text,
        "ocr_text": current_line.text,
    }


def build_text_recovery_plan(backup_xml_path: str | Path, current_xml_path: str | Path) -> dict:
    backup_lines = _non_empty_text_lines(extract_page_xml_lines(backup_xml_path))
    current_lines = _non_empty_text_lines(extract_page_xml_lines(current_xml_path))
    candidates_by_current: dict[str, list[dict]] = {}

    for current_line in current_lines:
        candidates = []
        for backup_line in backup_lines:
            candidate = _candidate_score(backup_line, current_line)
            if candidate is not None:
                candidates.append(candidate)
        candidates.sort(key=lambda item: item["score"], reverse=True)
        candidates_by_current[current_line.line_id] = candidates

    selected_by_current = {}
    skipped = []
    for current_line in current_lines:
        candidates = candidates_by_current.get(current_line.line_id, [])
        if not candidates:
            skipped.append({"current_line_id": current_line.line_id, "reason": "no_candidate"})
            continue
        best = candidates[0]
        second = candidates[1] if len(candidates) > 1 else None
        if second is not None and best["score"] - second["score"] < AMBIGUITY_MARGIN:
            skipped.append(
                {
                    "current_line_id": current_line.line_id,
                    "reason": "ambiguous",
                    "best_score": best["score"],
                    "second_score": second["score"],
                }
            )
            continue
        selected_by_current[current_line.line_id] = best

    matches = []
    used_backup_ids = set()
    for candidate in sorted(selected_by_current.values(), key=lambda item: item["score"], reverse=True):
        if candidate["backup_line_id"] in used_backup_ids:
            skipped.append(
                {
                    "current_line_id": candidate["current_line_id"],
                    "backup_line_id": candidate["backup_line_id"],
                    "reason": "backup_line_already_used",
                    "score": candidate["score"],
                }
            )
            continue
        used_backup_ids.add(candidate["backup_line_id"])
        matches.append(candidate)

    recovered_current_ids = {match["current_line_id"] for match in matches}
    unrecovered_line_ids = [line.line_id for line in current_lines if line.line_id not in recovered_current_ids]
    return {
        "matches": matches,
        "unrecovered_line_ids": unrecovered_line_ids,
        "skipped": skipped,
        "matched_line_count": len(matches),
        "current_line_count": len(current_lines),
        "backup_line_count": len(backup_lines),
    }


def build_latest_text_recovery_plan(manuscript_root: str | Path, page: str, current_xml_path: str | Path) -> dict:
    backup = latest_text_recovery_backup(manuscript_root, page)
    if not backup:
        return {
            "available": False,
            "reason": "no_backup",
            "matches": [],
            "unrecovered_line_ids": [],
            "skipped": [],
        }
    plan = build_text_recovery_plan(backup["backup_xml"], current_xml_path)
    plan["available"] = True
    plan["backup_id"] = backup.get("backup_id")
    plan["backed_up_at"] = backup.get("backed_up_at")
    return plan
