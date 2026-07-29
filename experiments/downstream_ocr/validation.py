from __future__ import annotations

import json
from pathlib import Path

from .pagexml import load_pagexml
from .splits import default_manuscript_paths, discover_page_ids


def validate_manuscript_pagexml(manuscript_root: str | Path, *, repair_geometry: bool = False) -> dict:
    paths = default_manuscript_paths(manuscript_root)
    page_ids = discover_page_ids(paths)
    page_results = []
    for page_id in page_ids:
        xml_path = paths.pagexml_dir / f"{page_id}.xml"
        try:
            page = load_pagexml(xml_path, strict=True, repair_geometry=repair_geometry)
            page_results.append(
                {
                    "page_id": page_id,
                    "xml_path": str(xml_path.resolve()),
                    "valid": True,
                    "line_count": len(page.lines),
                    "error": None,
                }
            )
        except Exception as exc:
            page_results.append(
                {
                    "page_id": page_id,
                    "xml_path": str(xml_path.resolve()),
                    "valid": False,
                    "line_count": None,
                    "error": str(exc),
                }
            )
    invalid_pages = [item for item in page_results if not item["valid"]]
    return {
        "manuscript_id": paths.manuscript_id,
        "page_count": len(page_results),
        "valid_page_count": len(page_results) - len(invalid_pages),
        "invalid_page_count": len(invalid_pages),
        "valid": not invalid_pages,
        "repair_geometry": bool(repair_geometry),
        "pages": page_results,
    }


def write_validation_report(report: dict, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return output
