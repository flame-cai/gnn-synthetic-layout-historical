"""Mark old layout backups whose recovery prompt no longer serves a purpose as answered.

A backup's prompt is only useful while the text it holds is not yet on the page.
This marks a page's latest backup answered ("migrated") only when recovering from
it could not restore anything the page lacks:

  - Recover would change no line, or
  - every line it would change was edited after that layout change, i.e. the
    page's text differs from the OCR reading recorded after the backup, so the
    page holds the newer text and recovering would put older text over it.

A page with any changed line still equal to that OCR reading (or with no OCR
record to compare against) is left alone and listed: its backup may hold the
only copy of corrected text. No page text is modified; only the backup's
metadata gains an "answer". Dry run unless --apply.

    python scripts/mark_old_text_recovery_backups.py            # report
    python scripts/mark_old_text_recovery_backups.py --apply    # mark
"""
import argparse
import json
import sys
import unicodedata
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT))

from text_recovery import (  # noqa: E402
    build_latest_text_recovery_plan,
    build_text_recovery_state,
    extract_page_xml_lines,
    record_text_recovery_answer,
)


def _norm(text):
    return unicodedata.normalize("NFC", text or "").replace("॥", "।।").replace(" ", "")


def classify(manuscript_root: Path, page: str) -> dict:
    current_xml = manuscript_root / "layout_analysis_output" / "page-xml-format" / f"{page}.xml"
    state = build_text_recovery_state(manuscript_root, page, current_xml_path=current_xml)
    result = {"page": page, "backup_id": state.get("backup_id"), "state": state}
    if not state.get("available"):
        return {**result, "verdict": "not_offered"}
    if state.get("answer"):
        return {**result, "verdict": "already_answered"}
    plan = build_latest_text_recovery_plan(manuscript_root, page, current_xml)
    current = {line.line_id: line.text for line in extract_page_xml_lines(current_xml)}
    changes = [
        match for match in plan["matches"]
        if (match.get("recovered_text") or "") != (current.get(str(match["current_line_id"])) or "")
    ]
    if not changes:
        return {**result, "verdict": "no_op", "changed": 0}

    registry_path = manuscript_root / "active_learning" / "recognition" / "registry.json"
    history = []
    if registry_path.exists():
        history = json.loads(registry_path.read_text(encoding="utf-8")).get("prediction_history_by_page", {}).get(page, [])
    backed_up_at = state.get("backed_up_at") or ""
    readings = [entry for entry in history if backed_up_at and entry.get("recorded_at", "") > backed_up_at]
    if not readings:
        return {**result, "verdict": "keep_prompt", "why": "no OCR reading recorded after the backup", "changed": len(changes)}
    reading = readings[0].get("predicted_lines", {})
    uncorrected = [
        match for match in changes
        if _norm(reading.get(str(match["current_line_id"]))) == _norm(current.get(str(match["current_line_id"])))
    ]
    if uncorrected:
        return {**result, "verdict": "keep_prompt", "changed": len(changes),
                "why": f"{len(uncorrected)} line(s) still hold the uncorrected OCR reading; the backup has their corrected text"}
    return {**result, "verdict": "edited_since", "changed": len(changes)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=APP_ROOT / "input_manuscripts")
    parser.add_argument("--apply", action="store_true", help="record the answers (default: report only)")
    args = parser.parse_args()

    counts = {}
    for backup_dir in sorted(args.root.glob("*/layout_analysis_output/text_recovery_backups/*")):
        if not backup_dir.is_dir():
            continue
        manuscript_root, page = backup_dir.parents[2], backup_dir.name
        row = classify(manuscript_root, page)
        verdict = row["verdict"]
        counts[verdict] = counts.get(verdict, 0) + 1
        label = f"{manuscript_root.name}/{page}"
        if verdict == "keep_prompt":
            print(f"KEEP PROMPT  {label}: {row['why']}")
        elif verdict in ("no_op", "edited_since"):
            if args.apply:
                record_text_recovery_answer(manuscript_root, page, row["backup_id"], "migrated",
                                            reason=verdict, changed_line_count=row.get("changed", 0))
            print(f"{'marked' if args.apply else 'would mark'}  {label}: {verdict} ({row.get('changed', 0)} line(s) differ)")
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
