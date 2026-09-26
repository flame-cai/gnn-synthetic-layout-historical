"""Read-only hygiene checks over the annotated manuscripts in app/input_manuscripts.

Per page it reports:
  structure   page XML parses, has lines, every line has coordinates and a baseline,
              no duplicate line ids, the page image exists
  review      the tool's review status, and whether the page text differs from its
              last ground-truth commit (edited or recovered but never committed)
  ocr         committed lines still identical to the OCR reading of the current
              layout (possibly never corrected)
  text        empty lines on transcribed pages, ॥ instead of ।।, ASCII | or Latin
              letters, characters the OCR model cannot learn, non-NFC text,
              invisible characters, stray whitespace, a line repeated on the page
  recovery    text-recovery backups whose prompt is still unanswered

Nothing is written into the manuscripts.

    python scripts/check_manuscript_hygiene.py --exclude-prefix newar_ --out report.json
"""
import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT))

from text_recovery import build_text_recovery_state, extract_page_xml_lines  # noqa: E402



def _ocr_charset() -> set[str]:
    """The local OCR model's character list, read from its source without importing it."""
    import ast
    source = (APP_ROOT / "recognition" / "recognize_manuscript_text_v2.py").read_text(encoding="utf-8")
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "SANSKRIT_CHARACTERS" for t in node.targets):
            return set(ast.literal_eval(node.value))
    raise RuntimeError("SANSKRIT_CHARACTERS not found")


CHARSET = _ocr_charset()
DOUBLE_DANDA = "॥"
INVISIBLE = {"​": "ZWSP", "‌": "ZWNJ", "‍": "ZWJ", "﻿": "BOM", " ": "NBSP", "­": "SHY"}
LATIN = re.compile(r"[A-Za-z]")
DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def _page_ids(manuscript_root: Path) -> list[str]:
    xml_dir = manuscript_root / "layout_analysis_output" / "page-xml-format"
    return sorted(p.stem for p in xml_dir.glob("*.xml"))


def _registry(manuscript_root: Path) -> dict:
    path = manuscript_root / "active_learning" / "recognition" / "registry.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _texts(xml_path: Path) -> dict[str, str]:
    return {line.line_id: line.text for line in extract_page_xml_lines(xml_path)}


def check_page(manuscript_root: Path, page: str, registry: dict, workflow_for) -> dict:
    xml_path = manuscript_root / "layout_analysis_output" / "page-xml-format" / f"{page}.xml"
    issues: list[str] = []
    row = {"page": page, "issues": issues}
    try:
        lines = extract_page_xml_lines(xml_path)
    except Exception as exc:  # noqa: BLE001
        issues.append(f"structure: page XML does not parse ({exc})")
        return row

    ids = [line.line_id for line in lines]
    texts = {line.line_id: line.text or "" for line in lines}
    with_text = [i for i in ids if texts[i].strip()]
    row.update(lines=len(lines), lines_with_text=len(with_text))
    if not lines:
        issues.append("structure: no text lines")
    duplicates = [i for i, n in Counter(ids).items() if n > 1]
    if duplicates:
        issues.append(f"structure: duplicate line ids {duplicates}")
    no_coords = [line.line_id for line in lines if len(line.coords) < 3]
    if no_coords:
        issues.append(f"structure: lines without a polygon {no_coords}")
    no_baseline = [line.line_id for line in lines if not line.baseline]
    if no_baseline:
        issues.append(f"structure: lines without a baseline {no_baseline}")
    images = manuscript_root / "images_resized"
    if not any((images / f"{page}{ext}").exists() for ext in (".jpg", ".jpeg", ".png", ".tif")):
        issues.append("structure: page image missing from images_resized/")

    # Review status as the GUI computes it.
    workflow = workflow_for(manuscript_root, page, texts)
    status = workflow.get("review_status")
    row["review_status"] = status

    # Text differing from the last ground-truth commit.
    revisions = registry.get("page_revisions", {}).get(page, [])
    commits = [r for r in revisions if r.get("save_intent") == "commit" and r.get("supervision_present")]
    if commits:
        last = commits[-1]
        rev_xml = (manuscript_root / "active_learning" / "recognition" / "revisions" / page
                   / f"rev_{int(last['revision_number']):04d}" / "page-xml-format" / f"{page}.xml")
        if rev_xml.exists():
            committed = {k: v for k, v in _texts(rev_xml).items() if (v or "").strip()}
            current = {k: v for k, v in texts.items() if v.strip()}
            if committed != current:
                changed = sorted(set(committed) ^ set(current)) + [k for k in committed if k in current and committed[k] != current[k]]
                issues.append(f"review: text differs from the last ground-truth commit (rev {last['revision_number']}, "
                              f"{last['created_at'][:10]}) on {len(set(changed))} line(s)")

    # Committed lines still equal to the OCR reading of this layout.
    if commits and with_text:
        fingerprint = workflow.get("_layout_fingerprint")
        history = registry.get("prediction_history_by_page", {}).get(page, [])
        readings = [h for h in history if fingerprint and h.get("layout_fingerprint") == fingerprint]
        if readings:
            predicted = readings[-1].get("predicted_lines", {})
            same = [i for i in with_text if len(texts[i].strip()) >= 8 and (predicted.get(i) or "").strip() == texts[i].strip()]
            long_lines = [i for i in with_text if len(texts[i].strip()) >= 8]
            row["lines_equal_to_ocr"] = f"{len(same)}/{len(long_lines)}"
            if long_lines and len(same) / len(long_lines) >= 0.5:
                issues.append(f"ocr: {len(same)} of {len(long_lines)} long lines are exactly the OCR reading of this "
                              f"layout; possibly committed without correction")

    # Text content.
    if with_text:
        empty = [i for i in ids if not texts[i].strip()]
        if empty:
            issues.append(f"text: {len(empty)} line(s) without text {empty}")
    all_text = "".join(texts.values())
    if DOUBLE_DANDA in all_text:
        issues.append(f"text: {all_text.count(DOUBLE_DANDA)} × ॥ (convention is ।।)")
    if "|" in all_text:
        issues.append(f"text: {all_text.count('|')} × ASCII | (danda typed as a pipe?)")
    latin = [i for i in with_text if LATIN.search(texts[i]) and DEVANAGARI.search(texts[i])]
    if latin:
        issues.append(f"text: Latin letters inside Devanagari lines {latin}: "
                      + "; ".join(repr(texts[i][:40]) for i in latin[:3]))
    outside = Counter(ch for ch in all_text if ch not in CHARSET and not ch.isspace() and ch not in INVISIBLE)
    if outside:
        issues.append("text: characters the OCR model cannot learn: "
                      + ", ".join(f"{ch!r} U+{ord(ch):04X} ×{n}" for ch, n in outside.most_common()))
    not_nfc = [i for i in with_text if unicodedata.normalize("NFC", texts[i]) != texts[i]]
    if not_nfc:
        issues.append(f"text: not NFC-normalized {not_nfc}")
    invisible = Counter(INVISIBLE[ch] for ch in all_text if ch in INVISIBLE)
    if invisible:
        issues.append("text: invisible characters " + ", ".join(f"{k} ×{n}" for k, n in invisible.items()))
    spaced = [i for i in with_text if texts[i] != texts[i].strip() or "  " in texts[i]]
    if spaced:
        issues.append(f"text: leading/trailing/double spaces {spaced}")
    repeated = [t for t, n in Counter(texts[i].strip() for i in with_text if len(texts[i].strip()) >= 8).items() if n > 1]
    if repeated:
        issues.append(f"text: same text on {len(repeated)} pair(s) of lines: " + "; ".join(repr(t[:40]) for t in repeated))

    recovery = build_text_recovery_state(manuscript_root, page, current_xml_path=xml_path)
    if recovery.get("available") and not recovery.get("answer"):
        issues.append("recovery: text-recovery backup with an unanswered prompt")
    return row


def _kind(issue: str) -> str:
    """An issue without its page-specific numbers and examples, for the per-manuscript summary."""
    head = re.split(r"[\[(:]", issue.split(": ", 1)[1], maxsplit=1)[0] if ": " in issue else issue
    return issue.split(":")[0] + ": " + re.sub(r"\d+", "N", head).replace("N × ", "").strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=APP_ROOT / "input_manuscripts")
    parser.add_argument("--exclude-prefix", action="append", default=[])
    parser.add_argument("--out", type=Path, help="write the full report as JSON")
    args = parser.parse_args()

    import os
    os.chdir(APP_ROOT)  # app.py resolves its checkpoint relative to app/
    # Loading a registry normally writes it straight back (and creates one where
    # none exists). A read-only check must not, least of all while the GUI server
    # may be writing the same file, so saving is disabled in this process.
    import manuscript_layout_registry
    import manuscript_ocr_registry
    manuscript_ocr_registry.ManuscriptOcrRegistry.save = lambda self: None
    manuscript_layout_registry.ManuscriptLayoutRegistry.save = lambda self: None
    import app as app_module

    def workflow_for(manuscript_root, page, texts):
        xml_path = manuscript_root / "layout_analysis_output" / "page-xml-format" / f"{page}.xml"
        payload = app_module.get_existing_text_content(str(xml_path)).get("text", {})
        workflow = app_module._build_page_workflow(manuscript_root, page, text_payload=payload)
        workflow["_layout_fingerprint"] = app_module.compute_page_layout_fingerprint(str(xml_path))
        return workflow

    report = []
    for manuscript_root in sorted(p for p in args.root.iterdir() if p.is_dir()):
        if any(manuscript_root.name.startswith(prefix) for prefix in args.exclude_prefix):
            continue
        pages = _page_ids(manuscript_root)
        if not pages:
            continue
        registry = _registry(manuscript_root)
        rows = [check_page(manuscript_root, page, registry, workflow_for) for page in pages]
        statuses = Counter(row.get("review_status") for row in rows)
        flagged = [row for row in rows if row["issues"]]
        print(f"\n## {manuscript_root.name}: {len(rows)} pages, {len(flagged)} with findings")
        print("   review status: " + ", ".join(f"{k} {v}" for k, v in statuses.most_common()))
        kinds = Counter(_kind(issue) for row in rows for issue in row["issues"])
        for kind, n in kinds.most_common():
            print(f"   {n:>3} pages: {kind}")
        report.append({"manuscript": manuscript_root.name, "review_status": dict(statuses), "pages": rows})
    if args.out:
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\nfull report: {args.out}")

if __name__ == "__main__":
    main()
