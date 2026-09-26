"""Apply the project's Devanagari text conventions to the ground truth of existing manuscripts.

Replacements (character for character, nothing else in a line changes):

  |   ASCII pipe typed as a danda      -> ।  (U+0964), so || -> ।। and |।। -> ।।।
  S s Latin letter typed as avagraha   -> ऽ  (U+093D), only directly after a Devanagari
                                          character and not inside a Latin word
  x   lowercase form of the X marker   -> X, only in a line with Devanagari and not
                                          inside a Latin word

How a page is changed, as in the earlier double-danda fix:

  * a page with a committed ground truth gets a Read Mode commit of the corrected
    text through the app's own save route (a new ground-truth revision; active
    learning off, so nothing trains);
  * any other page gets its PAGE-XML text updated in place, keeping confidences.

A page whose text differs from its last ground-truth commit is skipped and listed:
committing it would also commit someone's unreviewed edits.

Everything touched is backed up first under app/scripts/work/text_conventions_<time>/,
with a log of every replacement (applied.json). To undo: copy the backed-up files
back and delete the revision folders the log lists as created.

    python scripts/apply_text_conventions.py --exclude-prefix newar_ --exclude my_manuscript_testing
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT))

from text_recovery import extract_page_xml_lines  # noqa: E402

DANDA = "।"
AVAGRAHA = "ऽ"
DEVANAGARI = r"ऀ-ॿ"
RULES = [
    ("pipe_to_danda", re.compile(r"\|"), DANDA),
    ("latin_s_to_avagraha", re.compile(rf"(?<=[{DEVANAGARI}])[Ss](?![A-Za-z])"), AVAGRAHA),
    ("lowercase_x_marker", re.compile(r"(?<![A-Za-z])x(?![A-Za-z])"), "X"),
]


def convert(text: str) -> tuple[str, list[dict]]:
    """The converted line and every replacement made, with its context."""
    replacements = []
    for name, pattern, replacement in RULES:
        if name == "lowercase_x_marker" and not re.search(f"[{DEVANAGARI}]", text):
            continue
        for match in pattern.finditer(text):
            replacements.append({
                "rule": name,
                "at": match.start(),
                "context": text[max(0, match.start() - 10):match.end() + 10],
            })
        text = pattern.sub(replacement, text)
    return text, replacements


def committed_revisions(manuscript_root: Path) -> dict[str, dict]:
    """The latest supervised commit per page, from the manuscript's OCR registry."""
    registry = manuscript_root / "active_learning" / "recognition" / "registry.json"
    if not registry.exists():
        return {}
    latest = {}
    for page_id, revisions in json.loads(registry.read_text(encoding="utf-8")).get("page_revisions", {}).items():
        commits = [r for r in revisions if r.get("save_intent") == "commit" and r.get("supervision_present")]
        if commits:
            latest[page_id] = commits[-1]
    return latest


def committed_text(manuscript_root: Path, page_id: str, revision: dict) -> dict[str, str] | None:
    path = (manuscript_root / "active_learning" / "recognition" / "revisions" / page_id
            / f"rev_{int(revision['revision_number']):04d}" / "page-xml-format" / f"{page_id}.xml")
    if not path.exists():
        return None
    return {line.line_id: line.text for line in extract_page_xml_lines(path) if (line.text or "").strip()}


def backup(manuscript_root: Path, page_ids: list[str], destination: Path) -> None:
    target = destination / manuscript_root.name
    for page_id in page_ids:
        for relative in (
            Path("layout_analysis_output") / "page-xml-format" / f"{page_id}.xml",
            Path("node_corrections") / f"{page_id}.json",
        ):
            if (manuscript_root / relative).exists():
                (target / relative).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(manuscript_root / relative, target / relative)
    for relative in (Path("active_learning") / "recognition" / "registry.json", Path("active_learning") / "telemetry"):
        source = manuscript_root / relative
        if source.is_file():
            (target / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target / relative)
        elif source.is_dir():
            shutil.copytree(source, target / relative, dirs_exist_ok=True)


def revision_dirs(manuscript_root: Path, page_id: str) -> set[str]:
    folder = manuscript_root / "active_learning" / "recognition" / "revisions" / page_id
    return {path.name for path in folder.iterdir()} if folder.is_dir() else set()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=APP_ROOT / "input_manuscripts")
    parser.add_argument("--exclude-prefix", action="append", default=[])
    parser.add_argument("--exclude", action="append", default=[])
    args = parser.parse_args()
    root = args.root.resolve()

    os.chdir(APP_ROOT)  # app.py resolves its OCR checkpoint relative to app/
    import app as app_module
    from app import get_existing_text_content, update_page_text_content

    app_module.UPLOAD_FOLDER = str(root)
    client = app_module.app.test_client()

    # Plan every page first, so the backup covers exactly what is touched.
    plan, skipped = [], []
    for manuscript_root in sorted(p for p in root.iterdir() if p.is_dir()):
        if manuscript_root.name in args.exclude or any(manuscript_root.name.startswith(x) for x in args.exclude_prefix):
            continue
        commits = committed_revisions(manuscript_root)
        for xml_path in sorted((manuscript_root / "layout_analysis_output" / "page-xml-format").glob("*.xml")):
            page_id = xml_path.stem
            before = get_existing_text_content(str(xml_path))
            text, changes = {}, {}
            for line_id, value in before["text"].items():
                new_value, replacements = convert(value)
                text[line_id] = new_value
                if replacements:
                    changes[line_id] = {"before": value, "after": new_value, "replacements": replacements}
            if not changes:
                continue
            item = {"manuscript_root": manuscript_root, "page_id": page_id, "xml_path": xml_path,
                    "before": before, "text": text, "changes": changes, "commit": commits.get(page_id)}
            if item["commit"]:
                saved = committed_text(manuscript_root, page_id, item["commit"])
                current = {k: v for k, v in before["text"].items() if (v or "").strip()}
                if saved is None or saved != current:
                    skipped.append(f"{manuscript_root.name}/{page_id}: text differs from its last commit")
                    continue
            plan.append(item)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = APP_ROOT / "scripts" / "work" / f"text_conventions_{stamp}"
    by_manuscript: dict[Path, list[str]] = {}
    for item in plan:
        by_manuscript.setdefault(item["manuscript_root"], []).append(item["page_id"])
    for manuscript_root, page_ids in by_manuscript.items():
        backup(manuscript_root, page_ids, backup_dir)
    print(f"backed up {len(plan)} pages -> {backup_dir}")

    log = {"applied_at": stamp, "root": str(root), "backup": str(backup_dir), "skipped": skipped, "pages": []}
    for item in plan:
        manuscript_root, page_id = item["manuscript_root"], item["page_id"]
        manuscript = manuscript_root.name
        entry = {"manuscript": manuscript, "page_id": page_id, "changed_lines": item["changes"]}
        if item["commit"]:
            revisions_before = revision_dirs(manuscript_root, page_id)
            data = client.get(f"/semi-segment/{manuscript}/{page_id}").get_json()
            response = client.post(
                f"/semi-segment/{manuscript}/{page_id}",
                json={
                    "graph": data["graph"],
                    "baselineGraph": None,
                    "modifications": [],
                    "textlineLabels": [-1] * len(data["graph"]["nodes"]),
                    "textboxLabels": data["textbox_labels"],
                    "readingDirectionAnnotations": [],
                    "textContent": item["text"],
                    "runRecognition": False,
                    "recognitionEngine": "local",
                    "activeLearningEnabled": False,
                    "layoutActiveLearningEnabled": False,
                    "saveIntent": "commit",
                    "saveScope": "text_only",
                },
            )
            if response.status_code != 200:
                raise SystemExit(f"{manuscript}/{page_id}: commit failed {response.status_code} {response.get_json()}")
            entry["method"] = "read_mode_commit"
            entry["revisions_created"] = sorted(revision_dirs(manuscript_root, page_id) - revisions_before)
        else:
            update_page_text_content(item["xml_path"], text_content=item["text"], confidences=item["before"]["confidences"])
            entry["method"] = "text_update"
        after = get_existing_text_content(str(item["xml_path"]))["text"]
        if after != item["text"]:
            raise SystemExit(f"{manuscript}/{page_id}: the saved text is not the intended text")
        log["pages"].append(entry)
        counts = {}
        for change in item["changes"].values():
            for replacement in change["replacements"]:
                counts[replacement["rule"]] = counts.get(replacement["rule"], 0) + 1
        print(f"  {manuscript}/{page_id}: {entry['method']}, {counts}")
        for line_id, change in item["changes"].items():
            for replacement in change["replacements"]:
                if replacement["rule"] != "pipe_to_danda":
                    print(f"      line {line_id} {replacement['rule']}: …{replacement['context']}…")

    for line in skipped:
        print("SKIPPED", line)
    (backup_dir / "applied.json").write_text(json.dumps(log, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"log -> {backup_dir / 'applied.json'}")


if __name__ == "__main__":
    main()
