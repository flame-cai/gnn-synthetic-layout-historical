from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = REPO_ROOT / "app"
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from recognition.recognize_manuscript_text_v2_pretrained import (  # noqa: E402
    get_model_config,
    load_ocr_model,
    process_page_xml,
)


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _edit_distance(reference: str, hypothesis: str) -> int:
    previous = list(range(len(hypothesis) + 1))
    for reference_index, reference_character in enumerate(reference, start=1):
        current = [reference_index]
        for hypothesis_index, hypothesis_character in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[hypothesis_index] + 1,
                    previous[hypothesis_index - 1]
                    + (reference_character != hypothesis_character),
                )
            )
        previous = current
    return previous[-1]


def _ground_truth_by_line_id(xml_path: Path) -> dict[str, str]:
    root = ET.parse(xml_path).getroot()
    result = {}
    for line in root.iter():
        if _local_name(line.tag) != "TextLine":
            continue
        text = ""
        for descendant in line.iter():
            if _local_name(descendant.tag) == "Unicode":
                text = descendant.text or ""
                break
        result[str(line.get("id") or "")] = text
    return result


def _copy_metadata_without_reading_directions(source: Path, target: Path) -> None:
    payload = json.loads(source.read_text(encoding="utf-8"))
    for line in payload.get("line_metadata", []):
        if isinstance(line, dict):
            line["reading_direction_annotation"] = None
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _image_dirs(manuscript_root: Path) -> list[str]:
    candidates = [
        manuscript_root / "layout_analysis_output" / "images_resized",
        manuscript_root / "images_resized",
    ]
    return [str(path) for path in candidates if path.exists()]


def evaluate(manuscript_root: Path, output_root: Path, checkpoint_path: Path) -> dict:
    if output_root.exists():
        raise FileExistsError(f"Output directory already exists: {output_root}")
    pagexml_dir = manuscript_root / "layout_analysis_output" / "page-xml-format"
    output_pagexml_dir = output_root / "annotation_withheld_page_xml"
    output_pagexml_dir.mkdir(parents=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = get_model_config(str(checkpoint_path))
    model, converter = load_ocr_model(config, device)

    page_rows = []
    line_rows = []
    for source_xml in sorted(pagexml_dir.glob("*.xml")):
        page_id = source_xml.stem
        source_metadata = pagexml_dir / f"{page_id}_line_segmentation_metadata.json"
        if not source_metadata.exists():
            continue

        target_xml = output_pagexml_dir / source_xml.name
        target_metadata = output_pagexml_dir / source_metadata.name
        shutil.copy2(source_xml, target_xml)
        _copy_metadata_without_reading_directions(source_metadata, target_metadata)
        ground_truth = _ground_truth_by_line_id(source_xml)

        result = process_page_xml(
            target_xml,
            _image_dirs(manuscript_root),
            model,
            converter,
            config,
            device,
            line_segmentation_metadata_path=target_metadata,
        ) or {}
        selections = result.get("auto_orientation_selections", [])
        page_rows.append(
            {
                "page_id": page_id,
                "evaluated_line_count": len(selections),
                "rotated_line_count": sum(
                    item.get("selected_transform") == "rotate_180"
                    for item in selections
                ),
            }
        )

        for selection in selections:
            identity_text = selection["candidates"]["identity"]["text"]
            rotated_text = selection["candidates"]["rotate_180"]["text"]
            selected_text = selection["selected_text"]
            gt_text = ground_truth.get(str(selection.get("line_id") or ""), "")
            identity_distance = _edit_distance(gt_text, identity_text)
            rotated_distance = _edit_distance(gt_text, rotated_text)
            selected_distance = _edit_distance(gt_text, selected_text)
            alternate_distance = (
                rotated_distance
                if selection["selected_transform"] == "identity"
                else identity_distance
            )
            line_rows.append(
                {
                    "page_id": page_id,
                    "line_id": selection.get("line_id"),
                    "line_numeric_id": selection.get("line_numeric_id"),
                    "line_kind": selection.get("line_kind"),
                    "gt_text": gt_text,
                    "identity_text": identity_text,
                    "rotate_180_text": rotated_text,
                    "selected_text": selected_text,
                    "selected_transform": selection["selected_transform"],
                    "selection_reason": selection["reason"],
                    "identity_distance": identity_distance,
                    "rotate_180_distance": rotated_distance,
                    "selected_distance": selected_distance,
                    "selected_is_cer_oracle": selected_distance <= alternate_distance,
                    "selected_improves_identity": selected_distance < identity_distance,
                    "selected_worsens_identity": selected_distance > identity_distance,
                    "gt_char_count": len(gt_text),
                    "candidate_evidence": selection["candidates"],
                }
            )

    gt_chars = sum(row["gt_char_count"] for row in line_rows)
    identity_distance = sum(row["identity_distance"] for row in line_rows)
    rotated_distance = sum(row["rotate_180_distance"] for row in line_rows)
    selected_distance = sum(row["selected_distance"] for row in line_rows)
    kind_counts = Counter(row["line_kind"] for row in line_rows)

    by_line_kind = {}
    for line_kind in sorted(kind_counts):
        kind_rows = [row for row in line_rows if row["line_kind"] == line_kind]
        kind_gt_chars = sum(row["gt_char_count"] for row in kind_rows)
        kind_identity_distance = sum(row["identity_distance"] for row in kind_rows)
        kind_selected_distance = sum(row["selected_distance"] for row in kind_rows)
        by_line_kind[line_kind] = {
            "line_count": len(kind_rows),
            "rotated_line_count": sum(
                row["selected_transform"] == "rotate_180" for row in kind_rows
            ),
            "improved_identity_count": sum(
                row["selected_improves_identity"] for row in kind_rows
            ),
            "worsened_identity_count": sum(
                row["selected_worsens_identity"] for row in kind_rows
            ),
            "identity_cer": kind_identity_distance / max(kind_gt_chars, 1),
            "selected_cer": kind_selected_distance / max(kind_gt_chars, 1),
        }
    summary = {
        "manuscript_root": str(manuscript_root.resolve()),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "evaluation_mode": "all_reading_direction_annotations_withheld",
        "page_count": len(page_rows),
        "evaluated_line_count": len(line_rows),
        "line_kind_counts": dict(sorted(kind_counts.items())),
        "by_line_kind": by_line_kind,
        "rotated_line_count": sum(row["selected_transform"] == "rotate_180" for row in line_rows),
        "selector_cer_oracle_count": sum(row["selected_is_cer_oracle"] for row in line_rows),
        "selector_improved_identity_count": sum(row["selected_improves_identity"] for row in line_rows),
        "selector_worsened_identity_count": sum(row["selected_worsens_identity"] for row in line_rows),
        "gt_char_count": gt_chars,
        "identity_distance": identity_distance,
        "rotate_180_distance": rotated_distance,
        "selected_distance": selected_distance,
        "identity_cer": identity_distance / max(gt_chars, 1),
        "rotate_180_cer": rotated_distance / max(gt_chars, 1),
        "selected_cer": selected_distance / max(gt_chars, 1),
    }
    payload = {"summary": summary, "pages": page_rows, "lines": line_rows}
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "auto_orientation_evaluation.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate decoded-text auto-orientation on annotation-withheld curved OCR lines."
    )
    parser.add_argument("--manuscript-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=APP_ROOT / "recognition" / "pretrained_model" / "vadakautuhala.pth",
    )
    args = parser.parse_args(argv)
    payload = evaluate(args.manuscript_root, args.output_root, args.checkpoint)
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
