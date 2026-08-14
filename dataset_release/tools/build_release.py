"""Build the redistributable manuscript-layout dataset from the authoring tree.

This script is the sole provenance of everything under the release root. It is
shipped with the dataset so that every inclusion, exclusion and transformation is
auditable.

Usage:

    python tools/build_release.py \
        --source-root /path/to/repo/app/input_manuscripts \
        --release-root /path/to/repo/dataset_release

Layout produced:

    <release-root>/
        DATASET.md, SOURCES.bib, dataset_manifest.json, CHECKSUMS.sha256
        folds/<manuscript>.json
        tools/
        manuscripts/<manuscript>/
            SOURCE.md, processing_settings.json, pages.csv, lines.jsonl
            inputs/                 page images (or the withheld-raster notice)
            heatmaps/               CRAFT region-score maps, half scale
            labels/graph/           corrected layout graph
            labels/page_xml/        PAGE-XML: Coords + Baseline + Unicode
            labels/page_xml_baselines/  PAGE-XML: Baseline only
            labels/line_geometry/   per-line kind, topology, reading direction
            labels/line_images/     rectified line crops

The release is a *labels* dataset. It contains ground truth and the inputs the
ground truth is defined on. It deliberately does not contain machine
pre-correction state, correction deltas, or annotation timing.

EXCLUDED from the authoring tree, and why:

  * ``images/``                     pre-resize scans; ``inputs/`` is the raster
                                    every released coordinate is defined on.
  * ``layout_analysis_output/images_resized/``
                                    byte-identical duplicate of ``inputs/``.
  * ``layout_analysis_output/text_recovery_backups/``
                                    editor undo snapshots; no scientific content.
  * ``gnn-dataset/``                raw pre-correction CRAFT node proposals. Not
                                    ground truth. Its ``_dims.txt`` is duplicated
                                    byte-for-byte inside ``labels/graph/``.
  * ``node_corrections/``           per-page counts of what the annotator added
                                    and removed: a correction delta, not a label.
  * ``layout_analysis_output/layout_effort.json``
                                    annotation timing and edit telemetry.
  * per-line polygon-builder telemetry inside the line-segmentation metadata,
    including absolute paths from the authoring machine.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import re
import shutil
import sys
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

PAGE_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
NS = {"p": PAGE_NS}

# Directory names in the release. Authoring names are on the right.
MANUSCRIPTS_DIR = "manuscripts"
INPUTS_DIR = "inputs"                       # images_resized
HEATMAPS_DIR = "heatmaps"                   # heatmaps
LABELS_DIR = "labels"                       # layout_analysis_output
GRAPH_DIR = "graph"                         # gnn-format
PAGEXML_DIR = "page_xml"                    # page-xml-format
BASELINES_DIR = "page_xml_baselines"        # _baseline_page_xml
GEOMETRY_DIR = "line_geometry"              # *_line_segmentation_metadata.json
LINE_IMAGES_DIR = "line_images"             # image-format

# authoring id -> (released id, redistribute page rasters?)
MANUSCRIPTS = [
    ("yajn", "moderate_layout", False),
    ("dense", "dense_layout", True),
    ("circle_new", "circular_layout", True),
]

# Bibliographic source of each manuscript. `rights_basis` is why the page rasters
# are or are not redistributed; it is the load-bearing field, not decoration.
SOURCES = {
    "moderate_layout": {
        "bibtex_key": "Yajna",
        "work": "Yājñavalkyasmṛtiḥ (Ācārādhyāyaḥ)",
        "work_ascii": "Yajnavalakyasmritih (Acharadhyayah)",
        "author": "Unknown",
        "date": "Unknown",
        "holding_institution": "Lalchand Research Library, DAV College, Chandigarh, India",
        "url": "https://dav.splrarebooks.com/collection/view/yajnavalakyasmritih-acharadhyayah",
        "genre": "dharmaśāstra",
        "rights_basis": "Copyright retained by the holding institution; research use permitted, redistribution not permitted.",
        "page_rasters_redistributed": False,
        "pages_used": 15,
    },
    "dense_layout": {
        "bibtex_key": "Muhurta",
        "work": "Muhūrta Mārtaṇḍa",
        "work_ascii": "Muhurta Martanda (841 Gha Alm 4 Shlf 5 Devanagari Jyotish)",
        "author": "Unknown",
        "date": "Unknown",
        "holding_institution": "eGangotri Digital Preservation Trust (Dharmarth Trust J&K collection)",
        "url": "https://archive.org/details/MuhurtaMartanda841GhaAlm4Shlf5DevanagariJyotish",
        "archive_identifier": "MuhurtaMartanda841GhaAlm4Shlf5DevanagariJyotish",
        "genre": "jyotiṣa",
        "rights_basis": "CC0 1.0 Universal Public Domain Dedication (http://creativecommons.org/publicdomain/zero/1.0/), as recorded in the item metadata.",
        "page_rasters_redistributed": True,
        "pages_used": 7,
    },
    "circular_layout": {
        "bibtex_key": "Tantra",
        "work": "Tantrarāja, with Yantra and Mantra Uddhāra",
        "work_ascii": "Tantra Raj With Yantra And Mantra Uddhara (5890 1430 Ka Almira 26 Shlf 3 Devanagari Stotr)",
        "author": "Unknown",
        "date": "Unknown",
        "holding_institution": "eGangotri Digital Preservation Trust (Dharmarth Trust J&K collection)",
        "url": (
            "https://archive.org/details/"
            "TantraRajWithYantraAndMantraUddhara58901430KaAlmira26Shlf3DevanagariStotr"
        ),
        "archive_identifier": "TantraRajWithYantraAndMantraUddhara58901430KaAlmira26Shlf3DevanagariStotr",
        "genre": "tantra",
        "rights_basis": "CC0 1.0 Universal Public Domain Dedication (http://creativecommons.org/publicdomain/zero/1.0/), as recorded in the item metadata.",
        "page_rasters_redistributed": True,
        "pages_used": 9,
    },
}

SOURCES_BIB = r"""% Source manuscripts of the Sanskrit Manuscript Layout Regimes dataset.
% One entry per released manuscript directory:
%   Yajna   -> manuscripts/moderate_layout   (page rasters withheld; see DATASET.md section 7)
%   Muhurta -> manuscripts/dense_layout
%   Tantra  -> manuscripts/circular_layout
% Cite the manuscript whose pages you used, alongside the dataset itself.

@misc{Yajna,
  title        = "{Yajnavalakyasmritih (Acharadhyayah)}",
  author       = {Unknown},
  year         = {Unknown},
  howpublished = "{Lalchand Research Library, DAV College, Chandigarh, India.\\
                   Accessed online at: \url{https://dav.splrarebooks.com/collection/view/yajnavalakyasmritih-acharadhyayah}}"
}

@misc{Muhurta,
  title        = "{Muhurta Martanda (841 Gha Alm 4 Shlf 5 Devanagari Jyotish)}",
  author       = {Unknown},
  year         = {Unknown},
  howpublished = "{eGangotri Digital Preservation Trust.\\
                   Accessed online at: \url{https://archive.org/details/MuhurtaMartanda841GhaAlm4Shlf5DevanagariJyotish/mode/2up}}"
}

@misc{Tantra,
  title        = "{Tantra Raj With Yantra And Mantra Uddhara 5890 1430 Ka Almira 26 Shlf 3 Devanagari Stotr}",
  author       = {Unknown},
  year         = {Unknown},
  howpublished = "{eGangotri Digital Preservation Trust.\\
                   Accessed online at: \url{https://archive.org/details/TantraRajWithYantraAndMantraUddhara58901430KaAlmira26Shlf3DevanagariStotr/page/n45/mode/2up}}"
}
"""

# Per-line fields retained in labels/line_geometry/<page>.json. Everything else in
# the authoring file is polygon-builder diagnostics.
KEEP_LINE_KEYS = (
    "line_id",
    "line_custom",
    "line_numeric_id",
    "line_kind",
    "topology",
    "reading_direction_annotation",
    "crop_model",
    "local_s_min",
    "local_s_max",
    "local_n_min",
    "local_n_max",
    "line_half_width_px",
    "local_canvas_width_px",
    "local_canvas_height_px",
    "page_polygon_point_count",
    "fallback_used",
    "fallback_reason",
)

# Reading-direction fields retained. Dropped: annotation_id, frontend_line_id,
# source, status, updated_at, component_overlap_ratio -- editor bookkeeping.
KEEP_READING_DIRECTION_KEYS = (
    "resolved_line_numeric_id",
    "reading_direction",
    "cut_start",
    "cut_end",
    "cut_midpoint",
    "component_node_indices",
)

KEEP_GEOMETRY_SUMMARY_KEYS = (
    "geometry_source",
    "line_segmentation_strategy_name",
    "prepared_line_count",
    "source_text_line_count",
    "crop_model_counts",
    "topology_counts",
    "normalization_action_counts",
    "orientation_policy",
)

DEVANAGARI_RANGE = (0x0900, 0x097F)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def points_from(attr: str | None) -> list[list[int]]:
    if not attr:
        return []
    out = []
    for token in attr.split():
        if "," not in token:
            continue
        x_str, y_str = token.split(",", 1)
        try:
            out.append([int(round(float(x_str))), int(round(float(y_str)))])
        except ValueError:
            continue
    return out


def numeric_line_id(custom: str | None) -> int | None:
    if not custom:
        return None
    match = re.search(r"structure_line_id_(\d+)", custom)
    return int(match.group(1)) if match else None


def textbox_label(custom: str | None) -> int | None:
    if not custom:
        return None
    match = re.search(r"textbox_label_(-?\d+)", custom)
    return int(match.group(1)) if match else None


def devanagari_char_count(text: str) -> int:
    return sum(1 for ch in text if DEVANAGARI_RANGE[0] <= ord(ch) <= DEVANAGARI_RANGE[1])


# --------------------------------------------------------------------------
# folds (identical to experiments/downstream_ocr/splits.py)
# --------------------------------------------------------------------------


def make_three_folds(page_ids, *, train_size=3, fold_count=5, seed=42):
    ordered = tuple(sorted(page_ids))
    rng = random.Random(int(seed))
    folds = []
    for fold_index in range(fold_count):
        train = tuple(rng.sample(ordered, train_size))
        test = tuple(pid for pid in ordered if pid not in set(train))
        folds.append(
            {
                "fold_id": f"fold_{fold_index + 1}",
                "train_page_ids": list(train),
                "test_page_ids": list(test),
            }
        )
    return folds


# --------------------------------------------------------------------------
# per-manuscript build
# --------------------------------------------------------------------------


def slim_reading_direction(annotation) -> dict | None:
    if not isinstance(annotation, dict) or not annotation.get("reading_direction"):
        return None
    return {key: annotation[key] for key in KEEP_READING_DIRECTION_KEYS if key in annotation}


def slim_line_geometry(payload: dict, page_id: str, reading_direction_model: str | None) -> dict:
    geometry_summary = payload.get("geometry_summary") or {}
    lines = []
    for item in payload.get("line_metadata") or []:
        record = {key: item[key] for key in KEEP_LINE_KEYS if key in item}
        record["reading_direction_annotation"] = slim_reading_direction(
            item.get("reading_direction_annotation")
        )
        lines.append(record)
    return {
        "schema": "line_geometry-1",
        "page_id": page_id,
        "strategy_name": payload.get("strategy_name"),
        "reading_direction_model": reading_direction_model,
        "line_count": payload.get("line_count"),
        "geometry_summary": {
            key: geometry_summary[key]
            for key in KEEP_GEOMETRY_SUMMARY_KEYS
            if key in geometry_summary
        },
        "lines": lines,
    }


def write_source_card(dst: Path, release_id: str) -> None:
    source = SOURCES[release_id]
    rasters = (
        f"shipped in `{INPUTS_DIR}/`"
        if source["page_rasters_redistributed"]
        else f"**withheld**; see `{INPUTS_DIR}/DOWNLOAD.md` and `DATASET.md` section 7"
    )
    lines = [
        f"# Source of `{release_id}`",
        "",
        f"**{source['work']}** — *{source['work_ascii']}*",
        "",
        "| | |",
        "| --- | --- |",
        f"| Genre | {source['genre']} |",
        f"| Author | {source['author']} |",
        f"| Date | {source['date']} |",
        f"| Holding institution | {source['holding_institution']} |",
        f"| Source | <{source['url']}> |",
        f"| Pages in this release | {source['pages_used']} |",
        f"| Page rasters | {rasters} |",
        f"| Rights basis | {source['rights_basis']} |",
        "",
        f"Cite as `\\cite{{{source['bibtex_key']}}}`; the entry is in `SOURCES.bib` at the",
        "release root. Cite the manuscript alongside the dataset, not instead of it.",
        "",
    ]
    (dst / "SOURCE.md").write_text("\n".join(lines), encoding="utf-8")


def build_raster_manifest(src: Path, release_id: str, page_ids: list[str]) -> dict:
    """Record enough to verify a re-acquired raster without redistributing it."""
    from PIL import Image, __version__ as pillow_version

    entries = []
    for page_id in page_ids:
        source_path = src / "images" / f"{page_id}.jpg"
        derived_path = src / "images_resized" / f"{page_id}.jpg"
        with Image.open(derived_path) as image:
            width, height = image.size
        entries.append(
            {
                "page_id": page_id,
                "filename": f"{page_id}.jpg",
                "width": width,
                "height": height,
                "source_sha256": sha256_file(source_path),
                "source_bytes": source_path.stat().st_size,
                "derived_sha256": sha256_file(derived_path),
                "derived_bytes": derived_path.stat().st_size,
            }
        )
    return {
        "schema": "withheld_raster_manifest-1",
        "manuscript_id": release_id,
        "reason": "page rasters are under third-party copyright and are not redistributed",
        "target_longest_side": 3500,
        "derivation": (
            "PIL open -> LANCZOS downscale only if max(w, h) > 3500 -> convert to RGB "
            "if mode in {RGBA, P, LA} -> save as JPEG at Pillow's default quality"
        ),
        "reference_pillow_version": pillow_version,
        "pages": entries,
    }


def parse_page_xml(xml_path: Path) -> dict:
    root = ET.parse(xml_path).getroot()
    page = root.find("p:Page", NS)
    record = {
        "image_filename": page.get("imageFilename"),
        "image_width": int(page.get("imageWidth")),
        "image_height": int(page.get("imageHeight")),
        "regions": [],
    }
    for region_index, region in enumerate(page.findall("p:TextRegion", NS)):
        region_record = {
            "region_id": region.get("id"),
            "region_index": region_index,
            "region_custom": region.get("custom"),
            "textbox_label": textbox_label(region.get("custom")),
            "lines": [],
        }
        for line in region.findall("p:TextLine", NS):
            coords = line.find("p:Coords", NS)
            baseline = line.find("p:Baseline", NS)
            unicode_node = line.find("p:TextEquiv/p:Unicode", NS)
            text = unicode_node.text if unicode_node is not None and unicode_node.text else ""
            region_record["lines"].append(
                {
                    "line_id": line.get("id"),
                    "line_custom": line.get("custom"),
                    "line_numeric_id": numeric_line_id(line.get("custom")),
                    "polygon_points": points_from(coords.get("points") if coords is not None else None),
                    "baseline_points": points_from(baseline.get("points") if baseline is not None else None),
                    "text": text,
                }
            )
        record["regions"].append(region_record)
    return record


def build_manuscript(
    source_root: Path,
    release_root: Path,
    source_id: str,
    release_id: str,
    include_rasters: bool,
) -> dict:
    src = source_root / source_id
    dst = release_root / MANUSCRIPTS_DIR / release_id
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    src_lao = src / "layout_analysis_output"
    dst_labels = dst / LABELS_DIR
    src_pagexml = src_lao / "page-xml-format"

    page_ids = sorted(path.stem for path in src_pagexml.glob("*.xml"))

    # ---- inputs ----------------------------------------------------------
    shutil.copytree(src / "heatmaps", dst / HEATMAPS_DIR)
    shutil.copy2(src / "processing_settings.json", dst / "processing_settings.json")
    if include_rasters:
        shutil.copytree(src / "images_resized", dst / INPUTS_DIR)
    else:
        (dst / INPUTS_DIR).mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / "images" / "DOWNLOAD.md", dst / INPUTS_DIR / "DOWNLOAD.md")
        write_json(
            dst / INPUTS_DIR / "RASTER_MANIFEST.json",
            build_raster_manifest(src, release_id, page_ids),
        )

    # ---- labels ----------------------------------------------------------
    shutil.copytree(src_lao / "gnn-format", dst_labels / GRAPH_DIR)
    shutil.copytree(src_lao / "_baseline_page_xml", dst_labels / BASELINES_DIR)
    if include_rasters:
        shutil.copytree(src_lao / "image-format", dst_labels / LINE_IMAGES_DIR)

    (dst_labels / PAGEXML_DIR).mkdir(parents=True, exist_ok=True)
    for page_id in page_ids:
        shutil.copy2(src_pagexml / f"{page_id}.xml", dst_labels / PAGEXML_DIR / f"{page_id}.xml")

        reading_direction_model = None
        rd_path = src_pagexml / f"{page_id}_reading_direction_metadata.json"
        if rd_path.exists():
            reading_direction_model = json.loads(rd_path.read_text(encoding="utf-8")).get(
                "annotation_model"
            )
        ls_path = src_pagexml / f"{page_id}_line_segmentation_metadata.json"
        if ls_path.exists():
            payload = json.loads(ls_path.read_text(encoding="utf-8"))
            write_json(
                dst_labels / GEOMETRY_DIR / f"{page_id}.json",
                slim_line_geometry(payload, page_id, reading_direction_model),
            )

    write_source_card(dst, release_id)

    stats = build_indexes(dst, release_id, page_ids, include_rasters)

    write_json(
        release_root / "folds" / f"{release_id}.json",
        {
            "schema_version": 1,
            "manuscript_id": release_id,
            "split_seed": 42,
            "train_size": 3,
            "fold_count": 5,
            "page_ids": list(page_ids),
            "folds": make_three_folds(page_ids),
        },
    )
    return stats


def build_indexes(
    dst: Path, release_id: str, page_ids: list[str], include_rasters: bool
) -> dict:
    labels = dst / LABELS_DIR
    records: list[dict] = []

    line_kinds = Counter()
    reading_annotations = 0
    total_chars = 0
    total_devanagari = 0
    empty_text_lines = 0
    region_count = 0
    node_total = 0
    edge_total = 0
    component_total = 0
    page_rows = []

    for page_id in page_ids:
        page = parse_page_xml(labels / PAGEXML_DIR / f"{page_id}.xml")

        geometry_path = labels / GEOMETRY_DIR / f"{page_id}.json"
        geometry_by_id = {}
        if geometry_path.exists():
            for item in json.loads(geometry_path.read_text(encoding="utf-8")).get("lines", []):
                if item.get("line_numeric_id") is not None:
                    geometry_by_id[int(item["line_numeric_id"])] = item

        graph = labels / GRAPH_DIR
        textline_labels = [int(v) for v in (graph / f"{page_id}_labels_textline.txt").read_text().split()]
        edges = [ln for ln in (graph / f"{page_id}_edges.txt").read_text().splitlines() if ln.strip()]
        node_count = len(textline_labels)
        component_count = len(set(textline_labels))
        node_total += node_count
        edge_total += len(edges)
        component_total += component_count

        page_kinds = Counter()
        page_chars = 0
        page_lines = 0

        for region in page["regions"]:
            region_count += 1
            for line in region["lines"]:
                page_lines += 1
                numeric_id = line["line_numeric_id"]
                geometry = geometry_by_id.get(numeric_id, {})
                topology = geometry.get("topology") or {}
                kind = geometry.get("line_kind") or topology.get("line_kind")
                annotation = geometry.get("reading_direction_annotation") or {}
                text = line["text"]

                if kind:
                    line_kinds[kind] += 1
                    page_kinds[kind] += 1
                if annotation.get("reading_direction"):
                    reading_annotations += 1
                if not text.strip():
                    empty_text_lines += 1

                normalized = unicodedata.normalize("NFC", text)
                total_chars += len(normalized)
                page_chars += len(normalized)
                total_devanagari += devanagari_char_count(normalized)

                line_image = None
                if include_rasters and region["region_custom"] and numeric_id is not None:
                    candidate = (
                        Path(LABELS_DIR)
                        / LINE_IMAGES_DIR
                        / page_id
                        / region["region_custom"]
                        / f"line_{numeric_id}.jpg"
                    )
                    if (dst / candidate).exists():
                        line_image = candidate.as_posix()

                records.append(
                    {
                        "manuscript_id": release_id,
                        "page_id": page_id,
                        "image_width": page["image_width"],
                        "image_height": page["image_height"],
                        "region_id": region["region_id"],
                        "region_index": region["region_index"],
                        "region_custom": region["region_custom"],
                        "textbox_label": region["textbox_label"],
                        "line_id": line["line_id"],
                        "line_custom": line["line_custom"],
                        "line_numeric_id": numeric_id,
                        "line_kind": kind,
                        "is_closed": bool(topology.get("is_closed", False)),
                        "baseline_length_px": topology.get("baseline_length"),
                        "baseline_point_count": len(line["baseline_points"]),
                        "polygon_point_count": len(line["polygon_points"]),
                        "reading_direction": annotation.get("reading_direction"),
                        "reading_cut_midpoint": annotation.get("cut_midpoint"),
                        "orientation_action": topology.get("orientation_action"),
                        "text": text,
                        "text_length_nfc": len(normalized),
                        "line_image": line_image,
                    }
                )

        page_rows.append(
            {
                "manuscript_id": release_id,
                "page_id": page_id,
                "image_width": page["image_width"],
                "image_height": page["image_height"],
                "region_count": len(page["regions"]),
                "text_line_count": page_lines,
                "node_count": node_count,
                "edge_count": len(edges),
                "graph_component_count": component_count,
                "cycle_edge_count": len(edges) - (node_count - component_count),
                "text_chars_nfc": page_chars,
                "line_kinds": dict(sorted(page_kinds.items())),
            }
        )

    with (dst / "lines.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    with (dst / "pages.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [k for k in page_rows[0] if k != "line_kinds"] + ["line_kinds"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in page_rows:
            out = dict(row)
            out["line_kinds"] = json.dumps(out["line_kinds"], ensure_ascii=False)
            writer.writerow(out)

    return {
        "manuscript_id": release_id,
        "source": SOURCES[release_id],
        "page_count": len(page_ids),
        "page_ids": list(page_ids),
        "page_rasters_included": include_rasters,
        "text_region_count": region_count,
        "text_line_count": len(records),
        "line_kind_counts": dict(sorted(line_kinds.items())),
        "reading_direction_annotation_count": reading_annotations,
        "empty_transcription_line_count": empty_text_lines,
        "transcribed_chars_nfc": total_chars,
        "transcribed_devanagari_chars": total_devanagari,
        "node_count": node_total,
        "edge_count": edge_total,
        "graph_component_count": component_total,
        "cycle_edge_count": edge_total - (node_total - component_total),
    }


# --------------------------------------------------------------------------
# release-level artifacts
# --------------------------------------------------------------------------


def write_checksums(release_root: Path) -> int:
    entries = []
    for path in sorted(release_root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(release_root).as_posix()
        if rel == "CHECKSUMS.sha256" or "__pycache__" in rel:
            continue
        entries.append((sha256_file(path), rel))
    buffer = io.StringIO()
    for digest, rel in entries:
        buffer.write(f"{digest}  {rel}\n")
    (release_root / "CHECKSUMS.sha256").write_text(buffer.getvalue(), encoding="utf-8")
    return len(entries)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--release-root", required=True, type=Path)
    args = parser.parse_args()

    release_root: Path = args.release_root
    release_root.mkdir(parents=True, exist_ok=True)

    stats = []
    for source_id, release_id, include_rasters in MANUSCRIPTS:
        print(f"[build] {source_id} -> {release_id}", flush=True)
        stats.append(
            build_manuscript(args.source_root, release_root, source_id, release_id, include_rasters)
        )

    manifest = {
        "schema": "manuscript_layout_dataset/manifest-2",
        "dataset_name": "Sanskrit Manuscript Layout Regimes",
        "manuscripts": stats,
        "totals": {
            "manuscript_count": len(stats),
            "page_count": sum(s["page_count"] for s in stats),
            "text_region_count": sum(s["text_region_count"] for s in stats),
            "text_line_count": sum(s["text_line_count"] for s in stats),
            "transcribed_chars_nfc": sum(s["transcribed_chars_nfc"] for s in stats),
            "transcribed_devanagari_chars": sum(s["transcribed_devanagari_chars"] for s in stats),
            "node_count": sum(s["node_count"] for s in stats),
            "edge_count": sum(s["edge_count"] for s in stats),
            "cycle_edge_count": sum(s["cycle_edge_count"] for s in stats),
            "reading_direction_annotation_count": sum(
                s["reading_direction_annotation_count"] for s in stats
            ),
        },
        "split_protocol": {
            "seed": 42,
            "train_size": 3,
            "fold_count": 5,
            "sampling": "random.Random(42).sample over sorted page ids, five draws from one stream",
            "note": "Folds are resampled draws, not a partition: training sets overlap and pages recur across test sets.",
        },
    }
    write_json(release_root / "dataset_manifest.json", manifest)
    (release_root / "SOURCES.bib").write_text(SOURCES_BIB, encoding="utf-8")

    count = write_checksums(release_root)
    print(f"[build] wrote CHECKSUMS.sha256 with {count} entries", flush=True)
    print(json.dumps(manifest["totals"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
