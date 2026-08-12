"""Verify the integrity and internal consistency of the released dataset.

    python tools/verify_dataset.py --release-root .
    python tools/verify_dataset.py --release-root . --skip-checksums

Requires numpy and Pillow only. Exit status is 0 when every check passes, 1 otherwise.

Two independent classes of check are run.

1. Integrity. Every file listed in ``CHECKSUMS.sha256`` exists and hashes to the
   recorded digest, and no unlisted file is present.

2. Consistency. The label layers must agree with each other and with the inputs.
   These are the load-bearing invariants of the dataset; if one fails, some layer
   has been edited out of step with the others.

     C1  heatmap raster size  ==  (w, h) in ``labels/graph/<page>_dims.txt``
     C2  PAGE ``imageWidth/imageHeight``  ==  2 x heatmap size
     C3  input raster size  ==  PAGE ``imageWidth/imageHeight``   (when shipped)
     C4  normalized nodes  ==  unnormalized nodes / max(dims)
     C5  the node array has one row per entry in both label files
     C6  distinct text-line labels  ==  number of PAGE TextLines
     C7  #edges - (#nodes - #components)  ==  number of closed_circular lines
     C8  ``lines.jsonl`` has one record per PAGE TextLine, in PAGE order, and
         every referenced line-crop file exists
     C9  every fold in ``folds/`` partitions that manuscript's page set

Manuscript trees are read from ``<release-root>/manuscripts/<manuscript>/``;
``folds/`` is read from the release root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

PAGE_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
NS = {"p": PAGE_NS}

MANUSCRIPTS_DIR = "manuscripts"
MANUSCRIPTS = ("moderate_layout", "dense_layout", "circular_layout")


class Report:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.checks = 0

    def check(self, condition: bool, message: str) -> None:
        self.checks += 1
        if not condition:
            self.failures.append(message)

    def section(self, title: str) -> None:
        print(f"\n== {title}")

    def line(self, message: str) -> None:
        print(f"   {message}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_checksums(root: Path, report: Report) -> None:
    report.section("integrity")
    manifest_path = root / "CHECKSUMS.sha256"
    if not manifest_path.exists():
        report.check(False, "CHECKSUMS.sha256 is missing")
        return

    expected = {}
    for raw in manifest_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        digest, rel = raw.split("  ", 1)
        expected[rel] = digest

    present = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.as_posix()
        and path.name != "CHECKSUMS.sha256"
    }

    missing = sorted(set(expected) - present)
    unlisted = sorted(present - set(expected))
    report.check(not missing, f"{len(missing)} listed file(s) missing, e.g. {missing[:3]}")
    report.check(not unlisted, f"{len(unlisted)} unlisted file(s) present, e.g. {unlisted[:3]}")

    mismatched = [
        rel for rel, digest in expected.items()
        if (root / rel).exists() and sha256_file(root / rel) != digest
    ]
    report.check(not mismatched, f"{len(mismatched)} checksum mismatch(es), e.g. {mismatched[:3]}")
    report.line(f"{len(expected)} files listed, {len(mismatched)} mismatched, {len(missing)} missing")


def read_page_xml(path: Path) -> dict:
    root = ET.parse(path).getroot()
    page = root.find("p:Page", NS)
    line_ids = [
        line.get("id")
        for region in page.findall("p:TextRegion", NS)
        for line in region.findall("p:TextLine", NS)
    ]
    return {
        "width": int(page.get("imageWidth")),
        "height": int(page.get("imageHeight")),
        "line_ids": line_ids,
    }


def verify_manuscript(root: Path, manuscript: str, report: Report) -> None:
    report.section(f"consistency: {manuscript}")
    base = root / MANUSCRIPTS_DIR / manuscript
    labels = base / "labels"
    pagexml_dir = labels / "page_xml"
    graph_dir = labels / "graph"

    page_ids = sorted(p.stem for p in pagexml_dir.glob("*.xml"))
    report.check(bool(page_ids), f"{manuscript}: no PAGE-XML pages found")

    index = [json.loads(l) for l in (base / "lines.jsonl").read_text(encoding="utf-8").splitlines()]
    index_by_page: dict[str, list[dict]] = {}
    for record in index:
        index_by_page.setdefault(record["page_id"], []).append(record)

    rasters_shipped = any((base / "inputs").glob("*.jpg"))

    for page_id in page_ids:
        prefix = f"{manuscript}/{page_id}"
        dims = np.loadtxt(graph_dir / f"{page_id}_dims.txt")
        heatmap_size = Image.open(base / "heatmaps" / f"{page_id}.jpg").size
        page = read_page_xml(pagexml_dir / f"{page_id}.xml")

        # C1 / C2 / C3 -- one coordinate story across three rasters
        report.check(
            heatmap_size == (int(dims[0]), int(dims[1])),
            f"C1 {prefix}: heatmap {heatmap_size} != dims.txt {tuple(dims)}",
        )
        report.check(
            (page["width"], page["height"]) == (int(dims[0] * 2), int(dims[1] * 2)),
            f"C2 {prefix}: PAGE ({page['width']},{page['height']}) != 2x heatmap {heatmap_size}",
        )
        if rasters_shipped:
            raster = base / "inputs" / f"{page_id}.jpg"
            report.check(raster.exists(), f"C3 {prefix}: input raster missing")
            if raster.exists():
                report.check(
                    Image.open(raster).size == (page["width"], page["height"]),
                    f"C3 {prefix}: input raster size != PAGE imageWidth/imageHeight",
                )

        # C4 -- isotropic normalisation by the longest heatmap side
        unnormalized = np.atleast_2d(np.loadtxt(graph_dir / f"{page_id}_inputs_unnormalized.txt"))
        normalized = np.atleast_2d(np.loadtxt(graph_dir / f"{page_id}_inputs_normalized.txt"))
        report.check(
            np.allclose(normalized[:, :2], unnormalized[:, :2] / max(dims), atol=1e-5),
            f"C4 {prefix}: normalized nodes are not unnormalized / max(dims)",
        )

        # C5 -- graph arrays are aligned
        textline_labels = [int(v) for v in (graph_dir / f"{page_id}_labels_textline.txt").read_text().split()]
        textbox_labels = [int(v) for v in (graph_dir / f"{page_id}_labels_textbox.txt").read_text().split()]
        report.check(
            len(unnormalized) == len(textline_labels) == len(textbox_labels),
            f"C5 {prefix}: node/label array lengths disagree",
        )

        edges = [ln for ln in (graph_dir / f"{page_id}_edges.txt").read_text().splitlines() if ln.strip()]
        component_count = len(set(textline_labels))

        # C6 -- one connected component per PAGE TextLine
        report.check(
            component_count == len(page["line_ids"]),
            f"C6 {prefix}: {component_count} graph components != {len(page['line_ids'])} TextLines",
        )

        # C7 -- surplus edges are exactly the closed circular lines
        geometry_path = labels / "line_geometry" / f"{page_id}.json"
        closed = 0
        if geometry_path.exists():
            geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
            closed = sum(1 for item in geometry.get("lines", []) if item.get("line_kind") == "closed_circular")
        surplus = len(edges) - (len(unnormalized) - component_count)
        report.check(
            surplus == closed,
            f"C7 {prefix}: {surplus} surplus edge(s) but {closed} closed_circular line(s)",
        )

        # C8 -- flat index agrees with PAGE-XML, and crops exist
        records = index_by_page.get(page_id, [])
        report.check(
            [r["line_id"] for r in records] == page["line_ids"],
            f"C8 {prefix}: lines.jsonl line ids/order differ from PAGE-XML",
        )
        for record in records:
            if record.get("line_image"):
                report.check(
                    (base / record["line_image"]).exists(),
                    f"C8 {prefix}: missing line crop {record['line_image']}",
                )

    kinds = Counter(r["line_kind"] for r in index)
    report.line(
        f"{len(page_ids)} pages, {len(index)} text lines, kinds={dict(sorted(kinds.items()))}, "
        f"page rasters={'shipped' if rasters_shipped else 'withheld'}"
    )

    # C9 -- folds partition the page set
    folds_path = root / "folds" / f"{manuscript}.json"
    report.check(folds_path.exists(), f"C9 {manuscript}: folds file missing")
    if folds_path.exists():
        payload = json.loads(folds_path.read_text(encoding="utf-8"))
        report.check(
            sorted(payload["page_ids"]) == page_ids,
            f"C9 {manuscript}: folds page set != discovered pages",
        )
        for fold in payload["folds"]:
            train, test = set(fold["train_page_ids"]), set(fold["test_page_ids"])
            report.check(
                not (train & test) and sorted(train | test) == page_ids,
                f"C9 {manuscript}/{fold['fold_id']}: train/test do not partition the page set",
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--skip-checksums", action="store_true")
    args = parser.parse_args()

    report = Report()
    if not args.skip_checksums:
        verify_checksums(args.release_root, report)
    for manuscript in MANUSCRIPTS:
        verify_manuscript(args.release_root, manuscript, report)

    print()
    if report.failures:
        print(f"FAILED: {len(report.failures)} of {report.checks} checks")
        for failure in report.failures[:40]:
            print(f"  - {failure}")
        return 1
    print(f"OK: {report.checks} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
