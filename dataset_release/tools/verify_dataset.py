"""Verify the integrity and internal consistency of the released dataset.

    python tools/verify_dataset.py --release-root .
    python tools/verify_dataset.py --release-root . --skip-checksums

    python tools/verify_dataset.py --release-root . --skip-predictions

Requires numpy and Pillow only. Exit status is 0 when every check passes, 1 otherwise.

Three independent classes of check are run.

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

3. Predictions. ``multi-modal-LLM-outputs/`` holds baseline multi-modal LLM runs.
   They are predictions, never labels, so nothing about the ground truth depends
   on them; what must hold is that each run is complete and was produced against
   the inputs actually shipped here.

     P1  every run covers exactly that manuscript's released page set
     P2  every page has request.json, result.json and prediction.xml, at the
         paths the run manifest declares, and the two agree on the status
     P3  the manifest's success/failure counts match the per-page statuses
     P4  ``image_sha256`` == the released page raster (its ``derived_sha256``
         in ``inputs/RASTER_MANIFEST.json`` where the raster is withheld)
     P5  ``pagexml_sha256`` == the released PAGE-XML, for runs that were given
         the ground-truth layout as an input
     P6  prediction.xml parses as PAGE-XML and its imageFilename, imageWidth and
         imageHeight match the released page; request.json carries no absolute
         path from the authoring machine

Manuscript trees are read from ``<release-root>/manuscripts/<manuscript>/``;
``folds/`` and ``multi-modal-LLM-outputs/`` are read from the release root.
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
PREDICTIONS_DIR = "multi-modal-LLM-outputs"

# The input that makes a run layout-conditioned rather than end-to-end. Only for
# these is the released PAGE-XML part of what the model was shown (P5).
LAYOUT_INPUT = "ground_truth_layout_traces"


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


def released_image_digest(base: Path, page_id: str) -> str | None:
    """SHA-256 of the released page raster, from the withheld-raster manifest when
    the raster itself is not shipped. Both name the same bytes."""
    raster = base / "inputs" / f"{page_id}.jpg"
    if raster.exists():
        return sha256_file(raster)
    raster_manifest = base / "inputs" / "RASTER_MANIFEST.json"
    if raster_manifest.exists():
        for entry in json.loads(raster_manifest.read_text(encoding="utf-8"))["pages"]:
            if entry["page_id"] == page_id:
                return entry["derived_sha256"]
    return None


def is_release_relative(value: str) -> bool:
    """A path that stayed inside the release: no drive letter, no root, no UNC."""
    return bool(value) and "\\" not in value and ":" not in value and not value.startswith("/")


def verify_predictions(root: Path, report: Report) -> None:
    report.section("predictions: multi-modal LLM baselines")
    predictions_root = root / PREDICTIONS_DIR
    if not predictions_root.is_dir():
        report.check(False, f"P1: {PREDICTIONS_DIR}/ is missing")
        return

    statuses: Counter = Counter()
    run_total = 0
    request_total = 0

    for manuscript in MANUSCRIPTS:
        base = root / MANUSCRIPTS_DIR / manuscript
        manuscript_dir = predictions_root / manuscript
        report.check(manuscript_dir.is_dir(), f"P1 {manuscript}: no prediction runs shipped")
        if not manuscript_dir.is_dir():
            continue

        pagexml_dir = base / "labels" / "page_xml"
        released_pages = sorted(p.stem for p in pagexml_dir.glob("*.xml"))
        page_size = {}
        for page_id in released_pages:
            released = read_page_xml(pagexml_dir / f"{page_id}.xml")
            page_size[page_id] = (released["width"], released["height"])
        image_digest = {page_id: released_image_digest(base, page_id) for page_id in released_pages}
        pagexml_digest = {
            page_id: sha256_file(pagexml_dir / f"{page_id}.xml") for page_id in released_pages
        }

        for run_dir in sorted(d for d in manuscript_dir.iterdir() if d.is_dir()):
            run = f"{manuscript}/{run_dir.name}"
            run_total += 1
            manifest_path = run_dir / "manifest.json"
            report.check(manifest_path.exists(), f"P1 {run}: manifest.json missing")
            if not manifest_path.exists():
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

            # P1 -- the run covers exactly the pages this release ships
            report.check(
                sorted(manifest.get("page_ids") or []) == released_pages,
                f"P1 {run}: manifest page_ids != the manuscript's released pages",
            )
            report.check(
                sorted(manifest.get("pages") or {}) == released_pages,
                f"P1 {run}: per-page records do not cover the released pages",
            )

            run_statuses: Counter = Counter()
            for page_id in released_pages:
                prefix = f"{run}/{page_id}"
                declared = (manifest.get("pages") or {}).get(page_id) or {}
                page_dir = run_dir / "pages" / page_id
                request_path = page_dir / "request.json"
                result_path = run_dir / declared.get("result_path", f"pages/{page_id}/result.json")
                prediction_path = run_dir / declared.get(
                    "prediction_path", f"pages/{page_id}/prediction.xml"
                )

                # P2 -- the three per-page records exist where the manifest says
                report.check(request_path.exists(), f"P2 {prefix}: request.json missing")
                report.check(result_path.exists(), f"P2 {prefix}: {result_path.name} missing")
                report.check(prediction_path.exists(), f"P2 {prefix}: prediction.xml missing")
                if not (request_path.exists() and result_path.exists() and prediction_path.exists()):
                    continue

                request = json.loads(request_path.read_text(encoding="utf-8"))
                result = json.loads(result_path.read_text(encoding="utf-8"))
                request_total += 1
                run_statuses[result["status"]] += 1
                report.check(
                    declared.get("status") == result["status"],
                    f"P2 {prefix}: manifest status {declared.get('status')!r} != result {result['status']!r}",
                )

                # P4 -- the model was shown the raster this release ships
                report.check(
                    result.get("image_sha256") == image_digest.get(page_id),
                    f"P4 {prefix}: image_sha256 != the released page raster",
                )

                # P5 -- layout-conditioned runs consumed the released PAGE-XML
                if LAYOUT_INPUT in (request.get("input_order") or []):
                    report.check(
                        result.get("pagexml_sha256") == pagexml_digest.get(page_id),
                        f"P5 {prefix}: pagexml_sha256 != the released PAGE-XML",
                    )

                # P6 -- the prediction is PAGE-XML on this page's pixel grid
                try:
                    page = ET.parse(prediction_path).getroot().find("p:Page", NS)
                except ET.ParseError as error:
                    report.check(False, f"P6 {prefix}: prediction.xml does not parse ({error})")
                    continue
                report.check(page is not None, f"P6 {prefix}: prediction.xml has no Page element")
                if page is None:
                    continue
                report.check(
                    page.get("imageFilename") == f"{page_id}.jpg",
                    f"P6 {prefix}: prediction imageFilename {page.get('imageFilename')!r}",
                )
                report.check(
                    (int(page.get("imageWidth")), int(page.get("imageHeight"))) == page_size[page_id],
                    f"P6 {prefix}: prediction page size != the released page size",
                )
                for field in ("image_path", "pagexml_path"):
                    if field in request:
                        report.check(
                            is_release_relative(request[field]),
                            f"P6 {prefix}: {field} is not release-relative ({request[field]!r})",
                        )

            # P3 -- the run manifest's tally is the tally of its own pages
            report.check(
                manifest.get("success_count") == run_statuses.get("success", 0),
                f"P3 {run}: success_count != successful pages",
            )
            report.check(
                manifest.get("failure_count") == sum(run_statuses.values()) - run_statuses.get("success", 0),
                f"P3 {run}: failure_count != failed pages",
            )
            statuses.update(run_statuses)

    report.line(
        f"{run_total} runs, {request_total} page requests, "
        f"statuses={dict(sorted(statuses.items()))}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--skip-checksums", action="store_true")
    parser.add_argument("--skip-predictions", action="store_true")
    args = parser.parse_args()

    report = Report()
    if not args.skip_checksums:
        verify_checksums(args.release_root, report)
    for manuscript in MANUSCRIPTS:
        verify_manuscript(args.release_root, manuscript, report)
    if not args.skip_predictions:
        verify_predictions(args.release_root, report)

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
