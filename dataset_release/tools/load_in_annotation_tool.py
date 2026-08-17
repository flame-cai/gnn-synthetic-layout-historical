"""Rebuild a released manuscript into the layout the annotation tool loads.

The release and the annotation tool hold the same content in different shapes.
The release renames the directories for legibility (DATASET.md section 3.2) and
drops four things the tool needs:

  * ``gnn-dataset/<page>_dims.txt``  -- the tool lists a manuscript's pages from
    this directory, so without it the manuscript does not appear at all.  The
    release documents these files as byte-identical to the copies inside
    ``labels/graph/``, which is where this script takes them from.
  * ``<page>_reading_direction_metadata.json`` -- the release folds each line's
    reading direction into ``labels/line_geometry/``.  The tool reads it from a
    sidecar beside the PAGE-XML, so the sidecar is rebuilt here.
  * ``<page>_line_segmentation_metadata.json`` -- same content as
    ``labels/line_geometry/<page>.json``, under the name the tool looks for.
  * the pre-resize scans in ``images/`` -- not needed.  Every released
    coordinate is defined on ``inputs/``, which becomes ``images_resized/``.

Nothing is recomputed and no model runs: this is a copy, a rename, and one
sidecar rebuilt from fields the release already carries.

    python dataset_release/tools/load_in_annotation_tool.py
    python dataset_release/tools/load_in_annotation_tool.py --manuscript circular_layout
    python dataset_release/tools/load_in_annotation_tool.py --check

Then start the backend and the frontend and pick the manuscript from the list.

The released tree is never written to -- edits land only in the destination.
Saving a page re-derives its polygons and crops with the current production
line-segmentation strategy; on ``circular_layout`` page 11 a save with no edits
reproduced all 24 polygons and all 24 transcriptions byte-for-byte, and moved one
baseline point by one pixel.  That pixel is the tool's own round trip, not this
script: node positions are reloaded from ``_inputs_normalized.txt``, whose six
decimal places put ``710.0`` back as ``1419.999``, which truncates to 1419.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


RELEASE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = RELEASE_ROOT.parent
DEFAULT_DESTINATION = REPO_ROOT / "app" / "input_manuscripts"

# release directory -> tool directory, relative to the manuscript root
LABEL_DIRECTORIES = {
    "labels/graph": "layout_analysis_output/gnn-format",
    "labels/page_xml": "layout_analysis_output/page-xml-format",
    "labels/page_xml_baselines": "layout_analysis_output/_baseline_page_xml",
    "labels/line_images": "layout_analysis_output/image-format",
}


def released_manuscripts() -> list[str]:
    manuscripts_dir = RELEASE_ROOT / "manuscripts"
    return sorted(path.name for path in manuscripts_dir.iterdir() if path.is_dir())


def page_ids_for(manuscript_dir: Path) -> list[str]:
    graph_dir = manuscript_dir / "labels" / "graph"
    return sorted(path.name[: -len("_dims.txt")] for path in graph_dir.glob("*_dims.txt"))


def copy_tree(source: Path, destination: Path) -> int:
    """Copy a directory's contents, leaving anything already present alone."""
    if not source.exists():
        return 0
    destination.mkdir(parents=True, exist_ok=True)
    copied = 0
    for path in sorted(source.rglob("*")):
        if path.is_dir():
            continue
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied += 1
    return copied


def reading_direction_sidecar(line_geometry: dict, page_id: str) -> dict:
    """Rebuild the tool's reading-direction sidecar from the released line geometry.

    The release keeps every field the tool actually reads -- the cut endpoints,
    the tangent and the component node indices -- and drops only editor
    bookkeeping.  Of that bookkeeping, ``annotation_id`` and ``frontend_line_id``
    are the line's own numeric id in every released annotation, and the rest the
    tool defaults or recomputes on the next save.
    """
    annotations = []
    for line in line_geometry.get("lines", []):
        annotation = line.get("reading_direction_annotation")
        if not annotation:
            continue
        line_numeric_id = int(annotation["resolved_line_numeric_id"])
        annotations.append(
            {
                "annotation_id": str(line_numeric_id),
                "frontend_line_id": str(line_numeric_id),
                "component_node_indices": list(annotation.get("component_node_indices") or []),
                "cut_start": list(annotation["cut_start"]),
                "cut_end": list(annotation["cut_end"]),
                "cut_midpoint": list(annotation["cut_midpoint"]),
                "reading_direction": list(annotation["reading_direction"]),
                "source": "user_cross_cut",
                "updated_at": "",
                "status": "active",
                "resolved_line_numeric_id": line_numeric_id,
            }
        )
    return {
        "schema_version": 1,
        "page_id": str(page_id),
        "annotation_model": str(line_geometry.get("reading_direction_model") or "cross_cut_clockwise_90"),
        "line_annotations": sorted(annotations, key=lambda item: item["resolved_line_numeric_id"]),
        "stale_annotations": [],
    }


def convert(manuscript: str, destination_root: Path, *, force: bool, allow_missing_rasters: bool) -> dict:
    source = RELEASE_ROOT / "manuscripts" / manuscript
    if not source.is_dir():
        raise SystemExit(f"no released manuscript named {manuscript!r}")

    destination = destination_root / manuscript
    if destination.exists():
        if not force:
            raise SystemExit(
                f"{destination} already exists. Pass --force to replace it, or --dest to write elsewhere. "
                "Replacing discards any corrections made there in the tool."
            )
        shutil.rmtree(destination)

    page_ids = page_ids_for(source)
    rasters = sorted((source / "inputs").glob("*.jpg"))
    missing_rasters = [page for page in page_ids if not (source / "inputs" / f"{page}.jpg").exists()]
    if missing_rasters and not allow_missing_rasters:
        raise SystemExit(
            f"{manuscript}: {len(missing_rasters)} of {len(page_ids)} page rasters are not in the release.\n"
            f"  This manuscript's images are withheld for copyright. Follow\n"
            f"  {source / 'inputs' / 'DOWNLOAD.md'}\n"
            f"  and run tools/prepare_images.py to produce them, then re-run this script.\n"
            f"  Pass --allow-missing-rasters to build the tree anyway: the graph will load, but the\n"
            f"  page will render blank and saving will fail, because the polygon step needs the raster."
        )

    destination.mkdir(parents=True)
    counts = {"pages": len(page_ids), "rasters": 0, "heatmaps": 0, "labels": 0, "crops": 0, "sidecars": 0}

    # L1 rasters. Every released coordinate is defined on these.
    images_resized = destination / "images_resized"
    images_resized.mkdir()
    for raster in rasters:
        shutil.copy2(raster, images_resized / raster.name)
        counts["rasters"] += 1

    counts["heatmaps"] = copy_tree(source / "heatmaps", destination / "heatmaps")

    for release_dir, tool_dir in LABEL_DIRECTORIES.items():
        copied = copy_tree(source / release_dir, destination / tool_dir)
        if release_dir == "labels/line_images":
            counts["crops"] += copied
        else:
            counts["labels"] += copied

    # The page list is read from gnn-dataset, and only the dims files are needed
    # there: the node arrays beside them in labels/graph are the corrected set,
    # which the tool already prefers over anything in gnn-dataset.
    raw_dir = destination / "gnn-dataset"
    raw_dir.mkdir()
    for page_id in page_ids:
        shutil.copy2(source / "labels" / "graph" / f"{page_id}_dims.txt", raw_dir / f"{page_id}_dims.txt")

    # L6 under the two names the tool looks for.
    pagexml_dir = destination / "layout_analysis_output" / "page-xml-format"
    for page_id in page_ids:
        geometry_path = source / "labels" / "line_geometry" / f"{page_id}.json"
        if not geometry_path.exists():
            continue
        geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
        shutil.copy2(geometry_path, pagexml_dir / f"{page_id}_line_segmentation_metadata.json")
        (pagexml_dir / f"{page_id}_reading_direction_metadata.json").write_text(
            json.dumps(reading_direction_sidecar(geometry, page_id), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        counts["sidecars"] += 2

    settings = source / "processing_settings.json"
    if settings.exists():
        shutil.copy2(settings, destination / "processing_settings.json")

    (destination / "RELEASE_SOURCE.md").write_text(
        f"# {manuscript}\n\n"
        f"Rebuilt from `dataset_release/manuscripts/{manuscript}/` by\n"
        "`dataset_release/tools/load_in_annotation_tool.py`. Nothing here was recomputed.\n\n"
        "Saving a page in the annotation tool regenerates its `TextLine/Coords` and line crops\n"
        "with the current production line-segmentation strategy, so they may stop matching the\n"
        "released ones. The release itself is untouched.\n",
        encoding="utf-8",
    )

    counts["missing_rasters"] = len(missing_rasters)
    return counts


def check(manuscripts: list[str], destination_root: Path) -> int:
    """Load each converted manuscript through the annotation tool's own routes."""
    app_root = REPO_ROOT / "app"
    sys.path.insert(0, str(app_root))
    import os

    os.chdir(app_root)
    import app as backend

    client = backend.app.test_client()
    failures = 0

    for manuscript in manuscripts:
        pages_response = client.get(f"/manuscript/{manuscript}/pages")
        if pages_response.status_code != 200:
            print(f"  FAIL {manuscript}: /pages returned {pages_response.status_code}")
            failures += 1
            continue
        page_ids = pages_response.get_json().get("pages", [])
        expected = page_ids_for(RELEASE_ROOT / "manuscripts" / manuscript)
        if page_ids != expected:
            print(f"  FAIL {manuscript}: page list {page_ids} != released {expected}")
            failures += 1
            continue

        for page_id in page_ids:
            response = client.get(f"/semi-segment/{manuscript}/{page_id}")
            if response.status_code != 200:
                print(f"  FAIL {manuscript}/{page_id}: {response.status_code}")
                failures += 1
                continue
            data = response.get_json()
            problems = []
            if not data.get("layoutFromSavedGraph"):
                problems.append("layout did not come from the saved graph")
            if not data.get("graph", {}).get("nodes"):
                problems.append("no nodes")
            if not data.get("image"):
                problems.append("no page image")
            if not data.get("polygons"):
                problems.append("no polygons")
            expected_lines = len(
                {
                    line["line_numeric_id"]
                    for line in json.loads(
                        (
                            RELEASE_ROOT / "manuscripts" / manuscript / "labels" / "line_geometry" / f"{page_id}.json"
                        ).read_text(encoding="utf-8")
                    )["lines"]
                }
            )
            if len(data.get("polygons") or {}) != expected_lines:
                problems.append(f"{len(data.get('polygons') or {})} polygons, release has {expected_lines} lines")
            if problems:
                print(f"  FAIL {manuscript}/{page_id}: {'; '.join(problems)}")
                failures += 1

        annotated = sum(
            len(json.loads(path.read_text(encoding="utf-8")).get("line_annotations", []))
            for path in (destination_root / manuscript / "layout_analysis_output" / "page-xml-format").glob(
                "*_reading_direction_metadata.json"
            )
        )
        print(f"  ok   {manuscript}: {len(page_ids)} pages load, {annotated} reading-direction annotations")

    return failures


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--manuscript",
        action="append",
        help="a released manuscript id; repeatable. Default: all of them.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DESTINATION,
        help=f"where the tool reads manuscripts from (default: {DEFAULT_DESTINATION})",
    )
    parser.add_argument("--force", action="store_true", help="replace an existing destination manuscript")
    parser.add_argument(
        "--allow-missing-rasters",
        action="store_true",
        help="build the tree even when the page images are withheld from the release",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="after converting, load every page through the tool's routes and report",
    )
    args = parser.parse_args()

    manuscripts = args.manuscript or released_manuscripts()
    unknown = sorted(set(manuscripts) - set(released_manuscripts()))
    if unknown:
        raise SystemExit(f"unknown manuscript(s): {unknown}. Released: {released_manuscripts()}")

    destination_root = args.dest.resolve()
    destination_root.mkdir(parents=True, exist_ok=True)

    converted = []
    for manuscript in manuscripts:
        try:
            counts = convert(
                manuscript,
                destination_root,
                force=args.force,
                allow_missing_rasters=args.allow_missing_rasters,
            )
        except SystemExit as exit_error:
            print(f"{manuscript}: skipped\n{exit_error}\n")
            continue
        converted.append(manuscript)
        print(
            f"{manuscript} -> {destination_root / manuscript}\n"
            f"  {counts['pages']} pages, {counts['rasters']} rasters, {counts['heatmaps']} heatmaps, "
            f"{counts['labels']} label files, {counts['crops']} line crops, {counts['sidecars']} sidecars"
            + (f", {counts['missing_rasters']} rasters MISSING" if counts["missing_rasters"] else "")
        )

    if args.check and converted:
        print("\nloading through the annotation tool's routes:")
        failures = check(converted, destination_root)
        if failures:
            raise SystemExit(f"\n{failures} check(s) failed")
        print("\nall pages load")


if __name__ == "__main__":
    main()
