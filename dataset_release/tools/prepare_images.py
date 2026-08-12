"""Reconstruct the withheld page rasters of ``moderate_layout``.

The page images of ``moderate_layout`` are under third-party copyright and are
not part of this release. Acquire them yourself from the holding institution
(see ``manuscripts/moderate_layout/inputs/SCRAPE.md``), then run this
script to turn the downloaded scans into the exact raster that every coordinate
in the release is defined on:

    python tools/prepare_images.py \
        --source-dir /path/to/downloaded/scans \
        --manuscript-dir manuscripts/moderate_layout

The derivation is the pipeline's own preprocessing step, reproduced verbatim:

    open -> if max(w, h) > 3500: LANCZOS downscale so max(w, h) == 3500
         -> if mode in {RGBA, P, LA}: convert to RGB
         -> save as JPEG at Pillow's default quality

For ``moderate_layout`` no page exceeds 3500 px, so the step reduces to a JPEG
re-encode. Each result is checked against ``RASTER_MANIFEST.json``. A byte-level
match reproduces the release raster exactly; if the bytes differ (a different
libjpeg build, say) the maximum per-pixel deviation is reported so you can judge
whether the difference is perceptually and numerically negligible.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

TARGET_LONGEST_SIDE = 3500


def derive_raster_bytes(path: Path) -> bytes:
    with Image.open(path) as image:
        width, height = image.size
        if max(width, height) > TARGET_LONGEST_SIDE:
            scale = TARGET_LONGEST_SIDE / max(width, height)
            image = image.resize(
                (int(width * scale), int(height * scale)), Image.Resampling.LANCZOS
            )
        if image.mode in ("RGBA", "P", "LA"):
            image = image.convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, "JPEG")
    return buffer.getvalue()


def max_pixel_deviation(left: bytes, right: bytes) -> float | None:
    try:
        a = np.asarray(Image.open(io.BytesIO(left)).convert("RGB"), dtype=np.int16)
        b = np.asarray(Image.open(io.BytesIO(right)).convert("RGB"), dtype=np.int16)
    except Exception:
        return None
    if a.shape != b.shape:
        return None
    return float(np.abs(a - b).max())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path,
                        help="directory holding the scans you downloaded")
    parser.add_argument("--manuscript-dir", required=True, type=Path,
                        help="the release manuscript directory, e.g. manuscripts/moderate_layout")
    parser.add_argument("--dry-run", action="store_true",
                        help="verify only; do not write inputs/")
    args = parser.parse_args()

    target_dir = args.manuscript_dir / "inputs"
    manifest_path = target_dir / "RASTER_MANIFEST.json"
    if not manifest_path.exists():
        print(f"error: no raster manifest at {manifest_path}", file=sys.stderr)
        return 2

    import json

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(
        f"{manifest['manuscript_id']}: {len(manifest['pages'])} pages, "
        f"reference Pillow {manifest['reference_pillow_version']}, "
        f"this Pillow {Image.__version__ if hasattr(Image, '__version__') else __import__('PIL').__version__}"
    )

    exact = approx = missing = wrong_source = 0
    for entry in manifest["pages"]:
        source_path = args.source_dir / entry["filename"]
        if not source_path.exists():
            print(f"  MISSING  {entry['filename']}")
            missing += 1
            continue

        source_digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
        if source_digest != entry["source_sha256"]:
            print(
                f"  SOURCE?  {entry['filename']}: downloaded scan does not match the "
                f"scan used to build this release"
            )
            wrong_source += 1

        derived = derive_raster_bytes(source_path)
        derived_digest = hashlib.sha256(derived).hexdigest()
        if derived_digest == entry["derived_sha256"]:
            exact += 1
            status = "exact"
        else:
            reference_path = target_dir / entry["filename"]
            deviation = None
            if reference_path.exists():
                deviation = max_pixel_deviation(derived, reference_path.read_bytes())
            approx += 1
            status = (
                f"BYTES DIFFER (max pixel deviation {deviation:.0f}/255)"
                if deviation is not None
                else "BYTES DIFFER (no local reference to compare pixels against)"
            )
            print(f"  {status}  {entry['filename']}")

        if not args.dry_run:
            (target_dir / entry["filename"]).write_bytes(derived)

    print(
        f"\n{exact} exact, {approx} byte-divergent, {missing} missing, "
        f"{wrong_source} source mismatch"
    )
    if not args.dry_run and not missing:
        print(f"wrote {exact + approx} rasters to {target_dir}")
    return 0 if (missing == 0 and wrong_source == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
