# Obtaining the `moderate_layout` page rasters

The page images of this manuscript are not redistributed with the release. The
holding institution retains copyright and permits research use but not
redistribution, so this directory ships `RASTER_MANIFEST.json` and these notes
instead of the 15 JPEGs. Everything else about the manuscript — heatmaps, layout
graph, baselines, polygons, transcriptions, line geometry — is present and does
not depend on you completing this step. Only pixel-level work, crop-based
recognition and visual inspection do.

Retrieving the images is a two-part job: download them from the source viewer,
then run them through `tools/prepare_images.py` to obtain the exact raster every
coordinate in the release is defined on.

---

## 1. Where the images are

| | |
| --- | --- |
| Work | Yājñavalkyasmṛtiḥ (Ācārādhyāyaḥ) |
| Holding institution | Lalchand Research Library, DAV College, Chandigarh, India |
| Collection URL | <https://dav.splrarebooks.com/collection/view/yajnavalakyasmritih-acharadhyayah> |
| Rights | Copyright retained by the institution; research use permitted, redistribution not permitted |

The collection page shows a preview image which opens a JavaScript full-book
viewer. The viewer holds the whole manuscript, of which this release uses 15
pages.

Please observe the site's terms of service and `robots.txt`, and keep automated
access gentle — this is a small library server hosting a research collection.

---

## 2. Which images

These 15 files, named as the site serves them:

```text
233_0002.jpg  233_0003.jpg  233_0004.jpg  233_0005.jpg  233_0006.jpg
233_0007.jpg  233_0008.jpg  233_0009.jpg  233_0010.jpg  233_0011.jpg
233_0012.jpg  233_0013.jpg  233_0014.jpg  233_0015.jpg  233_0016.jpg
```

They are a contiguous run from the middle of the collection, not the first 15
pages of the viewer. `RASTER_MANIFEST.json` is the authoritative list: it carries
one entry per page with the filename, the expected dimensions, and a SHA-256 of
both the source file and the derived raster.

Keep the original filenames. The release keys every label layer on the page id,
which is the filename stem.

---

## 3. How to obtain them

The 15 pages are viewable at the collection URL:
<https://dav.splrarebooks.com/collection/view/yajnavalakyasmritih-acharadhyayah>

The most reliable route, and the one we would suggest first, is to **write to the
library**. Lalchand Research Library curates this collection and is used to
research enquiries; a request naming the manuscript and the folio range is often
answered with the scans, and it leaves the institution aware of the use its
material is being put to. That is worth more than saving an afternoon.

Failing that, the images can be read off the collection page by hand. A few
things about that page are worth knowing before you start, because each one has
cost us time:

- **The images sit behind the book viewer, not in a folder.** Opening the
  collection page gets you a preview; the full-resolution scan of a folio only
  becomes available once that folio is actually displayed in the viewer. There is
  no listing to fetch and no predictable address to guess.
- **The viewer is paged, and our 15 folios are in the middle.** You will pass
  through the earlier folios to reach them. Take the full-size image rather than
  the preview thumbnail.
- **The viewer will sometimes show you the same folio twice**, particularly if a
  page turn does not register. Check what you have saved against the list in
  section 2 rather than counting clicks.
- **Keep the name the server gives each file.** That name is the page id the
  whole release is keyed on; renaming the files disconnects them from every label
  layer. Put all 15 in one directory.
- **You do not have to finish in one sitting.** `RASTER_MANIFEST.json` lists all
  15 by name, so at any point you can see which ones you still need.

Whatever route you take: this is a small library server hosting a research
collection, and the courtesy owed to it is the ordinary one — go at human pace,
take only the 15 folios you need, and respect the site's terms of service and
`robots.txt`. The description above reflects the site as we found it, and a site
can be rebuilt at any time; treat it as orientation rather than as a recipe.

---

## 4. How to process them into the release rasters

Downloaded scans are not yet the raster the labels are defined on. Convert them
with:

```bash
python tools/prepare_images.py \
    --source-dir /path/to/downloaded/scans \
    --manuscript-dir manuscripts/moderate_layout
```

This reproduces the pipeline's own preprocessing step verbatim:

```text
open -> if max(w, h) > 3500: LANCZOS downscale so max(w, h) == 3500
     -> if mode in {RGBA, P, LA}: convert to RGB
     -> save as JPEG at Pillow's default quality
```

No page of this manuscript exceeds 3500 px, so in practice the derivation
reduces to a JPEG re-encode — which is why the result is stable enough to
checksum. All 15 pages reproduce byte-exactly under Pillow 11.3.0 and 12.3.0;
`reference_pillow_version` in the manifest records which of those the shipped
digests were last built under.

Each result is checked against `RASTER_MANIFEST.json`, which records two digests
per page so a failure is diagnosable: a mismatch on `source_sha256` means you
have a different scan, and a mismatch on `derived_sha256` alone means the right
scan and a different encoder.

Once `inputs/` holds the 15 derived rasters, run the release verifier to confirm
the images agree with the labels:

```bash
python tools/verify_dataset.py --release-root .
```
