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

## 3. How to download them

The 15 pages mentioned above can be manually downloaded from the link:
https://dav.splrarebooks.com/collection/view/yajnavalakyasmritih-acharadhyayah


The images are not reachable as a static directory listing; they are served
through the viewer, so a page has to be opened before its image URL exists. A
short Playwright script driving Chromium plus an ordinary HTTP client for the
image bytes is enough. The sequence is:

1. Open the collection URL and wait for the page to settle.
2. Click the preview image to open the full-book viewer.
3. Read the total page count from the viewer.
4. For each page: read the `src` of the currently active image, download it if
   that URL has not been seen before, then advance to the next page.
5. Stop when the next control becomes disabled, keeping whatever was already
   downloaded.

The selectors the viewer used at the time of writing:

| Purpose | CSS selector |
| --- | --- |
| Preview image that opens the viewer | `div.plates .preview .img-container` |
| Active full-book viewer | `div.full-book.section.active` |
| Total page count | `span.total` |
| Active page image | `.owl-item.active .main-img` |
| Next-page control | `li.next` |

These are implementation details of the current site and may well have changed.
Verify them in the browser's developer tools before relying on them, and if the
markup has moved, look for the equivalent elements: the preview, the viewer, the
page-count indicator, the active carousel image, and the next control.

Worth building in, because each one is a real failure mode of this particular
viewer:

- **Pacing.** A random pause of a few seconds between page turns.
- **Duplicate protection.** Track the image URLs already downloaded; the
  carousel can repeat an image when a click does not advance.
- **Waiting on change.** After clicking next, wait until the active image `src`
  actually differs from the previous one rather than trusting a fixed sleep.
- **Resume.** Skip a page whose file already exists at nonzero size, so an
  interrupted run does not start over.
- **Retries.** Two or three attempts with a growing delay on a failed download,
  then give up on that page rather than the whole run.
- **Original filenames.** Take the last path segment of the image URL, strip the
  query string, and keep the extension the server gives.

Only the 15 files listed above are needed, but the viewer is paged, so reaching
them means stepping through the pages before them.

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
