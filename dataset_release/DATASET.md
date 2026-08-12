# Sanskrit Manuscript Layout Regimes

A labels dataset for handwritten Devanagari manuscript page analysis: 31 folios from three
Sanskrit manuscripts, with human ground truth for text-line structure, region grouping,
reading direction and diplomatic transcription, plus the exact image inputs those labels are
defined on.

The three manuscripts span layout difficulty rather than sampling a population. One is a wide
landscape folio with long straight lines; one is densely packed with heavy marginalia; one is
dominated by yantra diagrams whose text lines curve and close into circles. Nine text lines in
this corpus are topological cycles; 172 are single glyphs. That range is the point.

**The one thing not to get wrong:** transcriptions, region labels, reading directions and the
layout graph's nodes and edges are human ground truth; the text-line groupings are connected
components of that human edge set; the `Coords` polygons are generated deterministically from
the human baselines plus a CRAFT heatmap. Evaluating polygon prediction against these `Coords`
measures agreement with a documented generator, not with a person. Section 5 gives the
provenance of every layer.

## At a glance

| | |
| --- | --- |
| Manuscripts | 3 — `moderate_layout`, `dense_layout`, `circular_layout` |
| Pages | 31 |
| Text regions | 596 |
| Text lines | 882 |
| Transcribed characters (NFC) | 27,618, of which 26,202 in the Devanagari block |
| Character vocabulary | 131 distinct code points |
| Layout graph nodes / edges | 13,074 / 12,201 |
| Cycle edges (closed circular lines) | 9 |
| Reading-direction annotations | 25, all in `circular_layout` |
| Rectified line crops | 566 |
| Page rasters | shipped for 2 of 3 manuscripts; withheld and byte-reproducible for the third |
| Files / verification checks | 915 / 842, all passing |
| Size | ≈ 40 MB unpacked, ≈ 35 MB as a gzipped tar |
| PAGE-XML schema | PRImA PAGE 2013-07-15 |

## Contents

1. [What this dataset is](#1-what-this-dataset-is)
2. [Quick start](#2-quick-start)
3. [Directory layout](#3-directory-layout)
4. [How the labels were produced](#4-how-the-labels-were-produced)
5. [The seven label layers](#5-the-seven-label-layers)
6. [Coordinate spaces](#6-coordinate-spaces)
7. [Rights, sources and citation](#7-rights-sources-and-citation)
8. [Evaluation protocol](#8-evaluation-protocol)
9. [Integrity and verification](#9-integrity-and-verification)
10. [Known limitations](#10-known-limitations)
11. [Provenance](#11-provenance)

---

## 1. What this dataset is

Inputs, and ground truth defined on those inputs. For each page: the page raster, a CRAFT
character-region heatmap computed from it, a graph over character-sized blobs whose edges say
"these two blobs are consecutive on one text line", PAGE-XML carrying baselines, polygons and
transcriptions, per-line geometry and reading directions, and rectified line crops.

### 1.1 The three manuscripts

| Release id | Work | Genre | Holding institution | Pages | Page rasters |
| --- | --- | --- | --- | ---: | --- |
| `moderate_layout` | Yājñavalkyasmṛtiḥ (Ācārādhyāyaḥ) | dharmaśāstra | Lalchand Research Library, DAV College, Chandigarh, India | 15 | withheld, reconstructible |
| `dense_layout` | Muhūrta Mārtaṇḍa | jyotiṣa | eGangotri Digital Preservation Trust (Dharmarth Trust J&K collection) | 7 | shipped |
| `circular_layout` | Tantrarāja, with Yantra and Mantra Uddhāra | tantra | eGangotri Digital Preservation Trust (Dharmarth Trust J&K collection) | 9 | shipped |

Full source records, rights bases and BibTeX keys: section 7,
`manuscripts/<manuscript>/SOURCE.md`, and `SOURCES.bib`.

### 1.2 The three regimes, in numbers

| | `moderate_layout` | `dense_layout` | `circular_layout` |
| --- | ---: | ---: | ---: |
| Pages | 15 | 7 | 9 |
| Page raster (px), landscape/portrait | 2500 × 912…967 | 3500 × 1511…1605 | 2504…2745 × 3500 |
| Distinct raster sizes | 10 | 6 | 9 |
| Text regions | 195 | 146 | 255 |
| Text lines | 316 | 311 | 255 |
| Maximum lines in one region | 8 | 16 | 1 |
| `horizontal_straight` | 282 | 289 | 91 |
| `vertical_straight` | 0 | 0 | 1 |
| `curved_open` | 2 | 7 | 29 |
| `closed_circular` | 0 | 0 | 9 |
| `point` | 32 | 15 | 125 |
| Reading-direction annotations | 0 | 0 | 25 |
| Lines with empty transcription | 7 | 2 | 5 |
| Transcribed chars (NFC) | 12,495 | 10,090 | 5,033 |
| … in the Devanagari block | 12,264 | 9,471 | 4,467 |
| Character vocabulary | 86 | 109 | 107 |
| Median line length (NFC chars) | 15 | 15 | 4 |
| Baseline points min / median / max | 1 / 7 / 59 | 1 / 7 / 61 | 1 / 2 / 66 |
| Polygon points min / median / max | 12 / 96 / 389 | 14 / 135 / 410 | 6 / 36 / 450 |
| Non-simple `Coords` polygons | 16 / 316 | 75 / 311 | 41 / 255 |
| Lines with `fallback_used` | 10 | 8 | 57 |
| Graph nodes / edges | 6,015 / 5,699 | 4,886 / 4,575 | 2,173 / 1,927 |
| Isolated nodes (degree 0) | 32 | 15 | 125 |
| Mean degree over non-isolated nodes | 1.905 | 1.878 | 1.882 |
| Maximum degree | 2 | 3 | 2 |
| Surplus edges beyond a spanning forest | 0 | 0 | 9 |

`moderate_layout` is the regular case: long straight lines, even spacing, sparse marginalia,
and 32 `point` lines that are folio numbers and short marginal marks.

`dense_layout` packs up to 16 lines into one region, with heavy marginal and interlinear
material. Adjacent lines' heatmap evidence competes, hence its high non-simple polygon rate.
It is the only manuscript with a node of degree 3 — exactly one, on page 23 — so a consumer
assuming each text line is a simple path meets exactly one violation in this corpus.

`circular_layout` is where ordinary document-layout assumptions break. Nine text lines are
closed rings, so nine components are cycles rather than trees; a ring has no canonical start
point or handedness, which is why all 25 reading-direction annotations are here, and without
them a rectified crop of a ring is ambiguous up to seam and direction. 125 of its 255 lines are
single glyphs — the cells of a yantra. Every region holds exactly one line, an artefact of
labelling rather than of the manuscript (section 10.5).

Three identities hold on every page of all three manuscripts, and are the skeleton of the
dataset:

```text
  #connected components of the edge set  ==  #PAGE TextLines            (check C6)
  #edges - (#nodes - #components)        ==  #closed_circular lines     (check C7)
  #isolated nodes  ==  #single-point baselines  ==  #point lines
```

The third is not separately checked but holds exactly: 32, 15 and 125. A `point` line is
precisely a text line whose graph component is a single node.

### 1.3 What is not in this release

No machine pre-correction state, no correction deltas, no annotation timing. Specifically
absent and not recoverable from the shipped files: raw pre-correction CRAFT node proposals,
per-page counts of nodes the annotator added or removed, and the annotation-effort log.

One consequence, stated plainly: **this release is not drop-in for the reference experiment
harness**, which hardcodes the authoring tree's directory names and expects the raw node
proposals. That is the trade made for legibility. Section 3.2 maps the names back.

### 1.4 Terminology

| Term | Meaning |
| --- | --- |
| **page raster** | The image in `inputs/`. Every coordinate in the release is defined on this pixel grid. |
| **heatmap space** | The CRAFT region-score grid, exactly half the page raster in each dimension. All graph node coordinates live here. |
| **node** | One local maximum of the region-score heatmap: approximately one Devanagari akṣara-sized ink blob. |
| **candidate edge** | An unordered node pair offered to a classifier as possibly same-line. Fixed heuristics, never learned. Not shipped; regenerable (section 4). |
| **layout graph** | The shipped node and edge set, in `labels/graph/`. Human ground truth. |
| **text line** | One connected component of the layout graph; one PAGE `TextLine`. |
| **region** | A `TextRegion`: a column, block, marginal note or diagram cell grouping one or more text lines. |
| **line kind** | One of `horizontal_straight`, `vertical_straight`, `curved_open`, `closed_circular`, `point`. Computed from baseline geometry. |
| **annotator** | The person who produced the labels. Referred to as they/them. |

---

## 2. Quick start

### 2.1 Verify the release

```bash
cd dataset_release
python tools/verify_dataset.py --release-root .
```

```text
== integrity
   915 files listed, 0 mismatched, 0 missing

== consistency: moderate_layout
   15 pages, 316 text lines, kinds={'curved_open': 2, 'horizontal_straight': 282, 'point': 32}, page rasters=withheld
…
== consistency: circular_layout
   9 pages, 255 text lines, kinds={'closed_circular': 9, 'curved_open': 29, 'horizontal_straight': 91, 'point': 125, 'vertical_straight': 1}, page rasters=shipped

OK: 842 checks passed
```

Requires `numpy` and `Pillow` only; exit status 0 on success, 1 otherwise. `--skip-checksums`
runs the consistency invariants alone. The checksum listing covers `DATASET.md` itself, so
editing this file reports a mismatch until `tools/build_release.py` regenerates the listing.

### 2.2 Load one page

Run from the release root. Reads every label layer for one page and cross-checks them.

```python
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(".")
MS, PAGE = "circular_layout", "11"
NS = {"p": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}
base = ROOT / "manuscripts" / MS

# L3 layout graph. Node coordinates are heatmap space; x2 converts to page raster.
g = base / "labels" / "graph"
nodes = np.loadtxt(g / f"{PAGE}_inputs_unnormalized.txt").reshape(-1, 3)[:, :2] * 2.0
edges = np.loadtxt(g / f"{PAGE}_edges.txt", dtype=int).reshape(-1, 2)
line_of_node = np.loadtxt(g / f"{PAGE}_labels_textline.txt", dtype=int)
region_of_node = np.loadtxt(g / f"{PAGE}_labels_textbox.txt", dtype=int)

# L5 PAGE-XML: polygon, baseline and transcription per line, all in page-raster pixels.
lines = {}
for tl in ET.parse(base / "labels" / "page_xml" / f"{PAGE}.xml").iter(f"{{{NS['p']}}}TextLine"):
    pts = lambda tag: [tuple(map(float, q.split(","))) for q in tl.find(tag, NS).get("points").split()]
    lines[tl.get("id")] = {"polygon": pts("p:Coords"), "baseline": pts("p:Baseline"),
                           "text": tl.findtext(".//p:Unicode", default="", namespaces=NS)}

# L6 line geometry: kind, topology and reading direction, keyed by line_id.
geom = {l["line_id"]: l
        for l in json.loads((base / "labels" / "line_geometry" / f"{PAGE}.json").read_text())["lines"]}

# Derived index: one record per text line, in PAGE-XML order.
index = [r for r in map(json.loads, (base / "lines.jsonl").read_text(encoding="utf-8").splitlines())
         if r["page_id"] == PAGE]

print(f"{len(nodes)} nodes, {len(edges)} edges, {len(set(line_of_node.tolist()))} components, "
      f"{len(lines)} TextLines, {len(set(region_of_node.tolist()))} regions")
for r in index:
    if r["line_kind"] == "closed_circular":
        print(r["line_id"], r["line_kind"], r["reading_direction"], repr(r["text"][:12]),
              "polygon pts:", len(lines[r["line_id"]]["polygon"]),
              "| cut:", geom[r["line_id"]]["reading_direction_annotation"]["cut_midpoint"])
        break
```

```text
192 nodes, 170 edges, 24 components, 24 TextLines, 24 regions
region_6_line_0 closed_circular [1.0, 0.0] 'हिंहींहुंहूं' polygon pts: 418 | cut: [1308.5714285714287, 790.7142857142858]
```

### 2.3 Reconstruct the withheld rasters

Acquire the 15 `moderate_layout` scans as specified in
`manuscripts/moderate_layout/inputs/SCRAPE.md`, then:

```bash
python tools/prepare_images.py \
    --source-dir /path/to/downloaded/scans \
    --manuscript-dir manuscripts/moderate_layout
```

Each result is checked against `manuscripts/moderate_layout/inputs/RASTER_MANIFEST.json`. All
15 pages reproduce byte-exactly under Pillow 11.3.0 and 12.3.0. Details in section 7.2.

---

## 3. Directory layout

```text
dataset_release/
├── DATASET.md
├── SOURCES.bib                       BibTeX for the three source manuscripts
├── dataset_manifest.json             machine-readable inventory and statistics
├── CHECKSUMS.sha256                  915 entries, every shipped file
├── folds/<manuscript>.json           5 folds, train_size 3, seed 42
├── tools/
│   ├── build_release.py              the sole provenance of this tree
│   ├── verify_dataset.py             842 integrity and consistency checks
│   └── prepare_images.py             reconstructs the withheld rasters
└── manuscripts/<manuscript>/         moderate_layout | dense_layout | circular_layout
    ├── SOURCE.md                     work, genre, holding institution, rights, BibTeX key
    ├── processing_settings.json      target_longest_side 3500, min_distance 20
    ├── pages.csv                     one row per page                        (derived index)
    ├── lines.jsonl                   one record per text line                (derived index)
    ├── inputs/                       L1 page rasters <page>.jpg
    ├── heatmaps/                     L2 CRAFT region-score maps <page>.jpg, half scale
    └── labels/
        ├── graph/                    L3 <page>_{dims,inputs_normalized,inputs_unnormalized,
        │                                        edges,labels_textline,labels_textbox}.txt
        ├── page_xml_baselines/       L4 <page>.xml   Baseline only
        ├── page_xml/                 L5 <page>.xml   Coords + Baseline + Unicode
        ├── line_geometry/            L6 <page>.json  kind, topology, unwrap frame, reading direction
        └── line_images/              L7 <page>/textbox_label_<K>/line_<N>.jpg
```

| Path | `moderate_layout` | `dense_layout` | `circular_layout` |
| --- | ---: | ---: | ---: |
| `inputs/` | 2 (`SCRAPE.md`, `RASTER_MANIFEST.json`) | 7 | 9 |
| `heatmaps/` | 15 | 7 | 9 |
| `labels/graph/` | 90 | 42 | 54 |
| `labels/page_xml_baselines/` | 15 | 7 | 9 |
| `labels/page_xml/` | 15 | 7 | 9 |
| `labels/line_geometry/` | 15 | 7 | 9 |
| `labels/line_images/` | 0 | 311 crops | 255 crops |

### 3.1 Page identifiers

Page ids are strings and their format differs by manuscript: `233_0002` … `233_0016` for
`moderate_layout`, bare integers for the other two (`10`, `17`, `22`, `23`, `24`, `25`, `28`;
and `11`, `14`, `15`, `28`, `35`, `49`, `56`, `64`, `65`). Every file belonging to a page is
named `<page_id>` plus a layer-specific suffix. Sort page ids as strings, or `moderate_layout`
reorders incorrectly. The authoritative list is `page_ids` in `dataset_manifest.json`.

### 3.2 Directory names, mapped back to the authoring tree

Directory names were made legible for the release; file names inside them were deliberately
left as the pipeline writes them. The paper and the authoring repository use the older names.

| Release | Authoring tree |
| --- | --- |
| `inputs/` | `images_resized/` |
| `labels/` | `layout_analysis_output/` |
| `labels/graph/` | `layout_analysis_output/gnn-format/` |
| `labels/page_xml/` | `layout_analysis_output/page-xml-format/` |
| `labels/page_xml_baselines/` | `layout_analysis_output/_baseline_page_xml/` |
| `labels/line_geometry/` | `layout_analysis_output/page-xml-format/<page>_line_segmentation_metadata.json` |
| `labels/line_images/` | `layout_analysis_output/image-format/` |

Manuscript identifiers map as `yajn` → `moderate_layout`, `dense` → `dense_layout`,
`circle_new` → `circular_layout`.

---

## 4. How the labels were produced

Nothing here is required in order to load the files. It is here so that the label semantics
are intelligible.

### 4.1 Why the layout is a graph

The layout ground truth is a graph over character-sized blobs in which an edge asserts that two
blobs are consecutive on one text line, and a text line is a connected component. Boxes fail
here: a line running around the rim of a circle has an axis-aligned box covering the whole
diagram, and two concentric rings give nested boxes with near-total mutual overlap, which
destroys both matcher assignment and the meaning of IoU. Pixel masks fail on dense folios,
where the descenders of one line touch the headline of the next and the merge corrupts the
reading order of a whole block. The graph handles straight lines, arcs, rings and isolated
glyphs in one formalism. The cost is that everything is conditioned on the node set: a
character with no node cannot belong to a line.

### 4.2 The pipeline

```text
  page scan (not released)
    │  LANCZOS to longest side 3500 if larger; RGB; JPEG
    ▼
  L1 page raster ── CRAFT (VGG16-BN, craft_mlt_25k.pth), region-score head only ──► L2 heatmap
    │                                                       (8-bit, 255 x score, half scale)
    │  min-max normalize per page; local maxima under maximum_filter(size=20), kept > 0.4
    ▼
  node set ──► candidate edges: collinearity heuristic (k=10, cosine <= -0.8)
    │                         U angular k-NN (k=50, nearest per 20-degree sector), bidirectional
    │  binary edge classification, then human review: add/delete nodes and edges,
    │  assign region labels, draw reading-direction cross-cuts
    ▼
  L3 layout graph ──► connected components ──► text lines ──► ordered nodes ──► L4 baselines
    │
    │  local_polygons_stable_unwrap_v1: assign heatmap components to the baseline,
    │  project into a local arclength/normal frame, clean, map back
    ▼
  L5 polygons + transcription ──► unwrap along arclength, apply reading direction ──► L6, L7
```

Notes that matter for interpreting the labels:

- **Peak detection is page-relative.** The region score is min-max normalized per page before
  thresholding at 0.4, so the threshold is relative to that page's strongest character evidence.
  `min_distance = 20` and `target_longest_side = 3500` are in `processing_settings.json`.
- **Candidate edges are not shipped**, only the human-accepted positive edges. They regenerate
  from the node coordinates: for each node the collinearity heuristic takes its 10 nearest
  neighbours and adds edges to the closest pair whose direction vectors from that node are
  nearly opposite (cosine below −0.8); the angular k-NN takes the 50 nearest and keeps the
  nearest in each 20-degree sector, guaranteeing the true edge set is a subset.
- **Components carry no order.** Ordering a component's nodes is a separate geometric step, and
  for a ring it is underdetermined — hence L6's reading directions.
- **Polygons are grown from the heatmap, not drawn** (section 5.5).

---

## 5. The seven label layers

| Layer | Path | Content | Provenance |
| --- | --- | --- | --- |
| **L1** | `inputs/` | page raster | machine — deterministic transform of the scan |
| **L2** | `heatmaps/` | CRAFT region-score map | machine |
| **L3** | `labels/graph/` | nodes, edges, region labels | **human** |
| **L3** | `labels/graph/` | text-line labels | derived — components of the human edge set |
| **L4** | `labels/page_xml_baselines/` | baselines | derived — the components, ordered |
| **L5** | `labels/page_xml/` | `TextLine/Coords` polygons | derived — generated from L4 + L2 |
| **L5** | `labels/page_xml/` | `TextEquiv/Unicode` | **human** |
| **L6** | `labels/line_geometry/` | line kind, topology, unwrap frame | derived — computed from L4 geometry |
| **L6** | `labels/line_geometry/` | reading directions | **human** |
| **L7** | `labels/line_images/` | rectified line crops | derived |

### 5.1 L1 — page raster

`manuscripts/<manuscript>/inputs/<page>.jpg`. Machine. RGB JPEG. Produced from the scan by:
open; if `max(w, h) > 3500`, LANCZOS downscale so `max(w, h) == 3500`; if the mode is `RGBA`,
`P` or `LA`, convert to RGB; save as JPEG at Pillow's default quality.

**Every coordinate in the release is expressed in this pixel grid.** Raster sizes vary per page
within a manuscript, so never assume a constant page size. For `moderate_layout` this directory
holds only `SCRAPE.md` and `RASTER_MANIFEST.json` (section 7.2).

### 5.2 L2 — CRAFT region-score heatmap

`manuscripts/<manuscript>/heatmaps/<page>.jpg`. Machine. 8-bit grayscale, written as
`255 × region_score`, exactly half the page raster in each dimension. Shipped for all three
manuscripts, including the one whose rasters are withheld.

Not a segmentation mask: a per-pixel character-centre likelihood, consumed by the polygon
builder as image evidence. Stored as JPEG, so the field is quantized to 8 bits and lossily
compressed; it is not a faithful float record. The image dimension is the floor of half the
raster (section 6).

### 5.3 L3 — layout graph

`manuscripts/<manuscript>/labels/graph/`, six files per page.

```text
11_dims.txt                 1364.5 1750.0
11_inputs_unnormalized.txt  674.999500 294.000000 0.000000
                            581.999250 296.000250 0.000000
                            …
11_inputs_normalized.txt    0.385714 0.168000 0.000000
                            0.332571 0.169143 0.000000
                            …
11_edges.txt                163 164          11_labels_textline.txt   0
                            47 53                                     0
                            50 52                                     …
                            …                11_labels_textbox.txt    0
                                                                      0
```

| File | Shape | Meaning |
| --- | --- | --- |
| `<page>_dims.txt` | one line, `"W H"` | Exactly half the page raster, possibly fractional. 12 of 31 pages have a fractional entry. |
| `<page>_inputs_unnormalized.txt` | N × 3 floats | Node `x y s` in heatmap pixels. `s` is a font-size slot, identically `0.0` throughout. |
| `<page>_inputs_normalized.txt` | N × 3 floats | The same array with all three columns divided by `max(W, H)` from `dims.txt`. |
| `<page>_edges.txt` | E × 2 ints | Undirected edges `u v`, deduplicated, pair order canonicalized. |
| `<page>_labels_textline.txt` | N ints | Text-line id per node: the connected-component id. **Derived.** |
| `<page>_labels_textbox.txt` | N ints | Region id per node. **Human**, with the exception below. |

Row order in the node array is the node index used by the edge list, both label files, and
`component_node_indices` in L6.

**Traps.** The third column is a font-size slot that is `0.0` for every node of every page; it
is not missing data, the reference preprocessing disables that feature. We do not use Font-size feature everywhere. This is an artifict of an abandoned feature.
`labels_textline.txt` is not an independent annotation — it is
`scipy.sparse.csgraph.connected_components` over the edge set, and re-running components
reproduces the partition exactly (verified on all 31 pages). `labels_textbox.txt` is a genuine
human label only where the annotator supplied one; elsewhere a per-component majority vote
assigned a fresh region id, and `circular_layout` received no region labelling at all
(section 10.5). Isolated nodes exist (32, 15, 125) and are the `point` lines: code that infers
line membership from edges alone will drop them.

### 5.4 L4 — PAGE-XML baselines

`manuscripts/<manuscript>/labels/page_xml_baselines/<page>.xml`. PAGE 2013-07-15. The layout
ground truth with nothing inferred from image evidence.

```xml
<?xml version='1.0' encoding='UTF-8'?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">
	<Metadata><Creator>GNN-Layout-Analysis</Creator><Created>2026-07-15T09:58:36.326011</Created></Metadata>
	<Page imageFilename="11.jpg" imageWidth="2729" imageHeight="3500">
		<TextRegion id="region_0" custom="textbox_label_0">
			<Coords points="1163,588 1436,588 1436,593 1163,593" />
			<TextLine id="region_0_line_0" custom="structure_line_id_0">
				<Baseline points="1163,592 1202,592 1237,592 1274,592 1307,592 1349,588 1394,593 1436,593" />
			</TextLine>
		</TextRegion>
		…
	</Page>
</PcGts>
```

`TextRegion` carries `custom="textbox_label_<K>"` and a `Coords`; each `TextLine` carries
`custom="structure_line_id_<N>"` and only a `Baseline`. Use this layer to evaluate line
grouping and line geometry without the confound of the polygon generator.

**Traps.** A `point` line's `Baseline` is a single coordinate pair; 172 lines are of that kind.
Region `Coords` bound the region's baselines, not its ink, and are degenerate for single-point
regions — `1311,995 1311,995 1311,995 1311,995` is a real example from `circular_layout`
page 11. Hence treat the "Baseline" as the real layout ground-truth instead of the "Coords" in the PAGE-XML.

### 5.5 L5 — PAGE-XML with polygons and transcriptions

`manuscripts/<manuscript>/labels/page_xml/<page>.xml`. Identical structure to L4 plus a
`Coords` and a `TextEquiv/Unicode` per `TextLine`. Same page, same line:

```xml
<TextLine id="region_0_line_0" custom="structure_line_id_0">
	<Coords points="1146,573 1146,581 1146,589 1146,596 1146,604 1146,612 1149,618 1153,620
	                1157,623 1163,623 1170,623 1176,623 1180,621 1183,619 1187,623 … " />
	<Baseline points="1163,592 1202,592 1237,592 1274,592 1307,592 1349,588 1394,593 1436,593" />
	<TextEquiv>
		<Unicode>प्रथमयंत्रमिदम्</Unicode>
	</TextEquiv>
</TextLine>
```

**The polygon is not human-drawn.** `local_polygons_stable_unwrap_v1` generates it from the
human baseline plus the CRAFT heatmap: heatmap connected components are assigned to the
baseline, projected into a local arclength/normal frame, cleaned, and mapped back. Object and
pixel metrics against it measure agreement with that generator; a model hugging the ink more
tightly than the generator will score worse, not better.

**The transcription is human.** Exactly one direct `TextEquiv/Unicode` per line, a diplomatic
transcription. 14 lines corpus-wide have an empty transcription (7 / 2 / 5), retained rather
than dropped because dropping them would silently change the denominator of every layout
metric.

**132 of 882 polygons (15.0%) are self-intersecting** (16 / 75 / 41), a property of the densely
sampled contour the generator emits. Apply a repair policy for geometric metrics and report
which; the reference evaluator rasterizes `Coords` as a filled contour in page-raster space and
re-extracts valid foreground contours. Text metrics are unaffected: they read only
`TextLine/TextEquiv/Unicode`.

### 5.6 L6 — line geometry and reading direction

`manuscripts/<manuscript>/labels/line_geometry/<page>.json`, one per page, schema
`line_geometry-1`.

```json
{
  "schema": "line_geometry-1",
  "page_id": "11",
  "strategy_name": "local_polygons_stable_unwrap_v1",
  "reading_direction_model": "cross_cut_clockwise_90",
  "line_count": 24,
  "geometry_summary": {
    "geometry_source": "baseline_heatmap", "prepared_line_count": 24, "source_text_line_count": 24,
    "crop_model_counts": {"local_polygon_stable_unwrap": 24},
    "topology_counts": {"horizontal_straight": 9, "point": 11, "closed_circular": 2, "curved_open": 2},
    "normalization_action_counts": {"preserved_left_to_right": 11, "annotated_cut_clockwise": 2},
    "orientation_policy": {"reading_order": "left_to_right", "circular_direction": "clockwise"}
  },
  "lines": [ … ]
}
```

Each entry of `lines`:

| Key | Meaning |
| --- | --- |
| `line_id` | matches `TextLine/@id` in the PAGE-XML, e.g. `region_6_line_0` |
| `line_custom` | matches `TextLine/@custom`, e.g. `structure_line_id_2` |
| `line_numeric_id` | the integer inside `line_custom` |
| `line_kind` | one of the five kinds |
| `topology` | object, fields listed below |
| `reading_direction_annotation` | object or `null`; present for 25 lines, all in `circular_layout` |
| `crop_model` | `"local_polygon_stable_unwrap"` |
| `local_s_min`, `local_s_max` | arclength bounds of the unwrap frame, page-raster px |
| `local_n_min`, `local_n_max` | signed normal bounds of the unwrap frame, page-raster px |
| `line_half_width_px` | estimated half-width of the line band |
| `local_canvas_width_px`, `local_canvas_height_px` | size of the rectified crop in `labels/line_images/` |
| `page_polygon_point_count` | number of points in the PAGE `Coords`; agrees with L5 |
| `fallback_used` | boolean; true where the heatmap gave no assignable component |
| `fallback_reason` | string or `null` |

`topology` carries `original_point_count`, `normalized_point_count`, `split_index`,
`mirror_pair_count`, `mirror_match_tolerance`, `mean_mirror_distance`, `max_mirror_distance`,
`was_out_and_back`, `out_and_back_detection`, `short_tail_trimmed`, `dominant_axis_deduped`,
`dominant_axis`, `repeated_near_point_count`, `normalization_actions`, `is_closed`,
`closed_path_tolerance`, `cut_index`, `baseline_length`, `line_kind`, `orientation_action`,
`reading_direction` and `reading_cut_point`.

`line_kind` is computed from the baseline alone:

| Kind | Rule |
| --- | --- |
| `closed_circular` | first and last baseline points within `closed_path_tolerance = 12` px |
| `point` | fewer than 2 points, or polyline length ≈ 0 |
| `horizontal_straight` | `chord / arclength ≥ 0.985` and the chord within 12° of horizontal |
| `vertical_straight` | `chord / arclength ≥ 0.985` and the chord within 12° of vertical |
| `curved_open` | everything else |

**Reading directions.** The model is `cross_cut_clockwise_90`: the annotator draws a cross-cut
segment across the line and the reading tangent is that cut rotated 90°, `t = unit(−dy, dx)`.
This holds exactly for all 25 annotations. A complete record, `circular_layout` page 11:

```json
"reading_direction_annotation": {
  "resolved_line_numeric_id": 2,
  "reading_direction": [1.0, 0.0],
  "cut_start":    [1308.5714285714287, 797.1428571428572],
  "cut_end":      [1308.5714285714287, 784.2857142857143],
  "cut_midpoint": [1308.5714285714287, 790.7142857142858],
  "component_node_indices": [24, 25, 26, 27, 28, 30, 31, 32, 34, 35, 40, … ]
}
```

The cut runs from `y = 797.14` to `y = 784.29` at constant `x`, so `(dx, dy) = (0, −12.857)`
and `unit(−dy, dx) = (1, 0)`, the recorded direction. Reading directions are authoritative
where present: use them rather than inferring orientation from decoded text. Where absent and
the line is `curved_open` or `closed_circular`, orientation is not determined by this dataset.

**Traps.** `cut_start`, `cut_end`, `cut_midpoint` and `topology.reading_cut_point` are in
**page-raster** space, while `component_node_indices` indexes the **heatmap**-space node array
of L3 (section 6.2). `fallback_used` is true on 75 of 882 lines (10 / 8 / 57), all with
`fallback_reason = "image_adaptive_binarization_no_assigned_components"`: those polygons fell
back to a band around the baseline instead of heatmap evidence and are the weakest geometry in
the release. Hence treat the "Baseline" as the real layout ground-truth instead of the "Coords" in the PAGE-XML.
`topology.line_kind` duplicates the entry-level `line_kind`; they agree.

### 5.7 L7 — rectified line crops

`manuscripts/<manuscript>/labels/line_images/<page>/textbox_label_<K>/line_<N>.jpg`, where
`<K>` is the region label and `<N>` the line index within that region. 566 crops: 311 for
`dense_layout`, 255 for `circular_layout`, none for `moderate_layout` (the crops are the
withheld image, in pieces). The exact path per line is the `line_image` field of `lines.jsonl`.

Each crop is the line resampled along its arclength/tangent frame into a straight strip, with
the annotated reading direction applied where one exists — close to a rotated crop for a
straight line, a genuine rectification cut at the annotated seam for a ring. Crop dimensions
are `local_canvas_width_px × local_canvas_height_px` in L6.

### 5.8 Derived indexes

Neither file introduces information; both exist so the dataset is usable without an XML parser.

**`manuscripts/<manuscript>/lines.jsonl`** — one JSON object per line, 882 corpus-wide, in
PAGE-XML order (check C8).

```json
{"manuscript_id": "circular_layout", "page_id": "11", "image_width": 2729, "image_height": 3500,
 "region_id": "region_0", "region_index": 0, "region_custom": "textbox_label_0", "textbox_label": 0,
 "line_id": "region_0_line_0", "line_custom": "structure_line_id_0", "line_numeric_id": 0,
 "line_kind": "horizontal_straight", "is_closed": false,
 "baseline_length_px": 273.46697191014505, "baseline_point_count": 8, "polygon_point_count": 125,
 "reading_direction": null, "reading_cut_midpoint": null,
 "orientation_action": "preserved_left_to_right",
 "text": "प्रथमयंत्रमिदम्", "text_length_nfc": 15,
 "line_image": "labels/line_images/11/textbox_label_0/line_0.jpg"}
```

Those 22 keys, in that order, are the complete record. `line_image` is a path relative to the
manuscript directory and is `null` for every `moderate_layout` record.

**`manuscripts/<manuscript>/pages.csv`** — one row per page, 31 corpus-wide.

```csv
manuscript_id,page_id,image_width,image_height,region_count,text_line_count,node_count,edge_count,graph_component_count,cycle_edge_count,text_chars_nfc,line_kinds
circular_layout,11,2729,3500,24,24,192,170,24,2,518,"{""closed_circular"": 2, ""curved_open"": 2, ""horizontal_straight"": 9, ""point"": 11}"
```

`line_kinds` is a JSON object embedded in a CSV field; parse it with a JSON parser after CSV
unquoting.

**`dataset_manifest.json`** — schema `manuscript_layout_dataset/manifest-2`. Per manuscript:
`manuscript_id`, `source` (section 7), `page_count`, `page_ids`, `page_rasters_included`,
`text_region_count`, `text_line_count`, `line_kind_counts`,
`reading_direction_annotation_count`, `empty_transcription_line_count`,
`transcribed_chars_nfc`, `transcribed_devanagari_chars`, `node_count`, `edge_count`,
`graph_component_count`, `cycle_edge_count`. Plus corpus `totals` and `split_protocol`.

**`manuscripts/<manuscript>/processing_settings.json`** — the preprocessing parameters,
`target_longest_side: 3500` and `min_distance: 20`, identical across manuscripts. Other keys
may appear and are not load-bearing.

---

## 6. Coordinate spaces

Three spaces, two conversions. This is the most common source of error.

| Space | Unit | Where it appears |
| --- | --- | --- |
| **raster** | `inputs/` pixels, `(W, H)` | all PAGE-XML `Coords` and `Baseline`; `imageWidth`/`imageHeight`; `lines.jsonl` geometry; L6 unwrap-frame bounds and reading-direction cut points |
| **heatmap** | `(W/2, H/2)` | `heatmaps/`; all node coordinates in `labels/graph/`; `dims.txt` |
| **normalized** | heatmap ÷ `max` of the two `dims.txt` values | `<page>_inputs_normalized.txt` |

```text
    raster  =  2 x heatmap
normalized  =  heatmap / max(dims.txt)
```

The normalization is isotropic, by the longest side, not per-axis: aspect ratio survives it and
the page aspect ratio remains recoverable from `dims.txt`.

### 6.1 A worked example

`circular_layout` page 11.

```text
labels/graph/11_dims.txt     "1364.5 1750.0"
heatmaps/11.jpg              1364 x 1750   (floor of dims)
inputs/11.jpg                2729 x 3500   (2 x dims, exactly)
labels/page_xml/11.xml       imageWidth="2729" imageHeight="3500"

labels/graph/11_inputs_unnormalized.txt   row 0:  674.999500 294.000000 0.000000
labels/graph/11_inputs_normalized.txt     row 0:    0.385714   0.168000 0.000000

max(dims)       = max(1364.5, 1750.0) = 1750.0
674.9995 / 1750 = 0.3857140…  ->  0.385714   ✓
     294 / 1750 = 0.168                      ✓
raster position = 2 x (674.9995, 294.0) = (1349.999, 588.0)
```

`dims.txt` holds half the raster exactly, so it may be fractional. The heatmap image is the
floor in each dimension; the raster is `int(2 × dims)`. Twelve of the thirty-one pages have a
fractional `dims.txt` entry. Fractional heatmap coordinates such as `674.999500` are normal:
node coordinates were authored in raster space and stored at half scale. Checks C1–C3 pin the
four page sizes to each other; C4 verifies the normalization to `atol=1e-5` on every page, and
the observed maximum deviation is 5 × 10⁻⁷.

### 6.2 The one asymmetry

Reading-direction cut coordinates are in raster space while `component_node_indices` in the
same record indexes the heatmap-space node array. For each of the 25 annotations the
`cut_midpoint` lies within 58 px (median 19 px) of the resolved line's PAGE-XML baseline, which
is raster space; read as heatmap coordinates it is at least 359 px from every point of that
baseline. On page 11, `topology.reading_cut_point` is `(1308.57, 790.71)` and the ring's
baseline begins at `1314,779` — the same place.

---

## 7. Rights, sources and citation

### 7.1 The three sources

Work, genre, holding institution and page count are in section 1.1. The rights position:

| | `moderate_layout` | `dense_layout` | `circular_layout` |
| --- | --- | --- | --- |
| Author / date | Unknown / Unknown | Unknown / Unknown | Unknown / Unknown |
| Rights basis | Copyright retained by the holding institution; research use permitted, redistribution not permitted | CC0 1.0 Universal Public Domain Dedication, as recorded in the item metadata | CC0 1.0 Universal Public Domain Dedication, as recorded in the item metadata |
| Page rasters | withheld | shipped | shipped |
| BibTeX key | `Yajna` | `Muhurta` | `Tantra` |

Source URLs, archive identifiers and the per-manuscript record are in
`manuscripts/<manuscript>/SOURCE.md`, in the `source` block of each manuscript entry in
`dataset_manifest.json`, and in `SOURCES.bib`.

The genres are not decorative. A dharmaśāstra text is continuous prose in long lines; a jyotiṣa
manual carries tables, marginal computations and interlinear glosses; a tantra with yantra and
mantra uddhāra carries diagrams whose cells and rims are themselves written in. The layout
regimes follow from the genres.

### 7.2 The withheld rasters

`moderate_layout` page rasters are under third-party copyright. The holding institution permits
research use but not redistribution, so `manuscripts/moderate_layout/inputs/` contains two
files instead of images.

`SCRAPE.md` gives the source URL, the 15 page filenames used (`233_0002.jpg` …
`233_0016.jpg`), and a functional specification for a Playwright-based scraper — viewer
interaction, page-count discovery, selectors, pacing, retry, resume, duplicate protection,
error isolation, validation checklist. It specifies behaviour, not an implementation.

`RASTER_MANIFEST.json` records the derivation recipe, the reference Pillow version, and per
page the dimensions and two SHA-256 digests:

```json
{
  "schema": "withheld_raster_manifest-1",
  "manuscript_id": "moderate_layout",
  "target_longest_side": 3500,
  "derivation": "PIL open -> LANCZOS downscale only if max(w, h) > 3500 -> convert to RGB if mode in {RGBA, P, LA} -> save as JPEG at Pillow's default quality",
  "reference_pillow_version": "11.3.0",
  "pages": [
    { "page_id": "233_0002", "filename": "233_0002.jpg", "width": 2500, "height": 940,
      "source_sha256": "495cc331e33c2a8f2be38210a69b15dca300c9a829071c5f3df091ca82a5c195",
      "source_bytes": 280341,
      "derived_sha256": "ff18358897c072e121dd4ec6d86455fcde4ab11a1b823ff50ee7cc4f009577ce",
      "derived_bytes": 364164 },
    …
  ]
}
```

Two digests rather than one, so a failure is diagnosable: a mismatch on `source_sha256` means a
different scan, a mismatch on `derived_sha256` alone means the right scan and a different
encoder. All 15 pages reproduce byte-exactly under Pillow 11.3.0 and 12.3.0; no page exceeds
3500 px, so the derivation reduces to a JPEG re-encode, which is why the result is that stable.

The `moderate_layout` heatmaps **are** shipped. A half-resolution 8-bit scalar
character-likelihood field is not a reproduction of the page — the manuscript cannot be read
from it and the page cannot be reconstructed from it — and the polygon and crop stages cannot
run without it. A deliberate judgement, not an oversight; delete the directory if you disagree,
nothing else depends on those files.

### 7.3 Citation

Cite the source manuscript whose pages you used, alongside the dataset itself, not instead of
it. `SOURCES.bib` holds one `@misc` entry per manuscript, keyed `Yajna`, `Muhurta` and
`Tantra`.

The licence for the annotation layer — layout graph, region labels, transcriptions and reading
directions — is not yet fixed.

---

## 8. Evaluation protocol

`folds/<manuscript>.json` ships five folds per manuscript with `train_size = 3` and
`seed = 42`, produced by five successive `random.Random(42).sample(sorted(page_ids), 3)` draws
from one shared random stream, with each fold's complement as its test set.

```json
{
  "schema_version": 1, "manuscript_id": "circular_layout", "split_seed": 42,
  "train_size": 3, "fold_count": 5,
  "page_ids": ["11", "14", "15", "28", "35", "49", "56", "64", "65"],
  "folds": [
    { "fold_id": "fold_1", "train_page_ids": ["14", "11", "49"],
      "test_page_ids": ["15", "28", "35", "56", "64", "65"] },
    …
    { "fold_id": "fold_4", "train_page_ids": ["65", "14", "35"],
      "test_page_ids": ["11", "15", "28", "49", "56", "64"] },
    …
  ]
}
```

**They are not partitioned k-folds.** Each fold is an independent draw: training sets overlap
and a page appears in several test sets — in `circular_layout`, page 14 is in the training set
of four of the five folds. Per-fold scores are therefore not independent, and pooling them as
if they were produces confidence intervals that are too narrow. Resample over
`(manuscript_id, page_id)` clusters, taking every appearance of a page together, or state
clearly what you did instead.

**Fold order is ladder order.** `train_page_ids` is ordered so that a 1/2/3-page ladder uses the
first, then the first two, then all three. Do not sort it.

**Three training pages is the whole budget.** These folds describe a low-supervision regime,
not large-scale training.

Check C9 verifies that every fold's train and test sets partition that manuscript's page set.

---

## 9. Integrity and verification

`tools/verify_dataset.py` runs 842 checks and currently passes all of them, in two independent
classes.

**Integrity.** Every file listed in `CHECKSUMS.sha256` exists and hashes to the recorded digest,
and no unlisted file is present: 915 files, 0 mismatched, 0 missing. The listing covers
`DATASET.md` itself, so editing this file reports a mismatch until `tools/build_release.py`
regenerates it.

```text
…
0cf687267335eea339c33a621e8c704d8752244d385071c9bf30f163972166ed  SOURCES.bib
7dadd405d17bd679a3fae93ae50d8b43f0a1432ead6fa67d8db79e63b4c228a7  dataset_manifest.json
2d4a7ba85dda3b37c7c03c3e6d28e9d1297182a8eb5fa4cb69b5a2afcfd39ae0  folds/circular_layout.json
c15534d456850ac62f928d91343641379bde93d835787689e3edf4a33892d8f2  folds/dense_layout.json
…
```

**Consistency.** Nine invariants tie the layers together. If one fails, some layer has been
edited out of step with the others.

| Id | Invariant | What it protects |
| --- | --- | --- |
| **C1** | heatmap image size == `int` of `labels/graph/<page>_dims.txt` | The heatmap belongs to this page and this scale. |
| **C2** | PAGE `imageWidth`/`imageHeight` == `int(2 × dims)` | The `raster = 2 × heatmap` conversion holds page by page. |
| **C3** | input raster size == PAGE `imageWidth`/`imageHeight`, where shipped | The shipped image is the one the coordinates were written against. |
| **C4** | normalized nodes == unnormalized nodes ÷ `max(dims)`, `atol=1e-5` | The normalization is isotropic and by the longest side. |
| **C5** | the node array has one row per entry in both label files | Node indices, text-line labels and region labels are aligned. |
| **C6** | distinct text-line labels == PAGE `TextLine` count | Text lines are the graph's components; L3 and L5 have not drifted. |
| **C7** | `#edges − (#nodes − #components)` == `closed_circular` count | The only cycles in the corpus are the closed circular lines. |
| **C8** | `lines.jsonl` has one record per PAGE `TextLine`, in PAGE order, and every referenced crop exists | The derived index is not stale and its paths resolve. |
| **C9** | every fold's train and test sets partition that manuscript's page set | The evaluation protocol applies to the pages actually shipped. |

C6 and C7 read the component count from the shipped text-line label file rather than
recomputing it. That the labels induce exactly the connected-component partition of the edge
set was verified independently on all 31 pages, and is trivial to re-verify: run
`scipy.sparse.csgraph.connected_components` over `<page>_edges.txt` and compare the partition
to `<page>_labels_textline.txt`.

---

## 10. Known limitations

**10.1 Thirty-one pages.** Three manuscripts, 31 pages, 882 text lines; per-manuscript
conclusions rest on 7 to 15 pages. Differences of a few percent between systems on one
manuscript are not resolvable at this size. This is a regime study, not a large-scale benchmark.

**10.2 The folds overlap.** Training sets share pages, test sets share pages, and per-fold
results are correlated. Any procedure treating the five folds as five independent trials is
mis-specified (section 8).

**10.3 The polygons are generated, and 15% are not simple.** `TextLine/Coords` was produced by
`local_polygons_stable_unwrap_v1` from the human baseline and the CRAFT heatmap and has never
been drawn or checked by hand. 132 of 882 lines (16 / 75 / 41) self-intersect. A further 75
lines carry `fallback_used = true`, meaning the polygon fell back to a band around the baseline
because no heatmap component could be assigned; those are the weakest geometry in the release
and are concentrated in `circular_layout` (57 of 75). Apply a repair policy for geometric
metrics and report which; filter on `fallback_used` when polygon quality matters.

**10.4 Fourteen lines have no transcription** (7 / 2 / 5), retained deliberately. Any text
metric must decide explicitly what an empty reference means.

**10.5 Region structure is a human label in only two of three manuscripts.**
`circular_layout` received no region labelling: all 255 of its regions were assigned
automatically, one per connected component. Its one-line-per-region statistic is a fact about
the assignment rule, not about the manuscript, and no region-grouping task should be evaluated
on it. `moderate_layout` and `dense_layout` carry human region labels.

**10.6 Reading directions exist only in `circular_layout`.** The other two manuscripts contain
9 `curved_open` lines between them with no explicit direction. Where a `curved_open` or
`closed_circular` line has no `reading_direction_annotation`, its reading orientation is not
determined by this dataset.

**10.7 Smaller sharp edges.**

- **`moderate_layout` is unusable for pixel-level work until rebuilt.** No page rasters and no
  line crops are shipped for it; object-level and text-level work is unaffected, but pixel
  metrics, crop-based recognition and visual inspection require `tools/prepare_images.py`
  first (section 7.2).
- **One node has degree 3**, on `dense_layout` page 23. Models assuming each text line is a
  simple path meet exactly one violation.
- **172 lines are single nodes.** A `point` line has no direction, no extent and a
  single-coordinate baseline. Metrics dividing by line length, and any orientation estimator,
  must handle them explicitly. They are 49% of `circular_layout`.
- **Region `Coords` can be degenerate**, collapsing to a repeated point.
- **The heatmaps are JPEG-compressed**: a lossy 8-bit record of a float field.
- **Page id formats differ by manuscript.** Sort as strings (section 3.1).
- **Node arrays carry a third column that is always `0.0`.** Not missing data.

---

## 11. Provenance

This tree is generated in full by `tools/build_release.py` from the authoring tree, and by
nothing else; every inclusion, exclusion and transformation is in that script, which ships here
for exactly that reason. The build renames the manuscript identifiers and directories
(section 3.2), copies the label layers verbatim, drops the excluded trees, rewrites the
line-segmentation metadata into `labels/line_geometry/`, derives `lines.jsonl` and `pages.csv`,
and regenerates the folds with the reference splitter.

| Excluded from the authoring tree | Reason |
| --- | --- |
| `images/` | Pre-resize scans. `inputs/` is the raster every released coordinate is defined on. |
| `layout_analysis_output/images_resized/` | Byte-identical duplicate of `inputs/`. |
| `layout_analysis_output/text_recovery_backups/` | Editor undo snapshots. No scientific content. |
| `gnn-dataset/` | Raw pre-correction CRAFT node proposals. Not ground truth. Its `_dims.txt` is duplicated byte-for-byte inside `labels/graph/`. |
| `node_corrections/` | Per-page counts of what the annotator added and removed: a correction delta, not a label. |
| `layout_analysis_output/layout_effort.json` | Annotation timing and edit telemetry. |
| Per-line polygon-builder telemetry | Diagnostic fields and absolute paths from the authoring machine. `labels/line_geometry/` keeps the fields a consumer reads. |
| `labels/line_images/` for `moderate_layout` | Those crops are the copyrighted image, in pieces. |

`dataset_manifest.json` carries the machine-readable inventory under schema
`manuscript_layout_dataset/manifest-2`; `CHECKSUMS.sha256` carries a digest for all 915 shipped
files. Regenerating the release from an unchanged authoring tree reproduces both.

This is the first release. There is no prior version to diff against.
