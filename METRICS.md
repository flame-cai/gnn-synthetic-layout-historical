# Evaluation Metrics: PAGE-XML vs PAGE-XML

This document describes the evaluation code that scores a predicted PAGE-XML
file against a ground-truth PAGE-XML file for the same page. It covers four
families of metric:

| Metric | Reads | Ignores | Direction |
| --- | --- | --- | --- |
| TextEdit (OmniDocBench v1.5 `simple_match`) | `TextLine/TextEquiv/Unicode` | geometry, line order, regions | lower is better |
| Devanagari-relaxed TextEdit | `TextLine/TextEquiv/Unicode` | geometry, line order, regions | lower is better |
| Page CER | `TextLine/TextEquiv/Unicode` + line polygon centroids (for reading order) | regions | lower is better |
| Layout: object P/R/G-F1 at IoU 0.50 and 0.75, pixel P/R/F1 | `TextLine/Coords` | text | higher is better |

These are the metrics reported in the paper *"Impact of Iterative Fine-Tuning
on Transcription Accuracy in Complex Historical Sanskrit Manuscripts"* and used
to score the multi-modal LLM baselines shipped in
[`dataset_release/`](./dataset_release/DATASET.md).

## What Is Committed

The rest of `experiments/` stays untracked (see `.gitignore`). Only the
evaluation code is committed, which is the import closure of
`experiments/downstream_ocr/metrics.py`:

| Path under `experiments/downstream_ocr/` | Role |
| --- | --- |
| `pagexml.py` | PAGE-XML loader: `load_pagexml()` returns a `PageXmlPage` of `TextLine`s (polygon, text, ids), with optional geometry repair |
| `text.py` | NFC + whitespace normalization and Levenshtein distance (RapidFuzz when installed, pure-Python fallback otherwise) |
| `metrics.py` | Per-page `evaluate_page()`, aggregate `evaluate_pages()`, Page CER, object and pixel layout metrics |
| `dataset/pagexml2pagexml_dataset.py` | TextEdit adapter: PAGE-XML → OmniDocBench items, `evaluate_pagexml_pairs()`, and the registered `pagexml2pagexml_dataset` |
| `devanagari_textedit.py` | Devanagari-relaxed TextEdit |
| `omnidocbench_v1_5/` | Minimal vendored OmniDocBench v1.5 source (Apache-2.0); see its `NOTICE.md` and `UPSTREAM.json` |
| `configs/pagexml_textedit.yaml` | Example OmniDocBench-style config for the registered dataset |
| `tests/test_pagexml_textedit.py` | Unit tests for both TextEdit metrics |

Not committed: the experiment runners and CLI (`runners.py`, `cli.py`), VLM
acquisition and caching, fold handling, report and figure generation
(including the page-clustered bootstrap confidence intervals), and the
GNN fine-tuning shim. Production GNN fine-tuning lives in
`src/gnn_training/gnn_finetuning.py`.

## Quick Start

Run from the repository root in the `gnn_layout` conda environment, so that
`experiments.downstream_ocr` is importable as a package.

```python
from experiments.downstream_ocr.pagexml import load_pagexml
from experiments.downstream_ocr.metrics import evaluate_page, evaluate_pages

gt = load_pagexml("path/to/gt/11.xml", repair_geometry=True)
pred = load_pagexml("path/to/pred/11.xml", repair_geometry=True)

record = evaluate_page(
    manuscript_id="circular_layout",
    fold_id="none",
    page_id="11",
    gt_page=gt,
    pred_page=pred,
)
print(record["textedit"], record["devanagari_textedit"], record["page_cer"])
print(record["object_g_f1_50"], record["object_g_f1_75"], record["pixel_f1"])

# Pool any number of page records.
aggregate = evaluate_pages([record]).aggregate
print(aggregate["mean_textedit"], aggregate["micro_page_cer"])
```

For example, scoring the released Gemini prediction for `circular_layout`
page 11 against its ground truth:

```python
root = "dataset_release"
gt = load_pagexml(f"{root}/manuscripts/circular_layout/labels/page_xml/11.xml",
                  repair_geometry=True)
pred = load_pagexml(f"{root}/multi-modal-LLM-outputs/circular_layout/gemini_e2e/pages/11/prediction.xml",
                    repair_geometry=True)
```

To score TextEdit alone over many pages, straight from file paths, without
parsing geometry:

```python
from pathlib import Path
from experiments.downstream_ocr.dataset import PageXmlPair, evaluate_pagexml_pairs

result = evaluate_pagexml_pairs([
    PageXmlPair(key="11", gt_xml_path=Path("gt/11.xml"), pred_xml_path=Path("pred/11.xml")),
    PageXmlPair(key="14", gt_xml_path=Path("gt/14.xml"), pred_xml_path=Path("pred/14.xml")),
])
result.official_result["Edit_dist"]   # ALL_page_avg, edit_whole, edit_sample_avg
result.page_metrics["11"]             # per-page TextEdit and Devanagari-relaxed fields
result.matches_by_key["11"]           # the line-to-line assignment, for inspection
```

A `pred_xml_path` that is `None` or does not exist is scored as an empty
prediction, not skipped.

Run the tests:

```bash
python -m unittest experiments.downstream_ocr.tests.test_pagexml_textedit
```

Dependencies, all in `requirements.txt` or pulled in by it: `numpy`, `scipy`,
`shapely`, `opencv-python`, `networkx`, `pandas`, `python-Levenshtein`,
`pylatexenc`, `beautifulsoup4`; `rapidfuzz` is optional.

## Input Contract

- PAGE-XML 2013-07-15 (`http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15`).
  The geometric loader is namespace-agnostic; the TextEdit file parser expects
  this namespace.
- Ground truth and prediction must share `imageWidth` and `imageHeight`.
  `evaluate_page()` raises `ValueError` otherwise; there is no rescaling.
- Lines are read from every `TextRegion/TextLine`. Region membership does not
  affect any metric.
- One transcription per line: the direct `TextEquiv` with `index="0"`,
  otherwise the smallest numeric `index`, otherwise the first in document
  order. Lines whose text is missing or whitespace-only are dropped from both
  TextEdit metrics (for ground truth and prediction alike). The file-based
  path logs each one it drops.

### Loading policy

`load_pagexml()` options decide what happens to bad geometry:

| Option | Effect |
| --- | --- |
| `strict=True` (default) | A `TextLine` without `Coords` raises |
| `strict=False` | A `TextLine` without `Coords` is skipped |
| `allow_empty_geometry=True` | A `TextLine` without `Coords` is kept with empty geometry. Use this for text-only predictions such as `sarvam_e2e` |
| `repair_geometry=True` | An invalid polygon (e.g. self-intersecting) is rasterized as a filled contour on the page grid and its valid foreground contours are re-extracted |

The harness that produced the paper's numbers loads both ground truth and
predictions with `repair_geometry=True`. That matters for the released
dataset: 132 of its 882 ground-truth polygons self-intersect (DATASET.md,
section 5.5). Report which repair policy you used. Text metrics are unaffected
by repair, because they read only `TextEquiv/Unicode`.

## Metric Definitions

### TextEdit (upstream-compatible)

An unordered, one-to-one line-matching edit distance, computed through the
vendored OmniDocBench v1.5 code (branch `v1_5`, commit
`59b103c4b47d3a01fada83491585d6512a40c0bc`).

1. Each non-empty `TextLine` becomes one atomic text item. Coordinates,
   baselines, XML order, reading order, region membership and IDs are ignored.
2. Each line is normalized with the official `textblock2unicode` and
   `clean_string`. `clean_string` keeps only Python `\w` characters and CJK.
   **On Devanagari it removes vowel signs, virama, anusvara, visarga and
   danda**: `श्रीगणेशाय नमः ।` normalizes to `शरगणशयनम`.
3. A pairwise cost matrix of normalized edit distances
   (`Levenshtein / max(len_gt, len_pred)`) is solved with Hungarian
   assignment (`simple_match`). Unmatched ground-truth lines count as full
   deletions. Unmatched predicted lines are concatenated into one insertion
   record.
4. Per page, `textedit = Σ edits / Σ max(len_gt, len_pred)` over that page's
   match records.

Aggregates, exactly as the official `call_Edit_dist` reports them:

| Field | Definition |
| --- | --- |
| `ALL_page_avg` → `mean_textedit`, `textedit_all_page_avg` | **Primary.** Unweighted mean of per-page ratios |
| `edit_whole` → `micro_textedit`, `textedit_edit_whole` | Pooled: Σ edits / Σ max-lengths over all pages |
| `edit_sample_avg` → `textedit_edit_sample_avg` | Mean of per-match-record ratios |

This metric uses PAGE TextLine granularity. It is not the paragraph-level
OmniDocBench leaderboard score. The vendored code changes one diagnostic-only
upstream bug and no part of the calculation (`omnidocbench_v1_5/NOTICE.md`).

### Devanagari-relaxed TextEdit

This uses the same unordered, one-to-one line matching with a normalization
that keeps Devanagari intact:

- NFC, then keep every Unicode letter (`L*`), combining mark (`M*`) and number
  (`N*`), plus danda `।`, double danda `॥` and the abbreviation sign `॰`.
- Whitespace, ZWJ/ZWNJ and other format controls, symbols, and all other
  punctuation are dropped.
- Lines that are empty after normalization are skipped on both sides.
- Matching is Hungarian on `Levenshtein / max(len)`. Unmatched ground-truth
  lines are deletions, and unmatched predictions are concatenated into one
  insertion record.

The primary field is `devanagari_textedit_all_page_avg` (aggregate
`mean_devanagari_textedit`), the unweighted mean of per-page ratios. The
pooled `micro_devanagari_textedit` and `devanagari_textedit_edit_sample_avg`
are reported alongside it. This metric adds to the upstream TextEdit and does
not replace it. Both are always reported.

### Page CER

This is an ordered, page-level character error rate.

1. Order lines geometrically by polygon centroid: top to bottom (`y`), then
   left to right (`x`), then by line id.
2. Normalize each line with NFC and collapse whitespace runs to one space.
   Lines that are empty after normalization are dropped.
3. Join the lines with single spaces, separately for ground truth and
   prediction.
4. `page_cer = Levenshtein(gt_text, pred_text) / len(gt_text)`. This is `0.0`
   when the ground truth has no text, and it can exceed `1.0` when the
   prediction is much longer than the ground truth.

No characters are stripped beyond whitespace, so matras, virama, danda and
punctuation all count. Because ordering uses geometry, Page CER also penalizes
line segmentation that breaks the reading order. Predictions without geometry
can pass `page_cer_predicted_lines_in_output_order=True` to keep the
prediction's XML line order instead. The harness does this for `sarvam_e2e`.
The ground truth is still ordered geometrically.

Aggregates: `micro_page_cer` (Σ distances / Σ ground-truth characters, the
headline figure), `mean_page_cer` and `median_page_cer`.

### Layout metrics

These compare `TextLine/Coords` polygons and ignore text.

**Object precision, recall and G-F1 at IoU 0.50 and 0.75.** A dense IoU
matrix is computed between every ground-truth and predicted polygon with
Shapely (an STRtree prunes non-overlapping pairs). At each threshold, pairs
with `IoU ≥ threshold` form a bipartite graph. A one-to-one matching takes the
maximum number of pairs, and among those the largest total IoU (NetworkX
`max_weight_matching` with `maxcardinality=True`). Each matched pair is a TP,
every unmatched prediction is an FP, and every unmatched ground-truth line is
an FN:

- `object_precision_t = TP / (TP + FP)`
- `object_recall_t = TP / (TP + FN)`
- `object_g_f1_t = 2TP / (2TP + FP + FN)`

**Pixel precision, recall and F1.** All ground-truth polygons, and separately
all predicted polygons, are rasterized as filled masks on the page grid.
Overlapping lines merge into one mask, so this measures region coverage, not
instance separation. TP, FP and FN are counted per pixel, and P/R/F1 use the
same formulas as above.

Aggregates pool the counts across pages (micro): `tp_50`, `fp_50`, `fn_50` →
`object_*_50`, likewise for `_75`, and `pixel_tp/fp/fn` → `pixel_*`.

Two cautions for the released dataset. The ground-truth polygons are
machine-generated from the human baseline plus the CRAFT heatmap, so a
prediction that hugs the ink more tightly than that generator scores worse,
not better (DATASET.md, section 5.5). Some methods return no geometry; for
those, pass `calculate_layout_metrics=False` and the layout fields are `None`
rather than zero.

## Per-Page Record And Aggregation

`evaluate_page()` returns one flat dict per page. It contains identifiers
(`manuscript_id`, `fold_id`, `page_id`, optional `method_id`), `status`,
`num_gt_lines` and `num_pred_lines`, plus the fields of every metric enabled
by `calculate_textedit`, `calculate_page_cer` and `calculate_layout_metrics`
(all `True` by default). Disabled metrics are written as `None`, and the
record carries `layout_metrics_available` and `page_cer_available`.

Failed predictions must stay in the record set. If `status` is one of
`metrics.FAILURE_STATUSES` (`api_timeout`, `api_error`, `empty_response`,
`json_parse_error`, `json_schema_error`, `html_parse_error`, `adapter_error`,
`other_output_error`), the prediction's lines are discarded and the page
scores as an empty prediction. Dropping failed pages instead would silently
change the denominator.

`evaluate_pages(records).aggregate` pools a list of records into the
aggregates named above. It also returns `page_count`, `valid_output_rate`
(share of records with `status == "success"`), `layout_metric_page_count`,
`page_cer_page_count` and `devanagari_textedit_page_count`. Records whose
layout or Page CER fields are `None` are excluded from those aggregates only.

`evaluate_page()` computes TextEdit per page in memory. The experiment harness
instead calls `evaluate_page(..., calculate_textedit=False)` and then
`evaluate_pagexml_pairs()` over all of a fold's pages, merging the result back
into each record. The two give the same per-page values. The batched form also
returns the official `call_Edit_dist` result and the match lists.

When aggregating over the released dataset's folds, note that they are
overlapping random draws rather than partitioned k-folds. Resample by
`(manuscript_id, page_id)` cluster for confidence intervals (DATASET.md,
section 8).

## OmniDocBench-Style Directory Evaluation

`pagexml2pagexml_dataset` is registered in the vendored `DATASET_REGISTRY`, so
the TextEdit adapter can be driven the way OmniDocBench drives its own
datasets. It pairs files by the same relative path under the ground-truth and
prediction roots, and a missing prediction file becomes an empty prediction.
See `configs/pagexml_textedit.yaml`:

```python
import yaml
from experiments.downstream_ocr.omnidocbench_v1_5.registry import DATASET_REGISTRY
import experiments.downstream_ocr.dataset  # registers pagexml2pagexml_dataset

cfg = yaml.safe_load(open("experiments/downstream_ocr/configs/pagexml_textedit.yaml"))
dataset = DATASET_REGISTRY.get("pagexml2pagexml_dataset")(cfg["end2end_eval"])
samples = dataset.samples["text_block"]   # match records for call_Edit_dist
```

## Reproducibility Metadata

`textedit_reproducibility_metadata()` (in `dataset/`) and
`devanagari_textedit_reproducibility_metadata()` return JSON-serializable
descriptions of each TextEdit metric: upstream commit, normalization,
selection and empty-line policies, matcher and aggregation field. Store them
next to any reported numbers.

## License

`experiments/downstream_ocr/omnidocbench_v1_5/` is a minimal extraction of
[OmniDocBench](https://github.com/opendatalab/OmniDocBench) under the Apache
License 2.0. Its `LICENSE` and `NOTICE.md` must travel with it.
