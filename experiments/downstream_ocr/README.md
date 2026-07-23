# Downstream OCR Comparison Harness

This package is an isolated experiment harness for comparing OCR methods on
manuscripts shaped like `app/input_manuscripts/yajn`.

It does not modify the production Flask app. Local OCR and fine-tuning methods
import the existing production helpers from `app/recognition`, then write all
run artifacts under the requested experiment output directory.

Local OCR inference also shares production decoded-text auto-orientation. For
unannotated `curved_open` and `closed_circular` unwrap crops, identity and
180-degree-rotated predictions are compared using Devanagari evidence without
using model confidence. The selected transform and both candidate predictions
are retained in each inference result's per-line `auto_orientation` metadata.
When PAGE XML already contains a persisted selected transform, shared dataset
preparation applies it to the line image before pairing that image with its
Unicode label and does not run auto-orientation on the already-oriented crop.
Explicit reading-direction annotations remain authoritative: shared preparation
ignores any stale persisted auto transform when an annotation is present.

## Initial Supported Workflows

Validate strict PAGE-XML geometry before any expensive run:

```powershell
conda run -n gnn_layout python -m experiments.downstream_ocr.cli validate-dataset `
  --manuscript-root app\input_manuscripts\yajn `
  --output-json app\tests\logs\downstream_ocr_yajn_validation.json
```

Validate with the experiment repair policy used by evaluation:

```powershell
conda run -n gnn_layout python -m experiments.downstream_ocr.cli validate-dataset `
  --manuscript-root app\input_manuscripts\yajn `
  --repair-geometry `
  --output-json app\tests\logs\downstream_ocr_yajn_validation_repaired.json
```

Repair mode is in-memory during evaluation. It follows the production crop
boundary in spirit: invalid PAGE `Coords` are rasterized as filled contours in
page coordinates, valid foreground contours are extracted, and those repaired
geometries are used for IoU, masks, and Page CER reading order. TextEdit reads
only direct `TextLine/TextEquiv/Unicode` transcriptions and never uses geometry.

Prepare production-style GT-layout OCR crops for a small smoke check:

```powershell
conda run -n gnn_layout python -m experiments.downstream_ocr.cli prepare-gt-layout `
  --manuscript-root app\input_manuscripts\yajn `
  --output-root app\tests\logs\downstream_ocr_yajn_prepare_smoke `
  --page-id 233_0002
```

This uses GT PAGE baselines plus the resized image and heatmap to regenerate
final `Coords`, line-segmentation metadata, and rectangular OCR line images
through the same production crop boundary used by the app.

Write only the reproducibility snapshot:

```powershell
conda run -n gnn_layout python -m experiments.downstream_ocr.cli snapshot-env `
  --output-root app\tests\logs\downstream_ocr_yajn_snapshot
```

Every run command also writes `<output-root>/reproducibility.json` with the
local git commit and dirty status, Python/runtime details, tracked dependency
versions, VCS direct-url metadata when available, and SHA-256 hashes for the
OCR/GNN checkpoint artifacts used by the harness.

Evaluate local OCR with GT layout, with and without the retained production
active-learning fine-tuning recipe:

```powershell
conda run -n gnn_layout python -m experiments.downstream_ocr.cli run-local-gt-layout `
  --manuscript-root app\input_manuscripts\yajn `
  --output-root app\tests\logs\downstream_ocr_yajn
```

Bounded one-page OCR smoke:

```powershell
conda run -n gnn_layout python -m experiments.downstream_ocr.cli run-methods `
  --manuscript-root app\input_manuscripts\yajn `
  --output-root app\tests\logs\downstream_ocr_yajn_ocr_smoke `
  --method-id annotation_tool_gt_layout `
  --fold-id fold_1 `
  --max-test-pages 1 `
  --write-diagnostics
```

Evaluate already generated PAGE-XML predictions:

```powershell
conda run -n gnn_layout python -m experiments.downstream_ocr.cli evaluate-existing `
  --manuscript-root app\input_manuscripts\yajn `
  --predictions-root path\to\prediction-tree `
  --method-id gemini_e2e `
  --output-root app\tests\logs\downstream_ocr_yajn_eval `
  --write-diagnostics
```

Convert saved Gemini/VLM JSON outputs into PAGE-XML and evaluate them:

```powershell
conda run -n gnn_layout python -m experiments.downstream_ocr.cli adapt-vlm-json `
  --manuscript-root app\input_manuscripts\yajn `
  --json-root path\to\raw-json `
  --output-root app\tests\logs\downstream_ocr_yajn_vlm_json `
  --method-id gemini_e2e `
  --write-diagnostics
```

The JSON root may be flat (`<json-root>/<page_id>.json`) or organized by
fold/method (`<json-root>/<fold_id>/<method_id>/<page_id>.json`). Adapter
failures and missing JSON files are written as empty PAGE predictions and
retained in the per-page `status`.

Run explicit methods across all standard folds:

```powershell
conda run -n gnn_layout python -m experiments.downstream_ocr.cli run-methods `
  --manuscript-root app\input_manuscripts\yajn `
  --output-root app\tests\logs\downstream_ocr_yajn `
  --method-id annotation_tool_gt_layout `
  --method-id annotation_tool_pred_layout_ft_1 `
  --method-id annotation_tool_gt_layout_ft_1 `
  --write-diagnostics
```

The local annotation-tool comparison contains eight variants:

- `annotation_tool_e2e`: predicted layout, base OCR checkpoint
- `annotation_tool_gt_layout`: human-corrected test layout, base OCR checkpoint
- `annotation_tool_pred_layout_ft_1/2/3`: human-corrected graph layout and Unicode text on 1/2/3 training pages, followed by fold-local GNN plus OCR fine-tuning and OCR inference on test layouts predicted by the matching fine-tuned GNN checkpoint
- `annotation_tool_gt_layout_ft_1/2/3`: the same 1/2/3-page checkpoints, followed by OCR inference on human-corrected test layouts

Within each fold, the harness trains one sequential 1/2/3-page OCR checkpoint
ladder and, when a predicted-layout fine-tuning method is requested, one
sequential GNN checkpoint ladder. Each corrected GNN training page is augmented
50 times with `src/configs/augment.yaml`. The current page's 50 variants plus a
deterministic 20% sample of every earlier page's variants are fine-tuned with
the unchanged model object; only fold training pages are used for training or
checkpoint selection. Continuation hyperparameters and the expected serialized
model/backbone classes are pinned separately in
`configs/gnn_finetuning.yaml`; the architecture is never reconstructed from
the generic GNN training config.

For GNN fine-tuning only, original CRAFT nodes missing from the corrected
graph are recovered with the configured image-space matching tolerance and
appended with text-line label `-1`. The ground-truth builder skips that label,
so every generated candidate edge incident to a recovered deleted node has
binary target `0`. Corrected unmatched nodes (including manual additions) are
left untouched. This does not change inference: isolated nodes are not deleted
or filtered.

At a given fine-tuning depth, the predicted-layout and corrected-layout test
variants still reuse the exact same OCR checkpoint. Human-corrected test
layouts bypass GNN inference. Predicted test layouts are regenerated at each
depth with that depth's fold-local GNN checkpoint. `annotation_tool_e2e`
continues to use the immutable pretrained GNN and OCR checkpoints.

One-fold, three-manuscript joint fine-tuning test:

```powershell
conda run -n gnn_layout python -m experiments.downstream_ocr.cli run-methods `
  --manuscript-root app\input_manuscripts\yajn `
  --manuscript-root app\input_manuscripts\dense `
  --manuscript-root app\input_manuscripts\circle_new `
  --output-root app\tests\logs\GNN_finetune_1_fold_test `
  --fold-id fold_1 `
  --method-id annotation_tool_e2e `
  --method-id annotation_tool_pred_layout_ft_1 `
  --method-id annotation_tool_pred_layout_ft_2 `
  --method-id annotation_tool_pred_layout_ft_3
```

## One-Time VLM Pre-Prediction

Paid VLM inference is a separate acquisition phase. Acquire every page once,
before creating or selecting folds:

Existing `gnn_layout` environments created before Sarvam support need the new
SDK once:

```powershell
conda run -n gnn_layout python -m pip install sarvamai
```

```powershell
conda run -n gnn_layout python -m experiments.downstream_ocr.cli prepredict-vlms `
  --manuscript-root app\input_manuscripts\yajn `
  --manuscript-root app\input_manuscripts\dense `
  --manuscript-root app\input_manuscripts\circle_new `
  --output-root app\tests\logs\downstream_ocr_vlm_cache `
  --provider-id gemini `
  --provider-id openai `
  --provider-id claude `
  --provider-id sarvam
```

The enabled provider methods are:

- `gemini_e2e`: `gemini-3.5-flash`
- `openai_e2e`: `gpt-5.6-terra`
- `claude_e2e`: `claude-sonnet-5`
- `sarvam_e2e`: Sarvam Vision Document Digitization (`sa-IN`, HTML)

Gemini, OpenAI, and Claude receive the same resized page image followed by the
exact `VLM_END_TO_END_PROMPT` string from `adapter.py`. Sarvam receives one
resized page image per Document Digitization job with `language=sa-IN` and
`output_format=html`; that API does not receive the shared prompt and does not
return line geometry. Keys are read from `app/.env` through `GEMINI_API_KEY`,
`OPENAI_API_KEY`, `CLAUDE_API_KEY`, and `SARVAM_API_KEY`. Keys and image bytes
are never written to the cache.

Sarvam HTML is converted immediately to PAGE-XML. Semantic text blocks such as
`p`, headings, list items, `header`, `footer`, `aside`, table cells, and
Sarvam's text-bearing classes (`sidebar`, `folio`, `formula`, and related
classes) end a text line. Each `br` inside a block also ends a text line. Nested
text blocks are emitted once rather than duplicated through their containers.
Content under `head`, `style`, `script`, templates, and SVG is ignored. Empty
fragments are discarded while DOM order and Unicode text are preserved. The
resulting page has one `TextRegion`; every `TextLine` contains
`TextEquiv/Unicode` plus empty `Coords` and `Baseline`. Because Sarvam supplies
no geometry, only the two unordered TextEdit metrics are evaluated for
`sarvam_e2e`; Page CER and layout metrics are explicitly unavailable.

The cache is immutable at the page/request level. Its request fingerprint pins
the provider, exact model, prompt or document-job parameters, image bytes,
template PAGE-XML, and the Sarvam output adapter where applicable.
Successful outputs and failures after retry exhaustion are both terminal. A
second acquisition command validates and reuses them without an API call.
An interrupted non-terminal page directory is refused because automatically
retrying it could duplicate a paid request. Use a new cache root after manual
inspection.

`--max-retries 3` means three retries after the initial attempt. Raw responses,
attempt metadata, normalized JSON when applicable, PAGE-XML, usage metadata,
and terminal status are retained per page.

DeepSeek V4-Flash is deliberately not registered: the official DeepSeek API
documents V4 as text-only and rejects image content.

Acquire Sarvam once for all three manuscripts, with up to eight active page
jobs. `--max-retries 3` applies the same policy as Gemini, OpenAI, and Claude:
one initial attempt plus up to three retries. Once a terminal result is cached,
later fold and report runs do not call Sarvam again:

```powershell
conda run -n gnn_layout python -m experiments.downstream_ocr.cli prepredict-vlms `
  --manuscript-root app\input_manuscripts\yajn `
  --manuscript-root app\input_manuscripts\dense `
  --manuscript-root app\input_manuscripts\circle_new `
  --output-root app\tests\logs\downstream_ocr_vlm_cache `
  --provider-id sarvam `
  --page-workers 8 `
  --timeout-seconds 600 `
  --request-spacing-seconds 6 `
  --max-retries 3
```

After that cache is complete, add `sarvam_e2e` to the existing five-fold run
and regenerate all per-manuscript and combined reports:

```powershell
conda run -n gnn_layout python -m experiments.downstream_ocr.cli run-methods `
  --manuscript-root app\input_manuscripts\yajn `
  --manuscript-root app\input_manuscripts\dense `
  --manuscript-root app\input_manuscripts\circle_new `
  --output-root app\tests\logs\ocr_5fold_new `
  --vlm-predictions-root app\tests\logs\downstream_ocr_vlm_cache `
  --method-id sarvam_e2e
```

The existing `folds.json` files are reused. Only Sarvam is materialized and
evaluated by this command; retained metrics for the other methods remain in
place and are included when each report is rebuilt.

Run folds strictly offline by pointing `run-methods` at the completed cache:

```powershell
conda run -n gnn_layout python -m experiments.downstream_ocr.cli run-methods `
  --manuscript-root app\input_manuscripts\yajn `
  --manuscript-root app\input_manuscripts\dense `
  --manuscript-root app\input_manuscripts\circle_new `
  --output-root app\tests\logs\downstream_ocr_all_manuscripts `
  --vlm-predictions-root app\tests\logs\downstream_ocr_vlm_cache `
  --method-id gemini_e2e `
  --method-id openai_e2e `
  --method-id claude_e2e `
  --method-id sarvam_e2e `
  --method-id annotation_tool_e2e
```

Before doing local OCR work, `run-methods` validates that every requested VLM
cache contains the exact current page set and matching fingerprints. It then
copies only each fold's test-page predictions into that fold's run directory.
This command never loads an API key or performs a VLM network request.

Cache validation is strict by default, including the PAGE-XML ground-truth
hash. If human PAGE-XML annotations changed but the page images are byte-for-byte
identical, pass `--allow-vlm-pagexml-drift`. This opt-in permits only the
PAGE-XML hash to differ: page set, image hash, provider/model, prompt, request
contract, cached terminal record, and PAGE image filename/dimensions/namespace
must still match. The resulting metric payload records
`preprediction_cache.pagexml_drift_allowed=true`.

Each evaluation command writes an automatic report under:

```text
<output-root>/report/
```

The report folder includes:

- `experiment_report.md`
- `summary_metrics.csv` and `summary_metrics.json`
- `fold_metrics.csv` and `fold_metrics.json`
- `per_page_metrics.csv`
- `layout_mode_comparisons.csv` and `layout_mode_comparisons.json`
- `table_1_off_the_shelf_models.csv` and `table_1_off_the_shelf_models.json`
- `table_2_annotation_tool_gains.csv` and `table_2_annotation_tool_gains.json`
- `vlm_usage.csv` and `vlm_usage.json`
- `textedit_metric.json` and `devanagari_textedit_metric.json`
- figures under `figures/`

The former `micro_page_cer_by_method.png` and `micro_textedit_by_method.png`
bar figures are no longer generated. They are replaced by two paper tables:

- Table 1 contains the enabled Gemini, OpenAI, Claude, and Sarvam off-the-shelf
  methods. Each row records its prompt/input contract; all rows use the same
  manuscript folds and test pages.
- Table 2, annotation-tool gains, compares 0/1/2/3-page fine-tuning with and
  without GT layout correction. It records Micro Page CER, upstream TextEdit,
  Devanagari-relaxed TextEdit, and active Layout Mode correction seconds per
  evaluated page when `layout_effort.json` is available.

Deterministic 95% page-cluster bootstrap confidence intervals use the unique
`(manuscript_id, page_id)` as the cluster unit, so repeated appearances of a
page across folds are resampled together. Read Mode fine-tuning effort is
intentionally not included in the layout timing estimate.

VLM methods record provider usage metadata when available. Each acquired
provider/page result is counted once, even when several folds reuse it.
Annotation-tool methods use local computation and have zero VLM API cost. USD
estimates are blank unless provider-specific rates are supplied:

```powershell
$env:GEMINI_INPUT_USD_PER_1M_TOKENS="..."
$env:GEMINI_OUTPUT_USD_PER_1M_TOKENS="..."
$env:OPENAI_INPUT_USD_PER_1M_TOKENS="..."
$env:OPENAI_OUTPUT_USD_PER_1M_TOKENS="..."
$env:CLAUDE_INPUT_USD_PER_1M_TOKENS="..."
$env:CLAUDE_OUTPUT_USD_PER_1M_TOKENS="..."
```

You can regenerate only the report for an existing run:

```powershell
conda run -n gnn_layout python -m experiments.downstream_ocr.cli write-report `
  --output-root app\tests\logs\downstream_ocr_yajn
```

To recompute both TextEdit metrics from retained PAGE-XML predictions and then
rebuild every manuscript and combined report under an existing
multi-manuscript run:

```powershell
conda run -n gnn_layout python -m experiments.downstream_ocr.cli refresh-textedit `
  --output-root app\tests\logs\ocr_5fold_new
```

This command preserves all non-TextEdit per-page and aggregate metric values.
It leaves the upstream TextEdit definition unchanged and adds or refreshes the
Devanagari-relaxed fields. Missing prediction XML is evaluated as an empty
prediction rather than skipped.

You can prepare the two paper tables for multiple existing manuscript runs at
once:

```powershell
conda run -n gnn_layout python -m experiments.downstream_ocr.cli write-combined-report `
  --input-root app\tests\logs\downstream_ocr_yajn `
  --input-root app\tests\logs\downstream_ocr_dense `
  --input-root app\tests\logs\downstream_ocr_circle_new `
  --output-root app\tests\logs\downstream_ocr_combined_tables
```

Alternatively, `run-methods` accepts repeated manuscript roots and writes a
combined report after the per-manuscript runs finish:

```powershell
conda run -n gnn_layout python -m experiments.downstream_ocr.cli run-methods `
  --manuscript-root app\input_manuscripts\yajn `
  --manuscript-root app\input_manuscripts\dense `
  --manuscript-root app\input_manuscripts\circle_new `
  --output-root app\tests\logs\downstream_ocr_all_manuscripts `
  --vlm-predictions-root app\tests\logs\downstream_ocr_vlm_cache `
  --method-id gemini_e2e `
  --method-id openai_e2e `
  --method-id claude_e2e `
  --method-id sarvam_e2e `
  --method-id annotation_tool_e2e `
  --method-id annotation_tool_gt_layout `
  --method-id annotation_tool_pred_layout_ft_1 `
  --method-id annotation_tool_gt_layout_ft_1 `
  --method-id annotation_tool_pred_layout_ft_2 `
  --method-id annotation_tool_gt_layout_ft_2 `
  --method-id annotation_tool_pred_layout_ft_3 `
  --method-id annotation_tool_gt_layout_ft_3
```

The existing prediction tree must be organized as:

```text
<predictions-root>/<fold_id>/<method_id>/<page_id>.xml
```

## Metrics

The evaluator implements:

- Object Precision/Recall/G-F1 at IoU 0.50 and 0.75
- Pixel Precision/Recall/F1 from polygon masks
- Page CER
- Upstream-compatible unordered PAGE-XML TextLine TextEdit using OmniDocBench
  v1.5 `simple_match`
- Devanagari-relaxed unordered PAGE-XML TextLine TextEdit
- pooled fold/manuscript aggregates and valid output rate

Failed model output is represented as an empty PAGE prediction and must keep
its failure `status` in the per-page record.

Upstream TextEdit treats each PAGE `TextLine` as one atomic unit. It extracts
one direct `TextEquiv/Unicode` transcription per line, applies the official
`textblock2unicode` and `clean_string` normalization, computes all pairwise
normalized edit costs, and uses the official Hungarian `simple_match`.
Coordinates, baselines, XML order, reading order, region membership, region
labels, IDs, image dimensions, and all geometry are ignored. The primary score
is official `Edit_dist.ALL_page_avg`, the mean of page-level ratios;
`edit_whole` and `edit_sample_avg` are retained as supplementary outputs.

Devanagari-relaxed TextEdit keeps the same unordered, one-to-one line-matching
idea but uses NFC and keeps all Unicode letters, combining marks, and numbers,
plus danda (`।`), double danda (`॥`), and the Devanagari abbreviation sign
(`॰`). This preserves vowel signs, virama, anusvara, visarga, nukta, and accent
marks that the upstream `clean_string` can remove. It ignores whitespace, most
punctuation, symbols, and format controls such as ZWJ/ZWNJ. Its primary score is
also an unweighted mean of page-level edit ratios. The original upstream
TextEdit fields remain unchanged for compatibility.

The pinned source is OmniDocBench branch `v1_5`, commit
`59b103c4b47d3a01fada83491585d6512a40c0bc`. The minimal vendored source,
Apache-2.0 license, modification notice, and upstream manifest are under
`omnidocbench_v1_5/`. The adapter is registered as
`pagexml2pagexml_dataset`; its example configuration is
`configs/pagexml_textedit.yaml`.

When `--write-diagnostics` is passed, evaluated pages get polygon overlay and
pixel-mask overlap PNGs under `<output-root>/diagnostics/`.

## VLM Adapter

`adapter.py` defines the VLM end-to-end prompt and a JSON-to-PAGE adapter that
accepts either `polygon_2d` or `box_2d`. Curved and circular lines should use
`polygon_2d`.
