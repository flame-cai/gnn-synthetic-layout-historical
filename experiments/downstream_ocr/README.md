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
geometries are used for IoU, masks, reading order, and TextEdit grouping.

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
  --method-id vlm_e2e `
  --output-root app\tests\logs\downstream_ocr_yajn_eval `
  --write-diagnostics
```

Convert saved Gemini/VLM JSON outputs into PAGE-XML and evaluate them:

```powershell
conda run -n gnn_layout python -m experiments.downstream_ocr.cli adapt-vlm-json `
  --manuscript-root app\input_manuscripts\yajn `
  --json-root path\to\raw-json `
  --output-root app\tests\logs\downstream_ocr_yajn_vlm_json `
  --method-id vlm_e2e `
  --write-diagnostics
```

The JSON root may be flat (`<json-root>/<page_id>.json`) or organized by
fold/method (`<json-root>/<fold_id>/<method_id>/<page_id>.json`). Adapter
failures and missing JSON files are written as empty PAGE predictions and
retained in the per-page `status`.

Run explicit methods across all three folds:

```powershell
conda run -n gnn_layout python -m experiments.downstream_ocr.cli run-methods `
  --manuscript-root app\input_manuscripts\yajn `
  --output-root app\tests\logs\downstream_ocr_yajn `
  --method-id annotation_tool_gt_layout `
  --method-id annotation_tool_pred_layout_ft_1 `
  --method-id annotation_tool_gt_layout_ft_1 `
  --write-diagnostics
```

The local annotation-tool comparison now contains seven variants:

- `annotation_tool_e2e`: predicted layout, base OCR checkpoint
- `annotation_tool_gt_layout`: human-corrected test layout, base OCR checkpoint
- `annotation_tool_pred_layout_ft_1/2/3`: human-corrected layout and Unicode text on 1/2/3 training pages, followed by OCR inference on predicted test layouts
- `annotation_tool_gt_layout_ft_1/2/3`: the same 1/2/3-page checkpoints, followed by OCR inference on human-corrected test layouts

Within each fold, the harness trains only one sequential 1/2/3-page checkpoint
ladder from corrected training-page layouts. At a given fine-tuning depth, the
predicted-layout and corrected-layout test variants reuse the exact same
checkpoint. Predicted held-out layouts are also prepared once per fold and
shared with `annotation_tool_e2e`, so the paired comparison changes the OCR
checkpoint or test-layout condition without rerunning a different layout
prediction for each method.

Gemini-backed methods are available as `vlm_e2e` and `gemini_gt_layout`, but
they make API calls and require `GEMINI_API_KEY` in `app/.env` or the process
environment.

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
- `gemini_usage.csv` and `gemini_usage.json`
- figures under `figures/`

The main Micro Page CER and Micro TextEdit figures include deterministic 95%
page-cluster bootstrap confidence intervals. The cluster unit is the unique
`(manuscript_id, page_id)`, so repeated appearances of a page across folds are
resampled together. The figures also report paired e2e-to-GT-layout relative
error reductions for Gemini and the Annotation Tool together with mean active
Layout Mode edit seconds per unique test page and its 95% confidence interval.
Three predicted-test-layout fine-tuning compartments appear immediately after
the off-the-shelf compartment. They are followed by the test-page layout
post-correction compartment and its 1/2/3-page fine-tuned variants. Read Mode
fine-tuning effort is intentionally not included in the timing estimate.

Gemini methods record SDK usage metadata when available, including prompt,
candidate, and total token counts. Annotation-tool methods use local
computation and are reported with zero Gemini API cost. USD estimates are left
blank unless you provide Gemini rates through environment variables:

```powershell
$env:GEMINI_INPUT_USD_PER_1M_TOKENS="..."
$env:GEMINI_OUTPUT_USD_PER_1M_TOKENS="..."
```

You can regenerate only the report for an existing run:

```powershell
conda run -n gnn_layout python -m experiments.downstream_ocr.cli write-report `
  --output-root app\tests\logs\downstream_ocr_yajn
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
- Line-group TextEdit (OmniDocBench-style)
- pooled fold/manuscript aggregates and valid output rate

Failed model output is represented as an empty PAGE prediction and must keep
its failure `status` in the per-page record.

When `--write-diagnostics` is passed, evaluated pages get polygon overlay and
pixel-mask overlap PNGs under `<output-root>/diagnostics/`.

## Gemini/VLM Adapter

`adapter.py` defines the VLM end-to-end prompt and a JSON-to-PAGE adapter that
accepts either `polygon_2d` or `box_2d`. Curved and circular lines should use
`polygon_2d`.
