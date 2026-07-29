# Production GUI App

This document describes the production behavior of the semi-automatic
annotation and OCR app under `app/`. Research verifier details live in
`RESEARCH_HARNESS.md`; this file is about GUI runtime behavior, saved
manuscript state, production text-line strategy adoption, local OCR inference,
Gemini OCR integration, exports, active-learning jobs, telemetry, and
operational validation.

## Purpose And Scope

The app supports manuscript digitization with human correction in the loop:

- layout correction by adding and deleting graph nodes
- graph correction by adding and deleting edges
- text-line and text-region grouping
- optional reading-direction annotation for ambiguous lines
- PAGE XML generation
- OCR through Gemini or the local OCR checkpoint family
- manual OCR text correction
- PAGE XML, line-image, OCR-training-format, resized-image, and overlay export
- manuscript-local OCR active learning for the local OCR model

The production app is not the research harness. It may use a strategy or recipe
that was selected by research gates, but changing production behavior must be
an explicit production adoption step.

## Runtime Source Files

Important production runtime files:

- `app/app.py`
- `app/frontend/`
- `app/gnn_inference.py`
- `app/device_leases.py`
- `app/job_orchestrator.py`
- `app/manuscript_ocr_registry.py`
- `app/ocr_active_learning_runtime.py`
- `app/ocr_model_manager.py`
- `app/profiling.py`
- `app/telemetry.py`
- `app/recognition/active_learning.py`
- `app/recognition/active_learning_recipe.py`
- `app/recognition/dataset.py`
- `app/recognition/ocr_defaults.py`
- `app/recognition/pagexml_line_dataset.py`
- `app/recognition/recognize_manuscript_text_v2_pretrained.py`
- `app/recognition/train.py`
- `app/recognition/line_segmentation/ocr_crops.py`
- `app/recognition/line_segmentation/reading_direction.py`
- `app/recognition/line_segmentation/registry.py`
- `app/recognition/line_segmentation/runtime_config.py`
- `app/recognition/line_segmentation/strategy_config.py`
- `scripts/adopt_text_line_strategy_for_app.py`

Important production/runtime tests:

- `app/tests/test_download_results_export_unit.py`
- `app/tests/test_job_orchestrator_unit.py`
- `app/tests/test_line_segmentation_strategy_unit.py`
- `app/tests/test_manuscript_ocr_registry_unit.py`
- `app/tests/test_profiling_unit.py`
- `app/tests/test_read_mode_line_image_previews_unit.py`
- `app/tests/test_recognition_active_learning_backend_unit.py`
- `app/tests/test_recognition_active_learning_unit.py`
- `app/tests/test_recognition_telemetry_unit.py`
- `app/tests/test_strategy_ablation_config_unit.py`
- `app/tests/test_strategy_adoption_unit.py`
- `app/tests/test_strategy_aware_ocr_crops_unit.py`
- `app/tests/test_auto_orientation_unit.py`

## Manuscript Runtime State

The Flask backend stores manuscripts under:

```text
app/input_manuscripts/<manuscript>/
```

Important manuscript-local paths include:

- `images/`: uploaded original images.
- `images_resized/`: images resized for layout processing.
- `heatmaps/`: CRAFT heatmaps.
- `gnn-dataset/`: initial graph-format files from preprocessing.
- `processing_settings.json`: upload-time processing settings, including
  `target_longest_side`, `min_distance`, and optional line-segmentation
  overrides such as `BINARIZE_THRESHOLD`. It also records the opt-in
  `pipeline_visualization` setting.
- `visualizations/<page>/`: optional, disposable pipeline-stage images and a
  `manifest.json`. These are diagnostics only and are never read by layout,
  OCR, exports, or active learning.
- `node_corrections/`: legacy cumulative node correction summaries used by the
  download metrics export.
- `layout_analysis_output/gnn-format/`: corrected graph, labels, and dimensions
  written by layout saves.
- `layout_analysis_output/_baseline_page_xml/`: baseline-only PAGE XML produced
  from the corrected graph before the production text-line strategy writes final
  `Coords`.
- `layout_analysis_output/page-xml-format/`: final PAGE XML plus sibling
  line-segmentation and reading-direction metadata sidecars.
- `layout_analysis_output/image-format/`: app line images organized by page and
  text region.
- `layout_analysis_output/images_resized/`: resized images copied beside saved
  PAGE XML for export and OCR.
- `layout_analysis_output/text_recovery_backups/<page>/`: timestamped PAGE XML
  snapshots and metadata taken before a Layout Mode save regenerates an existing
  page. These snapshots are used only by Text Recovery.
- `overlay_exports/`: generated page overlay JPEGs from the `/save-overlay`
  route.
- `active_learning/recognition/`: manuscript-local OCR registry, revisions,
  checkpoints, prepared pages, telemetry, profiling, and job state.

Uploading a manuscript name that already exists currently replaces that
manuscript directory in `app/app.py`. Treat manuscript names as mutable working
folders, not archival identifiers.

## Supported Source Images

The production preprocessor recognizes source JPEG (`.jpg`, `.jpeg`), PNG, BMP,
TIFF (`.tif`, `.tiff`), WebP, AVIF, and JPEG 2000 (`.jp2`) images. Decoding AVIF
and JPEG 2000 depends on the installed Pillow build exposing those codecs. A
source image is rejected when both dimensions are under 600 px; otherwise the
upload-time `target_longest_side` setting controls whether it is resized before
layout processing. The uploaded originals remain under `images/`, while the
processed layout page is written under `images_resized/`.

## Optional Pipeline Visualizations

The upload screen has a **Save pipeline visualizations (background)** switch.
It is off by default. When enabled, `app/pipeline_visualization.py` records the
following real artifacts for every uploaded page under
`visualizations/<page>/`:

1. original resized page image;
2. CRAFT heatmap;
3. GNN preprocessing overlay: angular-KNN candidate connectivity, categorical
   heuristic-degree one-hot node colors, and categorical heuristic-overlap
   one-hot edge colors;
4--5. one predicted-versus-human-corrected graph comparison, written only after
   a human layout correction. It uses the color-blind-safe Okabe-Ito palette:
   blue means human-added/predicted-missing and vermilion means
   human-deleted/predicted-extra;
6. the exact processed OCR line images, including configured unwrapped crops,
   both as a contact sheet and as `06_processed_line_images/` files preserving
   their text-region hierarchy;
7--8. one per-line OCR comparison. Predicted and human-corrected Unicode text
   are grapheme-aligned in identically sized cells, one directly above the
   other, with the same blue/vermilion correction semantics and per-line CED
   (grapheme edit distance) and CER.

Stages 1--3 are queued after upload. Stages 4--6 are queued after a layout
save, but the combined graph comparison is written only when the frontend
supplies a real pre-correction versus corrected graph difference. The combined
OCR comparison is written once a committed Read Mode save has both XML
snapshots. The manifest marks unavailable downstream stages as `pending`; the
app never invents ground truth before a human save.

## Headless Layout-Correction Overlay

`app/visualize_layout_corrections.py` creates one standalone image without
starting Flask or the frontend:

```powershell
$env:CONDA_NO_PLUGINS='true'
conda run -n gnn_layout python app/visualize_layout_corrections.py app/input_manuscripts/<manuscript>/images/<page>.jpg
```

The script re-runs the production GNN against the original `gnn-dataset`
inputs in a temporary directory, compares that graph with the corrected graph
under `layout_analysis_output/gnn-format`, and writes only
`visualizations/<page>_layout_corrections.png`. Manuscript-local output is
rejected outside `visualizations/`.

The page background is converted to grayscale. The overlay uses orange
(`#f58231`) for prediction-missing/human-added nodes and edges, blue
(`#4363d8`) for prediction-extra/human-deleted nodes and edges, and black for
unchanged graph elements. Every graph element has a thin black outline, and
node radius is required to exceed edge width. The output omits the correction
legend by default so it retains the source page dimensions; pass `--legend` to
append the legend. Counts are net graph
differences; an edit that was later undone has no surviving location to draw.

For a committed Read Mode text save, the sidecar also copies the current PAGE
XML immediately before the save to `07_ocr_predictions_page.xml`, then copies
the updated PAGE XML to `08_ocr_ground_truth_page.xml`. These two snapshots are
created synchronously around the XML update so a later background task cannot
replace the OCR prediction before it is captured. The background sidecar then
writes `07_08_ocr_text_correction_diff.jpg`, its UTF-8 metrics JSON, and one
losslessly rendered PNG per line under `07_08_ocr_text_correction_diff/`.

The sidecar is best effort: it has a separate daemon worker, serializes its own
writes, and logs an error without changing the upload, layout-save, or OCR
result. To enable it by default for non-browser uploads, set
`APP_PIPELINE_VISUALIZATION=true`; a manuscript's saved upload setting takes
precedence. `max_line_previews` in `processing_settings.json` controls the
contact-sheet cap (default 24, bounded to 1--100).

## Production Text-Line Strategy

The production app strategy is stored in
`app/recognition/line_segmentation/strategy_config.py` as
`production_strategy_name`.

The current checked-in role pins are:

- research benchmark: `local_polygons_stable_unwrap_v1`
- research proposed: unset
- production app: `local_polygons_stable_unwrap_v1`

The production app and the current research benchmark currently name the same
registered strategy implementation: `local_polygons_stable_unwrap_v1`
(`LocalPolygonsStableUnwrapStrategy` in
`app/recognition/line_segmentation/local_polygons_stable_unwrap.py`). There is
not a separate production-only strategy class.

They do not necessarily run with identical runtime knobs. The research harness
gets role configs from `app/tests/precommit_gate_config.py`; the checked-in
benchmark config for `local_polygons_stable_unwrap_v1` is currently only
`BINARIZE_THRESHOLD=0.45`. The GUI runtime gets config from
`get_strategy_runtime_config()` in
`app/recognition/line_segmentation/runtime_config.py`. For production, that
config currently sets the same threshold and also enables:

- `image_fallback_when_no_heatmap_components=True`
- `anchor_window_clip_enabled=True`

So the current production app intentionally deviates from the benchmark in
human-corrected edge cases, while sharing the same strategy code:

- If a corrected text line has no assigned heatmap components, production can
  remap a wider local strip around the corrected baseline, run adaptive
  binarization on the page image, assign nearby foreground islands to the line's
  anchors, and use those island bounds plus padding for PAGE `Coords`. If no
  plausible foreground is found, it still falls back to the historical
  minimum-width baseline band.
- For short or sparse non-closed corrected lines, production can clip the
  remaining heatmap/fallback rectangles to spacing-aware anchor windows around
  the human-corrected node or baseline anchors. Multi-anchor windows scale from
  local anchor spacing; one-node point lines use fixed point caps because there
  is no spacing estimate.

Anchor-window clipping is a final narrowing pass after the existing local
polygon cleanup, including adjacent-line up/down cleanup. It only intersects
the remaining rectangles with the anchor neighborhood; it does not replace the
earlier cleanup. Research harness benchmark runs do not get these
production-only behaviors unless the harness config explicitly enables the same
keys, and such a run should be recorded as a production-runtime comparison
rather than the checked-in benchmark baseline.

Registered strategies currently include:

- `legacy_axis_bound_v1`
- `local_polygons_v1`
- `local_polygons_stable_unwrap_v1`
- `local_tangent_band_v1`
- `local_polygons_hstraight_smooth_unwrap_v1`

Only `legacy_axis_bound_v1`, `local_polygons_v1`, and
`local_polygons_stable_unwrap_v1` are currently marked independent enough for
research or production role pins. `local_tangent_band_v1` and
`local_polygons_hstraight_smooth_unwrap_v1` remain registered for historical or
experimental use but are rejected for production adoption.

Production role validation requires:

- the strategy is registered
- the strategy is marked `production_role_independent=True`
- the strategy has explicit runtime config in
  `app/recognition/line_segmentation/runtime_config.py`

Production adoption is done with:

```powershell
$env:CONDA_NO_PLUGINS='true'
conda run -n gnn_layout python scripts/adopt_text_line_strategy_for_app.py --strategy <strategy_name> --reason "<reason>" --apply
```

Production adoption updates only `production_strategy_name` and
`production_adoption_history`. It must not modify research benchmark/proposed
roles or ablation gates. In particular, do not change
`app/tests/pipeline_ablation_experiment.py`,
`app/tests/recognition_finetuning_experiment.py`,
`app/tests/precommit_gate_config.py`, or `scripts/run_precommit_eval.py` when
the task is only to adopt a strategy in the app.

Existing PAGE XML, OCR line images, and active-learning checkpoint lineage are
not migrated automatically when the production strategy changes.

## Production Layout Save Boundary

Production layout saves start from the live corrected GUI graph, not from the
research eval fixtures. The route is:

- `POST /semi-segment/<manuscript>/<page>`

The frontend sends `saveScope` and `saveIntent`:

- `saveScope == "layout"` regenerates layout artifacts.
- `saveScope == "text_only"` updates text in existing PAGE XML.
- `saveIntent == "commit"` is an explicit user save.
- `saveIntent == "draft"` is background Text Review autosave state.

For layout saves, `generate_xml_and_images_for_page()` in `app/gnn_inference.py`
does the production layout work:

1. Converts the corrected GUI graph into graph-format files.
2. Includes manual node and edge edits.
3. Recomputes connected-component text-line labels from saved graph edges.
4. Includes text-region labels.
5. Resolves optional reading-direction annotations.
6. Writes baseline PAGE XML under `_baseline_page_xml/`.
7. Loads `production_strategy_name`.
8. Loads explicit production runtime config, plus manuscript processing
   overrides from `processing_settings.json`.
9. Applies `apply_text_line_segmentation_strategy(...)`.
10. Writes final PAGE `TextLine/Coords`.
11. Writes sibling `<page>_line_segmentation_metadata.json`.
12. Writes app line images through the shared OCR crop layer.
13. Copies the resized page image into `layout_analysis_output/images_resized/`.

### Region Defaults And GNN Adaptation Boundary

Text-region labels are resolved after connected components are recomputed from
the corrected graph. If a component contains valid manual region labels, the
most common label is applied to every node in that component. If it has no
valid label, the app allocates an unused non-negative label to that component.
Consequently, each unannotated text line is saved in its own `TextRegion`; the
app does not place all unannotated lines in one shared region. Region folders
for line images and the PAGE XML hierarchy use those resolved labels.

The production app always performs layout inference with its pre-trained GNN.
It does not queue GNN fine-tuning jobs, select GNN checkpoints, or promote a
fine-tuned GNN. The fold-local GNN fine-tuning used by
`experiments/downstream_ocr/` is an isolated evaluation workflow and cannot
change the GNN used by production layout saves.

### Corrected Graph To OCR Line Image

The layout-save pipeline turns a corrected graph prediction into an OCR input
as follows:

1. GNN edges plus manual node and edge corrections are materialized as
   connected text-line components and ordered PAGE `Baseline` polylines.
2. The production strategy builds a PAGE `Coords` polygon around each baseline.
   The current `local_polygons_stable_unwrap_v1` strategy starts from the
   CRAFT heatmap, applies baseline-local cleanup and component assignment, and
   records line topology and crop metadata. Production-only image fallback can
   derive local foreground from the resized page when a corrected line has no
   usable heatmap component; anchor-window clipping narrows sparse open-line
   geometry after that cleanup.
3. The shared crop layer reads the saved PAGE geometry and metadata. It uses
   stable arclength/tangent sampling and vectorized remap grids to unwrap
   `curved_open` and `closed_circular` lines through their local polygons,
   masks outside-polygon pixels to the page median color, and trims the result.
   If metadata is absent or invalid, or an unwrap guard fails, it writes the
   historical masked PAGE `Coords` crop instead.
4. The resulting rectangular crop is saved under
   `image-format/<page>/textbox_label_<region>/line_<id>.jpg` and is the image
   supplied to local OCR. The same crop contract is reused for line-image
   export and OCR active-learning preparation.

This boundary is intentionally based on saved PAGE geometry and metadata, so
OCR, exports, and fine-tuning preparation do not need to reconstruct the live
graph or reload a heatmap.

For text-only saves, `update_page_text_content()` updates PAGE `TextEquiv`
content in place and does not regenerate layout geometry or line images.

The save route also keeps the legacy `node_corrections/<page>.json` counters.
Those counters are used by the ZIP export's `node_metrics.json`; they are not
the active-learning or human-effort source of truth. Use
`active_learning/telemetry/human_interventions.json` for unified
intervention logging.

## Text Recovery After Layout Regeneration

Before a Layout Mode save replaces an existing page's PAGE XML, the backend
copies that XML and writes backup metadata under:

```text
layout_analysis_output/text_recovery_backups/<page>/
```

The most recent snapshot is available to Read Mode through:

- `POST /recover-text/<manuscript>/<page>`

Recovery does not restore old geometry. It compares the prior non-empty line
text with the current page's lines, using text similarity and, where available,
PAGE `Coords` overlap. Ambiguous or low-confidence candidates are excluded. For
accepted matches, the route writes the recovered text into the current PAGE XML
and clears that line's recognition confidence; unmatched current lines retain
their existing text. The frontend presents the resulting text for review.

Text Recovery is not an OCR supervision event or an active-learning job. A user
must review the recovered text and make a normal Text Review `text_only` commit
before it can become OCR ground truth.

## Reading Direction And Layout Staleness

Layout mode supports optional reading-direction annotations for line orientation
and unwrap ambiguity. In the frontend, holding `q` and hovering records a
cross-line cut. The backend normalizes the cut into:

- component node indices
- cut start, end, and midpoint
- a unit reading-direction tangent
- source and timestamp metadata

On save, `app/recognition/line_segmentation/reading_direction.py` resolves each
annotation against the current final text-line components by node overlap. An
annotation that cannot be normalized or no longer matches a current component is
stored as stale and is not used for crop orientation.

Reading-direction metadata is saved beside PAGE XML as:

```text
<page>_reading_direction_metadata.json
```

For a single-node text line, the resolved reading direction supplies the local
station axis and the cross-cut supplies the local normal axis used for PAGE
`Coords` construction and OCR crop unwrapping. Without a valid annotation, point
baselines keep the historical horizontal default.

The layout fingerprint used by Text Review state includes canonicalized PAGE
`Coords`, canonicalized `Baseline` points, and the reading-direction metadata
payload. If the layout changes after OCR prediction or text review, the page
workflow can report stale states such as `stale_layout` or
`ground_truth_stale_layout`. This protects reread/review workflows, but it is
separate from validating the line-segmentation metadata sidecar.

## Production OCR Crop Boundary

OCR crop preparation is centralized in:

```text
app/recognition/line_segmentation/ocr_crops.py
```

The production crop contract is:

- saved PAGE `TextLine/Coords`
- saved PAGE `Baseline`
- optional sibling line-segmentation metadata
- optional reading-direction metadata copied into line metadata

The same crop boundary is used by:

- app line-image export in `app/gnn_inference.py`
- local OCR inference in `app/recognition/recognize_manuscript_text_v2_pretrained.py`
- active-learning page preparation in `app/recognition/pagexml_line_dataset.py`
- manuscript-aware OCR inference in `app/ocr_model_manager.py`

Line-segmentation metadata is saved beside PAGE XML as:

```text
<page>_line_segmentation_metadata.json
```

If strategy metadata is valid and requests a supported unwrap crop model, the
crop layer unwraps the line. If metadata is missing, malformed, unsupported,
non-unwrapped for an unwrap strategy, or unwrap guards fail, OCR crop
preparation falls back to the historical masked PAGE `Coords` crop.

During local OCR inference, unwrapped `curved_open` and `closed_circular` lines
without a valid reading-direction annotation use decoded-text auto-orientation.
The runtime recognizes both the identity crop and its 180-degree rotation, then
selects the decoded string with stronger Devanagari evidence. Devanagari
letters, marks, and numbers are positive evidence; letters and numbers from
other scripts are negative evidence; punctuation and whitespace are neutral.
Model confidence is not used, and exact evidence ties preserve the identity
crop. Annotated curved lines, straight lines, point lines, and masked crop
fallbacks retain the single-orientation path. Production writes the selected
transform into the PAGE `TextEquiv/@custom` audit metadata.

That persisted transform is the orientation contract for subsequent use of the
line. Read Mode applies it when serving the line-image preview, and text-only
saves preserve it while replacing the Unicode text. Active-learning dataset
preparation and the downloadable `ocr-training-format` apply the same transform
to the image paired with the corrected Unicode label. The layout crop on disk
remains the canonical, unmodified crop; oriented preview and training images
are derived from it so the transform is applied exactly once.
An explicit Layout Mode reading-direction annotation always takes precedence.
If stale auto-orientation metadata is also present, preview, fine-tuning
preparation, and OCR-training export ignore it; the next text save removes it.

With the current production strategy, new layout saves write metadata requesting:

```text
crop_model = "local_polygon_stable_unwrap"
```

That crop path uses stable arclength/tangent unwrapping with vectorized remap
grids and a masked fallback. Existing pages without usable metadata continue to
use masked PAGE `Coords` crops.

Active-learning revision snapshots copy PAGE XML, line-segmentation metadata,
reading-direction metadata, and the resized page image. Revision training then
prepares OCR crops from the snapshot PAGE XML plus snapshot metadata; it does
not require heatmaps.

## Local OCR And Gemini Runtime

The local OCR runtime uses the EasyOCR-style Sanskrit checkpoint family. The
base pretrained checkpoint is:

```text
app/recognition/pretrained_model/vadakautuhala.pth
```

Do not treat that base checkpoint as mutable. Fine-tuned checkpoints belong in
manuscript-local runtime artifact folders.

The app is manuscript-aware. Local OCR inference loads the current manuscript
checkpoint from the manuscript OCR registry instead of assuming one global
active model forever. If no manuscript-local active checkpoint exists, the app
falls back to the base checkpoint. If the active checkpoint is missing, the
registry tries the previous active checkpoint and then the base checkpoint,
recording the fallback.

The backend exposes reader capabilities at:

- `GET /recognition/readers`

The default reader is local OCR. Gemini is available only when `GEMINI_API_KEY`
is configured in the server environment. Gemini request timeout is controlled
by `GEMINI_OCR_TIMEOUT_SECONDS`; invalid values fall back to 45 seconds and
valid values are clamped to 5 through 300 seconds. Gemini failures return
retryable recovery metadata with local OCR listed as the fallback reader.

OCR predictions are not supervised training data by themselves. Both local OCR
and Gemini predictions are recorded in the manuscript OCR registry with:

- predicted lines
- recognition engine
- checkpoint id and checkpoint path for local OCR
- confidences when available
- layout fingerprint
- recorded timestamp

Corrected Gemini predictions can still become supervised local OCR training data
after the user commits them in Text Review. Gemini is a prediction source, not a
checkpoint lineage.

## OCR Active-Learning Runtime

The runtime stores manuscript-local active-learning state under:

```text
input_manuscripts/<manuscript>/active_learning/recognition/
```

This state includes:

- `registry.json`
- page revisions
- revision snapshots of PAGE XML, sidecars, and images
- active checkpoint lineage
- candidate checkpoint lineage
- promotion summaries
- fallback state
- `needs_rebase` tracking
- pending jobs
- prepared pages
- training artifacts
- active-learning telemetry under `active_learning/telemetry/` and profiling summaries

Only foreground Text Review commit saves with non-empty text become supervised
OCR ground truth. The exact supervised OCR boundary is:

- `saveIntent == "commit"`
- `saveScope == "text_only"`
- at least one non-empty corrected text line

Draft autosaves, raw OCR predictions, and Page Layout saves may update PAGE XML,
layout lineage, or recoverability state, but they are not OCR supervision. A
layout save with text present is still not supervised OCR input unless it is a
Text Review `text_only` commit.

Prepared lines with a graph-derived PAGE baseline of one or two nodes are kept
in PAGE XML, line-image exports, and OCR inference, but are excluded from OCR
fine-tuning. The fine-tuning corpus requires at least three baseline nodes and
records skipped short lines with the reason
`baseline_node_count_below_minimum` in its manifest. This protects the
recognition model from being adapted on extremely short graph crops without
discarding those lines from the user-facing workflow.

The frontend currently autosaves dirty Text Review drafts about every 20 seconds
while Text Review is active. Draft saves create revision/recovery state but do
not enqueue OCR training.

When active learning is enabled and a new supervised commit revision is saved:

- if the page has not already contributed to a promoted checkpoint, the runtime
  queues an `ocr_fine_tune` job using the active checkpoint as the parent and
  approved historical revisions as replay data
- if the page has already contributed to a promoted checkpoint and is changed,
  the registry marks `needs_rebase` and may queue an `ocr_rebase` job over the
  latest approved supervised revisions

After each successful OCR fine-tune step, the runtime compacts checkpoint
artifacts by copying the selected sibling checkpoint to:

```text
active_learning/recognition/checkpoints/<checkpoint_id>/model.pth
```

The registry points at that stable `model.pth`. The runtime keeps small metadata
and logs, including `fine_tune_metadata.json` and `selector_metrics.json`, but
removes regenerated heavyweight materialization such as sibling `.pth` files in
`training_run/`, the step `dataset/`, the step `lmdb/`, and prepared-page
training scratch directories. This does not change future fine-tuning inputs:
rebase and history replay use the base checkpoint plus revision snapshots under
`active_learning/recognition/revisions/`, not deleted LMDB or training scratch
directories.

After promotion, obsolete manuscript checkpoint directories are pruned when they
are not one of:

- the base checkpoint
- the current active checkpoint
- the previous active checkpoint used for fallback
- the in-flight candidate
- a parent or candidate checkpoint referenced by a pending OCR job

Pruned checkpoint records remain in `registry.json` with `status="pruned"` for
lineage/audit purposes, but their model directory is removed. Set
`OCR_RUNTIME_COMPACT_CHECKPOINT_ARTIFACTS=0` to skip per-candidate compaction or
`OCR_RUNTIME_PRUNE_OBSOLETE_CHECKPOINTS=0` to keep obsolete checkpoint
directories during debugging.

The active runtime recipe is based on
`DEFAULT_OCR_ACTIVE_LEARNING_RECIPE` in
`app/recognition/active_learning_recipe.py`:

- `training_policy=page_plus_random_history`
- `history_sample_line_count=10`
- `width_policy=batch_max_pad`
- `oversampling_policy=none`
- `augmentation_policy=none`
- `lr_scheduler=none`
- `optimizer=adadelta`
- `lr=0.2`
- `num_iter=60`
- `curve_metric=early_weighted_page_cer`
- `regression_guard_abs=0.005`

The shared default recipe uses `sibling_checkpoint_strategy=page_cer_selector`.
The GUI runtime overrides this to `best_norm_ed` unless
`OCR_RUNTIME_SIBLING_CHECKPOINT_STRATEGY` is set.

The GUI runtime does not rerun the full research verifier on every save.
Runtime candidate checkpoints are directly promoted after successful training
and recorded in the manuscript OCR registry. Future hyperparameter changes
should go back through the research verifier before replacing this runtime
recipe.

## Job Orchestration And Device Leases

The app uses a generic job orchestrator for background work. The active
production integration is OCR fine-tune and OCR rebase.

The orchestrator supports:

- queued job records
- numeric priorities
- direct handlers and isolated child-process jobs
- OCR fine-tune and rebase job types
- exclusive resource leases such as `gpu`
- cancellation
- requeue-on-cancel behavior
- queue wait and runtime status reporting
- listener callbacks that mirror job state into the manuscript OCR registry

Interactive local OCR calls `prepare_for_interactive_ocr(...)`. If a lower
priority GPU job is running, the orchestrator marks it for cancellation and
requeue, sets manuscript status to `paused_for_ocr`, and waits briefly for the
resource to clear before running interactive OCR.

The device lease manager is intentionally simple: one owner per resource name.
It does not implement multi-GPU scheduling or memory-aware placement.

## Telemetry And Profiling

Production saves and jobs record structured telemetry and coarse profiling
summaries.

Telemetry includes:

- page save events in `telemetry/page_events.jsonl`
- job lifecycle events in `telemetry/job_events.jsonl`
- per-revision page edit summaries in `telemetry/page_edit_summary.json`
- unified per-page and per-manuscript human-effort summaries in
  `telemetry/human_interventions.json`
- layout edit metrics such as node additions, node deletions, edge additions,
  edge deletions, reset-heuristic count, text-region annotation deltas,
  reading-direction annotation deltas, and total layout intervention count
- text edit metrics measured against the built-in reader prediction, including
  changed line count, total edit distance, page CER, mean line CER, and
  per-line diffs
- active-learning entry decisions
- promotion and fallback summaries

Profiling includes:

- job wall time
- CUDA availability
- device name
- peak CUDA memory allocated and reserved when CUDA is available
- optional sampled CUDA traces

Set `ACTIVE_LEARNING_PROFILE_CUDA=1` to capture a sampled CUDA trace. The
current helper samples once per job family by writing a marker under the
manuscript profiling root.

Layout-save timing for the synchronous Layout Mode save to PAGE XML boundary is
off by default. Set `LAYOUT_SAVE_TIMING_ENABLED=1` before starting the Flask
backend to append chunked timing records to:

```text
app/input_manuscripts/<manuscript>/layout_analysis_output/profiling/layout_save_timings.jsonl
```

Each record covers `generate_xml_and_images_for_page()` in `app/gnn_inference.py`,
including graph materialization, connected-component label writing, reading
direction metadata, baseline PAGE XML, production `Coords` strategy application,
app line-image export, resized-image copy, and cleanup. Optional
`LAYOUT_SAVE_TIMING_LOG_DIR` and `LAYOUT_SAVE_TIMING_LOG_PATH` override the
default destination for local profiling runs.

Generated telemetry and profiling artifacts are runtime evidence, not research
source-of-truth config. A checked-in evaluator that turns these artifacts into
manuscript-level effort curves is still a known gap.

## Read Mode Previews And Exports

Read Mode can serve processed line-image previews through:

- `GET /line-image/<manuscript>/<page>/<line_numeric_id>`

The preview payload intentionally includes only lines that are non-straight
according to line-segmentation metadata or have a reading-direction annotation.
This keeps the Text Review view focused on lines where crop orientation is most
likely to matter.

Read Mode also uses the Devanagari keyboard implementation in
`app/frontend/src/typing-utils/devanagariInputUtils.js`. Its browser-free
regression guide and test command are in [TYPING_TESTING.md](./TYPING_TESTING.md):

```powershell
npm --prefix app/frontend run test:typing
```

The results ZIP route is:

- `GET /download-results/<manuscript>`

The ZIP currently includes:

- `page-xml-format/`
- `image-format/`
- `images_resized/` for annotated pages
- `ocr-training-format/` with text-line images and `gt.txt` rows for lines that
  have non-empty PAGE `Unicode` text and a matching saved line image
- `node_metrics.json` derived from the legacy `node_corrections/` summaries

The route returns `404` if no annotated PAGE XML exists for the manuscript.

The overlay export route is:

- `POST /save-overlay/<manuscript>/<page>`

It draws the current graph over the original page image when available, falls
back to the resized page image, and writes:

```text
overlay_exports/<page>_overlay.jpg
```

## Known Production Limitations

- Restart, interruption, and rebuild hardening for the live OCR runtime is still
  shallow. See `docs/exec-plans/tech-debt-tracker.md`.
- The runtime records useful telemetry, but there is not yet a checked-in
  evaluator that turns it into human-effort curves.
- `app/app.py` still mixes route handlers, legacy node-correction logging,
  PAGE-XML generation orchestration, OCR recognition startup, and
  active-learning queueing.
- `app/app.py` still appends `app/recognition` to `sys.path` for local OCR
  imports.
- The device lease manager provides exclusive single-resource locking, not full
  multi-device scheduling.
- The slow OCR study can still hit Windows `conda run` Unicode output issues;
  trust generated artifacts or use a direct environment Python fallback as
  described in `RESEARCH_HARNESS.md`.

## Runtime Validation Commands

Run these from the repository root in the `gnn_layout` environment.

Job orchestrator:

```powershell
$env:CONDA_NO_PLUGINS='true'
conda run -n gnn_layout python -m unittest app.tests.test_job_orchestrator_unit -v
```

Manuscript OCR registry:

```powershell
$env:CONDA_NO_PLUGINS='true'
conda run -n gnn_layout python -m unittest app.tests.test_manuscript_ocr_registry_unit -v
```

OCR active-learning runtime and backend behavior:

```powershell
$env:CONDA_NO_PLUGINS='true'
conda run -n gnn_layout python -m unittest app.tests.test_recognition_active_learning_unit -v
conda run -n gnn_layout python -m unittest app.tests.test_recognition_active_learning_backend_unit -v
```

Telemetry and profiling:

```powershell
$env:CONDA_NO_PLUGINS='true'
conda run -n gnn_layout python -m unittest app.tests.test_recognition_telemetry_unit -v
conda run -n gnn_layout python -m unittest app.tests.test_profiling_unit -v
```

Production strategy, crop behavior, and adoption:

```powershell
$env:CONDA_NO_PLUGINS='true'
conda run -n gnn_layout python -m unittest app.tests.test_line_segmentation_strategy_unit -v
conda run -n gnn_layout python -m unittest app.tests.test_strategy_ablation_config_unit -v
conda run -n gnn_layout python -m unittest app.tests.test_strategy_adoption_unit -v
conda run -n gnn_layout python -m unittest app.tests.test_strategy_aware_ocr_crops_unit -v
```

Read Mode previews and export behavior:

```powershell
$env:CONDA_NO_PLUGINS='true'
conda run -n gnn_layout python -m unittest app.tests.test_read_mode_line_image_previews_unit -v
conda run -n gnn_layout python -m unittest app.tests.test_download_results_export_unit -v
```

Research gates, slow OCR policy studies, and text-line strategy promotion are
documented in `RESEARCH_HARNESS.md`.
