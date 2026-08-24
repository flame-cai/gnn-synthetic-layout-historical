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
- manuscript-local active learning for the local OCR model and the layout GNN,
  as two separate opt-ins

The production app is not the research harness. It may use a strategy or recipe
that was selected by research gates, but changing production behavior must be
an explicit production adoption step.

## Runtime Source Files

Important production runtime files:

- `app/app.py`
- `app/frontend/`
- `app/gnn_inference.py`
- `app/active_learning_jobs.py`
- `app/device_leases.py`
- `app/job_orchestrator.py`
- `app/layout_active_learning_runtime.py`
- `app/manuscript_layout_registry.py`
- `app/manuscript_ocr_registry.py`
- `app/ocr_active_learning_runtime.py`
- `app/ocr_model_manager.py`
- `app/pretrained_gnn/gnn_active_learning.yaml`
- `src/gnn_training/gnn_finetuning.py`
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
- `app/tests/test_active_learning_joint_e2e.py`
- `app/tests/test_job_orchestrator_unit.py`
- `app/tests/test_layout_active_learning_e2e.py`
- `app/tests/test_layout_active_learning_unit.py`
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
- `overlay_exports/`: generated page overlay JPEGs from the `/save-overlay`
  route.
- `active_learning/recognition/`: manuscript-local OCR registry, revisions,
  checkpoints, prepared pages, telemetry, profiling, and job state.
- `active_learning/layout/`: the same for the layout GNN — registry, corrected
  graph snapshots per revision, cached augmentations, checkpoints, training runs,
  and profiling.
- `active_learning/telemetry/`: shared across both lineages.

Uploading a manuscript name that already exists currently replaces that
manuscript directory in `app/app.py`. Treat manuscript names as mutable working
folders, not archival identifiers.

## Optional Pipeline Visualizations

Pipeline visualizations were a temporary diagnostic and the upload-screen switch
has been removed. The capability remains: set `APP_PIPELINE_VISUALIZATION=true`,
or set `pipeline_visualization.enabled` in a manuscript's
`processing_settings.json`, which takes precedence. When enabled, `app/pipeline_visualization.py` records the
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

For text-only saves, `update_page_text_content()` updates PAGE `TextEquiv`
content in place and does not regenerate layout geometry or line images.

The save route also keeps the legacy `node_corrections/<page>.json` counters.
Those counters are used by the ZIP export's `node_metrics.json`; they are not
the active-learning or human-effort source of truth. Use
`active_learning/telemetry/human_interventions.json` for unified
intervention logging.

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

## Layout Active-Learning Runtime

The app fine-tunes two models, not one. OCR active learning improves the
recognition model from corrected text; layout active learning improves the
text-line GNN from corrected graphs. They are separate opt-ins with separate
lineages, and the GUI exposes one toggle each:

- **Improve Future Reading** — OCR. Default on, preserving previous behaviour.
- **Improve Future Layout** — layout GNN. Default off; it trains a second model.

Layout state is manuscript-local and lives beside the OCR state:

```text
input_manuscripts/<manuscript>/active_learning/layout/
    registry.json
    checkpoints/<checkpoint_id>/model.pt
    revisions/<page>/rev_NNNN/gnn-format/     snapshot of the six corrected files
    augmentations/<page>/rev_NNNN/            cached 50 variants for replay
    training/<candidate_id>/
    profiling/
```

### Supervision boundary

Layout supervision is a **Layout Mode commit save with at least one node**:

- `saveIntent == "commit"`
- `saveScope == "layout"`
- the saved graph has at least one node

Saving asserts the layout is correct, so a page the human did not have to edit
still counts. Re-saving an unchanged graph is deduplicated by a content hash
over node positions (rounded to 0.1 px), the canonicalised edge set, region
labels, and reading directions, so it records no new revision and queues no
job. Draft autosaves and Read Mode `text_only` commits never touch layout
lineage, exactly as Layout Mode saves never become OCR supervision.

Each non-duplicate supervised revision snapshots the six `gnn-format` files
before any job runs, so training uses the graph as it was saved even if the
user keeps editing while the job is queued.

### Recipe

`app/pretrained_gnn/gnn_active_learning.yaml`, whose hyperparameters are
identical to the canonical experiment recipe
`experiments/downstream_ocr/configs/gnn_finetuning_no_deleted_nodes.yaml`:
50 augmentations per page, 20% history replay, Adam 1e-3 with 5-epoch warmup,
10 epochs, focal loss, selection on `val_textline_f1_score`, continuing the
serialized checkpoint's model object rather than rebuilding the architecture.

`deleted_node_supervision` is **disabled**, and this matters more in the GUI
than in the experiment: deleting spurious CRAFT nodes is a primary Layout Mode
action, so app pages routinely sit in the high-deletion regime where
re-inserting deleted nodes as all-negative edge supervision collapsed held-out
layout quality (`circle_new` fold_4: G-F1 0.577 -> 0.214). Keeping it off also
means training never needs the raw `gnn-dataset/` proposals. Evidence:
`experiments/downstream_ocr/GNN_SUPERVISION_FINDINGS.md`.

The production copy exists because `experiments/` is untracked and production
must not import untracked code. Keep the two in step, and take any change back
through the research harness first.

### Training schedules

Two schedules, same hyperparameters; only the data schedule differs.

- **Incremental** (`gnn_fine_tune`, one per supervised save): continues the
  manuscript's active checkpoint on the new page's 50 augmentations plus a
  deterministic 20% replay of each already-consumed page. This is the
  experiment recipe, and it is O(1) per save.
- **Pooled** (`gnn_backfill`, `gnn_rebase`): trains one checkpoint from the
  pretrained base over every saved page at once, each contributing all 50
  augmentations to every epoch, with no history subsample. Used when every page
  is already in hand, where a ladder would only approximate what pooling can do
  exactly — and where pooling is also roughly half the GPU work at ten pages.

Backfill is offered from the toggle when a manuscript has corrected pages the
layout model has not learned from yet, including pages corrected before this
feature existed: those are adopted into the registry and snapshotted first.
Recorrecting a page that already shaped the active checkpoint marks
`needs_rebase` and queues one pooled rebase rather than stacking a second step
for the same page.

### Contention: page load never blocks

This is the one place layout learning deliberately differs from OCR. Local OCR
inference is an explicit user action, so `prepare_for_interactive_ocr(...)` can
preempt and requeue a running job. GNN inference runs on **every page open**, so
the same strategy would both stall navigation and starve training, since a
requeued GNN step restarts from scratch.

Instead, `GET /semi-segment/<manuscript>/<page>` predicts with the newest
*promoted* layout checkpoint and ignores queued or running work. Nothing in the
layout runtime calls `preempt_for_interactive`. Consequences worth knowing:

- A page opened while a fine-tune is in flight uses the previous checkpoint.
  The status line says training is running; the page does not wait.
- If CUDA is out of memory because a training job holds the GPU, GNN inference
  falls back to CPU rather than failing. Inference is small, so this is a slower
  page load, not a broken one.
- `load_model_once` keys its cache on the resolved checkpoint path, so a
  promotion is picked up by the next page load with no restart.
- The registry resolves active -> previous active -> pretrained base, so a
  missing or half-written checkpoint degrades page load instead of ending it.

A page that already has saved `gnn-format` edges is returned from disk and never
re-predicted, so improving the model never overwrites human layout corrections.
Fine-tuning only affects pages not yet corrected.

The reverse also holds: training must not start underneath imminent inference.
Automatic layout jobs (`gnn_fine_tune`, `gnn_rebase`) are queued with a
`not_before` head start, `LAYOUT_RUNTIME_TRAINING_START_DELAY_SECONDS` (default
25), and every interactive read pushes that deadline back through
`defer_layout_training_for_interactive_work(...)`. A user-requested backfill is
not deferred. Preemption remains the backstop for a job that did start: local
OCR calls `prepare_for_interactive_ocr(...)`, which cancels and requeues any
running GPU job including a layout one. This deferral was added because the
first real GUI session preempted and restarted 5 of 6 layout jobs; the head
start removes the race rather than resolving it.

### Replace With New Layout

The layout counterpart of Read Mode's Replace With New Reading, and the only way
to re-run the GNN on a page that already has a saved graph:

    GET /semi-segment/<manuscript>/<page>?relayout=1

It re-predicts edges over the page's **current** node set, so manual node
additions and deletions survive, and it carries the saved per-node region labels
forward because the node set is unchanged. Text-line labels are not carried
forward: they are the components of the edge set being replaced. Nothing is
written — the user keeps the result by saving the page or discards it by
reloading. The response reports `layoutFromSavedGraph`, which the GUI uses to
show the button only where there is something to replace.

### Checkpoint retention

Each layout checkpoint is roughly 51 MB and one is produced per corrected page,
so promotion prunes checkpoint directories the fallback ladder can no longer
reach. Protected: base, active, previous active, the in-flight candidate, and
anything a pending job still needs. Pruned records remain in `registry.json`
with `status="pruned"` for lineage. Set
`LAYOUT_RUNTIME_PRUNE_OBSOLETE_CHECKPOINTS=0` to keep them while debugging.

Layout learning is a follow-up to the save, never a gate on it. Layout artifacts
are written before the runtime is called, and a failure there is caught, logged,
and reported without failing the save.

### Job types and routing

`gnn_fine_tune`, `gnn_rebase` and `gnn_backfill` join `ocr_fine_tune` and
`ocr_rebase` on the same orchestrator, the same exclusive `gpu` lease, and one
shared state listener. `app/active_learning_jobs.py` routes both isolated job
execution and job-state events to the runtime that owns the job type, so neither
lineage writes into the other's registry. Layout jobs are isolated child
processes like OCR jobs.

Because the orchestrator runs isolated jobs as daemonic processes, and a
daemonic process may not have children, graph augmentation falls back to serial
execution there. Augmenting one page is sub-second either way.

### Routes

- `GET /manuscript/<name>/layout-active-learning` — status, including how many
  corrected pages the layout model has and has not learned from.
- `POST /manuscript/<name>/layout-active-learning/backfill` — queue one pooled
  run over every corrected page.

`POST /semi-segment/<manuscript>/<page>` accepts `layoutActiveLearningEnabled`
alongside the existing `activeLearningEnabled`, and returns
`layoutActiveLearning` plus `layoutActiveLearningQueuedJobIds`.

## Job Orchestration And Device Leases

The app uses a generic job orchestrator for background work. The active
production integrations are OCR fine-tune/rebase and layout GNN
fine-tune/rebase/backfill.

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

## Text Recovery After A Layout Change

A Layout Mode save regenerates `TextLine/Coords` and line ids and is sent with an
empty text payload, so any text the user had already corrected in Read Mode does
not survive the save. Before regenerating, the save route copies the current PAGE
XML to:

```text
layout_analysis_output/text_recovery_backups/<page>/<backup_id>.xml
```

`POST /recover-text/<manuscript>/<page>` matches backup lines onto the current
lines with `0.75 * text_similarity + 0.25 * bbox_IoU` over whitespace- and
joiner-stripped text, rejects a best match within `0.04` of the runner-up as
ambiguous, and never reuses a backup line twice. Unmatched lines are returned in
`unrecovered_line_ids` and drawn in red in Text Review.

Recovery is opt-in, so the risk is a user who edits the fresh reading without
pressing `Recover Text` and strands the earlier work. Each backup therefore also
records `had_read_mode_annotations`, classified at backup time by
`ManuscriptOcrRegistry.has_read_mode_annotations(...)`. Page text counts as
annotated when either:

- the OCR registry has a Text Review revision for the page — a supervised
  commit, or a draft autosave, which only fires on a dirty text draft (Layout
  Mode saves are non-supervised commits and do not qualify); or
- the page carries Unicode text that is not identical to the last prediction
  this app recorded for it.

The second condition exists because in-tool history is not the only source of
annotated text. Manuscripts transcribed elsewhere and imported as PAGE XML —
`dense_layout`, `moderate_layout`, `circular_layout`, and the legacy conversions
described in each manuscript's `CONVERSION.md` — carry human transcription on
every page with no registry revisions and no predictions, and a
revision-only test called them unannotated. Across the manuscripts currently in
`app/input_manuscripts/`, the revision-only test protected 97 of 134 transcribed
pages; the current test protects all 134. Text that still matches the recorded
prediction verbatim is not an annotation and does not warn.

The flag is surfaced in `pageWorkflow.text_recovery` and is `null` on backups
written before it existed; consumers must treat `null` as "may hold
annotations".

Read Mode raises a modal on the first edit attempt — clicking a line, tabbing to
one, or typing into a focused one — when recovery is available and
`had_read_mode_annotations` is not `false`. The choice ("Recover Previous Text"
or "Continue Without Recovering") is remembered per backup id in `localStorage`,
so the user is asked once per layout change rather than once per page visit, and
pressing `Recover Text` directly settles it the same way.

This classification is read-only with respect to active learning. It adds one
registry load to a layout save, reads revisions and the recorded prediction, and
changes no supervision boundary, job, or checkpoint lineage.

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

Layout active learning. The unit tests are fast and need no GPU; the two
end-to-end suites train for real and need the pretrained GNN checkpoint, the
released ground-truth graphs, and (for the joint suite) the base OCR
checkpoint. They skip cleanly when those are absent:

```powershell
$env:CONDA_NO_PLUGINS='true'
conda run -n gnn_layout python -m unittest app.tests.test_layout_active_learning_unit -v
conda run -n gnn_layout python -m unittest app.tests.test_layout_active_learning_e2e -v
conda run -n gnn_layout python -m unittest app.tests.test_active_learning_joint_e2e -v
```

`test_layout_active_learning_e2e` is the evidence that fine-tuning works, in
two directions. It trains on the three pages of the released `circular_layout`
fold_1 and scores the six held-out pages on their ground-truth node set, so
predicted and ground-truth edges are directly comparable. It also runs a
falsifiable check: one page corrected under a deliberately absurd convention
(every node isolated, no edges) must make the model decline to connect nodes on
a different, never-seen page. Observed on an RTX 6000 Ada:

| Check | Held-out measure | Before | After |
| --- | --- | --- | --- |
| Normal, 3 pages, incremental | text-line F1 | 0.7962 | 0.9063 |
| Normal, 3 pages, pooled backfill | text-line F1 | 0.7962 | 0.9086 |
| Extreme, 1 page, all edges deleted | predicted positive-edge rate | 0.1083 | 0.0001 |

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
