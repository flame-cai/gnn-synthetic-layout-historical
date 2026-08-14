# Research Harness

This repository uses an LLM-assisted, external-verifier-driven research harness to
improve the manuscript digitization pipeline in small, reviewable steps.

The core idea is simple:

1. An agent or researcher proposes one narrow pipeline change.
2. The change is implemented as a proposed research strategy or recipe.
3. External verifier code runs the same task against the current benchmark and
   the proposed change.
4. The gate writes evidence.
5. A human reviews the evidence.
6. Promotion is explicit. Production adoption is a separate explicit step.

The harness is currently used for two parts of the pipeline:

- local OCR model fine-tuning hyperparameters
- text-line segmentation strategy and OCR crop preparation

The same pattern can be reused for other pipeline stages if the stage boundary,
metrics, source-of-truth config, and promotion workflow are made explicit first.

## Purpose And Scope

The harness exists to let LLMs help with research changes without letting the
LLM decide success by itself.

The LLM may inspect code, propose a strategy, implement code, and update config.
The external verifier decides whether the proposed change is eligible for
promotion. The verifier must be deterministic enough to rerun, strict enough to
catch regressions, and explicit enough that a fresh clone can tell which
strategy is the benchmark.

This document describes the checked-in harness behavior. It is not a substitute
for reading the source files named below before changing behavior.

## Shared Verifier-Driven Harness Contract

### Generic Pattern

For any pipeline stage, the harness should follow this pattern:

1. Define the exact stage to optimize.
2. Name the inputs and outputs.
3. Keep a checked-in benchmark role.
4. Keep a checked-in proposed role when a comparison is active.
5. Keep the benchmark and proposed implementations independent.
6. Keep deployed production strategy code independent from active research
   strategy code where a production role exists.
7. Run the same verifier against both roles.
8. Compare the proposed role against source-controlled metrics and regression
   rules.
9. Write local evidence artifacts.
10. Promote the proposed role only through an explicit command.
11. Adopt a promoted strategy for production only through a separate explicit
    command.
12. Retain old strategy code and promotion history for rollback.

The only intentional variable in a gate run should be the strategy or recipe
under test. Datasets, model checkpoints, metric code, thresholds, and verifier
entrypoints should be shared between benchmark and proposed roles.

### Evidence Versus Source Of Truth

Generated artifacts are evidence. They are not the source of truth.

The source of truth must live in checked-in code or checked-in markdown. For the
text-line harness, the role pins live in
`app/recognition/line_segmentation/strategy_config.py`. Gate artifacts are
written under `app/tests/logs/` and can be regenerated.

Evidence artifacts should record enough detail to review a run:

- strategy names
- dataset names
- metric values
- comparison operators
- thresholds or allowed regressions
- local artifact paths
- timestamps
- enough run metadata to detect stale evidence

Important conclusions should not live only in ignored logs.

### Benchmark, Proposed, And Production Roles

The research roles are:

- `benchmark`: the current research baseline.
- `proposed`: the candidate being tested against the benchmark.

The production role is:

- `production`: the strategy used by future GUI layout saves.

For text-line segmentation, these names are stored as:

- `benchmark_strategy_name`
- `proposed_strategy_name`
- `production_strategy_name`

Research roles must be independent implementations. The text-line strategy
registry enforces this with `research_role_independent=True`. A strategy that
delegates to another registered text-line strategy is rejected for benchmark or
proposed use.

Production defaults must also be marked independent and must have explicit
runtime config in `app/recognition/line_segmentation/runtime_config.py`.

Current limitation: the code allows the research benchmark and production app
role to point to the same registered strategy after explicit production
adoption. The current checked-in state does this with
`local_polygons_stable_unwrap_v1`. That is weaker than the strict rule that
research and production must share no code at all. The enforced invariant today
is that benchmark/proposed research roles cannot delegate to another strategy,
and production adoption must be explicit and production-configured.

### Promotion Versus Production Adoption

Promotion and production adoption are different operations.

Research promotion changes the research harness. For text-line segmentation,
`scripts/promote_text_line_strategy.py --apply` moves the proposed strategy into
the benchmark role, clears the proposed role, and appends research promotion
history. It preserves `production_strategy_name`.

Production adoption changes future GUI behavior. For text-line segmentation,
`scripts/adopt_text_line_strategy_for_app.py --apply` changes
`production_strategy_name` and appends production adoption history. It does not
modify research benchmark/proposed roles.

A gate run may recommend promotion, but it must not silently rewrite tracked
source during pre-commit. Existing PAGE XML, OCR crops, and active-learning
checkpoint lineage are not migrated by either command.

### Non-Negotiable Evaluation Rules

- Scope one stage precisely before implementing a proposed change.
- Name the stage inputs and outputs.
- Use the same verifier code for benchmark and proposed roles.
- Keep benchmark and proposed strategy code independent.
- Do not implement a proposed strategy as a wrapper around the current
  benchmark or production strategy.
- Do not call another registered strategy's `apply(...)` from the method under
  comparison.
- Do not import strategy-owned geometry constants, crop constants, or helper
  functions from the competing method. Shared role-neutral infrastructure such
  as dataclasses, PAGE XML helpers, registry plumbing, low-level geometry
  primitives, and OCR crop execution is allowed.
- Keep production strategy code independent from active research strategy code
  unless an explicitly documented adoption workflow has made the same strategy
  the production default.
- Keep metric thresholds and regression allowances in checked-in source.
- Keep generated artifacts separate from source-of-truth config.
- Do not let pre-commit silently promote or adopt a strategy.
- Keep old strategy code and promotion history.
- Make rollback possible.
- Do not treat a GUI runtime promotion as research proof.
- Do not treat research promotion as GUI rollout.

## Evaluation Datasets

**New harnesses build on `dataset_release/`. `app/tests/eval_dataset/` and
`app/tests/eval_dataset_v2/` are retained locally, are no longer tracked, and
are not distributed.**

Both legacy datasets were the substrate on which the retained OCR fine-tuning
recipe and the promoted text-line unwrapping strategy were refined. The gate
thresholds in `app/tests/precommit_gate_config.py` and the promotion provenance
in `app/recognition/line_segmentation/strategy_config.py` were calibrated
against them and remain the evidence base for the current benchmark. Those
records are not rewritten by this change. Three existing gates still read the
local copies and continue to work on a machine that has them; a fresh clone does
not, and the fix for a fresh clone is to migrate the gate, not to restore the
data.

### Directory contract

A recognition evaluation dataset is three directories plus a page ordering.
`RecognitionEvalDatasetConfig` needs `images_dir`, `pagexml_dir` and
`heatmaps_dir`; pages are ordered by sorted image stem, then split by
`fine_tune_page_count` and `eval_page_start_index` / `eval_page_end_index`. A
released manuscript satisfies that contract directly:

| Config field | Legacy path | `dataset_release` path |
| --- | --- | --- |
| `images_dir` | `app/tests/<dataset>/images` | `dataset_release/manuscripts/<manuscript>/inputs` |
| `pagexml_dir` | `app/tests/<dataset>/labels/PAGE-XML` | `dataset_release/manuscripts/<manuscript>/labels/page_xml` |
| `heatmaps_dir` | `app/tests/<dataset>/heatmaps` | `dataset_release/manuscripts/<manuscript>/heatmaps` |

`labels/unicode_output/` in `eval_dataset` has no reader anywhere in the tree
and has no counterpart in the release; per-line Unicode lives inside the PAGE
XML, which is what the harness actually reads.

### What the released manuscripts correspond to

| Legacy dataset | Pages | Nearest released manuscript | Relationship |
| --- | --- | --- | --- |
| `eval_dataset` | 15 (`233_0002`…`233_0016`) | `moderate_layout` | **Same manuscript.** Heatmaps are byte-identical. Images are the pre-resize DAV scans; the release is defined on the resized raster. PAGE XML is an **older annotation snapshot** (`Created` 2026-01-31 against the release's 2026-07-23) with different region labels. |
| `eval_dataset_v2` | 5 (`page_2`…`page_6`, 2500×2500) | `circular_layout` | **Different manuscript**, same regime. Not present in the release in any form. `circular_layout` is a role substitute, not the same data. |

### Why a swap is not drop-in

Three consequences follow, and a migrated gate must handle all three.

1. **Thresholds do not transfer.** `max_curve_metric_value`,
   `max_final_page_cer`, `min_first_step_gain` and the strategy-ablation
   regression allowances were calibrated on the legacy pages and labels.
   Changing the ground truth — even for the same manuscript, because the
   annotations were revised — moves every metric. A migrated gate must be
   recalibrated against the new dataset and its new numbers recorded in
   checked-in source, exactly as the current ones are.
2. **`moderate_layout` page rasters are withheld.** They are under third-party
   copyright and are not in the release. A clone must reconstruct them with
   `dataset_release/tools/prepare_images.py` before that manuscript can drive
   any image-dependent gate. `dense_layout` and `circular_layout` ship their
   rasters and work immediately.
3. **The circular gate loses its manuscript.** `eval_dataset_v2` has no
   released counterpart, so migrating gate 3 means re-establishing a circular
   baseline on `circular_layout` from scratch, not porting a threshold.

### The retained calibration is narrower than the release

The retained OCR fine-tuning recipe and the promoted text-line unwrapping
strategy are the choices the recorded evidence supports, and they remain the
trusted defaults. They are not, however, known to be optimal on
`dataset_release`, and nothing in the recorded evidence claims they are. Two
limits of the basis they were selected on:

- **Coverage.** They were established on 20 pages of two manuscripts: one that
  is in the release but under **superseded annotations**, and one that is **not
  in the release at all**. The release spans three manuscripts, 31 pages and 882
  text lines across three deliberately contrasting layout regimes.
- **Observability.** Neither legacy dataset labels the properties the unwrapping
  strategy actually selects on. Their PAGE XML carries only
  `structure_line_id_N`; there is no line kind and no reading direction, so
  curved, closed and single-point lines could only be measured indirectly
  through aggregate page CER. The release labels them explicitly — 38
  `curved_open`, 9 `closed_circular`, 172 `point`, plus 25 human reading
  directions and per-line arclength/normal frame parameters — so a strategy can
  be scored on the 47 hard cases directly rather than inferred from a page mean.

The release also ships five documented folds per manuscript, so a re-derivation
gets a stated split protocol and page-cluster confidence intervals instead of a
single fixed page prefix.

Re-optimizing the fine-tuning hyperparameters and the unwrapping strategy
against `dataset_release` is therefore open work, and the natural first use of
it. Doing so means re-deriving thresholds rather than porting them (see above),
and recording the new evidence the same way the current evidence is recorded:
in checked-in source, with the promotion command, not silently.

### Why the legacy data is no longer tracked

`app/tests/eval_dataset/images/` holds the original DAV scans of the
`moderate_layout` manuscript byte-for-byte. Those images are licensed for
research use but not for redistribution, which is why the release withholds
them and ships `DOWNLOAD.md` plus a checksum manifest instead. Tracking them here
contradicted that. `eval_dataset_v2` is untracked in the same change because it
is 90 MB serving one gate.

Removing them from tracking removes them from the current tip only; they remain
in earlier commits. Purging them from history would require a rewrite and a
force-push, which is a separate decision.

## Current Harness Instances

The OCR work has three layers that should be understood together: the offline
OCR active-learning research harness, the surrogate OCR fine-tuning pre-commit
gate, and the GUI-safe OCR active-learning runtime documented in
`PRODUCTION.md`.

### 1. OCR Fine-Tuning Hyperparameter Harness

#### Stage Boundary

This harness optimizes the local OCR model continuation recipe.

Inputs:

- the base OCR checkpoint at
  `app/recognition/pretrained_model/vadakautuhala.pth`
- PAGE XML with corrected text
- prepared line images and ground-truth text
- a sequence of pages used for fine-tuning
- held-out pages used for OCR evaluation

Outputs:

- fine-tuned OCR checkpoint candidates
- OCR predictions written back into PAGE XML
- page-level and line-level CER metrics
- curve metrics across sequential fine-tuning steps
- metadata describing the training recipe and selected checkpoint

This stage does not change text-line geometry. When the verifier needs crops, it
uses the shared PAGE XML line dataset preparation code.

#### Source Of Truth Files

The main source files are:

- `app/recognition/active_learning_recipe.py`
- `app/recognition/active_learning.py`
- `app/recognition/train.py`
- `app/recognition/pagexml_line_dataset.py`
- `app/recognition/dataset.py`
- `app/recognition/ocr_defaults.py`
- `app/ocr_active_learning_runtime.py`
- `app/manuscript_ocr_registry.py`
- `app/job_orchestrator.py`
- `app/tests/recognition_finetuning_config.py`
- `app/tests/precommit_gate_config.py`
- `app/tests/recognition_finetuning_experiment.py`
- `app/tests/test_recognition_active_learning_unit.py`
- `app/tests/test_recognition_finetuning_page_plus_history_unit.py`
- `app/tests/test_recognition_finetuning_precommit_unit.py`
- `app/tests/test_recognition_finetuning_precommit_e2e.py`
- `app/tests/test_recognition_finetuning_e2e.py`

Current limitation: this OCR hyperparameter harness was implemented as a policy
study and retained production recipe. It does not currently have persistent
checked-in `benchmark_recipe` and `proposed_recipe` role pins equivalent to the
text-line strategy role config. Future OCR recipe changes should add or emulate
that role discipline before replacing the retained recipe.

#### Tuned Variables

The OCR study code can vary:

- continuation policy
- number of sampled history lines
- OCR width policy
- oversampling policy
- augmentation policy
- learning-rate scheduler
- optimizer
- learning rate
- training iteration count
- checkpoint sibling selection strategy
- shuffle behavior

The active code path currently keeps only the selected recipe as the default
runtime recipe.

The current harness supports explicit OCR width policies
`global_2000_pad` and `batch_max_pad`, bounded CER-weighted oversampling,
OCR-only augmentation policies `none`, `background_only`, and
`background_plus_rotation`, and learning-rate scheduler plumbing for `none`,
`step`, and `cosine`. It also supports sibling checkpoint selection between
`best_accuracy.pth` and `best_norm_ED.pth`; the CER-aligned selector remains
available for verifier use.

#### Retained Production Recipe

The retained recipe is defined by `DEFAULT_OCR_ACTIVE_LEARNING_RECIPE`:

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

The default recipe object uses `sibling_checkpoint_strategy=page_cer_selector`.
The GUI active-learning runtime overrides that default to `best_norm_ed` unless
`OCR_RUNTIME_SIBLING_CHECKPOINT_STRATEGY` is set.

#### Historical Study Evidence

The retained historical winner is:

- `wb_on_an_hist10_sn_optd_lr200000u`

Meaning:

- `page_plus_random_history`
- `history_sample_line_count=10`
- `batch_max_pad`
- no oversampling
- no augmentation
- no scheduler
- `optimizer=Adadelta`
- `lr=0.2`
- `num_iter=60`

Retained metrics:

- `curve_metric_value=0.22151451085911972`
- `final_page_cer=0.13784355179704016`
- `first_step_gain=0.0572938689217759`

Earlier cumulative and page-only studies are preserved conclusions rather than
live code paths. The retained conclusions are that broad and focused sweeps
established `batch_max_pad + no oversampling + no augmentation` as the stable
structural stack worth keeping; strict page-only continuation was viable but
weaker and substantially more guard-sensitive on `eval_dataset`; hybrid replay
beat the earlier baselines on the primary curve metric and final-page CER; and
Adam remained guard-sensitive even after replaying historical lines, so the
trusted recipe remains Adadelta with `lr=0.2` and `num_iter=60`.

Generated run artifacts may include:

- `curve_metrics.json`
- `per_page.csv`
- `per_line.csv`
- `selector_metrics.json`
- `fine_tune_metadata.json`
- plots and run summaries

Those artifacts are local evidence. The selected recipe in source is the runtime
source of truth.

#### Runtime Adoption Contract

The GUI runtime does not rerun the full research verifier on every save.

Runtime behavior is:

- Text Review commit saves can create supervised OCR revisions.
- Draft autosaves do not create supervised OCR training data.
- Layout saves do not create supervised OCR training data.
- If active learning is enabled and the revision is new, the runtime queues OCR
  fine-tune or rebase jobs.
- Runtime state is manuscript-local under
  `input_manuscripts/<manuscript>/active_learning/recognition/`.
- Candidate checkpoints are promoted through the manuscript OCR registry, not by
  editing global research config.

The commit boundary is implemented in `app/ocr_active_learning_runtime.py`:
`save_intent == "commit"`, `save_scope == "text_only"`, and at least one
non-empty text line are required for supervised OCR input.

#### Future Hyperparameter Change Workflow

To change OCR hyperparameters:

1. Define the exact recipe change.
2. Keep the current retained recipe as the benchmark.
3. Implement the proposed recipe without changing production runtime defaults.
4. Run the OCR verifier against both recipes.
5. Compare the primary curve metric and secondary metrics.
6. Save evidence artifacts.
7. Update `DEFAULT_OCR_ACTIVE_LEARNING_RECIPE` only after review.
8. Update tests that pin the recipe.
9. Keep old study code and evidence summaries.

Do not replace the GUI runtime recipe based only on a single manual run or a
candidate checkpoint that happened to improve one manuscript.

#### Validation Commands

From the repository root:

```powershell
$env:CONDA_NO_PLUGINS='true'
conda run -n gnn_layout python -m unittest app.tests.test_recognition_finetuning_precommit_unit -v
```

Slow pre-commit OCR gate:

```powershell
$env:CONDA_NO_PLUGINS='true'
conda run -n gnn_layout python -m unittest app.tests.test_recognition_finetuning_precommit_e2e -v
```

Slow OCR policy study:

```powershell
$env:CONDA_NO_PLUGINS='true'
conda run -n gnn_layout python -m unittest discover -s app/tests -p "test_recognition_finetuning_e2e.py" -v
```

On Windows, if `conda run` fails after the study completes because of console
encoding, trust the saved artifact folder more than the wrapper output.

Targeted OCR active-learning unit tests:

```powershell
$env:CONDA_NO_PLUGINS='true'
conda run -n gnn_layout python -m unittest app.tests.test_recognition_active_learning_unit -v
```

If `conda run` output is unreliable on Windows, use an activated environment or
the environment Python directly:

```powershell
conda activate gnn_layout
python -m unittest discover -s app/tests -p "test_recognition_finetuning_e2e.py" -v
```

```powershell
C:\Users\intro\miniconda3\envs\gnn_layout\python.exe -m unittest discover -s app/tests -p "test_recognition_finetuning_e2e.py" -v
C:\Users\intro\miniconda3\envs\gnn_layout\python.exe -m unittest app.tests.test_recognition_finetuning_precommit_e2e -v
```

Update the direct interpreter path if the local Conda installation differs.

### 2. Text-Line Segmentation To OCR Crops Harness

#### Stage Boundary

This harness optimizes the stage that turns text-line baselines into PAGE
`TextLine/Coords` and OCR-ready crops.

Research harness inputs:

- resized manuscript page image
- CRAFT heatmap
- source PAGE XML with `Baseline` polylines
- benchmark/proposed strategy name
- strategy config

Research harness outputs:

- regenerated PAGE XML with `TextLine/Coords`
- line-segmentation metadata
- OCR-ready line crops
- OCR and geometry metrics

Production inputs are different. The GUI starts from the live corrected graph,
manual node and edge edits, text-line labels, text-region labels, text content,
and optional reading-direction annotations. Production first writes baseline
PAGE XML, then applies `production_strategy_name` to generate final PAGE
`Coords`.

#### Source Of Truth Files

The main source files are:

- `app/recognition/line_segmentation/strategy_config.py`
- `app/recognition/line_segmentation/registry.py`
- `app/recognition/line_segmentation/runtime_config.py`
- `app/recognition/line_segmentation/types.py`
- `app/recognition/line_segmentation/legacy_axis_bound.py`
- `app/recognition/line_segmentation/local_polygons.py`
- `app/recognition/line_segmentation/local_polygons_stable_unwrap.py`
- `app/recognition/line_segmentation/ocr_crops.py`
- `app/recognition/line_segmentation/reading_direction.py`
- `app/recognition/pagexml_line_dataset.py`
- `app/gnn_inference.py`
- `app/app.py`
- `app/tests/precommit_gate_config.py`
- `app/tests/pipeline_ablation_experiment.py`
- `app/tests/recognition_finetuning_experiment.py`
- `app/tests/test_ci_e2e.py`
- `app/tests/test_recognition_finetuning_precommit_e2e.py`
- `app/tests/test_circular_recognition_finetuning_precommit_e2e.py`
- `scripts/run_precommit_eval.py`
- `scripts/promote_text_line_strategy.py`
- `scripts/adopt_text_line_strategy_for_app.py`
- `docs/pipeline-improvement/text-line-segmentation/strategy-promotion-workflow.md`
- `docs/pipeline-improvement/text-line-segmentation/strategy-promotion-record.md`
- `.githooks/pre-commit`

#### Strategy Roles

The checked-in role config currently stores:

- research benchmark: `local_polygons_stable_unwrap_v1`
- research proposed: `null`
- production app: `local_polygons_stable_unwrap_v1`

`get_strategy_role_config()` validates the config on read.

Research benchmark and proposed roles must pass
`validate_research_role_strategy()`. Production must pass
`validate_production_role_strategy()` and must have runtime config.

A proposed strategy must be configured before comparison gates can run. If
`proposed_strategy_name` is `null`, the strategy ablation gates will refuse to
run the proposed role. The e2e unittest entrypoints skip in this idle state so
normal test discovery does not report a misleading failure when there is no
candidate to evaluate; the underlying gate helpers still fail closed if called
without a proposed strategy.

Do not change the gates to benchmark-only mode or no-proposed-strategy mode as
part of production adoption work. That is a separate research-harness behavior
change and needs explicit approval.

#### Current Strategies

Registered text-line strategies include:

- `legacy_axis_bound_v1`
- `local_polygons_v1`
- `local_polygons_stable_unwrap_v1`
- `local_tangent_band_v1`
- `local_polygons_hstraight_smooth_unwrap_v1`

Currently role-independent strategies:

- `legacy_axis_bound_v1`
- `local_polygons_v1`
- `local_polygons_stable_unwrap_v1`

Currently registered but rejected for benchmark/proposed/production role pins:

- `local_tangent_band_v1`
- `local_polygons_hstraight_smooth_unwrap_v1`

`local_polygons_stable_unwrap_v1` owns stable local-polygon unwrap behavior,
baseline endpoint anchors, and ambiguous joined-heatmap component splitting. It
is the current research benchmark and current production app strategy.

Its stable unwrap crop path is intentionally more specific than "use local
polygons." The strategy requests `crop_model="local_polygon_stable_unwrap"` in
line metadata. The OCR crop layer then uses stable arclength/tangent unwrapping
with smoothed centerline sampling, endpoint-exclusive sampling for closed loops,
vectorized remap grids, and median-color masking outside the PAGE `Coords`
polygon. If unwrap geometry guards fail, the crop layer falls back to the
historical masked PAGE `Coords` crop instead of emitting a broken unwrap.

#### Production Crop Preparation Boundary

Production layout saves call `generate_xml_and_images_for_page()` in
`app/gnn_inference.py`.

That function:

1. Converts the corrected graph into baseline PAGE XML.
2. Resolves reading-direction annotations.
3. Loads the production strategy name.
4. Loads explicit production runtime config.
5. Applies `apply_text_line_segmentation_strategy()`.
6. Writes final PAGE XML and line-segmentation metadata.
7. Creates app line images through the shared OCR crop layer.

OCR crop preparation is centralized in
`app/recognition/line_segmentation/ocr_crops.py`. If strategy metadata is valid
and requests an unwrap crop model, the crop layer unwraps the line. If metadata
is missing, malformed, unsupported, or unwrap fails, it falls back to the
historical masked PAGE `Coords` crop.

#### Reading-Direction Metadata

Reading-direction annotations are stored separately from PAGE XML.

The frontend records cross-cut annotations. The backend resolves each annotation
to a final text-line component by node overlap and writes a sidecar JSON file
with active and stale annotations.

The crop layer uses resolved reading direction and cut midpoint metadata to
choose a stable unwrap direction for ambiguous lines. For single-node point
baselines, the resolved reading direction also defines the local station axis
used by strategy metadata and OCR crop unwrapping; missing annotations retain
the horizontal point-baseline default.

#### Active-Learning Ground-Truth Boundary

Text-line strategy changes can affect OCR crops, and OCR crops affect active
learning. The supervised OCR boundary is still text review, not layout save.

A save creates supervised OCR ground truth only when:

- `saveIntent` is `commit`
- `saveScope` is `text_only`
- corrected text contains at least one non-empty line

Draft saves and layout saves may preserve state, but they do not become OCR
training ground truth.

#### External Verifier Gates

These gates compare a proposed text-line strategy against the current research
benchmark. When `proposed_strategy_name` is `null`, the unittest entrypoints
below skip with an explicit inactive-gate message. Configure a proposed strategy
before using these commands as promotion evidence.

##### Pretrained Full-Pipeline Gate

Entrypoint:

- `app/tests/test_ci_e2e.py`

Implementation:

- `app/tests/pipeline_ablation_experiment.py`

Dataset:

- `app/tests/eval_dataset/` (local only; not tracked -- see
  [Evaluation datasets](#evaluation-datasets))

What it does:

1. Uploads the eval dataset through the Flask test client.
2. Runs semi-segmentation with the pretrained GNN.
3. Saves generated graph output.
4. Prepares OCR crops with each strategy role.
5. Runs local OCR predictions with the pretrained OCR model.
6. Evaluates predicted PAGE XML against ground truth.

Absolute role threshold:

- `page_cer <= 0.40`

Benchmark/proposed comparison:

- primary metric: `page_cer`
- proposed must be `<= benchmark + 0.01`

Latest artifacts:

- `app/tests/logs/pipeline_ablation_latest.md`
- `app/tests/logs/pipeline_ablation_latest.json`

##### OCR Fine-Tuning Strategy Ablation Gate

Entrypoint:

- `app/tests/test_recognition_finetuning_precommit_e2e.py`

Implementation:

- `app/tests/recognition_finetuning_experiment.py`

Dataset:

- `app/tests/eval_dataset/` (local only; not tracked -- see
  [Evaluation datasets](#evaluation-datasets))

What it does:

1. Regenerates PAGE `Coords` from `Baseline` plus heatmap for each strategy
   role.
2. Prepares OCR crops.
3. Runs the retained hybrid OCR fine-tuning recipe.
4. Evaluates OCR improvement over sequential fine-tuning steps.
5. Compares proposed strategy metrics against benchmark strategy metrics.

Geometry guards before OCR:

- `source_line_coverage >= 0.90`
- `heatmap_box_assignment_rate >= 0.90`

The `baseline_heatmap` path must not read PAGE `Coords` for fallback or
equivalence. PAGE baselines do not preserve the manual node-add/delete history
needed to reconstruct every corrected graph point exactly, so the ablation gate
must regenerate geometry from `Baseline`, heatmap, and page image only.

Recipe:

- `page_plus_random_history`
- `history_sample_line_count=10`
- `batch_max_pad`
- no oversampling
- no augmentation
- no scheduler
- `Adadelta`
- `lr=0.2`
- `num_iter=60`

Benchmark/proposed comparison:

- primary metric: `curve_metric_value`
- proposed must be `<= benchmark + 0.02`
- `final_page_cer` must be `<= benchmark + 0.02`
- `first_step_gain` must be `>= benchmark - 0.02`

Latest artifacts:

- `app/tests/logs/recognition_finetune_ablation_latest.md`
- `app/tests/logs/recognition_finetune_ablation_latest.json`
- `app/tests/logs/recognition_finetune_ablation_latest.txt`

Note: absolute OCR thresholds such as `max_curve_metric_value=0.26`,
`max_final_page_cer=0.18`, and `min_first_step_gain=0.04` are used by the
standalone recognition pre-commit helper. The text-line strategy ablation
launcher compares benchmark and proposed roles.

##### Circular OCR Fine-Tuning Strategy Ablation Gate

Entrypoint:

- `app/tests/test_circular_recognition_finetuning_precommit_e2e.py`

Implementation:

- `app/tests/recognition_finetuning_experiment.py`

Dataset:

- `app/tests/eval_dataset_v2/` (local only; not tracked -- see
  [Evaluation datasets](#evaluation-datasets))

The checked test expects:

- ordered pages: `page_2`, `page_3`, `page_4`, `page_5`, `page_6`
- fine-tune pages: `page_2`, `page_3`, `page_4`
- evaluation pages: `page_5`, `page_6`

Benchmark/proposed comparison:

- primary metric: `curve_metric_value`
- proposed must be strictly `< benchmark`
- secondary metric comparisons are recorded but are not blocking when strict
  primary improvement is required

Latest artifacts:

- `app/tests/logs/circular_ocr_ablation_latest.md`
- `app/tests/logs/circular_ocr_ablation_latest.json`
- `app/tests/logs/circular_ocr_ablation_latest.txt`

#### Aggregate Promotion Evidence

`scripts/run_precommit_eval.py` runs the three strategy gates. If all phases run
and pass, it calls `write_strategy_promotion_evidence()` in
`scripts/promote_text_line_strategy.py`.

The aggregate evidence:

- verifies all latest artifacts use the expected study modes
- verifies all gate artifacts agree on benchmark and proposed strategy names
- records each gate's primary metric and pass/fail state
- records artifact paths and modification times
- recommends promotion only if all gates passed

Local evidence is written to:

- `app/tests/logs/strategy_promotion_latest.md`
- `app/tests/logs/strategy_promotion_latest.json`

A checked-in promotion record is also generated because `app/tests/logs/` is
ignored and local artifacts may not exist in a fresh clone:

- `docs/pipeline-improvement/text-line-segmentation/strategy-promotion-record.md`

The launcher cleans up large passing role-run directories by default after each
phase writes its latest aliases. This removes bulky benchmark/proposed OCR run
trees, including per-step model folders, while retaining the latest artifacts
and compact summary directories needed for promotion evidence. Passing OCR plots
are copied into retained summary directories before cleanup. Failed phases keep
their role-run directories for debugging. Set `CLEAN_UP=0` before running
`scripts/run_precommit_eval.py` to keep full passing role artifacts.

#### Research Promotion Workflow

1. Implement an independent proposed strategy.
2. Register it in `app/recognition/line_segmentation/registry.py`.
3. Mark it `research_role_independent=True`.
4. Add research strategy config if needed.
5. Set `proposed_strategy_name` in
   `app/recognition/line_segmentation/strategy_config.py`.
6. Run the three verifier gates.
7. Review the generated evidence.
8. Promote with:

```powershell
$env:CONDA_NO_PLUGINS='true'
conda run -n gnn_layout python scripts/promote_text_line_strategy.py --candidate <proposed_strategy> --previous-benchmark <current_benchmark> --metrics app/tests/logs/strategy_promotion_latest.json --apply
```

Promotion updates only the research role config. It preserves production app
strategy config.

Use the promotion script in dry-run mode first by omitting `--apply`, then
inspect the reported before/after role config and the source diff. Only rerun
with `--apply` after the evidence and diff have been reviewed.

The promotion script refuses to write when:

- the promotion evidence file is missing
- the evidence is stale relative to the referenced gate artifacts
- any required gate failed
- the evidence benchmark or proposed strategy does not match the CLI arguments
- the candidate or previous benchmark strategy is not registered
- the candidate or previous benchmark is not valid for a research role
- the current checked-in benchmark/proposed roles no longer match the requested
  promotion

#### Production Adoption Workflow

Only adopt a strategy for production after deciding it should affect future GUI
saves.

The strategy must:

- be registered
- be marked `production_role_independent=True`
- have explicit runtime config in
  `app/recognition/line_segmentation/runtime_config.py`

Adopt with:

```powershell
$env:CONDA_NO_PLUGINS='true'
conda run -n gnn_layout python scripts/adopt_text_line_strategy_for_app.py --strategy <strategy_name> --reason "<reason>" --apply
```

Production adoption affects future layout saves. It does not rewrite old PAGE
XML, old line images, or active-learning checkpoint lineage.

Production adoption must not modify research harness code or ablation gates. Do
not change `app/tests/pipeline_ablation_experiment.py`,
`app/tests/recognition_finetuning_experiment.py`,
`app/tests/precommit_gate_config.py`, or `scripts/run_precommit_eval.py` when
the task is only to adopt a strategy in the app.

#### Hook Behavior

`.githooks/pre-commit` currently exits immediately at the top:

```sh
exit 0
```

Because of that guard, installing the hook will not automatically run the
strategy gates on commit. If the guard is removed, the hook calls
`scripts/run_precommit_eval.py`.

`scripts/run_precommit_eval.py` supports skip environment variables:

- `SKIP_EVAL_HOOK=1`
- `SKIP_PIPELINE_EVAL_HOOK=1`
- `SKIP_RECOGNITION_FT_HOOK=1`
- `SKIP_CIRCULAR_RECOGNITION_FT_HOOK=1`

If any phase is skipped, aggregate promotion evidence is not refreshed.

If `proposed_strategy_name` is `null`, the e2e gate unittests skip instead of
running benchmark-only. A skipped gate is not promotion evidence and must not be
used to refresh aggregate promotion evidence.

#### Validation Commands

Full three-phase launcher:

```powershell
$env:CONDA_NO_PLUGINS='true'
conda run -n gnn_layout python scripts/run_precommit_eval.py
```

Fast pretrained full-pipeline gate:

```powershell
$env:CONDA_NO_PLUGINS='true'
conda run -n gnn_layout python -m unittest discover -s app/tests -p "test_ci_e2e.py" -v
```

OCR strategy ablation gate:

```powershell
$env:CONDA_NO_PLUGINS='true'
conda run -n gnn_layout python -m unittest app.tests.test_recognition_finetuning_precommit_e2e -v
```

Circular OCR strategy ablation gate:

```powershell
$env:CONDA_NO_PLUGINS='true'
conda run -n gnn_layout python -m unittest app.tests.test_circular_recognition_finetuning_precommit_e2e -v
```

## Adapting The Harness To Another Pipeline Stage

### Required Stage Contract

Before building a new harness instance, write down:

- the pipeline stage name
- the exact inputs
- the exact outputs
- the code entrypoint that owns the stage
- the upstream assumptions
- the downstream contract
- the dataset or fixtures used by the verifier
- what is allowed to vary
- what must stay fixed

Examples of future stages:

- CRAFT character detection
- GNN edge classification
- text-region grouping
- text-line reading-order annotation
- OCR post-processing
- runtime speed or memory optimization

### Required Metrics And Regression Rules

Every harness instance needs checked-in metric rules:

- primary metric
- secondary metrics
- metric direction
- absolute thresholds where needed
- allowed regressions against benchmark
- strict improvement requirements where needed
- missing-data behavior
- failure message format

For manuscript digitization, useful metrics may include:

- page CER
- line CER
- first-step OCR gain
- final-page OCR CER
- weighted curve metric
- AP at IoU threshold
- F1 score
- node add/delete counts
- edge add/delete counts
- human correction time or edit burden
- runtime and memory usage

### Required Source-Of-Truth Config

Do not put role pins only in logs.

A new harness instance should have checked-in config for:

- benchmark role
- proposed role
- production role if the stage affects the GUI
- strategy or recipe config
- datasets
- thresholds
- allowed regressions
- promotion history
- production adoption history if applicable

The config should be readable in a fresh clone without local artifact folders.

### Required Evidence Artifacts

Evidence artifacts should be regenerated by the verifier and should include:

- metrics JSON
- human-readable summary
- per-page or per-sample CSV where useful
- plots where useful
- run config
- environment metadata
- model checkpoint references
- dataset page ids
- strategy names
- stale-evidence checks where promotion depends on artifacts

Ignored local artifacts are acceptable as evidence, but durable conclusions must
be copied into checked-in config or checked-in summaries.

### Required Promotion And Adoption Boundaries

For each new harness instance, define two separate operations:

- research promotion
- production adoption

Research promotion should update only the benchmark/proposed research roles and
append promotion history.

Production adoption should update only production runtime config and append
adoption history.

Neither operation should silently migrate old outputs. If migration is required,
it should be a third explicit workflow with its own verifier and rollback plan.
