# Production GUI App

This document describes the production behavior of the semi-automatic
annotation and OCR app under `app/`. Research verifier details live in
`RESEARCH_HARNESS.md`; this file is about the GUI runtime, saved manuscript
state, production text-line strategy adoption, local OCR inference, and
active-learning jobs.

## Purpose And Scope

The app supports manuscript digitization with human correction in the loop:

- layout correction by adding and deleting graph nodes
- graph correction by adding and deleting edges
- text-line and text-region grouping
- PAGE XML generation
- OCR through Gemini or the local OCR checkpoint family
- manual OCR text correction
- PAGE XML and line-image export
- manuscript-local OCR active learning for the local OCR model

The production app is not the research harness. It may use a strategy or recipe
that was selected by research gates, but changing production behavior must be an
explicit production adoption step.

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
- `app/recognition/train.py`
- `app/recognition/line_segmentation/ocr_crops.py`
- `app/recognition/line_segmentation/strategy_config.py`
- `app/recognition/line_segmentation/runtime_config.py`
- `scripts/adopt_text_line_strategy_for_app.py`

Important production/runtime tests:

- `app/tests/test_job_orchestrator_unit.py`
- `app/tests/test_manuscript_ocr_registry_unit.py`
- `app/tests/test_recognition_active_learning_unit.py`
- `app/tests/test_recognition_active_learning_backend_unit.py`
- `app/tests/test_recognition_telemetry_unit.py`
- `app/tests/test_strategy_adoption_unit.py`
- `app/tests/test_strategy_aware_ocr_crops_unit.py`

## Production Text-Line Strategy

The production app strategy is stored in
`app/recognition/line_segmentation/strategy_config.py` as
`production_strategy_name`.

The current production app strategy is:

- `local_polygons_stable_unwrap_v1`

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
research eval fixtures.

The save path:

1. Converts the corrected graph into baseline PAGE XML.
2. Includes manual node and edge edits.
3. Includes text-line labels and text-region labels.
4. Resolves optional reading-direction annotations.
5. Applies `production_strategy_name`.
6. Writes final PAGE `TextLine/Coords`.
7. Writes sibling `<page>_line_segmentation_metadata.json`.
8. Writes app line images through the shared OCR crop layer.

The production crop contract is saved PAGE `Coords` plus optional strategy
metadata. App line-image export, local OCR inference, and active-learning
revision training use that saved contract. Missing, malformed,
legacy/delegated, unsupported, non-unwrapped, or unwrap-guard-failing metadata
falls back to the historical masked PAGE `Coords` crop.

This separation matters because production and research crop preparation do not
start from the same operational geometry. Production starts from the live
corrected graph and saved PAGE `Coords`; research strategy ablation starts from
PAGE `Baseline`, heatmap, and page image, then regenerates `Coords` through the
strategy under test.

## Local OCR Runtime

The local OCR runtime uses the EasyOCR-style Sanskrit checkpoint family. The
base pretrained checkpoint is:

- `app/recognition/pretrained_model/vadakautuhala.pth`

Do not treat that base checkpoint as mutable. Fine-tuned checkpoints belong in
manuscript-local runtime artifact folders.

The app is manuscript-aware. Local OCR inference loads the current manuscript
checkpoint from the manuscript OCR registry instead of assuming one global
active model forever. If no manuscript-local active checkpoint exists, the app
falls back to the base checkpoint.

The canonical OCR recipe lives in
`app/recognition/active_learning_recipe.py` and is shared by runtime and
pre-commit config code. The GUI runtime defaults sibling checkpoint selection to
`best_norm_ed` through `OCR_RUNTIME_SIBLING_CHECKPOINT_STRATEGY`; the
CER-aligned selector remains available in the shared OCR code.

## OCR Active-Learning Runtime

The runtime stores manuscript-local active-learning state under:

```text
input_manuscripts/<manuscript>/active_learning/recognition/
```

This state includes:

- manuscript-local OCR registry JSON
- page revisions
- revision snapshots of PAGE XML and images
- active checkpoint lineage
- candidate checkpoint lineage
- promotion summaries
- fallback state
- `needs_rebase` tracking
- prepared pages
- training artifacts
- telemetry and profiling summaries

Only foreground Text Review commit saves with non-empty text become supervised
OCR ground truth. OCR predictions, draft autosaves, and Page Layout saves may
update PAGE XML or layout lineage, but they are not OCR supervision.

The supervised OCR save boundary is:

- `saveIntent == "commit"`
- `saveScope == "text_only"`
- at least one non-empty corrected text line

Draft saves are recoverability state. They do not enqueue OCR fine-tuning.

When active learning is enabled and a new supervised commit revision is saved,
the app may enqueue an OCR fine-tune job. If a previously consumed revision is
changed, the registry marks `needs_rebase` and may enqueue a rebase job over the
approved supervised revisions.

## Job Orchestration And Device Leases

The app uses a generic job orchestrator for background work. It supports:

- queued job records
- priorities
- isolated OCR fine-tune and rebase jobs
- GPU device leases
- cancellation
- requeue-on-cancel behavior
- queue wait and runtime status reporting

This keeps heavy OCR training work outside the interactive request path and lets
interactive OCR remain responsive while background jobs are queued or running.

## Telemetry And Profiling

Production saves and jobs record structured telemetry and coarse profiling
summaries. Optional sampled CUDA traces may be collected when enabled.

Telemetry is used to explain what the app did and why, including:

- page save events
- layout edit metrics
- text edit metrics
- OCR job lifecycle events
- queued/running/requeued/canceled status
- active-learning entry decisions
- promotion or fallback summaries

Generated telemetry and profiling artifacts are runtime evidence, not research
source-of-truth config.

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

Telemetry:

```powershell
$env:CONDA_NO_PLUGINS='true'
conda run -n gnn_layout python -m unittest app.tests.test_recognition_telemetry_unit -v
```

Production strategy adoption and crop behavior:

```powershell
$env:CONDA_NO_PLUGINS='true'
conda run -n gnn_layout python -m unittest app.tests.test_strategy_adoption_unit -v
conda run -n gnn_layout python -m unittest app.tests.test_strategy_aware_ocr_crops_unit -v
```
