# EVAL.md

This document explains the current verifier-driven evaluation harness, the research-only promotion workflow for text-line segmentation strategy research, and the separate production adoption workflow for the app. It is also written as a reusable template for future agents who want to apply the same harness to a different pipeline stage.

## Core Principle

The harness compares one checked-in `benchmark_strategy` against one checked-in `proposed_strategy` using the same plumbing and the same external verifiers.

The production app separately reads one checked-in `production_strategy`. Research promotion does not change that production app pin.

Generated log artifacts are evidence.

Checked-in source config is the source of truth.

That split matters because local artifacts under `app/tests/logs/` may not exist in a fresh checkout, while source config must always reveal:

- which strategy currently owns the benchmark role
- which strategy is the current promotion candidate
- which strategy the app uses for future layout saves and regenerations
- what the research promotion and production adoption histories were

## Current Harness Target

The current target stage is the text-line segmentation step that converts:

- resized page images
- heatmaps
- PAGE-XML `Baseline` polylines derived from GNN predictions

into:

- PAGE-space `TextLine/Coords` polygons
- OCR-ready text-line crops used by recognition fine-tuning and inference

This stage exists because the historical cropper worked mainly for horizontal lines. The current research harness evaluates whether a newer strategy improves curved, circular, and vertical lines without breaking the horizontal baseline.

## Source Of Truth Files

The current checked-in strategy-role state lives in:

- `app/recognition/line_segmentation/strategy_config.py`

The promotion and evidence tooling lives in:

- `scripts/run_precommit_eval.py`
- `scripts/promote_text_line_strategy.py`

The production adoption tooling lives in:

- `scripts/adopt_text_line_strategy_for_app.py`

The strategy comparison config lives in:

- `app/tests/precommit_gate_config.py`
- `app/tests/recognition_finetuning_config.py`

The strategy docs live in:

- `docs/pipeline-improvement/text-line-segmentation/local-tangent-band-v1-architecture.md`
- `docs/pipeline-improvement/text-line-segmentation/strategy-promotion-workflow.md`

## Strategy Roles

The checked-in current role mapping for this harness is:

- benchmark: `local_polygons_v1`
- proposed: not configured
- production app: `local_polygons_v1`

`local_polygons_v1` is the current research benchmark after harness promotion on 2026-05-22 and the current production app strategy after explicit production adoption on 2026-05-23. Configure the next proposed strategy before running the next three-gate ablation cycle.

`legacy_axis_bound_v1` is the preserved historical benchmark and remains registered for rollback, historical comparison, and legacy-page fallback behavior.

`local_tangent_band_v1` is the first generalized strategy for vertical, curved, and circular text. It keeps the older behavior for simple horizontal lines by delegating those cases back to the legacy implementation.

`local_polygons_v1` builds PAGE `Coords` in a baseline-local frame from heatmap contour evidence, applies local top/bottom cleanup, and requests local-polygon unwrapping for OCR crops. Its contour-based local mask projection avoids treating large page-axis-aligned heatmap boxes as the true normal height for circular text. Its research-harness heatmap contours are binarized at `0.45` before local cleanup so weak detached-mark evidence reaches the boundary trimmer; open-line final-mask normal padding stays at zero, while closed circular lines retain a separate final-mask override. `app/tests/precommit_gate_config.py` attaches that research override by strategy name so it remains in force when the strategy changes from proposed to benchmark.

`production_strategy_name` is independent of the research roles. The app uses it for future PAGE `Coords` generation during layout saves/regenerations, and production OCR crop preparation routes through the same strategy-aware crop layer. Existing PAGE XML, existing OCR line images, and active-learning lineage are not migrated automatically when the production strategy changes.

The crop-preparation boundary is intentional. The production GUI prepares OCR line images from saved PAGE `TextLine/Coords` and optional sibling line-segmentation metadata through `app/recognition/line_segmentation/ocr_crops.py`. For new `local_polygons_v1` saves, line metadata requests `crop_model="local_polygon_unwrap"` so local OCR, app line-image export, and active-learning revision training consume the same unwrapped median-background crop used by the benchmark harness. Missing, malformed, stale, legacy, or unsupported metadata falls back to the existing masked PAGE `Coords` crop. The research OCR ablation gates instead start from PAGE `Baseline` plus the page image and heatmap, regenerate `Coords` through the selected strategy, and then use the same crop decision layer on the regenerated geometry. Because those paths do not have the same operational contract, production adoption remains explicit rather than a silent consequence of harness promotion.

Reading-direction annotations are optional production metadata. The layout GUI writes `<page>_reading_direction_metadata.json` from cross-line `O` gestures. The app resolves annotations by component overlap on save, marks stale annotations instead of guessing, and includes the sidecar in layout fingerprints and active-learning revision snapshots. If no active annotation exists, the strategy uses script defaults: horizontal left-to-right, vertical top-to-bottom, and circular clockwise with a top cut.

## Three External Verifier Gates

All three gates run benchmark and proposed through the same strategy-aware implementation path.

When `proposed_strategy_name` is not configured, the fast pipeline and OCR strategy-ablation launchers run the benchmark role and record the strategy comparison as skipped instead of failing the gate on missing proposal state. Configure a proposed strategy before using the gates as promotion evidence.

### 1. Pretrained Full-Pipeline Gate

Purpose:

- verify that end-to-end manuscript processing still works with pretrained CRAFT, GNN, and OCR
- catch regressions in the layout-to-crop stage before OCR fine-tuning even begins

Dataset:

- `app/tests/eval_dataset/`

Command:

```powershell
$env:CONDA_NO_PLUGINS='true'
conda run -n gnn_layout python -m unittest discover -s app/tests -p "test_ci_e2e.py" -v
```

Study mode:

- `pipeline_strategy_ablation_gate`

Primary comparison metric:

- `page_cer`

Diagnostic metrics written to artifacts but not used for pass/fail:

- `line_cer_50`
- `line_cer_75`
- `line_cer_range`

Success rule:

- benchmark role must pass its own `page_cer` threshold
- proposed role must pass its own `page_cer` threshold
- proposed `page_cer` may regress by at most `0.01` absolute versus benchmark

Latest artifact files:

- `app/tests/logs/pipeline_ablation_latest.json`
- `app/tests/logs/pipeline_ablation_latest.md`

### 2. OCR Fine-Tuning Strategy Ablation Gate

Purpose:

- test whether the strategy produces OCR crops that work well with the retained surrogate recognition fine-tuning recipe on the regular evaluation manuscript

Dataset:

- `app/tests/eval_dataset/`

Command:

```powershell
$env:CONDA_NO_PLUGINS='true'
conda run -n gnn_layout python -m unittest app.tests.test_recognition_finetuning_precommit_e2e -v
```

Study mode:

- `recognition_strategy_ablation_gate`

Recipe shape:

- `fine_tune_page_count=3`, configured in `app/tests/precommit_gate_config.py` for the pre-commit gate while slower research studies may use a longer prefix
- `page_plus_random_history`
- `history_sample_line_count=10`
- `width_policy=batch_max_pad`
- `oversampling_policy=none`
- `augmentation_policy=none`
- `optimizer=Adadelta`
- `lr=0.2`
- `num_iter=60`

Primary comparison metric:

- `curve_metric_value`

Additional comparison metrics:

- `final_page_cer`
- `first_step_gain`

Success rule:

- benchmark role must pass
- proposed role must pass
- proposed may regress by at most `0.02` absolute against the benchmark on the ablation comparison

Blocking dataset thresholds:

- `curve_metric_value <= 0.26`
- `final_page_cer <= 0.18`
- `first_step_gain >= 0.04`

Geometry guard before OCR fine-tuning:

- `source_line_coverage >= 0.90`
- `heatmap_box_assignment_rate >= 0.90`

Latest artifact files:

- `app/tests/logs/recognition_finetune_ablation_latest.json`
- `app/tests/logs/recognition_finetune_ablation_latest.md`
- `app/tests/logs/recognition_finetune_ablation_latest.txt`

### 3. Circular OCR Fine-Tuning Strategy Ablation Gate

Purpose:

- focus on the circular and non-horizontal failure mode that motivated the new strategy

Dataset:

- `app/tests/eval_dataset_v2/`

Command:

```powershell
$env:CONDA_NO_PLUGINS='true'
conda run -n gnn_layout python -m unittest app.tests.test_circular_recognition_finetuning_precommit_e2e -v
```

Study mode:

- `circular_recognition_strategy_ablation_gate`

Primary comparison metric:

- `curve_metric_value`

Additional comparison metrics:

- `final_page_cer`
- `first_step_gain`

Success rule:

- benchmark role must pass
- proposed role must pass
- proposed must be strictly better on the primary metric
- no absolute regression allowance is granted on the primary metric for this gate

Blocking behavior:

- only the strict primary comparison is blocking in this circular gate
- secondary metrics are still reported, but they are informational when strict primary improvement mode is active

Dataset-level thresholds remain:

- `curve_metric_value <= 0.26`
- `final_page_cer <= 0.18`
- `first_step_gain >= 0.04`

Latest artifact files:

- `app/tests/logs/circular_ocr_ablation_latest.json`
- `app/tests/logs/circular_ocr_ablation_latest.md`
- `app/tests/logs/circular_ocr_ablation_latest.txt`

## Aggregate Promotion Evidence

When all three gates run and pass through `scripts/run_precommit_eval.py`, the launcher writes:

- `app/tests/logs/strategy_promotion_latest.json`
- `app/tests/logs/strategy_promotion_latest.md`
- `docs/pipeline-improvement/text-line-segmentation/strategy-promotion-record.md`

The launcher deletes passing gate role-run directories by default after each phase writes its latest aliases. This removes the large benchmark/proposed OCR artifact trees, including per-step `models/` folders, while preserving the latest aliases and small gate summary directories required for promotion evidence. Passing OCR role plots are copied into each retained summary directory under `plots/` before role cleanup. Failed phases keep role-run directories for diagnosis. Set `CLEAN_UP=0` before running the launcher to retain full passing role artifacts.

This aggregate file summarizes:

- benchmark strategy name
- proposed strategy name
- pass/fail state for each gate
- primary metric comparison for each gate
- whether promotion is recommended

The files under `app/tests/logs/` are local generated artifacts and may be ignored in a fresh checkout. The checked-in promotion record is the durable summary to review and commit with a promotion. It is still evidence only; it does not change the benchmark role by itself.

## Research Harness Promotion Workflow

Harness promotion is explicit and research-only:

1. Run `scripts/run_precommit_eval.py`.
2. Inspect the latest gate artifacts and aggregate promotion evidence.
3. Run the promotion script in dry-run mode first.
4. Run it again with `--apply` only if the diff is correct.
5. Review the source diff.
6. Commit the already-reviewed config and doc changes.

Dry run:

```powershell
$env:CONDA_NO_PLUGINS='true'
conda run -n gnn_layout python scripts/promote_text_line_strategy.py --candidate local_polygons_v1 --previous-benchmark local_tangent_band_v1 --metrics app/tests/logs/strategy_promotion_latest.json
```

Apply:

```powershell
$env:CONDA_NO_PLUGINS='true'
conda run -n gnn_layout python scripts/promote_text_line_strategy.py --candidate local_polygons_v1 --previous-benchmark local_tangent_band_v1 --metrics app/tests/logs/strategy_promotion_latest.json --apply
```

The promotion script refuses to write when:

- the evidence file is missing
- the evidence file is stale relative to its referenced artifacts
- any required gate failed
- the evidence names a different benchmark or candidate than the CLI request
- the candidate strategy is not registered

On success it updates `app/recognition/line_segmentation/strategy_config.py`, moves the candidate into the research benchmark slot, clears the proposed slot, and appends `research_promotion_history`.

It does not change `production_strategy_name` or `production_adoption_history`. A harness promotion must not be treated as a GUI/app rollout.

## Production Adoption Workflow

Production adoption is separate and explicit. It chooses the strategy used by future app layout saves/regenerations and the crop behavior honored by production OCR preparation metadata.

Dry run:

```powershell
$env:CONDA_NO_PLUGINS='true'
conda run -n gnn_layout python scripts/adopt_text_line_strategy_for_app.py --strategy local_tangent_band_v1 --reason "adopt after reviewed harness evidence"
```

Apply:

```powershell
$env:CONDA_NO_PLUGINS='true'
conda run -n gnn_layout python scripts/adopt_text_line_strategy_for_app.py --strategy local_tangent_band_v1 --reason "adopt after reviewed harness evidence" --apply
```

The adoption script validates that the named strategy is registered. On success it updates only `production_strategy_name` and `production_adoption_history`; it does not mutate the research benchmark/proposed roles and does not require verifier evidence.

The 2026-05-23 production adoption used the same command shape with `--strategy local_polygons_v1` and recorded the change in `production_adoption_history`.

No migration happens automatically. Existing PAGE XML, existing OCR line images, and active-learning checkpoint lineage remain as they are. GUI OCR inference, app line-image export, and active-learning training all read saved PAGE `Coords`; when a sibling strategy metadata sidecar exists, they use it only to decide the derived OCR crop representation. Missing metadata remains a valid masked-crop fallback.

## Hook Behavior

The checked-in hook launcher is:

- `.githooks/pre-commit`

After `python scripts/install_git_hooks.py`, a normal `git commit` runs the three gates through `scripts/run_precommit_eval.py`.

The hook does not promote automatically.

Intentional bypasses remain available:

- `git commit --no-verify`
- `SKIP_EVAL_HOOK=1`
- `SKIP_PIPELINE_EVAL_HOOK=1`
- `SKIP_RECOGNITION_FT_HOOK=1`
- `SKIP_CIRCULAR_RECOGNITION_FT_HOOK=1`

If any gate is skipped, the launcher does not refresh the aggregate promotion evidence.

## Validation Commands

Promotion unit tests:

```powershell
$env:CONDA_NO_PLUGINS='true'
conda run -n gnn_layout python -m unittest app.tests.test_strategy_promotion_unit -v
conda run -n gnn_layout python -m unittest app.tests.test_strategy_adoption_unit -v
```

Strategy configuration and crop behavior:

```powershell
$env:CONDA_NO_PLUGINS='true'
conda run -n gnn_layout python -m unittest app.tests.test_strategy_aware_ocr_crops_unit -v
conda run -n gnn_layout python -m unittest app.tests.test_strategy_ablation_config_unit -v
conda run -n gnn_layout python -m unittest app.tests.test_line_segmentation_strategy_unit -v
conda run -n gnn_layout python -m unittest app.tests.test_recognition_finetuning_precommit_unit -v
```

Full launcher:

```powershell
$env:CONDA_NO_PLUGINS='true'
conda run -n gnn_layout python scripts/run_precommit_eval.py
```

## How To Adapt This Harness To Another Pipeline Stage

Future agents should preserve the same structure instead of inventing a new one-off evaluation loop.

1. Specify the stage boundary precisely.
   Inputs: exact files, tensors, records, or PAGE elements consumed.
   Outputs: exact files, tensors, records, or PAGE elements produced.

2. Create stable strategy names.
   Keep one benchmark and one proposed role in checked-in config.

3. Reuse the same external verifiers for both roles.
   The only intentional variable should be the strategy itself.

4. Define success criteria in checked-in source.
   Include primary metric, secondary metrics, thresholds, and regression allowances.

5. Separate evidence from source-of-truth config.
   Gate artifacts can be regenerated; strategy-role config must remain readable in a fresh clone.

6. Keep promotion explicit.
   A gate run may recommend promotion, but it should not silently rewrite tracked source during pre-commit.

7. Retain old strategy code and research promotion history.
   Research branches need rollback and historical comparison, not destructive replacement.


okay now please make the following changes:
1) Add a checked-in promotion record doc generated by the script, since app/tests/logs/ is ignored. Maybe in C:\Users\intro\OneDrive\Documents\MEGA\CAI-FLAME\gnn-synthetic-layout-historical\docs\pipeline-improvement\text-line-segmentation?
