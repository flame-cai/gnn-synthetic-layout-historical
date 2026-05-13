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

The checked-in initial role mapping for this harness is:

- benchmark: `legacy_axis_bound_v1`
- proposed: `local_tangent_band_v1`
- production app: `legacy_axis_bound_v1`

`legacy_axis_bound_v1` is the preserved historical benchmark.

`local_tangent_band_v1` is the first generalized strategy for vertical, curved, and circular text. It keeps the older behavior for simple horizontal lines by delegating those cases back to the legacy implementation.

`production_strategy_name` is independent of the research roles. The app uses it for future PAGE `Coords` generation during layout saves/regenerations. Existing PAGE XML, existing OCR line images, and active-learning lineage are not migrated automatically when the production strategy changes.

The crop-preparation boundary is intentional. The production GUI currently prepares OCR line images from existing PAGE `TextLine/Coords`; GUI OCR inference crops directly from those `Coords`, and GUI active-learning training still defaults to `pagexml_coords`. The research OCR ablation gates instead start from PAGE `Baseline` plus the page image and heatmap, regenerate `Coords` through the selected strategy, and then prepare OCR crops from the regenerated geometry. Because those paths do not have the same operational contract, adopting a strategy for the app requires explicit production infrastructure decisions rather than a silent consequence of harness promotion.

Production integration work that is intentionally not hidden inside harness promotion includes deciding when to regenerate PAGE `Coords`, how to avoid migrating existing pages unexpectedly, how to record the geometry/crop strategy used for active-learning samples, and whether GUI OCR should later use local-tangent unwrapped crops rather than direct masked crops from PAGE `Coords`.

## Three External Verifier Gates

All three gates run benchmark and proposed through the same strategy-aware implementation path.

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

Additional comparison metrics:

- `line_cer_50`
- `line_cer_75`
- `line_cer_range`

Success rule:

- benchmark role must pass its own thresholds
- proposed role must pass its own thresholds
- proposed metrics may regress by at most `0.01` absolute versus benchmark on the comparison metrics

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

This aggregate file summarizes:

- benchmark strategy name
- proposed strategy name
- pass/fail state for each gate
- primary metric comparison for each gate
- whether promotion is recommended

This file is still evidence only. It does not change the benchmark role by itself.

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
conda run -n gnn_layout python scripts/promote_text_line_strategy.py --candidate local_tangent_band_v1 --previous-benchmark legacy_axis_bound_v1 --metrics app/tests/logs/strategy_promotion_latest.json
```

Apply:

```powershell
$env:CONDA_NO_PLUGINS='true'
conda run -n gnn_layout python scripts/promote_text_line_strategy.py --candidate local_tangent_band_v1 --previous-benchmark legacy_axis_bound_v1 --metrics app/tests/logs/strategy_promotion_latest.json --apply
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

Production adoption is separate and explicit. It chooses the strategy used by future app layout saves and regenerations.

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

No migration happens automatically. Existing PAGE XML, existing OCR line images, and active-learning checkpoint lineage remain as they are. GUI OCR inference continues to crop from existing PAGE `Coords`, and GUI active-learning training still defaults to `pagexml_coords`.

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
