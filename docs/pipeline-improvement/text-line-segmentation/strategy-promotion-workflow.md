# Strategy Promotion And Production Adoption Workflow

This document records the checked-in lifecycle controls for text-line segmentation strategies. There are two separate workflows:

- harness promotion: move a passing proposed strategy into the research benchmark role
- production adoption: explicitly choose the strategy used by future app saves and regenerations

Harness promotion is not an app rollout.

## Source Of Truth

The checked-in role mapping lives in:

    app/recognition/line_segmentation/strategy_config.py

That file owns:

- `benchmark_strategy_name`
- `proposed_strategy_name`
- `production_strategy_name`
- `research_promotion_history`
- `production_adoption_history`

Generated artifacts under `app/tests/logs/` are evidence only.

## Current Initial State

The current checked-in state is:

- research benchmark: `legacy_axis_bound_v1`
- research proposed: `local_tangent_band_v1`
- production app default: `legacy_axis_bound_v1`

`legacy_axis_bound_v1` remains available after any research promotion for rollback, historical comparison, and production pinning.

## Why The Lifecycles Are Separate

The production app and the research harness currently prepare OCR line images from different starting geometry.

Production GUI/runtime path:

- app save/regeneration writes PAGE `TextLine/Coords` using `production_strategy_name`
- GUI OCR inference crops line images directly from the existing PAGE `Coords`
- GUI active-learning training still defaults to `pagexml_coords`

Research verifier path:

- strategy ablation gates start from PAGE `Baseline` plus page image and heatmap
- the selected strategy regenerates PAGE-space `TextLine/Coords`
- OCR verifier crops are then prepared from that regenerated geometry

This is the main reason harness promotion must not be treated as production adoption. Cleanly adopting a baseline-derived strategy such as `local_tangent_band_v1` in the app is not just a config flip for OCR behavior. It needs explicit infrastructure decisions around when to regenerate PAGE `Coords`, how to preserve existing pages without migration, how to record which geometry/crop strategy produced active-learning samples, and whether/when GUI OCR should use local-tangent unwrapped crops instead of direct masked crops from PAGE `Coords`.

## Harness Promotion

Harness promotion is evidence-gated. It changes only the research roles and research history.

Prerequisite:

```powershell
$env:CONDA_NO_PLUGINS='true'
conda run -n gnn_layout python scripts/run_precommit_eval.py
```

If all three gates run, the launcher writes:

- `app/tests/logs/strategy_promotion_latest.json`
- `app/tests/logs/strategy_promotion_latest.md`

Dry run:

```powershell
$env:CONDA_NO_PLUGINS='true'
conda run -n gnn_layout python scripts/promote_text_line_strategy.py --candidate local_tangent_band_v1 --previous-benchmark legacy_axis_bound_v1 --metrics app/tests/logs/strategy_promotion_latest.json
```

Apply only after reviewing the dry-run output:

```powershell
$env:CONDA_NO_PLUGINS='true'
conda run -n gnn_layout python scripts/promote_text_line_strategy.py --candidate local_tangent_band_v1 --previous-benchmark legacy_axis_bound_v1 --metrics app/tests/logs/strategy_promotion_latest.json --apply
```

Successful harness promotion:

1. moves the candidate into `benchmark_strategy_name`
2. clears `proposed_strategy_name`
3. appends `research_promotion_history`

It does not change:

- `production_strategy_name`
- `production_adoption_history`
- existing PAGE XML
- existing OCR line images
- GUI OCR crop behavior

## Production Adoption

Production adoption is the explicit app rollout step. It does not require verifier evidence because it is an operational decision after review.

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

Successful production adoption:

1. updates `production_strategy_name`
2. appends `production_adoption_history`

It does not change:

- `benchmark_strategy_name`
- `proposed_strategy_name`
- `research_promotion_history`
- existing PAGE XML
- existing OCR line images
- active-learning checkpoint lineage

Production adoption affects future layout saves and regenerations only. Existing manuscripts and pages are not migrated automatically.

## App OCR Non-Changes

This workflow does not integrate local-tangent OCR unwrapping into the GUI runtime.

Current GUI behavior remains:

- app save/regeneration uses `production_strategy_name` for PAGE `Coords`
- GUI OCR inference still crops from existing PAGE `Coords`
- GUI active-learning training still defaults to `pagexml_coords`

The `baseline_heatmap` verifier path remains a research/test harness path unless explicitly requested.
