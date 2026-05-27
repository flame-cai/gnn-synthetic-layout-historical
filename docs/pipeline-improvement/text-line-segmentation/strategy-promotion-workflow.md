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
The durable generated evidence summary is checked in at:

    docs/pipeline-improvement/text-line-segmentation/strategy-promotion-record.md

## Current State

The current checked-in state is:

- research benchmark: `local_polygons_v1`
- research proposed: not configured
- production app default: `local_polygons_v1`

`local_polygons_v1` was adopted for production on 2026-05-23 after runtime support landed for strategy-owned config and reading-direction metadata. `legacy_axis_bound_v1` remains available after any research promotion or production adoption for rollback, historical comparison, and legacy-page fallback behavior.
Configure a new `proposed_strategy_name` before running the next strategy ablation cycle.
Research strategy configs that must survive role changes are keyed by strategy in `app/tests/precommit_gate_config.py`; the current local-polygons research config keeps `BINARIZE_THRESHOLD=0.45` whether it is benchmark or proposed.
If no proposed strategy is configured, do not change the ablation gates to run benchmark-only as part of production adoption. Configure a proposed research strategy before running comparison gates again, or treat any benchmark-only health check as a separate research-harness maintenance change with its own review.

## Why The Lifecycles Are Separate

The production app and the research harness prepare OCR line images from different starting geometry, even though they now share the same crop decision layer.

Production GUI/runtime path:

- app save/regeneration writes PAGE `TextLine/Coords` using `production_strategy_name`
- app line-image export, GUI OCR inference, and GUI active-learning training read saved PAGE `Coords`
- optional line-segmentation metadata sidecars decide whether the OCR crop stays masked or uses a strategy-specific derived crop

Research verifier path:

- strategy ablation gates start from PAGE `Baseline` plus page image and heatmap
- the selected strategy regenerates PAGE-space `TextLine/Coords`
- OCR verifier crops are then prepared from that regenerated geometry through the shared crop layer

This is the main reason harness promotion must not be treated as production adoption. Cleanly adopting a baseline-derived strategy such as `local_polygons_v1` in the app is not just a config flip for research metrics. Production adoption changes future PAGE `Coords` generation and the strategy metadata that production OCR crop preparation honors. Existing pages remain valid and fall back to masked PAGE `Coords` crops when metadata is missing.

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
- `docs/pipeline-improvement/text-line-segmentation/strategy-promotion-record.md`

The launcher deletes passing gate role-run directories by default after each phase writes its latest aliases. This prunes the large benchmark/proposed OCR artifact trees, including `models/`, while retaining the latest aliases and gate summary directories needed for promotion evidence. Passing OCR role plots are copied into each retained gate summary directory under `plots/` before cleanup. Failed phases keep role-run directories for debugging. Set `CLEAN_UP=0` before the launcher command to keep full passing role artifacts.

The `app/tests/logs/` files are local generated artifacts. Commit the checked-in promotion record with any research promotion so future readers can review the gate summary without needing ignored logs.

Dry run with the command from the 2026-05-22 promotion record:

```powershell
$env:CONDA_NO_PLUGINS='true'
conda run -n gnn_layout python scripts/promote_text_line_strategy.py --candidate local_polygons_v1 --previous-benchmark local_tangent_band_v1 --metrics app/tests/logs/strategy_promotion_latest.json
```

Apply only after reviewing the dry-run output:

```powershell
$env:CONDA_NO_PLUGINS='true'
conda run -n gnn_layout python scripts/promote_text_line_strategy.py --candidate local_polygons_v1 --previous-benchmark local_tangent_band_v1 --metrics app/tests/logs/strategy_promotion_latest.json --apply
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
- production OCR crop behavior

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
- research harness code or pre-commit ablation gate behavior
- existing PAGE XML
- existing OCR line images
- active-learning checkpoint lineage

Production adoption affects future layout saves/regenerations and the crop metadata produced for newly saved pages only. Existing manuscripts and pages are not migrated automatically.
When adopting the current research benchmark for production, do not edit `app/tests/pipeline_ablation_experiment.py`, `app/tests/recognition_finetuning_experiment.py`, `app/tests/precommit_gate_config.py`, or `scripts/run_precommit_eval.py` unless the task is explicitly a separate research-harness change. Production adoption should be limited to app/runtime support and the production role fields in `app/recognition/line_segmentation/strategy_config.py`.

The 2026-05-23 production adoption command was:

```powershell
$env:CONDA_NO_PLUGINS='true'
conda run -n gnn_layout python scripts/adopt_text_line_strategy_for_app.py --strategy local_polygons_v1 --reason "Adopt current research benchmark with strategy-owned runtime config and reading-direction metadata support." --apply
```

## App OCR Behavior

The GUI runtime now uses the shared strategy-aware crop layer:

- app line-image export calls `crop_line_record_for_ocr(...)`
- local OCR inference discovers sibling line-segmentation metadata and falls back safely when it is absent
- active-learning revision snapshots preserve metadata sidecars and train from saved PAGE `Coords` plus metadata

With the current production pin `local_polygons_v1`, new layout saves write metadata that requests `crop_model="local_polygon_unwrap"`. Production OCR then uses local-polygon unwrapping and median-color background through the shared crop layer. Existing pages without usable metadata still use the historical masked PAGE `Coords` crop.

The app also stores optional reading-direction sidecars next to PAGE XML. Layout-mode `O` gestures write cross-line cuts that resolve open-line 180-degree ambiguity and circular unwrap start/direction. Active-learning snapshots copy this sidecar with PAGE XML and line-segmentation metadata. Missing, malformed, or stale annotations fall back to script defaults.

The `baseline_heatmap` verifier path remains a research/test harness path unless explicitly requested.
