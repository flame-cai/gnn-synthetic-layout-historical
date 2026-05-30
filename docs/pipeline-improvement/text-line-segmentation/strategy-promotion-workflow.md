# Strategy Promotion And Production Adoption Workflow

This document records the checked-in lifecycle controls for text-line segmentation strategies. There are three independent role pins and two separate apply workflows:

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

- research benchmark: `local_polygons_stable_unwrap_v1`
- research proposed: unset
- production app default: `local_polygons_stable_unwrap_v1`

`local_polygons_v1` was adopted for production on 2026-05-23 after runtime support landed for strategy-owned config and reading-direction metadata. `legacy_axis_bound_v1` remains available after any research promotion or production adoption for rollback, historical comparison, and legacy-page fallback behavior.
`local_polygons_stable_unwrap_v1` was promoted into the research benchmark role on 2026-05-30 after passing the 2026-05-29 gate evidence. It is now a frozen independently owned implementation rather than a wrapper around the production `local_polygons_v1` class. It was separately adopted for production on 2026-05-30 after adding explicit app runtime config and marking the implementation production-independent.
Research strategy configs that must survive role changes are keyed by strategy in `app/tests/precommit_gate_config.py`; the current local-polygons and stable-unwrap research configs keep `BINARIZE_THRESHOLD=0.45` whether either role changes.
If no proposed strategy is configured, do not change the ablation gates to run benchmark-only as part of production adoption. Configure a proposed research strategy before running comparison gates again, or treat any benchmark-only health check as a separate research-harness maintenance change with its own review.

## Lifecycle Scope

The intended release train is:

1. an independent research proposed strategy competes against an independent research benchmark strategy
2. if all comparison gates pass, `scripts/promote_text_line_strategy.py` makes the proposed strategy the new `benchmark_strategy_name` and clears `proposed_strategy_name`
3. if the same winner should become the app default, `scripts/adopt_text_line_strategy_for_app.py` adopts that strategy as `production_strategy_name` in a separate apply step after production-readiness validation passes

The third step is explicit by design. It lets a research winner stop at the benchmark role when its evidence is valid for the harness but its production PAGE `Coords`, metadata, runtime config, or app-save behavior has not been reviewed yet.

## Strategy Independence Rules

Research benchmark/proposed roles must be independent implementations. A strategy used in either role must not delegate to another registered text-line strategy, subclass another strategy implementation, or import strategy-owned geometry/crop model code from the method it is competing against. Shared role-neutral infrastructure is allowed: request/result dataclasses, PAGE XML parsing/writing helpers, low-level geometry primitives, OCR crop execution, and registry plumbing.

Production defaults must also be independent app implementations. A strategy cannot be adopted for `production_strategy_name` unless it is marked production-independent and has an explicit production runtime config in `app/recognition/line_segmentation/runtime_config.py`.

The code enforces these rules through:

- `research_role_independent` and `production_role_independent` attributes on strategy classes
- `validate_strategy_role_config_payload_roles(...)` in the checked-in role config accessor
- `validate_research_role_strategy(...)` in research gate config and promotion
- `validate_production_role_strategy(...)` plus runtime-config validation in production adoption

Wrapper strategies such as `local_polygons_hstraight_smooth_unwrap_v1` are allowed to remain registered for experiments, but they must not be promoted or adopted until they are converted into owned implementations.

When creating a new proposed strategy:

1. put its strategy-owned implementation in its own module
2. do not call another registered strategy's `apply(...)`
3. do not import another strategy module's strategy-owned constants or helper functions
4. set `research_role_independent = True` only after the implementation is independently owned
5. add its research config to `app/tests/precommit_gate_config.py`
6. leave `production_role_independent = False` until an explicit app rollout adds runtime config and production validation
7. when the goal is production rollout after a successful benchmark promotion, run the adoption workflow separately and record that operational decision in `production_adoption_history`

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

This is the main reason harness promotion must not be treated as hidden production adoption. Cleanly adopting a baseline-derived strategy such as `local_polygons_v1` in the app is not just a config flip for research metrics. Production adoption changes future PAGE `Coords` generation and the strategy metadata that production OCR crop preparation honors. Existing pages remain valid and fall back to masked PAGE `Coords` crops when metadata is missing.

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

Dry run with the command from the latest promotion record:

```powershell
$env:CONDA_NO_PLUGINS='true'
conda run -n gnn_layout python scripts/promote_text_line_strategy.py --candidate local_polygons_stable_unwrap_v1 --previous-benchmark local_polygons_v1 --metrics app/tests/logs/strategy_promotion_latest.json
```

Apply only after reviewing the dry-run output:

```powershell
$env:CONDA_NO_PLUGINS='true'
conda run -n gnn_layout python scripts/promote_text_line_strategy.py --candidate local_polygons_stable_unwrap_v1 --previous-benchmark local_polygons_v1 --metrics app/tests/logs/strategy_promotion_latest.json --apply
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

Production adoption is the explicit app rollout step. It does not require new verifier evidence when adopting the already reviewed research winner, but it does require the strategy to be production-independent, configured for the app runtime, and validated against app save/regeneration behavior.

Dry run:

```powershell
$env:CONDA_NO_PLUGINS='true'
conda run -n gnn_layout python scripts/adopt_text_line_strategy_for_app.py --strategy <production_ready_strategy_name> --reason "adopt after reviewed app rollout evidence"
```

Apply:

```powershell
$env:CONDA_NO_PLUGINS='true'
conda run -n gnn_layout python scripts/adopt_text_line_strategy_for_app.py --strategy <production_ready_strategy_name> --reason "adopt after reviewed app rollout evidence" --apply
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
When adopting a research benchmark for production, first make it production-independent, add an explicit runtime config, and validate app saves/regenerations. Do not edit `app/tests/pipeline_ablation_experiment.py`, `app/tests/recognition_finetuning_experiment.py`, `app/tests/precommit_gate_config.py`, or `scripts/run_precommit_eval.py` unless the task is explicitly a separate research-harness change. Production adoption should be limited to app/runtime support and the production role fields in `app/recognition/line_segmentation/strategy_config.py`.

The 2026-05-23 production adoption command, when `local_polygons_v1` was the reviewed research benchmark, was:

```powershell
$env:CONDA_NO_PLUGINS='true'
conda run -n gnn_layout python scripts/adopt_text_line_strategy_for_app.py --strategy local_polygons_v1 --reason "Adopt current research benchmark with strategy-owned runtime config and reading-direction metadata support." --apply
```

The 2026-05-30 production adoption command, after `local_polygons_stable_unwrap_v1` became the reviewed research benchmark and gained production runtime config, was:

```powershell
$env:CONDA_NO_PLUGINS='true'
conda run -n gnn_layout python scripts/adopt_text_line_strategy_for_app.py --strategy local_polygons_stable_unwrap_v1 --reason "Adopt current research benchmark for production after stable unwrap runtime validation." --apply
```

## App OCR Behavior

The GUI runtime now uses the shared strategy-aware crop layer:

- app line-image export calls `crop_line_record_for_ocr(...)`
- local OCR inference discovers sibling line-segmentation metadata and falls back safely when it is absent
- active-learning revision snapshots preserve metadata sidecars and train from saved PAGE `Coords` plus metadata

With the current production pin `local_polygons_stable_unwrap_v1`, new layout saves write metadata that requests `crop_model="local_polygon_stable_unwrap"`. Production OCR then uses the stable local-polygon unwrap through the shared crop layer. Existing pages without usable metadata still use the historical masked PAGE `Coords` crop.

The app also stores optional reading-direction sidecars next to PAGE XML. Layout-mode `O` gestures write cross-line cuts that resolve open-line 180-degree ambiguity and circular unwrap start/direction. Active-learning snapshots copy this sidecar with PAGE XML and line-segmentation metadata. Missing, malformed, or stale annotations fall back to script defaults.

The `baseline_heatmap` verifier path remains a research/test harness path unless explicitly requested.
