# Strategy Promotion Workflow

This document records the checked-in promotion workflow for text-line segmentation strategies.

## Source Of Truth

The checked-in benchmark/proposed role mapping lives in:

    app/recognition/line_segmentation/strategy_config.py

That file is the source of truth for:

- `benchmark_strategy_name`
- `proposed_strategy_name`
- `promotion_history`

Generated artifacts under `app/tests/logs/` are evidence only.

## Current Initial State

The current checked-in initial state for this harness is:

- benchmark: `legacy_axis_bound_v1`
- proposed: `local_tangent_band_v1`

`legacy_axis_bound_v1` remains available after promotion for rollback and historical comparison.

## Evidence Generation

Run the three-gate launcher:

```powershell
$env:CONDA_NO_PLUGINS='true'
conda run -n gnn_layout python scripts/run_precommit_eval.py
```

If all three gates run, the launcher writes:

- `app/tests/logs/strategy_promotion_latest.json`
- `app/tests/logs/strategy_promotion_latest.md`

Those files summarize:

- benchmark strategy name
- proposed strategy name
- pass/fail status for each gate
- primary metric comparisons
- whether promotion is recommended

## Promotion

Review the evidence first, then run a dry run:

```powershell
$env:CONDA_NO_PLUGINS='true'
conda run -n gnn_layout python scripts/promote_text_line_strategy.py --candidate local_tangent_band_v1 --previous-benchmark legacy_axis_bound_v1 --metrics app/tests/logs/strategy_promotion_latest.json
```

Apply only after review:

```powershell
$env:CONDA_NO_PLUGINS='true'
conda run -n gnn_layout python scripts/promote_text_line_strategy.py --candidate local_tangent_band_v1 --previous-benchmark legacy_axis_bound_v1 --metrics app/tests/logs/strategy_promotion_latest.json --apply
```

The script refuses to write when the evidence is missing, stale, failing, mismatched, or names an unregistered strategy.

## Promotion Result

Successful promotion does three things:

1. moves the candidate into `benchmark_strategy_name`
2. clears `proposed_strategy_name` so the next research iteration must set a new candidate explicitly
3. appends a history entry with evidence paths, primary metric summaries, timestamp, and tool identity

This workflow keeps source changes reviewable and avoids the unsafe pattern of mutating tracked files inside Git pre-commit after the candidate commit index is already built.
