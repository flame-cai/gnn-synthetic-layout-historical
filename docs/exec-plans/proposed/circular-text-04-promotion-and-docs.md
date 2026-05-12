# Add Explicit Strategy Promotion Workflow And Documentation

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This document is maintained in accordance with `PLANS.md` from the repository root.

This is the fourth plan in the circular text support series. It depends on the strategy interface, ablation gates, and first proposed strategy from:

- `docs/exec-plans/proposed/circular-text-01-strategy-interface.md`
- `docs/exec-plans/proposed/circular-text-02-ablation-gates.md`
- `docs/exec-plans/proposed/circular-text-03-generalized-cropping-unwrapping.md`

It uses `docs/exec-plans/proposed/circular-text-support.md` as the research source and must not overwrite that file.

## Purpose / Big Picture

After this change, strategy promotion will be explicit, reviewable, and documented. When the proposed strategy passes all gates, a developer can run a promotion script that updates checked-in config so the proposed strategy becomes the new benchmark. A normal pre-commit hook will run gates and report pass/fail evidence, but it will not silently mutate files after Git has already built the commit index.

The observable behavior is a safe workflow:

1. Run the ablation gates.
2. Inspect latest artifacts.
3. Run an explicit promotion command.
4. Review the config and documentation diff.
5. Commit the already-reviewed changes.

## Progress

- [x] (2026-05-09 22:31 IST) Confirmed the research source asks for promotion from `proposed_strategy` to `benchmark_strategy` after gates pass.
- [x] (2026-05-09 22:31 IST) Confirmed normal pre-commit mutation is unsafe because `.git/hooks/pre-commit` runs after Git has prepared the candidate commit index.
- [x] (2026-05-09 22:31 IST) Confirmed the current checked-in `.githooks/pre-commit` begins with `exit 0`, so hook activation needs an explicit decision rather than being assumed.
- [x] (2026-05-12 17:42 IST) Added checked-in strategy role config at `app/recognition/line_segmentation/strategy_config.py` with benchmark, proposed, and promotion history state.
- [x] (2026-05-12 17:42 IST) Added `scripts/promote_text_line_strategy.py` with dry-run default, `--apply`, evidence validation, and idempotent history handling.
- [x] (2026-05-12 17:42 IST) Updated `scripts/run_precommit_eval.py` to write aggregate promotion evidence and print the explicit promotion command without mutating tracked config.
- [x] (2026-05-12 17:42 IST) Activated `.githooks/pre-commit` as a launcher once hooks are installed and documented the explicit bypass variables.
- [x] (2026-05-12 17:42 IST) Updated `README.md`, added `EVAL.md`, updated `VISION.md`, and refreshed text-line strategy docs with the promotion workflow.
- [x] (2026-05-12 17:42 IST) Added promotion unit tests for dry run, apply, idempotence, missing evidence, failing gates, mismatched evidence, and unregistered candidates.
- [ ] Run the full validation suite and record outcomes.

## Surprises & Discoveries

- Observation: a pre-commit hook should not silently promote strategy config after gates pass.
  Evidence: Git runs pre-commit after the developer has staged files for the commit. If the hook mutates tracked config at that point, the working tree changes are not automatically included in the commit the user is making.

- Observation: the current hook file is present but inert.
  Evidence: `.githooks/pre-commit` contains `exit 0` before the launcher code. This may be intentional local safety. Any activation should be documented and reviewable.

## Decision Log

- Decision: promotion must be an explicit script or command, not silent mutation inside `scripts/run_precommit_eval.py` or `.githooks/pre-commit`.
  Rationale: the promotion changes tracked source config. Mutating tracked files during pre-commit would create hidden unstaged changes and confuse reproducibility.
  Date/Author: 2026-05-09 / Codex

- Decision: gate artifacts are evidence, but checked-in config remains the source of truth for active benchmark/proposed strategy names.
  Rationale: artifacts under `app/tests/logs/` may not exist in a fresh checkout. A future agent must be able to understand strategy state from source files and docs.
  Date/Author: 2026-05-09 / Codex

- Decision: promotion should preserve the previous benchmark name in history metadata instead of deleting it.
  Rationale: future ablations need to understand what changed and why. Keeping a promotion history makes results interpretable.
  Date/Author: 2026-05-09 / Codex

- Decision: use a checked-in Python module for strategy role config rather than environment variables or generated logs.
  Rationale: the benchmark/proposed mapping must remain readable in a fresh checkout and easy to update through a normal source diff.
  Date/Author: 2026-05-12 / Codex

- Decision: activate `.githooks/pre-commit` as a launcher once the user installs hooks, but keep promotion as a separate explicit command.
  Rationale: automatic gate execution is useful; automatic tracked-source mutation during pre-commit is not.
  Date/Author: 2026-05-12 / Codex

## Outcomes & Retrospective

Implemented on 2026-05-12 except for the final validation run inventory. The checked-in source-of-truth file is `app/recognition/line_segmentation/strategy_config.py`. The explicit promotion command is:

    python scripts/promote_text_line_strategy.py --candidate <strategy> --previous-benchmark <strategy> --metrics app/tests/logs/strategy_promotion_latest.json --apply

The launcher now writes promotion evidence to:

    app/tests/logs/strategy_promotion_latest.json
    app/tests/logs/strategy_promotion_latest.md

The main documentation updates landed in:

    README.md
    EVAL.md
    VISION.md
    docs/pipeline-improvement/text-line-segmentation/local-tangent-band-v1-architecture.md
    docs/pipeline-improvement/text-line-segmentation/strategy-promotion-workflow.md

## Context and Orientation

The strategy series introduces two stable strategy names:

    legacy_axis_bound_v1
    local_tangent_band_v1

`legacy_axis_bound_v1` is the benchmark strategy extracted from current behavior. `local_tangent_band_v1` is the first proposed strategy for curved, vertical, and circular text.

The ablation gates from plan 02 compare benchmark and proposed roles. The roles are not permanent strategy names. After promotion, the strategy that was proposed becomes the benchmark for the next research iteration, and a newer proposed strategy can be added.

Promotion means updating checked-in configuration so the default benchmark strategy changes. It does not mean deleting old code. It also does not mean editing generated logs. Generated logs are run evidence only.

Important current files:

    scripts/run_precommit_eval.py
    .githooks/pre-commit
    app/tests/precommit_gate_config.py
    app/tests/recognition_finetuning_config.py
    app/recognition/active_learning_recipe.py
    README.md
    EVAL.md
    VISION.md
    docs/exec-plans/proposed/circular-text-support.md

Plan 04 should add a small checked-in source of truth for strategy roles. A good target is:

    app/recognition/line_segmentation/strategy_config.py

or, if the project prefers data files:

    app/recognition/line_segmentation/strategy_config.json

Use a Python module if the existing codebase patterns make imports simpler. Use JSON if human editability and script updates are more important. Whichever form is chosen, document it in `Decision Log` and keep it easy to diff.

## Plan of Work

Create one checked-in strategy role config with this information:

    benchmark_strategy_name="legacy_axis_bound_v1"
    proposed_strategy_name="local_tangent_band_v1"
    promotion_history=[]

After `local_tangent_band_v1` passes all gates, the promotion script should update it to:

    benchmark_strategy_name="local_tangent_band_v1"
    proposed_strategy_name="<unset or next strategy name>"

and append a promotion history entry with:

    promoted_strategy_name
    previous_benchmark_strategy_name
    gate_artifact_paths
    gate_metric_summary
    promotion_timestamp
    author_or_tool

Add a script:

    scripts/promote_text_line_strategy.py

The script should support a dry run by default and require `--apply` to write files. It should accept:

    --candidate local_tangent_band_v1
    --previous-benchmark legacy_axis_bound_v1
    --metrics app/tests/logs/strategy_promotion_latest.json
    --apply

If the metric path is omitted, the script may read the latest known ablation JSONs from `app/tests/logs/`, but it must print exactly which files it used. It should refuse promotion if any required artifact is missing, stale, names a different candidate, names a different benchmark, or reports failure.

Create or update an aggregate promotion evidence file when gates pass:

    app/tests/logs/strategy_promotion_latest.json
    app/tests/logs/strategy_promotion_latest.md

This file should summarize:

- full-pipeline ablation status on `eval_dataset`
- OCR fine-tuning ablation status on `eval_dataset`
- circular OCR fine-tuning ablation status on `eval_dataset_v2`
- benchmark strategy name
- proposed strategy name
- primary metric comparison for each gate
- whether promotion is recommended

This aggregate file is still evidence, not source of truth. The promotion script reads it only to verify a proposed config change.

Update `scripts/run_precommit_eval.py` so after all phases pass it prints whether promotion is recommended and the exact explicit command to run. It should not call the promotion script with `--apply`.

Update `.githooks/pre-commit` intentionally. If the project wants hooks active, remove the leading `exit 0` and keep the hook as a launcher only. If the project wants hooks disabled by default, keep `exit 0` and document that developers should run `scripts/run_precommit_eval.py` manually. Do not leave the file's behavior unexplained.

Update docs:

`README.md` should explain that pre-commit evaluation now has three strategy ablation gates and list the manual promotion command.

`EVAL.md` should explain the three-gate evaluation architecture, the circular `eval_dataset_v2` split, and the rule that gate artifacts are evidence while checked-in config is source of truth.

`VISION.md` should explain the iterative strategy improvement loop: proposed strategies are compared against the benchmark, promoted explicitly after gates pass, then a new proposed strategy can be researched.

If OCR fine-tuning behavior changed in any way while implementing the previous plans, update the relevant ExecPlan files and any current OCR docs in the same pass.

## Concrete Steps

Work from the repository root:

    cd c:\Users\intro\OneDrive\Documents\MEGA\CAI-FLAME\gnn-synthetic-layout-historical

Add or update:

    app/recognition/line_segmentation/strategy_config.py
    scripts/promote_text_line_strategy.py
    scripts/run_precommit_eval.py
    .githooks/pre-commit
    app/tests/test_strategy_promotion_unit.py
    README.md
    EVAL.md
    VISION.md

Promotion unit tests should create temporary strategy config and metrics files under `app/tests/_tmp_strategy_promotion_unit/`. They should cover:

- dry run does not modify config.
- `--apply` updates config when all three gates passed for the requested candidate and benchmark.
- running the same promotion twice is idempotent and does not append duplicate history entries.
- script refuses when a metrics file is missing.
- script refuses when any gate failed.
- script refuses when metrics candidate or benchmark does not match CLI arguments.
- script refuses when the candidate is not registered in the strategy registry.

The promotion script should use standard-library JSON and path handling only. Do not invoke Git from the script. Let the user review diffs with normal `git diff`.

## Validation and Acceptance

Run promotion unit tests:

    $env:CONDA_NO_PLUGINS='true'; conda run -n gnn_layout python -m unittest app.tests.test_strategy_promotion_unit -v

Expected result: all promotion refusal, dry-run, apply, and idempotence tests pass.

Run strategy and gate unit tests:

    $env:CONDA_NO_PLUGINS='true'; conda run -n gnn_layout python -m unittest app.tests.test_recognition_finetuning_precommit_unit -v
    $env:CONDA_NO_PLUGINS='true'; conda run -n gnn_layout python -m unittest app.tests.test_line_segmentation_strategy_unit -v
    $env:CONDA_NO_PLUGINS='true'; conda run -n gnn_layout python -m unittest app.tests.test_strategy_ablation_config_unit -v

Run the three integration gates:

    $env:CONDA_NO_PLUGINS='true'; conda run -n gnn_layout python -m unittest discover -s app/tests -p "test_ci_e2e.py" -v
    $env:CONDA_NO_PLUGINS='true'; conda run -n gnn_layout python -m unittest app.tests.test_recognition_finetuning_precommit_e2e -v
    $env:CONDA_NO_PLUGINS='true'; conda run -n gnn_layout python -m unittest app.tests.test_circular_recognition_finetuning_precommit_e2e -v

Run the complete launcher:

    $env:CONDA_NO_PLUGINS='true'; conda run -n gnn_layout python scripts/run_precommit_eval.py

Expected result: the launcher runs all phases and writes promotion evidence. It prints an explicit command similar to:

    python scripts/promote_text_line_strategy.py --candidate local_tangent_band_v1 --previous-benchmark legacy_axis_bound_v1 --metrics app/tests/logs/strategy_promotion_latest.json --apply

Run the promotion dry run:

    $env:CONDA_NO_PLUGINS='true'; conda run -n gnn_layout python scripts/promote_text_line_strategy.py --candidate local_tangent_band_v1 --previous-benchmark legacy_axis_bound_v1 --metrics app/tests/logs/strategy_promotion_latest.json

Expected result: no files are modified, and the script prints the config changes it would make.

Run promotion apply only after all gates passed:

    $env:CONDA_NO_PLUGINS='true'; conda run -n gnn_layout python scripts/promote_text_line_strategy.py --candidate local_tangent_band_v1 --previous-benchmark legacy_axis_bound_v1 --metrics app/tests/logs/strategy_promotion_latest.json --apply

Expected result: the checked-in strategy config changes, a promotion history entry is added, and rerunning the command does not duplicate that entry.

For long OCR runs on Windows, use direct interpreter fallback:

    C:\Users\intro\miniconda3\envs\gnn_layout\python.exe scripts/run_precommit_eval.py
    C:\Users\intro\miniconda3\envs\gnn_layout\python.exe scripts/promote_text_line_strategy.py --candidate local_tangent_band_v1 --previous-benchmark legacy_axis_bound_v1 --metrics app/tests/logs/strategy_promotion_latest.json --apply

Acceptance requires a clean, reviewable diff showing source config and docs changes. It is not acceptable for promotion to happen only in generated logs.

## Idempotence and Recovery

The promotion script must be safe to rerun. If the requested candidate is already the benchmark and the same metrics evidence is already recorded, it should report that no change is needed and exit successfully.

If promotion fails validation, the script must leave config unchanged. Implement writes by preparing the full new file content first, then replacing the file only after all checks pass.

If `.githooks/pre-commit` is activated, document how to bypass intentionally with `git commit --no-verify` and keep existing skip variables:

    SKIP_EVAL_HOOK=1
    SKIP_PIPELINE_EVAL_HOOK=1
    SKIP_RECOGNITION_FT_HOOK=1
    SKIP_CIRCULAR_RECOGNITION_FT_HOOK=1

Do not remove old strategy code during promotion. A promoted benchmark may still be needed for historical comparison and rollback.

## Artifacts and Notes

Promotion evidence should be concise and checked by tests. A minimal `strategy_promotion_latest.json` shape is:

    {
      "study_mode": "strategy_promotion_evidence",
      "benchmark_strategy_name": "legacy_axis_bound_v1",
      "proposed_strategy_name": "local_tangent_band_v1",
      "promotion_recommended": true,
      "gate_results": {
        "pipeline_eval_dataset": {"passed": true, "primary_metric_name": "page_cer"},
        "ocr_eval_dataset": {"passed": true, "primary_metric_name": "curve_metric_value"},
        "circular_ocr_eval_dataset_v2": {"passed": true, "primary_metric_name": "curve_metric_value"}
      }
    }

This JSON is evidence only. The strategy config file remains the source of truth.

## Interfaces and Dependencies

Required script interface:

    python scripts/promote_text_line_strategy.py --candidate <strategy> --previous-benchmark <strategy> --metrics <path> [--apply]

Required config accessors:

    get_benchmark_strategy_name() -> str
    get_proposed_strategy_name() -> str | None
    get_strategy_role_config() -> dict

Required promotion script behavior:

- default dry run
- `--apply` required for writes
- no Git commands
- no writes outside the repository
- refuses missing, stale, mismatched, or failing evidence
- idempotent when repeated

Use only standard-library file and JSON tooling for promotion. Continue to use existing unittest and conda validation commands.

## Change Note

Initial split plan created on 2026-05-09. This plan keeps promotion separate from pre-commit execution so a successful gate run never creates hidden unstaged source mutations.
