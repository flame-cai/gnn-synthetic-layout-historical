# Add Strategy Ablation Gates For Pre-Commit Evaluation

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This document is maintained in accordance with `PLANS.md` from the repository root.

This is the second plan in the circular text support series. It depends on the shared strategy interface from `docs/exec-plans/proposed/circular-text-01-strategy-interface.md` and uses `docs/exec-plans/proposed/circular-text-support.md` as the research source.

## Purpose / Big Picture

After this change, pre-commit evaluation will compare a benchmark text-line segmentation strategy against a proposed strategy in three gates: a pretrained full-pipeline gate on `eval_dataset`, an OCR fine-tuning gate on `eval_dataset`, and a circular-layout OCR fine-tuning gate on `eval_dataset_v2`. The observable behavior is that one command, `python scripts/run_precommit_eval.py`, runs all configured gates and writes logs showing benchmark results, proposed results, and whether the proposed strategy is acceptable.

This plan does not implement the first proposed circular strategy itself. Until plan 03 lands, the proposed strategy can be configured to a placeholder or to the same implementation as the benchmark only for plumbing tests. The gate framework must still run benchmark and proposed through the same implementation path so that later strategy changes are true ablations.

## Progress

- [x] (2026-05-09 22:31 IST) Confirmed the current launcher has two phases in `scripts/run_precommit_eval.py`: full pipeline and recognition fine-tune.
- [x] (2026-05-09 22:31 IST) Confirmed current config registries only include `eval_dataset` in `app/tests/precommit_gate_config.py` and `app/tests/recognition_finetuning_config.py`.
- [x] (2026-05-09 22:31 IST) Confirmed `app/tests/eval_dataset_v2/` exists with images and heatmaps for ordered pages `page_2` through `page_6`, and PAGE-XML labels under `labels/PAGE-XML/`.
- [x] (2026-05-09 22:31 IST) Confirmed `app/tests/circular_OCR_test/` has only `__pycache__` entries and no checked-in source implementation for a circular gate.
- [x] (2026-05-10 02:25 IST) Added shared ablation config data classes and registry entries for benchmark and proposed strategies in `app/tests/precommit_gate_config.py`.
- [x] (2026-05-10 02:25 IST) Reworked the pretrained full-pipeline gate so benchmark and proposed use the same strategy-aware pipeline through `app/tests/pipeline_ablation_experiment.py`.
- [x] (2026-05-10 02:25 IST) Reworked the OCR fine-tuning gate so benchmark and proposed use the same strategy-aware crop preparation and OCR recipe through `run_recognition_strategy_ablation_gate(...)`.
- [x] (2026-05-10 02:25 IST) Implemented the circular OCR fine-tuning gate from source using `app/tests/eval_dataset_v2` and `run_circular_recognition_strategy_ablation_gate(...)`.
- [x] (2026-05-10 02:25 IST) Updated `scripts/run_precommit_eval.py` to run all three ablation gates and report strategy-specific artifacts.
- [x] (2026-05-10 02:25 IST) Added focused unit tests plus the new circular e2e test.
- [x] (2026-05-10 02:25 IST) Ran all validation commands in this plan and recorded results.

## Surprises & Discoveries

- Observation: the current `.githooks/pre-commit` begins with `exit 0`, so the hook is disabled even though `scripts/run_precommit_eval.py` exists.
  Evidence: reading `.githooks/pre-commit` shows `#!/bin/sh`, then `exit 0`, then the launcher code. This plan updates the launcher; the hook activation and promotion workflow are handled by plan 04.

- Observation: the circular dataset is already laid out like the existing eval dataset, but it is not registered in the OCR config.
  Evidence: `app/tests/eval_dataset_v2/images/` and `heatmaps/` contain `page_2.jpg` through `page_6.jpg`, and `labels/PAGE-XML/` contains `page_2.xml` through `page_6.xml`.

- Observation: previous circular-looking logs under `app/tests/logs/` are generated evidence, not checked-in implementation.
  Evidence: `rg "circular_ocr" app/tests` finds latest log summaries, while `app/tests/circular_OCR_test/` contains no `.py` sources.

- Observation: `eval_dataset_v2` contains very tall, narrow circular crops under the legacy strategy, and `batch_max_pad` could shrink a single-item batch to only a few pixels wide after height normalization.
  Evidence: the first circular e2e run failed with `Given input size: (128x25x1). Calculated output size: (128x12x0). Output size is too small`. Inspecting prepared crops showed examples such as `page_3/test/word_0002.png` with original size `45x877`.

## Decision Log

- Decision: all gates should use role names `benchmark` and `proposed`, but source strategy names remain explicit strings such as `legacy_axis_bound_v1` and `local_tangent_band_v1`.
  Rationale: role names make reports easy to read, while strategy names make artifacts reproducible and allow future promotion without rewriting historical results.
  Date/Author: 2026-05-09 / Codex

- Decision: first two gates may allow a small configured regression, while the circular gate must require strict improvement on the primary OCR curve metric.
  Rationale: the research source says slightly inferior is acceptable for horizontal `eval_dataset` gates, but not for the circular-layout gate.
  Date/Author: 2026-05-09 / Codex

- Decision: implement the circular gate from source in the normal `app/tests` modules, not under `app/tests/circular_OCR_test/`.
  Rationale: that directory currently contains only cache files and no importable source. New checked-in tests should follow the existing top-level `app/tests/test_*.py` pattern.
  Date/Author: 2026-05-09 / Codex

- Decision: default `PRECOMMIT_PROPOSED_LINE_STRATEGY` to `legacy_axis_bound_v1` for this plan, while keeping it as a single environment-configurable value.
  Rationale: plan 03 has not added `local_tangent_band_v1` yet. Running both roles with the benchmark strategy verifies the ablation plumbing end to end without silently inventing a placeholder geometry algorithm. Setting `PRECOMMIT_PROPOSED_LINE_STRATEGY=local_tangent_band_v1` before plan 03 will fail clearly with the strategy registry's unknown-strategy error.
  Date/Author: 2026-05-10 / Codex

- Decision: when a gate is marked `strict_primary_improvement_required` but benchmark and proposed are configured to the same strategy name, treat equality as acceptable compatibility-mode plumbing.
  Rationale: strict improvement cannot be demonstrated until the proposed circular strategy exists. The comparison remains strict for distinct strategy names, and the config still records that the circular gate is intended to require strict improvement after plan 03.
  Date/Author: 2026-05-10 / Codex

- Decision: set a minimum padded OCR width of 32 pixels for `batch_max_pad`.
  Rationale: ratio-preserving padding can otherwise produce a tensor too narrow for the CTC OCR CNN on tall circular crops. The minimum preserves content aspect ratio while adding right padding, matching the existing padding semantics.
  Date/Author: 2026-05-10 / Codex

## Outcomes & Retrospective

Implemented on 2026-05-10. The launcher now runs three phases named `Full Pipeline Strategy Ablation Gate`, `Recognition Fine-Tune Strategy Ablation Gate`, and `Circular Recognition Fine-Tune Strategy Ablation Gate`. The latest artifacts are `app/tests/logs/pipeline_ablation_latest.{md,json}`, `app/tests/logs/recognition_finetune_ablation_latest.{md,json,txt}`, and `app/tests/logs/circular_ocr_ablation_latest.{md,json,txt}`. For this plumbing milestone both benchmark and proposed are configured to `legacy_axis_bound_v1`; the environment variable `PRECOMMIT_PROPOSED_LINE_STRATEGY` is the single switch for plan 03 to point proposed at `local_tangent_band_v1`.

The first completed benchmark/proposed results from the final launcher run were:

- Pipeline `eval_dataset`: benchmark `page_cer=0.3309031044214487`, proposed `page_cer=0.3309031044214487`, comparison passed with `max_allowed_regression_abs=0.01`.
- Recognition `eval_dataset`: benchmark `curve_metric_value=0.23554526422635289`, proposed `curve_metric_value=0.2355072004060141`, comparison passed with `max_allowed_regression_abs=0.005`.
- Circular recognition `eval_dataset_v2`: benchmark `curve_metric_value=0.9447010869565217`, proposed `curve_metric_value=0.9447010869565217`, comparison passed in same-strategy compatibility mode.

## Context and Orientation

The current pre-commit evaluation has two active Python phases:

`scripts/run_precommit_eval.py` defines `PIPELINE_PHASE`, which runs:

    python -m unittest discover -s tests -p "test_ci_e2e.py" -v

from the `app/` directory. This test lives at `app/tests/test_ci_e2e.py`. It uploads `app/tests/eval_dataset/images/`, runs CRAFT plus GNN through the Flask app, saves PAGE-XML, runs local OCR, evaluates against `app/tests/eval_dataset/labels/PAGE-XML/`, and writes `app/tests/logs/ci_eval_results_latest.txt` and `.json`.

The launcher also defines `RECOGNITION_PHASE`, which runs:

    python -m unittest tests.test_recognition_finetuning_precommit_e2e -v

from the `app/` directory. This test calls `run_recognition_precommit_gate(...)` in `app/tests/recognition_finetuning_experiment.py`. That gate prepares OCR line images from PAGE `Baseline` plus heatmap through `prepare_page_datasets(...)`, fine-tunes the OCR model with the canonical hybrid recipe, evaluates held-out pages, and writes `app/tests/logs/recognition_finetune_precommit_latest.*`.

The current OCR recipe must remain:

    training_policy=page_plus_random_history
    history_sample_line_count=10
    width_policy=batch_max_pad
    oversampling_policy=none
    augmentation_policy=none
    lr_scheduler=none
    optimizer=adadelta
    lr=0.2
    num_iter=60

The current OCR recipe is represented by `app/recognition/active_learning_recipe.py` and pulled into test config through `app/tests/precommit_gate_config.py` and `app/tests/recognition_finetuning_config.py`.

The new circular dataset config is:

    dataset name: eval_dataset_v2
    root: app/tests/eval_dataset_v2
    images: app/tests/eval_dataset_v2/images
    heatmaps: app/tests/eval_dataset_v2/heatmaps
    PAGE-XML: app/tests/eval_dataset_v2/labels/PAGE-XML
    ordered pages: page_2, page_3, page_4, page_5, page_6
    fine-tune pages: page_2, page_3, page_4
    evaluation pages: page_5, page_6

An ablation gate is a test runner that executes the same dataset, same OCR recipe, same evaluation metrics, and same artifact writing twice: once with the benchmark strategy and once with the proposed strategy. Only the strategy name and strategy config should differ.

## Plan of Work

Add a strategy ablation config layer in `app/tests/precommit_gate_config.py`. Keep existing dataset dataclasses but add a shared structure such as:

    @dataclass(frozen=True)
    class StrategyRoleConfig:
        role: str
        strategy_name: str
        strategy_config: dict = field(default_factory=dict)

    @dataclass(frozen=True)
    class StrategyAblationConfig:
        benchmark: StrategyRoleConfig
        proposed: StrategyRoleConfig
        max_allowed_regression_abs: float
        strict_primary_improvement_required: bool

The initial benchmark role must be `legacy_axis_bound_v1`. The initial proposed role should be read from a single config value, defaulting to `local_tangent_band_v1` after plan 03. While this plan is implemented before plan 03, allow tests to override the proposed strategy to `legacy_axis_bound_v1` so plumbing can be verified without a missing strategy.

Extend `PipelinePrecommitDatasetConfig` with strategy ablation settings and artifact basenames. The full-pipeline gate should write:

    app/tests/logs/pipeline_ablation_latest.json
    app/tests/logs/pipeline_ablation_latest.md
    app/tests/logs/<timestamp>_pipeline_<role>_<dataset>/

Rework `app/tests/test_ci_e2e.py` or extract its core runner into a helper module so the full-pipeline gate can call the same function for both roles. A good split is:

    app/tests/pipeline_ablation_experiment.py

with a function:

    run_pipeline_strategy_ablation_gate(dataset_name: str = "eval_dataset") -> dict

The returned dict should contain `study_mode="pipeline_strategy_ablation_gate"`, a `dataset_name`, a `strategy_results` mapping keyed by `benchmark` and `proposed`, and a `comparison` object that says whether proposed passed.

The full-pipeline comparison should keep the existing absolute thresholds from `PipelinePrecommitDatasetConfig` and add a relative strategy comparison. Use these metrics from `app/tests/evaluate.py`: `page_cer`, `line_cer_50`, `line_cer_75`, and `line_cer_range`. For `eval_dataset`, proposed passes if it satisfies the existing absolute thresholds and each primary metric is no worse than benchmark by more than `max_allowed_regression_abs`. Start with `max_allowed_regression_abs=0.01` for CER-like metrics unless the first implementation records a better justified value in `Decision Log`.

Extend `RecognitionPrecommitDatasetConfig` or add a sibling config for OCR ablations. The OCR fine-tuning gate on `eval_dataset` should call the same helper for each role:

    run_recognition_strategy_ablation_gate(dataset_name: str = "eval_dataset") -> dict

It should call the existing `_prepare_study_inputs(...)` and `_run_single_policy_run(...)` only through a strategy-aware dataset config. That config must pass the role's strategy name and config into `prepare_page_datasets(...)`. Do not duplicate OCR training code for benchmark and proposed.

The OCR `eval_dataset` comparison should keep existing blocking thresholds from `app/tests/precommit_gate_config.py` and also compare proposed against benchmark. Proposed passes if:

- proposed `curve_metric_value <= benchmark curve_metric_value + max_allowed_regression_abs`
- proposed `final_page_cer <= benchmark final_page_cer + max_allowed_regression_abs`
- proposed `first_step_gain >= benchmark first_step_gain - max_allowed_regression_abs`

Start with `max_allowed_regression_abs=0.005` for OCR curve metrics because the existing regression guard also uses `0.005`.

Register `eval_dataset_v2` in `app/tests/recognition_finetuning_config.py`:

    DATASET_CONFIGS["eval_dataset_v2"] = RecognitionEvalDatasetConfig(
        name="eval_dataset_v2",
        images_dir=TESTS_ROOT / "eval_dataset_v2" / "images",
        pagexml_dir=TESTS_ROOT / "eval_dataset_v2" / "labels" / "PAGE-XML",
        heatmaps_dir=TESTS_ROOT / "eval_dataset_v2" / "heatmaps",
        fine_tune_page_count=3,
        eval_page_start_index=3,
        eval_page_end_index=5,
    )

Add a circular OCR gate config in `app/tests/precommit_gate_config.py`. It should use the same OCR recipe as `eval_dataset`. Its primary comparison is stricter: proposed must have a lower `curve_metric_value` than benchmark. If adding secondary checks, they must not let a worse primary metric pass. The test name required by this series is:

    app/tests/test_circular_recognition_finetuning_precommit_e2e.py

It should call:

    run_circular_recognition_strategy_ablation_gate(dataset_name: str = "eval_dataset_v2") -> dict

and assert `study_mode="circular_recognition_strategy_ablation_gate"`, both roles ran, the dataset pages are `page_2` through `page_6`, and proposed passed the strict comparison.

Update `scripts/run_precommit_eval.py` so `PHASES` contains three user-facing phases:

1. `Full Pipeline Strategy Ablation Gate`
2. `Recognition Fine-Tune Strategy Ablation Gate`
3. `Circular Recognition Fine-Tune Strategy Ablation Gate`

Keep skip environment variables specific and documented:

    SKIP_PIPELINE_EVAL_HOOK=1
    SKIP_RECOGNITION_FT_HOOK=1
    SKIP_CIRCULAR_RECOGNITION_FT_HOOK=1
    SKIP_EVAL_HOOK=1

The launcher should continue to prefer the `gnn_layout` interpreter and should still write artifacts under `app/tests/logs/`.

## Concrete Steps

Work from the repository root:

    cd c:\Users\intro\OneDrive\Documents\MEGA\CAI-FLAME\gnn-synthetic-layout-historical

Implement the shared config additions in:

    app/tests/precommit_gate_config.py
    app/tests/recognition_finetuning_config.py

Extract reusable full-pipeline logic out of `app/tests/test_ci_e2e.py` into:

    app/tests/pipeline_ablation_experiment.py

Keep `app/tests/test_ci_e2e.py` as a unittest entry point. It may become a small test that calls `run_pipeline_strategy_ablation_gate("eval_dataset")` and asserts the returned result passed.

Update OCR experiment code in:

    app/tests/recognition_finetuning_experiment.py

Add role-aware wrappers without changing the lower-level OCR fine-tuning behavior. A single-policy role run should still write `curve_metrics.json`, `per_page.csv`, `per_line.csv`, `selector_metrics.json`, `fine_tune_metadata.json`, and plots in the same artifact format.

Add tests:

    app/tests/test_strategy_ablation_config_unit.py
    app/tests/test_recognition_finetuning_precommit_unit.py
    app/tests/test_circular_recognition_finetuning_precommit_e2e.py

Update:

    scripts/run_precommit_eval.py

The launcher should print artifact paths for the latest full-pipeline ablation, OCR ablation, and circular OCR ablation summaries.

## Validation and Acceptance

Use the `gnn_layout` conda environment. From the repository root, run:

    $env:CONDA_NO_PLUGINS='true'; conda run -n gnn_layout python -m unittest app.tests.test_strategy_ablation_config_unit -v

Expected result: unit tests confirm `eval_dataset` and `eval_dataset_v2` configs, role names, strategy names, ordered pages, and comparison rules.

Run the required focused OCR unit tests:

    $env:CONDA_NO_PLUGINS='true'; conda run -n gnn_layout python -m unittest app.tests.test_recognition_finetuning_precommit_unit -v

Expected result: existing OCR recipe assertions still pass, with added assertions for strategy-aware config.

Run the pretrained full-pipeline gate:

    $env:CONDA_NO_PLUGINS='true'; conda run -n gnn_layout python -m unittest discover -s app/tests -p "test_ci_e2e.py" -v

Expected result: both benchmark and proposed role results are present in `app/tests/logs/pipeline_ablation_latest.json`, and the test fails with a useful comparison message if proposed violates configured tolerances.

Run the OCR fine-tuning gate:

    $env:CONDA_NO_PLUGINS='true'; conda run -n gnn_layout python -m unittest app.tests.test_recognition_finetuning_precommit_e2e -v

Expected result: both roles run through the same OCR policy code on `eval_dataset`, latest artifacts are written, and proposed is allowed only the configured small tolerance.

Run the circular OCR fine-tuning gate:

    $env:CONDA_NO_PLUGINS='true'; conda run -n gnn_layout python -m unittest app.tests.test_circular_recognition_finetuning_precommit_e2e -v

Expected result: the gate uses `eval_dataset_v2`, fine-tunes on `page_2`, `page_3`, and `page_4`, evaluates on `page_5` and `page_6`, writes `app/tests/logs/circular_ocr_ablation_latest.*`, and requires proposed to be strictly better on the primary curve metric.

Run the complete launcher:

    $env:CONDA_NO_PLUGINS='true'; conda run -n gnn_layout python scripts/run_precommit_eval.py

Expected result: all three phases pass and the launcher prints artifact paths for each phase. If a phase fails, it should print the failing role, metric, observed values, thresholds, and latest artifact paths.

For long OCR runs on Windows, use the direct interpreter fallback if needed:

    C:\Users\intro\miniconda3\envs\gnn_layout\python.exe -m unittest app.tests.test_recognition_finetuning_precommit_e2e -v
    C:\Users\intro\miniconda3\envs\gnn_layout\python.exe -m unittest app.tests.test_circular_recognition_finetuning_precommit_e2e -v
    C:\Users\intro\miniconda3\envs\gnn_layout\python.exe scripts/run_precommit_eval.py

The acceptance criterion is not just passing tests. Inspect the latest JSON artifacts and verify that benchmark and proposed were both run, that they name their strategies, and that no role uses PAGE `Coords` as a fallback when strategy-based baseline geometry is required.

## Idempotence and Recovery

All gates should be rerunnable. They may overwrite `*_latest.*` files under `app/tests/logs/` and create new timestamped run directories. Timestamped run directories are evidence only and should not be required for a fresh checkout. The launcher prunes passing benchmark/proposed role-run directories by default after latest aliases are copied, while copying passing OCR role plots into the retained summary directory under `plots/`; failed phases retain their full role runs for debugging. Set `CLEAN_UP=0` when a passing launcher run needs full role artifacts.

Do not write to `C:\temp` or outside this repository. If a test needs temporary files, use `app/tests/_tmp_*` or a timestamped directory under `app/tests/logs/`.

If a proposed strategy is not implemented yet, plumbing unit tests may explicitly configure proposed to `legacy_axis_bound_v1`. The real e2e gates should fail clearly when configured to a missing proposed strategy rather than silently falling back to benchmark. Record any temporary bypass in `Decision Log`.

If OCR runs fail due to the Windows `conda run` console encoding issue after artifacts are written, treat saved artifact folders as evidence and rerun with the direct interpreter command shown above.

## Artifacts and Notes

Expected latest artifacts after this plan:

    app/tests/logs/pipeline_ablation_latest.md
    app/tests/logs/pipeline_ablation_latest.json
    app/tests/logs/recognition_finetune_ablation_latest.md
    app/tests/logs/recognition_finetune_ablation_latest.json
    app/tests/logs/circular_ocr_ablation_latest.md
    app/tests/logs/circular_ocr_ablation_latest.json

Generated logs currently present under `app/tests/logs/` are not source of truth. They may guide manual debugging, but checked-in config and source code must define gate membership and thresholds.

Validation completed on 2026-05-10:

    $env:CONDA_NO_PLUGINS='true'; conda run -n gnn_layout python -m unittest app.tests.test_strategy_ablation_config_unit -v
    Result: OK, 3 tests passed.

    $env:CONDA_NO_PLUGINS='true'; conda run -n gnn_layout python -m unittest app.tests.test_recognition_finetuning_precommit_unit -v
    Result: OK, 7 tests passed.

    $env:CONDA_NO_PLUGINS='true'; conda run -n gnn_layout python -m unittest app.tests.test_recognition_active_learning_unit -v
    Result: OK, 9 tests passed. This includes the new minimum-width `batch_max_pad` assertion.

    $env:CONDA_NO_PLUGINS='true'; conda run -n gnn_layout python -m unittest discover -s app/tests -p "test_ci_e2e.py" -v
    Result: OK, 1 full-pipeline ablation e2e test passed.

    $env:CONDA_NO_PLUGINS='true'; conda run -n gnn_layout python -m unittest app.tests.test_recognition_finetuning_precommit_e2e -v
    Result: OK, 1 OCR fine-tuning ablation e2e test passed.

    $env:CONDA_NO_PLUGINS='true'; conda run -n gnn_layout python -m unittest app.tests.test_circular_recognition_finetuning_precommit_e2e -v
    First result: failed with a too-narrow CNN input from circular crops. After adding the minimum padded OCR width, rerun result: OK, 1 circular OCR ablation e2e test passed.

    $env:CONDA_NO_PLUGINS='true'; conda run -n gnn_layout python scripts/run_precommit_eval.py
    Result: OK. The launcher ran all three ablation phases and printed the latest artifact paths.

Plan update note, 2026-05-10: Recorded the completed implementation, the temporary same-strategy proposed default, the circular narrow-crop fix, and validation evidence so the plan reflects the checked-in behavior.

## Interfaces and Dependencies

Required new or updated public functions:

    run_pipeline_strategy_ablation_gate(dataset_name: str = "eval_dataset") -> dict
    run_recognition_strategy_ablation_gate(dataset_name: str = "eval_dataset") -> dict
    run_circular_recognition_strategy_ablation_gate(dataset_name: str = "eval_dataset_v2") -> dict

Required strategy role fields in each result:

    role
    strategy_name
    strategy_config
    run_dir
    status
    metrics
    summary_path
    metrics_path

Required comparison fields:

    benchmark_role
    proposed_role
    primary_metric_name
    benchmark_value
    proposed_value
    operator
    allowed_regression_abs
    passed
    failure_message

The gates should continue to use existing dependencies and evaluation helpers: `app/tests/evaluate.py`, `app/recognition/active_learning.py`, and `app/tests/recognition_finetuning_experiment.py`. Do not introduce a new test runner framework.

## Change Note

Initial split plan created on 2026-05-09. The reason for this split is to isolate evaluation overhaul from both the strategy extraction and the new circular geometry algorithm, making failures easier to attribute.
