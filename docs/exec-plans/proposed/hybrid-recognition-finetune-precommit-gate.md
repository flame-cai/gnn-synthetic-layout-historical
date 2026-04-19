# Add A Hybrid OCR Fine-Tuning Pre-Commit Gate

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This document is maintained in accordance with `PLANS.md` from the repository root.

## Purpose / Big Picture

After this change, every normal commit in this repository will be screened by two GUI-free gates instead of one. The existing gate will still prove that the full automatic pipeline, using pretrained CRAFT, pretrained GNN, and the pretrained recognition checkpoint, can process the evaluation manuscript end to end and stay under the current PAGE-level CER thresholds. The new gate will prove something different: that the recognition fine-tuning stack, using the currently best hybrid continuation recipe, can still improve OCR on fixed held-out pages when it is fed perfect line crops and ground-truth text pairs.

The user-visible proof will be that one pre-commit hook invocation prints two named phases, writes two families of artifacts under `app/tests/logs/`, and exits nonzero only if either the existing full-pipeline gate fails or the new recognition fine-tuning gate falls outside its configured acceptance thresholds. The new gate is intentionally not the real human-in-the-loop pipeline. It is a surrogate safety check for the OCR fine-tuning code and shared scaffolding while the repo still lacks live GUI-triggered training for CRAFT, GNN, and OCR together.

## Progress

- [x] (2026-04-19 15:05 IST) Audited the current hook launcher in `.githooks/pre-commit`, the current launcher script in `scripts/run_precommit_eval.py`, and the existing end-to-end gate in `app/tests/test_ci_e2e.py`.
- [x] (2026-04-19 15:10 IST) Confirmed that the currently best hybrid OCR recipe is already implemented in the offline harness through `training_policy=page_plus_random_history` and `run_page_plus_random_history_experiment(...)`.
- [x] (2026-04-19 15:12 IST) Confirmed that the current hook is effectively disabled because `.githooks/pre-commit` starts with `exit 0`.
- [x] (2026-04-19 16:01 IST) Added `app/tests/precommit_gate_config.py` as the shared pre-commit dataset and threshold registry, and refactored `app/tests/test_ci_e2e.py` to read its dataset settings from that registry instead of hardcoding `eval_dataset`.
- [x] (2026-04-19 16:01 IST) Added `get_precommit_hybrid_recognition_gate_config(...)` plus `run_recognition_precommit_gate(...)`, wiring the exact hybrid recipe into a single-policy surrogate OCR gate with dedicated latest aliases.
- [x] (2026-04-19 16:01 IST) Added `app/tests/test_recognition_finetuning_precommit_unit.py` and `app/tests/test_recognition_finetuning_precommit_e2e.py` to lock the recipe, thresholds, and warning-only regression semantics.
- [x] (2026-04-19 16:01 IST) Extended `scripts/run_precommit_eval.py` to run the full-pipeline gate first and the OCR fine-tuning gate second, added `SKIP_PIPELINE_EVAL_HOOK` and `SKIP_RECOGNITION_FT_HOOK`, and reactivated `.githooks/pre-commit` by removing the unconditional `exit 0`.
- [x] (2026-04-19 16:01 IST) Updated `README.md`, `EVAL.md`, `VISION.md`, and `AGENTS.md` so the docs now describe the two-phase hook and the surrogate scope of the OCR gate.
- [x] (2026-04-19 16:01 IST) Verified the implementation with the new unit test, the refactored full-pipeline gate, the new OCR pre-commit e2e, and the launcher entrypoint.

## Surprises & Discoveries

- Observation: the repository already has almost all of the mechanics needed for the new gate.
  Evidence: `app/tests/recognition_finetuning_config.py`, `app/tests/recognition_finetuning_experiment.py`, and `app/recognition/active_learning.py` already implement the hybrid continuation regime, the OCR dataset preparation flow from PAGE-XML ground truth, and the metric reporting we need.

- Observation: the current hook is not merely minimal; it is currently bypassed unconditionally.
  Evidence: `.githooks/pre-commit` begins with `exit 0`, so the later Python-launch logic never runs.

- Observation: the current hook launcher knows how to find the `gnn_layout` interpreter, but it only knows how to run one test command.
  Evidence: `scripts/run_precommit_eval.py` resolves one Python command and then runs only `python -m unittest discover -s tests -p "test_ci_e2e.py" -v`.

- Observation: the existing OCR hybrid entrypoint is a research comparison, not a commit gate.
  Evidence: `run_page_plus_random_history_experiment(...)` runs two policies, ranks winners, writes study-style summaries, and treats the regression guard as a hard pass/fail criterion for each policy candidate.

- Observation: the current full-pipeline gate is hardcoded around one dataset and one manuscript name.
  Evidence: `app/tests/test_ci_e2e.py` hardcodes `eval_dataset`, `ci_eval_dataset`, the dataset directories, and the expected page count in the test body and class constants.

- Observation: the current best hybrid OCR artifact gives us a concrete source of truth for initial thresholds.
  Evidence: `app/tests/logs/20260419_132843_ocrft_pagehist_eval_dataset/metrics.json` records `curve_metric_value=0.22191428022294832`, `final_page_cer=0.14735729386892177`, `first_step_gain=0.05137420718816066`, and `max_regression=0.001691331923890066` for `wb_on_an_hist10_sn_optd_lr200000u`.

- Observation: the easiest safe implementation path was to extend the existing single-policy runner rather than clone it.
  Evidence: `app/tests/recognition_finetuning_experiment.py` now lets `_run_single_policy_run(...)` switch between hard-fail and warning-only regression-guard modes, so the research studies keep strict behavior while the surrogate pre-commit gate can finish all steps and still report `max_regression`.

- Observation: a top-level dataset-keyed JSON payload was easy to add even for the first single-dataset rollout.
  Evidence: `app/tests/logs/recognition_finetune_precommit_latest.json` now stores results under `dataset_results["eval_dataset"]`, which avoids baking a permanent single-dataset assumption into the new gate aliases.

## Decision Log

- Decision: keep the old pretrained end-to-end gate and add the new OCR fine-tuning gate alongside it rather than replacing the old gate.
  Rationale: the two checks validate different failure surfaces. The old gate proves that CRAFT, GNN, PAGE-XML generation, and OCR inference still work together with pretrained checkpoints. The new gate proves that the OCR fine-tuning harness and its shared scaffolding still behave well under the best known hybrid continuation recipe.
  Date/Author: 2026-04-19 / Codex

- Decision: build the new commit gate around one explicit hybrid OCR recipe instead of rerunning a multi-policy study.
  Rationale: a commit gate should be deterministic, faster than a research sweep, and easy to interpret. The purpose is not to rediscover the best policy on every commit; it is to verify that the current best-known recipe still works.
  Date/Author: 2026-04-19 / Codex

- Decision: continue to compute the regression guard with `regression_guard_abs=0.005`, but make regression-guard failure warning-only for the new pre-commit gate.
  Rationale: the user explicitly asked that regression guard failure should not by itself fail the new hook. The gate should still record `max_regression` and `regression_guard_passed` for human diagnosis, but the exit code should be driven by `curve_metric_value`, `final_page_cer`, and `first_step_gain`.
  Date/Author: 2026-04-19 / Codex

- Decision: move gate thresholds and dataset membership into a shared checked-in registry instead of scattering them across unittest constants.
  Rationale: both current pre-commit checks only use `eval_dataset` today, but the user wants a design that can grow to multiple datasets later. A registry makes that growth additive and reviewable.
  Date/Author: 2026-04-19 / Codex

- Decision: document the new gate explicitly as a surrogate for the OCR fine-tuning subsystem, not as proof that the real human-in-the-loop pipeline is already solved.
  Rationale: the current gate uses ground-truth segmented lines and ground-truth text pairs, while the future product vision involves upstream CRAFT, GNN, scaffolding, save-cycle events, and human corrections driving fine-tuning for all three model families.
  Date/Author: 2026-04-19 / Codex

- Decision: reuse `_run_single_policy_run(...)` with a `regression_guard_mode` switch instead of creating a second mostly-duplicated OCR runner implementation.
  Rationale: the pre-commit gate still needs the same artifacts, step records, and metric computation as the research harness. A mode switch preserves one implementation path while keeping the research runs strict and the surrogate gate warning-only on regression-guard failures.
  Date/Author: 2026-04-19 / Codex

## Outcomes & Retrospective

The implementation is complete. The repository now has a two-phase pre-commit path backed by one shared checked-in registry: the original pretrained full-pipeline gate and a new surrogate OCR fine-tuning gate for the best-known hybrid continuation recipe. The conceptual constraint remained the same all the way through implementation: the new OCR gate is documented as a subsystem guard, not as proof that the future full human-in-the-loop training loop already exists.

The main practical outcome is that the repo now catches regressions in two different places before commit. The full-pipeline gate still protects CRAFT plus GNN plus PAGE-XML plus OCR inference together. The new OCR gate protects the fine-tuning stack and shared OCR scaffolding using perfect segmentation inputs, dedicated thresholds, and warning-only regression-guard reporting.

Verification completed on 2026-04-19 with these commands:

- `C:\Users\intro\miniconda3\envs\gnn_layout\python.exe -m unittest app.tests.test_recognition_finetuning_precommit_unit -v`
- `C:\Users\intro\miniconda3\envs\gnn_layout\python.exe -m unittest discover -s app/tests -p "test_ci_e2e.py" -v`
- `C:\Users\intro\miniconda3\envs\gnn_layout\python.exe -m unittest app.tests.test_recognition_finetuning_precommit_e2e -v`
- `C:\Users\intro\miniconda3\envs\gnn_layout\python.exe scripts/run_precommit_eval.py`

The calibrated initial OCR thresholds remained unchanged from the plan (`0.26`, `0.18`, `0.04`) because the verified run stayed comfortably inside them.

## Context and Orientation

The current repository has one GUI-free integration gate wired for pre-commit use and one family of slower OCR research studies.

The existing pre-commit gate lives in these files:

- `.githooks/pre-commit`
- `scripts/install_git_hooks.py`
- `scripts/run_precommit_eval.py`
- `app/tests/test_ci_e2e.py`

That gate uploads the evaluation manuscript into the Flask app, runs CRAFT and GNN inference, saves PAGE-XML, runs OCR recognition, evaluates PAGE-level and line-level CER against the ground truth in `app/tests/eval_dataset/labels/PAGE-XML/`, and fails if the aggregate thresholds are exceeded. It is a full automatic pipeline test and it intentionally does not fine-tune any model.

The OCR fine-tuning research harness lives in these files:

- `app/tests/recognition_finetuning_config.py`
- `app/tests/recognition_finetuning_experiment.py`
- `app/recognition/active_learning.py`
- `app/tests/test_recognition_finetuning_e2e.py`
- `app/tests/test_recognition_finetuning_page_plus_history_unit.py`

That harness prepares perfect line crops from PAGE-XML ground truth, fine-tunes OCR checkpoints sequentially, evaluates later pages, and writes artifact folders under `app/tests/logs/`. The hybrid continuation recipe we want for the new gate is:

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
- `background_plus_rotation_variant_count=10`
- `shuffle_train_each_epoch=True`

The best completed artifact for that recipe is:

- `app/tests/logs/20260419_132843_ocrft_pagehist_eval_dataset/`

Its winning policy metrics are:

- `curve_metric_value=0.22191428022294832`
- `final_page_cer=0.14735729386892177`
- `first_step_gain=0.05137420718816066`
- `max_regression=0.001691331923890066`

The new pre-commit gate must reuse this hybrid recipe, but it must not reuse the whole two-policy study shape. A commit gate should run one explicit policy, compare the resulting metrics to checked-in thresholds, and produce one clear pass or fail result per dataset.

This new gate is not the future end state of the product. In the real long-term workflow, a human will upload a manuscript, CRAFT will propose character regions, a GNN will propose text-line structure, the user will correct nodes and edges, the app will save PAGE-XML and OCR edits, and all three model families may eventually be fine-tuned from those corrections. The gate in this plan is narrower: it only verifies the current OCR fine-tuning subsystem under perfect segmentation inputs so that OCR code and shared scaffolding regressions are caught early while the broader active-learning system is still under construction.

## Plan of Work

Start by introducing a shared checked-in pre-commit registry module, for example `app/tests/precommit_gate_config.py`. This file should define small, explicit dataclasses for the two gate families rather than leaving their assumptions spread across many test files. One dataclass should describe the existing full-pipeline dataset gate: dataset name, manuscript name, images directory, PAGE-XML ground truth directory, expected page count, and the existing PAGE-CER and line-CER thresholds. A second dataclass should describe the OCR fine-tuning gate: dataset name, the OCR dataset config name, the exact hybrid recipe, the three blocking thresholds `max_curve_metric_value`, `max_final_page_cer`, and `min_first_step_gain`, plus the non-blocking regression guard settings. The registry should initially contain only `eval_dataset`, but the structure must make adding `foo_dataset` later a one-file additive change.

Refactor `app/tests/test_ci_e2e.py` to read from the new pre-commit dataset registry instead of hardcoding `eval_dataset` constants throughout the test. Keep the observed behavior the same for the one current dataset. The point of this refactor is not to change the current gate's thresholds or flow. The point is to remove the future assumption that the repo will forever have only one pre-commit dataset. Use `subTest` or a small loop so one test module can later screen multiple configured datasets without duplicating the whole test file.

Next, add a dedicated pre-commit OCR gate config helper in `app/tests/recognition_finetuning_config.py`. Do not make the broader research helpers the source of truth for the commit gate. Research helpers should stay free to explore, while the commit gate should be explicit and stable. Add a helper with a stable signature such as `get_precommit_hybrid_recognition_gate_config(name="eval_dataset")` that returns exactly one `RecognitionEvalDatasetConfig` for the best-known hybrid recipe above. That helper should set the policy fields directly rather than relying on whichever policy won the last research sweep. The exact recipe must be visible in code review.

Then extend `app/tests/recognition_finetuning_experiment.py` with a new single-policy runner for the commit gate. A good name is `run_recognition_precommit_gate(dataset_name="eval_dataset")`. This function should prepare the dataset once, run only the explicit hybrid recipe, and return a compact payload that contains:

- the dataset name
- the policy descriptor
- the resulting `curve_metric_value`
- the resulting `final_page_cer`
- the resulting `first_step_gain`
- `regression_guard_passed`
- `max_regression`
- the artifact paths
- a derived boolean for each blocking threshold

The runner should still compute the regression guard from the same sequential steps and the same `regression_guard_abs=0.005`, but it must treat that guard as diagnostic only. If `regression_guard_passed` is false and the three blocking metrics still satisfy their thresholds, the runner should mark the dataset result as passed with a warning. The summary output should say this plainly rather than hiding it in JSON.

Seed the first threshold set from the current best hybrid artifact and keep those thresholds in the new pre-commit registry, not inside the unittest body. For `eval_dataset`, the initial thresholds should be deliberately looser than the current best artifact so the hook catches meaningful regressions without turning routine commits into noise. The first implementation should start with:

- `max_curve_metric_value = 0.26`
- `max_final_page_cer = 0.18`
- `min_first_step_gain = 0.04`

These values are intentionally above or below the current best run with clear headroom. They are not meant to define the long-term target. They are meant to create a stable first commit gate that is strict enough to catch breakage and loose enough to tolerate normal machine-to-machine variation. If a clean implementation pass shows that these thresholds are too tight or too loose on the actual hook environment, the final implementation should adjust the numbers in the registry and record that calibration decision in this ExecPlan and `EVAL.md`.

Add two new tests for the new gate. First add a fast config-level test, for example `app/tests/test_recognition_finetuning_precommit_unit.py`, that verifies the gate helper returns the exact hybrid recipe, the warning-only regression semantics are preserved, and the threshold registry is wired correctly. Then add the real slow gate test, for example `app/tests/test_recognition_finetuning_precommit_e2e.py`, that calls `run_recognition_precommit_gate(...)`, asserts that all configured OCR pre-commit datasets pass the three blocking thresholds, asserts that the summary payload still includes `regression_guard_passed` and `max_regression`, and asserts that the report files are written to stable latest aliases. The slow test should not rely on the GUI and should not run a multi-policy sweep.

Choose stable artifact names for the new gate that do not collide with the research-study aliases. A clear initial choice is:

- `app/tests/logs/recognition_finetune_precommit_latest.md`
- `app/tests/logs/recognition_finetune_precommit_latest.json`
- `app/tests/logs/recognition_finetune_precommit_latest.txt`

If the implementation groups multiple OCR pre-commit datasets into one invocation later, the latest JSON should include a top-level `dataset_results` map keyed by dataset name rather than assuming a single dataset forever.

After the new test exists, extend `scripts/run_precommit_eval.py` so it runs two phases in order using the same interpreter-resolution logic it already has. The first phase should remain the current full-pipeline test. The second phase should run the new OCR fine-tuning pre-commit test. The launcher should print clear phase headers, stop on the first hard failure, and print the artifact summary paths for whichever phase failed. Preserve `GNN_LAYOUT_PYTHON` support. Add phase-specific skip variables while keeping `SKIP_EVAL_HOOK=1` as the umbrella override. A good first set is:

- `SKIP_EVAL_HOOK=1` to skip both phases
- `SKIP_PIPELINE_EVAL_HOOK=1` to skip only the old full-pipeline phase
- `SKIP_RECOGNITION_FT_HOOK=1` to skip only the new OCR fine-tuning phase

Then update `.githooks/pre-commit` to remove the unconditional `exit 0` and rely on the launcher again. Keep the existing "find Python, run the launcher, fail with a readable message if no interpreter is available" behavior. The hook should remain a thin shell wrapper and should not learn evaluation logic itself.

Finally, update the repository documentation. `README.md` should describe that pre-commit now runs two GUI-free evaluation phases and explain where their latest summaries are written. `EVAL.md` should explicitly distinguish the two pre-commit gates from the broader offline research studies and from the future true human-in-the-loop pipeline. It should state plainly that the new OCR pre-commit gate uses perfect line segmentation from PAGE-XML ground truth and therefore validates the OCR fine-tuning subsystem, not the full interactive active-learning product. `VISION.md` should explain why this is still valuable: it gives the repo a stable surrogate guard while the longer path toward fine-tuning CRAFT, GNN, and OCR from real user corrections is still under construction. `AGENTS.md` should be updated so future agents know the new hook exists, where its source of truth lives, and how it differs from the broader research harness.

## Concrete Steps

From the repository root, confirm the current hook state and the current launch path before editing anything:

    Get-Content .githooks\pre-commit
    Get-Content scripts\run_precommit_eval.py
    Get-Content app\tests\test_ci_e2e.py

Expected result: the shell hook currently starts with `exit 0`, the launcher only runs `test_ci_e2e.py`, and the current end-to-end test is hardcoded around `eval_dataset`.

Inspect the hybrid OCR research entrypoint and current winning artifact:

    Get-Content app\tests\recognition_finetuning_config.py
    Get-Content app\tests\recognition_finetuning_experiment.py
    Get-Content app\tests\logs\recognition_finetune_results_latest.json

Expected result: the hybrid recipe is already implemented through `page_plus_random_history`, the current entrypoint runs a research comparison rather than a single gate recipe, and the latest artifact records the best-known metric values that will seed the new threshold registry.

After implementing the new registry and the new OCR pre-commit gate, run the fast regression coverage:

    C:\Users\intro\miniconda3\envs\gnn_layout\python.exe -m unittest app.tests.test_recognition_finetuning_precommit_unit -v

Expected result: the test proves that the exact hybrid recipe is wired into the gate helper, that the blocking thresholds are loaded from the registry rather than hardcoded in the test body, and that regression-guard failure is treated as warning-only in the gate result model.

Run the existing full-pipeline gate directly to ensure the refactor preserved its behavior:

    C:\Users\intro\miniconda3\envs\gnn_layout\python.exe -m unittest discover -s app/tests -p "test_ci_e2e.py" -v

Expected result: the test still uploads the evaluation manuscript, runs CRAFT plus GNN plus OCR end to end, and writes `app/tests/logs/ci_eval_results_latest.txt` and `.json`.

Run the new slow OCR pre-commit gate directly:

    C:\Users\intro\miniconda3\envs\gnn_layout\python.exe -m unittest app.tests.test_recognition_finetuning_precommit_e2e -v

Expected result: the test runs the single explicit hybrid recipe, writes dedicated latest aliases such as `app/tests/logs/recognition_finetune_precommit_latest.json`, and passes when the three blocking metrics satisfy the configured thresholds even if the regression guard reports a warning.

After both tests pass directly, run the launcher:

    python scripts/run_precommit_eval.py

or on Windows:

    py -3 scripts/run_precommit_eval.py

Expected result: the launcher prints two named phases, runs the old gate first and the new gate second, and exits with success only if both phases pass. If one phase fails, the launcher prints which phase failed and where its latest report file lives.

Finally, install the hook into git and confirm the shell wrapper is active:

    python scripts/install_git_hooks.py
    git config --get core.hooksPath

Expected result: `core.hooksPath` prints `.githooks`, `.githooks/pre-commit` is executable, and the file no longer exits immediately at the top.

## Validation and Acceptance

Acceptance for this plan is behavioral.

First, the repository must still have the current full-pipeline GUI-free pre-commit gate. It must remain required, and it must still fail commits when the pretrained CRAFT plus pretrained GNN plus pretrained OCR path breaks or exceeds its current CER thresholds.

Second, the repository must gain a second GUI-free pre-commit gate for OCR fine-tuning that uses perfect line crops and ground-truth text pairs rather than the GUI or predicted segmentation. The point of this gate is to protect the OCR fine-tuning subsystem and shared scaffolding, not to simulate the final product experience.

Third, the new OCR gate must run exactly one explicit recipe for each configured dataset, not a policy sweep. For `eval_dataset`, that recipe must be the current best hybrid continuation recipe: `page_plus_random_history`, `history_sample_line_count=10`, `batch_max_pad`, no oversampling, no augmentation, no scheduler, Adadelta, `lr=0.2`, `num_iter=60`, `early_weighted_page_cer`, `regression_guard_abs=0.005`, `background_plus_rotation_variant_count=10`, and `shuffle_train_each_epoch=True`.

Fourth, the new OCR gate must report at least these metrics per dataset:

- `curve_metric_value`
- `final_page_cer`
- `first_step_gain`
- `regression_guard_passed`
- `max_regression`

Fifth, the new OCR gate must fail only on the three blocking thresholds `curve_metric_value`, `final_page_cer`, and `first_step_gain`. Regression guard failure must be visible in the reports and console output, but it must not by itself fail the commit.

Sixth, both the old and new gates must be driven from checked-in dataset and threshold registries rather than from hardcoded `eval_dataset` paths buried deep inside unittest bodies. The first implementation may still configure only one dataset, but adding a second one later must be an additive config change rather than a redesign.

Seventh, the docs must explain the nuance clearly. `README.md`, `EVAL.md`, `VISION.md`, and `AGENTS.md` must all make it clear that the new OCR pre-commit gate is a surrogate verifier for the current recognition fine-tuning recipe and not yet the same thing as the future CRAFT plus GNN plus OCR human-in-the-loop active-learning loop.

## Idempotence and Recovery

This plan should be implemented additively. The current pipeline gate must remain working while the new OCR gate is being added. The safest sequence is: introduce the new config and runner, add the new tests, extend the launcher, then remove the unconditional `exit 0` from `.githooks/pre-commit` only after both phases pass locally.

The hook installer `scripts/install_git_hooks.py` is already idempotent and should remain so. Running it multiple times should only keep `core.hooksPath=.githooks` configured and ensure the hook file is executable.

If the new OCR gate thresholds prove too strict or too loose on the actual gate environment, the recovery path is to rerun the gate manually, inspect the generated latest JSON, adjust the checked-in threshold registry, and record the calibration change in this ExecPlan and `EVAL.md`. Do not hide threshold changes inside test-body magic numbers.

If the hook becomes too expensive for local iteration, use the explicit skip variables documented above. Do not reintroduce an unconditional `exit 0` at the top of the hook, because that silently disables all protection for everyone.

## Artifacts and Notes

Important current source-of-truth files and artifacts for this work are:

- `.githooks/pre-commit`
- `scripts/run_precommit_eval.py`
- `scripts/install_git_hooks.py`
- `app/tests/test_ci_e2e.py`
- `app/tests/recognition_finetuning_config.py`
- `app/tests/recognition_finetuning_experiment.py`
- `app/tests/logs/ci_eval_results_latest.json`
- `app/tests/logs/recognition_finetune_results_latest.json`
- `app/tests/logs/20260419_132843_ocrft_pagehist_eval_dataset/metrics.json`

The initial OCR gate thresholds in the registry should be seeded from the current hybrid artifact but with explicit headroom. The first checked-in values should be:

- `max_curve_metric_value = 0.26`
- `max_final_page_cer = 0.18`
- `min_first_step_gain = 0.04`

The first stable artifact aliases for the new OCR gate should be distinct from the broader research-study aliases. A concise initial shape is:

- `recognition_finetune_precommit_latest.md`
- `recognition_finetune_precommit_latest.json`
- `recognition_finetune_precommit_latest.txt`

If a future implementation adds multiple OCR pre-commit datasets, the JSON should look like a dataset-keyed map rather than a single flat object. The point is to keep the first implementation simple without baking in a permanent single-dataset assumption.

## Interfaces and Dependencies

In `app/tests/precommit_gate_config.py`, define explicit dataclasses and helpers for both gate families. The exact class names may vary, but the end state must include a checked-in registry that can answer these questions:

- which datasets belong to the full-pipeline pre-commit gate
- which datasets belong to the OCR fine-tuning pre-commit gate
- what thresholds apply to each configured dataset

In `app/tests/recognition_finetuning_config.py`, expose a stable helper equivalent to:

    def get_precommit_hybrid_recognition_gate_config(name: str = "eval_dataset") -> RecognitionEvalDatasetConfig:
        ...

That helper must return the exact single hybrid recipe used by the new gate.

In `app/tests/recognition_finetuning_experiment.py`, expose a stable public entrypoint equivalent to:

    def run_recognition_precommit_gate(dataset_name: str = "eval_dataset") -> dict:
        ...

The returned payload must include at least:

- `dataset_name`
- `policy`
- `curve_metrics`
- `threshold_results`
- `warnings`
- `summary_path`
- `metrics_path`
- `fine_tune_metadata_path`

In `app/tests`, add:

- `test_recognition_finetuning_precommit_unit.py`
- `test_recognition_finetuning_precommit_e2e.py`

In `scripts/run_precommit_eval.py`, the launcher must end this work capable of running two separate test commands in sequence while reusing the same resolved interpreter path. It must keep backward-compatible support for `GNN_LAYOUT_PYTHON` and `conda run -n gnn_layout`.

In `.githooks/pre-commit`, the hook must remain a thin shell entrypoint that delegates to the Python launcher rather than embedding repo logic directly.

Revision note, 2026-04-19: this ExecPlan was created after the hybrid OCR continuation recipe stabilized enough to justify a dedicated commit gate. The planning pass also discovered that the current shell hook is effectively disabled by an unconditional `exit 0`, so reactivating the hook path is now part of the required implementation rather than an optional cleanup.

Revision note, 2026-04-19 16:01 IST: the implementation is complete. The plan was updated to reflect the shared pre-commit registry, the new surrogate OCR gate, the launcher and hook changes, the docs sync, and the local verification commands that passed.
