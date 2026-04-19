# Page-Only Sequential OCR Continuation Study for Explicit Low-LR Policies

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This document is maintained in accordance with `PLANS.md` from the repository root.

## Purpose / Big Picture

After this change, the repository supports a separate OCR experiment path that keeps the held-out pages, CER metrics, and regression guard the same as the focused cumulative verifier, but changes the sequential fine-tuning regime to page-only continuation. In plain language, step 6 no longer rebuilds a train set from pages 1 through 6. Instead, it loads the checkpoint produced after step 5 and fine-tunes that checkpoint only on page 6.

The user-visible proof is a timestamped artifact folder under `app/tests/logs/` whose per-step metadata shows that each post-baseline step trains on exactly one page and that each step's `base_checkpoint` equals the previous step's `output_checkpoint`. The initial requested study under this regime is a four-policy comparison on `eval_dataset`.

## Progress

- [x] (2026-04-19 11:xx IST) Read `PLANS.md` and the current OCR harness to confirm the required ExecPlan format and the current sequential fine-tuning behavior.
- [x] (2026-04-19 11:xx IST) Confirmed from source that the runner already had a dormant `training_policy == "page_only"` branch, but no dedicated reproducible study path or dedicated latest aliases.
- [x] (2026-04-19 12:xx IST) Added explicit page-only policy configuration helpers, a dedicated page-only experiment entrypoint, unambiguous micro-LR slugging, and additive step metadata for checkpoint carry-forward and train-dataset page counts.
- [x] (2026-04-19 12:xx IST) Added fast unit coverage in `app/tests/test_recognition_finetuning_page_only_unit.py` and slow end-to-end coverage in `app/tests/test_recognition_finetuning_page_only_e2e.py`.
- [x] (2026-04-19 12:32 IST) Ran the slow page-only study and produced `app/tests/logs/20260419_123216_ocrft_pageonly_eval_dataset/`.
- [x] (2026-04-19 13:xx IST) Updated `EVAL.md`, `VISION.md`, and this ExecPlan so the documentation matches the implemented page-only path and its saved artifacts.

## Surprises & Discoveries

- Observation: the core page-only continuation behavior already existed in the experiment runner.
  Evidence: `app/tests/recognition_finetuning_experiment.py` already switched to `training_page_ids = [train_page_id]` when `dataset_config.training_policy == "page_only"`, while still carrying `current_checkpoint` forward.

- Observation: the low-level fine-tune dataset builder did not need a new trainer.
  Evidence: `app/recognition/active_learning.py::prepare_incremental_finetune_dataset()` already obeyed the exact list of prepared pages it was given, so a one-element page list naturally produced a one-page train set.

- Observation: the original slow-test expectation that every policy would complete all 9 continuation steps was too strict.
  Evidence: the saved study `app/tests/logs/20260419_123216_ocrft_pageonly_eval_dataset/` records all four policies, but three policies failed the regression guard early and stopped with shorter per-step traces while still producing valid metrics and summaries.

- Observation: sub-`0.001` learning rates needed a dedicated slug helper for follow-up studies.
  Evidence: earlier ad hoc runs at `0.00005` and `0.00001` both collapsed to `lr0000` under the older helper, while the page-only study now distinguishes them as `lr000050u` and `lr000010u`.

- Observation: strict page-only continuation changed the optimizer picture.
  Evidence: the page-only winner is `wb_on_an_sn_opta_lr000010u` with `curve_metric_value=0.24025369978858352`, while the Adadelta page-only candidates both failed the regression guard in this study.

## Decision Log

- Decision: implement page-only continuation as a separate experiment path rather than replacing the cumulative focused matrix.
  Rationale: the cumulative 24-run matrix is already part of the repository's evidence chain and should remain reproducible. Page-only continuation is an alternative study regime, not a correction that invalidates the prior study.
  Date/Author: 2026-04-19 / Codex

- Decision: reuse the existing `training_policy="page_only"` control path instead of creating a second OCR trainer or dataset-preparation stack.
  Rationale: the desired semantics were already expressible by the current runner. The missing work was configuration, observability, artifact naming, and tests.
  Date/Author: 2026-04-19 / Codex

- Decision: keep the evaluation metric definitions, held-out pages, regression guard, plot generation, and artifact file layout the same.
  Rationale: the user wanted to isolate the effect of the training regime itself rather than re-open the entire policy-search space.
  Date/Author: 2026-04-19 / Codex

- Decision: treat per-policy regression-guard failures as recorded study outcomes rather than as study-level fatal errors.
  Rationale: the saved artifact is more useful when it preserves both passed and failed policy runs. The acceptance condition is that the run records the requested four policies faithfully, not that every candidate survives all nine continuation steps.
  Date/Author: 2026-04-19 / Codex

- Decision: give the page-only study its own stable latest aliases.
  Rationale: `recognition_finetune_results_latest.*` already refers to the cumulative default study. The page-only follow-up should not silently overwrite that evidence chain.
  Date/Author: 2026-04-19 / Codex

## Outcomes & Retrospective

The page-only continuation study was implemented end to end. The repository now exposes `get_page_only_followup_policy_configs(...)` in `app/tests/recognition_finetuning_config.py` and `run_page_only_continuation_experiment(...)` in `app/tests/recognition_finetuning_experiment.py`. The study writes its own timestamped artifact directories and its own stable latest aliases:

- `app/tests/logs/recognition_finetune_page_only_latest.md`
- `app/tests/logs/recognition_finetune_page_only_latest.json`
- `app/tests/logs/recognition_finetune_page_only_latest.png`
- `app/tests/logs/recognition_finetune_page_only_latest.txt`

The first completed artifact is `app/tests/logs/20260419_123216_ocrft_pageonly_eval_dataset/`. Its winning passed policy is `wb_on_an_sn_opta_lr000010u`, meaning Adam with `lr=0.00001` and `num_iter=200` on the `batch_max_pad / none / none / none` structural stack. That run achieved:

- `curve_metric_value=0.24025369978858352`
- `final_page_cer=0.17061310782241015`
- `first_step_gain=0.04439746300211417`
- `max_regression=0.0012684989429175286`

The plan also clarified a practical lesson for later work: strict continuation semantics and "all policies must complete all steps" are different things. The winning page-only policy completed the full 9-page continuation, but the study as a whole still includes shorter failed traces for policies that crossed the regression guard.

## Context and Orientation

The OCR research harness lives mainly in:

- `app/tests/recognition_finetuning_config.py`
- `app/tests/recognition_finetuning_experiment.py`
- `app/recognition/active_learning.py`
- `app/recognition/train.py`
- `app/tests/test_recognition_finetuning_page_only_unit.py`
- `app/tests/test_recognition_finetuning_page_only_e2e.py`

A "sequential fine-tuning step" means one cycle of training a checkpoint and then re-evaluating that checkpoint on the held-out evaluation pages. The `eval_dataset` ordering is determined by sorting the filenames under `app/tests/eval_dataset/images/`. In this repository, the first 9 pages are the fine-tune pages and the later 6 pages are the evaluation pages. That means the fine-tune sequence is `233_0002` through `233_0010`, while the evaluation pages remain `233_0011` through `233_0016`.

Under the page-only regime described here:

- step 0 evaluates the pretrained OCR checkpoint with no fine-tuning
- step 1 fine-tunes only on `233_0002`
- step 2 fine-tunes only on `233_0003` while loading the checkpoint produced after step 1
- step 6 fine-tunes only on `233_0007` while loading the checkpoint produced after step 5

The initial page-only study runs exactly these four policies and no others:

- Adam, `lr=0.00005`, `num_iter=60`
- Adam, `lr=0.00001`, `num_iter=200`
- Adadelta, `lr=0.2`, `num_iter=60`
- Adadelta, `lr=0.05`, `num_iter=200`

All four share the same structural settings:

- `training_policy=page_only`
- `width_policy=batch_max_pad`
- `oversampling_policy=none`
- `augmentation_policy=none`
- `lr_scheduler=none`
- `curve_metric=early_weighted_page_cer`
- `regression_guard_abs=0.005`
- `background_plus_rotation_variant_count=10`
- `shuffle_train_each_epoch=True`

## Plan of Work

The implementation begins in `app/tests/recognition_finetuning_config.py`. Add `get_page_only_followup_policy_configs(name="eval_dataset")`, which starts from the existing dataset config, keeps the 9-page fine-tune span and later held-out evaluation pages, sets `training_policy="page_only"`, and returns exactly four fully materialized `RecognitionEvalDatasetConfig` objects for the four explicit policies above.

Next, update `app/tests/recognition_finetuning_experiment.py` to expose `run_page_only_continuation_experiment(dataset_name="eval_dataset")`. That entrypoint should prepare the shared dataset inputs once, run each explicit config through the existing single-policy runner, rank the passed policies with the same `early_weighted_page_cer`, `final_page_cer`, and `first_step_gain` logic used elsewhere in the harness, and write a top-level summary and metrics payload with `study_mode="page_only_policy_continuation"`.

Still in the experiment runner, make the saved artifacts prove the continuation semantics directly. Each step record must include at least:

- `training_policy`
- `training_page_ids`
- `train_dataset_page_count`
- `base_checkpoint`
- `output_checkpoint`

The saved metadata for the winning policy must show `train_dataset_page_count == 1` for every post-baseline step. Step 6 must show `training_page_ids=["233_0007"]` and a `base_checkpoint` equal to step 5's `output_checkpoint`.

Add a page-only slug helper that can encode micro-scale learning rates without collisions. The implemented helper uses micro-units and writes tokens such as `lr000050u`, `lr000010u`, `lr200000u`, and `lr050000u`.

Finally, add regression coverage. The fast unit test in `app/tests/test_recognition_finetuning_page_only_unit.py` proves that the config helper returns the exact four policies, that the slug helper stays unique for both Adam micro-LR runs, and that page-only sequencing uses one train page and previous-step checkpoint carry-forward. The slow test in `app/tests/test_recognition_finetuning_page_only_e2e.py` proves that the full study writes the expected artifact family and that the winning run carries the checkpoint forward correctly.

## Concrete Steps

From the repository root, inspect the relevant control flow:

    Get-Content app\tests\recognition_finetuning_config.py
    Get-Content app\tests\recognition_finetuning_experiment.py
    Get-Content app\recognition\active_learning.py

Expected result: `training_policy` exists, the default cumulative path remains intact, and the page-only study uses a dedicated public entrypoint rather than piggybacking on the cumulative latest aliases.

Run the fast page-only unit coverage:

    C:\Users\intro\miniconda3\envs\gnn_layout\python.exe -m unittest app.tests.test_recognition_finetuning_page_only_unit -v

Expected result: the tests pass quickly and prove that:

- the config helper returns exactly four page-only policies
- the two Adam micro-LR runs have distinct slugs
- each sequential training step receives exactly one page
- each step after step 1 loads the previous step's output checkpoint

Run the slow page-only study:

    C:\Users\intro\miniconda3\envs\gnn_layout\python.exe -m unittest app.tests.test_recognition_finetuning_page_only_e2e -v

Expected result: a timestamped directory appears under `app/tests/logs/` with a name of the form `YYYYMMDD_HHMMSS_ocrft_pageonly_eval_dataset`, and the stable latest aliases under `app/tests/logs/recognition_finetune_page_only_latest.*` are refreshed.

Inspect the saved summaries:

    Get-Content app\tests\logs\recognition_finetune_page_only_latest.md
    Get-Content app\tests\logs\recognition_finetune_page_only_latest.json
    Get-Content app\tests\logs\20260419_123216_ocrft_pageonly_eval_dataset\policies\wb_on_an_sn_opta_lr000010u\fine_tune_metadata.json

Expected result: the top-level summary reports exactly four requested policy runs, the winning policy is `wb_on_an_sn_opta_lr000010u`, and the saved step metadata shows one train page per post-baseline step plus explicit checkpoint carry-forward.

## Validation and Acceptance

Acceptance for this plan is behavioral.

First, the repository must expose a dedicated page-only experiment path that is separate from the cumulative focused matrix. Running the page-only slow test must create a new page-only artifact directory and must not overwrite `recognition_finetune_results_latest.*`.

Second, the page-only study must keep the existing evaluation framework intact. The saved metrics must still include the same curve metric, regression guard, `winning_policy`, and winner-by-metric map. The intended variable is the train-dataset construction per step, not the scoring logic.

Third, every post-baseline step in the winning page-only run must prove that its train dataset contains exactly one page. The saved metadata must show `train_dataset_page_count == 1` and a one-element `training_page_ids` list for each post-baseline step.

Fourth, the continuation semantics must be explicit and verifiable. For the `eval_dataset` ordering used here, the artifact must show that the step trained on `233_0007` loads the checkpoint output by the preceding step trained on `233_0006`.

Fifth, the study must record exactly four requested policies with explicit optimizer, LR, and iteration-count reporting.

Sixth, the artifact slugs must be unambiguous across all four policies, especially for `0.00005` versus `0.00001`.

Seventh, policy-level regression-guard failures are acceptable as long as they are recorded faithfully. A failed policy may stop before the ninth continuation step and still remain a valid part of the saved study. The acceptance requirement is that the four requested policies are run and summarized, not that every candidate survives all nine continuation pages.

## Idempotence and Recovery

This plan is additive. The cumulative 24-run matrix and its historical artifacts remain intact. Re-running the page-only study creates a new timestamped directory every time instead of mutating old evidence.

If the page-only study fails mid-run, the correct recovery is to fix the cause and rerun the same slow-test command. Because artifacts are timestamped, retries do not require manual cleanup. If the page-only latest aliases were refreshed before a problem was fully understood, rerunning the study will refresh those aliases again to the newest successful page-only run without affecting the cumulative-study aliases.

## Artifacts and Notes

Important evidence for this plan includes:

- `app/tests/logs/20260418_231746_ocrft_eval_dataset/summary.md`
- `app/tests/logs/20260418_231746_ocrft_eval_dataset/metrics.json`
- `app/tests/logs/20260419_123216_ocrft_pageonly_eval_dataset/summary.md`
- `app/tests/logs/20260419_123216_ocrft_pageonly_eval_dataset/metrics.json`
- `app/tests/logs/20260419_123216_ocrft_pageonly_eval_dataset/policies/wb_on_an_sn_opta_lr000010u/fine_tune_metadata.json`
- `app/tests/logs/recognition_finetune_page_only_latest.md`
- `app/tests/logs/recognition_finetune_page_only_latest.json`

The page-only winner's step-level proof lives in `fine_tune_metadata.json`. In that file:

- step 5 trains on `233_0006` and writes `output_checkpoint=...step_05_233_0006...`
- step 6 trains on `233_0007`
- step 6 stores `base_checkpoint=...step_05_233_0006...`
- every post-baseline step stores `train_dataset_page_count=1`

## Interfaces and Dependencies

In `app/tests/recognition_finetuning_config.py`, the page-only follow-up is exposed through:

    def get_page_only_followup_policy_configs(name: str = "eval_dataset") -> tuple[RecognitionEvalDatasetConfig, ...]:
        ...

Each returned config has `training_policy="page_only"` and one of the four requested `(optimizer, lr, num_iter)` combinations.

In `app/tests/recognition_finetuning_experiment.py`, the page-only study is exposed through:

    def run_page_only_continuation_experiment(dataset_name: str = "eval_dataset") -> dict:
        ...

The returned dictionary includes at least:

- `study_mode`
- `run_dir`
- `summary_path`
- `metrics_path`
- `policy_runs`
- `winning_policy`
- `winning_policies_by_metric`

The page-only policy summary and per-step metadata now record:

- `training_policy`
- `training_page_ids`
- `train_dataset_page_count`
- `base_checkpoint`
- `output_checkpoint`
- `optimizer`
- `lr`
- `num_iter`

The additive tests for this work live in:

- `app/tests/test_recognition_finetuning_page_only_unit.py`
- `app/tests/test_recognition_finetuning_page_only_e2e.py`

Revision note, 2026-04-19: this ExecPlan now records the implemented page-only continuation path, the first completed artifact at `app/tests/logs/20260419_123216_ocrft_pageonly_eval_dataset/`, the winning policy `wb_on_an_sn_opta_lr000010u`, and the decision to preserve failed policies as valid recorded evidence rather than requiring every candidate to finish all nine continuation steps.
