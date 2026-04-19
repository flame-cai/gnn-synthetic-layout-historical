# Page-Plus-Random-History OCR Continuation Study

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This document is maintained in accordance with `PLANS.md` from the repository root.

## Purpose / Big Picture

After this change, the repository supports a hybrid OCR continuation regime that sits between cumulative training and strict page-only continuation. For each post-baseline step, the training set contains:

- all lines from the newly added page
- 10 randomly sampled lines drawn from any earlier fine-tune pages
- shuffled training order

The first-page edge case is handled explicitly. When there are no earlier fine-tune pages yet, the study trains only on the new page and records that no historical lines were available.

The user-visible proof is a timestamped artifact folder under `app/tests/logs/` whose saved metadata shows both the new page and the sampled history lines for each step, including deterministic sampling seeds, sampled page ids, sampled line references, and the step-1 zero-history fallback.

## Progress

- [x] (2026-04-19 12:xx IST) Read the existing page-only continuation implementation and confirmed that the train-dataset builder already accepted explicit prepared-page lists and additive metadata fields.
- [x] (2026-04-19 12:xx IST) Added `history_sample_line_count` to `RecognitionEvalDatasetConfig` and introduced `get_page_plus_random_history_policy_configs(...)` for the two requested policies.
- [x] (2026-04-19 12:xx IST) Implemented deterministic history-line sampling and additive manifest metadata in `app/recognition/active_learning.py`.
- [x] (2026-04-19 12:xx IST) Added the dedicated public entrypoint `run_page_plus_random_history_experiment(...)` plus hybrid-specific summary output and latest aliases in `app/tests/recognition_finetuning_experiment.py`.
- [x] (2026-04-19 13:xx IST) Added fast unit coverage in `app/tests/test_recognition_finetuning_page_plus_history_unit.py` and slow end-to-end coverage in `app/tests/test_recognition_finetuning_page_plus_history_e2e.py`.
- [x] (2026-04-19 13:28 IST) Ran the requested two-policy hybrid study and produced `app/tests/logs/20260419_132843_ocrft_pagehist_eval_dataset/`.
- [x] (2026-04-19 14:xx IST) Updated `EVAL.md`, `VISION.md`, and this ExecPlan so the documentation matches the implemented hybrid path and its saved artifacts.

## Surprises & Discoveries

- Observation: the page-only implementation already exposed most of the plumbing needed for the hybrid regime.
  Evidence: `fine_tune_checkpoint_on_pages(...)` already accepted explicit prepared pages and a base checkpoint path, so the missing piece was sample selection and metadata, not a new trainer.

- Observation: `training_page_ids` alone was too ambiguous once historical replay lines were introduced.
  Evidence: the hybrid train set can contain a new current page plus sampled lines from multiple earlier pages, so the saved manifest needed additive fields such as `current_page_ids`, `history_source_page_ids`, `history_sample_page_ids`, and `history_sample_line_refs`.

- Observation: the first-page edge case needed explicit metadata rather than silent fallback behavior.
  Evidence: in the winning artifact's `fine_tune_metadata.json`, step 1 records `history_source_page_ids=[]`, `history_sample_requested_count=10`, `history_sample_line_count=0`, and `history_sample_seed=null`.

- Observation: deterministic replay-line sampling mattered for reproducibility.
  Evidence: `app/recognition/active_learning.py` now records a stable `history_sample_seed` per step, and the winning artifact records concrete values such as `15930781408265787669` for step 2.

- Observation: the Adadelta hybrid candidate outperformed both the focused cumulative winner and the page-only winner on this dataset.
  Evidence: `wb_on_an_hist10_sn_optd_lr200000u` reached `curve_metric_value=0.22191428022294832` and `final_page_cer=0.14735729386892177`, compared with the cumulative winner `wb_on_an_sn_optd_lr0200` at `0.22700365173938108` and `0.15137420718816066`, and the page-only winner `wb_on_an_sn_opta_lr000010u` at `0.24025369978858352` and `0.17061310782241015`.

- Observation: Adam remained guard-sensitive even after replaying earlier lines.
  Evidence: the requested Adam hybrid policy `wb_on_an_hist10_sn_opta_lr000050u` failed the regression guard with `max_regression=0.006131078224101472`.

## Decision Log

- Decision: name the new regime `training_policy="page_plus_random_history"`.
  Rationale: the policy needs a name that is explicit about both the current-page requirement and the replay-buffer-like history sampling without implying full cumulative training.
  Date/Author: 2026-04-19 / Codex

- Decision: sample history lines without replacement and cap the sample count at the number of available earlier lines.
  Rationale: the user asked for 10 randomly sampled earlier lines, not weighted replay or repeated draws of the same line. The step-1 edge case and any low-line-count page prefixes should therefore degrade gracefully to fewer than 10 history lines.
  Date/Author: 2026-04-19 / Codex

- Decision: keep the train-set shuffle at the loader level and also shuffle the selected logical samples before materialization.
  Rationale: the user explicitly asked that the data be shuffled. The implementation now shuffles the selected current-page and history samples before they are materialized, while the existing loader-level epoch shuffle remains in place.
  Date/Author: 2026-04-19 / Codex

- Decision: preserve the cumulative and page-only artifact families and give the hybrid study its own stable latest aliases.
  Rationale: each study family answers a different research question. Keeping separate latest aliases avoids collapsing their evidence chains into one filename.
  Date/Author: 2026-04-19 / Codex

- Decision: limit the first hybrid follow-up to the two policies requested by the user.
  Rationale: this work was about testing the new data regime, not reopening the full optimizer and LR search space. The initial comparison should therefore stay small and exact.
  Date/Author: 2026-04-19 / Codex

## Outcomes & Retrospective

The hybrid continuation path was implemented end to end. The repository now exposes:

- `get_page_plus_random_history_policy_configs(...)` in `app/tests/recognition_finetuning_config.py`
- `run_page_plus_random_history_experiment(...)` in `app/tests/recognition_finetuning_experiment.py`

The hybrid study writes its own stable latest aliases:

- `app/tests/logs/recognition_finetune_page_plus_history_latest.md`
- `app/tests/logs/recognition_finetune_page_plus_history_latest.json`
- `app/tests/logs/recognition_finetune_page_plus_history_latest.png`
- `app/tests/logs/recognition_finetune_page_plus_history_latest.txt`

The first completed artifact is `app/tests/logs/20260419_132843_ocrft_pagehist_eval_dataset/`. Its winning passed policy is `wb_on_an_hist10_sn_optd_lr200000u`, meaning Adadelta with `lr=0.2`, `num_iter=60`, and `history_sample_line_count=10` on the `batch_max_pad / none / none / none` structural stack. That run achieved:

- `curve_metric_value=0.22191428022294832`
- `final_page_cer=0.14735729386892177`
- `first_step_gain=0.05137420718816066`
- `max_regression=0.001691331923890066`

The saved metadata proves that the first step used only the new page and that later steps used the new page plus exactly 10 sampled earlier lines when available. This makes the hybrid study a reproducible bridge between strict continuation and replay-style stabilization.

## Context and Orientation

The hybrid continuation work touches:

- `app/tests/recognition_finetuning_config.py`
- `app/tests/recognition_finetuning_experiment.py`
- `app/recognition/active_learning.py`
- `app/tests/test_recognition_finetuning_page_plus_history_unit.py`
- `app/tests/test_recognition_finetuning_page_plus_history_e2e.py`

The `eval_dataset` ordering is the same as the other OCR studies. The fine-tune pages are `233_0002` through `233_0010`, and the evaluation pages are `233_0011` through `233_0016`.

A "hybrid continuation step" means:

- evaluate the current checkpoint on the held-out pages
- select all lines from the newly added current page
- collect all lines from earlier fine-tune pages as the history pool
- sample up to 10 of those earlier lines without replacement
- combine the current-page lines and sampled history lines
- shuffle the logical training samples
- fine-tune the current checkpoint on that mixed train set

The first requested hybrid study runs exactly these two policies and no others:

- `training_policy=page_plus_random_history`, `history_sample_line_count=10`, `optimizer=adam`, `lr=0.00005`, `num_iter=60`
- `training_policy=page_plus_random_history`, `history_sample_line_count=10`, `optimizer=adadelta`, `lr=0.2`, `num_iter=60`

Both policies keep the same structural settings:

- `width_policy=batch_max_pad`
- `oversampling_policy=none`
- `augmentation_policy=none`
- `lr_scheduler=none`
- `curve_metric=early_weighted_page_cer`
- `regression_guard_abs=0.005`
- `background_plus_rotation_variant_count=10`
- `shuffle_train_each_epoch=True`

## Plan of Work

Start in `app/tests/recognition_finetuning_config.py`. Extend `RecognitionEvalDatasetConfig` with `history_sample_line_count: int = 0` and add `get_page_plus_random_history_policy_configs(name="eval_dataset")`. That helper should return exactly two configs, both with `training_policy="page_plus_random_history"` and `history_sample_line_count=10`, while differing only on the two requested `(optimizer, lr, num_iter)` combinations.

Next, update `app/recognition/active_learning.py`. Add a helper that chooses the logical training samples for one continuation step. It must:

- always include all current-page lines
- draw up to `history_sample_line_count` lines from earlier pages without replacement
- use a deterministic step-specific seed
- record enough metadata to prove what happened later

The saved manifest must include at least:

- `current_page_ids`
- `current_page_line_count`
- `history_source_page_ids`
- `history_source_line_count`
- `history_sample_requested_count`
- `history_sample_line_count`
- `history_sample_seed`
- `history_sample_page_ids`
- `history_sample_line_refs`

It must also record a `sample_origin` marker per materialized sample so the train set can later be audited as "current page" versus "history replay."

Then update `app/tests/recognition_finetuning_experiment.py`. The existing single-policy runner should gain a new branch for `training_policy == "page_plus_random_history"`. At step `n`, it should fine-tune on the current page `n` plus a history pool built from pages `1` through `n-1`. The public entrypoint `run_page_plus_random_history_experiment(dataset_name="eval_dataset")` should run the exact two requested policies, write `study_mode="page_plus_random_history_followup"`, and refresh the hybrid latest aliases without touching the cumulative or page-only latest aliases.

Add coverage in `app/tests/test_recognition_finetuning_page_plus_history_unit.py`. The unit tests should prove three things:

- the config helper returns the exact two requested policies
- the dataset-preparation helper includes the current page plus 10 earlier sampled lines when available and zero history lines on the first step
- the logical sample order is shuffled before materialization

The slow test in `app/tests/test_recognition_finetuning_page_plus_history_e2e.py` should prove that the end-to-end study writes the expected artifact family, that the first step records no history, that later steps record bounded history replay, and that the requested two policy descriptors survive intact in the saved summary.

## Concrete Steps

From the repository root, inspect the relevant control flow:

    Get-Content app\tests\recognition_finetuning_config.py
    Get-Content app\recognition\active_learning.py
    Get-Content app\tests\recognition_finetuning_experiment.py

Expected result: the hybrid regime is expressed as config and experiment-runner behavior rather than as a second OCR trainer.

Run the fast hybrid unit coverage:

    C:\Users\intro\miniconda3\envs\gnn_layout\python.exe -m unittest app.tests.test_recognition_finetuning_page_plus_history_unit -v

Expected result: the tests pass quickly and prove that:

- the helper returns exactly the two requested hybrid policies
- step 1 uses zero history lines
- later steps use at most 10 history lines
- current-page lines and history lines are combined and shuffled

Run the slow hybrid study:

    C:\Users\intro\miniconda3\envs\gnn_layout\python.exe -m unittest app.tests.test_recognition_finetuning_page_plus_history_e2e -v

Expected result: a timestamped directory appears under `app/tests/logs/` with a name of the form `YYYYMMDD_HHMMSS_ocrft_pagehist_eval_dataset`, and the stable latest aliases under `app/tests/logs/recognition_finetune_page_plus_history_latest.*` are refreshed.

Inspect the saved summaries:

    Get-Content app\tests\logs\recognition_finetune_page_plus_history_latest.md
    Get-Content app\tests\logs\recognition_finetune_page_plus_history_latest.json
    Get-Content app\tests\logs\20260419_132843_ocrft_pagehist_eval_dataset\policies\wb_on_an_hist10_sn_optd_lr200000u\fine_tune_metadata.json

Expected result: the top-level summary reports exactly the two requested policy runs, the winning policy is `wb_on_an_hist10_sn_optd_lr200000u`, step 1 records zero history lines, and later steps record a non-empty historical replay sample whose size never exceeds 10.

## Validation and Acceptance

Acceptance for this plan is behavioral.

First, the repository must expose a dedicated hybrid experiment path that is separate from both the cumulative focused matrix and the page-only continuation study.

Second, each post-baseline hybrid step must include all current-page lines. The saved manifest and fine-tune metadata must distinguish those current-page lines from the replayed historical lines.

Third, each post-baseline hybrid step after the first must use at most 10 earlier lines sampled from the pool of previous fine-tune pages. The saved metadata must record both the history pool and the sampled subset.

Fourth, the first-page edge case must be explicit and benign. The saved metadata for the first training step must show that no earlier history pages existed and that the train set therefore used only the new page.

Fifth, the train data must be shuffled. The unit coverage must prove that the selected logical samples are not left in deterministic "current page first, then history lines" order before materialization.

Sixth, the study must record exactly the two requested policies with explicit `history_sample_line_count`, optimizer, LR, and iteration-count reporting.

Seventh, the saved hybrid latest aliases must not overwrite the cumulative or page-only latest aliases.

## Idempotence and Recovery

This plan is additive. The cumulative and page-only study paths remain intact. Re-running the hybrid study creates a new timestamped directory every time instead of mutating old evidence.

If the hybrid study fails mid-run, the correct recovery is to fix the cause and rerun the same slow-test command. Because the artifacts are timestamped, retries do not require cleanup. The hybrid latest aliases can be refreshed safely by rerunning the study without affecting the other study families.

## Artifacts and Notes

Important evidence for this plan includes:

- `app/tests/logs/20260418_231746_ocrft_eval_dataset/summary.md`
- `app/tests/logs/20260419_123216_ocrft_pageonly_eval_dataset/summary.md`
- `app/tests/logs/20260419_132843_ocrft_pagehist_eval_dataset/summary.md`
- `app/tests/logs/20260419_132843_ocrft_pagehist_eval_dataset/metrics.json`
- `app/tests/logs/20260419_132843_ocrft_pagehist_eval_dataset/policies/wb_on_an_hist10_sn_optd_lr200000u/fine_tune_metadata.json`
- `app/tests/logs/recognition_finetune_page_plus_history_latest.md`
- `app/tests/logs/recognition_finetune_page_plus_history_latest.json`

The winning hybrid artifact proves the edge-case behavior directly:

- step 1 records `history_source_page_ids=[]`
- step 1 records `history_sample_line_count=0`
- step 2 records `history_source_page_ids=["233_0002"]`
- step 2 records `history_sample_line_count=10`
- later steps record deterministic `history_sample_seed` values and sampled page ids

## Interfaces and Dependencies

In `app/tests/recognition_finetuning_config.py`, the hybrid follow-up is exposed through:

    def get_page_plus_random_history_policy_configs(name: str = "eval_dataset") -> tuple[RecognitionEvalDatasetConfig, ...]:
        ...

The config dataclass now includes:

    history_sample_line_count: int = 0

In `app/tests/recognition_finetuning_experiment.py`, the hybrid study is exposed through:

    def run_page_plus_random_history_experiment(dataset_name: str = "eval_dataset") -> dict:
        ...

The returned dictionary includes at least:

- `study_mode`
- `run_dir`
- `summary_path`
- `metrics_path`
- `policy_runs`
- `winning_policy`
- `winning_policies_by_metric`

In `app/recognition/active_learning.py`, the continuation dataset-preparation path now accepts:

- `history_source_pages: list[PreparedPageDataset] | None = None`
- `history_sample_line_count: int = 0`

The saved manifest and fine-tune metadata now record:

- `current_page_ids`
- `history_source_page_ids`
- `history_sample_page_ids`
- `history_sample_line_refs`
- `history_sample_line_count`
- `history_sample_seed`
- `sample_origin`

The additive tests for this work live in:

- `app/tests/test_recognition_finetuning_page_plus_history_unit.py`
- `app/tests/test_recognition_finetuning_page_plus_history_e2e.py`

Revision note, 2026-04-19: this ExecPlan records the implemented hybrid continuation path, the first completed artifact at `app/tests/logs/20260419_132843_ocrft_pagehist_eval_dataset/`, and the result that `wb_on_an_hist10_sn_optd_lr200000u` outperformed both the focused cumulative winner and the page-only winner on `eval_dataset` while Adam remained regression-guard-sensitive.
