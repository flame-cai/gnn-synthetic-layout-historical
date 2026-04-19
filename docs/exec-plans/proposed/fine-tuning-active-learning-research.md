# Focused 9-Page OCR Policy Follow-Up with LR and Optimizer Sweep

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This document is maintained in accordance with `PLANS.md` from the repository root.

## Purpose / Big Picture

After this change, the slow OCR verifier will run one focused 9-page follow-up study on `eval_dataset` instead of rerunning the older broad blocker-first policy search as its main mode. The new follow-up will compare only the three shortlisted structural policy stacks from the 2026-04-17 study, but it will now sweep four learning rates and both optimizers. The user-visible proof is that one long verifier command will produce a timestamped artifact folder under `app/tests/logs/` containing exactly 24 focused policy runs, explicit optimizer-aware run names, separate winners for the primary curve metric, `final_page_cer`, and `first_step_gain`, and the usual OCR study artifacts.

All other current follow-up assumptions remain the same unless this plan says otherwise. The study still fine-tunes cumulatively on the first 9 pages of `eval_dataset`, still evaluates on the later held-out pages, still keeps `lr_scheduler=none`, still preserves the 2026-04-17 artifacts as historical evidence, and still treats Windows-safe long-run execution as part of the deliverable.

## Progress

- [x] (2026-04-17 18:45 IST) Recorded the completed 5-page broad OCR study as the baseline for this follow-up plan.
- [x] (2026-04-17 18:46 IST) Narrowed the next study scope to the three shortlisted policy stacks requested by the user.
- [x] (2026-04-18 12:xx IST) Revised the follow-up specification to expand the focused study from a 9-run learning-rate-only sweep to a 24-run matrix over learning rate and optimizer, while keeping the 9-page setup and shortlist intact.
- [x] (2026-04-18 12:xx IST) Recorded the requirement that the `background_plus_rotation` policy should use heavier augmentation sampling: 10 extra augmented variants per logical training sample or oversampled replica.
- [x] (2026-04-18 12:xx IST) Confirmed from source that per-epoch train-order randomization is possible and is already aligned with the current loader design, so the implementation should make that behavior explicit and testable rather than redesigning dataset materialization around epoch order.
- [x] (2026-04-18 22:xx IST) Replaced the default slow-study runner with a fixed 24-run focused matrix while keeping the historical blocker-first path callable for reproduction.
- [x] (2026-04-18 22:xx IST) Updated `app/tests/recognition_finetuning_config.py` so the default slow-study path now exposes the 9-page follow-up shape, focused structural shortlist, optimizer list, LR list, and `background_plus_rotation_variant_count=10`.
- [x] (2026-04-18 22:xx IST) Extended study reporting and per-step metadata so every run records optimizer, LR, shuffle policy, and augmentation-variant count without deleting existing artifact keys.
- [x] (2026-04-18 22:xx IST) Replaced the focused-study policy slug encoding with the optimizer-aware `..._opt{a|d}_lrNNNN` format so `lr=0.001` remains unambiguous.
- [x] (2026-04-18 22:xx IST) Materialized 10 independently seeded `background_plus_rotation` variants per logical training sample or oversampled replica while keeping validation data unaugmented.
- [x] (2026-04-18 22:xx IST) Made the train-loader epoch-shuffle guarantee explicit with a seeded `torch.Generator`, loader metadata, and deterministic unit coverage.
- [x] (2026-04-18 22:xx IST) Preserved the Windows-safe long-run execution path and UTF-8 artifact logging in the focused-study workflow.
- [x] (2026-04-19 01:xx IST) Re-ran the targeted OCR unit tests and completed one full slow focused OCR verifier run, producing `app/tests/logs/20260418_231746_ocrft_eval_dataset/`.
- [x] (2026-04-19 01:xx IST) Updated `EVAL.md` and `VISION.md` in the same implementation pass so the documentation matches the focused-study matrix and data-sampling behavior.
- [x] (2026-04-19 01:41 IST) Ran a one-off low-LR Adam follow-up on the `wb_on_an` stack with `lr=0.00005`, `num_iter=60`, and wrote artifact `app/tests/logs/20260419_014120_ocrft_eval_dataset/`.
- [x] (2026-04-19 02:03 IST) Ran a second one-off low-LR Adam follow-up on the same `wb_on_an` stack with `lr=0.00001`, `num_iter=600`, and wrote artifact `app/tests/logs/20260419_020333_ocrft_eval_dataset/`.

## Surprises & Discoveries

- Observation: the optimizer toggle already exists in the OCR training code, so the optimizer sweep does not require inventing a second training stack.
  Evidence: `app/recognition/train.py` chooses `optim.Adam(...)` when `opt.adam` is true and `optim.Adadelta(...)` otherwise, and `app/recognition/active_learning.py::_build_finetune_options()` already passes an `adam` option through.

- Observation: per-epoch train-order randomization is already possible at the loader layer and should not be implemented by rematerializing files every epoch.
  Evidence: `app/recognition/dataset.py::Batch_Balanced_Dataset` builds the train `DataLoader` with `shuffle=True`, and `get_batch()` recreates the iterator after `StopIteration`, which is the epoch-equivalent boundary in the current iteration-based trainer.

- Observation: the current `background_plus_rotation` policy only creates one augmented variant per logical sample.
  Evidence: `app/recognition/active_learning.py::_materialize_split_dataset()` currently adds only one `("bgrot", "background_plus_rotation")` variant when augmentation is enabled for that policy.

- Observation: the current policy-slug encoding cannot safely represent the new study matrix.
  Evidence: `app/tests/recognition_finetuning_experiment.py::_policy_slug()` does not encode optimizer at all, and `_short_float_token(0.001)` would collapse to `000`, which would make `0.001` indistinguishable from `0.0` in file names.

- Observation: the historical 2026-04-17 slugs are optimizer-implicit rather than optimizer-neutral.
  Evidence: `app/tests/recognition_finetuning_config.py` defaults `training_overrides["adam"] = False`, so the checked-in historical slugs such as `wb_oc_an_sn020` refer to Adadelta-based runs at their stated learning rate.

- Observation: the current slow-study test still assumes the older 5-page, multi-axis search behavior.
  Evidence: `app/tests/test_recognition_finetuning_e2e.py` currently expects only 6 sequential steps and only checks that there were at least 2 policy runs.

- Observation: the completed focused 24-run matrix strongly favors Adadelta on `eval_dataset`; every Adam run failed the regression guard in the first full study artifact.
  Evidence: `app/tests/logs/20260418_231746_ocrft_eval_dataset/summary.md` lists all eight Adam configurations as `status=failed`.

- Observation: some Adadelta configurations also fail the regression guard, so the study runner must preserve those runs as recorded evidence instead of aborting the whole matrix after all policies finish.
  Evidence: `app/tests/logs/20260418_231746_ocrft_eval_dataset/summary.md` shows failed Adadelta runs such as `wb_oc_ar_sn_optd_lr0010`, `wb_oc_an_sn_optd_lr0001`, and `wb_on_an_sn_optd_lr0500`.

- Observation: the first focused 24-run matrix did not rule Adam out categorically; it only showed that the Adam learning rates in that matrix were too aggressive for this dataset.
  Evidence: the one-off artifact `app/tests/logs/20260419_014120_ocrft_eval_dataset/summary.md` shows an Adam run with `width_policy=batch_max_pad`, `oversampling_policy=none`, `augmentation_policy=none`, `lr=0.00005`, and `num_iter=60` that passed the regression guard and recorded `curve_metric_value=0.22449356140688065` and `final_page_cer=0.1477801268498943`, both better than the first focused-study winner `wb_on_an_sn_optd_lr0200`.

- Observation: pushing Adam lower and longer did not improve this policy stack.
  Evidence: the second one-off artifact `app/tests/logs/20260419_020333_ocrft_eval_dataset/summary.md` shows that `lr=0.00001` with `num_iter=600` still passed, but it regressed to `curve_metric_value=0.23295022102633095`, `final_page_cer=0.1553911205073996`, and `first_step_gain=0.05095137420718815`, while `per_step_train_seconds` grew from a final-step value of `130.71853569999803` in the 60-iteration run to `1012.6422151000006`.

- Observation: the current policy-slug encoding is still ambiguous for ad hoc sub-`0.001` one-off sweeps.
  Evidence: both one-off Adam runs wrote under the slug `wb_on_an_sn_opta_lr0000`, but their `metrics.json` files record different true learning rates (`5e-05` and `1e-05` respectively), so the JSON metadata is currently the source of truth for those experiments.

## Decision Log

- Decision: keep the 2026-04-17 broad 5-page study as historical evidence, but make the default slow verifier target the focused 9-page follow-up.
  Rationale: the broad sweep already reduced the search space. The next useful uncertainty is not width, oversampling, or scheduler discovery in the abstract; it is the stability of the three shortlisted structural stacks under the requested LR and optimizer sweep.
  Date/Author: 2026-04-17 to 2026-04-18 / Codex

- Decision: preserve the three shortlisted structural stacks and expand each of them across a Cartesian product of four learning rates and two optimizers.
  Rationale: this keeps the study focused while honoring the new request. The new run count is exactly `3 structural stacks x 4 learning rates x 2 optimizers = 24 policy runs`.
  Date/Author: 2026-04-18 / Codex

- Decision: keep `lr_scheduler=none` fixed in this phase and sweep only `lr in {0.001, 0.01, 0.2, 0.5}` with `optimizer in {Adadelta, Adam}`.
  Rationale: the previous plan already chose to stop spending study budget on scheduler shape. The new user request explicitly shifts the uncertainty to learning-rate sensitivity and optimizer choice.
  Date/Author: 2026-04-18 / Codex

- Decision: do not redesign fine-tune dataset preparation around epoch-specific file ordering.
  Rationale: the current training loop is iteration-based and already reshuffles at the `DataLoader` boundary. The clean implementation is to make that loader behavior explicit, deterministic, and testable, not to rewrite LMDB or copied image order at every epoch.
  Date/Author: 2026-04-18 / Codex

- Decision: materialize 10 extra `background_plus_rotation` variants per logical training sample or oversampled replica, and leave `background_only` behavior unchanged.
  Rationale: the user explicitly chose `10` when asked how much heavier augmentation sampling should be. Limiting the change to `background_plus_rotation` avoids unintentionally altering other policy families.
  Date/Author: 2026-04-18 / Codex

- Decision: replace the focused-study policy slug with an explicit optimizer-aware and LR-safe format.
  Rationale: the new matrix needs unique, human-readable artifact folders. The focused-study slug format will be `width_oversampling_augmentation_scheduler_optimizer_lr`, where optimizer tokens are `optd` for Adadelta and `opta` for Adam, and LR tokens are `lr0001`, `lr0010`, `lr0200`, and `lr0500`.
  Date/Author: 2026-04-18 / Codex

- Decision: keep existing artifact files and keys stable wherever possible, and make new reporting additive.
  Rationale: `AGENTS.md` says OCR research artifact format should stay stable unless there is a strong reason to change it. The new implementation should add fields such as optimizer, variant count, and winner-by-metric rather than removing old keys like `winning_policy`.
  Date/Author: 2026-04-18 / Codex

- Decision: treat per-policy regression-guard failures as recorded run outcomes, not as a study-level fatal error once the 24-run matrix has completed.
  Rationale: the acceptance criteria are about enumerating all 24 runs and ranking the passed ones, not about requiring every candidate to pass. Returning the completed matrix preserves evidence and keeps the slow verifier aligned with the artifact-first workflow.
  Date/Author: 2026-04-19 / Codex

- Decision: interpret the first focused matrix's Adam failures as evidence about the tested Adam LR band, not as evidence that Adam is unusable on `eval_dataset`.
  Rationale: the `20260419_014120_ocrft_eval_dataset` one-off run passed the regression guard and beat the focused-matrix winner on both `curve_metric_value` and `final_page_cer`, so the correct conclusion is that Adam needs a much lower LR regime than the one encoded in the 24-run sweep.
  Date/Author: 2026-04-19 / Codex

- Decision: treat the sub-`0.001` one-off `metrics.json` files as the canonical record for exact LR values until the slug helper is extended for ad hoc sweeps below the planned matrix range.
  Rationale: both low-LR Adam one-offs collapsed to the same `lr0000` slug token even though they represent materially different experiments.
  Date/Author: 2026-04-19 / Codex

## Outcomes & Retrospective

This revision keeps the spirit of the earlier focused follow-up but broadens it in a controlled way. The study is still much narrower than the original broad 11-policy blocker-first sweep, but it is no longer just a 9-run LR check. It is now a 24-run focused comparison that preserves the three shortlisted structural stacks, increases the fine-tune span to 9 pages, compares both optimizers, tests the four requested learning rates, makes `background_plus_rotation` materially richer, and treats train-order reshuffling as an explicit supported behavior.

Nothing in this plan should overwrite or rename the historical 2026-04-17 artifact folders. Those baseline artifacts remain part of the repository’s evidence chain and should continue to be cited directly.

The implementation completed the focused follow-up end to end. The default slow verifier now runs the 24-run matrix, the dataset and training metadata record optimizer, shuffle, and augmentation multiplicity explicitly, and the unit coverage now protects the new slugging, augmentation, and shuffle behavior.

The first completed focused artifact is `app/tests/logs/20260418_231746_ocrft_eval_dataset/`. In that run, the best passed policy on all three tracked rankings was `wb_on_an_sn_optd_lr0200`, with `curve_metric_value=0.22700365173938108`, `final_page_cer=0.15137420718816066`, and `first_step_gain=0.05433403805496828`.

The study also surfaced an important boundary condition: the matrix should not assume every candidate passes the regression guard. Many Adam runs failed outright, and a few Adadelta runs also failed, but keeping those runs in the top-level summary turned out to be more useful than aborting the whole study after completion.

Two newer one-off Adam follow-ups refine that conclusion. The first, `app/tests/logs/20260419_014120_ocrft_eval_dataset/`, kept the same `wb_on_an` structure but dropped Adam to `lr=0.00005` with `num_iter=60`. That run passed the regression guard and improved on the first focused-study winner for the primary curve metric (`0.22449356140688065` vs `0.22700365173938108`) and `final_page_cer` (`0.1477801268498943` vs `0.15137420718816066`), though it was still slightly worse on `first_step_gain` (`0.05243128964059196` vs `0.05433403805496828`).

The second one-off, `app/tests/logs/20260419_020333_ocrft_eval_dataset/`, pushed Adam lower and longer with `lr=0.00001` and `num_iter=600`. That run also passed and had `max_regression=0.0`, but it did not beat the shorter `lr=0.00005` run on any tracked score and it was far more expensive in wall time. The practical takeaway is that low-LR Adam is now a real contender for the `wb_on_an` stack, but the promising region looks closer to short-horizon runs around `5e-05` than to very long runs at `1e-05`.

## Context and Orientation

The current OCR research harness lives mainly in `app/tests/recognition_finetuning_experiment.py`, `app/tests/recognition_finetuning_config.py`, `app/recognition/active_learning.py`, `app/recognition/dataset.py`, `app/recognition/train.py`, and `app/tests/test_recognition_finetuning_e2e.py`.

The dataset ordering in `app/tests/eval_dataset/images/` is:

- `233_0002`
- `233_0003`
- `233_0004`
- `233_0005`
- `233_0006`
- `233_0007`
- `233_0008`
- `233_0009`
- `233_0010`
- `233_0011`
- `233_0012`
- `233_0013`
- `233_0014`
- `233_0015`
- `233_0016`

The current default config fine-tunes on the first 5 pages and evaluates on pages `233_0011` through `233_0016`. This follow-up changes the fine-tune span to the first 9 pages while keeping evaluation on the remaining later pages. The focused follow-up must therefore fine-tune cumulatively on `233_0002` through `233_0010` and evaluate on `233_0011` through `233_0016`. Because the curve metric scores every sequential fine-tune step plus the pretrained baseline, the follow-up now produces 10 sequential checkpoints per policy run: step 0 for the pretrained model and steps 1 through 9 after each cumulative page is added.

The 2026-04-17 broad study produced these historically important results and they must remain visible in the revised plan:

- Primary metric winner: `wb_oc_an_sn020`
  Meaning: `width_policy=batch_max_pad`, `oversampling_policy=cer_weighted`, `augmentation_policy=none`, `lr_scheduler=none`, `lr=0.2`, and implicitly `optimizer=Adadelta`
  Evidence: `curve_metric_value=0.25063928319742274`

- Best `final_page_cer`: `wb_on_an_sn020`
  Evidence: `final_page_cer=0.18266384778012684`

- Best `first_step_gain`: `wb_on_an_sn020`
  Evidence: `first_step_gain=0.05433403805496828`

The three shortlisted structural stacks remain the same, but they now represent structure only, not the full run identity:

- Structural stack `wb_oc_ar`: `width_policy=batch_max_pad`, `oversampling_policy=cer_weighted`, `augmentation_policy=background_plus_rotation`
- Structural stack `wb_oc_an`: `width_policy=batch_max_pad`, `oversampling_policy=cer_weighted`, `augmentation_policy=none`
- Structural stack `wb_on_an`: `width_policy=batch_max_pad`, `oversampling_policy=none`, `augmentation_policy=none`

For this follow-up, each structural stack must be expanded across:

- learning rates `0.001`, `0.01`, `0.2`, and `0.5`
- optimizers `Adadelta` and `Adam`
- fixed `lr_scheduler=none`

That produces exactly 24 focused policy runs. The historical slug names without optimizer remain valid only for the already-existing 2026-04-17 artifacts. New focused-study runs must use explicit optimizer-aware slugs so their paths and summaries are unambiguous.

The user also asked whether randomized train order per epoch is possible when batch size is always 1. In this repository, the answer is yes. The training loop is not written in epochs; it is written in iterations. The epoch-equivalent event is when the current train iterator exhausts the materialized train dataset and `Batch_Balanced_Dataset.get_batch()` recreates the iterator. Because the train loader already uses `shuffle=True`, the implementation should preserve that design, make it explicit in the metadata and tests, and avoid creating a second, more complicated epoch-level dataset-preparation system.

## Plan of Work

In `app/tests/recognition_finetuning_config.py`, convert the dataset configuration from a single broad-search default into an explicit focused-follow-up spec while preserving the ability to reproduce the old broad study separately. Keep the `eval_dataset` command entrypoint stable for users, but add fields that fully describe the new focused matrix: `fine_tune_page_count=9`, `eval_page_start_index=9`, `eval_page_end_index=15`, `training_policy="cumulative"`, `lr_scheduler="none"`, `focused_learning_rates=(0.001, 0.01, 0.2, 0.5)`, `focused_optimizers=("adadelta", "adam")`, `focused_structural_policies=(wb_oc_ar, wb_oc_an, wb_on_an)`, `background_plus_rotation_variant_count=10`, and `shuffle_train_each_epoch=True`. Keep `training_overrides["adam"]` as the low-level training switch consumed by `train.py`, but expose optimizer choice as a first-class study setting in the config so reporting and study expansion do not have to infer meaning from a boolean buried in overrides.

In `app/tests/recognition_finetuning_experiment.py`, replace the current blocker-first search loop with an explicit focused matrix runner for the default slow-study path. The implementation should build one `RecognitionEvalDatasetConfig`-derived run config per structural stack, optimizer, and LR combination, then run all 24 of them. The old axis-by-axis search code should remain callable as a separate historical path or helper for reproducibility, but it must no longer be the default code path behind the slow verifier. The study runner must stop thinking in terms of “challenger vs baseline” decisions and instead think in terms of a fixed experiment matrix and post-hoc rankings. The top-level `summary.md` and `metrics.json` for the study must report all 24 policy runs and three separate winners: best primary curve metric, best `final_page_cer`, and best `first_step_gain`. The existing `winning_policy` field should be retained and should continue to mean “winner on the primary curve metric,” while a new additive map such as `winning_policies_by_metric` should record all three winners.

Still in `app/tests/recognition_finetuning_experiment.py`, replace the focused-study slug helper with a format that safely distinguishes optimizer and the newly requested low LR. Use this exact segment structure for new focused-study runs:

- width token: `wg` or `wb`
- oversampling token: `on` or `oc`
- augmentation token: `an`, `ab`, or `ar`
- scheduler token: `sn`, `ss`, or `sc`
- optimizer token: `optd` for Adadelta and `opta` for Adam
- LR token: `lr0001`, `lr0010`, `lr0200`, `lr0500`, where the numeric part is `int(lr * 1000)` zero-padded to four digits

That yields example focused-study slugs such as `wb_oc_ar_sn_optd_lr0200`, `wb_oc_ar_sn_opta_lr0200`, and `wb_oc_ar_sn_optd_lr0001`. The implementation must not try to keep using the old `_short_float_token()` behavior for focused runs because it cannot safely encode `0.001`.

In `app/recognition/active_learning.py`, extend fine-tune dataset preparation so `background_plus_rotation` creates more than one augmented realization per logical sample. Do not change `apply_ocr_training_augmentation()` into a multi-image API. Keep that function as the single-image augmentation kernel and change `_materialize_split_dataset()` instead. When `augmentation_policy == "background_plus_rotation"` and augmentation is enabled for the train split, materialize the base sample plus 10 independently seeded augmented variants. Name those variants deterministically as `bgrot01` through `bgrot10` so the artifact tree and `gt_train.txt` remain readable. Preserve the current behavior for `augmentation_policy == "none"` and `augmentation_policy == "background_only"`, and keep the validation split unaugmented. Because `cer_weighted` oversampling happens before augmentation materialization, the multiplicative effect must be explicit in the manifest: a hard sample with replication factor 4 under `background_plus_rotation` will produce `4 x (1 base + 10 augmented) = 44` train items. Record the variant count and the effective materialized counts in `manifest.json` and in each step’s `fine_tune_metadata.json`.

Also in `app/recognition/active_learning.py`, keep train-order randomization at the loader level rather than the dataset-file level. The plan does not require re-copying or re-writing samples between epochs. Instead, pass through enough metadata so each fine-tune step records that train-order reshuffling is enabled, and thread `background_plus_rotation_variant_count`, optimizer name, and shuffle policy into the metadata payloads returned by `prepare_incremental_finetune_dataset()` and `fine_tune_checkpoint_on_pages()`.

In `app/recognition/dataset.py`, make the train-shuffle contract explicit. Keep `shuffle=True` in the training `DataLoader`. Add an explicit `torch.Generator` seeded from `opt.manualSeed` when constructing the train loader so the shuffle sequence is deterministic across runs but still changes on each iterator recreation. Treat iterator exhaustion and recreation as the epoch-equivalent boundary in this trainer. The implementation must not promise file-level epoch reshuffling in the materialized dataset directory; it must promise loader-level train-order reshuffling for each full pass through the train set. The validation loader behavior can remain unchanged.

In `app/recognition/train.py`, preserve the existing optimizer selection logic but make the training summary and logged options explicit about optimizer name and shuffle policy. The returned `training_summary` should include at least `optimizer_name`, `lr_scheduler`, `final_lr`, and `shuffle_train_each_epoch`. This is an additive change. It exists so downstream fine-tune metadata and study summaries can explain exactly what happened without reverse-engineering a boolean.

In `app/tests/test_recognition_active_learning_unit.py`, add focused regression coverage for the new mechanics. Add a unit test that `background_plus_rotation` dataset materialization produces 10 extra variants per logical sample and leaves the validation split unaugmented. Add a unit test that focused-study policy slugs remain unique for both optimizer choices and for `lr=0.001`. Add a deterministic loader-level shuffle test that verifies two full passes through a small synthetic training set use the same samples but not the same order when `shuffle_train_each_epoch=True`. If exact-order assertions prove too brittle across Torch versions, assert the explicit code-level invariants instead: the training `DataLoader` was constructed with `shuffle=True`, a dedicated generator seeded from `manualSeed` exists, and iterator exhaustion recreates the iterator before more batches are drawn.

In `app/tests/test_recognition_finetuning_e2e.py`, update the slow verifier expectation to the new follow-up shape. The winning policy’s sequential steps should now be 10 total entries, not 6. The study should now produce exactly 24 policy runs, not merely “at least 2.” The returned result should still expose `winning_policy`, but it must also expose the additive ranking structure for the three winning metrics. The test should continue to tolerate the fact that the identity of the winners may vary across metrics, and it should fail if the old broad-search path is still running by default.

Finally, update `EVAL.md`, `VISION.md`, and this ExecPlan in the same implementation pass. `EVAL.md` should describe the new 24-run focused follow-up, the explicit LR and optimizer sweep, the 10-variant `background_plus_rotation` sampling, and the loader-level epoch shuffle guarantee. `VISION.md` should reflect that the OCR research harness now supports optimizer sweeps and richer augmentation sampling in the focused follow-up.

## Concrete Steps

From the repository root, inspect the current OCR study baseline before making changes:

    Get-Content app\tests\logs\recognition_finetune_results_latest.md
    Get-Content app\tests\logs\recognition_finetune_results_latest.json

Expected result: the current latest summary still reflects the earlier 5-page study and shows `wb_oc_an_sn020` as the primary winner and `wb_on_an_sn020` as the best `final_page_cer` and `first_step_gain` policy.

After implementing the focused-study matrix and metadata changes, run the fast OCR unit tests:

    conda activate gnn_layout
    python -m unittest app.tests.test_recognition_active_learning_unit -v

Expected result: the selector, width-policy, oversampling, augmentation, slug-encoding, augmentation-multiplicity, and shuffle-policy tests all pass.

Then run the slow focused OCR study without `conda run`:

    conda activate gnn_layout
    python -m unittest discover -s app/tests -p "test_recognition_finetuning_e2e.py" -v

Equivalent direct-interpreter form:

    C:\Users\intro\miniconda3\envs\gnn_layout\python.exe -m unittest discover -s app/tests -p "test_recognition_finetuning_e2e.py" -v

Expected result: a new timestamped folder appears under `app/tests/logs/`; the top-level summary names exactly 24 focused policy runs; every run records its optimizer and LR explicitly; the winning-policy map includes separate winners for the primary metric, `final_page_cer`, and `first_step_gain`; and the study completes without the earlier `conda run` Unicode wrapper failure.

After the slow run completes, inspect the generated latest summary files:

    Get-Content app\tests\logs\recognition_finetune_results_latest.md
    Get-Content app\tests\logs\recognition_finetune_results_latest.json

Expected result: the latest summary now refers to the 9-page focused follow-up, lists the 24-run matrix, and shows explicit optimizer-aware slugs such as `..._optd_lr0200` or `..._opta_lr0010`.

## Validation and Acceptance

Acceptance for this plan is behavioral:

1. The OCR verifier runs a 9-page focused follow-up on `eval_dataset` without falling back to the old broad blocker-first policy search as its default slow-study mode.
2. The focused study compares exactly 24 runs: the three shortlisted structural stacks crossed with learning rates `0.001`, `0.01`, `0.2`, `0.5` and optimizers `{Adadelta, Adam}`.
3. Every focused run keeps `lr_scheduler=none`.
4. The new policy-slug format uniquely represents optimizer and LR, including `lr=0.001`, and no focused-study runs collide on path names.
5. The `background_plus_rotation` train split materializes 10 extra augmented variants per logical training sample or oversampled replica; validation remains unaugmented.
6. Per-epoch-equivalent train-order randomization is explicitly supported at the loader level and recorded in metadata. The implementation does not rely on re-materializing the train dataset between epochs to achieve this.
7. The study summary reports separate winners for the primary curve metric, `final_page_cer`, and `first_step_gain`, while keeping the existing `winning_policy` field for the primary metric winner.
8. Every policy run still writes the standard OCR artifacts, including `curve_metrics.json`, `per_page.csv`, `per_line.csv`, `selector_metrics.json`, `fine_tune_metadata.json`, and plots.
9. The Windows-safe long-run path remains the recommended way to execute the study, and long output no longer depends on `conda run` behaving correctly.

## Idempotence and Recovery

The focused-study code path must remain additive. Historical 2026-04-17 artifact folders must not be deleted or renamed. New study runs must continue to write to timestamped artifact directories so repeated experiments append evidence rather than overwrite it.

If the new focused-study default path fails, the historical broad-search runner should remain callable for reproduction, but it should not be the default path used by `test_recognition_finetuning_e2e.py`. If the Windows-safe runner fails, the fallback remains an activated `gnn_layout` shell using the direct `python -m unittest ...` form, not `conda run`.

If the increased `background_plus_rotation` multiplicity makes a run too slow or too large, the correct recovery is to inspect the generated manifest and metadata to verify counts, then adjust only the explicit variant-count field in config in a later plan revision. Do not silently reduce augmentation multiplicity in code without updating the plan and the docs.

## Artifacts and Notes

Key historical evidence that must remain available while implementing this plan:

- `app/tests/logs/20260417_155737_ocrft_eval_dataset/summary.md`
- `app/tests/logs/20260417_155737_ocrft_eval_dataset/metrics.json`
- `app/tests/logs/20260417_155737_ocrft_eval_dataset/policies/wb_oc_an_sn020/summary.md`
- `app/tests/logs/20260417_155737_ocrft_eval_dataset/policies/wb_on_an_sn020/summary.md`
- `app/tests/logs/20260417_155737_ocrft_eval_dataset/policies/wb_oc_ar_sn020/summary.md`

Those files are the shortlist baseline and should continue to be cited directly instead of paraphrased from memory.

Newer one-off Adam low-LR evidence that should also remain available:

- `app/tests/logs/20260419_014120_ocrft_eval_dataset/summary.md`
- `app/tests/logs/20260419_014120_ocrft_eval_dataset/metrics.json`
- `app/tests/logs/20260419_020333_ocrft_eval_dataset/summary.md`
- `app/tests/logs/20260419_020333_ocrft_eval_dataset/metrics.json`

For those two one-off artifacts, use `metrics.json` rather than the folder slug as the source of truth for exact LR because both runs currently encode as `wb_on_an_sn_opta_lr0000`.

Example focused-study slug names after this change:

- `wb_oc_ar_sn_optd_lr0001`
- `wb_oc_ar_sn_opta_lr0001`
- `wb_oc_an_sn_optd_lr0200`
- `wb_on_an_sn_opta_lr0500`

For the focused-study rankings, the primary winner is the passed run with the lowest `curve_metric_value`. The `final_page_cer` winner is the passed run with the lowest `final_page_cer`. The `first_step_gain` winner is the passed run with the highest `first_step_gain`. If ties occur, break them by lower `final_page_cer`, then lower `max_regression`, then lexicographically by slug so the output is deterministic.

## Interfaces and Dependencies

The implementation should keep the low-level OCR training interface stable and make the new study controls explicit at the config and reporting layers.

In `app/tests/recognition_finetuning_config.py`, the `RecognitionEvalDatasetConfig` dataclass must end this work with explicit focused-study fields for:

- the 9-page cumulative follow-up shape
- the exact LR list `(0.001, 0.01, 0.2, 0.5)`
- the optimizer list `("adadelta", "adam")`
- the three structural shortlist descriptors
- `background_plus_rotation_variant_count = 10`
- `shuffle_train_each_epoch = True`

In `app/tests/recognition_finetuning_experiment.py`, every run descriptor written to JSON must include:

- `width_policy`
- `oversampling_policy`
- `augmentation_policy`
- `lr_scheduler`
- `optimizer`
- `lr`
- `curve_metric`
- `regression_guard_abs`
- `background_plus_rotation_variant_count`
- `shuffle_train_each_epoch`

In `app/recognition/active_learning.py`, the manifest and fine-tune metadata must gain additive fields that capture:

- optimizer name
- shuffle policy
- `background_plus_rotation_variant_count`
- effective train materialized count after oversampling and augmentation
- variant labels for augmented samples

In `app/recognition/train.py`, the returned training summary must gain additive fields for:

- `optimizer_name`
- `shuffle_train_each_epoch`

No existing artifact files should be removed. Existing keys such as `winning_policy`, `curve_metric_value`, and the standard CSV/JSON file names must remain available.

Revision note, 2026-04-18: this document was rewritten to preserve the existing 9-page focused follow-up while expanding the requested study matrix to learning rates `0.001`, `0.01`, `0.2`, and `0.5`, adding both Adadelta and Adam, explicitly confirming loader-level per-epoch shuffle support, and specifying 10 extra `background_plus_rotation` samples per logical training sample or oversampled replica.

Revision note, 2026-04-19: implementation completed. The plan now records the concrete code changes, the first full focused-study artifact at `app/tests/logs/20260418_231746_ocrft_eval_dataset/`, the winning policy `wb_on_an_sn_optd_lr0200`, and the decision to treat failed policy runs as preserved study evidence rather than a study-level fatal error.

Revision note, 2026-04-19 (later): two one-off low-LR Adam follow-ups were added to the evidence chain. The `lr=0.00005`, `num_iter=60` run showed that Adam can pass the regression guard and beat the first focused-study winner on the main curve metric and final CER, while the `lr=0.00001`, `num_iter=600` run showed that going lower and much longer did not improve the outcome and exposed a remaining sub-`0.001` slug-precision caveat for ad hoc sweeps.
