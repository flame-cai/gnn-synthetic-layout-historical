# Focused 9-Page OCR Policy Follow-Up

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This document is maintained in accordance with `PLANS.md` from the repository root.

## Purpose / Big Picture

After this change, the OCR verifier will no longer rerun the full broad 5-page policy sweep as its main research mode. Instead, it will run a focused 9-page follow-up study that builds directly on the 2026-04-17 findings. The user-visible proof is that one slow verifier command will compare only the three shortlisted policy stacks `wb_oc_ar_sn020`, `wb_oc_an_sn020`, and `wb_on_an_sn020`, each at learning rates `0.01`, `0.2`, and `0.8`, write a clean artifact folder under `app/tests/logs/`, and do so without hitting the same Windows Unicode printing failure that affected the earlier `conda run` execution.

## Progress

- [x] (2026-04-17 18:45 IST) Recorded the completed 5-page broad OCR study as the baseline for this follow-up plan.
- [x] (2026-04-17 18:46 IST) Narrowed the next study scope to the three shortlisted policy stacks requested by the user.
- [ ] Add a focused-study mode to `app/tests/recognition_finetuning_experiment.py` so it runs only the shortlisted policy stacks instead of the full axis-by-axis policy search.
- [ ] Update `app/tests/recognition_finetuning_config.py` so the focused study uses 9 sequential fine-tune pages and keeps evaluation on the remaining later pages.
- [ ] Add a fixed shortlist policy map that expands `wb_oc_ar_sn020`, `wb_oc_an_sn020`, and `wb_on_an_sn020` into explicit policy fields so there is no ambiguity in future runs.
- [ ] Extend the focused study to sweep learning rates `0.01`, `0.2`, and `0.8` while keeping `lr_scheduler=none`.
- [ ] Emit summary rankings for primary metric, final-page CER, and first-step gain, because the 2026-04-17 study showed those leaders are not necessarily the same.
- [ ] Make the slow-study execution path Windows-safe by avoiding the `conda run` output trap and by capturing UTF-8 logs in the artifact folder.
- [ ] Re-run the targeted OCR unit tests and the slow OCR verifier in the updated focused-study mode.

## Surprises & Discoveries

- Observation: the broad 5-page study already reduced the large search space substantially.
  Evidence: `app/tests/logs/20260417_155737_ocrft_eval_dataset/summary.md` ranks `wb_oc_an_sn020` first on the primary curve metric, while `wb_on_an_sn020` is best on both `final_page_cer` and `first_step_gain`.

- Observation: `batch_max_pad` was the only width policy worth carrying forward.
  Evidence: `wg_on_an_sn020` had `curve_metric_value=0.27947246551897714`, while `wb_on_an_sn020` improved that to `0.2510721836303232` and also cut train time materially.

- Observation: CER-weighted oversampling helps the repository's chosen primary metric, but not every secondary metric.
  Evidence: `wb_oc_an_sn020` beat `wb_on_an_sn020` on `curve_metric_value`, but `wb_on_an_sn020` remained better on `final_page_cer` and `first_step_gain`.

- Observation: the most aggressive augmentation in the current shortlist did not collapse quality, but it also did not win the primary metric.
  Evidence: `wb_oc_ar_sn020` stayed regression-guard safe and finished with `final_page_cer=0.1835095137420719`, but its primary `curve_metric_value=0.259710057384476` remained worse than `wb_oc_an_sn020`.

- Observation: the previous slow-study command can finish the actual work and still fail at the wrapper layer on Windows.
  Evidence: the 2026-04-17 run wrote valid artifacts under `app/tests/logs/20260417_155737_ocrft_eval_dataset/`, but the shell surfaced a `UnicodeEncodeError` from the `conda run` wrapper while printing long output.

## Decision Log

- Decision: keep the current broad 5-page study as historical evidence, but make the next implementation target a focused 9-page follow-up rather than another full policy sweep.
  Rationale: the broad sweep already did its job by pruning the search space to three meaningful contenders.
  Date/Author: 2026-04-17 / Codex

- Decision: compare only `wb_oc_ar_sn020`, `wb_oc_an_sn020`, and `wb_on_an_sn020` in this phase.
  Rationale: those are the three stacks the user explicitly wants to carry forward, and they preserve the important trade-off between early curve quality, final-page CER, and augmentation.
  Date/Author: 2026-04-17 / Codex

- Decision: keep `lr_scheduler=none` fixed in this focused follow-up and sweep only `lr in {0.01, 0.2, 0.8}`.
  Rationale: the broader scheduler study did not beat the current best stack, so the next uncertainty is learning-rate sensitivity under the shortlisted policy stacks, not scheduler shape.
  Date/Author: 2026-04-17 / Codex

- Decision: change the canonical long-study command on Windows away from `conda run`.
  Rationale: the encoding failure happened after the study work completed, which makes it a tooling problem rather than a model problem. The repository should stop treating the fragile wrapper as the preferred path for long OCR runs.
  Date/Author: 2026-04-17 / Codex

## Outcomes & Retrospective

This plan starts from a completed milestone rather than a blank slate. The 2026-04-17 broad study turned the OCR verifier from a design idea into a working research harness with policy search, selector evidence, and reproducible artifacts. The remaining work is now focused and practical: extend the study to 9 fine-tune pages, compare only the three shortlisted stacks, sweep the three requested learning rates, and harden the Windows execution path so future runs fail only when the study itself fails.

Nothing in this plan should discard or overwrite the 2026-04-17 artifacts. Those artifacts are now part of the repository's research evidence chain.

## Context and Orientation

The current OCR research harness lives mainly in `app/tests/recognition_finetuning_experiment.py`, `app/tests/recognition_finetuning_config.py`, `app/recognition/active_learning.py`, and `app/tests/test_recognition_finetuning_e2e.py`.

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

The current default config fine-tunes on the first 5 pages and evaluates on pages `233_0011` through `233_0016`. This follow-up plan changes the fine-tune span to the first 9 pages, which means the focused study should fine-tune sequentially on `233_0002` through `233_0010` and continue evaluating on `233_0011` through `233_0016`. The curve metric therefore changes from `K=5` to `K=9`.

The three shortlisted policy slugs expand to:

- `wb_oc_ar_sn020`: `width_policy=batch_max_pad`, `oversampling_policy=cer_weighted`, `augmentation_policy=background_plus_rotation`, `lr_scheduler=none`, `lr=0.2`
- `wb_oc_an_sn020`: `width_policy=batch_max_pad`, `oversampling_policy=cer_weighted`, `augmentation_policy=none`, `lr_scheduler=none`, `lr=0.2`
- `wb_on_an_sn020`: `width_policy=batch_max_pad`, `oversampling_policy=none`, `augmentation_policy=none`, `lr_scheduler=none`, `lr=0.2`

For this follow-up, each of those stacks must be rerun at `lr=0.01`, `lr=0.2`, and `lr=0.8`. That produces exactly 9 policy runs.

## Plan of Work

First, update `app/tests/recognition_finetuning_config.py` so the focused study can express a 9-page setup without destroying the current broad-study baseline. The safest path is to add a second explicit config or study mode rather than changing the existing default in place. The new mode should set `fine_tune_page_count=9`, keep `eval_page_start_index=9` and `eval_page_end_index=15`, keep `validation_ratio=0.0`, and continue using cumulative training.

Next, update `app/tests/recognition_finetuning_experiment.py` so the study runner can accept an explicit shortlist of policy descriptors instead of always rebuilding the full width, oversampling, augmentation, and scheduler search tree. The implementation should keep the existing broad-search path intact for historical reproducibility, then add a focused path that expands the three fixed slugs above and sweeps only the requested learning rates. The focused study summary must rank policy runs on:

1. the primary curve metric
2. `final_page_cer`
3. `first_step_gain`

This ranking split is required because the current best-by-curve run is not the same as the best-by-final-CER run.

Then, update the study artifact writer so the focused mode writes a direct shortlist manifest. The top-level run summary should name the 9-page setup, the exact pages used for fine-tuning and evaluation, the 9 compared policy runs, the winner on the primary metric, the winner on `final_page_cer`, and the winner on `first_step_gain`.

After the policy matrix is narrowed, fix the Windows execution path. Do not keep `conda run` as the preferred command for long OCR study output. Add a repo-local runner or explicit helper command that uses the already-selected interpreter and writes stdout and stderr to UTF-8 artifact files. The implementation can be a small Python or PowerShell helper, but it must keep all writes inside the repository and must not require any external logging service. The important behavioral guarantee is that long verifier output can be printed or saved without the wrapper crashing on Unicode encoding.

Finally, update `app/tests/test_recognition_finetuning_e2e.py` so the slow verifier exercises the focused study mode instead of the old broad policy sweep. Keep `app/tests/test_recognition_active_learning_unit.py` as the fast coverage layer for selector, padding, oversampling, and augmentation mechanics.

## Concrete Steps

From the repository root, inspect the current OCR study artifacts before making changes:

    Get-Content app\tests\logs\recognition_finetune_results_latest.md
    Get-Content app\tests\logs\recognition_finetune_results_latest.json

Expected result: the files show `wb_oc_an_sn020` as the primary winner and `wb_on_an_sn020` as the best `final_page_cer` and `first_step_gain` policy in the 5-page study.

After implementing the focused-study mode, run the fast OCR unit tests:

    conda activate gnn_layout
    python -m unittest app.tests.test_recognition_active_learning_unit -v

Expected result: selector, width-policy, oversampling, and augmentation tests all pass.

Then run the slow focused OCR study without `conda run`:

    conda activate gnn_layout
    python -m unittest discover -s app/tests -p "test_recognition_finetuning_e2e.py" -v

Equivalent direct-interpreter form:

    C:\Users\intro\miniconda3\envs\gnn_layout\python.exe -m unittest discover -s app/tests -p "test_recognition_finetuning_e2e.py" -v

Expected result: a new timestamped folder appears under `app/tests/logs/`, the top-level summary names 9 focused policy runs, and the study completes without the earlier `UnicodeEncodeError` wrapper failure.

## Validation and Acceptance

Acceptance for this plan is behavioral:

1. The OCR verifier can run a 9-page focused follow-up study on `eval_dataset` without re-running the old full policy tree.
2. The focused study compares exactly 9 runs: the three shortlisted policy stacks at learning rates `0.01`, `0.2`, and `0.8`.
3. The focused run summary reports separate winners for the primary curve metric, `final_page_cer`, and `first_step_gain`.
4. Every focused run writes the standard OCR artifacts, including `curve_metrics.json`, `per_line.csv`, `selector_metrics.json`, and plots.
5. The slow-study execution path is Windows-safe: long output does not fail because of the earlier `conda run` Unicode printing problem.
6. The current broad-study code path remains available for historical reproduction unless it is explicitly retired in a later plan.

## Idempotence and Recovery

The focused-study code path should be additive. If the new 9-page mode fails, the previous broad-study harness should still be runnable. Artifact folders must remain timestamped and append-only so failed exploratory runs do not destroy earlier evidence.

If the Windows-safe runner is introduced and it fails, the fallback is an activated `gnn_layout` shell running the direct `python -m unittest ...` command, not `conda run`.

## Artifacts and Notes

Key evidence that must remain available while implementing this plan:

- `app/tests/logs/20260417_155737_ocrft_eval_dataset/summary.md`
- `app/tests/logs/20260417_155737_ocrft_eval_dataset/metrics.json`
- `app/tests/logs/20260417_155737_ocrft_eval_dataset/policies/wb_oc_an_sn020/summary.md`
- `app/tests/logs/20260417_155737_ocrft_eval_dataset/policies/wb_on_an_sn020/summary.md`
- `app/tests/logs/20260417_155737_ocrft_eval_dataset/policies/wb_oc_ar_sn020/summary.md`

Those files are the baseline for the shortlist and should be cited in later summaries instead of being paraphrased from memory.

Revision note, 2026-04-17: this document was rewritten from the earlier broad-search proposal after the 5-page OCR study completed. The new version narrows the search to the three shortlisted policies, extends the study to 9 fine-tune pages, adds the requested learning-rate sweep, and explicitly addresses the Windows output-encoding failure seen in the previous long run.
