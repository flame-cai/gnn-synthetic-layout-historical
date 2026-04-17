# Local OCR Active-Learning Optimization, Verifier Stabilization, and Corpus Policy Study

## Summary

This plan replaces the older proposed OCR fine-tuning plans and keeps scope offline-first. It builds on the already-implemented pipeline in `app/recognition/active_learning.py`, `app/recognition/pagexml_line_dataset.py`, and `app/tests/recognition_finetuning_experiment.py`, then turns that pipeline into a reliable research harness for reducing manual OCR correction effort page by page.

The first task is to stabilize the verifier, because the current blocker is still checkpoint selection: `active_learning.py` hard-prefers `best_norm_ED.pth`, while the 2026-04-16 session report shows `best_accuracy.pth` can be better on the actual page-CER objective. Only after that selector is CER-aligned should new width, sampling, augmentation, and scheduler experiments be compared.

A width audit on the current `eval_dataset` already justifies the padding work. On fine-tune pages `233_0002` through `233_0006`, the current `imgH=50`, `imgW=2000`, `PAD=True` path yields median resized width about `254` pixels and median padding fraction about `0.8805`; `32/92` lines are `<=10` characters. With batch size fixed to `1`, the correct “width bucketing” implementation for this phase is batch-max padding, which degenerates to true per-sample arbitrary-width handling instead of padding every line to 2000 pixels.

The primary success metric will be an early-weighted page-CER learning-curve score, lower-is-better: `sum((K-s+1) * page_CER_s) / sum(K-s+1)` over steps `s=0..K`, with `K=5`. The verifier will also enforce a regression guard: no fine-tune step may worsen aggregate page CER by more than `0.005` absolute versus the previous step. Secondary metrics will be final page CER, first-step gain, per-step train time, and length-stratified line CER for short (`<=10` chars), medium (`11-30`), and long (`>30`) lines.

## Implementation Changes

- In `app/recognition/active_learning.py`, replace the static checkpoint preference with CER-aligned sibling checkpoint selection. After every fine-tune run, evaluate `best_accuracy.pth` and `best_norm_ED.pth` on the full current fine-tune corpus, choose the lower-CER checkpoint, and write `selector_metrics.json` with both scores, the chosen file, and the reason. Keep `validation_ratio=0.0` for training in this phase; the selector corpus is only for ranking sibling checkpoints, not for reporting the main benchmark.

- In `app/tests/recognition_finetuning_experiment.py` and `app/tests/recognition_finetuning_config.py`, add explicit experiment-policy fields: `width_policy`, `oversampling_policy`, `augmentation_policy`, `lr_scheduler`, `regression_guard_abs`, and `curve_metric`. Extend artifacts to include `curve_metrics.json`, `per_line.csv`, and `selector_metrics.json`. `per_line.csv` must log page id, line id, gt length, predicted text, edit distance, line CER, confidence, resized width, and pad fraction.

- In `app/recognition/dataset.py` and `app/recognition/ocr_defaults.py`, add `width_policy` with exactly two modes for this phase: `global_2000_pad` for the current baseline and `batch_max_pad` for the experimental path. `batch_max_pad` must resize to `imgH=50`, cap only when natural width exceeds `imgW=2000`, and otherwise pad only to the batch max width. Because batch size is fixed at `1`, this removes the global-width padding while preserving CTC-compatible variable sequence lengths.

- In `app/tests/recognition_finetuning_experiment.py`, make the first ablation a locked two-way comparison between `global_2000_pad` and `batch_max_pad`, both using the same checkpoint selector and current baseline training setup `Adadelta`, `lr=0.2`, `num_iter=60`, `batch_size=1`, `workers=0`. Promote `batch_max_pad` only if it improves the primary learning-curve metric and passes the regression guard.

- In `app/recognition/active_learning.py`, add difficulty scoring for the newly corrected page before each fine-tune step. Score every line with the current base checkpoint, using `line_cer = edit_distance(pred, gt) / max(len(gt), 1)`. Materialize oversampling by bounded duplication in the fine-tune corpus with `replication = 1 + floor(3 * min(line_cer, 1.0))`, capped at `4`. This keeps perfect lines at `1x`, hardest lines at `4x`, and normalizes for short-line length automatically.

- Keep oversampling policy deliberately narrow in this phase: compare only `none` versus `cer_weighted`, under the best width policy. Report the primary curve metric plus short/medium/long line CER so we can detect whether hard-line oversampling helps overall without damaging short-line learning.

- Add OCR-only augmentation during corpus materialization, never to evaluation crops. Background augmentation must jitter the page median fill by `delta` in `{-8, -4, 0, 4, 8}`, clipped to `[0,255]`. Rotation augmentation must use fixed-canvas affine rotation inside the existing crop rectangle, with requested angle sampled in `[-5°, 5°]`; if the foreground bbox would clip outside the crop, reduce the angle or skip rotation. Compare `none`, `background_only`, and `background_plus_rotation` only after width policy and oversampling are settled.

- In `app/recognition/train.py`, add scheduler plumbing with exactly `none`, `step`, and `cosine`. Keep optimizer fixed to `Adadelta` in this phase and keep `batch_size=1` fixed. Search `lr` over `{0.05, 0.1, 0.2}` for `none`, and initial `lr` over `{0.1, 0.2}` for `step` and `cosine`, with `num_iter=60` unchanged. Only run this schedule study on the best width/sampling/augmentation stack.

- Keep the experiment order fixed and blocker-first: selector fix and verifier artifacts, width-policy comparison, CER-weighted oversampling comparison, augmentation comparison, then LR/scheduler comparison. Do not mix axes before the previous one has a winner.

- Keep all experiment outputs inside repo-local artifact directories under `app/tests/logs/`, keep run names short to avoid Windows path-length issues under OneDrive, and continue using repo-local JSON/CSV/plot writes only. Do not add any writes outside the repository for this phase.

- When this plan is executed outside Plan Mode, remove only the superseded proposed plans `docs/exec-plans/proposed/fine-tuning-easyOCR.md` and `docs/exec-plans/proposed/local-ocr-active-learning-next-steps.md`. Keep `docs/exec-plans/active/old-recognition-finetuning-session-report-2026-04-16.md` and `docs/exec-plans/active/old-recognition-finetuning-failure-log.md` as historical evidence.

## Test Plan

- Add a targeted checkpoint-selector test that creates sibling candidates and verifies the lower-CER checkpoint is selected, regardless of whether it is `best_accuracy.pth` or `best_norm_ED.pth`.

- Add a width-policy test for `AlignCollate` that proves `global_2000_pad` still pads to 2000 and `batch_max_pad` pads only to the current batch max width.

- Add an oversampling-materialization test that verifies replication counts are derived from normalized line CER and are capped at `4`.

- Add an augmentation test that verifies output size is unchanged, the foreground remains non-empty, and rotation never spills outside the fixed crop rectangle.

- Keep `app/tests/test_recognition_finetuning_e2e.py` as the slow verifier entrypoint, but make it emit the new curve metric, regression-guard result, and length-bucket metrics for every policy run.

- Run all commands from PowerShell using the `gnn_layout` environment. The canonical command for the slow verifier remains: `$env:CONDA_NO_PLUGINS='true'; conda run -n gnn_layout python -m unittest discover -s app/tests -p "test_recognition_finetuning_e2e.py" -v`.

- Acceptance for the new baseline is: CER-aligned selector enabled, reproducible artifacts written, and the current baseline beats the old failing selector behavior. Acceptance for each new policy axis is: improved early-weighted page-CER curve, no regression-guard failure, and no harmful shift in short-line CER.

## Assumptions and Defaults

- Scope stays offline-first in this phase. No GUI-triggered background fine-tuning, telemetry integration, or model-promotion work is included here.

- Training and inference batch size stay fixed at `1` everywhere. Width bucketing for this phase therefore means batch-max padding, which is equivalent to per-sample arbitrary-width handling.

- The main benchmark remains `eval_dataset` with pages `1-5` used for sequential fine-tuning and pages `10-15` used for evaluation, matching the current experiment structure.

- `vadakautuhala.pth` remains immutable; every fine-tuned checkpoint stays versioned in the experiment artifact folder.

- Windows/OneDrive stability is a first-class constraint. Keep `workers=0`, keep artifact paths short, and keep all reads and writes inside the repository while running through the `gnn_layout` PowerShell environment.
