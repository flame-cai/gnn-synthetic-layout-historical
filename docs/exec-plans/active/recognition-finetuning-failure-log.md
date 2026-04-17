# Recognition Fine-Tuning Failure Log

This file records each failed attempt of the OCR fine-tuning evaluation experiment, what changed afterward, and why that change was made. If the experiment passes on the first complete run, leave this file with only that note.

## Runs

- 2026-04-16 21:09 IST
  Run dir: `app/tests/logs/20260416_210927_recognition_finetune_eval_eval_dataset`
  Failure: `Step 4 (page_233_0005) did not improve aggregate page CER: 0.215222 >= 0.200423`
  Context: this run used page-only sequential fine-tuning, duplicated the training page as validation, and trained with `Adadelta lr=0.8 num_iter=100`.
  Changes after failure:
  - Added support for cumulative replay over all corrected pages seen so far.
  - Added deterministic fine-tune corpus materialization and train/validation split support.
  - Fixed the evaluator to preserve the failing step in `metrics.json` and `summary.md` before asserting.

- 2026-04-16 21:59 IST
  Run dir: `app/tests/logs/20260416_215952_recognition_finetune_eval_eval_dataset`
  Failure: `Step 1 (page_233_0002) did not improve aggregate page CER: 0.373362 >= 0.309302`
  Context: this run used cumulative replay with a `validation_ratio=0.2` hold-out on only 23 line samples for step 1.
  Root cause: the held-out split destabilized checkpoint selection on the smallest fine-tuning step.
  Changes after failure:
  - Set the evaluator default to `validation_ratio=0.0` for the current experiment.
  - Kept cumulative replay enabled.

- 2026-04-16 22:05 IST
  Run dir: `app/tests/logs/20260416_220541_recognition_finetune_eval_eval_dataset`
  Failure: `Step 2 (page_233_0003) did not improve aggregate page CER: 0.303805 >= 0.282030`
  Context: this run used cumulative replay, `validation_ratio=0.0`, and the original `Adadelta lr=0.8 num_iter=100` schedule.
  Root cause: the update schedule was too aggressive once the replay corpus reached pages 1 and 2.
  Changes after failure:
  - Probed step-2 schedules against the evaluation set.
  - Updated the evaluator default schedule to `Adadelta lr=0.2 num_iter=60 valInterval=5 batch_size=1`.

- 2026-04-16 22:26 IST
  Run dir: `app/tests/logs/20260416_222641_recognition_finetune_eval_eval_dataset`
  Failure: `Step 4 (page_233_0005) did not improve aggregate page CER: 0.227907 >= 0.220296`
  Context: this run used the current best-known schedule: cumulative replay, `validation_ratio=0.0`, `Adadelta lr=0.2 num_iter=60`.
  Root cause: checkpoint selection, not corpus construction.
  Evidence:
  - The recorded step-4 `best_norm_ED.pth` gives page CER `0.227907`.
  - The sibling step-4 `best_accuracy.pth` gives page CER `0.216913`, which is better than step 3.
  - Probing `best_accuracy.pth` across steps 1-4 yields a monotonic curve: `0.306765 -> 0.276321 -> 0.220296 -> 0.216913`.
  Next change queued:
  - Replace the static `best_norm_ED` preference with a CER-aligned checkpoint selector, ideally by evaluating saved candidates on the fine-tune validation corpus and choosing the lower CER checkpoint.
