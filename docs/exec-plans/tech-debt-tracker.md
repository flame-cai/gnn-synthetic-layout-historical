# Tech Debt Tracker

This file is the short list of the highest-priority technical debts that are currently slowing reliable progress in this repository. It is intentionally brief and should stay focused on debts that materially affect correctness, evaluation credibility, or the path to the active-learning vision.

## Highest-Priority Debts

- OCR checkpoint selection is not CER-aligned.
  Why it matters: the sequential OCR fine-tuning evaluator is implemented, but later steps can still fail because the code prefers the wrong saved checkpoint from the same training run.
  Evidence: in `app/tests/logs/20260416_222641_recognition_finetune_eval_eval_dataset`, step 4 fails with selected checkpoint CER `0.227907`, while the sibling `best_accuracy.pth` gives `0.216913`.
  Recommended next action: evaluate sibling checkpoints on a validation corpus and choose the lowest-CER candidate instead of hard-preferring `best_norm_ED.pth`.

- The repository has no formal OCR model registry or promotion rule.
  Why it matters: background fine-tuning cannot be safely integrated into the app until the code can distinguish between a candidate checkpoint and the currently active checkpoint.
  Recommended next action: add a small JSON-backed model registry with candidate registration, validation-gated promotion, and rollback.

- Human correction effort is not yet logged as structured evaluation data.
  Why it matters: `EVAL.md` and `VISION.md` define success in terms of reduced manual effort, but the app still lacks first-class OCR correction-burden telemetry.
  Recommended next action: log per-page text edits, edit-distance summaries, save cycles, fine-tune triggers, and checkpoint ids from the save and recognition flows in `app/app.py`.

- The Flask app still relies on a direct `sys.path` append for local recognition imports.
  Why it matters: the OCR package has been cleaned up enough for offline use, but the app entrypoint still uses an import shortcut that hides packaging debt and makes future background workers harder to isolate cleanly.
  Recommended next action: replace the `sys.path.append(...)` pattern in `app/app.py` with package-safe imports and one consistent app-root import strategy.

- Active-learning evaluation exists only for one OCR dataset and one manuscript sequence.
  Why it matters: the current OCR benchmark is useful as a development signal, but it is too narrow to support strong product or research claims.
  Recommended next action: extend `app/tests/recognition_finetuning_config.py` and the evaluator to support multiple datasets, page sequences, and annotation-budget policies.
