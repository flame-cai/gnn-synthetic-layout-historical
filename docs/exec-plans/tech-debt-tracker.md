# Tech Debt Tracker

This file is the short list of the highest-priority technical debts that are currently slowing reliable progress in this repository. It should stay focused on debts that materially affect correctness, evaluation credibility, or the path to the active-learning vision.

## Highest-Priority Debts

- The repository still has no formal OCR model registry or promotion rule.
  Why it matters: the offline OCR verifier can now produce candidate checkpoints and rank them, but the app still has no safe concept of active model, candidate model, rollback target, or promotion decision.
  Recommended next action: add a small JSON-backed OCR model registry per manuscript with active, candidate, previous-active, verifier summary, and rollback metadata.

- The slow OCR study still has a Windows-specific output failure mode when run through `conda run`.
  Why it matters: the 2026-04-17 verifier completed and wrote correct artifacts, but the wrapper crashed while printing long Unicode-heavy output because of a `cp1252` encoding issue. That is a reliability problem for both humans and future agents.
  Evidence: the study artifacts were written under `app/tests/logs/20260417_155737_ocrft_eval_dataset/`, but the shell surfaced a `UnicodeEncodeError` from the `conda run` wrapper after completion.
  Recommended next action: make the slow-study entrypoint or its helper script capture UTF-8 output safely and document a direct-interpreter fallback as a first-class execution path.

- The OCR study benchmark is still too narrow.
  Why it matters: the current broad study used one dataset and 5 sequential fine-tune pages. That is good enough for development guidance, but not yet strong enough for product or research claims.
  Recommended next action: run the focused 9-page follow-up study on `eval_dataset` comparing only `wb_oc_ar_sn020`, `wb_oc_an_sn020`, and `wb_on_an_sn020` across learning rates `0.01`, `0.2`, and `0.8`, then extend to additional manuscript sequences.

- The GUI still has no safe OCR fine-tuning integration path.
  Why it matters: the backend can recognize text and the offline verifier can fine-tune checkpoints, but the app has no job queue, no training-versus-inference coordination, and no status model for candidate checkpoints.
  Recommended next action: implement GUI integration only after the focused 9-page OCR study stabilizes the policy choice, then add a background job manager that prevents training and inference from contending for the same device.

- Human correction effort is still not logged as structured evaluation data.
  Why it matters: `VISION.md` and `EVAL.md` define success in terms of reducing manual effort, but the current backend only logs node corrections and does not yet log OCR text edits, fine-tune triggers, or checkpoint ids per page.
  Recommended next action: add structured per-page logging in `app/app.py` for OCR edits, save cycles, recognition requests, fine-tune requests, and model lineage.

- The Flask app still relies on a direct `sys.path.append(...)` hack for local OCR imports.
  Why it matters: the offline OCR package has matured, but `app/app.py` still treats `app/recognition/` as an import side path instead of a clean package dependency. That makes future background workers and isolated job runners harder to build safely.
  Evidence: `app/app.py` appends `app/recognition` to `sys.path` before importing OCR helpers.
  Recommended next action: replace the path hack with package-safe imports and one consistent app-root import strategy.

## Recently Retired Debts

- OCR checkpoint selection was previously not CER-aligned.
  Status: retired on 2026-04-17.
  Evidence: `app/recognition/active_learning.py` now evaluates sibling checkpoints and writes `selector_metrics.json`, and the broad OCR study used that selector successfully across policy runs.

- The OCR verifier previously lacked policy metadata and per-line artifacts.
  Status: retired on 2026-04-17.
  Evidence: the current harness now writes `curve_metrics.json`, `per_line.csv`, `selector_metrics.json`, and policy summaries under each study run.
