# Tech Debt Tracker

This file is the short list of the highest-priority technical debts that are currently slowing reliable progress in this repository. It should stay focused on debts that materially affect correctness, evaluation credibility, or the path to the active-learning vision.

## Highest-Priority Debts

- The live OCR runtime exists, but restart, interruption, and rebuild hardening is still shallow.
  Why it matters: manuscript-local registry, candidate promotion, GPU preemption, and rebase behavior now exist, but most coverage is still unit-scale and the runtime has not yet been stress-tested across longer real annotation sessions or broader missing-artifact recovery cases.
  Recommended next action: expand restart/interruption/rebuild coverage around `app/job_orchestrator.py`, `app/manuscript_ocr_registry.py`, and `app/ocr_active_learning_runtime.py`, then validate the queue and fallback behavior on longer manuscript sessions.

- The slow OCR study still has a Windows-specific output failure mode when run through `conda run`.
  Why it matters: the 2026-04-17 verifier completed and wrote correct artifacts, but the wrapper crashed while printing long Unicode-heavy output because of a `cp1252` encoding issue. That is a reliability problem for both humans and future agents.
  Evidence: the study artifacts were created successfully, but the shell surfaced a `UnicodeEncodeError` from the `conda run` wrapper after completion.
  Recommended next action: make the slow-study entrypoint or its helper script capture UTF-8 output safely and document a direct-interpreter fallback as a first-class execution path.

- The OCR study benchmark is still too narrow.
  Why it matters: the retained hybrid verifier on `eval_dataset` is good enough for development guidance, but one manuscript sequence is still not strong enough for product or research claims.
  Recommended next action: rerun the retained hybrid `page_plus_random_history` recipe on additional manuscript sequences and a small set of history replay sizes before treating the current recipe as generally stable.

- Manuscript-local telemetry is now recorded, but there is still no checked-in evaluator that turns it into effort curves.
  Why it matters: the runtime writes page and job events, edit distances, queue transitions, and checkpoint lineage under `input_manuscripts/<manuscript>/active_learning/recognition/telemetry/`, but that data is not yet summarized into manuscript-level effort trends or cross-manuscript reports.
  Recommended next action: add a small evaluator that consumes `page_events.jsonl` and `job_events.jsonl` and writes manuscript-level effort summaries for review.

- The save and recognition routes still mix legacy logging and new orchestration concerns in one module.
  Why it matters: `app/app.py` still handles legacy `node_corrections/*.json` writes, PAGE-XML generation, OCR runtime queueing, and background recognition thread startup inside the same route handlers. That makes the next round of CRAFT or GNN workflow work harder to reason about safely.
  Recommended next action: split route-level workflow helpers and decide whether the legacy `node_corrections` sidecar should remain alongside the newer structured telemetry.

- The Flask app still relies on a direct `sys.path.append(...)` hack for local OCR imports.
  Why it matters: the offline OCR package has matured, but `app/app.py` still treats `app/recognition/` as an import side path instead of a clean package dependency. That makes future background workers and isolated job runners harder to build safely.
  Evidence: `app/app.py` appends `app/recognition` to `sys.path` before importing OCR helpers.
  Recommended next action: replace the path hack with package-safe imports and one consistent app-root import strategy.

## Recently Retired Debts

- The repository previously had no formal OCR model registry or promotion rule.
  Status: retired on 2026-04-20.
  Evidence: `app/manuscript_ocr_registry.py` now persists the manuscript-local registry, and `app/ocr_active_learning_runtime.py` plus `app/tests/test_manuscript_ocr_registry_unit.py` exercise active, candidate, and previous-active behavior.

- The GUI previously had no safe OCR fine-tuning integration path.
  Status: retired on 2026-04-20.
  Evidence: `app/job_orchestrator.py`, `app/device_leases.py`, `app/ocr_active_learning_runtime.py`, and `app/ocr_model_manager.py` now provide queueing, device coordination, manuscript-aware checkpoint loading, and rebase handling for the live runtime.

- Human correction effort was previously not logged as structured evaluation data.
  Status: retired on 2026-04-20.
  Evidence: `app/telemetry.py` now writes structured page/job telemetry, `app/ocr_active_learning_runtime.py` writes `page_events.jsonl` and `job_events.jsonl`, and `app/tests/test_recognition_telemetry_unit.py` covers the core metric helpers.

- OCR checkpoint selection was previously not CER-aligned.
  Status: retired on 2026-04-17.
  Evidence: `app/recognition/active_learning.py` now supports both CER-aligned sibling evaluation and an explicit direct-preference mode, writes `selector_metrics.json`, and the broad OCR study used the selector successfully across policy runs.

- The OCR verifier previously lacked policy metadata and per-line artifacts.
  Status: retired on 2026-04-17.
  Evidence: the current harness now writes `curve_metrics.json`, `per_line.csv`, `selector_metrics.json`, and policy summaries under each study run.
