# OCR Active Learning Refinement Plan

## Summary

The next useful work is to move OCR fine-tuning from “implemented and protected by a surrogate gate” toward “measurably improving real annotation sessions.” The plan is to harden the live GUI runtime, add manuscript-level effort reporting, make slow OCR studies easier to run reliably on Windows, and extend the research harness so future manuscript sequences and replay sizes can be compared without changing core code.

This keeps the `VISION.md` goal central: corrected pages should become training data, later pages should require less manual correction, and every claim should be backed by verifier output or telemetry.

## Key Changes

- Add a manuscript effort reporter for OCR active learning.
  - Consume manuscript-local `page_events.jsonl`, `job_events.jsonl`, and registry state from `input_manuscripts/<manuscript>/active_learning/recognition/`.
  - Output a Markdown and JSON summary with per-page OCR text edit distance, changed line count, active checkpoint id, whether the page entered active learning, queued/running/completed job counts, and simple trend metrics.
  - Treat this as the first checked-in evaluator for real GUI correction effort, not a replacement for the surrogate OCR pre-commit gate.

- Harden live runtime recovery paths.
  - Add tests for app restart with pending jobs, missing active checkpoint fallback, interrupted fine-tune jobs, rebase after a consumed page is edited, duplicate saves, draft-to-commit transitions, and missing revision snapshots.
  - Keep current direct promotion behavior unchanged, but make every promotion/fallback/rebase state visible in the registry and effort report.
  - Preserve the current runtime recipe: `page_plus_random_history`, `history_sample_line_count=10`, `batch_max_pad`, no oversampling, no augmentation, Adadelta `lr=0.2`, `num_iter=60`, runtime sibling checkpoint default `best_norm_ed`.

- Add a Windows-safe OCR study launcher.
  - Introduce one small script that runs the slow OCR verifier through the selected `gnn_layout` Python executable and captures UTF-8 output safely.
  - Keep `scripts/run_precommit_eval.py` unchanged for the two-phase pre-commit path.
  - The new launcher should print the artifact directory, command, Python path, dataset name, recipe, and final pass/fail status.

- Make the OCR research harness easier to extend.
  - Keep `eval_dataset` as the only required checked-in dataset.
  - Add config structure for future OCR study datasets and history replay sizes without requiring code edits in the runner.
  - Default future replay-size comparison to `history_sample_line_count` values `5`, `10`, and `20`, with `10` remaining the current trusted production default until broader evidence says otherwise.
  - Keep Adam experiments out of the default gate because current evidence shows guard sensitivity; allow Adam only as explicit research config.

- Keep documentation synchronized.
  - Update `EVAL.md`, `VISION.md`, `AGENTS.md`, and the relevant ExecPlan after implementation.
  - Do not cite generated local artifact paths as durable docs. Copy durable conclusions, exact recipes, thresholds, and dataset assumptions into checked-in Markdown.

## Public Interfaces

- New CLI-style entrypoint:
  - `python scripts/run_ocr_study.py --dataset eval_dataset`
  - Optional flags: `--python <path>`, `--history-sizes 5,10,20`, `--output-root <path>`.
  - Default behavior uses the current trusted hybrid recipe and writes local artifacts without changing pre-commit behavior.

- New effort-report entrypoint:
  - `python -m app.tests.summarize_ocr_active_learning_effort --manuscript-root <path>`
  - Writes `ocr_effort_summary.json` and `ocr_effort_summary.md` under the manuscript active-learning recognition folder.

- No change to existing save/recognition API request shapes.
- No change to the current pre-commit thresholds or blocking metrics.

## Test Plan

- Unit tests:
  - Effort reporter handles empty telemetry, one-page sessions, multi-page sessions, draft saves, commit saves, missing job logs, and missing checkpoint records.
  - Registry/runtime tests cover fallback from missing active checkpoint to previous/base checkpoint.
  - Runtime tests cover rebase after editing a consumed page and duplicate saves not queueing redundant jobs.
  - Study launcher tests verify Python resolution, UTF-8 log capture, argument parsing, and nonzero exit reporting without running the full slow OCR study.

- Integration checks:
  - `conda run -n gnn_layout python -m unittest app.tests.test_recognition_active_learning_backend_unit -v`
  - `conda run -n gnn_layout python -m unittest app.tests.test_manuscript_ocr_registry_unit -v`
  - `conda run -n gnn_layout python -m unittest app.tests.test_recognition_telemetry_unit -v`
  - `conda run -n gnn_layout python -m unittest app.tests.test_recognition_finetuning_precommit_unit -v`
  - Slow validation when practical: direct environment Python running `app.tests.test_recognition_finetuning_precommit_e2e`.

- Acceptance scenarios:
  - After a GUI-like commit save with corrected OCR text, the effort report shows the page entered active learning and records text correction effort.
  - After a trained checkpoint is missing, the registry falls back predictably and records the fallback.
  - After re-saving a page already consumed into a checkpoint, the runtime marks `needs_rebase` and queues or reports rebuild work.
  - The new slow-study launcher completes or fails with readable UTF-8 logs and prints the artifact directory.

## Assumptions

- The current OCR recipe remains the production default until broader studies prove a better option.
- `eval_dataset` remains the only required benchmark dataset in the repository.
- Additional manuscript sequences may be added later, but this plan only adds the structure needed to register and compare them cleanly.
- Generated artifacts remain local and untracked; durable conclusions must be copied into checked-in docs.
- This plan focuses only on text-line recognition OCR fine-tuning, not GNN text-line segmentation, text-region grouping, or CRAFT fine-tuning.
