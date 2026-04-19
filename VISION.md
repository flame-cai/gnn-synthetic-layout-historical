# VISION.md

This repository exists to reduce the total human effort required to digitize historical manuscripts while preserving or improving output quality. The tool is already useful as a semi-automatic annotation system. The long-term vision is to make it progressively better page by page through active learning.

## Product Vision

The semi-automatic tool in `app/` digitizes manuscripts in two stages:

1. Layout analysis: text-line segmentation, text-region grouping, and PAGE-XML creation.
2. Text recognition: recognizing text from segmented lines and updating PAGE-XML with Unicode text.

That means the repository is naturally active-learning-friendly for four tasks:

1. Character detection or a future CRAFT-like surrogate model.
2. GNN edge classification for text-line grouping.
3. Text-region grouping.
4. Text recognition.

By active learning here, we mean the user corrects model mistakes, those corrections become new training data, and later pages should require less manual work than earlier pages.

## Current Reality

The repository does not yet provide production-ready GUI fine-tuning for any of the four tasks.

However, the OCR side is no longer only aspirational. As of 2026-04-19, the repository has a real offline OCR active-learning research harness in:

- `app/recognition/active_learning.py`
- `app/tests/recognition_finetuning_config.py`
- `app/tests/recognition_finetuning_experiment.py`
- `app/tests/test_recognition_finetuning_e2e.py`
- `app/tests/test_recognition_finetuning_page_only_e2e.py`
- `app/tests/test_recognition_finetuning_page_plus_history_e2e.py`

The key completed study artifacts are:

- `app/tests/logs/20260418_231746_ocrft_eval_dataset/`
- `app/tests/logs/20260419_123216_ocrft_pageonly_eval_dataset/`
- `app/tests/logs/20260419_132843_ocrft_pagehist_eval_dataset/`

Those studies establish five important facts:

1. The default slow OCR verifier can now run a focused 9-page, 24-run matrix rather than the older blocker-first search path.
2. The harness now supports explicit optimizer sweeps, page-only continuation, and a page-plus-random-history continuation mode that replays 10 sampled lines from earlier pages at each step.
3. In the completed focused cumulative follow-up, the best passed policy on the repository's primary metric, final-page CER, and first-step gain is `wb_on_an_sn_optd_lr0200`.
4. In the completed page-only follow-up, strict single-page continuation is viable, but only one of the four tested policies passed the regression guard all the way through the 9-page continuation.
5. In the completed page-plus-random-history follow-up, `wb_on_an_hist10_sn_optd_lr200000u` beat both the focused cumulative winner and the page-only winner on the primary metric and final-page CER, while Adam still failed the regression guard.

So the repository is now in a transitional state:

- the OCR verifier is real
- the OCR research harness is useful and can compare cumulative, page-only, and replay-buffer-like continuation regimes
- the GUI fine-tuning loop is still future work

## Desired End State

For a manuscript with multiple pages, the user should eventually experience the following:

1. Page 1 requires the most manual correction.
2. The corrected page is turned into fine-tuning data.
3. Fine-tuning runs in the background at a safe time.
4. The next model version performs better on the next unseen page.
5. Manual effort continues to drop over later pages.

Success is not defined by a single accuracy number. Success means:

- lower total manual effort
- stable or improving output quality
- predictable and reversible model promotion
- a smooth annotation workflow that does not freeze or confuse the user

## Immediate Direction

The next OCR milestone is no longer proving that continuation-style verifier runs are feasible. The repository now has three working offline regimes: cumulative training, strict page-only continuation, and page-plus-random-history continuation. The next uncertainty is which continuation regime is strong enough and stable enough to deserve promotion toward GUI-backed workflows.

Before the frontend is allowed to trigger OCR fine-tuning, the repository should:

- validate the page-plus-random-history regime on more than one manuscript sequence and more than one history replay size
- decide whether page-plus-random-history is the right default successor to page-only continuation
- investigate why Adam remains regression-guard-sensitive in the continuation follow-ups
- add a small OCR model registry with active, candidate, and rollback metadata
- keep the Windows-safe direct-interpreter execution path first-class for long OCR verifier runs

Only after those pieces exist should the repository promote GUI integration work for OCR fine-tuning.

## Broader Research Direction

Over time, the same active-learning pattern should apply to all major tasks:

- OCR text recognition
- text-line segmentation
- text-region grouping
- eventually a trainable surrogate for character detection

CRAFT itself is not currently fine-tunable in this repository because the relevant training path is not available here. If active learning for character detection becomes important, the expected path is to introduce a surrogate model that can learn from user node additions and deletions.

## Non-Negotiable Product Constraints

- The app must remain usable even when research code is changing.
- A newly trained model must never silently replace the active model without a recorded promotion rule.
- Training and inference must not fight each other for the same device in a way that degrades the user experience.
- Every research claim should be backed by saved artifacts under `app/tests/logs/`.
- Human correction burden must eventually become a logged first-class metric rather than an anecdotal claim.
