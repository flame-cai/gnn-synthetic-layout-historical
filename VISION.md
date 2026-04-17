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

However, the OCR side is no longer only aspirational. As of 2026-04-17, the repository has a real offline OCR active-learning research harness in:

- `app/recognition/active_learning.py`
- `app/tests/recognition_finetuning_config.py`
- `app/tests/recognition_finetuning_experiment.py`
- `app/tests/test_recognition_finetuning_e2e.py`

The latest completed study is:

- `app/tests/logs/20260417_155737_ocrft_eval_dataset/`

That study established three important facts:

1. `batch_max_pad` is materially better than global padding for the current OCR verifier.
2. CER-weighted oversampling helps the primary early-weighted curve metric, even though it does not win on final-page CER.
3. The current best overall 5-page stack by the repository's primary metric is `wb_oc_an_sn020`, while `wb_on_an_sn020` is still the best on final-page CER and first-step gain.

So the repository is now in a transitional state:

- the OCR verifier is real
- the OCR research harness is useful
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

The next OCR milestone is narrower than full GUI integration.

Before the frontend is allowed to trigger OCR fine-tuning, the repository should complete a focused verifier-driven follow-up study that:

- increases sequential fine-tuning from 5 pages to 9 pages
- compares only the three shortlisted policies `wb_oc_ar_sn020`, `wb_oc_an_sn020`, and `wb_on_an_sn020`
- sweeps learning rates `{0.01, 0.2, 0.8}`
- hardens the Windows execution path so long verifier output does not fail because of `conda run` encoding issues

Only after that follow-up study should the repository promote GUI integration work for OCR fine-tuning.

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
