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
- `app/tests/precommit_gate_config.py`
- `app/tests/recognition_finetuning_config.py`
- `app/tests/recognition_finetuning_experiment.py`
- `app/tests/test_recognition_finetuning_precommit_e2e.py`
- `app/tests/test_recognition_finetuning_e2e.py`

The key completed study artifacts are:

- `app/tests/logs/recognition_finetune_results_latest.json`
- `app/tests/logs/recognition_finetune_precommit_latest.json`

Those studies establish five important facts:

1. Earlier broad and focused sweeps were enough to identify the structural stack worth keeping: `batch_max_pad + no oversampling + no augmentation`.
2. Strict page-only continuation is viable, but it was materially weaker and more regression-guard-sensitive than the retained hybrid recipe on `eval_dataset`.
3. The live slow OCR verifier now keeps only the hybrid `page_plus_random_history` regime rather than carrying all earlier study modes in active code.
4. In the completed page-plus-random-history follow-up, `wb_on_an_hist10_sn_optd_lr200000u` beat both the earlier cumulative winner and the page-only winner on the primary metric and final-page CER.
5. Adam remained regression-guard-sensitive even in the hybrid regime, so the trusted recipe remains Adadelta `lr=0.2`, `num_iter=60`.

The repository now also has a two-phase pre-commit screen:

- a pretrained full-pipeline gate that verifies CRAFT plus GNN plus OCR still work together on the fixed evaluation manuscript
- a surrogate OCR fine-tuning gate that runs the best-known hybrid continuation recipe on perfect PAGE-XML-derived line crops and blocks commits only on `curve_metric_value`, `final_page_cer`, and `first_step_gain`

So the repository is now in a transitional state:

- the OCR verifier is real
- the pre-commit path now protects both the pretrained full pipeline and the current OCR fine-tuning subsystem
- the OCR research harness now keeps one replay-buffer-like continuation regime in active code while preserving the earlier cumulative and page-only conclusions in docs
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

The next OCR milestone is no longer deciding which offline continuation regime to keep. That decision is already made in the live harness: the retained path is `page_plus_random_history` plus the dedicated surrogate pre-commit gate for the trusted Adadelta recipe. The next uncertainty is whether that retained hybrid recipe generalizes strongly enough to deserve promotion into GUI-backed workflows.

Before the frontend is allowed to trigger OCR fine-tuning, the repository should:

- validate the retained page-plus-random-history regime on more than one manuscript sequence and more than one history replay size
- investigate why Adam remains regression-guard-sensitive in the continuation follow-ups
- add a small OCR model registry with active, candidate, and rollback metadata
- keep the surrogate pre-commit gate honest about its scope: it validates OCR fine-tuning under perfect segmentation inputs, not the full future human correction loop
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
