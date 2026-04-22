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

The repository still does not provide production-ready GUI fine-tuning for all four tasks.

However, the OCR side is no longer only aspirational or GUI-free. As of 2026-04-20, the repository now has a first-pass save-triggered GUI OCR active-learning runtime on top of the existing offline harness.

As of 2026-04-22, the live GUI runtime carries one runtime-specific sibling-checkpoint knob: it currently defaults the sibling checkpoint choice to `best_norm_ED.pth` through `OCR_RUNTIME_SIBLING_CHECKPOINT_STRATEGY`. The offline harness and surrogate gate can still use the CER-aligned selector when that is the better fit.

The relevant live runtime files are:

- `app/recognition/active_learning_recipe.py`
- `app/job_orchestrator.py`
- `app/device_leases.py`
- `app/manuscript_ocr_registry.py`
- `app/ocr_active_learning_runtime.py`
- `app/ocr_model_manager.py`
- `app/telemetry.py`
- `app/profiling.py`

The relevant offline research and regression files remain:

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
- the GUI now has a first-pass OCR active-learning loop with manuscript-local registry, promotion, rebase detection, telemetry, and profiling
- the GUI runtime is still intentionally narrow and needs more hardening before it should be treated as fully mature product behavior

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

The next OCR milestone is no longer deciding which offline continuation regime to keep or whether the GUI may trigger OCR fine-tuning at all. Both decisions are now made in code: the retained path is `page_plus_random_history`, and the GUI save flow can trigger manuscript-local OCR fine-tuning through the recorded runtime recipe with direct promotion after training.

The next uncertainty is how robust that first-pass live runtime is across more manuscripts and longer annotation sessions. The highest-value follow-up work is:

- validate the retained page-plus-random-history regime on more than one manuscript sequence and more than one history replay size
- investigate why Adam remains regression-guard-sensitive in the continuation follow-ups
- harden restart, interruption, and rebuild behavior in the live manuscript registry/orchestrator path
- keep the surrogate pre-commit gate honest about its scope: it validates OCR fine-tuning under perfect segmentation inputs, not the full future human correction loop
- turn the new manuscript-local telemetry into manuscript-level effort summaries and trend reports
- keep the Windows-safe direct-interpreter execution path first-class for long OCR verifier runs

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
