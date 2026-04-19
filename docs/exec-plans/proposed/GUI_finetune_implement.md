do the test reuse the same code which the app uses? or will use?
**IMPORTANT**
1) fine-tuning happens after every page save. when user will be manually annotating the next page (by adding/deleting nodes/edges in layout mode, or by making correction to recognized text in recogntion mode), the fine-tuning of the recognition model should happen in the background, using the best fine-tuning recipe we have found 

The best recipe ((with hybrid continuation regime):
- training_policy=page_plus_random_history
- history_sample_line_count=10
- width_policy=batch_max_pad
- oversampling_policy=none
- augmentation_policy=none
- lr_scheduler=none
- optimizer=adadelta
- lr=0.2
- num_iter=60
- curve_metric=early_weighted_page_cer
- regression_guard_abs=0.005
- background_plus_rotation_variant_count=10
- shuffle_train_each_epoch=True

2) Write code with the entire pipeline (including the human in the loop) in mind. In the pipeline, CRAFT, GNN, and the recogntion model require the GPU at different times. Also note that eventually we would want to fine-tune all three CRAFT, GNN, and the recognition model in the background after each page save. We will thus need good backend orchestration framework in place which handles all nuances. For example on nuance is that when the user uploads a 200 page manuscript, CRAFT first needs to process all 200 - this is a significant bottleneck. So in the future, to make the UX smooth we might want to process the 200 in batches of 10. Once CRAFT processing is done, a graph is constructed on the CRAFT output, which a GNN does binary edge classification on, to get segmented textlines. Then the human in the loop and frontend comes into play. In the frontend, the human can manually make corrections to by adding/deleting nodes (this data can be used to fine-tune CRAFT), or by adding/deleting edges (this data can be used to fine-tune the GNN). Once they save the page, or go the recognition model, a page-XML (without recognized text) is created. This is where the recognition model fine-tuning recipe we have optimized will come in - to iteratively self-improve on the task of recognizing text-content from the segmented text-lines.

3) While digitizing a new target manuscript, help me track when GPU is being used the most, and what the bottle necks are, by CUDA profiling and logging. 

4) Implement this Active learning mode (for iterative finetuning of only the recognition model), but make the integration future-proof, by refering to EVAL.md and VISION.md and other relevant files.

5) Ensure the changes we do to the GUI and backend with this plan, do not fail the evaluation pre-commit checks.





# Bring OCR Fine-Tuning Into The GUI Safely

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This document is maintained in accordance with `PLANS.md` from the repository root.

## Purpose / Big Picture

After this change, the semi-automatic app will be able to turn corrected OCR pages into background fine-tuning jobs without turning the annotation workflow into a fragile research demo. The user-visible proof will be that the UI can show OCR model status, queue fine-tuning for a manuscript, keep serving recognition from a stable active checkpoint, and promote a new candidate checkpoint only after a recorded verification step says it is safe to do so.

This plan is intentionally future-facing. It should be implemented only after the OCR hyperparameter and policy search has stabilized enough to define a trusted background fine-tuning recipe.

## Progress

- [x] (2026-04-17 19:00 IST) Preserved the original intent of the seed notes: do not train and infer at the same time, understand what differs between offline research and GUI active learning, keep two rolling models per document, and keep the UX smooth.
- [x] (2026-04-17 19:02 IST) Confirmed the current backend already supports local OCR, Gemini OCR, save flows, and background recognition threads, but not fine-tuning job orchestration.
- [ ] Add a per-manuscript OCR model registry with active, candidate, and previous-active checkpoint metadata.
- [ ] Add backend APIs for requesting fine-tuning, checking job status, inspecting candidate metrics, promoting a candidate, and rolling back.
- [ ] Add a background OCR fine-tuning worker that reuses the offline research utilities instead of inventing a second training stack.
- [ ] Add coordination so OCR inference and OCR fine-tuning do not contend for the same device at the same time. 
- [ ] Add coordination so OCR inference and OCR fine-tuning can happen on consumer grade GPU's. Do GPU optimization in this direction, without affecting the functioning.
- [ ] Add frontend status surfaces in `app/frontend/src/components/ManuscriptViewer.vue` so the user can see model status, pending jobs, promotion results, and failures.
- [ ] Add structured logging for OCR edits, save cycles, fine-tune triggers, job durations, candidate outcomes, and promotions.
- [ ] Add verifier-gated promotion rules so a newly trained checkpoint never silently replaces the active checkpoint without evidence.

## Surprises & Discoveries

- Observation: the repository already has most of the offline mechanics needed for OCR fine-tuning.
  Evidence: `app/recognition/active_learning.py`, `app/recognition/pagexml_line_dataset.py`, and `app/tests/recognition_finetuning_experiment.py` already prepare PAGE-XML line crops, fine-tune checkpoints, rank sibling checkpoints, and evaluate policy runs.

- Observation: the current Flask app is not structured as a job system yet.
  Evidence: `app/app.py` uses direct request handlers and lightweight background threads for auto-recognition, but it has no model registry, queue, or verifier-backed promotion concept.

- Observation: the UI already has a place where OCR controls live.
  Evidence: `app/frontend/src/components/ManuscriptViewer.vue` already contains recognition-mode controls, auto-recognition toggles, save flows, and page navigation, so GUI fine-tuning should extend that component rather than inventing a separate screen first.

- Observation: training and inference cannot be treated as harmlessly concurrent on the same manuscript and device.
  Evidence: the current OCR path uses a global loaded model in `app/app.py`, while the offline OCR study shows fine-tuning steps can take tens of seconds. Running recognition and training at the same time would create unpredictable device contention and a poor annotation experience.

- Observation: GUI promotion rules still cannot rely on one vague "best" label.
  Evidence: the retained OCR verifier and the surrogate pre-commit gate both track `curve_metric_value`, `final_page_cer`, and `first_step_gain` separately, so the GUI will need an explicit promotion policy rather than a hand-wavy "use the latest trained model" rule.

## Decision Log

- Decision: do not train and infer at the same time on the same OCR device for the same manuscript.
  Rationale: this preserves responsiveness and prevents fine-tuning from degrading the page the user is actively working on.
  Date/Author: 2026-04-17 / Codex

- Decision: keep two rolling OCR models per target document.
  Rationale: the app needs a stable active checkpoint and a separate candidate checkpoint so it can train, evaluate, promote, and roll back safely.
  Date/Author: 2026-04-17 / Codex

- Decision: reuse the offline OCR fine-tuning utilities instead of creating a new GUI-specific training stack.
  Rationale: the repository already has a verified offline pipeline for dataset preparation, checkpoint selection, and artifact writing. GUI integration should orchestrate that code, not fork it.
  Date/Author: 2026-04-17 / Codex

- Decision: require an explicit verifier-backed promotion step before a candidate becomes active.
  Rationale: the app should never silently switch checkpoints based only on training completion.
  Date/Author: 2026-04-17 / Codex

- Decision: smooth UX is a hard requirement, not a nice-to-have.
  Rationale: this is a human-in-the-loop product. If the UI becomes confusing or blocks annotation work, the active-learning story fails even if the underlying training code is correct.
  Date/Author: 2026-04-17 / Codex

## Outcomes & Retrospective

This plan has not been implemented yet. What exists today is the offline verifier and fine-tuning harness, not GUI-triggered OCR fine-tuning. The goal of this plan is to bridge that gap safely, after the OCR policy search has stabilized enough to choose a trusted recipe.

The main lesson from the current codebase is that GUI fine-tuning should be treated as orchestration, model registry, and UX work. The low-level OCR training loop already exists.

## Context and Orientation

The relevant backend file is `app/app.py`. It currently:

- loads the local OCR model lazily
- runs local OCR or Gemini OCR
- saves PAGE-XML corrections
- supports background auto-recognition threads
- logs node corrections

The relevant frontend file is:

- `app/frontend/src/components/ManuscriptViewer.vue`

That component already manages:

- layout mode versus recognition mode
- save and save-and-next actions
- auto-recognition toggles
- recognition engine choice
- recognition text editing

The relevant offline OCR fine-tuning files are:

- `app/recognition/active_learning.py`
- `app/recognition/pagexml_line_dataset.py`
- `app/tests/recognition_finetuning_config.py`
- `app/tests/recognition_finetuning_experiment.py`
- `app/tests/test_recognition_finetuning_e2e.py`

The key architectural difference between the offline harness and the future GUI flow is this:

- the offline harness is allowed to own the whole run and sweep many policies in sequence
- the GUI flow must protect a human operator's current session, maintain a stable active model, and treat training as a background candidate-generation process

## Plan of Work

Start by adding a per-manuscript OCR registry in a repo-local directory under the manuscript root. The registry should record at least:

- manuscript id
- immutable base checkpoint
- current active checkpoint
- current candidate checkpoint, if any
- previous active checkpoint for rollback
- model lineage and creation timestamps
- verifier summary for each candidate
- current job state such as idle, training, verifying, ready_for_promotion, failed

This registry must live inside the repository, not in an external database, because the current system is file-oriented and the evaluation artifacts already live on disk.

Next, add a background fine-tuning worker. It should not live inside a request handler thread. It should consume explicit fine-tune jobs, call the existing OCR dataset-preparation and fine-tuning utilities, write artifacts under the manuscript or test-log root, and update the registry as the candidate progresses through training and verification. A single manuscript should never have more than one active OCR fine-tuning job at a time.

Then, add concurrency control so training and inference do not run at the same time on the same OCR device. The core rule is simple: recognition requests should continue using the stable active checkpoint, while training runs only when the device lease is free and the user is not waiting on OCR inference for that manuscript. If the user requests recognition while a fine-tune job is holding the lease, the app must either defer the recognition request with clear UI status or cancel or pause the fine-tune job according to a documented rule. Do not rely on optimistic GPU sharing.

After that, add verifier-gated promotion. A completed training job should produce a candidate checkpoint and a verifier summary. Promotion should be a separate state transition, not an automatic side effect of training finishing. The initial rule can be conservative: only promote if the candidate passes the chosen verifier recipe and the registry update is written successfully. If promotion fails or verification regresses, keep the current active checkpoint unchanged and record the failure.

Once the backend state model exists, extend `app/frontend/src/components/ManuscriptViewer.vue` with OCR fine-tuning status surfaces. The minimum UI should show:

- current OCR model state for the manuscript
- whether a fine-tune job is idle, queued, training, verifying, ready, or failed
- when the active checkpoint last changed
- whether the current page save has produced new training data
- a clear action to request fine-tuning when the feature is enabled
- clear explanations when recognition is deferred because training owns the device

Finally, add structured logging and evaluation hooks. The GUI flow must log OCR edits, save cycles, fine-tune requests, training durations, verifier outcomes, promotions, rollbacks, and the checkpoint id used for each page. Without that logging, the repository still cannot measure whether GUI fine-tuning actually reduces manual effort.

## Concrete Steps

Before implementation, inspect the current OCR research and UI touchpoints:

    Get-Content app\app.py
    Get-Content app\frontend\src\components\ManuscriptViewer.vue
    Get-Content app\recognition\active_learning.py
    Get-Content app\tests\recognition_finetuning_experiment.py

Expected result: you can trace the current save flow, recognition flow, OCR training utilities, and verifier artifact layout.

During implementation, add backend smoke coverage for the registry and queue logic. For example, add tests that:

- create a candidate checkpoint entry without changing the active checkpoint
- reject promotion when verifier status is not passing
- promote successfully and preserve a rollback pointer
- block or defer recognition when a manuscript-scoped OCR training job holds the device lease

After backend implementation, run the existing OCR unit tests and any new registry or queue tests:

    conda activate gnn_layout
    python -m unittest app.tests.test_recognition_active_learning_unit -v

After frontend implementation, add a small scripted smoke test or at minimum a documented manual test that:

1. opens a manuscript
2. edits OCR text
3. saves the page
4. requests fine-tuning
5. sees status change from idle to training to verifying to ready
6. confirms that recognition still uses the active checkpoint until promotion

## Validation and Acceptance

Acceptance is:

1. The app can create a fine-tuning job from corrected OCR pages without blocking the current save flow.
2. The system keeps two rolling models per manuscript: a stable active model and a separate candidate model.
3. OCR inference and OCR fine-tuning do not run at the same time on the same device lease.
4. A finished training job does not silently replace the active checkpoint.
5. Promotion and rollback are explicit, logged, and recoverable.
6. The UI exposes enough status that a user can tell whether the model is training, waiting, ready, promoted, or failed.
7. The implementation writes structured logs that later evaluation code can use to measure human effort and model lineage.

## Idempotence and Recovery

Registry updates must be idempotent and crash-safe. If the app dies mid-training, the active checkpoint must still be well-defined on restart. If promotion fails halfway through, the previous active checkpoint must remain available and the registry must make that clear.

Fine-tuning jobs should be restartable or safely disposable. Never leave the app in a state where the UI believes a candidate is active but the checkpoint file or verifier evidence does not exist.

## Artifacts and Notes

The original seed notes that motivated this plan were:

- do not train and infer at the same time; train when the human is annotating
- check what is fundamentally different in offline research and GUI active learning
- keep two models per target document rolling
- UI and UX experience should be smooth

This rewritten ExecPlan preserves those requirements and expands them into concrete backend, frontend, verification, and recovery work.

Revision note, 2026-04-17: this document was rewritten from a four-line seed note into a full ExecPlan. The new version preserves the original intent while specifying the registry, job, promotion, rollback, concurrency, logging, and UI work needed to make GUI OCR fine-tuning safe after hyperparameter search stabilizes.
