# Bring Save-Triggered OCR Active Learning Into The GUI Safely

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This document is maintained in accordance with `PLANS.md` from the repository root.

## Purpose / Big Picture

After this change, the semi-automatic app will gain one minimal new user-facing control, `Active Learning`, and it will be on by default. When a user makes a real page save, the backend will persist a page revision, decide whether that revision contains usable OCR supervision, and, if it does, queue a background OCR fine-tuning job for that manuscript using the repository's current best hybrid recipe:

- `training_policy=page_plus_random_history`
- `history_sample_line_count=10`
- `width_policy=batch_max_pad`
- `oversampling_policy=none`
- `augmentation_policy=none`
- `lr_scheduler=none`
- `optimizer=adadelta`
- `lr=0.2`
- `num_iter=60`
- `curve_metric=early_weighted_page_cer`
- `regression_guard_abs=0.005`
- `background_plus_rotation_variant_count=10`
- `shuffle_train_each_epoch=True`

The intended user-visible behavior is sequential and manuscript-specific. After page 1 is corrected and saved, the background worker fine-tunes a manuscript-local OCR checkpoint while the user annotates page 2. Recognition on page 3 should then use the best promoted checkpoint trained from page 1. After page 2 is corrected and saved, the next background job should produce the checkpoint used for page 4, and so on. If the user closes the app and returns later, the backend must load the best promoted checkpoint for that manuscript automatically rather than falling back to the global pretrained model.

This plan intentionally implements only OCR active learning now, but it does so on top of a generic backend orchestrator. The same orchestration layer must be able to schedule future CRAFT preprocessing, future GNN work, and eventual CRAFT and GNN fine-tuning without redesigning the app a second time.

## Progress

- [x] (2026-04-17 19:00 IST) Preserved the original seed intent: do not let background training degrade the annotation workflow, keep per-document model lineage, and keep the UX smooth.
- [x] (2026-04-17 19:02 IST) Confirmed the current backend already supports local OCR, Gemini OCR, PAGE-XML generation on save, and lightweight background recognition threads, but not fine-tuning orchestration.
- [x] (2026-04-19 17:59 IST) Rewrote this plan around the current best hybrid OCR recipe, save-triggered background training, manuscript-specific checkpoint lineage, future-proof backend orchestration, structured edit telemetry, CUDA profiling, and the requirement that existing headless pre-commit checks must remain green.
- [x] (2026-04-20 10:06 IST) Extracted one canonical OCR active-learning recipe into `app/recognition/active_learning_recipe.py` and pointed the runtime and pre-commit config helpers at that shared source.
- [x] (2026-04-20 10:06 IST) Added a generic app-level job orchestrator and device-lease manager in `app/job_orchestrator.py` and `app/device_leases.py`.
- [x] (2026-04-20 10:06 IST) Added a per-manuscript OCR registry and page-revision ledger in `app/manuscript_ocr_registry.py`, including revision snapshots under each manuscript runtime root.
- [x] (2026-04-20 10:06 IST) Added save-triggered OCR active-learning jobs, recorded automatic promotion, and rebase detection/runtime queueing in `app/ocr_active_learning_runtime.py`.
- [x] (2026-04-20 10:06 IST) Replaced the single global OCR model assumption on the live route path with `app/ocr_model_manager.py` and manuscript-aware checkpoint selection in `app/app.py`.
- [x] (2026-04-20 10:06 IST) Added structured telemetry for node edits, edge edits, OCR text edits, save intents, checkpoint lineage, queue events, and active-learning outcomes in `app/telemetry.py` plus manuscript runtime logs.
- [x] (2026-04-20 10:06 IST) Added coarse profiling summaries and optional sampled CUDA traces in `app/profiling.py`.
- [x] (2026-04-20 10:06 IST) Kept the frontend changes minimal by adding the `Active Learning` toggle, status text, and explicit `saveIntent` wiring in `app/frontend/src/components/ManuscriptViewer.vue`.
- [x] (2026-04-22 10:23 IST) Added a configurable live-runtime sibling checkpoint strategy override so GUI-triggered OCR jobs now default to `best_norm_ED.pth` through `OCR_RUNTIME_SIBLING_CHECKPOINT_STRATEGY`, while leaving the CER-based selector available in the shared OCR code path.
- [x] (2026-04-22 15:02 IST) Removed the stale manuscript-local protected-set promotion guard from the live runtime so GUI-triggered OCR jobs now promote directly after training and registry writes, without carrying dead verifier-bank code paths.
- [ ] (2026-04-20 10:06 IST) Added new headless backend tests and reran both unchanged gates individually: `test_ci_e2e.py` and `app.tests.test_recognition_finetuning_precommit_e2e`; the only remaining redundant rerun is `scripts/run_precommit_eval.py`.

## Surprises & Discoveries

- Observation: the current save route already writes the exact artifacts that the OCR active-learning path will need later.
  Evidence: `app/app.py` route `save_correction(...)` already calls `generate_xml_and_images_for_page(...)`, which writes PAGE-XML and cropped line images before any background recognition is started.

- Observation: the current app already performs background OCR work after save, but it does so with an ad hoc thread and no persisted job state.
  Evidence: `app/app.py` starts `threading.Thread(...)` inside `save_correction(...)` when `runRecognition` is true.

- Observation: the current OCR runtime assumes one global loaded model, which is incompatible with manuscript-specific checkpoints.
  Evidence: `app/app.py` owns a single `OCR_GLOBAL_CONTEXT` and `get_ocr_context()` loads only `OCR_MODEL_PATH`, not a manuscript-selected checkpoint.

- Observation: the frontend already distinguishes between explicit save actions and background autosave behavior, and that distinction matters for active learning.
  Evidence: `app/frontend/src/components/ManuscriptViewer.vue` calls `saveModifications(true)` every 20 seconds in recognition mode, while explicit actions such as `saveCurrentPage()` and `saveAndGoNext()` call `saveModifications()` without the background flag.

- Observation: treating every autosave as a training trigger would create pathological churn.
  Evidence: the current recognition-mode autosave interval is 20 seconds, so blindly fine-tuning on every autosave would thrash the queue, waste GPU time, and make page revision lineage noisy.

- Observation: the current backend logs node edits but not the other human-effort metrics that matter for active learning.
  Evidence: `app/app.py` currently writes `node_corrections/<page>.json` with node counts, but there is no equivalent persisted log for edge edits, OCR text edits, save intents, checkpoint ids, or queue behavior.

- Observation: the current repository already has the low-level OCR training and evaluation primitives needed for live integration.
  Evidence: `app/recognition/active_learning.py` exposes `prepare_page_datasets(...)`, `run_checkpoint_on_prepared_pages(...)`, and `fine_tune_checkpoint_on_pages(...)`, while `app/tests/precommit_gate_config.py` and `app/tests/recognition_finetuning_config.py` already encode the trusted hybrid recipe.

- Observation: the existing save path overwrote the newest PAGE-XML in place, so safe rebase support required explicit revision snapshots rather than only a ledger row.
  Evidence: the implemented runtime now copies each non-duplicate saved page into `input_manuscripts/<manuscript>/active_learning/recognition/revisions/<page>/rev_<n>/` before queueing OCR work.

- Observation: backward compatibility for the existing GUI-free tests was preserved by treating a missing `activeLearningEnabled` field as false and a missing `saveIntent` field as `commit`.
  Evidence: the unchanged pretrained full-pipeline gate `app/tests/test_ci_e2e.py` still passed after the route wiring moved onto the new runtime modules.

## Decision Log

- Decision: the GUI flow should trigger OCR active learning automatically after a real page save instead of asking the user to click a separate "fine-tune now" button.
  Rationale: the requested workflow is page-by-page specialization with minimal UI churn. A manual promotion or fine-tune button would slow the operator and would not match the desired "save page N, benefit on page N+2" behavior.
  Date/Author: 2026-04-19 / Codex

- Decision: the frontend will expose `Active Learning` as a small toggle that defaults to on, but the backend must treat a missing `activeLearningEnabled` field as false for backward compatibility.
  Rationale: the real app should default to active learning, while existing tests and older clients should not accidentally start new behavior just because the backend changed.
  Date/Author: 2026-04-19 / Codex

- Decision: OCR fine-tuning will trigger only on commit saves, not on draft autosaves.
  Rationale: recognition-mode autosave exists for recoverability, not for model-lineage promotion. Training on every 20-second autosave would cause duplicate jobs and unstable provenance.
  Date/Author: 2026-04-19 / Codex

- Decision: every save still enters the orchestration pipeline, but the OCR branch will enqueue work only when the saved page revision includes usable text supervision.
  Rationale: layout-only saves should still be recorded for future CRAFT and GNN active learning, but OCR cannot fine-tune on a page that has polygons without corrected text.
  Date/Author: 2026-04-19 / Codex

- Decision: the local OCR model family remains the active-learning target even if the user chose Gemini for recognition on a given page.
  Rationale: corrected PAGE-XML text and line crops are still valuable supervision for the local OCR model. Engine choice for prediction and the model family being improved should be decoupled.
  Date/Author: 2026-04-19 / Codex

- Decision: manuscript checkpoints must promote automatically through a recorded rule, not through an untracked implicit swap and not through a new manual promotion button.
  Rationale: `VISION.md` requires that a newly trained model never silently replaces the active one without a recorded promotion rule. Automatic promotion is acceptable only when it is deterministic, logged, and recoverable.
  Date/Author: 2026-04-19 / Codex

- Decision: the live GUI runtime should hard-prefer `best_norm_ED.pth` after each OCR fine-tune step unless explicitly reconfigured back to the CER-based sibling selector.
  Rationale: the GUI request here is to skip the extra page-CER sibling-selection pass in the background training flow while keeping an explicit escape hatch for future reversal. Wiring the choice through `OCR_RUNTIME_SIBLING_CHECKPOINT_STRATEGY` keeps the runtime change narrow and reversible without deleting the selector implementation used elsewhere.
  Date/Author: 2026-04-22 / Codex

- Decision: the live GUI runtime should promote the trained candidate directly after a successful fine-tune step and registry write, without a manuscript-local protected-set guard.
  Rationale: the protected-set gate was later judged to be stale deployment logic for this runtime. The retained live behavior is now intentionally simple: train with the recorded recipe, select the sibling checkpoint according to the configured sibling strategy, then promote and mark the revision consumed.
  Date/Author: 2026-04-22 / Codex

- Decision: editing a page that has already been consumed into the promoted manuscript lineage must mark the manuscript as needing an OCR rebase.
  Rationale: if page 1 changes after checkpoints trained through page 5 already exist, the lineage is no longer semantically clean. The safe first implementation is to keep the current active checkpoint for inference, record divergence, and queue a rebuild from the nearest safe ancestor, typically the base checkpoint.
  Date/Author: 2026-04-19 / Codex

- Decision: the app needs one generic orchestration layer with job priorities and explicit device leases, even though only OCR active learning is implemented in the first pass.
  Rationale: the long-term pipeline will need GPU time for CRAFT, GNN, and OCR at different stages. Designing a narrow OCR-only worker now would force a second scheduler rewrite later.
  Date/Author: 2026-04-19 / Codex

- Decision: interactive inference must outrank background training, and background training may be canceled and requeued if an interactive GPU request arrives.
  Rationale: preserving the annotation experience matters more than squeezing every background training second out of the GPU. Requeueing a short 60-iteration OCR job is safer than blocking the user behind it.
  Date/Author: 2026-04-19 / Codex

- Decision: profiling must be two-tiered: coarse always-on metrics and sampled detailed CUDA traces.
  Rationale: the user wants bottleneck visibility during real digitization, but full tracing on every request would itself become a bottleneck. Always-on summaries plus sampled traces preserve observability without making the UI sluggish.
  Date/Author: 2026-04-19 / Codex

- Decision: the first pass should surface active-learning state through the existing save and page-load responses instead of adding a larger new management route or UI panel.
  Rationale: the request explicitly asked for minimal frontend churn. Reusing `save_correction(...)` and `get_page_prediction(...)` for the `AL: ...` status line kept the UI change small while still exposing manuscript-local runtime state.
  Date/Author: 2026-04-20 / Codex

## Outcomes & Retrospective

This plan is now implemented in a first-pass form. The backend has a shared production OCR recipe, a generic orchestrator plus GPU lease manager, a manuscript-local OCR registry with revision snapshots, save-triggered fine-tune and rebase jobs, manuscript-aware checkpoint loading for local OCR, structured telemetry, and coarse profiling. The frontend gained only the requested `Active Learning` toggle, status text, and explicit save-intent wiring. The live runtime now records which sibling checkpoint strategy it used for each OCR fine-tune step, currently defaulting that runtime-specific choice to `best_norm_ED.pth`, and it promotes directly after training.

The main lesson from implementation matched the earlier design expectation: the hard part was not the OCR trainer itself. The hard part was durable orchestration and provenance. The route changes were straightforward only after the registry, revision snapshotting, and queue/event plumbing existed.

The main remaining work is validation depth rather than feature absence. New focused unit coverage is in place and both unchanged gates still passed when rerun directly, but the two-phase launcher `scripts/run_precommit_eval.py` was not rerun because it would only repeat those same two checks.

## Context and Orientation

The current backend lives primarily in `app/app.py`. It currently:

- uploads manuscripts under `input_manuscripts/<manuscript>/`
- writes PAGE-XML and line images during `save_correction(...)`
- runs manual recognition through `/recognize-text`
- optionally starts background recognition threads after save
- logs node corrections only
- loads one global OCR checkpoint through `OCR_MODEL_PATH` and `OCR_GLOBAL_CONTEXT`

The current frontend lives in `app/frontend/src/components/ManuscriptViewer.vue`. It currently:

- toggles layout mode and recognition mode
- saves layout and text changes through `saveModifications(...)`
- sends `runRecognition` and `recognitionEngine`
- performs a recognition-mode autosave every 20 seconds
- exposes only "Auto-Recognize on Save", not active learning

The OCR training and evaluation primitives already exist in:

- `app/recognition/active_learning.py`
- `app/recognition/pagexml_line_dataset.py`
- `app/recognition/ocr_defaults.py`
- `app/tests/precommit_gate_config.py`
- `app/tests/recognition_finetuning_config.py`
- `app/tests/recognition_finetuning_experiment.py`

The current regression guards that must continue to pass live in:

- `app/tests/test_ci_e2e.py`
- `app/tests/test_recognition_active_learning_unit.py`
- `app/tests/test_recognition_finetuning_precommit_unit.py`
- `app/tests/test_recognition_finetuning_precommit_e2e.py`
- `scripts/run_precommit_eval.py`

Several terms in this plan are precise and must be implemented that way.

A "commit save" means an intentional user action that should create durable lineage. In this repository that includes `Save`, `Save & Next`, page-navigation saves, and mode-transition saves that the user explicitly requested. A "draft save" means recoverability-only autosave while the user is still editing. Draft saves must persist the page, but they must not enqueue OCR fine-tuning.

A "supervised page revision" means one saved page version whose PAGE-XML line polygons and Unicode text are both available. OCR active learning can only train on supervised revisions.

A "device lease" means the exclusive right for one job to use a scarce runtime resource, most importantly the GPU. Device leases must be explicit because CRAFT, GNN, and OCR will all eventually compete for the same GPU.

A "rebase" means rebuilding the manuscript-specific OCR lineage because an already-consumed earlier page was edited again after later checkpoints had been trained.

The best known OCR recipe remains the exact hybrid recipe summarized in `app/tests/logs/recognition_finetune_results_latest.json`. The GUI implementation must reuse those settings, but the live promotion rule will differ from the offline curve metric because the GUI does not have future-page ground truth during normal use.

## Plan of Work

Start by extracting the canonical OCR active-learning recipe into production code. The settings currently exist indirectly through `app/tests/precommit_gate_config.py` and `app/tests/recognition_finetuning_config.py`. That is acceptable for evaluation, but it is the wrong place for runtime code to depend on. Create a shared production module such as `app/recognition/active_learning_recipe.py` that defines the locked OCR recipe once. Then make both the runtime worker and the pre-commit config helpers import from that production definition. The tests should still own thresholds and datasets, but not a second copy of the runtime recipe.

Next, add a generic orchestration layer in the app root, not inside `app/recognition/`. The first-pass file split should be explicit:

- `app/job_orchestrator.py` for persisted job records, priorities, queueing, and worker coordination
- `app/device_leases.py` for GPU and CPU lease acquisition and release
- `app/manuscript_ocr_registry.py` for the per-manuscript OCR registry and page-revision ledger
- `app/ocr_active_learning_runtime.py` for the OCR-specific glue that turns saved page revisions into fine-tune, verify, promote, and rebase jobs
- `app/telemetry.py` for page-edit diffs, job events, and manuscript summaries
- `app/profiling.py` for coarse timing and GPU summaries plus optional detailed traces

Do not hide this state in memory. Each manuscript needs a durable OCR runtime directory under its own manuscript root, for example:

- `input_manuscripts/<manuscript>/active_learning/recognition/registry.json`
- `input_manuscripts/<manuscript>/active_learning/recognition/checkpoints/<checkpoint_id>/`
- `input_manuscripts/<manuscript>/active_learning/recognition/telemetry/page_events.jsonl`
- `input_manuscripts/<manuscript>/active_learning/recognition/telemetry/job_events.jsonl`
- `input_manuscripts/<manuscript>/active_learning/recognition/profiling/`

The registry must be the source of truth. It should record:

- the immutable base checkpoint
- the current active checkpoint
- the previous active checkpoint for rollback
- any in-flight candidate checkpoint
- lineage metadata describing which page revisions were consumed
- the most recent promoted page index or revision sequence
- whether the manuscript currently needs a rebase
- whether active learning is enabled for this manuscript
- the last successful promotion summary
- the queue state or enough information to reconstruct pending work after restart

The registry must also maintain a page-revision ledger. Every commit save should create or update a page-revision record that includes at least the page id, a monotonically increasing revision number, a content hash, whether text supervision is present, which engine produced the initial text, and whether that revision has already been consumed into the active OCR lineage. This ledger is what allows the backend to deduplicate repeated saves, to skip layout-only revisions for OCR training, and to detect when an already-consumed earlier page has changed and the manuscript now requires a rebase.

Once the registry and queue exist, rework the save flow in `app/app.py`. The current `save_correction(...)` route should keep writing PAGE-XML and line images immediately, because that keeps the user-visible behavior stable. After persistence succeeds, it should classify the save as `draft` or `commit`, compute structured edit metrics, update the manuscript registry, and enqueue downstream jobs without waiting for them to finish. Save latency must remain bounded by normal file I/O plus queue insertion, not by OCR training.

The save route must distinguish three important OCR cases. First, a layout-only commit save should persist the revision and maybe create future CRAFT/GNN training evidence, but it should not enqueue an OCR fine-tune because there is no text supervision yet. Second, a recognition commit save with text should enqueue one OCR active-learning step if active learning is enabled and this page revision has not already been consumed. Third, a draft autosave should update the draft page state and text recovery data, but it should not create OCR lineage or queue work. Add an explicit `saveIntent` field to the frontend payload so the backend does not have to infer this from timing or route shape.

The OCR active-learning runtime should model the per-manuscript sequence explicitly. If page 1 is the newest supervised commit and no manuscript-local checkpoint exists yet, the runtime fine-tunes from the base checkpoint on page 1 and promotes the result directly after a successful training step. If page 2 later becomes supervised, the runtime fine-tunes from the currently active page-1 checkpoint using the locked hybrid recipe on the new page plus up to 10 sampled history lines from earlier approved pages. Later pages continue the same pattern. Each job must record which parent checkpoint it started from, which page revision it added, and which history lines were replayed before promotion.

The promotion rule must be automatic and recorded. The live GUI does not use the offline `early_weighted_page_cer` curve metric directly because that metric depends on future held-out pages, and it no longer carries a manuscript-local verifier-bank gate. The runtime records the candidate checkpoint metadata, promotes that candidate after the successful fine-tune step, and marks the triggering revision as consumed into the lineage. This automatic promotion rule satisfies the repository constraint that a new model must not silently replace the active model without a recorded rule, while also keeping the frontend simple. There should be no manual promotion button in the first implementation.

The recognition runtime must stop assuming one global checkpoint. Replace `OCR_GLOBAL_CONTEXT` with a manuscript-aware recognition model manager that can answer "load the active checkpoint for manuscript X" deterministically. To control GPU memory pressure, the manager should keep only the currently needed local OCR model on the GPU and unload or replace it when another manuscript or checkpoint is requested. Recognition requests must always use the active checkpoint that was current when the request began. If a newer checkpoint is promoted while a recognition request is already running, the request should finish on its original checkpoint and the new checkpoint should only affect later requests.

This change also needs explicit handling for layout-mode and recognition-mode edge cases. When the user switches from layout mode into recognition mode, the backend should ensure the latest layout save has already written PAGE-XML and line crops before running recognition. If the user edits nodes or edges on a page that already had corrected text, that page now has a new revision and its previous OCR training evidence is stale. If that page was already part of the promoted lineage, mark the manuscript as needing a rebase, keep the current active checkpoint for inference until the rebuild finishes, and queue a background rebuild job from the base checkpoint through the current approved pages in order. The first implementation should favor correctness and traceability over clever partial reuse.

The backend orchestrator must be future-proof for the rest of the human-in-the-loop pipeline. Even though the first live implementation only adds OCR active learning, the job model must already support at least these families:

- bulk upload preprocessing such as CRAFT on page batches of 10
- graph construction and GNN inference after CRAFT results exist
- OCR inference for recognition mode
- OCR fine-tuning and OCR rebase work
- future CRAFT fine-tuning from node edits
- future GNN fine-tuning from edge edits

Do not hardcode "one OCR queue." Introduce job priorities instead. Interactive recognition must outrank background fine-tuning, and background fine-tuning must outrank bulk preprocessing. A good initial rule is:

- priority 0: user-blocking inference and page-open work
- priority 1: save-followup work needed for the next human step
- priority 2: OCR background fine-tuning and rebase work
- priority 3: bulk upload preprocessing and future low-priority rebuilds

Background OCR fine-tuning should not be allowed to bottleneck the user. The first implementation should therefore run fine-tune jobs in a child process or otherwise isolate them so they can be canceled cleanly if an interactive GPU job arrives. If cancellation happens, discard the partial candidate, record the interruption in telemetry, and requeue the exact same job from its original parent checkpoint rather than resuming from a half-trained intermediate state. This preserves the recipe semantics while keeping interactive behavior predictable.

Add telemetry and profiling in parallel with the orchestration work, not after it. The telemetry layer should record per page:

- save intent (`draft` or `commit`)
- nodes added and deleted
- edges added and deleted
- whether text changed
- OCR edit distance between the last machine prediction and the saved corrected text
- changed line count
- checkpoint id used for the prediction the user edited
- recognition engine used to obtain that prediction
- whether the page revision entered OCR active learning

This is what will let later evaluation answer the real question in `EVAL.md`: whether manual effort is dropping over time. The per-page OCR edit metric should compare the saved text against the exact prediction the user saw, not only against the previous XML contents, so the app needs to preserve the last recognition payload per page revision.

Profiling should record both queueing cost and actual model cost. Every OCR inference and OCR fine-tune job should emit a compact JSON summary with at least wall time, queue wait time, device, peak CUDA memory allocated, peak CUDA memory reserved, page counts or line counts, and whether the job was canceled or completed. In addition, `app/profiling.py` should support sampled `torch.profiler` traces when CUDA is available. Enable these traces for the first run of each job family per manuscript and under an explicit environment flag such as `ACTIVE_LEARNING_PROFILE_CUDA=1`. Save the traces under the manuscript profiling folder so real digitization runs can be inspected afterward without drowning the normal UX in trace overhead.

The frontend changes should stay intentionally small. In `app/frontend/src/components/ManuscriptViewer.vue`, add:

- an `Active Learning` toggle near the existing auto-recognition controls
- `localStorage` persistence for that toggle
- a `saveIntent` field in the save payload so the backend can distinguish draft autosave from commit save
- a small status line such as `AL: idle`, `AL: training page 7`, `AL: paused for OCR`, `AL: needs rebase`, or `AL: failed`

Do not add a new panel for candidate models, manual promotions, or profiling controls in this first pass. The user asked for minimal frontend changes, and the orchestration and telemetry complexity belongs in the backend.

Finally, update the documentation that must move with behavior. If this plan is implemented, `EVAL.md`, `VISION.md`, `AGENTS.md`, and the relevant test-doc comments must all be updated in the same pass so the repository's stated evaluation story matches the code. `EVAL.md` in particular should describe the new GUI-safe OCR active-learning telemetry and should explicitly distinguish the live manuscript-local direct-promotion runtime from the fixed surrogate pre-commit gate. The pre-commit gate remains the headless code-regression guard; it is not replaced by the GUI runtime.

## Concrete Steps

From the repository root, inspect the current save, recognition, and OCR active-learning touchpoints before editing:

    Get-Content app\app.py
    Get-Content app\frontend\src\components\ManuscriptViewer.vue
    Get-Content app\recognition\active_learning.py
    Get-Content app\tests\precommit_gate_config.py
    Get-Content app\tests\recognition_finetuning_config.py

Expected result: you can trace the current save flow, the current autosave behavior, the current single-global-model OCR assumption, and the current hybrid recipe definition.

Implement the production recipe extraction first. After that change, confirm that both runtime and tests read the same recipe source:

    C:\Users\intro\miniconda3\envs\gnn_layout\python.exe -m unittest app.tests.test_recognition_active_learning_unit -v
    C:\Users\intro\miniconda3\envs\gnn_layout\python.exe -m unittest app.tests.test_recognition_finetuning_precommit_unit -v

Expected result: the OCR utility tests still pass, and the pre-commit config test proves that the trusted runtime recipe has not drifted.

After the orchestration, registry, and telemetry modules exist, add new focused unit coverage. A reasonable first split is:

- `app/tests/test_job_orchestrator_unit.py`
- `app/tests/test_manuscript_ocr_registry_unit.py`
- `app/tests/test_recognition_active_learning_backend_unit.py`
- `app/tests/test_recognition_telemetry_unit.py`

Run them with:

    C:\Users\intro\miniconda3\envs\gnn_layout\python.exe -m unittest app.tests.test_job_orchestrator_unit -v
    C:\Users\intro\miniconda3\envs\gnn_layout\python.exe -m unittest app.tests.test_manuscript_ocr_registry_unit -v
    C:\Users\intro\miniconda3\envs\gnn_layout\python.exe -m unittest app.tests.test_recognition_active_learning_backend_unit -v
    C:\Users\intro\miniconda3\envs\gnn_layout\python.exe -m unittest app.tests.test_recognition_telemetry_unit -v

Expected result: these tests prove, without the GUI, that commit saves enqueue the right jobs, draft saves do not, the registry promotes and rolls back cleanly, rebase is detected when an older page changes, and telemetry captures edit-distance metrics and checkpoint ids.

After the backend tests are passing, rerun the existing pre-commit gates unchanged:

    $env:CONDA_NO_PLUGINS='true'
    conda run -n gnn_layout python -m unittest discover -s app/tests -p "test_ci_e2e.py" -v

    $env:CONDA_NO_PLUGINS='true'
    conda run -n gnn_layout python -m unittest app.tests.test_recognition_finetuning_precommit_e2e -v

    $env:CONDA_NO_PLUGINS='true'
    conda run -n gnn_layout python scripts/run_precommit_eval.py

Expected result: the pretrained full-pipeline gate and the surrogate OCR pre-commit gate still pass without requiring the GUI or a live active-learning worker.

Then run one manual end-to-end app scenario with a small manuscript:

1. Start the Flask backend and frontend.
2. Open a manuscript with `Active Learning` enabled.
3. Correct page 1 in recognition mode and click `Save & Next`.
4. Confirm that the save returns immediately, the next page opens, and a background OCR training job appears in the manuscript telemetry or status.
5. While page 2 is being annotated, confirm that page-1 fine-tuning runs in the background.
6. After page-1 promotion completes, open page 3 recognition and confirm that the backend reports the page-1-derived checkpoint id as the active inference checkpoint.
7. Close the app, reopen it, and reopen the same manuscript. Confirm that the same promoted checkpoint is loaded.
8. Edit page 1 again after later pages exist. Confirm that the manuscript is marked `needs rebase`, that the current active checkpoint remains stable for inference until rebuild, and that a rebuild job is queued.

To inspect profiling output during that manual run, enable detailed CUDA traces once:

    $env:ACTIVE_LEARNING_PROFILE_CUDA='1'

Expected result: the manuscript profiling directory gains compact summary JSON files for OCR inference and OCR fine-tuning, plus sampled detailed traces when CUDA is available.

## Validation and Acceptance

Acceptance for this plan is behavioral.

First, the frontend change is minimal. The app gains an `Active Learning` option that defaults to on and a small status display, but it does not gain a large new training-management screen.

Second, commit saves must be cheap. `save_correction(...)` must still write PAGE-XML and line images and return promptly. OCR fine-tuning must happen after the save in the background, not inline with the request.

Third, the OCR active-learning path must use the exact trusted hybrid recipe listed at the top of this plan. The runtime and the pre-commit harness must share one canonical recipe definition so they cannot drift silently.

Fourth, manuscript-specific sequential behavior must work. After page 1 is corrected and saved, the manuscript gets a promoted checkpoint derived from page 1. Page 3 recognition uses that checkpoint. After page 2 is corrected and saved, the next promoted checkpoint becomes the one used on page 4, and so on.

Fifth, reopening the app must not lose manuscript progress. When the user comes back later, the backend must load the best promoted checkpoint for that manuscript automatically. If the latest active checkpoint is missing, the backend must fall back deterministically to the previous active checkpoint or the base checkpoint and record the degradation.

Sixth, layout-mode and recognition-mode edge cases must be handled explicitly. Layout-only saves must not try to fine-tune OCR without text supervision. Recognition-mode autosaves must not create OCR lineage. Switching back to recognition after layout edits must use the latest saved PAGE-XML and the current active checkpoint.

Seventh, edits to already-consumed earlier pages must trigger safe rebase behavior. The app must detect the divergence, record it, preserve current inference stability, and queue a rebuild rather than pretending the old lineage is still clean.

Eighth, interactive recognition must outrank background training. Manual recognition requests and page-open OCR work must not sit behind a long background fine-tune job. Background training may be canceled and requeued to preserve UX.

Ninth, the implementation must produce the human-effort telemetry needed by `EVAL.md`. For every page, the repository must be able to answer how many nodes and edges were edited, how much recognized text was changed, which checkpoint produced the original prediction, and whether edits are trending downward over the manuscript.

Tenth, the implementation must emit coarse performance and resource metrics plus sampled CUDA traces so future work can identify whether bottlenecks are in CRAFT, GNN, OCR inference, OCR fine-tuning, queue delays, memory pressure, or upload-time batch preprocessing.

Eleventh, the existing GUI-free pre-commit gates must still pass. This work must not break `test_ci_e2e.py`, the OCR active-learning unit tests, the surrogate OCR pre-commit gate, or `scripts/run_precommit_eval.py`.

## Idempotence and Recovery

Registry writes must be atomic. Write to a temporary file and replace the target only after the new content is complete. The active checkpoint pointer must never reference a checkpoint directory that does not exist on disk.

Commit saves must be deduplicated by page revision hash. Saving the same recognized text and layout twice should not enqueue duplicate OCR jobs.

Draft saves are repeatable and safe. They may refresh recovery state, but they must not mutate manuscript lineage or promotion state.

If the app crashes during a background OCR fine-tune, the partial candidate must remain unpromoted. On restart, the registry scan should mark the interrupted job accordingly and requeue it from the original parent checkpoint if the triggering page revision is still the newest unconsumed supervised revision.

If a candidate training step finishes but the promotion write fails, the registry must leave the old active checkpoint unchanged and record the candidate as a promotion failure without silently advancing the active checkpoint.

If the user disables `Active Learning` mid-manuscript, the backend should stop enqueueing new OCR jobs for that manuscript but must not delete existing checkpoints or telemetry. Re-enabling it later should continue from the last valid active checkpoint and the current page-revision ledger.

If profiling is unavailable or CUDA is absent, the coarse profiling JSON should still be written with CPU-only fields and an explicit `cuda_available=false` marker rather than failing the page workflow.

## Artifacts and Notes

The key current sources of truth for the OCR recipe and evaluation constraints are:

- `EVAL.md`
- `VISION.md`
- `app/tests/precommit_gate_config.py`
- `app/tests/recognition_finetuning_config.py`
- `app/tests/logs/recognition_finetune_results_latest.json`
- `app/tests/logs/recognition_finetune_precommit_latest.json`

The runtime artifacts added by this plan should stay inside each manuscript root so they move with the manuscript and stay within the repository:

- `input_manuscripts/<manuscript>/active_learning/recognition/registry.json`
- `input_manuscripts/<manuscript>/active_learning/recognition/checkpoints/`
- `input_manuscripts/<manuscript>/active_learning/recognition/telemetry/page_events.jsonl`
- `input_manuscripts/<manuscript>/active_learning/recognition/telemetry/job_events.jsonl`
- `input_manuscripts/<manuscript>/active_learning/recognition/telemetry/page_edit_summary.json`
- `input_manuscripts/<manuscript>/active_learning/recognition/profiling/`

The profiling summaries should make it possible to answer these concrete questions after a real manuscript run:

- How much time did OCR inference spend waiting on the queue?
- How much time did OCR fine-tuning spend training versus verifying?
- Which jobs were canceled because the user needed the GPU?
- What were peak CUDA memory usage and batch sizes?
- Are OCR text-edit counts per page falling as the manuscript progresses?

The orchestrator is deliberately broader than the first OCR-only implementation because the long-term pipeline is:

1. upload manuscript
2. run CRAFT on page batches
3. construct graphs and run GNN inference
4. let the human correct nodes and edges
5. save PAGE-XML and recognition edits
6. run OCR active learning now
7. later add CRAFT and GNN active learning on the same orchestration substrate

## Interfaces and Dependencies

The exact file names may vary slightly, but the end state must provide stable interfaces equivalent to the following.

In `app/recognition/active_learning_recipe.py`, define a shared production recipe object or dataclass equivalent to:

    class OcrActiveLearningRecipe:
        training_policy: str = "page_plus_random_history"
        history_sample_line_count: int = 10
        width_policy: str = "batch_max_pad"
        oversampling_policy: str = "none"
        augmentation_policy: str = "none"
        lr_scheduler: str = "none"
        optimizer: str = "adadelta"
        lr: float = 0.2
        num_iter: int = 60
        curve_metric: str = "early_weighted_page_cer"
        regression_guard_abs: float = 0.005
        background_plus_rotation_variant_count: int = 10
        shuffle_train_each_epoch: bool = True

In `app/job_orchestrator.py`, define explicit job types and states, for example:

    class JobType(str, Enum):
        OCR_INFER = "ocr_infer"
        OCR_FINE_TUNE = "ocr_fine_tune"
        OCR_REBASE = "ocr_rebase"
        CRAFT_BATCH_INFER = "craft_batch_infer"
        GNN_PAGE_INFER = "gnn_page_infer"

    class JobPriority(IntEnum):
        INTERACTIVE = 0
        SAVE_FOLLOWUP = 1
        BACKGROUND_TRAINING = 2
        BULK_PREPROCESS = 3

The orchestrator must provide stable operations equivalent to:

    enqueue(job: QueuedJob) -> str
    cancel(job_id: str, reason: str) -> None
    start_workers() -> None
    shutdown_workers() -> None
    get_job_status(job_id: str) -> dict

In `app/device_leases.py`, provide a small lease manager equivalent to:

    acquire(resource_name: str, owner: str, priority: int) -> LeaseToken
    release(token: LeaseToken) -> None
    current_owner(resource_name: str) -> str | None

In `app/manuscript_ocr_registry.py`, provide stable methods equivalent to:

    load_registry(manuscript_root: Path) -> ManuscriptOcrRegistry
    record_page_revision(page_id: str, revision_payload: dict) -> PageRevision
    active_checkpoint() -> Path
    mark_candidate(candidate_payload: dict) -> None
    promote_candidate(candidate_id: str, promotion_summary: dict) -> None
    mark_rebase_needed(reason: str, changed_page_id: str) -> None
    pending_ocr_work() -> list[dict]

In `app/ocr_active_learning_runtime.py`, provide stable glue functions equivalent to:

    handle_post_save(manuscript: str, page: str, save_intent: str, active_learning_enabled: bool, recognition_engine: str, text_payload: dict | None) -> dict
    run_ocr_finetune_job(job_payload: dict) -> dict
    rebuild_manuscript_lineage(job_payload: dict) -> dict

These functions must reuse the existing OCR helpers in `app/recognition/active_learning.py`, especially:

- `prepare_page_datasets(...)`
- `run_checkpoint_on_prepared_pages(...)`
- `fine_tune_checkpoint_on_pages(...)`

In `app/telemetry.py`, define a stable helper equivalent to:

    compute_text_edit_metrics(predicted_lines: dict[str, str], saved_lines: dict[str, str]) -> dict

The result must include at least total edit distance, changed line count, and per-line diffs.

In `app/profiling.py`, define helpers equivalent to:

    summarize_gpu_job(job_name: str, metadata: dict, fn: Callable[[], T]) -> tuple[T, dict]
    maybe_write_cuda_trace(job_name: str, output_dir: Path, enabled: bool, fn: Callable[[], T]) -> T

In `app/app.py`, the routes must keep their existing HTTP purpose but gain the new orchestration semantics. `save_correction(...)` must remain the save entrypoint. `recognize_text(...)` must remain the manual OCR entrypoint. Both should delegate to the new orchestration and model-manager modules rather than continuing to own ad hoc global OCR state directly.

Revision note, 2026-04-19 17:59 IST: this ExecPlan was rewritten to incorporate the requested save-triggered active-learning workflow, the exact best hybrid OCR recipe, manuscript-specific checkpoint lineage, restart-safe automatic promotion, future-proof orchestration for CRAFT/GNN/OCR resource contention, structured edit telemetry, CUDA profiling, and the requirement that the existing headless pre-commit checks remain unaffected.

Revision note, 2026-04-20 10:06 IST: this ExecPlan was updated after implementation to record the shipped first-pass runtime, the added backend/test files, the revision-snapshot discovery, the preserved backward-compatibility contract, and the exact validation that was completed versus still pending.


