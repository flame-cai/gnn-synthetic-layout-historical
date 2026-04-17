# guided hyper parameter search
- what were session 1 hyperparams
- learning rate trajectories over pages
- have two evals before every commit! one pretrained one active learning
- padding rules for short lines? no resize? what's the best way?
- recheck that text-line segmentation is consistent between the tool and evaluation experiment
- instead of 5 pages, do 10 pages experiment, with rolling window average lineant test
- simplify, abstract, find invariants, while maintaining the exact the same functioning!
- make it very rapidly iterative at the line level instead of page level in milestone 5. never abruptly update a text-line which the user is editing, or has editing. But you can update the lines not touched by the user on the same page! OLD: Add a new backend option such as `activeLearningEnabled` to the save payload. When it is false, the current behavior should remain unchanged. When it is true and the recognition engine is local OCR, the save path should enqueue a background fine-tune job that uses the corrected page plus the replay buffer to produce a candidate checkpoint. When the job completes, it should run the promotion rule from Milestone 2. Only if promotion succeeds should the new checkpoint become active for later pages. SAMPLE THE BAD PERFORMING LINES PROPORTIONALLY WITH THE DATASET.
- be selective using confidence scores - oversample a bit
- make an actual GUI test, where we just copy paste GT, to simulat human corrections..





# Local OCR Active Learning Next Steps

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This document must be maintained in accordance with [PLANS.md](/c:/Users/intro/OneDrive/Documents/MEGA/CAI-FLAME/gnn-synthetic-layout-historical/PLANS.md). It incorporates the current implementation state and evidence recorded in [docs/exec-plans/active/recognition-finetuning-session-report-2026-04-16.md](/c:/Users/intro/OneDrive/Documents/MEGA/CAI-FLAME/gnn-synthetic-layout-historical/docs/exec-plans/active/recognition-finetuning-session-report-2026-04-16.md), but it repeats all necessary context here so a new contributor can execute it without reading any other plan first.

## Purpose / Big Picture

After this work, the local OCR path will behave like a real active-learning system instead of a one-off fine-tuning script. A user will be able to correct page text, save the page, let the repository fine-tune the local OCR model safely in the background, and then see better recognition on later pages without risking silent model regressions. The proof will not be only a lower character error rate on one fixed split. The proof will be that page-by-page correction effort drops over time under a controlled sequential experiment, with saved artifacts that explain exactly why a newly trained model was accepted or rejected.

The immediate goal is not to add more user interface. The immediate goal is to make the current OCR update loop trustworthy. Once the update loop is trustworthy, the next goal is to make it selective so we train on the most useful corrections instead of retraining blindly after every save. Once it is selective, the next goal is to make it automatic and measurable inside the app.

## Progress

- [x] (2026-04-16 23:05 IST) Reviewed the current OCR fine-tuning state, the sequential evaluator artifacts, `EVAL.md`, `VISION.md`, and the active session report.
- [x] (2026-04-16 23:10 IST) Wrote this proposed blocker-first roadmap for the next phase of OCR active learning.
- [ ] Implement a character-error-rate-aligned checkpoint selector and rerun the sequential evaluator until the step-0 to step-5 curve passes monotonically on `eval_dataset`.
- [ ] Add model-promotion guardrails so a newly fine-tuned checkpoint only becomes the active OCR model if it beats the current active model on a reserved internal validation set.
- [ ] Add structured active-learning telemetry so every correction, save, fine-tune, model promotion, and next-page inference can be measured as human-effort data instead of only accuracy data.
- [ ] Add budget-aware sample selection and replay policies so the system trains on the most informative corrected lines instead of replaying all past lines forever.
- [ ] Integrate the proven update loop into the app as an opt-in background process after page save, without blocking the user and without swapping checkpoints mid-page.
- [ ] Expand evaluation from “does CER drop on one dataset?” to “does manual correction burden fall page-by-page across datasets and annotation budgets?”

## Surprises & Discoveries

- Observation: the current OCR fine-tuning path is much closer to working than the raw failing test suggests.
  Evidence: the latest full run on `eval_dataset` reaches step 4 with a descending selected-checkpoint curve through step 3, and the only failure is that the selector chose a weaker step-4 checkpoint.

- Observation: the dominant blocker is now checkpoint selection, not PAGE-XML parsing, data conversion, or fine-tune orchestration.
  Evidence: in `app/tests/logs/20260416_222641_recognition_finetune_eval_eval_dataset`, step 4 failed because `best_norm_ED.pth` produced page CER `0.227907`, while the sibling `best_accuracy.pth` produced `0.216913`, which would have preserved monotonic improvement.

- Observation: cumulative replay is necessary for OCR active learning in this repository.
  Evidence: earlier page-only sequential tuning failed sooner, while cumulative replay improved the curve to step 4 before checkpoint selection became the only blocker.

- Observation: a naive held-out validation split hurts the earliest fine-tune steps because the corpus is extremely small.
  Evidence: the 20 percent hold-out experiment failed at step 1 on only 23 line samples. This means the repository needs a smarter model-selection set than “remove a random fifth of the newest tiny corpus from training.”

- Observation: the app already has a background-recognition pattern that can be reused for background fine-tuning later, but it currently lacks model-version safety.
  Evidence: [app/app.py](/c:/Users/intro/OneDrive/Documents/MEGA/CAI-FLAME/gnn-synthetic-layout-historical/app/app.py:674) already starts background recognition threads after save, but there is no atomic model registry or promotion rule for local OCR checkpoints.

## Decision Log

- Decision: treat checkpoint selection as the first implementation milestone and do not add GUI-triggered fine-tuning until the offline sequential evaluator passes.
  Rationale: a background feature that occasionally promotes a worse checkpoint would erode trust immediately. The repository already has enough evidence that the remaining risk is post-training model selection, so the safest path is to fix that before exposing more automation.
  Date/Author: 2026-04-16 / Codex

- Decision: keep the short-term active-learning policy cumulative and page-sequential, but make the medium-term policy budget-aware and selective.
  Rationale: cumulative replay is the simplest policy that currently works. However, replaying every corrected line forever is not scalable and does not reflect active learning well. The system should first become reliable under cumulative replay, then graduate to more selective update policies.
  Date/Author: 2026-04-16 / Codex

- Decision: optimize future model promotion around character error rate, not around internal training metrics such as normalized edit distance alone.
  Rationale: the user-facing outcome is fewer text corrections. Character error rate is directly tied to that burden in this repository, while the current internal checkpoint metrics have already selected the wrong model.
  Date/Author: 2026-04-16 / Codex

- Decision: delay any “train after every save” default until the repository can prove bounded latency, atomic checkpoint promotion, and monotonic next-page benefit on offline experiments.
  Rationale: active learning is only useful if it reduces human effort without making the tool unpredictable or slow between pages.
  Date/Author: 2026-04-16 / Codex

## Outcomes & Retrospective

This is a proposed plan, so no new code is implemented by this document. The main outcome of the current research state is clarity: the repository no longer needs broad exploratory OCR work first. The next work is well-scoped. The offline fine-tuning path exists, the sequential evaluator exists, and the next phase is to make model selection, model promotion, and manual-effort measurement robust enough for real active learning.

The most important lesson so far is that active learning here is not mainly a question of “can we fine-tune the model?” The repository can already fine-tune the model. The real question is whether the system can decide when to fine-tune, what to fine-tune on, which resulting model to trust, and whether that trust leads to less human correction on later pages. That is the lens for every milestone below.

## Context and Orientation

The local OCR model lives under `app/recognition/`. The pretrained checkpoint is [app/recognition/pretrained_model/vadakautuhala.pth](/c:/Users/intro/OneDrive/Documents/MEGA/CAI-FLAME/gnn-synthetic-layout-historical/app/recognition/pretrained_model/vadakautuhala.pth). The current package-safe OCR defaults and checkpoint helpers live in [app/recognition/ocr_defaults.py](/c:/Users/intro/OneDrive/Documents/MEGA/CAI-FLAME/gnn-synthetic-layout-historical/app/recognition/ocr_defaults.py:1). The PAGE-XML crop-preparation path lives in [app/recognition/pagexml_line_dataset.py](/c:/Users/intro/OneDrive/Documents/MEGA/CAI-FLAME/gnn-synthetic-layout-historical/app/recognition/pagexml_line_dataset.py:1). The fine-tune orchestration and prediction generation live in [app/recognition/active_learning.py](/c:/Users/intro/OneDrive/Documents/MEGA/CAI-FLAME/gnn-synthetic-layout-historical/app/recognition/active_learning.py:1).

The current evaluator for OCR fine-tuning lives in [app/tests/recognition_finetuning_experiment.py](/c:/Users/intro/OneDrive/Documents/MEGA/CAI-FLAME/gnn-synthetic-layout-historical/app/tests/recognition_finetuning_experiment.py:1), with dataset settings in [app/tests/recognition_finetuning_config.py](/c:/Users/intro/OneDrive/Documents/MEGA/CAI-FLAME/gnn-synthetic-layout-historical/app/tests/recognition_finetuning_config.py:1) and the slow test entrypoint in [app/tests/test_recognition_finetuning_e2e.py](/c:/Users/intro/OneDrive/Documents/MEGA/CAI-FLAME/gnn-synthetic-layout-historical/app/tests/test_recognition_finetuning_e2e.py:1).

The page-level OCR metric used by this repository is character error rate, abbreviated as CER. Character error rate is the edit distance between predicted and ground-truth text divided by the ground-truth text length. In plain language, it measures how many character insertions, deletions, or substitutions are needed to correct the OCR output. The repository’s current PAGE-XML evaluation logic lives in [app/tests/evaluate.py](/c:/Users/intro/OneDrive/Documents/MEGA/CAI-FLAME/gnn-synthetic-layout-historical/app/tests/evaluate.py:1).

An active-learning system in this repository means the following concrete loop. The user corrects OCR mistakes on one page. Those corrections become new supervised training examples. The repository fine-tunes the local OCR model on those corrections, with some replay from older corrected examples to avoid forgetting. Then the tool uses the improved checkpoint on later pages and ideally requires fewer corrections. That “fewer corrections” outcome must be measured, not assumed.

There are four terms of art that matter in this plan:

An acquisition policy is the rule that decides which corrected examples should be used for the next update. In this repository, that could mean “use every corrected line from the saved page,” “use only the lines that had low OCR confidence,” or “use a balanced subset of hard and representative lines.”

A replay buffer is the saved pool of earlier corrected examples that are mixed into later fine-tuning runs so the model does not forget older handwriting patterns. In this repository, the replay buffer will be a versioned on-disk collection of corrected line images plus text labels.

A checkpoint selector is the rule that chooses which trained checkpoint from a fine-tuning run becomes the candidate model. The current repository writes multiple checkpoints such as `best_accuracy.pth` and `best_norm_ED.pth`, but the wrong one is sometimes chosen.

A promotion rule is the rule that decides whether the candidate model is allowed to replace the currently active model. This rule must be stricter than “training finished successfully.” It should require beating the active model on a reserved validation set and should leave the current model unchanged if the candidate does not pass.

## Plan of Work

The work should proceed in six milestones, in order, without skipping ahead. The order matters because later active-learning features depend on the earlier ones being trustworthy.

### Milestone 1: Make checkpoint selection CER-aligned and robust

The first edit area is [app/recognition/active_learning.py](/c:/Users/intro/OneDrive/Documents/MEGA/CAI-FLAME/gnn-synthetic-layout-historical/app/recognition/active_learning.py:1). Extend the fine-tuning result model so each training run records every candidate checkpoint that was written, not just the one currently selected. Add a post-training evaluation function that can run OCR inference over a small selection corpus and compute true character error rate for each candidate checkpoint.

This selection corpus must not be the evaluation pages from `app/tests/eval_dataset`. It should be an internal validation reservoir built from corrected lines already available to the update loop. In the short term, when the reservoir is tiny, it is acceptable to evaluate candidate checkpoints on a deterministic subset of the replay corpus even if those lines were also used for training, because the goal at this stage is ranking sibling checkpoints from the same run, not publishing a research number. What matters is that the selector becomes CER-aligned and deterministic.

Add a helper module, preferably `app/recognition/model_selection.py`, if `active_learning.py` becomes too large. This module should expose one function that receives a list of checkpoint paths plus a prepared dataset and returns the selected checkpoint, the per-checkpoint CER scores, and the ranking rationale. Update `fine_tune_metadata.json` to record the full comparison, not just the chosen file name.

Then update [app/tests/recognition_finetuning_experiment.py](/c:/Users/intro/OneDrive/Documents/MEGA/CAI-FLAME/gnn-synthetic-layout-historical/app/tests/recognition_finetuning_experiment.py:1) so the step record includes the selector’s evidence. The summary and metrics files should say whether `best_accuracy.pth` or `best_norm_ED.pth` was chosen and why.

The acceptance for this milestone is that the slow test on `eval_dataset` passes step 0 through step 5 monotonically with the current cumulative schedule, or else fails later for a different, now well-explained reason. If the run still fails, the artifacts must show the CER score of each candidate checkpoint so the next change can be driven by evidence instead of by guesswork.

### Milestone 2: Add model registry and promotion guardrails

Once checkpoint selection is fixed, the next risk is unsafe promotion. Right now there is no formal distinction between a candidate model produced by a background fine-tune and the active model the app should trust. Introduce that distinction explicitly.

Add a small model-registry layer under `app/recognition/`, preferably a new file such as `app/recognition/model_registry.py`. This registry should define the active OCR model, the current candidate model, and the metadata needed to explain where each came from. Store the registry as a JSON document in a predictable location inside the app data area or a repository-controlled local path used by the tests.

The registry should support four operations: read the currently active model, register a new candidate checkpoint, promote a candidate to active if it passes the promotion rule, and roll back to the previous active checkpoint if something later fails. The promotion rule should compare the candidate model against the current active model on a reserved internal validation reservoir, using the same CER evaluator introduced in Milestone 1.

The app should never load a half-written checkpoint. Use atomic file replacement for the registry update. The easiest safe pattern on Windows is to write a temporary JSON file and rename it into place only after the candidate has passed validation.

The acceptance for this milestone is an automated test that simulates an active model plus a candidate model, verifies that a worse candidate is rejected, verifies that a better candidate is promoted, and verifies that the active-model pointer does not change when promotion fails.

### Milestone 3: Add active-learning telemetry and correction-burden logging

Active learning in this repository is not only about model quality. It is about reducing total human effort. That means the repository must log the effort itself.

Add structured event logging to the save and recognition flows in [app/app.py](/c:/Users/intro/OneDrive/Documents/MEGA/CAI-FLAME/gnn-synthetic-layout-historical/app/app.py:598), especially around `save_correction`, local OCR recognition, and any future fine-tune trigger. Every event should write a small JSON record that includes manuscript name, page id, recognition engine, active checkpoint id, whether recognition ran, whether fine-tuning ran, how long it took, and how many lines were changed.

For OCR-specific correction burden, the save path should record at least the number of corrected lines, the sum of per-line edit distances between pre-save OCR text and post-save corrected text, and the number of save cycles needed for that page. If the frontend already sends enough information to infer these counts, compute them directly. If not, extend the saved payload minimally and document the new fields.

Create a new helper module such as `app/recognition/active_learning_metrics.py` so the burden calculations are not buried inside Flask route functions. The helper should also compute line-length buckets and confidence buckets, because later acquisition policies will need those.

The acceptance for this milestone is a scripted test that saves a corrected page, verifies that a structured telemetry record was written, and shows that the record contains the active model id, corrected-line count, edit-distance summary, and timestamps for save and recognition. This milestone does not yet require the GUI to show these numbers, only that the backend captures them reliably.

### Milestone 4: Introduce budget-aware acquisition and replay policies

Once the update loop is trustworthy and measurable, stop treating all corrected lines equally. This is where the repository becomes active-learning-aware in a meaningful sense.

Add a new configuration block in [app/tests/recognition_finetuning_config.py](/c:/Users/intro/OneDrive/Documents/MEGA/CAI-FLAME/gnn-synthetic-layout-historical/app/tests/recognition_finetuning_config.py:1) for acquisition policy and replay policy. Keep the default policy conservative and transparent. The first recommended policy set is:

The acquisition policy should remain “use all corrected lines from the newest saved page” for the first online implementation, because that matches user expectations and avoids a hidden “why was my correction ignored?” problem.

The replay policy should become bounded instead of unbounded. Use a fixed maximum replay size in lines, not in pages, and sample from the replay buffer in a balanced way across page ids, line lengths, and prior confidence buckets. The reason is straightforward: otherwise a long manuscript can overfit to whichever early pages happened to accumulate the most corrected lines.

After that conservative policy works, add one experimental acquisition strategy for research comparison: “priority replay,” meaning replay more examples whose original OCR confidence was low or whose prior correction edit distance was high. In plain language, this means the system should spend more training budget on the lines it used to get most wrong.

Store the per-line metadata needed for this policy inside the replay manifest. That means each corrected line in the replay buffer should carry at least its page id, label text, line image path, original OCR prediction, original OCR confidence, and edit distance to the correction.

The acceptance for this milestone is not just passing tests. It is an offline experiment that compares at least three policies on the same page sequence: cumulative replay-all, bounded balanced replay, and bounded priority replay. The output should be a budget-versus-CER plot and a budget-versus-correction-burden plot saved under `app/tests/logs/`.

### Milestone 5: Integrate opt-in background OCR fine-tuning into the app

Only after Milestones 1 through 4 are working should the app begin background fine-tuning after page save. Reuse the existing background-thread style in [app/app.py](/c:/Users/intro/OneDrive/Documents/MEGA/CAI-FLAME/gnn-synthetic-layout-historical/app/app.py:674), but do not wire the feature directly into the current save path without a safety layer.

Add a new backend option such as `activeLearningEnabled` to the save payload. When it is false, the current behavior should remain unchanged. When it is true and the recognition engine is local OCR, the save path should enqueue a background fine-tune job that uses the corrected page plus the replay buffer to produce a candidate checkpoint. When the job completes, it should run the promotion rule from Milestone 2. Only if promotion succeeds should the new checkpoint become active for later pages.

Do not replace the local OCR model for the page that is currently open. The point of the background update is to improve later pages, not to change the current page out from under the user. The safest behavioral rule is: “a newly promoted model applies only to future recognition requests.”

Add a small status file or registry field so the app can report whether background fine-tuning is idle, running, failed, or promoted. This should be enough for the frontend to show a future status indicator, even if no frontend change is made in this milestone.

The acceptance for this milestone is a backend test that simulates a save with `activeLearningEnabled=true`, verifies that a background job writes a candidate checkpoint and registry update, verifies that promotion occurs only after validation passes, and verifies that a subsequent local OCR request uses the promoted checkpoint instead of the old one.

### Milestone 6: Expand evaluation into real active-learning experiments

The final milestone turns the OCR fine-tune evaluator from a useful regression test into a real active-learning research harness.

Extend [app/tests/recognition_finetuning_experiment.py](/c:/Users/intro/OneDrive/Documents/MEGA/CAI-FLAME/gnn-synthetic-layout-historical/app/tests/recognition_finetuning_experiment.py:1) so it can compare multiple policies, multiple datasets, and multiple annotation budgets. A budget here means the maximum amount of human correction you allow before each update. In this repository, reasonable budget units are corrected lines, corrected characters, or corrected pages.

The minimal research matrix should include:

The current sequential full-page policy, where every corrected line from a page is used before moving on.

A bounded replay policy, where old lines are sampled from a fixed-size replay buffer.

A priority replay policy, where harder historical lines receive more replay weight.

At least two datasets once more are available, because active-learning conclusions based on one manuscript style are not trustworthy enough for product decisions.

The output should include the metrics emphasized in `EVAL.md`: page CER, line-level CER, fine-tune duration, next-page quality after fine-tune, and correction burden. It should also include plots that show how quickly quality improves as the annotation budget increases.

The acceptance for this milestone is that a human can open the generated experiment folder, inspect the JSON and summary files, and answer three concrete questions without reading the code: which policy was used, how much correction budget it consumed, and whether the next-page burden fell over time.

## Concrete Steps

Work from the repository root `c:\Users\intro\OneDrive\Documents\MEGA\CAI-FLAME\gnn-synthetic-layout-historical`. Use the requested environment:

    $env:CONDA_NO_PLUGINS='true'
    C:\Users\intro\miniconda3\envs\gnn_layout\python.exe -m unittest discover -s app/tests -p "test_recognition_finetuning_e2e.py" -v

Run that command before Milestone 1 changes to confirm the current known failure. The expected failure at the time of writing is a step-4 monotonicity failure in the latest OCR fine-tuning run.

For Milestone 1, the main cycle should be:

    $env:CONDA_NO_PLUGINS='true'
    C:\Users\intro\miniconda3\envs\gnn_layout\python.exe -m unittest discover -s app/tests -p "test_recognition_finetuning_e2e.py" -v

Inspect the latest artifacts under `app/tests/logs/` after every run, especially:

    app/tests/logs/recognition_finetune_results_latest.md
    app/tests/logs/recognition_finetune_results_latest.json

After Milestone 1, add a direct selector test command, for example:

    $env:CONDA_NO_PLUGINS='true'
    C:\Users\intro\miniconda3\envs\gnn_layout\python.exe -m unittest discover -s app/tests -p "test_recognition_model_selection.py" -v

This new test should compare sibling checkpoints on a prepared validation corpus and verify that the lower-CER checkpoint is selected.

For Milestones 2 and 3, add targeted backend tests and run:

    $env:CONDA_NO_PLUGINS='true'
    C:\Users\intro\miniconda3\envs\gnn_layout\python.exe -m unittest discover -s app/tests -p "test_recognition_model_registry.py" -v
    C:\Users\intro\miniconda3\envs\gnn_layout\python.exe -m unittest discover -s app/tests -p "test_active_learning_telemetry.py" -v

For Milestone 5, add a backend flow test that exercises save plus background fine-tune plus promotion. Run:

    $env:CONDA_NO_PLUGINS='true'
    C:\Users\intro\miniconda3\envs\gnn_layout\python.exe -m unittest discover -s app/tests -p "test_local_ocr_active_learning_background.py" -v

At every stopping point, update this proposed plan or move it to `docs/exec-plans/active/` when implementation begins, and append a short note at the bottom explaining what changed and why.

## Validation and Acceptance

The repository should not consider OCR active learning ready until all of the following behaviors are observable.

First, the offline sequential OCR experiment must pass on `eval_dataset`. A passing run means the saved summary under `app/tests/logs/` shows step 0 through step 5 with a monotonically descending aggregate page CER curve.

Second, the checkpoint selector must be explainable. Given a fine-tune run with multiple saved checkpoints, the artifacts must record every candidate checkpoint, its internal validation CER, and the exact reason the winner was chosen.

Third, model promotion must be safe. A deliberately worse candidate checkpoint must be rejected without changing the active OCR model pointer. A better candidate must be promoted atomically.

Fourth, human-effort telemetry must exist. After saving a corrected page, the repository must be able to show how many lines were corrected, how many character edits were applied relative to the OCR output, and whether a fine-tune was triggered.

Fifth, the background active-learning path must be stable. A page save with active learning enabled must return immediately, start a background fine-tune job, and make any promoted checkpoint visible only to future pages, not the page already on screen.

Sixth, the research harness must compare at least one conservative baseline policy and one selective active-learning policy under the same annotation budget, with plots saved to disk.

## Idempotence and Recovery

All experiment runs must continue writing to timestamped directories under `app/tests/logs/` so reruns never overwrite old evidence. The only overwrite allowed is the lightweight “latest” pointer files already used by the evaluator.

The model registry must support safe recovery. If a candidate checkpoint fails validation, crashes during promotion, or later proves invalid, the active checkpoint path must remain unchanged. Every promoted checkpoint must retain a pointer to the previous active checkpoint so the repository can roll back intentionally without guessing.

Background jobs must be restart-safe. If the app restarts while a fine-tune is running, the next startup should either resume from a cleanly written candidate artifact or discard the incomplete candidate and keep the active model unchanged. Do not leave the registry in a half-promoted state.

## Artifacts and Notes

The most important current evidence lives here:

- [docs/exec-plans/active/recognition-finetuning-session-report-2026-04-16.md](/c:/Users/intro/OneDrive/Documents/MEGA/CAI-FLAME/gnn-synthetic-layout-historical/docs/exec-plans/active/recognition-finetuning-session-report-2026-04-16.md:1)
- [docs/exec-plans/active/recognition-finetuning-failure-log.md](/c:/Users/intro/OneDrive/Documents/MEGA/CAI-FLAME/gnn-synthetic-layout-historical/docs/exec-plans/active/recognition-finetuning-failure-log.md:1)
- [app/tests/logs/20260416_222641_recognition_finetune_eval_eval_dataset](/c:/Users/intro/OneDrive/Documents/MEGA/CAI-FLAME/gnn-synthetic-layout-historical/app/tests/logs/20260416_222641_recognition_finetune_eval_eval_dataset)

The single strongest short transcript from the current evidence is this:

    Step 3 selected checkpoint page CER: 0.220296
    Step 4 selected checkpoint page CER: 0.227907
    Step 4 sibling best_accuracy checkpoint page CER: 0.216913

This proves that the current update loop is close enough to succeed and that checkpoint selection should be the first milestone.

## Interfaces and Dependencies

At the end of Milestone 1, `app/recognition/active_learning.py` or a sibling helper module must expose a CER-aligned selector with a stable signature equivalent to:

    select_best_checkpoint(
        checkpoint_paths: list[Path],
        prepared_validation_pages: dict[str, PreparedPageDataset],
        output_root: Path,
    ) -> dict

The returned dictionary must include the selected checkpoint path and the per-checkpoint CER scores.

At the end of Milestone 2, `app/recognition/model_registry.py` must expose stable functions equivalent to:

    load_model_registry(registry_path: Path) -> dict
    register_candidate_model(registry_path: Path, candidate_metadata: dict) -> dict
    promote_candidate_model(registry_path: Path, candidate_id: str) -> dict
    rollback_active_model(registry_path: Path) -> dict

At the end of Milestone 3, `app/recognition/active_learning_metrics.py` must expose helpers equivalent to:

    compute_line_edit_burden(previous_text: str, corrected_text: str) -> dict
    summarize_page_correction_burden(line_records: list[dict]) -> dict
    write_active_learning_event(event_path: Path, payload: dict) -> None

At the end of Milestone 4, the replay manifest format must store enough metadata to support balanced and priority sampling. Each record must include the corrected text, line image path, page id, original OCR prediction, original OCR confidence, and correction edit distance.

At the end of Milestone 5, `app/app.py` must be able to route an opt-in local OCR active-learning save flow without blocking the request thread and without changing the current active model until promotion succeeds.

Revision note: created on 2026-04-16 to define the next expert-recommended phase after the first OCR fine-tuning implementation. The plan is intentionally blocker-first: fix checkpoint selection, then make model promotion safe, then measure correction burden, then add selective active-learning behavior, then integrate it into the app.
