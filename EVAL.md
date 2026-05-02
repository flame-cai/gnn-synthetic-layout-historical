# EVAL.md

This document is the evaluation blueprint for the repository. It covers both the current regression gate and the slower OCR active-learning research harness. It should be treated as the source of truth when changing evaluation thresholds, adding datasets, or extending the fine-tuning workflow.

The short version is this: evaluation in this repository is not just about model accuracy on a fixed benchmark. The long-term goal is to reduce the total human effort required to digitize a manuscript page by page while preserving or improving output quality.

## Why Evaluation Exists Here

The repository has two connected products:

1. The GNN layout analysis core in `src/`
2. The semi-automatic annotation and recognition tool in `app/`

The app already lets a human correct:

- character-detection failures by adding and deleting nodes
- graph mistakes by adding and deleting edges
- text-region grouping by assigning textbox labels
- OCR mistakes by editing text

That means evaluation here has to answer two different questions:

1. Does the automatic pipeline still work after a code change?
2. Are we reducing manual effort over time as corrections are fed back into model updates?

The first question is covered by the existing headless regression gate. The second question is the main active-learning direction.

## Current Source Of Truth

The current evaluation code lives in:

- `app/tests/precommit_gate_config.py`
- `app/tests/test_ci_e2e.py`
- `app/tests/test_recognition_active_learning_unit.py`
- `app/tests/test_recognition_finetuning_precommit_unit.py`
- `app/tests/test_recognition_finetuning_precommit_e2e.py`
- `app/tests/test_recognition_finetuning_e2e.py`
- `app/tests/test_recognition_finetuning_page_plus_history_unit.py`
- `app/tests/evaluate.py`
- `app/tests/recognition_finetuning_config.py`
- `app/tests/recognition_finetuning_experiment.py`
- `app/recognition/active_learning.py`
- `app/recognition/active_learning_recipe.py`
- `app/job_orchestrator.py`
- `app/manuscript_ocr_registry.py`
- `app/ocr_active_learning_runtime.py`
- `app/ocr_model_manager.py`
- `app/profiling.py`
- `app/telemetry.py`
- `app/tests/test_job_orchestrator_unit.py`
- `app/tests/test_manuscript_ocr_registry_unit.py`
- `app/tests/test_recognition_active_learning_backend_unit.py`
- `app/tests/test_recognition_telemetry_unit.py`
- `app/recognition/pagexml_line_dataset.py`
- `app/tests/eval_dataset/`
- `.githooks/pre-commit`
- `scripts/run_precommit_eval.py`

## Evaluation Layers

### Level 0: Fast structural checks

Purpose: catch obvious breakage cheaply.

Examples:

- PAGE-XML parsing checks
- OCR selector unit tests
- padding-policy tests
- oversampling and augmentation tests

These checks are necessary but not sufficient. They prove that narrow mechanics still behave correctly.

### Level 1: Pre-commit regression gates

Purpose: screen normal commits with two complementary GUI-free checks.

Current entrypoints:

- `app/tests/test_ci_e2e.py`
- `app/tests/test_recognition_finetuning_precommit_e2e.py`
- `app/tests/precommit_gate_config.py`

The first gate runs:

1. manuscript upload
2. CRAFT plus GNN inference
3. PAGE-XML generation
4. OCR
5. evaluation against PAGE-XML ground truth

The second gate runs one explicit hybrid OCR continuation recipe on perfect PAGE-XML-derived line crops and ground-truth text pairs, records `curve_metric_value`, `final_page_cer`, `first_step_gain`, `regression_guard_passed`, and `max_regression`, and compares only the three blocking metrics against checked-in thresholds. Regression-guard failure is preserved as a warning, not a hard failure, for this gate.

These two checks intentionally validate different failure surfaces. The first remains the required pretrained full-pipeline gate for changes that can affect the app, OCR, PAGE-XML generation, evaluation code, or GNN inference. The second is a surrogate guard for the OCR fine-tuning subsystem even now that the GUI has a first-pass live OCR active-learning runtime, because the live runtime promotes on manuscript-local verifier-bank evidence rather than the fixed held-out benchmark used by the surrogate gate.

### Level 2: Offline OCR active-learning research harness

Purpose: evaluate whether sequential OCR fine-tuning improves later pages in a reproducible way.

Current entrypoint:

- `app/tests/test_recognition_finetuning_e2e.py`

This harness does not use CRAFT or GNN segmentation. Instead it:

1. prepares perfect line crops from PAGE-XML ground truth
2. fine-tunes the local OCR checkpoint sequentially on earlier pages
3. evaluates later pages after each update
4. writes full local run artifacts for debugging and later review

This harness is offline-first and research-oriented. It remains the slower policy-study path, separate from the live GUI runtime.

### Level 3: Live GUI and human-in-the-loop evaluation

Purpose: measure whether real user correction effort drops as the system learns.

As of 2026-04-20, the first pass of this layer exists. The save and recognition flows now record manuscript-local page revisions, checkpoint lineage, page/job telemetry, and profiling artifacts under `input_manuscripts/<manuscript>/active_learning/recognition/`. Commit saves with corrected text can enqueue OCR fine-tune or rebase jobs, and local OCR recognition loads the current manuscript checkpoint rather than assuming one global model forever.

This layer is still early. The live runtime now uses direct promotion after background OCR fine-tuning. It still does not answer every human-effort question the long-term evaluation story will need.

## Current OCR Harness State

As of 2026-04-20, the OCR active-learning harness has the following implemented features:

- one canonical production OCR recipe in `app/recognition/active_learning_recipe.py`
- configurable sibling checkpoint strategy support between `best_accuracy.pth` and `best_norm_ED.pth`; the live GUI runtime currently defaults to `best_norm_ED.pth` through `OCR_RUNTIME_SIBLING_CHECKPOINT_STRATEGY`, while the CER-aligned selector remains available for research and gate runs
- explicit width policies: `global_2000_pad` and `batch_max_pad`
- bounded CER-weighted oversampling
- OCR-only augmentation policies, including 10 extra `background_plus_rotation` variants per logical train sample or oversampled replica
- one retained page-plus-random-history continuation path where each post-baseline step fine-tunes on the newly added page plus up to 10 randomly sampled lines from earlier fine-tune pages, with deterministic sampling metadata and a first-page fallback to current-page-only training
- deterministic loader-level train reshuffling using a seeded `torch.Generator`, with shuffle policy recorded in metadata
- additive train-dataset metadata that separates `current_page_ids`, `history_source_page_ids`, `history_sample_page_ids`, `history_sample_line_refs`, and per-sample origin markers
- richer local artifacts including `curve_metrics.json`, `per_page.csv`, `per_line.csv`, `selector_metrics.json`, `fine_tune_metadata.json`, winners-by-metric metadata, and plots
- a shared checked-in pre-commit gate registry in `app/tests/precommit_gate_config.py` so dataset membership and thresholds are no longer buried in unittest bodies
- a dedicated surrogate OCR fine-tuning pre-commit runner in `app/tests/recognition_finetuning_experiment.py` exposed through `run_recognition_precommit_gate(...)`
- focused unit coverage for selector choice, width policy, oversampling, augmentation multiplicity, slug encoding, shuffle behavior, and page-plus-history sampling
- a GUI-safe app runtime with `app/job_orchestrator.py`, `app/device_leases.py`, `app/manuscript_ocr_registry.py`, `app/ocr_active_learning_runtime.py`, `app/ocr_model_manager.py`, `app/telemetry.py`, and `app/profiling.py`
- manuscript-local page-revision snapshots, active/candidate checkpoint lineage, recorded automatic promotion, rebase detection, and active-checkpoint persistence across app restarts
- structured save/recognition telemetry including save intent, node and edge edit counts, OCR text edit distance against the last prediction, checkpoint ids, and job queue events
- coarse profiling summaries plus optional sampled CUDA traces saved under each manuscript runtime folder

Relevant files:

- `app/recognition/active_learning.py`
- `app/recognition/active_learning_recipe.py`
- `app/recognition/dataset.py`
- `app/recognition/ocr_defaults.py`
- `app/recognition/train.py`
- `app/job_orchestrator.py`
- `app/manuscript_ocr_registry.py`
- `app/ocr_active_learning_runtime.py`
- `app/ocr_model_manager.py`
- `app/profiling.py`
- `app/telemetry.py`
- `app/tests/precommit_gate_config.py`
- `app/tests/recognition_finetuning_config.py`
- `app/tests/recognition_finetuning_experiment.py`

## Latest Completed OCR Studies

The shared dataset assumptions are still:

- ordered pages: `233_0002` through `233_0016`
- sequential fine-tune pages: `233_0002` through `233_0010`
- evaluation pages: `233_0011` through `233_0016`
- batch size: `1`
- primary metric: `early_weighted_page_cer`
- regression guard: no step may worsen aggregate page CER by more than `0.005` absolute versus the previous step

The retained live study family is the hybrid page-plus-random-history follow-up. Generated study artifacts are local run outputs, so the durable findings are copied here instead of relying on artifact files being present in a GitHub checkout.

Important results from the 2-policy hybrid follow-up:

- Primary-metric winner: `wb_on_an_hist10_sn_optd_lr200000u`
  Meaning: `training_policy=page_plus_random_history`, `history_sample_line_count=10`, `width_policy=batch_max_pad`, `oversampling_policy=none`, `augmentation_policy=none`, `lr_scheduler=none`, `optimizer=Adadelta`, `lr=0.2`, `num_iter=60`
  Evidence: `curve_metric_value=0.22151451085911972`
- Best `final_page_cer`: `wb_on_an_hist10_sn_optd_lr200000u`
  Evidence: `final_page_cer=0.13784355179704016`
- Best `first_step_gain`: `wb_on_an_hist10_sn_optd_lr200000u`
  Evidence: `first_step_gain=0.0572938689217759`
- The first page correctly used no history lines, while later steps recorded both the candidate history pool and the 10 sampled history lines in metadata.
  Evidence: step 1 stores `history_source_page_ids=[]` and `history_sample_line_count=0`; step 2 stores `history_source_page_ids=["233_0002"]` and `history_sample_line_count=10`

The earlier broad and page-only studies are no longer live code paths, but their conclusions still matter:

- the broad and focused sweeps established `batch_max_pad + no oversampling + no augmentation + Adadelta lr=0.2` as the strongest stable structural stack worth keeping
- the best retained cumulative result reached `curve_metric_value=0.22700365173938108` and `final_page_cer=0.15137420718816066`
- the best page-only result reached `curve_metric_value=0.24025369978858352` and `final_page_cer=0.17061310782241015`, but three of the four page-only policies failed the regression guard
- the hybrid replay recipe improved on both earlier baselines on the primary curve metric and final-page CER, while the Adam hybrid candidate still failed the regression guard

The pre-commit OCR result is intentionally narrower than the broader study. It always represents one explicit best-known hybrid recipe and one thresholded dataset result, not a policy sweep. The blocking thresholds are checked in through `app/tests/precommit_gate_config.py`: `curve_metric_value <= 0.26`, `final_page_cer <= 0.18`, and `first_step_gain >= 0.04`.

## Required Artifacts

Serious evaluation runs should write a dedicated local artifact folder. Those artifacts are evidence for a run, but they are generated outputs and should not be the only place a checked-in research conclusion is recorded.

The current OCR study shape is:

    <local-artifact-root>/<timestamp>_ocrft_<dataset>/
      config.json
      summary.md
      metrics.json
      gt_eval_subset/
      prepared_pages/
      policies/
        <policy_slug>/
          config.json
          summary.md
          metrics.json
          curve_metrics.json
          per_page.csv
          per_line.csv
          selector_metrics.json
          fine_tune_metadata.json
          predicted_page_xml/
          plots/

Interpretation:

- `config.json`: dataset setup, device info, and policy values
- `summary.md`: readable summary of the run
- `metrics.json`: machine-readable aggregate metrics
- `curve_metrics.json`: primary curve metric and regression-guard outcome
- `per_page.csv`: page-level metrics for each fine-tune step
- `per_line.csv`: per-line OCR detail including width and padding metadata
- `selector_metrics.json`: sibling checkpoint selection evidence, either CER ranking details or an explicit configured direct preference
- `fine_tune_metadata.json`: training options, selected checkpoint, and timing

This OCR artifact layout should be preserved unless there is a strong reason to change it.

## Current Commands

### Full-pipeline pre-commit gate

From repository root:

    $env:CONDA_NO_PLUGINS='true'
    conda run -n gnn_layout python -m unittest discover -s app/tests -p "test_ci_e2e.py" -v

### OCR unit tests

From repository root:

    $env:CONDA_NO_PLUGINS='true'
    conda run -n gnn_layout python -m unittest app.tests.test_recognition_active_learning_unit -v

Hybrid continuation unit coverage:

    $env:CONDA_NO_PLUGINS='true'
    conda run -n gnn_layout python -m unittest app.tests.test_recognition_finetuning_page_plus_history_unit -v

Hybrid OCR pre-commit unit coverage:

    $env:CONDA_NO_PLUGINS='true'
    conda run -n gnn_layout python -m unittest app.tests.test_recognition_finetuning_precommit_unit -v

### OCR fine-tuning surrogate pre-commit gate

From repository root:

    $env:CONDA_NO_PLUGINS='true'
    conda run -n gnn_layout python -m unittest app.tests.test_recognition_finetuning_precommit_e2e -v

This gate uses perfect PAGE-XML-derived line crops and ground-truth text pairs. It does not validate upstream CRAFT or GNN behavior, and it should not be described as if it were the final interactive active-learning loop.

### Two-phase launcher

From repository root:

    $env:CONDA_NO_PLUGINS='true'
    conda run -n gnn_layout python scripts/run_precommit_eval.py

or:

    C:\Users\intro\miniconda3\envs\gnn_layout\python.exe scripts/run_precommit_eval.py

The launcher runs the full-pipeline gate first and the OCR fine-tuning surrogate gate second. `SKIP_EVAL_HOOK=1` skips both phases. `SKIP_PIPELINE_EVAL_HOOK=1` skips only the pretrained full-pipeline phase. `SKIP_RECOGNITION_FT_HOOK=1` skips only the OCR fine-tuning surrogate phase.

### Slow OCR study verifiers

On Windows, prefer an activated `gnn_layout` shell or the direct environment Python for these commands. Very long Unicode-heavy output can trigger a `conda run` printing failure even after the study itself has finished and written artifacts.

Preferred forms:

    conda activate gnn_layout
    python -m unittest discover -s app/tests -p "test_recognition_finetuning_e2e.py" -v

or:

    C:\Users\intro\miniconda3\envs\gnn_layout\python.exe -m unittest discover -s app/tests -p "test_recognition_finetuning_e2e.py" -v

If the wrapper crashes after artifact creation, inspect the artifact directory printed by the runner before assuming the study itself failed.

## Metrics By Task

### 1. Character detection or a future surrogate CRAFT task

What matters:

- missing character centers
- false character centers
- downstream effect on text-line segmentation
- amount of manual node correction required

Current reality:

- the app logs node corrections in `app/app.py`
- there is no formal node-level benchmark yet

### 2. GNN edge classification for text-line segmentation

What matters:

- PAGE-level CER
- line-level CER at polygon IoU thresholds
- how easy graph mistakes are to repair

Current reality:

- automatic evaluation already computes PAGE-level and line-level CER through `app/tests/evaluate.py`
- edge-edit telemetry now lands in manuscript-local page events, but there is still no benchmarked graph-repair study tied to those logs

### 3. Text-region grouping

What matters:

- correct region assignment
- how often users need to merge or split regions manually

Current reality:

- the UI supports region grouping
- there is no formal region-grouping evaluation dataset yet

### 4. Text recognition

What matters:

- OCR CER
- confidence summaries
- first-step gain
- final-page CER
- early-weighted curve quality across sequential fine-tuning steps
- eventually, human text-correction effort

Current reality:

- automatic OCR evaluation exists
- an offline sequential OCR fine-tuning harness exists
- the GUI now records manuscript-local checkpoints and promotes them through a recorded direct-promotion runtime rule that is intentionally narrower than the offline study harness

## Human Effort Metrics

Because the long-term goal is active learning, human effort has started to become a first-class metric.

The current live runtime logs, per page:

- nodes added
- nodes deleted
- edges added
- edges deleted
- OCR text edit distance against the last prediction the user saw
- changed line count
- save intent
- whether the revision entered OCR active learning
- prediction engine
- active checkpoint id and candidate checkpoint id

Important gaps remain:

- total correction time is still not logged
- textbox-edit telemetry is still not first-class
- the repository does not yet summarize effort trends across a whole manuscript automatically

## Current Evaluation Limits

The repository has meaningful OCR research infrastructure now, but the current benchmark is still narrow.

Current limits:

- only one OCR evaluation dataset is wired into the harness
- the cumulative, page-only, and page-plus-random-history studies still use only one manuscript sequence
- page-only continuation is now implemented, but many page-only policies terminate early on the regression guard, so "all policies reach all nine continuation pages" is not a realistic acceptance rule
- page-plus-random-history has only been tested with `history_sample_line_count=10` and random line replay from prior pages; other replay sizes and sampling strategies are still unexplored
- Adam remains guard-sensitive in both page-only and page-plus-random-history follow-ups, so optimizer-specific tuning is still unresolved
- the live GUI runtime has only been exercised against the current local-manuscript flow, not a broad benchmark suite
- the live GUI runtime is intentionally local and lightweight, so it is not a replacement for the surrogate pre-commit gate
- manuscript-local telemetry exists, but there is still no checked-in evaluator that turns those logs into cross-manuscript effort curves

## Near-Term Recommended Additions

The next high-value steps are:
1. Harden the live manuscript registry and rebuild path with more restart and interrupted-job coverage.
2. Add a checked-in evaluator that turns manuscript-local telemetry into manuscript-level effort summaries.
3. Wrap the direct-interpreter OCR-study path in a small helper so Windows-safe UTF-8 execution is the easiest path, not just the documented path.
4. Extend the OCR benchmark to multiple manuscript sequences now that the cumulative, page-only, hybrid follow-up, and live GUI runtime all exist.

## Definition Of Success

Evaluation maturity in this repository means:

- every significant code change can be screened by automatic headless gates that cover both the pretrained full pipeline and the current OCR fine-tuning subsystem
- every OCR research claim can be tied to an artifact-producing run and exact settings, with durable conclusions copied into checked-in docs when they matter beyond one local workspace
- the OCR verifier can rank policies by the repository's chosen primary metric
- GUI OCR fine-tuning promotes models only through a recorded direct-promotion runtime rule
- human correction burden is eventually measured directly rather than inferred indirectly
