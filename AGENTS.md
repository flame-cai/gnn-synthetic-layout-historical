# AGENTS.md

This file is for coding agents working in this repository. It explains the repository layout, maps the papers to the codebase, and documents the current verification and OCR fine-tuning research state so future agents can work from the actual source of truth instead of older assumptions.

## Repository Structure

The repository has two main products.

### 1. `src/`: Graph Neural Network text-line segmentation core

`src/` contains the implementation of the paper `Towards Text-Line Segmentation of Historical Documents Using Graph Neural Networks` by Kartik Chincholikar, Kaushik Gopalan, and Mihir Hasabnis.

The pipeline is:

1. Use CRAFT to detect character regions from a manuscript image and produce a heatmap.
2. Convert the heatmap into a point cloud of character centers and radii.
3. Build a heuristic graph using geometric priors.
4. Add extra candidate edges so true same-line links are available to the model.
5. Build node and edge features from geometry and heuristic metadata.
6. Train a GNN to classify candidate edges as keep or delete.
7. Convert kept edges into connected components and recover text lines.
8. Export predictions as graph labels, PAGE-XML, and cropped line images.

Important directories:

- `src/gnn_inference/`: end-to-end automatic inference.
- `src/synthetic_data_gen/`: synthetic layout generation and augmentation.
- `src/gnn_training/gnn_data_preparation/`: graph preprocessing for training.
- `src/gnn_training/training/`: GNN training code.
- `src/configs/`: YAML configs for synthetic generation, augmentation, preprocessing, and training.

### 2. `app/`: Semi-automatic annotation and OCR tool

`app/` contains the Flask backend, the frontend, and the OCR code used for local recognition research.

The app currently supports:

- semi-automatic text-line correction by adding and deleting nodes
- graph correction by adding and deleting edges
- manual grouping of text lines into text regions
- OCR using either the local Sanskrit checkpoint or Gemini
- PAGE-XML export and cropped line-image export

Important app paths:

- `app/app.py`: Flask backend and current save and recognition routes.
- `app/frontend/`: browser UI.
- `app/recognition/`: local OCR code, training code, active-learning utilities, and pretrained checkpoint handling.
- `app/tests/`: headless evaluation dataset, OCR fine-tuning verifier, unit tests, and logs.

## Current OCR Research Harness

As of 2026-04-20, the repository has three OCR layers that matter together:

- the offline OCR active-learning research harness
- the surrogate OCR fine-tuning pre-commit gate
- a first-pass GUI-safe OCR active-learning runtime that records manuscript-local checkpoints, page revisions, telemetry, and profiling under `input_manuscripts/<manuscript>/active_learning/recognition/`

The source-of-truth files are:

- `app/device_leases.py`
- `app/job_orchestrator.py`
- `app/manuscript_ocr_registry.py`
- `app/ocr_active_learning_runtime.py`
- `app/ocr_model_manager.py`
- `app/profiling.py`
- `app/recognition/active_learning.py`
- `app/recognition/active_learning_recipe.py`
- `app/recognition/pagexml_line_dataset.py`
- `app/recognition/dataset.py`
- `app/recognition/ocr_defaults.py`
- `app/recognition/train.py`
- `app/telemetry.py`
- `app/tests/precommit_gate_config.py`
- `app/tests/recognition_finetuning_config.py`
- `app/tests/recognition_finetuning_experiment.py`
- `app/tests/test_job_orchestrator_unit.py`
- `app/tests/test_manuscript_ocr_registry_unit.py`
- `app/tests/test_recognition_active_learning_unit.py`
- `app/tests/test_recognition_active_learning_backend_unit.py`
- `app/tests/test_recognition_finetuning_page_plus_history_unit.py`
- `app/tests/test_recognition_finetuning_precommit_unit.py`
- `app/tests/test_recognition_finetuning_precommit_e2e.py`
- `app/tests/test_recognition_finetuning_e2e.py`
- `app/tests/test_recognition_telemetry_unit.py`
- `scripts/run_precommit_eval.py`
- `.githooks/pre-commit`

The current harness supports:

- one canonical production OCR recipe in `app/recognition/active_learning_recipe.py` shared by runtime and pre-commit config code
- configurable sibling checkpoint strategy support between `best_accuracy.pth` and `best_norm_ED.pth`, with the live GUI runtime defaulting to `best_norm_ED.pth` through `OCR_RUNTIME_SIBLING_CHECKPOINT_STRATEGY` and the CER-aligned selector still available
- explicit OCR width policies: `global_2000_pad` and `batch_max_pad`
- bounded CER-weighted oversampling
- OCR-only augmentation policies: `none`, `background_only`, `background_plus_rotation`
- LR scheduler plumbing: `none`, `step`, `cosine`
- one retained continuation regime: `page_plus_random_history`
- deterministic history replay metadata with `history_sample_line_count=10`
- a shared checked-in pre-commit gate registry for dataset membership and thresholds
- a dedicated surrogate OCR pre-commit gate using the exact hybrid recipe `page_plus_random_history + batch_max_pad + no oversampling + no augmentation + Adadelta lr=0.2 + num_iter=60`
- manuscript-local OCR registries with durable page-revision snapshots, active/candidate checkpoint lineage, automatic fallback, and `needs_rebase` tracking
- save-triggered OCR job queueing that distinguishes commit saves from draft autosaves through explicit `saveIntent`
- a generic app-level job orchestrator with priorities, GPU device leases, and isolated OCR fine-tune/rebase jobs that can be canceled and requeued for interactive OCR
- manuscript-aware local OCR inference that loads the current manuscript checkpoint instead of assuming one global active model forever
- structured page/job telemetry and coarse profiling summaries, with optional sampled CUDA traces
- run artifacts including `curve_metrics.json`, `per_page.csv`, `per_line.csv`, `selector_metrics.json`, `fine_tune_metadata.json`, and plots

Earlier cumulative and page-only studies are now treated as preserved conclusions rather than live code paths. The retained conclusions are:

- the broad and focused sweeps established `batch_max_pad + no oversampling + no augmentation` as the stable structural stack worth keeping
- strict page-only continuation was viable but weaker and substantially more guard-sensitive on `eval_dataset`
- the hybrid replay recipe beat both earlier baselines on the primary curve metric and final-page CER
- Adam remained guard-sensitive even after replaying historical lines, so the trusted recipe remains Adadelta `lr=0.2`, `num_iter=60`

The canonical hybrid study aliases are:

- `app/tests/logs/recognition_finetune_results_latest.md`
- `app/tests/logs/recognition_finetune_results_latest.json`
- `app/tests/logs/recognition_finetune_results_latest.txt`

The current checked-in hybrid summary records these important results:

- Primary metric winner: `wb_on_an_hist10_sn_optd_lr200000u`
  Meaning: `page_plus_random_history`, `history_sample_line_count=10`, `batch_max_pad`, `none`, `none`, `optimizer=Adadelta`, `lr=0.2`, `num_iter=60`
  Evidence: `curve_metric_value=0.22151451085911972`
- Best `final_page_cer`: `wb_on_an_hist10_sn_optd_lr200000u`
  Evidence: `final_page_cer=0.13784355179704016`
- Best `first_step_gain`: `wb_on_an_hist10_sn_optd_lr200000u`
  Evidence: `first_step_gain=0.0572938689217759`

The latest dedicated surrogate pre-commit aliases are:

- `app/tests/logs/recognition_finetune_precommit_latest.md`
- `app/tests/logs/recognition_finetune_precommit_latest.json`
- `app/tests/logs/recognition_finetune_precommit_latest.txt`

See:

- `app/tests/logs/recognition_finetune_results_latest.md`
- `app/tests/logs/recognition_finetune_results_latest.json`

## Verification Commands

Run verification in the `gnn_layout` environment.

Two-phase pre-commit launcher:

From repository root:

    $env:CONDA_NO_PLUGINS='true'
    conda run -n gnn_layout python scripts/run_precommit_eval.py

Fast pretrained full-pipeline gate:

From repository root:

    $env:CONDA_NO_PLUGINS='true'
    conda run -n gnn_layout python -m unittest discover -s app/tests -p "test_ci_e2e.py" -v

Targeted OCR unit tests:

From repository root:

    $env:CONDA_NO_PLUGINS='true'
    conda run -n gnn_layout python -m unittest app.tests.test_recognition_active_learning_unit -v

Hybrid OCR pre-commit unit tests:

From repository root:

    $env:CONDA_NO_PLUGINS='true'
    conda run -n gnn_layout python -m unittest app.tests.test_recognition_finetuning_precommit_unit -v

Slow surrogate OCR pre-commit gate:

From repository root:

    $env:CONDA_NO_PLUGINS='true'
    conda run -n gnn_layout python -m unittest app.tests.test_recognition_finetuning_precommit_e2e -v

Slow OCR verifier and policy-study entrypoint:

Prefer an activated `gnn_layout` shell or the environment Python directly. On Windows, very long Unicode-heavy output printed through `conda run` can hit a `cp1252` encoding bug even when the test itself completes and writes artifacts.

Recommended PowerShell forms:

    conda activate gnn_layout
    python -m unittest discover -s app/tests -p "test_recognition_finetuning_e2e.py" -v

Or, if you want to bypass `conda run` entirely:

    C:\Users\intro\miniconda3\envs\gnn_layout\python.exe -m unittest discover -s app/tests -p "test_recognition_finetuning_e2e.py" -v

    C:\Users\intro\miniconda3\envs\gnn_layout\python.exe -m unittest app.tests.test_recognition_finetuning_precommit_e2e -v

If you use the direct interpreter form, update the path to match the local machine if needed.

## Documentation Map

Read these before changing behavior:

- `README.md`: introduction, install and usage instructions
- `ENGINEERING_DOCTRINE.md`: repository working norms
- `VISION.md`: long-term product and research direction
- `EVAL.md`: current evaluation architecture and OCR verifier (recognition model) reality
- `PLANS.md`: required format for execution plans
- `docs/exec-plans/tech-debt-tracker.md`: current high-priority debts

Current OCR-related plan documents:

- `docs/exec-plans/proposed/fine-tuning-page-plus-random-history-research.md`
- `docs/exec-plans/proposed/GUI_finetune_implement.md`
- `docs/exec-plans/proposed/hybrid-recognition-finetune-precommit-gate.md`

Historical OCR evidence should be preserved, not deleted casually:

- `docs/exec-plans/completed/old-recognition-finetuning-session-report-2026-04-16.md`
- `docs/exec-plans/completed/old-recognition-finetuning-failure-log.md`

## Guidance For Future Agents

- If you change OCR fine-tuning behavior, update `EVAL.md`, `VISION.md`, and the relevant ExecPlan in the same pass.
- If you change the policy-study harness, keep the artifact format stable unless there is a strong reason to change it.
- Do not treat `vadakautuhala.pth` as mutable. Fine-tuned checkpoints belong in run-local artifact folders.
- Keep writes inside the repository. OneDrive and Windows path length are real constraints here.
- If you run the slow OCR verifier on Windows, trust the saved artifact folder more than the raw `conda run` stdout if the wrapper crashes with a Unicode printing error after the study has completed.
