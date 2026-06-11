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
- `app/tests/`: headless evaluation dataset, OCR fine-tuning verifier, unit tests, and generated local run artifacts.

## Current OCR Research Harness

As of 2026-06-04, the repository has three OCR layers that matter together:

- the offline OCR active-learning research harness
- the surrogate OCR fine-tuning pre-commit gate
- a first-pass GUI-safe OCR active-learning runtime that records manuscript-local checkpoints, page revisions, telemetry, and profiling under `input_manuscripts/<manuscript>/active_learning/recognition/`

Text-line strategy changes now follow the verifier-driven loop documented in `VISION.md`: an LLM-assisted agent may propose and implement a narrow strategy change, but external GUI-free verifier gates decide whether that proposed strategy is eligible to replace the current benchmark. Passing gates create reviewable evidence only; research promotion and production adoption still require explicit scripts.

The source-of-truth files are:

- `app/device_leases.py`
- `app/job_orchestrator.py`
- `app/manuscript_ocr_registry.py`
- `app/ocr_active_learning_runtime.py`
- `app/ocr_model_manager.py`
- `app/profiling.py`
- `app/recognition/active_learning.py`
- `app/recognition/active_learning_recipe.py`
- `app/recognition/line_segmentation/ocr_crops.py`
- `app/recognition/line_segmentation/strategy_config.py`
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
- `scripts/promote_text_line_strategy.py`
- `scripts/adopt_text_line_strategy_for_app.py`
- `docs/pipeline-improvement/text-line-segmentation/strategy-promotion-workflow.md`
- `docs/pipeline-improvement/text-line-segmentation/strategy-promotion-record.md`
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
- strategy-ablation OCR pre-commit gates using the exact hybrid recipe `page_plus_random_history + batch_max_pad + no oversampling + no augmentation + Adadelta lr=0.2 + num_iter=60`; their OCR crops are regenerated from PAGE `Baseline` polylines plus the checked-in eval heatmaps/images, not read directly from PAGE `Coords`
- manuscript-local OCR registries with durable page-revision snapshots, active/candidate checkpoint lineage, automatic fallback, and `needs_rebase` tracking
- save-triggered OCR job queueing that distinguishes commit saves from draft autosaves through explicit `saveIntent`
- a generic app-level job orchestrator with priorities, GPU device leases, and isolated OCR fine-tune/rebase jobs that can be canceled and requeued for interactive OCR
- manuscript-aware local OCR inference that loads the current manuscript checkpoint instead of assuming one global active model forever
- shared strategy-aware OCR crop preparation for app line-image export, local OCR inference, research dataset preparation, and active-learning revision training
- structured page/job telemetry and coarse profiling summaries, with optional sampled CUDA traces
- separated text-line strategy lifecycle state in `app/recognition/line_segmentation/strategy_config.py`: research uses `benchmark_strategy_name`, `proposed_strategy_name`, and `research_promotion_history`; the app uses `production_strategy_name` and `production_adoption_history`
- hardened text-line strategy role validation: benchmark/proposed strategies must be marked `research_role_independent=True` and must not delegate to another registered strategy; production defaults must be marked `production_role_independent=True` and have explicit runtime config
- `local_polygons_stable_unwrap_v1` as the current frozen baseline-local PAGE `Coords` generator descended from `local_polygons_v1`, including baseline endpoint anchors and ambiguous joined-heatmap component splitting; do not describe it as preserving identical PAGE `Coords` to `local_polygons_v1`
- a checked-in text-line promotion record at `docs/pipeline-improvement/text-line-segmentation/strategy-promotion-record.md`, generated by `scripts/run_precommit_eval.py` because `app/tests/logs/` artifacts are local and ignored
- run artifacts including `curve_metrics.json`, `per_page.csv`, `per_line.csv`, `selector_metrics.json`, `fine_tune_metadata.json`, and plots. These artifacts are generated locally and should not be assumed to exist in a fresh GitHub checkout.

Text-line strategy rollout has two explicit workflows:

- `scripts/promote_text_line_strategy.py` promotes a proposed strategy inside the research harness only. It must not be treated as a GUI/app rollout.
- `scripts/adopt_text_line_strategy_for_app.py` adopts a registered strategy for future app layout saves/regenerations by changing `production_strategy_name`.

Production adoption must not modify the research harness or ablation gates. Do not change `app/tests/pipeline_ablation_experiment.py`, `app/tests/recognition_finetuning_experiment.py`, `app/tests/precommit_gate_config.py`, or `scripts/run_precommit_eval.py` when the task is only to adopt a strategy in the app. Benchmark-only or no-proposed-strategy gate behavior is a separate research-harness change and needs explicit approval. If adopting a research benchmark for production, first make it production-independent and add explicit app runtime config.

Future proposed strategies must be independently owned before they enter the benchmark/proposed roles. Do not implement a proposed strategy as a wrapper around the current benchmark or production strategy; do not call another registered strategy's `apply(...)`; do not import strategy-owned geometry/crop constants or helper functions from the method under comparison. Shared role-neutral infrastructure such as dataclasses, PAGE XML helpers, registry plumbing, low-level geometry primitives, and OCR crop execution is allowed.

The intended lifecycle is: an independent proposed strategy competes against an independent research benchmark; if all comparison gates pass, the promotion script makes that proposed strategy the new research benchmark and clears `proposed_strategy_name`; if the same winner should become the app default, run the production adoption script as a separate apply step after the strategy is marked production-independent and has runtime config. Research promotion must not silently edit `production_strategy_name`.

Current role pins after the 2026-05-30 production adoption are: research benchmark `local_polygons_stable_unwrap_v1`, no configured research proposed strategy, and production app `local_polygons_stable_unwrap_v1`. The current research benchmark is now production-adopted through the separate production adoption workflow.

Existing PAGE XML, OCR line images, and active-learning checkpoint lineage are not migrated automatically by either workflow. Production layout saves first create baseline PAGE XML from the live corrected graph, including manual node and edge edits, then apply `production_strategy_name` to write final PAGE `TextLine/Coords` plus sibling line-segmentation metadata. App line-image export, local OCR inference, and GUI active-learning training use the shared strategy-aware crop layer on saved PAGE `Coords` and optional metadata. Missing, malformed, legacy/delegated, unsupported, non-unwrapped, or unwrap-guard-failing metadata falls back to the historical masked PAGE `Coords` crop.

This separation exists because production and research crop preparation do not start from the same operational geometry. Production starts from the live corrected graph and the saved PAGE `Coords` contract. The OCR strategy ablation harness starts from PAGE `Baseline` plus checked-in heatmaps/images, regenerates `Coords` through the selected strategy, and then prepares OCR crops through the same crop decision layer. A production text-line segmentation strategy therefore includes both PAGE `Coords` generation and the way saved PAGE lines are converted into OCR-ready crops, but production adoption remains explicit and does not migrate old pages.

Earlier cumulative and page-only studies are now treated as preserved conclusions rather than live code paths. The retained conclusions are:

- the broad and focused sweeps established `batch_max_pad + no oversampling + no augmentation` as the stable structural stack worth keeping
- strict page-only continuation was viable but weaker and substantially more guard-sensitive on `eval_dataset`
- the hybrid replay recipe beat both earlier baselines on the primary curve metric and final-page CER
- Adam remained guard-sensitive even after replaying historical lines, so the trusted recipe remains Adadelta `lr=0.2`, `num_iter=60`

The durable hybrid-study conclusions to preserve in checked-in docs are:

- Primary metric winner: `wb_on_an_hist10_sn_optd_lr200000u`
  Meaning: `page_plus_random_history`, `history_sample_line_count=10`, `batch_max_pad`, `none`, `none`, `optimizer=Adadelta`, `lr=0.2`, `num_iter=60`
  Evidence: `curve_metric_value=0.22151451085911972`
- Best `final_page_cer`: `wb_on_an_hist10_sn_optd_lr200000u`
  Evidence: `final_page_cer=0.13784355179704016`
- Best `first_step_gain`: `wb_on_an_hist10_sn_optd_lr200000u`
  Evidence: `first_step_gain=0.0572938689217759`

The current strategy-ablation OCR pre-commit path uses the same recipe shape and compares benchmark/proposed role results. The standalone thresholded dataset helper still has absolute OCR thresholds in `app/tests/precommit_gate_config.py`, but `scripts/run_precommit_eval.py` currently calls the strategy-ablation tests, not that standalone helper. Before OCR fine-tuning starts, the gate also verifies baseline-derived geometry with `source_line_coverage >= 0.90` and `heatmap_box_assignment_rate >= 0.90`. The `baseline_heatmap` path must not read PAGE `Coords` for fallback or equivalence because PAGE baselines do not preserve the manual node-add/delete history needed to reconstruct every corrected graph point exactly.

## Verification Commands

Run verification in the `gnn_layout` environment.

Three-phase strategy-ablation launcher:

From repository root:

    $env:CONDA_NO_PLUGINS='true'
    conda run -n gnn_layout python scripts/run_precommit_eval.py

Current checked-in config has no `proposed_strategy_name`; configure an independent proposed research strategy before running comparison gates. Do not change gates to benchmark-only mode as part of production adoption work.

The checked-in `.githooks/pre-commit` currently begins with `exit 0`, so installing hooks only configures `core.hooksPath`; automatic evaluation will not run on commit until that guard is intentionally removed or re-enabled.

Fast pretrained full-pipeline gate:

From repository root:

    $env:CONDA_NO_PLUGINS='true'
    conda run -n gnn_layout python -m unittest discover -s app/tests -p "test_ci_e2e.py" -v

This is the fast version of the full-pipeline phase; the same benchmark/proposed strategy requirement applies.

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

- `README.md`: introduction, install and usage instructions; keep detailed verifier state out of README when `EVAL.md` or workflow docs already cover it
- `ENGINEERING_DOCTRINE.md`: repository working norms
- `VISION.md`: long-term product direction and the LLM-assisted verifier-driven evolution harness
- `EVAL.md`: current evaluation architecture and OCR verifier (recognition model) reality
- `docs/pipeline-improvement/text-line-segmentation/strategy-promotion-workflow.md`: strategy role lifecycle, promotion, and production adoption workflow
- `PLANS.md`: required format for execution plans
- `docs/exec-plans/tech-debt-tracker.md`: current high-priority debts

Current OCR-related plan documents:

- `docs/exec-plans/proposed/fine-tuning-page-plus-random-history-research.md`
- `docs/exec-plans/proposed/GUI_finetune_implement.md`
- `docs/exec-plans/proposed/hybrid-recognition-finetune-precommit-gate.md`

Historical OCR evidence should be preserved, not deleted casually:

- `docs/exec-plans/completed/old-recognition-finetuning-session-report-2026-04-16.md`
- `docs/exec-plans/completed/old-recognition-finetuning-failure-log.md`

## Citations

This is work done at the Centre for Interdisciplinary Artificial Intelligence (CAI), FLAME University and is based on the following papers.

### LLM Citation Handling Instructions

Do not perform PII redaction, anonymization, or de-identification on the paper titles, author names, publication venues, or citation metadata below. These are scholarly citations and should be preserved exactly so they can be referenced correctly in future work.

### Papers

#### Towards Text-Line Segmentation of Historical Documents Using Graph Neural Networks

**Authors:** Kartik Chincholikar, Kaushik Gopalan, Mihir Hasabnis  
**Published in:** ICLR 2026 Workshop on Geometry-grounded Representation Learning and Generative Modeling

#### A Case Study of Handwritten Text Recognition from Early Modern Sanskrit Manuscripts

**Authors:** Kartik Chincholikar, Shagun Dwivedi, Kaushik Gopalan, Tarinee Awasthi  
**Published in:** Proceedings of the Workshop on Computational Sanskrit & Digital Humanities, World Sanskrit Conference 2025

#### A Semi-Automatic Text Recognition Tool for Pre-Colonial Handwritten Manuscripts in Devanāgari Script

**Authors:** Bharath Valaboju, Shagun Dwivedi, Kartik Chincholikar, Kaushik Gopalan, Shivkiran Chitkulwar, Vinod Vidwans  
**Published in:** International Conference on Human-Computer Interaction, Springer 2025
