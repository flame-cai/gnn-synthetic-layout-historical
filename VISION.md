# VISION.md

This repository exists to reduce the total human effort required to digitize historical manuscripts. The product goal is not only better raw model accuracy, but a digitization workflow where corrected pages make later pages cheaper to annotate, easier to review, and safer to promote.

The repository currently has two connected products:

1. `src/`: the graph neural network text-line segmentation core.
2. `app/`: the semi-automatic annotation tool, OCR stack, evaluation harnesses, and active-learning runtime.

## Product Vision

The manuscript pipeline has two user-visible stages:

1. Layout analysis: detect characters, group them into text lines, group text lines into text regions, and write PAGE-XML geometry.
2. Text recognition: turn segmented text-line images into Unicode text and let the annotator correct the output.

The long-term product target is a manuscript-local learning loop:

1. Earlier pages need the most correction.
2. Those corrections become training or adaptation signals.
3. Background jobs run safely without freezing the app.
4. Later pages require less manual intervention.
5. Every promotion is explicit, reviewable, and reversible.

Success is therefore multi-objective:

- lower human correction burden
- stable or improved output quality
- explicit promotion rules
- reproducible evidence for research claims
- a frontend workflow that remains predictable while background work happens

## Verifier-Driven Evolution Harness

The repository now has a reusable research harness for step-by-step improvement of a specific pipeline stage with LLM-assisted code changes and external verifier metrics.

The generic pattern is:

1. Define one narrow pipeline stage with explicit inputs and outputs.
2. Keep one checked-in `benchmark_strategy`.
3. Implement one checked-in `proposed_strategy`.
4. Run the same external verifiers against both roles.
5. Aggregate the gate evidence.
6. Promote the proposed strategy explicitly inside the research harness only after all gates pass.
7. Adopt a strategy for production app use through a separate explicit command.
8. Keep the old strategy code and history for rollback and future ablations.

This harness is intentionally broader than text-line segmentation. A future agent can adapt it to another pipeline component if it first writes down:

- the exact input contract
- the exact output contract
- the primary quality metrics
- the acceptable regression rules
- the research promotion command, production adoption command, and source-of-truth config

The important design rule is that generated logs are evidence, but checked-in source config is the source of truth for which strategy currently owns the research benchmark role and which strategy the app uses in production.

## Current Harness Instance: Text-Line Segmentation To OCR Crops

The active non-OCR harness target is the step that converts:

- resized manuscript page images
- heatmaps
- PAGE-XML `Baseline` polylines predicted from the GNN graph

into:

- PAGE-space `TextLine/Coords` polygons
- OCR-ready text-line crops

The current checked-in strategy role source of truth is:

- `app/recognition/line_segmentation/strategy_config.py`

The current concrete strategies are:

- benchmark: `legacy_axis_bound_v1`
- proposed: `local_tangent_band_v1`
- production app: `legacy_axis_bound_v1`

`legacy_axis_bound_v1` preserves the historical axis-aligned behavior. `local_tangent_band_v1` is the generalized strategy for vertical, curved, and circular text while preserving horizontal behavior through selective legacy delegation.

The current explicit workflow is:

1. A researcher suggests a new `proposed_strategy`.
2. Agents implement the code and update docs/config.
3. The developer runs or triggers the three pre-commit gates.
4. `scripts/run_precommit_eval.py` writes gate artifacts and aggregate promotion evidence.
5. The developer reviews the evidence and runs `scripts/promote_text_line_strategy.py --apply`.
6. The research benchmark role moves forward in checked-in config, while old strategy code remains available for comparison.
7. A separate operator decision runs `scripts/adopt_text_line_strategy_for_app.py --apply` if the app should use a different production strategy.

This keeps promotion and production rollout reviewable. The pre-commit path does not silently mutate tracked config after Git has already prepared the commit, and research promotion is not a GUI/app rollout.

## Current Evaluation And Promotion State

The text-line segmentation harness currently uses three external verifier gates:

1. A pretrained full-pipeline strategy ablation gate on `app/tests/eval_dataset/`.
2. A surrogate OCR fine-tuning strategy ablation gate on `app/tests/eval_dataset/`.
3. A circular-layout OCR fine-tuning strategy ablation gate on `app/tests/eval_dataset_v2/`.

The launcher is:

- `scripts/run_precommit_eval.py`

The explicit promotion command is:

- `scripts/promote_text_line_strategy.py`

The explicit production adoption command is:

- `scripts/adopt_text_line_strategy_for_app.py`

The current strategy docs live under:

- `docs/pipeline-improvement/text-line-segmentation/`

The detailed evaluation architecture, thresholds, artifacts, and adaptation guidance live in:

- `EVAL.md`

## OCR Active Learning Reality

The OCR side of the repository remains the most mature active-learning subsystem. The important checked-in files are still:

- `app/recognition/active_learning.py`
- `app/recognition/active_learning_recipe.py`
- `app/ocr_active_learning_runtime.py`
- `app/manuscript_ocr_registry.py`
- `app/job_orchestrator.py`
- `app/tests/precommit_gate_config.py`
- `app/tests/recognition_finetuning_config.py`
- `app/tests/recognition_finetuning_experiment.py`

The retained OCR continuation recipe remains the hybrid `page_plus_random_history` recipe with:

- `history_sample_line_count=10`
- `width_policy=batch_max_pad`
- `oversampling_policy=none`
- `augmentation_policy=none`
- `optimizer=Adadelta`
- `lr=0.2`
- `num_iter=60`

The repository therefore has two complementary promotion/adoption concepts today:

1. manuscript-local OCR checkpoint promotion inside the runtime
2. repository-level text-line segmentation research promotion through checked-in config
3. repository-level text-line segmentation production adoption through checked-in config

They follow the same product rule: promotion or adoption must be explicit, reviewable, and not silent.

## Broader Research Direction

Over time, the same verifier-driven improvement pattern should be extended to other stages:

- character detection or a trainable CRAFT-like surrogate
- GNN text-line segmentation
- text-region grouping
- OCR post-processing or crop normalization

For each new stage, future agents should preserve the same discipline:

- stable benchmark/proposed role names
- one checked-in role config
- external verifier artifacts
- explicit research promotion after review
- explicit production adoption when app behavior should change
- historical code retained for comparison and rollback

## Non-Negotiable Constraints

- The app must remain usable while research code changes.
- A successful gate run must never silently edit tracked source config.
- Promotions and production adoptions must be explicit, reviewable, and reproducible.
- Generated artifacts must not be the only place where important conclusions live.
- Human-effort reduction should become a first-class logged metric, not only an anecdotal goal.
