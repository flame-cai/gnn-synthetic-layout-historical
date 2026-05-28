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
7. Adopt a strategy for production app use through a separate explicit command that changes future PAGE geometry generation and the crop metadata honored by production OCR preparation.
8. Keep the old strategy code and history for rollback and future ablations.

This harness is intentionally broader than text-line segmentation. A future agent can adapt it to another pipeline component if it first writes down:

- the exact input contract
- the exact output contract
- the primary quality metrics
- the acceptable regression rules
- the research promotion command, production adoption command, and source-of-truth config

The important design rule is that generated logs are evidence, but checked-in source config is the source of truth for which strategy currently owns the research benchmark role and which strategy the app uses in production. Because generated logs under `app/tests/logs/` are ignored, `scripts/run_precommit_eval.py` also refreshes `docs/pipeline-improvement/text-line-segmentation/strategy-promotion-record.md` as the checked-in promotion evidence summary.

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

- benchmark: `local_polygons_v1`
- proposed: not configured
- production app: `local_polygons_v1`

`local_polygons_v1` is the current research benchmark after promotion on 2026-05-22 and the current production app strategy after explicit adoption on 2026-05-23. It uses baseline-local heatmap contour masks so circular OCR crops are not sized from page-axis-aligned heatmap boxes. Its heatmap threshold is `0.45` before local cleanup so weak detached-mark evidence reaches the boundary trimmer; open-line final-mask normal padding stays at zero, while closed circular lines retain a separate final-mask override. `legacy_axis_bound_v1` preserves the historical axis-aligned behavior and remains available for rollback and legacy-page fallback behavior. `local_tangent_band_v1` remains available as the earlier generalized benchmark for vertical, curved, and circular text while preserving horizontal behavior through selective legacy delegation.

Production now has a shared OCR crop preparation boundary in `app/recognition/line_segmentation/ocr_crops.py`. App line-image export, local OCR inference, and active-learning revision preparation all read saved PAGE `Coords` and optional strategy metadata through that layer. New `local_polygons_v1` layout saves write metadata that asks this layer for local-polygon unwrapping with a median-color background. Pages without usable metadata continue through the masked PAGE `Coords` fallback.

Layout mode also has optional intra-line reading-direction annotation. The `O` shortcut records a cross-line cut, resolves it to the final line by component overlap, and stores it in a reading-direction sidecar. Open lines use the local cut tangent to resolve 180-degree ambiguity; circular lines use it to choose both unwrap start station and direction. Missing or stale annotations fall back to script-specific defaults.

The current explicit workflow is:

1. A researcher suggests a new `proposed_strategy`.
2. Agents implement the code and update docs/config.
3. The developer runs or triggers the three pre-commit gates.
4. `scripts/run_precommit_eval.py` writes gate artifacts, aggregate promotion evidence, and the checked-in promotion record.
5. The developer reviews the evidence and runs `scripts/promote_text_line_strategy.py --apply`.
6. The research benchmark role moves forward in checked-in config, while old strategy code remains available for comparison.
7. A separate operator decision runs `scripts/adopt_text_line_strategy_for_app.py --apply` if the app should use a different production strategy for future geometry and crop behavior.

This keeps promotion and production rollout reviewable. The pre-commit path does not silently mutate tracked config after Git has already prepared the commit, and research promotion is not a GUI/app rollout.

## Current Evaluation And Promotion State

The text-line segmentation harness currently uses three external verifier gates:

1. A pretrained full-pipeline strategy ablation gate on `app/tests/eval_dataset/`.
2. A surrogate OCR fine-tuning strategy ablation gate on `app/tests/eval_dataset/`.
3. A circular-layout OCR fine-tuning strategy ablation gate on `app/tests/eval_dataset_v2/`.

The surrogate OCR gates keep the retained hybrid recipe, but the checked-in pre-commit registry controls their fine-tuning page prefix separately from slower research studies. The regular OCR gate now defaults to three fine-tuning pages so the commit guard stays bounded while still measuring the same held-out evaluation pages.

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

In the production GUI, reviewed ground truth is a save-state contract rather than a separate lock button. `Save Page` and `Save & Next Page` in Text Review commit reviewed text as supervised OCR ground truth. Draft autosaves, raw OCR predictions, and Page Layout saves remain recoverability or layout-lineage states and do not enter OCR fine-tuning.

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
