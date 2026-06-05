# VISION.md

# VISION
This repository exists to reduce the total human effort required to digitize historical manuscripts. The goal is to build an historical manuscript digitization workflow, where previously corrected pages (annotated data) is used to train AI models which make better predictions on lsubsequent pages, reducing the burden of annotation continuously in a loop.

The repository currently has two connected products:
1. `src/`: the graph neural network text-line segmentation core (this is for doing GNN improvements and research)
2. `app/`: this contain the full-fledged Semi-automatic annotation tool with a frontend, a backend, an Verifier-Driven Evolution Harness (for the text-line segmentation strategy), and an active-learning runtime (for the OCR model).

## Semi-automatic annotation tool pipeline:
In a broad sense, the digitization pipeline is as follows (and it requires manual human annotation at various stages):

Step 1: CRAFT based character detection: In the first step of the pipeline, we detect characters on the page using CRAFT. Sometimes CRAFT makes mistakes. Hence the Layout Mode GUI allows the user to manually add or delete false negatives and false positive character detections respectively.

Step 2: GNN based Binary edge classification: Once the characters are detected, we use a GNN to connect characters in such a way that characters belonging to the same text-line are connected together. But sometimes the GNN makes mistakes. Hence the Layout Mode GUI allows the user to manually add or delete edges which have been incorrectly classified (false negatives and false positives). The GNN pipeline currently only does automatic binary edge classification, but we want the GNN to automatically perform the following tasks too (which are currently only performed by the human manually using the GUI):
- text-region labelling: all characters (nodes) belonging to the text-region should have the same node labels. This is only done manually in the GUI right now.
- text-line orientation annotation: classifying the orientation of the text-line (which decides the read-order of the text-lines based on the script being used). This is only done manually in the GUI right now.


Step 3: Once the graph based text-lines are formed, they are converted to the standard PAGE-XML baselines. These baselines are used, along with the CRAFT heatmaps, and the original images, to get PAGE-XML Coords. The Coords are used to prepare text-line images for inference and training of the local Built-in OCR model. This is our the text-line segmentation strategy, which is not AI based but is based on algorithms and traditional computer vision algorithms. There is no human manual correction involved here as of now (although the text-line orientation annotations in Step 2 do influence the orientation of the text-line images prepared for the OCR model in Step 3)

Step 4: Once the text-line images are prepared for the OCR model (a CNN-BiLSTM-CTC), the OCR model perdicts the text content from the text-line images. This predicted text content contains mistakes. Hence the GUI Read Mode allows users to manually correct the predicted text (measured by Character Error Rate).

## DEVELOPMENT MOTIFS
There are also two main motifs of development we have in the vision:


### Active learning: 
We to iteratively improve AI models (CRAFT,GNN,OCR Models) using manual human annotations, so that the AI models make better subsequent predictions, reducing the burden of manual annotation after each training iteration. Current we are using this to iteratively improve the OCR model, but we aim to do the same for CRAFT and GNN too.

The long-term product target is a manuscript-local active learning loop:

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

#### OCR Active Learning Reality

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

We also do manuscript-local OCR checkpoint promotion inside the runtime (a better model replaces the previous model)


### Verifier-Driven Evolution Harness: 
Any change we make to the pipeline, should pass an external verifier. We are currently using this motif to refine and improve Step 3, i.e the text-line segmentation strategy, but we can adapt this improve any other part of the pipeline too.


#### Current Verifier-Driven Evolution Harness

The repository now has a reusable research harness for step-by-step improvement of a specific pipeline stage with LLM-assisted code changes and external verifier metrics.

The generic pattern is:

1. Define one narrow pipeline stage with explicit inputs and outputs.
2. Keep one checked-in `benchmark_strategy`.
3. Implement one checked-in `proposed_strategy`, with independent code.
4. Run the same external verifiers against both roles.
5. Aggregate the gate evidence.
6. Promote the proposed strategy explicitly inside the research harness only after all gates pass.
7. Adopt a strategy for production app use through a separate explicit command that changes future PAGE geometry generation and the crop metadata honored by production OCR preparation.
8. Keep the old strategy code and history for rollback and future ablations.
9. The implementation in code of the `benchmark_strategy`, `proposed_strategy` and the production app strategy should be independent

This harness is intentionally broader than text-line segmentation. A future agent can adapt it to another pipeline component if it first writes down:

- the exact input contract
- the exact output contract
- the primary quality metrics
- the acceptable regression rules
- the research promotion command, production adoption command, and source-of-truth config

The important design rule is that generated logs are evidence, but checked-in source config is the source of truth for which strategy currently owns the research benchmark role and which strategy the app uses in production. Because generated logs under `app/tests/logs/` are ignored, `scripts/run_precommit_eval.py` also refreshes `docs/pipeline-improvement/text-line-segmentation/strategy-promotion-record.md` as the checked-in promotion evidence summary.

#### Current Harness Instance: Text-Line Segmentation To OCR Crops

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

- benchmark: `local_polygons_stable_unwrap_v1`
- proposed: unset
- production app: `local_polygons_stable_unwrap_v1`

`local_polygons_stable_unwrap_v1` is the current research benchmark after promotion on 2026-05-30 and the current production app strategy after explicit adoption on 2026-05-30. It owns a frozen baseline-local PAGE geometry implementation descended from `local_polygons_v1`, keeps the `0.45` heatmap threshold before local cleanup, and changes the OCR crop representation to the stable arclength/tangent unwrap. Its PAGE `Coords` generation now also handles two production-relevant geometry edge cases: baseline endpoint anchors extend open-line polygons when a corrected baseline reaches beyond heatmap evidence, and ambiguous joined heatmap components are split by nearest baseline before local polygon construction. `legacy_axis_bound_v1` preserves the historical axis-aligned behavior and remains available for rollback and legacy-page fallback behavior. `local_tangent_band_v1` is still registered as historical strategy code, but its research and production role-independent flags are false, so role validation rejects it as a benchmark, proposed, or production pin.

There is currently no configured proposed research strategy. Production and the research harness both call the registered text-line strategy through `apply_text_line_segmentation_strategy(...)` when they need to generate PAGE `TextLine/Coords`. The production GUI first converts the live corrected graph, including manual node and edge edits, into a baseline PAGE XML file and then applies `production_strategy_name`. The research harness starts from checked-in PAGE `Baseline` labels plus the page image and heatmap, then applies the benchmark/proposed strategy role. When those role names point to `local_polygons_stable_unwrap_v1`, both paths reuse the same PAGE `Coords` generation code, but their upstream baseline sources and runtime config are intentionally different.

Production now has a shared OCR crop preparation boundary in `app/recognition/line_segmentation/ocr_crops.py`. App line-image export, local OCR inference, and active-learning revision preparation all read saved PAGE `Coords` and optional strategy metadata through that layer. New `local_polygons_stable_unwrap_v1` layout saves write metadata that asks this layer for stable local-polygon unwrapping. Pages without usable metadata continue through the masked PAGE `Coords` fallback.

Layout mode also has optional intra-line reading-direction annotation. The `q` shortcut records a cross-line cut, resolves it to the final line by component overlap, and stores it in a reading-direction sidecar. Open lines use the local cut tangent to resolve 180-degree ambiguity; circular lines use it to choose both unwrap start station and direction. Missing or stale annotations fall back to script-specific defaults.

The current explicit workflow is:

1. A researcher suggests a new `proposed_strategy`.
2. Agents implement the code and update docs/config.
3. The developer runs or triggers the three pre-commit gates.
4. `scripts/run_precommit_eval.py` writes gate artifacts, aggregate promotion evidence, and the checked-in promotion record.
5. The developer reviews the evidence and runs `scripts/promote_text_line_strategy.py --apply`.
6. The research benchmark role moves forward in checked-in config, while old strategy code remains available for comparison.
7. A separate operator decision runs `scripts/adopt_text_line_strategy_for_app.py --apply` if the app should use a different production strategy for future geometry and crop behavior.

This keeps promotion and production rollout reviewable. The pre-commit path does not silently mutate tracked config after Git has already prepared the commit, and research promotion is not a GUI/app rollout.

#### Current Evaluation And Promotion State

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


#### Broader Research Direction

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

- The app must remain usable while research harness code changes.
- A successful gate run must never silently edit tracked source config.
- Promotions and production adoptions must be explicit, reviewable, and reproducible.
- Generated artifacts must not be the only place where important conclusions live.
- Human-effort reduction should become a first-class logged metric, not only an anecdotal goal.
- The implementation in code of the `benchmark_strategy`, `proposed_strategy` and the production app strategy should be independent.
