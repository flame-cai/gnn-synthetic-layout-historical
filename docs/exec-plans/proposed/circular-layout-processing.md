# Circular OCR Evolution Harness Implementation Plan

## Summary
Build a GUI-free, human-supervised experiment harness for improving circular, curved, and vertical text-line preparation before OCR. The harness must preserve existing horizontal-line behavior, keep PAGE-XML page-space `Coords` separate from OCR-ready unwrapped crops, and gate each iteration against a checked-in accepted baseline.

Implementation is opt-in under `app/tests/circular_OCR_test/` plus supporting `app/` test modules. `scripts/run_precommit_eval.py` may be modified only to add the circular experiment launcher phase/profile.

## Architecture
- Add a small circular experiment package under `app/tests/circular_OCR_test/`:
  - `config.py`: dataset paths, page split, OCR recipe, baseline path, artifact naming.
  - `geometry.py`: PAGE-XML baseline parsing, gnn-format parsing, open-curve normalization, circular topmost cut.
  - `segmentation_strategies.py`: strategy registry and implementations.
  - `pagexml_rewrite.py`: copied PAGE-XML writer that updates only page-space `TextLine/Coords`.
  - `ocr_unwrap.py`: converts page-space `Coords` plus `Baseline` into OCR-ready horizontal crop candidates.
  - `experiment.py`: prepares pages, runs OCR active-learning recipe, records metrics, compares against baseline.
  - `baseline.py`: loads `baseline.json`, compares metric direction and minimum delta.
- Add unittest entrypoints outside or inside that package:
  - `app/tests/test_circular_ocr_unit.py`
  - `app/tests/test_circular_ocr_precommit_e2e.py`
- Reuse existing OCR harness code from `app/tests/recognition_finetuning_experiment.py` and `app/recognition/active_learning.py` rather than duplicating fine-tuning logic.

## Durable Research State
Create these checked-in files under `app/tests/circular_OCR_test/`:
- `CIRCULAR_EXPERIMENT_README.md`: purpose, gates, artifact locations, durable state files.
- `IDEAS.md`: curated surviving idea pool, with one idea id per candidate hypothesis.
- `BASELINE.md`: how `baseline.json` is produced and manually accepted.
- `baseline.json`: current accepted baseline consumed by the gate.
- `RUNBOOK.md`: one-iteration workflow: read idea, implement one hypothesis, run gates, inspect artifacts, report.
- `STRATEGY_CONTRACT.md`: segmentation and unwrapping interfaces, invariants, and artifact boundaries.
- `FAILURE_MODES.md`: initial known failures and later evidence-backed failures.

Initial `FAILURE_MODES.md` entries must include: diagonal tangent over/under-cropping, clockwise/counterclockwise ambiguity, glyph upright ambiguity, topmost-cut mismatch, OCR-confidence misselection, and accidental replacement of page-space `Coords` with unwrapped crop geometry.

## Dataset And Metrics
- Dataset: `app/tests/eval_dataset_v2`.
- Input sources:
  - page images: `images_resized/`
  - heatmaps: `heatmaps/`
  - gnn labels and points: `layout_analysis_output/gnn-format/`
  - PAGE-XML text and baselines: `layout_analysis_output/page-xml-format/`
- Page split must be derived from sorted page ids:
  - fine-tune count: 3, currently `page_2`, `page_3`, `page_4`
  - eval pages: remaining last 2, currently `page_5`, `page_6`
- OCR recipe must match the trusted hybrid recipe:
  - `page_plus_random_history`
  - `history_sample_line_count=10`
  - `width_policy=batch_max_pad`
  - no oversampling
  - no augmentation
  - no LR scheduler
  - Adadelta `lr=0.2`
  - `num_iter=60`
- Report the existing metric family:
  - `curve_metric_value`
  - `first_step_gain`
  - `final_page_cer`
  - `regression_guard_passed`
  - `max_regression`
  - per-page CER rows
  - per-line prediction rows

## Strategy Contract
Segmentation and OCR unwrapping are two separate steps.

Segmentation input:
- page image
- heatmap
- gnn points and text-line labels
- PAGE-XML `Baseline` and `TextEquiv`
- strategy config

Segmentation output:
- copied PAGE-XML with updated page-space `TextLine/Coords`
- page-space polygons only
- metadata including strategy name, line ids, cut point, tangent/normal settings, and source inputs

OCR unwrapping input:
- copied PAGE-XML
- page image
- page-space `Coords`
- `Baseline`
- unwrapping config

OCR unwrapping output:
- OCR-ready horizontal crop images
- `PreparedPageDataset`-compatible manifests
- orientation candidate metadata
- selected candidate and rejected candidate scores

The unwrapped horizontal rectangle must never be written as PAGE-XML `Coords`.

## Initial Strategy Set
Implement two strategies before optimization begins.

`current_implementation_control`:
- Reproduce the current axis-bound behavior as the known-bad control.
- Use current heatmap connected components, gnn labels, page x/y padding, rectangle bridging, and masked bounding-rectangle OCR crops.
- Run this once to create the initial checked-in `baseline.json`.

`local_tangent_band_v1`:
- Treat every line as an ordered open curve.
- If the baseline is circular or nearly closed, cut at the topmost point first.
- Sample local tangents along the baseline.
- Build local normal bands around the baseline so padding means “across the text line” rather than page x/y.
- Use heatmap evidence and gnn-labeled points to keep line-local foreground and reject adjacent-line spillover.
- Produce page-space polygons from the union of local bands.
- Keep horizontal lines behaviorally equivalent by making a straight baseline produce the same kind of horizontal band.

## Orientation Selection
For each OCR crop, generate up to four candidates:
- `forward`
- `reversed`
- `forward_vertical_flip`
- `reversed_vertical_flip`

Run OCR inference on all candidates using the current checkpoint for that step. Select by OCR confidence or uncertainty only. Log:
- candidate image path
- predicted text
- confidence score
- selected candidate name
- rejected candidate names
- selector reason
- line id and page id

Ground-truth text must not be used for orientation selection.

## Gate Behavior
Add a circular pre-commit gate:
- unittest: `python -m unittest app.tests.test_circular_ocr_precommit_e2e -v`
- artifacts under `app/tests/logs/<timestamp>_circular_ocr_eval_dataset_v2/`
- latest pointers:
  - `app/tests/logs/circular_ocr_latest.md`
  - `app/tests/logs/circular_ocr_latest.json`
  - `app/tests/logs/circular_ocr_latest.txt`

Pass/fail:
- load `app/tests/circular_OCR_test/baseline.json`
- read `primary_blocking_metric_name`
- apply `metric_direction`
- require at least `minimum_improvement_delta`
- fail if the metric is missing, NaN, or not improved enough
- print artifact paths on both pass and fail

Baseline JSON must include dataset name, page split, strategy name, baseline status, segmentation config, unwrapping config, metrics, primary metric, direction, minimum delta, date recorded, and command used.

## Pre-Commit Integration
Modify `scripts/run_precommit_eval.py`:
- Add `CIRCULAR_LAYOUT_PHASE`.
- Add skip flag `SKIP_CIRCULAR_LAYOUT_HOOK=1`.
- Preserve existing global `SKIP_EVAL_HOOK=1`.
- Preserve existing full-pipeline and recognition fine-tuning phases.
- Add an experiment profile, preferably `PRECOMMIT_EVAL_PROFILE=circular_layout`, that runs:
  - existing full-pipeline gate on `eval_dataset`
  - new circular layout OCR gate on `eval_dataset_v2`
- Normal repository pre-commit behavior must still be able to run the existing recognition fine-tuning gate.

## Tests
Unit tests:
- gnn-format parser loads points, dims, labels, and detects mismatches.
- PAGE-XML parser extracts `TextEquiv`, `Baseline`, and `Coords`.
- circular baseline cutting chooses the topmost point and records it.
- straight, vertical, curved, and circular baselines all normalize to the same open-curve representation.
- unwrapped crop metadata is separate from page-space `Coords`.
- orientation candidate generator produces the expected four variants.
- orientation selector uses confidence metadata, not ground-truth text.
- baseline comparator handles lower-is-better, higher-is-better, delta, missing metric, and malformed JSON.

E2E gates:
- `python -m unittest app.tests.test_circular_ocr_precommit_e2e -v`
- `python -m unittest discover -s app/tests -p "test_ci_e2e.py" -v`
- experiment profile through `scripts/run_precommit_eval.py`

Do not run `test_recognition_finetuning_e2e.py` by default for this workflow.

## Safety Rules
- Experimental curved segmentation is opt-in for the circular harness.
- GUI save, recognition, and normal PAGE-XML crop behavior remain unchanged unless explicitly enabled.
- Any shared OCR crop changes must include a regression test for normal horizontal PAGE-XML polygon crops.
- If circular metrics improve but the full-pipeline `eval_dataset` gate regresses, the iteration is not acceptable unless the behavior is isolated behind an experiment-only flag.
- The agent must implement one hypothesis per iteration and must not auto-commit during the supervised phase.
- Failed iterations keep artifact evidence and update `FAILURE_MODES.md` with cause, evidence, status, and related `IDEAS.md` id.


# Context:


This is an experiment (or rather an experiment system) which combines the generative capabilities of LLMs with external verifier metrics to perform evolutionary search in python code space, under human supervision.

The purpose of this experiment system is to find a good strategy to process segmentation of circular, curved, and vertical text lines present in historical manuscripts, such that the downstream CER is low, and active learning works. Hence the main improvements to be done are not in the OCR Model, but in how the text-line image data is prepared before the OCR Model.

### INSPIRATION:
Every curved line, when seen locally is a straight line. We can get good information of the curvature using the 'Baseline' in the PAGE-XML (and the gnn-format labels). There are only two types of text-lines circle and curved line (of which straight line is a special case). Once we cut the circle, we want to treat it as a curved line. Hence use the ENGINEERING_DOCTRINE to find invariants, abstraction and tread every line the same (with minimal if/else edge case last mile handing).


### IMPLEMENTATION AND SETUP:
To actually implement and setup the experiment system, we want to take inspiration from `app/tests/test_recognition_finetuning_e2e.py`, in the sense that we would like to experiment with various parts of the pipeline and select the best configuration which gets the best score on external verification metrics (GUI-free headless manner). However instead of working with 'app\tests\eval_dataset', will be working with 'app\tests\eval_dataset_v2', which has different file structure, and needs to be handled differently as mentioned below:

'test_recognition_finetuning_e2e.py' we assume the text line segmentation to be perfect, and directly crop out the lines from the page using the PAGE-XML (the same way the GUI tool does it.) However, as eval_dataset_v2 contains circular text-lines, vertical text-lines, and curvy text-lines - we can assume the labels in the gnn-format (eval_dataset_v2\layout_analysis_output\gnn-format) to be the perfect ground truth. However, the way we segment text-lines using these ground truth labels in gnn-format, and the heatmaps (eval_dataset_v2\heatmaps) works well only for relatively straight, horizontal lines. It fails for text-lines in a circular layout, very curved lines, and vertical lines. Hence _this conversion pipeline_ from gnn-format labels, to the text-line images (similar to the ones we use in test_recognition_finetuning_e2e.py) needs to optimized and evolutionarily searched - because we have the downstream ground-truth text content of each page in the PAGE-XML files in (eval_dataset_v2\layout_analysis_output\page-xml-format).

Note that the PAGE-XML files contain the grount-truth text (TextEquiv), and also the ground-truth polylines (Baseline) based on the gnn-format labels. However the (Coords) are not the groun-truth bounding polygons. These Coords denote the bounding polygons obtained using the current flawed pipeline, which only work well for relatively straight, horizontal lines, and not circular, curved, or vertical lines. 

PcGts
├── @xmlns
├── Metadata
│   ├── Creator
│   └── Created
└── Page
    ├── @imageFilename
    ├── @imageWidth
    ├── @imageHeight
    └── TextRegion*
        ├── @id
        ├── @custom
        ├── Coords
        │   └── @points
        └── TextLine*
            ├── @id
            ├── @custom
            ├── Baseline
            │   └── @points
            ├── Coords
            │   └── @points
            └── TextEquiv
                └── Unicode

For reference gnn-format labels are as follows:
- page_2_dims.txt (heatmap image dimension)
1250.0 1250.0

- page_2_inputs_normalized.txt (coordinate x, coordinate y, font_size)
0.489600 0.077600 0.000004
0.452000 0.078400 0.000004
0.470400 0.078400 0.000005
...


- page_2_inputs_unnormalized.txt (coordinate x, coordinate y, font_size)
612.000000 97.000000 0.005600
565.000000 98.000000 0.005600
588.000000 98.000000 0.006400
...

- page_2_labels_textline.txt (points belonging to the same text-line have the same label)
0
0
0
...


### IMPLEMENTATION DETAILS:
The existing heuristic text-line segmentation logic works well for horizontal lines. It also effectively uses the heatmap, along with additional heuristics to not include text from adjacent (above and below) lines. Please study this heuristic, and attempt to generalize this strategy to any curved text-line (including circular text lines), WITHOUT making it worse on horizontal lines. This heuristic has been tuned for horizontal lines, having in mind that some scripts like devanagari have diacritic marks, matras extend out from the main text line (and which need to be included), but matras and diacritics from adjacent lines need to be excluded. Carefully study this.
- Treat curved text-line as a generalization. That is, a straight line is a special case of the curved line.
- For a circular text line, we will have to 'cut' it at the topmost point, to convert it into a curved line topologically.
- We must make sure this generalization does not affect the performance on horizontal lines in 'eval_dataset'. So the previous tests should pass after this change (eventually)
- When we will convert a curved line or a vertical line to a horizontal line as required, there is a chance that this text-line image will be upside down (as the conversion is ambiguous). To fix this, we can use the _uncertainty_ of the OCR model to decide which orientation is the right one.
- Hence we will actually need to create new copies of the PAGE-XML (with updated Coords, everything else fixed) when trying different strategies. 


### OPTIMIZATIONS:
Hence, in this experiment, we do not want to optimize the OCR model fine-tuning configuration. Instead we want to optimize :
- how vertical, curved, and circular text-line images are prepared from the gnn-format ground-truths.
- how vertical, curved, and circular text-line images are converted to horizontal text-line image rectangles (as required to finetune and run inference by the OCR model)
- the converted horizontal line orientation ambiguity decision using OCR model _uncertainty_. So we will need to try out both orientations.


### EVALUATION, VERIFICATION, EVOLUTIONARY SEARCH:
Every strategy we try, should do well on the following tests:
- pre-trained models test on 'eval_dataset' (with attempted new generalized line-segmentation using gnn predictions and heatmap)
- active learning OCR model test on 'eval_dataset_v2' (attempted new generalized line-segmentation using ground truth gnn-format labels (or ground-truth Baseline) and heatmaps. New copies of the PAGE-XML ground truth will be created atleast temporarily with updated Coords)
- Do not check using the the active learning OCR model test on 'eval_dataset'. As it is time consuming, and can be done later.


### OCR MODE CONFIG:
'eval_dataset_v2' has only 5 pages. We want to use first 3 pages for iterative fine-tuning, and last 2 for verification and metric calculation (which will eventually help us select). Please use the same metrics and the best OCR model config we use in app/tests/test_recognition_finetuning_e2e.py.

### HUMAN INPUT:
This is an experiment which combines the generative capabilities of LLMs with external verifier metrics to perform evolutionary search in python code space. Please prepare this experiment as an expert researcher with good logging, configuration and segmentation strategy handling. Please create a new directory 'circular_OCR_test' for this experiment. The directory will have a file IDEAS.md which both you and the human can read/write/update with new ideas we have to try out. This IDEAS.md will be like a good genetic 'pool'. Only the good ideas survive, the bad ones will be removed. This will allow us to juxtapose ideas and iterate. Please think carefully and prepare this evolutionary experiment system well.
Only make modifications inside 'app/', but feel free to change any code inside it for each experiment. 

### IMPORTANT WORKFLOW INSTRUCTIONS: 
Initially, do not perform multiple experiments fully autonomously. At the start I would like the following workflow:
- we want to have this as a new pre-commit check, along with the existing pre-trained model only check. 'test_recognition_finetuning_e2e.py' will be disabled as it is time-consuming.
- you will then change everything that needs to be optimized, and then I will manually commit. For the commit to pass, performance on the new pre-commit check must have improved than before, and performance on the existing pre-trained check should not worsen.
- once the commit passes. I will then manually update the IDEAS.md file in 'circular_OCR_test', and then I will ask you to again make optimizations based on it. Then I will manually commit again with hope that it passes.
- after a few iterations, i might ask you to go full automatic mode and try out multiple experiments. But we don't want this at this start. 
- please write any other scaffolding documents with instructions other than IDEAS.md which will help with this experiment. This will help not dilute the context, and allow reuse.

Additional workflow details for the initial human-supervised phase:

- Each iteration should be a single, reviewable hypothesis, not a broad sweep. The agent should read `app/tests/circular_OCR_test/IDEAS.md`, pick or propose one specific idea, implement only that idea, run the required gates, and report the metric/artifact evidence. Do not combine unrelated segmentation, unwrapping, orientation-selection, and OCR-policy changes in the same iteration unless the user explicitly asks for a larger combined experiment.
- The user remains the commit owner in this phase. The agent should not commit automatically unless explicitly asked. The expected loop is: agent changes code and/or experiment config, agent runs verification, user reviews the diff and artifacts, user manually commits if the result is worth keeping.
- The experiment-oriented gate should run the existing pretrained full-pipeline gate on `eval_dataset` and the new circular layout OCR gate on `eval_dataset_v2`. It should not run the existing recognition fine-tuning gate on `eval_dataset` by default for this workflow, because that gate is time-consuming and is not the optimization target of this circular-layout experiment.
- The circular gate should compare the current result against a checked-in accepted baseline JSON. After a successful and manually accepted iteration, the human should intentionally update the baseline JSON to the new accepted result before asking for the next optimization. This makes each later iteration compete against the latest accepted strategy rather than an old stale baseline.
- If an iteration improves the circular metric but worsens the pretrained full-pipeline gate on `eval_dataset`, the iteration should be treated as not acceptable for commit unless the user explicitly decides to isolate the behavior behind an experiment-only flag. The default safety rule is that circular improvements must not regress the existing horizontal-line workflow.
- If an iteration fails, keep the evidence. The agent should summarize why it failed, what artifact or metric proved the failure, and whether the idea should be rejected, revised, or deferred. Failed ideas are useful research evidence and should not simply disappear.
- The first several iterations should avoid autonomous experiment chains. Full automatic mode should only begin after the experiment harness, baseline comparison, artifact logging, and failure-mode documentation have proven stable across a few manual iterations.

Recommended reusable scaffolding documents under `app/tests/circular_OCR_test/`:

- `CIRCULAR_EXPERIMENT_README.md`: entry point for humans and agents. It should describe the experiment purpose, the two gates to run for this workflow, where artifacts are written, and which files are durable checked-in research state.
- `IDEAS.md`: human-maintained idea pool. We only keep the good ideas, and will remove the bad ones altogether.
- `BASELINE.md` and `baseline.json`: explanation and machine-readable record of the current accepted baseline. The JSON should be the file the circular gate reads for pass/fail comparison.
- `RUNBOOK.md`: exact one-iteration procedure. It should explain how to choose one idea, run the gates, inspect artifacts, report metrics, and decide whether the user should update the baseline.
- `STRATEGY_CONTRACT.md`: interface and invariants for segmentation and unwrapping strategies. It should state that PAGE-XML `Coords` are page-space segmentation geometry, while unwrapped OCR crops are separate artifacts.
- `FAILURE_MODES.md`: prioritized list of known and discovered failure modes. Each entry should include a short name, symptom, evidence, likely cause, priority, related idea ids from `IDEAS.md`, and current status. Initial entries should include diagonal tangent over/under-cropping near 45/135/225/315 degrees, clockwise/counterclockwise ambiguity, glyph upright ambiguity, topmost-cut mismatch, OCR-confidence misselection, and accidental replacement of page-space `Coords` with unwrapped crop geometry.

The purpose of these scaffolding documents is to make the workflow reusable without forcing every future prompt to restate the whole research protocol. They should be concise and operational. The docs should preserve decisions, evidence, and failure modes; they should not become a place for speculative implementation details that are not tied to a testable strategy.


We must make sure that each strategy we try does not break any upstream or downstream code. Although this experiment is headless, we must also ensure any GUI functioning will not break, and there will be no unintended effects.

### ADDITIONAL CLARIFICATIONS FROM FEASIBILITY REVIEW (2026-05-05):

These clarifications are additive and should be treated as binding design decisions for future implementation work.

#### Segmentation vs OCR unwrapping:
The PAGE-XML `Coords` and the OCR-ready horizontal text-line image are two different artifacts and must be produced in two separate steps.

Step 1 is segmentation in page coordinate space. This step updates the `TextLine/Coords` polygons in copied PAGE-XML files. These `Coords` remain page-space polygons that describe where the text line is on the manuscript image.

Step 2 is unwrapping for OCR. This step consumes the page image plus the updated `Coords` and `Baseline`, then creates horizontal OCR-ready line images and metadata. The unwrapped horizontal rectangle must not be treated as the PAGE-XML `Coords`, because it no longer represents the original page-space layout.

#### Circular text cut convention:
For circular text lines, cut at the topmost point before treating the line as an open curved line. This is acceptable for `eval_dataset_v2` because the dataset was annotated with this convention. Future code should still record the chosen cut point in run metadata so failures can be diagnosed when a page or future dataset violates this convention.

The circular processing order should be:

1. identify the circular baseline or circular line geometry,
2. cut it at the topmost point,
3. run or generalize segmentation in the resulting open-curve frame,
4. update page-space `Coords`,
5. unwrap the segmented curved line into OCR-ready horizontal crop candidates,
6. use OCR uncertainty or confidence to select the best orientation candidate.

#### Orientation ambiguity:
Trying only a simple image flip may not fully solve clockwise versus counterclockwise ambiguity. For circular and strongly curved text, candidate generation should explicitly represent reading direction and glyph orientation. At minimum, try forward and reversed reading order. If glyph upright direction is still ambiguous, also try vertically flipped variants, yielding up to four candidates:

- forward,
- reversed,
- forward plus vertical flip,
- reversed plus vertical flip.

The OCR uncertainty or confidence selector should choose among these candidates. The selector decision, candidate scores, selected candidate name, and rejected candidate names should be logged in the experiment artifacts.

#### Observed current failure mode:
The current segmentation behavior appears to work reasonably well on horizontal and vertical portions of circular text when judged in the global page frame. It crops too much or too little near diagonal tangent regions, especially around roughly 45, 135, 225, and 315 degrees. This supports the hypothesis that global x/y padding and rectangle-bridging are the wrong abstraction for curved text. New strategies should use local tangent and local normal directions along the baseline so padding means "along the line" and "across the line" rather than "page x" and "page y".

#### Experiment directory:
The experiment system, checked-in baseline file, strategy configs, logs documentation, and `IDEAS.md` should live under:

    app/tests/circular_OCR_test/

This directory should be used for reusable scaffolding and checked-in research state. Generated run artifacts may be written under `app/tests/logs/` or a run-local subdirectory, but durable baselines and decisions needed for commit gating must be checked in under `app/tests/circular_OCR_test/`.

#### Pre-commit integration:
The circular layout experiment should become a third phase in the explicit pre-commit launcher, alongside the existing full-pipeline gate and the existing OCR fine-tuning gate. It is acceptable to modify:

    scripts/run_precommit_eval.py

for this integration, even though most experiment implementation should stay under `app/`.

The intended three launcher phases are:

1. existing pretrained full-pipeline gate on `eval_dataset`,
2. existing recognition fine-tuning gate on `eval_dataset`,
3. new circular layout OCR gate on `eval_dataset_v2`.

Clarification: for this circular-layout experiment workflow, do not run phase 2 by default. The existing recognition fine-tuning gate on `eval_dataset` should remain available for the normal repository pre-commit launcher, but the experiment-oriented command/profile should run only:

1. the existing pretrained full-pipeline gate on `eval_dataset`, and
2. the new circular layout OCR gate on `eval_dataset_v2`.

In practical terms, implementation may either add a dedicated experiment launcher/profile or document that circular-layout experiment commits should run with the existing recognition fine-tuning phase skipped, for example via `SKIP_RECOGNITION_FT_HOOK=1`. The circular experiment must still preserve the existing recognition gate code path; it should not delete or weaken that gate.

The third phase should have its own skip environment variable, for example `SKIP_CIRCULAR_LAYOUT_HOOK=1`, and should print its own artifact paths when it passes or fails.

#### Circular gate pass/fail rule:
The circular gate should require improvement over a checked-in baseline JSON for the current strategy. Do not use fixed blocking thresholds yet.

Before the first circular-layout implementation, create the checked-in baseline from the current implementation, even though the current implementation is known to perform badly on circular and strongly curved lines. IMPORTANT: This baseline is the control condition: it represents the repo's current behavior before circular-layout strategy work starts. The first improved strategy must beat this control baseline before it is considered acceptable.

The baseline JSON should be stored in `app/tests/circular_OCR_test/` and should record at least:

- dataset name,
- page split,
- strategy name, for example `current_implementation_control`,
- baseline status, for example `known_bad_control_baseline`,
- relevant segmentation and unwrapping config,
- metric names (use the same success metrics we have for app/tests/test_recognition_finetuning_e2e.py),
- baseline metric values,
- primary blocking metric name,
- metric direction, for example `lower_is_better`,
- minimum improvement delta, so tiny nondeterministic metric noise does not pass the gate,
- date recorded,
- command used to produce the baseline.

The circular experiment should report the same success metrics used by `app/tests/test_recognition_finetuning_e2e.py`, including the curve metric, first-step gain, and final-page CER style metrics, and it should also follow a similar page-wise finetuning recipe.

For the first implementation, the circular gate should compare the current run against that checked-in current-implementation baseline and pass only if the selected primary metric improves according to the metric direction recorded in the baseline file. If multiple metrics are reported, the gate must still have one explicit primary blocking metric to avoid ambiguous pass/fail behavior.

#### Dataset split:
For `eval_dataset_v2`, use the first 3 pages for iterative OCR fine-tuning and the last 2 pages for verification and metric calculation. With the current page names, this means:

- fine-tune pages: `page_2`, `page_3`, `page_4`
- evaluation pages: `page_5`, `page_6`

The code should derive these from sorted page ids through config rather than hard-coding only these names inside algorithmic code.

#### Documentation and safety:
When this plan is converted into a full implementation plan, preserve the existing GUI behavior by default. Experimental segmentation and OCR unwrapping should be opt-in for the circular test harness until it is proven not to regress the existing `eval_dataset` full-pipeline gate.

Any implementation that changes OCR crop generation shared by the GUI must include tests that prove existing PAGE-XML polygon crops still work for normal horizontal lines.
