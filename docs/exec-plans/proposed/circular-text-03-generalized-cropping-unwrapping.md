This research plan will enable us to combine the generative capabilities of LLMs with external verifier metrics to perform step by step evolutionary search in python code space - with the purpose to improve how this application segments text-lines (using the resized_images, heatmaps and the respective predictions of the GNN i.e the <Baseline points="x1,y1 x2,y2 .."/> in page-xml). In other words, we want to improve the part of the pipeline which converts the GNN predictions <Baseline points="x1,y1 x2,y2 .."/> in page-xml) to the text-line images which are fed to the downstream OCR model for fine-tuning or inference.

The application already has such a text-line segmentation strategy. We will call this the "benchmark_strategy". However this strategy is flawed and works only for horizontal text-lines (not circular, curved, or vertical lines). To fix these flaws, we will be implementing a new "proposed_strategy". The goal is that if the new "proposed_strategy" passes 3 pre-commit external verifier checks (more on these verifier checks later), it will be promoted, and will become the new "benchmark_strategy", and then finally the git commit will happen. This will then allow us to go to the next round, where we will try an even newer "proposed_strategy" and check it's performance against the new "benchmark_strategy" using the same external evaluator checks, and promote it if the checks pass, and git commit again. Hence we would like to overhaul the current existing pre-commit check mechanism.


We will implement this plan in the following step:

Step 1: Understand the code, and precisely scope what it is that needs ablations to be iteratively improved. (IMPLEMENTED)

Step 2: Setup 3 external verifier precommit checks. Each verifier will have 2 runs, ablating and comparing "proposed_strategy" vs "benchmark_strategy". Each of the 3 external-verifier pre-commit checks should use the exact same implementation for each ablation strategy. (IMPLEMENTED)

Step 3: Implement the first iteration of the "proposed_strategy", with the aim to handle circular, curved and vertical lines better, while maintaining performance on horizontal lines. (TO BE IMPLEMENTED)

Step 4: Setup strategy promotion mechanism and configuration (from "proposed_strategy" promotes to "benchmark_strategy") if pre-commit checks passes. (NOT TO BE IMPLEMENTED RIGHT NOW)

  

### STEP 1 (IMPLEMENTED)

In this step, you should understand the code, and precisely scope what part of the pipeline is will be needing ablations to be iteratively improved:

We want to improve how this application segments text-lines (using the resized_images, heatmaps and the respective predictions of the GNN i.e the <Baseline points="x1,y1 x2,y2 .."/> in page-xml). In other words, we want to improve the part of the pipeline which converts the GNN predictions (<Baseline/> in page-xml) to the text-line images which are fed to the downstream OCR model for fine-tuning or inference.

Understand how the GNN predictions (<Baseline points="x1,y1 x2,y2 .."/> in page-xml)  are converted to text-line images, which are then fed to the downstream OCR model for fine-tuning or inference. We want to improve precisely this part of the pipeline.

Once done understanding, edit and refactor the code, such that it becomes modular and ablation friendly. Regarding this, it is very important to keep in mind that each of the 3 external-verifier pre-commit checks should use the _exact same implementation_ for each ablation strategy. So we want to setup the code in a way that when we will implement the "proposed_strategy", we will reuse the code of the "proposed_strategy" and "benchmark_strategy" for all 3 pre-commit checks.

### STEP 2 Implement 3 external verifier checks (IMPLEMENTED)

I want you to overhaul the pre-commit evaluation, with one primary motive: Perform ablations on the text-line segmentation strategies - "benchmark_strategy" vs "proposed_strategy" new generalized text-line segmentation. In the future, when we want to try a newer version of text-line segmentation, the "proposed_strategy" will become the "benchmark_strategy", and the newer version will become the "proposed_strategy".

This will allow us to iteratively improve with each git commit in this branch. Hence we want to create overhaul the evaluation framework with the motto:

"every proposed will become a baselines, only to be replaced by a new proposed in the future."

The 3 external verifier checks which we want to implement are:

- pre-trained-pipeline gate (2 tests on 'eval_data', "benchmark_strategy" vs "proposed_strategy" ablation)

- active-learning-ocr-finetuning gate (2 tests on 'eval_data', "benchmark_strategy" vs "proposed_strategy" ablation)

- circular-layout-ocr-finetuning gate (2 tests on 'eval_data_2', "benchmark_strategy" vs "proposed_strategy" ablation)

We already have pre-trained-pipeline gate, and active-learning-ocr-finetuning gate implemented, and will need to modify them to fit this new evaluation framework. However, circular-layout-ocr-finetuning gate is not yet implemented, so we need to implement this from scratch:

  

#### implementation details:
##### pre-trained-pipeline gate

This gate works with tests\eval_dataset.

Modify this gate such that we scope out which part of the pipeline we are improving upon iteratively and make it modular such that doing ablations is easy.

##### active-learning-ocr-finetuning gate

This gate works with tests\eval_dataset.

Modify this gate such that we scope out which part of the pipeline we are improving upon iteratively and make it modular such that doing ablations is easy.

##### circular-layout-ocr-finetuning gate

This gate works with tests\eval_dataset_v2. 'eval_dataset_v2' has only 5 pages. We want to use first 3 pages for iterative fine-tuning, and last 2 for verification and metric calculation (which will be used to compare "benchmark_strategy" with "proposed_strategy"). Other than this difference, this test is very similar to active-learning-ocr-finetuning gate, so please use the same metrics and the same OCR model fine-tuning config and recipe.

Write this gate such that we scope out which part of the pipeline we are improving upon iteratively and make it modular such that doing ablations is easy.

#### What success means for each gate:

- pre-trained-pipeline gate - "proposed_strategy" as good as or better than "benchmark_strategy" (slightly inferior also OK)

- active-learning-ocr-finetuning gate - "proposed_strategy" as good as or better than "benchmark_strategy" (slightly inferior also OK)

- circular-layout-ocr-finetuning gate - "proposed_strategy" should be strictly better than "benchmark_strategy" (slightly inferior NOT OK)


### STEP 3 Implement Proposed_strategy (Implementing this)

We want to implement a proposed strategy which will take in the <Baseline points="x1,y1 x2,y2 .."/> from the page-xml, the heatmap, the original image, and then output text-line images in a format which can be fed to the downstream OCR model for finetuning or inference.

The proposed strategy should be able to handle vertical text, curved text, and circular text, along with standard horizontal text. For this our main inspiration is:

- Every curved line, when seen locally is a straight line. We can get good information of the curvature using the 'Baseline' in the PAGE-XML

- straight line is a special case of curved line

Hence we want this proposed_strategy to be a generalization of the current benchmark_strategy. In other words, the benchmark_strategy should be a special case of the proposed_strategy.

##### heuristic-adjacent line-text-cropping:

We will want to use a generalized version of the heuristic-adjacent line-text-cropping which the benchmark_strategy uses:

This heuristic smartly crops out text coming from the "top" or "bottom" from adjacent lines. But generalize this for curved lines, as this heuristic has been tuned for horizontal lines, having in mind that some scripts like devanagari have diacritic marks, matras extend out from the main text line (and which need to be included), but matras and diacritics from adjacent lines need to be excluded (they come from the "top and bottom"). Carefully study this.

##### OCR unwrapping

For horizontal line, we assume no OCR unwrapping would be required. For vertical line OCR unwrapping would just be rotation. For curved lines we will need an OCR unwrapping strategy.

#### How to handle each type of lines:

Hence use the ENGINEERING_DOCTRINE to find invariants, abstraction and treat every line the same (with minimal if/else edge case last mile handing).

We want to handle all lines in two steps:

STEP 1: heuristic-adjacent-line-text-cropping (a more generalized form of benchmark_strategy)

STEP 2: OCR unwrapping (unwrapping the curve if present, and set background to median page color)

Hence the workflow should be something like:

heuristic-adjacent-line-text-cropping and OCR unwrapping are two separate steps.

heuristic-adjacent-line-text-cropping input:

- page image

- heatmap

- PAGE-XML `Baseline` and `TextEquiv`

- strategy config

  
heuristic-adjacent-line-text-cropping output:

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

  
The unwrapped horizontal rectangle must never be written as PAGE-XML `Coords`.

The right orientation after conversion of curved lines too is ambiguous. We need orientation handling mechanism.

##### Horizontal lines:

We will use the same heuristic-adjacent-line-text-cropping. These lines should be handled exactly like how the benchmark_strategy handles them, but decomposed into the two required steps:

STEP 1: heuristic-adjacent-line-text-cropping

STEP 2: OCR unwrapping

##### Vertical lines:

heuristic-adjacent-line-text-cropping needs to be modified such that it works with Vertical lines. Then these need to be converted to horizontal lines. The right orientation after conversion is ambiguous. We need orientation handling mechanism.

STEP 1: heuristic-adjacent-line-text-cropping

STEP 2: OCR unwrapping

##### Curved lines:

STEP 1: heuristic-adjacent-line-text-cropping

This needs to be modified such that it works with curved lines (and lines of shape S and C and U etc).

The current version of the heuristic-adjacent-line-text-cropping crops too much or too little near diagonal tangent regions, especially around roughly 45, 135, 225, and 315 degrees. This supports the hypothesis that global x/y padding and rectangle-bridging are the wrong abstraction for curved text. New strategies should use local tangent and local normal directions along the baseline so padding means "along the line" and "across the line" rather than "page x" and "page y". Once heuristic-adjacent-line-text-cropping is done (we will get the <Coords/> in the page-xml and we can create new page-xmls with everything else the same but with new <Coords/>). These `Coords` remain page-space polygons that describe where the text line is on the manuscript image.

STEP 2: OCR unwrapping

Step 2 is unwrapping for OCR. This step consumes the page image, the updated `Coords`, and the `Baseline` and then then creates horizontal OCR-ready line images and metadata. The horizontal OCR-ready line images should contain ONLY the contents of the respective line's new `Coords` in an unwrapped transformed way. Once the new `Coords` are transformed and unwrapped, to we will fit a tight enclosing rectangle around them.

The background of this rectangular horizontal OCR-ready line images (the part outside the transformed unwrapped `Coords`) should be set to the median background color of the page the text-line is from.


##### Circular lines:

We will cut a circle always at the top-most point, as that's where we start annotating the text.

Once we cut the circle, we want to process it just like a curved line.

STEP 1: heuristic-adjacent-line-text-cropping

STEP 2: OCR unwrapping

#### orientation handling mechanism

After the two steps (heuristic-adjacent-line-text-cropping and OCR unwrapping), the step of orientation handling still remains, as there is a chance that this text-line image will be upside down (as the conversion is ambiguous).

There are 4 possible orientation ambiguities, but we can narrow them down using domain knowledge which can be configured. Right now we would be processing Sanskrit text lines. So:

- reading order will always left to right

- always clockwise for circular layouts

This info, which will be configurable will be used to reduce the ambiguity.

But some ambiguity will still remain. To fix this perhaps we can run the OCR model on all possible orientations, and then "smartly" select the right one based on the statistics OCR model outputs, and the OCR model uncertainty.

To get these statistics, we can also perhaps use the three page training data in "eval_dataset_v2", and the fact that predictions of the OCR model on correctly oriented lines, would be less that those with incorrectly oriented lines. Thus we can use this, to get statistics of the correct orientations.

### Step 4: (Not To be implemented right now)

Setup strategy promotion mechanism and configuration (from "proposed_strategy" promotes to "benchmark_strategy") if pre-commit checks passes. Perform clean up and additional checks. Once everything is done, prompt use to manually commit. This should trigger the pre-commit checks, and if they pass, the strategy promotion will happen. If they fail, we should be able to see good logs on what failed. Write good configuration files for the strategy promption framework.

  

# IMPORTANT NOTE: PAGE Baseline Out-And-Back Topology
Make sure to keep this in mind!

## PAGE Baseline Out-And-Back Topology

Some PAGE `Baseline` polylines in `eval_dataset` and `eval_dataset_v2` are not single start-to-end reading paths. They are out-and-back paths.

The topology is:

    p0, p1, p2, ..., pt, ..., p2, p1

The baseline first moves outward from `p0` to a turning point `pt`, then retraces the same path in reverse.

This means the full baseline point sequence should not always be treated as the intended single reading path. Any code that follows the baseline in order may process the same line twice unless it handles this topology.

## Topological Handling

First detect the out-and-back split point `t`.

A candidate split point `t` is valid when the points after `t` mirror the points before `t` in reverse order:

    p[t + 1] ≈ p[t - 1]
    p[t + 2] ≈ p[t - 2]
    p[t + 3] ≈ p[t - 3]
    ...

After finding the split point, use only the forward path:

    p0, p1, p2, ..., pt

Record in metadata that the baseline was normalized, including:

- original point count
- normalized point count
- split index
- mirror-match tolerance
- mean mirror distance
- max mirror distance

## Circle vs Open Path

After the out-and-back split, classify topology using the forward path only.

If the forward path returns to its start:

    p0 ≈ pt

then it is a closed path.

If the forward path ends elsewhere:

    p0 != pt

then it is an open path.

So the topology-only rule is:

    1. Detect out-and-back mirroring.
    2. Keep the forward path `p0 ... pt`.
    3. If `distance(p0, pt)` is within a small tolerance, classify as closed/circular.
    4. Otherwise classify as open/non-circular.

This rule distinguishes closed circular baselines from open baselines without relying on bounding boxes, tangent rotation, or visual appearance.

# IMPORTANT NOTE: 
We must use a generalized version of heuristic-adjacent-line-text-cropping (which respect line curvature), and then unwrap the mask as per the line curvature. Once unwrapped, we set the background to median page color. Hence the final processed text-line images, should have a foreground (unwrapped mask), and background (median page color)

## Additional Notes:

Please clean up existing pre-commit checks and remove stale code. We only want the new 3 pre-commit checks mentions above going ahead.

While doing this overhaul, please clean up any stale code if you find any. Follow ENGINEERING_DOCTRINE.md to keep the code base maintainable, and future proof. Do not be afraid to do a big overhaul if you think it will be more future proof, and more in spirit of this "benchmark_strategy" vs "proposed_strategy" ablation vision. MORE IMPORTANTLY, note that the current evaluation suite will only test the "benchmark_strategy" vs "proposed_strategy" in the context of text-lines segmentation (conversion of GNN output to OCR Model input). However, in the future "benchmark_strategy" vs "proposed_strategy" can be any ablation in the entire pipeline - you may also get more context regarding this from EVAL.md, and VISION.md. Do not let EVAL.md corrupt your context.

Refer to AGENTS.md to know common hiccups like which conda environment to use, how to fix for unicode output errors in windows, where to write.

Also please think critically and try to find flaws, bugs or unexpected subtle effects which might happen in upstream code or downstream code due to this implementation. Please ask me any clarifications if you feel the need to.

use conda environment gnn_layout

avoid windows permission denied error, write everything to the current directory only.no saving in C:\\temp


# Implement Local Tangent Band Segmentation And OCR Unwrapping (this is just a general template. Not precise instructions. Feel free to make adjustments as required)

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This document is maintained in accordance with `PLANS.md` from the repository root.

This is the third plan in the circular text support series. It depends on the shared strategy API from `docs/exec-plans/proposed/circular-text-01-strategy-interface.md` and the ablation gates from `docs/exec-plans/proposed/circular-text-02-ablation-gates.md`. It uses `docs/exec-plans/proposed/circular-text-support.md` as the research source and must not overwrite that file.

## Purpose / Big Picture

After this change, the repository will have a first proposed text-line segmentation strategy named `local_tangent_band_v1`. The strategy will handle horizontal, vertical, curved, and circular text using the same geometric idea: every curved line is locally straight, so padding and cleanup should be measured along the local tangent and normal of the PAGE `Baseline`, not only along global page x/y axes.

The observable behavior is that `local_tangent_band_v1` can be selected as the proposed strategy in the ablation gates. It should preserve horizontal-line performance on `eval_dataset` while improving circular-layout OCR fine-tuning on `eval_dataset_v2`.

## Progress

- [x] (2026-05-09 22:31 IST) Read the research source and identified required behavior: local tangent/normal bands, circular top-point cut, separate OCR unwrapping, vertical/curved handling, median-background masking, orientation candidate selection, and detailed metadata.
- [ ] Implement `local_tangent_band_v1` as a registered strategy under `app/recognition/line_segmentation/`.
- [ ] Implement separate OCR unwrapping that consumes copied PAGE-XML `Coords` plus `Baseline`, without writing unwrapped rectangles as PAGE `Coords`.
- [ ] Add orientation candidate generation and deterministic selection metadata.
- [ ] Add supervised orientation calibration for labeled fine-tuning pages and ground-truth-free orientation inference for held-out validation pages.
- [ ] Add synthetic unit tests for horizontal, vertical, curved, and circular baselines.
- [ ] Run all ablation gates and record whether the proposed strategy meets the plan 02 comparison rules.

## Surprises & Discoveries

- Observation: the current legacy algorithm contains a vertical branch, but its core polygon construction still depends on axis-aligned boxes and rectangle bridging.
  Evidence: `app/segment_from_point_clusters.py` and `src/gnn_inference/segment_from_point_clusters.py` contain `detect_line_type(...)`, `analyze_and_clean_blob(...)`, and `get_bboxes_for_lines(...)`, then draw axis-aligned rectangles into a mask before extracting a contour.

- Observation: the current OCR crop preparation masks a PAGE-space polygon into an axis-aligned bounding rectangle and fills background with the page median color.
  Evidence: `app/recognition/pagexml_line_dataset.py::_masked_line_crop(...)` computes `cv2.boundingRect(polygon)`, fills a new image with `np.median(processing_image)`, and copies only pixels inside the shifted polygon mask.

- Observation: the circular OCR fine-tuning gate has labels on the first three pages, so orientation choice can be supervised on those pages without leaking validation labels.
  Evidence: plan 02 defines `eval_dataset_v2` with fine-tune pages `page_2`, `page_3`, and `page_4`, and evaluation pages `page_5` and `page_6`. The PAGE-XML for the fine-tune pages contains `TextEquiv/Unicode` ground-truth text.

## Decision Log

- Decision: `local_tangent_band_v1` must write page-space PAGE `Coords` only, and OCR unwrapping must be a separate step.
  Rationale: PAGE `Coords` describe location on the manuscript page. An unwrapped OCR rectangle is a recognition input representation and loses page-space geometry.
  Date/Author: 2026-05-09 / Codex

- Decision: circular baselines should be cut at the top-most point by default.
  Rationale: the research source says circular annotation starts at the top, and a stable cut point makes unwrapped line images deterministic.
  Date/Author: 2026-05-09 / Codex

- Decision: orientation selection should write all candidate scores to metadata, even when the selected orientation is obvious.
  Rationale: circular and vertical text introduce ambiguity. Debugging bad OCR output requires knowing which candidates were considered and why one was selected.
  Date/Author: 2026-05-09 / Codex

- Decision: use PAGE ground-truth text only to calibrate orientation on labeled fine-tuning pages, never to choose orientation on held-out validation pages.
  Rationale: the first three circular fine-tuning pages have ground-truth labels. For those pages, each orientation candidate can be OCR'd and compared against the known line text; the correct candidate should have lower line CER than wrong orientations. That creates supervised labels and calibration statistics. During validation or inference, the selector may use the calibration learned from training pages plus OCR confidence and uncertainty features, but it must not compute CER against validation ground truth.
  Date/Author: 2026-05-09 / Codex

## Outcomes & Retrospective

Not yet implemented. At completion, record the final geometry parameters, representative metadata snippets, and benchmark/proposed results from the three ablation gates.

## Context and Orientation

The shared strategy API from plan 01 accepts a page image, heatmap, source PAGE-XML with `Baseline` and text, a strategy config, and writes copied PAGE-XML with updated page-space `Coords`.

The ablation gates from plan 02 run benchmark and proposed strategies through the same implementation path. This plan supplies the first real proposed strategy: `local_tangent_band_v1`.

Important definitions:

A tangent is the local direction of travel along a baseline. On a horizontal line it points left-to-right. On a curved line it changes gradually.

A normal is the direction perpendicular to the tangent. Padding along the normal means "above and below the line" in the line's own local coordinate system.

A local tangent band is a polygon or mask built around a baseline by measuring distance along tangents and normals. For Sanskrit manuscript text, the band must include matras and diacritics belonging to the line while excluding strokes from adjacent lines.

Unwrapping means sampling pixels from a curved or vertical text-line region and placing them into a horizontal OCR-ready image. Unwrapping changes image representation for OCR only. It must not replace PAGE `Coords`.

Orientation selection means choosing which direction an unwrapped line image should be read. Horizontal Sanskrit is left-to-right. Circular layouts in this project default to clockwise reading order. Even with those assumptions, some vertical and curved cases may be upside down after unwrapping.

In the circular OCR fine-tuning gate, orientation selection has two different contexts. Fine-tuning pages are labeled pages, so their PAGE-XML `TextEquiv/Unicode` text may be used to discover the correct orientation for those same training crops. Validation pages are held-out pages, so their ground-truth text must not be used for orientation choice. Line CER means character error rate: the Levenshtein edit distance between OCR prediction and ground-truth text divided by the ground-truth length. The right orientation should normally have lower CER than wrong orientations, because the OCR model should read the correctly oriented image more accurately.

## Plan of Work

Implement `local_tangent_band_v1` under the strategy package created in plan 01:

    app/recognition/line_segmentation/local_tangent_band.py

Register it in:

    app/recognition/line_segmentation/registry.py

The strategy should parse every PAGE `TextLine` with non-empty `Baseline` and text. For each line, parse the baseline into page-space points, remove duplicate consecutive points, and resample it at stable arc-length intervals. Start with a default sample spacing of 6 pixels, configurable as `baseline_sample_spacing_px`.

For every sampled baseline station, compute a tangent vector from neighboring points and a normal vector by rotating the tangent by 90 degrees. The implementation must handle short baselines by falling back to the first and last distinct points. If tangent length is zero, skip that station and record the skip in metadata.

Load the page image and heatmap using the same image helpers used by the legacy strategy. Resize the heatmap to page-image dimensions before extracting connected components. For each heatmap box, compute its center and assign it to the nearest baseline station among all text lines if the normal-distance and tangent-distance checks pass. Keep per-line lists of assigned boxes and record unassigned box counts.

Construct page-space `Coords` for each line using local bands. A concrete first implementation can do this:

1. Project each assigned heatmap box center into the nearest baseline station's local tangent/normal frame.
2. Estimate normal extents from assigned boxes plus configurable padding. Use defaults that mimic legacy behavior for horizontal lines: normal padding near 0.7 of local component height and tangent padding near 0.5 of local component width.
3. Build a top polyline by offsetting baseline stations along the positive normal and a bottom polyline by offsetting along the negative normal.
4. Join top and reversed bottom into one page-space polygon and clip points to the page bounds.
5. If no heatmap boxes are assigned, fall back to a minimum-width band around the baseline and mark the line with `fallback_reason="no_assigned_heatmap_boxes"`.

The implementation may use `shapely` to validate and simplify polygons if it is already available in the environment, because `app/recognition/pagexml_line_dataset.py` already imports it. Do not add a new dependency. If the polygon is invalid, repair with `buffer(0)` and record `polygon_repaired=true`.

For circular baselines, detect likely closure. Use a configurable rule such as:

    close_distance_px <= max(24, 0.05 * baseline_arc_length)

or total tangent rotation above a high threshold. When a baseline is circular, rotate the baseline point sequence so the first point is the top-most point, meaning minimum page y and then minimum x as tie-breaker. If `reading_direction="clockwise"` is configured, ensure the unwrapped sampling direction follows clockwise order. Record `is_circular`, `cut_point`, `cut_policy="top_point"`, `closed_distance_px`, and `reading_direction` in line metadata.

For vertical baselines, do not special-case the geometry. A vertical line should naturally have a vertical tangent and horizontal normal. Only orientation candidates later need special handling.

For curved baselines, avoid global rectangle bridging. If a line has separated assigned components, connect them along the baseline band rather than drawing global x/y rectangles. The band polygon itself is the bridge.

Implement separate OCR unwrapping in a module such as:

    app/recognition/line_segmentation/unwrap.py

Expose a function shaped like:

    prepare_unwrapped_page_line_dataset(
        pagexml_path: Path,
        page_image_path: Path,
        output_root: Path,
        unwrap_config: Mapping[str, object] | None = None,
    ) -> PreparedPageDataset

This function should be compatible with `PreparedPageDataset` and `PreparedLineRecord` from `app/recognition/pagexml_line_dataset.py`. It should parse `Coords`, `Baseline`, and `TextEquiv`, create one OCR-ready image per text line, write the same flat `finetune_dataset/test/word_*.png`, `gt.txt`, `image-format/<page>/...`, and `manifest.json` layout, and include unwrapping metadata in the manifest.

For horizontal `legacy_axis_bound_v1`, the unwrap mode can remain `axis_aligned_mask_v1`, equivalent to the current `_masked_line_crop(...)`. For `local_tangent_band_v1`, add `baseline_ribbon_v1`:

1. Resample the baseline in reading order.
2. For each output x-coordinate, choose the corresponding baseline station by arc length.
3. For each output y-coordinate, sample along the station normal within the selected top/bottom band width.
4. Use `cv2.remap` or equivalent interpolation to sample the page image.
5. Mask pixels outside the page-space `Coords` polygon and fill them with the median background color of the source page.
6. Tight-crop the unwrapped result to the non-background content with a small configurable margin.

The median background fill must use the page median grayscale value for grayscale output, matching the current crop behavior. If the app needs RGB later, define that as a separate config rather than mixing output modes.

Implement orientation candidate generation. A first deterministic version should support:

- `identity`
- `rotate_180`
- `rotate_90_clockwise`
- `rotate_90_counterclockwise`
- `reverse_path`, which unwraps the baseline in the opposite order

Use config to narrow candidates:

    script_direction="left_to_right"
    circular_reading_direction="clockwise"
    orientation_selection="ocr_confidence"

Add a supervised calibration path for labeled fine-tuning pages. For `eval_dataset_v2`, the first three pages are fine-tuning pages and their ground-truth line texts are available. For each line on those pages, generate every allowed orientation candidate, run the current OCR scorer on each candidate, and compute line CER against that line's PAGE `TextEquiv/Unicode` text. Select the candidate with the lowest line CER as the supervised oracle orientation for that training line. If two candidates tie on CER, break ties using higher OCR confidence, then lower uncertainty, then the configured reading-direction prior, then candidate name order. Store the full candidate list and the selected oracle candidate in `orientation_metadata.json`.

Use those supervised oracle decisions to compute calibration statistics that can be applied later without labels. At minimum, record per-candidate and aggregate features such as mean OCR confidence, confidence margin between best and second-best candidate, prediction length ratio against the training label, blank ratio, entropy or uncertainty if available, geometric line class, rotation family, and whether the candidate matched the supervised oracle. This can start as a transparent rule-based calibration rather than a learned classifier: for example, choose the candidate with the best calibrated score built from confidence, uncertainty, blank ratio, and reading-direction priors. If enough labeled examples exist, a later plan may replace this with the orientation MLP described in `docs/exec-plans/proposed/orientation-mlp.md`, but this plan should keep the calibration simple and inspectable.

Separate label-only diagnostics from label-free inference features. `oracle_cer`, `matched_oracle`, and prediction length ratio against the ground-truth text are useful for analyzing training-page calibration, but they are not legal inputs to validation-page orientation selection. The calibrated inference score must be computed only from label-free features available at inference time, such as OCR confidence, OCR uncertainty, blank ratio, raw prediction length, geometry class, candidate transform name, and reading-direction priors.

For validation pages and normal inference, do not use PAGE ground-truth text in orientation selection. The selector may run OCR on orientation candidates and may use the calibration statistics learned from fine-tuning pages, OCR confidence, OCR uncertainty, blank ratio, output length, geometry class, and configured reading-direction priors. It must not compute CER, edit distance to ground truth, or any feature that requires validation `TextEquiv/Unicode`. The evaluation code may later compare final predictions against validation ground truth, but that happens after orientation has already been selected.

When OCR confidence selection is available, run the local OCR model on candidates and choose the candidate with the best calibrated inference score. If full OCR scoring is too expensive for unit tests, abstract it behind a scorer interface and provide a deterministic test scorer. Candidate scores should include at least candidate name, selected boolean, predicted text when available, mean confidence when available, uncertainty when available, blank ratio or output length when available, supervised line CER only when the page role is training, and rejection reason.

Do not train a new orientation MLP in this plan. `docs/exec-plans/proposed/orientation-mlp.md` is separate research and should not be mixed into this first strategy implementation.

Write metadata files under the caller's output root:

    segmentation_metadata.json
    unwrap_metadata.json
    orientation_metadata.json

For every line, metadata should include:

    strategy_name
    line_id
    line_custom
    line_numeric_id
    baseline_point_count
    resampled_point_count
    is_horizontal
    is_vertical
    is_curved
    is_circular
    cut_point
    tangent_settings
    normal_settings
    assigned_heatmap_box_count
    coords_area
    polygon_repaired
    unwrap_mode
    selected_orientation
    orientation_candidates
    orientation_selection_mode
    orientation_oracle_cer
    orientation_calibration_features
    ground_truth_used_for_orientation

## Concrete Steps

Work from the repository root:

    cd c:\Users\intro\OneDrive\Documents\MEGA\CAI-FLAME\gnn-synthetic-layout-historical

Add or update:

    app/recognition/line_segmentation/local_tangent_band.py
    app/recognition/line_segmentation/orientation.py
    app/recognition/line_segmentation/unwrap.py
    app/recognition/line_segmentation/types.py
    app/recognition/line_segmentation/registry.py
    app/recognition/pagexml_line_dataset.py
    app/tests/test_line_segmentation_strategy_unit.py
    app/tests/test_local_tangent_band_unit.py
    app/tests/test_orientation_selection_unit.py
    app/tests/test_recognition_active_learning_unit.py
    app/tests/test_circular_recognition_finetuning_precommit_e2e.py

Create synthetic test images in temporary test directories, not checked-in binary fixtures unless the test genuinely needs them. Synthetic tests should draw dark strokes on light backgrounds and matching heatmap blobs.

Unit tests should cover:

- horizontal line: `local_tangent_band_v1` produces a valid polygon with similar bounds to `legacy_axis_bound_v1`.
- vertical line: the strategy produces a tall page-space polygon and unwrap output whose width is greater than height after rotation or baseline ribbon sampling.
- simple arc: the strategy produces a curved band polygon whose points are not only a single axis-aligned rectangle.
- circular line: a closed baseline is cut at the top-most point and metadata records `cut_policy="top_point"`.
- masking: pixels outside the unwrapped `Coords` mask are set to the page median background.
- orientation metadata: all considered candidates and the selected candidate are written.
- supervised orientation calibration: on labeled fine-tuning pages, the candidate with the lowest CER against the line ground truth is marked as the oracle orientation, and the metadata records `ground_truth_used_for_orientation=true`.
- validation orientation inference: on held-out pages, the orientation selector records candidate OCR confidence and uncertainty features, selects an orientation without any ground-truth CER field, and records `ground_truth_used_for_orientation=false`.

Then configure `local_tangent_band_v1` as the proposed strategy for ablation gates from plan 02.

## Validation and Acceptance

Run focused unit tests:

    $env:CONDA_NO_PLUGINS='true'; conda run -n gnn_layout python -m unittest app.tests.test_local_tangent_band_unit -v
    $env:CONDA_NO_PLUGINS='true'; conda run -n gnn_layout python -m unittest app.tests.test_line_segmentation_strategy_unit -v

Expected result: synthetic horizontal, vertical, curved, circular, masking, and metadata tests pass.

Run the orientation calibration unit tests:

    $env:CONDA_NO_PLUGINS='true'; conda run -n gnn_layout python -m unittest app.tests.test_orientation_selection_unit -v

Expected result: a deterministic fake OCR scorer makes the lowest-CER candidate win on a labeled training page, and the same selector refuses to read or use ground-truth text on a held-out validation page.

Run the OCR pre-commit unit tests:

    $env:CONDA_NO_PLUGINS='true'; conda run -n gnn_layout python -m unittest app.tests.test_recognition_finetuning_precommit_unit -v

Expected result: recipe and gate config tests still pass, now with `local_tangent_band_v1` available as a valid proposed strategy.

Run the pretrained full-pipeline gate:

    $env:CONDA_NO_PLUGINS='true'; conda run -n gnn_layout python -m unittest discover -s app/tests -p "test_ci_e2e.py" -v

Expected result: proposed performance on `eval_dataset` is within the configured small regression tolerance against `legacy_axis_bound_v1`.

Run the OCR fine-tuning gate:

    $env:CONDA_NO_PLUGINS='true'; conda run -n gnn_layout python -m unittest app.tests.test_recognition_finetuning_precommit_e2e -v

Expected result: proposed OCR fine-tuning performance on `eval_dataset` is within the configured small regression tolerance.

Run the circular OCR gate:

    $env:CONDA_NO_PLUGINS='true'; conda run -n gnn_layout python -m unittest app.tests.test_circular_recognition_finetuning_precommit_e2e -v

Expected result: proposed `local_tangent_band_v1` is strictly better than benchmark `legacy_axis_bound_v1` on the circular gate primary metric. The latest artifact should show fine-tune pages `page_2`, `page_3`, `page_4` and evaluation pages `page_5`, `page_6`. Orientation metadata for fine-tune pages may contain supervised oracle CER values. Orientation metadata for evaluation pages must not contain validation ground-truth CER or any marker that ground truth was used for orientation selection.

Run the complete launcher:

    $env:CONDA_NO_PLUGINS='true'; conda run -n gnn_layout python scripts/run_precommit_eval.py

Expected result: all three ablation gates pass and write latest artifacts under `app/tests/logs/`.

For long OCR runs on Windows, use:

    C:\Users\intro\miniconda3\envs\gnn_layout\python.exe -m unittest app.tests.test_circular_recognition_finetuning_precommit_e2e -v
    C:\Users\intro\miniconda3\envs\gnn_layout\python.exe scripts/run_precommit_eval.py

Acceptance must include inspecting representative metadata. Confirm the unwrapped OCR rectangles are not written into PAGE `Coords`; PAGE `Coords` remain page-space polygons.

## Idempotence and Recovery

The strategy should be deterministic. Re-running on the same inputs and output directory may overwrite copied PAGE-XML and metadata files for that run, but it should not mutate source PAGE-XML under `app/tests/eval_dataset*`.

If `local_tangent_band_v1` regresses horizontal `eval_dataset`, do not loosen gates immediately. First inspect `segmentation_metadata.json`, compare assigned heatmap counts and polygon areas against `legacy_axis_bound_v1`, and tune config defaults. Record any threshold change in `Decision Log` with metrics evidence.

If orientation selection is unstable because OCR confidence is noisy, keep all candidates in metadata and use deterministic tie-breaking: prefer the supervised lowest-CER oracle only on labeled training pages; otherwise prefer the candidate implied by calibrated training-page statistics, then configured reading direction, then shortest rotation, then candidate name order.

If a bug causes validation PAGE `TextEquiv/Unicode` to be read during orientation inference, treat it as a test-blocking data leak. The fix is to pass page role explicitly into the orientation selector and to make the scorer API reject ground-truth text unless `page_role="train"` and `allow_ground_truth_orientation_labels=true`.

If a circular line has no reliable closure, process it as a curved line and record `is_circular=false` with the closure distance. Do not force a circular cut on open baselines.

## Artifacts and Notes

The most important outputs for debugging are:

    segmentation_metadata.json
    unwrap_metadata.json
    orientation_metadata.json
    manifest.json
    app/tests/logs/circular_ocr_ablation_latest.json
    app/tests/logs/circular_ocr_ablation_latest.md

Generated crop and metadata artifacts are evidence only. Durable conclusions about the strategy should be copied into checked-in docs by plan 04 if the strategy is promoted.

## Interfaces and Dependencies

Required strategy name:

    local_tangent_band_v1

Required config defaults:

    baseline_sample_spacing_px=6
    normal_padding_scale=0.7
    tangent_padding_scale=0.5
    min_band_half_height_px=12
    circular_cut_policy=top_point
    circular_reading_direction=clockwise
    script_direction=left_to_right
    unwrap_mode=baseline_ribbon_v1
    orientation_selection=calibrated_ocr_uncertainty
    allow_ground_truth_orientation_labels=true

Required unwrapping function:

    prepare_unwrapped_page_line_dataset(
        pagexml_path,
        page_image_path,
        output_root,
        unwrap_config=None,
    ) -> PreparedPageDataset

Required orientation function:

    select_orientation_candidate(
        candidates,
        page_role,
        scorer,
        ground_truth_text=None,
        calibration=None,
        config=None,
    ) -> OrientationSelectionResult

This function must reject `ground_truth_text` unless `page_role="train"` and `allow_ground_truth_orientation_labels=true`. Its result must expose the selected candidate, all candidate scores, whether ground truth was used, and the label-free features used for inference scoring.

Use existing dependencies: `cv2`, `numpy`, `shapely`, `skimage.io`, and the existing OCR inference helpers in `app/recognition/active_learning.py` when candidate scoring needs model predictions. Do not add a new external OCR or geometry library.

## Change Note

Initial split plan created on 2026-05-09. This plan isolates the first proposed geometry and unwrapping algorithm from the gate framework so algorithm failures can be debugged without changing evaluation plumbing.

Updated on 2026-05-09 to add supervised orientation calibration on labeled fine-tuning pages and an explicit no-ground-truth rule for validation-page orientation inference. This preserves the original orientation candidate plan while making use of the three labeled circular fine-tuning pages safely.
