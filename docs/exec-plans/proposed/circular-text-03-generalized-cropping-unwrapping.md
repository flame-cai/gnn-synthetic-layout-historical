# Implement Local Tangent Band Segmentation And OCR Unwrapping

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
- [ ] Add synthetic unit tests for horizontal, vertical, curved, and circular baselines.
- [ ] Run all ablation gates and record whether the proposed strategy meets the plan 02 comparison rules.

## Surprises & Discoveries

- Observation: the current legacy algorithm contains a vertical branch, but its core polygon construction still depends on axis-aligned boxes and rectangle bridging.
  Evidence: `app/segment_from_point_clusters.py` and `src/gnn_inference/segment_from_point_clusters.py` contain `detect_line_type(...)`, `analyze_and_clean_blob(...)`, and `get_bboxes_for_lines(...)`, then draw axis-aligned rectangles into a mask before extracting a contour.

- Observation: the current OCR crop preparation masks a PAGE-space polygon into an axis-aligned bounding rectangle and fills background with the page median color.
  Evidence: `app/recognition/pagexml_line_dataset.py::_masked_line_crop(...)` computes `cv2.boundingRect(polygon)`, fills a new image with `np.median(processing_image)`, and copies only pixels inside the shifted polygon mask.

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

When OCR confidence selection is available, run the local OCR model on candidates and choose the candidate with the best score. If full OCR scoring is too expensive for unit tests, abstract it behind a scorer interface and provide a deterministic test scorer. Candidate scores should include at least candidate name, selected boolean, predicted text when available, mean confidence when available, blank ratio or output length when available, and rejection reason.

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

## Concrete Steps

Work from the repository root:

    cd c:\Users\intro\OneDrive\Documents\MEGA\CAI-FLAME\gnn-synthetic-layout-historical

Add or update:

    app/recognition/line_segmentation/local_tangent_band.py
    app/recognition/line_segmentation/unwrap.py
    app/recognition/line_segmentation/types.py
    app/recognition/line_segmentation/registry.py
    app/recognition/pagexml_line_dataset.py
    app/tests/test_line_segmentation_strategy_unit.py
    app/tests/test_local_tangent_band_unit.py
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

Then configure `local_tangent_band_v1` as the proposed strategy for ablation gates from plan 02.

## Validation and Acceptance

Run focused unit tests:

    $env:CONDA_NO_PLUGINS='true'; conda run -n gnn_layout python -m unittest app.tests.test_local_tangent_band_unit -v
    $env:CONDA_NO_PLUGINS='true'; conda run -n gnn_layout python -m unittest app.tests.test_line_segmentation_strategy_unit -v

Expected result: synthetic horizontal, vertical, curved, circular, masking, and metadata tests pass.

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

Expected result: proposed `local_tangent_band_v1` is strictly better than benchmark `legacy_axis_bound_v1` on the circular gate primary metric. The latest artifact should show fine-tune pages `page_2`, `page_3`, `page_4` and evaluation pages `page_5`, `page_6`.

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

If orientation selection is unstable because OCR confidence is noisy, keep all candidates in metadata and use deterministic tie-breaking: prefer the candidate implied by configured reading direction, then shortest rotation, then candidate name order.

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
    orientation_selection=ocr_confidence

Required unwrapping function:

    prepare_unwrapped_page_line_dataset(
        pagexml_path,
        page_image_path,
        output_root,
        unwrap_config=None,
    ) -> PreparedPageDataset

Use existing dependencies: `cv2`, `numpy`, `shapely`, `skimage.io`, and the existing OCR inference helpers in `app/recognition/active_learning.py` when candidate scoring needs model predictions. Do not add a new external OCR or geometry library.

## Change Note

Initial split plan created on 2026-05-09. This plan isolates the first proposed geometry and unwrapping algorithm from the gate framework so algorithm failures can be debugged without changing evaluation plumbing.
