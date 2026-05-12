# `local_tangent_band_v1` Architecture

This document is the blueprint for the current proposed text-line segmentation strategy, `local_tangent_band_v1`. It describes the implemented pipeline that converts PAGE-XML `Baseline` polylines, page images, and heatmaps into PAGE-space `Coords` polygons and OCR-ready line crops.

The important design rule is that PAGE-XML `Coords` remain manuscript page geometry. OCR unwrapping is a separate downstream representation and must never be written back as PAGE `Coords`.

## Purpose

`local_tangent_band_v1` generalizes the legacy horizontal cropper to vertical, curved, and circular text. Its core invariant is that every curved text line is locally straight, so padding is measured along each baseline's local tangent and normal rather than along global page x/y axes.

The strategy is registered under:

    app/recognition/line_segmentation/registry.py

and implemented by:

    app/recognition/line_segmentation/local_tangent_band.py
    app/recognition/line_segmentation/geometry.py
    app/recognition/line_segmentation/unwrap.py

## Pipeline Overview

The pipeline has two explicit phases.

Phase 1, page-space line segmentation, reads the page image, heatmap, and source PAGE-XML. It writes a copied PAGE-XML with updated page-space `TextLine/Coords` polygons and writes strategy metadata.

Phase 2, OCR crop preparation, reads the copied PAGE-XML, page image, `Coords`, and `Baseline`. It writes OCR-ready horizontal crop images and a `PreparedPageDataset` manifest. It does not modify PAGE-XML geometry.

## Inputs

The strategy application entry point is:

    apply_text_line_segmentation_strategy(
        page_image_path,
        heatmap_path,
        source_pagexml_path,
        output_pagexml_path,
        strategy_name="local_tangent_band_v1",
        strategy_config={...},
        metadata_path=...
    )

Required inputs are:

- page image, usually the resized manuscript page
- heatmap, usually generated from CRAFT character detection
- source PAGE-XML containing `TextLine/Baseline` and optional `TextEquiv/Unicode`
- strategy config

The strategy does not use PAGE `Coords` as a fallback when generating proposed polygons.

## Step-By-Step Blueprint

### 1. Load PAGE Baseline Records

`load_baseline_records(...)` parses each PAGE `TextLine` and extracts:

- `line_id`
- `line_custom`
- numeric line id from `structure_line_id_*`
- `Baseline` points
- normalized text from `TextEquiv/Unicode`

For OCR training, empty-text lines are skipped by default. The app save path can still pass `include_empty_text_lines=True`.

### 2. Normalize Baseline Topology

Each line's raw `Baseline` is normalized by `normalize_baseline_topology(...)` in `geometry.py`.

The first normalization handles PAGE out-and-back paths. Some baselines are stored as:

    p0, p1, ..., pt, ..., p1

The code detects a split point `t` where points after `t` mirror points before `t` in reverse order. It then keeps only:

    p0, p1, ..., pt

The second normalization classifies whether the forward path is closed. A path is closed when its first and last points are within `closed_path_tolerance_px`.

Closed paths are cut at the top-most point, because the circular annotation convention starts there. The configured circular direction is `clockwise`.

Open paths are optionally reversed to respect the configured `left_to_right` reading order when the line is dominantly horizontal.

The metadata records:

- original and normalized point counts
- out-and-back split index
- mirror tolerance and mirror distances
- closed/open classification
- top cut index
- baseline length
- line kind
- orientation action

### 3. Classify Line Kind

The normalized baseline is classified as one of:

- `horizontal_straight`
- `vertical_straight`
- `curved_open`
- `closed_circular`
- `point`

Straightness is based on chord length divided by polyline length. Horizontal and vertical classification also use the configured angle tolerance.

### 4. Preserve Legacy Horizontal Behavior

For `horizontal_straight` and `point` lines, `local_tangent_band_v1` preserves the old benchmark behavior by delegating PAGE-space polygon generation to `legacy_axis_bound_v1`.

This is controlled by:

    preserve_horizontal_with_legacy=True

This keeps standard horizontal manuscript behavior stable while allowing local tangent geometry for vertical, curved, and circular lines.

### 5. Extract Heatmap Components

For non-preserved lines, the strategy loads the heatmap, resizes it to page-image dimensions, thresholds it with `BINARIZE_THRESHOLD`, and extracts connected-component bounding boxes using the existing canonical legacy helper.

Each heatmap box stores:

- x/y position
- width and height
- center point
- max side length

### 6. Assign Components To Baselines

Every heatmap component is assigned to the nearest normalized baseline. Distance is computed to the nearest point on the polyline, not to an axis-aligned rectangle.

The assignment guard rejects a component when its distance exceeds:

    max(component_max_distance_px, component.max_side * component_distance_scale)

For accepted components, the strategy records both:

- distance from component center to baseline
- component half-extent along the local normal

The local-normal half-extent is important for vertical text. A vertical line's component height is mostly along the line, so using the longest side as line thickness would make the crop too wide.

### 7. Estimate Local Band Width

For each local line, the strategy estimates a half-width across the baseline from assigned components:

    baseline_distance + normal_half_extent

It uses the 90th percentile for robustness, then applies:

    half_width = percentile90 * normal_pad_scale + normal_pad_px

The result is clamped between `minimum_half_width_px` and `maximum_half_width_px`.

### 8. Build PAGE-Space Local Band Polygon

The normalized baseline is converted into a polygon by sampling each baseline point's local tangent and normal.

For an open path:

- extend the first point backward along the tangent by `along_pad`
- extend the last point forward along the tangent by `along_pad`
- offset every point by `+normal * half_width`
- offset every point by `-normal * half_width`
- combine the two offset chains into a closed polygon

For a closed circular path:

- do not add endpoint extension
- use the top-cut closed path order
- offset around the local normal and close the polygon

All polygon points are clipped to page-image bounds. These polygons are the PAGE `Coords`.

### 9. Write Copied PAGE-XML And Metadata

The strategy removes any existing `TextLine/Coords` from the copied PAGE-XML and inserts the new page-space polygon before the `Baseline`.

The result metadata includes:

- strategy name
- source/output paths
- line counts
- line metadata
- geometry summary

The geometry summary includes guard values such as:

- `source_line_coverage`
- `heatmap_box_count`
- `assigned_box_count`
- `heatmap_box_assignment_rate`
- topology counts
- orientation policy

## OCR Unwrapping Blueprint

OCR crop preparation happens in:

    app/recognition/pagexml_line_dataset.py

For `local_tangent_band_v1`, OCR unwrapping is applied only when a line's strategy metadata says:

    crop_model == "local_tangent_band"

Lines delegated to legacy geometry use the old axis-aligned masked crop.

### 1. Read PAGE Geometry

`load_pagexml_lines(...)` reads each prepared PAGE `TextLine` and keeps:

- page-space `Coords`
- normalized `Baseline`
- ground-truth text
- region and line ids

### 2. Re-Normalize The Baseline

`unwrap_line_crop_for_ocr(...)` normalizes the baseline again using the same topology rules. This keeps unwrapping independent from PAGE writing and makes the OCR step reproducible from the copied PAGE-XML.

### 3. Estimate Output Rectangle Size

The output width is approximately the normalized baseline length:

    output_width = ceil(baseline_length / sample_spacing_px)

The output height is twice the estimated half-width of the PAGE polygon around the normalized baseline.

This provides the sanity check required by the plan: OCR crop length should track baseline length after out-and-back splitting.

### 4. Build Remap Coordinates

The normalized baseline is sampled at one-pixel spacing. For each output column:

- find the baseline center point
- find the local tangent
- compute the local normal
- map output rows across the normal from `-half_width` to `+half_width`

OpenCV `cv2.remap(...)` samples the page image into this horizontal rectangle.

### 5. Mask Foreground And Fill Background

The PAGE-space `Coords` polygon is rasterized into a page mask. The same remap grid unwraps that mask into OCR space.

Pixels outside the unwrapped mask are filled with the median page color. This is the strict foreground/background requirement: the OCR crop contains the transformed text-line region, and outside-mask background is page-median color.

### 6. Record Orientation Metadata

The current implementation records deterministic orientation metadata:

- candidate transforms: `identity`, `rotate_180`
- selected transform: `identity`
- selection mode: supervised text-equivalent baseline order when text exists, otherwise geometry-only baseline order
- configured reading order and circular direction
- topology orientation action

The current v1 does not run OCR-confidence orientation search. That remains a future strategy iteration.

## Outputs

Strategy outputs:

- copied PAGE-XML with page-space `Coords`
- metadata JSON
- `TextLineSegmentationResult`

OCR preparation outputs:

- app-style line images under `image-format/`
- OCR fine-tuning images under `finetune_dataset/test/`
- `gt.txt`
- `manifest.json`
- `PreparedPageDataset`

Each prepared record can carry `crop_metadata`, including unwrap and orientation details.

## Config Defaults

Important `local_tangent_band_v1` defaults are:

    BINARIZE_THRESHOLD = 0.5098
    mirror_match_tolerance_px = 4.0
    closed_path_tolerance_px = 12.0
    min_mirror_pairs = 3
    straightness_chord_ratio = 0.985
    horizontal_angle_degrees = 12.0
    component_max_distance_px = 90.0
    component_distance_scale = 4.0
    minimum_half_width_px = 18.0
    maximum_half_width_px = 180.0
    normal_pad_px = 8.0
    normal_pad_scale = 1.15
    along_pad_scale = 0.5
    preserve_horizontal_with_legacy = True
    reading_order = "left_to_right"
    circular_direction = "clockwise"

## Current Gate Behavior

`local_tangent_band_v1` is the default proposed strategy through:

    PRECOMMIT_PROPOSED_LINE_STRATEGY

when the environment variable is not set.

The three ablation gates compare:

- benchmark: `legacy_axis_bound_v1`
- proposed: `local_tangent_band_v1`

Current acceptance behavior:

- pipeline gate allows `0.01` absolute regression on CER-like metrics
- regular OCR fine-tuning gate allows `0.02` absolute regression
- circular OCR fine-tuning gate requires strict primary curve-metric improvement

Circular secondary metrics are reported but are not blocking when strict primary improvement is enabled.

## Known Limits

The v1 strategy is intentionally conservative.

It does not yet perform OCR-confidence orientation selection across rotated/flipped candidate crops. It records deterministic metadata instead.

It does not promote the strategy to benchmark. Promotion belongs to the next plan.

It does not remove or replace `legacy_axis_bound_v1`. The legacy strategy remains the benchmark and also serves as the horizontal special case for the proposed strategy.

## Validation Commands

Use the `gnn_layout` conda environment from the repository root.

    $env:CONDA_NO_PLUGINS='true'; conda run -n gnn_layout python -m unittest app.tests.test_line_segmentation_strategy_unit -v
    $env:CONDA_NO_PLUGINS='true'; conda run -n gnn_layout python -m unittest app.tests.test_strategy_ablation_config_unit -v
    $env:CONDA_NO_PLUGINS='true'; conda run -n gnn_layout python -m unittest app.tests.test_recognition_finetuning_precommit_unit -v
    $env:CONDA_NO_PLUGINS='true'; conda run -n gnn_layout python -m unittest app.tests.test_circular_recognition_finetuning_precommit_e2e -v
    $env:CONDA_NO_PLUGINS='true'; conda run -n gnn_layout python -m unittest app.tests.test_recognition_finetuning_precommit_e2e -v
    $env:CONDA_NO_PLUGINS='true'; conda run -n gnn_layout python -m unittest discover -s app/tests -p "test_ci_e2e.py" -v
