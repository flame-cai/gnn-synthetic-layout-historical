# local_polygons_v1 Strategy Sketch

This document sketches a proposed text-line segmentation and OCR crop strategy named
`local_polygons_v1`. It is not an implementation plan yet. The goal is to record the
geometry idea, the risks found during investigation, and the TODOs that should be
completed before implementation starts.

## Problem

`legacy_axis_bound_v1` produces useful OCR crops for mostly horizontal text because
`segmentLinesFromPointClusters(...)` applies several practical heuristics before it
builds PAGE `TextLine/Coords`:

- heatmap thresholding and connected-component boxes
- dynamic padding around each component
- local connected-component cleanup inside each padded crop
- removal of small components touching the crop's top or bottom boundary
- rectangular mask union and bridging between nearby disconnected box groups
- final contour extraction and median-color fill outside the generated polygon

Those heuristics are valuable because they reject unnecessary ink from adjacent
lines above and below the target line.

`local_tangent_band_v1` solves a different problem. It builds a broad local tangent
band around a baseline, then unwraps that band. This handles curved and circular
layout, but the band itself is often too permissive. In
`app/tests/logs/20260512_135724_ocrft_circular_ocr_ablation_proposed_eval_dataset_v2`,
the local-tangent metadata shows that the PAGE/ribbon mask covers nearly the whole
unwrapped rectangle. For example, `page_2` line `0` has output size
`6665 x 156` and `foreground_pixel_count=1035333`, so only about `0.42%` of the
rectangle is outside the mask and replaced with page median color. The crop therefore
keeps page background texture across almost the entire ribbon.

The desired next strategy is not "band then unwrap". It is:

```text
legacy polygon heuristics in a local coordinate system,
then unwrap from those tighter polygons
```

## Core Idea

For every text line, construct a local frame around the normalized PAGE baseline:

```text
s = distance along baseline
n = signed distance along the local normal
```

In this `(s, n)` frame, each line is treated as locally horizontal.

The legacy idea of top and bottom generalizes to:

```text
top    = negative normal side
bottom = positive normal side
```

For a normal horizontal line, `s` is approximately page `x` and `n` is approximately
`page_y - baseline_y`. That means the new strategy can match the spirit of
`segmentLinesFromPointClusters(...)` for horizontal lines without delegating to
`legacy_axis_bound_v1`.

## Proposed Pipeline

### 1. Normalize Baselines

Normalize every raw PAGE `Baseline` before any local geometry is computed.

Required behavior:

- remove graph-walk backtracking artifacts such as `A -> B -> C -> B`
- handle longer out-and-back paths
- preserve true closed circular paths
- choose a stable seam for closed paths
- enforce reading direction and circular direction
- expose the normalized baseline and topology metadata to both geometry generation
  and OCR unwrapping

This should happen before `local_polygons_v1` is implemented, because baseline
normalization affects `local_tangent_band_v1`, the proposed strategy, and every
future strategy that uses baseline-local coordinates.

### 2. Extract Heatmap Components

Reuse the legacy heatmap thresholding primitive:

```text
heatmap -> threshold -> external contours -> component bounding boxes
```

The implementation should preserve the meaning of the existing defaults:

- `BINARIZE_THRESHOLD`
- `BBOX_PAD_V`
- `BBOX_PAD_H`
- `CC_SIZE_THRESHOLD_RATIO`

The strategy may later add local-only settings, but the first version should keep
the old knobs recognizable so ablations remain interpretable.

### 3. Assign Components To Lines

Assign heatmap components to normalized baselines by nearest baseline point, not by
axis-aligned containment alone.

For each component, record:

- nearest line id
- nearest baseline station `s`
- signed normal offset `n`
- local tangent and normal
- component extent along `s`
- component extent along `n`
- distance-to-baseline and rejection reason when unassigned

This is where local "above" and "below" become meaningful for vertical, curved, and
circular text.

### 4. Project Components Into Local Space

For each assigned component, project either its rectangle corners or its heatmap
mask pixels into the line-local `(s, n)` frame.

The recommended first implementation is conservative:

- start with component rectangles projected into local space
- retain enough per-component metadata to debug false inclusions
- add mask-pixel projection only if rectangle projection proves too coarse

Projected components should be represented as local rectangles or small local masks
on a local canvas.

### 5. Run Legacy-Style Cleanup In Local Space

Apply the old component cleanup in local coordinates:

- pad along `n` using the legacy vertical padding idea
- pad along `s` using the legacy horizontal padding idea
- crop the original page through the local remap for the padded local box
- binarize the local crop
- remove small connected components touching the local top or bottom boundary
- for near-vertical text, do not special-case page `x`; the local normal axis already
  defines top and bottom

This is the main difference from `local_tangent_band_v1`: local polygons are produced
from cleaned component evidence, not from a broad baseline ribbon.

### 6. Build A Tight Local Polygon

Union the cleaned local boxes or masks for one line.

Then:

- bridge nearby disconnected groups in increasing `s` order
- use local line height as the bridge height, matching the spirit of legacy bridging
- extract a contour in local `(s, n)` space
- simplify the contour while preserving the cleaned mask area

The output at this stage is a local polygon, not yet PAGE `Coords`.

### 7. Map The Local Polygon Back To PAGE Space

Convert local polygon boundary points back to page coordinates:

```text
page_point = baseline_point_at_s + n * local_normal_at_s
```

The resulting PAGE `Coords` are not globally axis-aligned for curved text. They are
axis-aligned in the local line frame, which is the intended generalization of the
legacy axis-bound polygon.

### 8. Prepare OCR Crop By Unwrapping The Tight Polygon

Use the baseline-local remap that `local_tangent_band_v1` already uses, but mask with
the tight local polygon rather than the broad line band.

Expected OCR crop behavior:

```text
inside cleaned local polygon  -> original page pixels
outside cleaned local polygon -> page median color
```

This should make curved and circular line crops visually closer to legacy crops:
text-line content is preserved, while unrelated adjacent-line/background regions are
suppressed.

## Why Horizontal Lines Should Not Delegate To Legacy

The proposed strategy should be one general algorithm. For horizontal lines, the
local coordinate frame collapses to the normal page frame:

```text
s ~= page x
n ~= page y - baseline_y
```

Therefore, if the strategy preserves thresholding, padding, top/bottom cleanup,
bridging, and contour extraction semantics, horizontal behavior can be made
equivalent or near-equivalent to legacy without a special delegation branch.

This matters because delegation hides defects. A general strategy must prove that
the local-coordinate formulation reproduces the horizontal benchmark before it is
trusted for curved and circular text.

## Pre-Implementation TODOs

### TODO 1: Harden Baseline Normalization First

Do not start `local_polygons_v1` by patching OCR crops. Fix the baseline topology
layer first.

The immediate failure case is:

```text
756,816 -> 806,814 -> 856,818 -> 806,814
```

This is a short retraced graph walk: `A -> B -> C -> B`.

Current `normalize_baseline_topology(...)` detects out-and-back paths only when
there are at least `min_mirror_pairs=3`. For `A-B-C-B`, there is only one mirror
pair, so current normalization leaves it as `curved_open`. `local_tangent_band_v1`
then builds and unwraps a folded band.

Required normalization work:

- detect short retraced tails such as `A-B-C-B` and `A-B-C-D-C-B`
- avoid turning branched walks such as `A-B-C-A-C-B` into fake closed loops
- when repeated near-points form an otherwise straight horizontal or vertical line,
  collapse to unique points ordered along the dominant axis
- keep true circular paths classified as `closed_circular`
- expose enough metadata to distinguish `short_tail_trimmed`,
  `dominant_axis_deduped`, and existing `out_and_back` normalization

### TODO 2: Scan Baselines Before Changing Behavior

Initial scan performed on 2026-05-16 over:

- `app/tests/eval_dataset`
- `app/tests/eval_dataset_v2`
- `app/input_manuscripts`

Scan results:

| Root | XML files | Text lines | Lines with baseline | Current out-and-back | Current closed | Short retraced-tail candidates | Non-adjacent repeats | Straight after dedupe |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `app/tests/eval_dataset` | 15 | 305 | 279 | 218 | 0 | 31 | 249 | 31 |
| `app/tests/eval_dataset_v2` | 5 | 25 | 25 | 25 | 10 | 0 | 25 | 0 |
| `app/input_manuscripts` | 495 | 14342 | 14245 | 640 | 0 | 59 | 699 | 58 |

Interpretation:

- `eval_dataset_v2` mostly uses long out-and-back paths already handled by current
  normalization.
- `eval_dataset` contains many short retraced-tail baselines that current
  normalization misses.
- local manuscripts also contain short retraced-tail candidates, so this is not only
  a test-artifact problem.

Before implementation, turn this scan into a repeatable test or diagnostic script
so future strategy changes can report how many baselines are normalized by each rule.

### TODO 3: Decide Whether To Fix Baseline Generation Too

The source of many retraced baselines is `trace_component_with_backtracking(...)` in
`app/gnn_inference.py`. That DFS trace appends parent nodes while it backtracks,
which is appropriate for visiting all graph edges but not always appropriate for a
PAGE text-line baseline.

Implementation choices:

- harden only `normalize_baseline_topology(...)`
- replace the graph baseline trace with a cleaner path extraction method
- do both, but keep normalization as a defensive layer for old PAGE XML

The safest order is:

1. harden normalization
2. add tests proving old PAGE XML is repaired at read time
3. consider changing baseline generation after the strategy work is stable

### TODO 4: Define Local Polygon Metadata

`local_polygons_v1` should emit enough metadata for debugging and crop selection:

- `crop_model = "local_polygon_unwrap"`
- `line_kind`
- `topology`
- component assignment counts
- rejected component counts by reason
- local canvas dimensions
- local polygon point count before and after simplification
- median-background fraction in the OCR crop
- whether the crop used rectangle projection or heatmap-mask projection
- seam location for closed lines

The shared OCR cropper must choose the new unwrapping behavior from per-line
metadata, not from page-level strategy name alone.

### TODO 5: Protect Horizontal Parity

Add tests before curved/circular optimization:

- synthetic horizontal line: local polygon crop should match legacy masked crop
  within a tight pixel tolerance
- real `eval_dataset` horizontal pages: line counts and page-level CER should stay
  close to `legacy_axis_bound_v1`
- no `legacy_axis_bound_delegate` crop model should appear for this strategy
- `baseline_heatmap` strategy preparation should report stable
  `source_line_coverage` and `heatmap_box_assignment_rate`

## Expected Trickle-Down Effects

### Strategy Registry And Role Config

Adding `local_polygons_v1` means updating the strategy registry, tests, and role
config. It should enter as a proposed research strategy, not as production behavior.
Research promotion and production adoption must remain separate.

### OCR Crop Layer

The shared crop layer currently distinguishes:

- legacy masked crop
- local tangent unwrap
- local tangent legacy delegate fallback

`local_polygons_v1` needs a new crop model. It should not be forced through
`local_tangent_band` metadata because that name means broad-band geometry today.

### PAGE XML Size And Compatibility

Local masks mapped back to page space may produce many polygon vertices. Add contour
simplification and point-count metadata. Keep PAGE `Coords` in page coordinates and
never write unwrapped rectangles back to PAGE XML.

### Circular Seam Handling

Closed circular lines need a seam. A poor seam can split connected text components
across `s=0`. The strategy should cut closed paths at a stable top point, then either
duplicate a small seam overlap in local space or bridge across the seam before final
contour extraction.

### Component Competition Between Lines

Nearest-baseline assignment can still assign adjacent-line ink to the wrong line
when baselines are close. Consider adding competition rules:

- reject components whose nearest and second-nearest baselines are too close
- reject components with normal offset beyond a robust line-width estimate
- log ambiguous components separately from unassigned components

### Background Semantics

Legacy crops preserve original page pixels inside the generated polygon and set only
outside-polygon pixels to median color. `local_polygons_v1` should initially preserve
that semantic. A stricter ink-only mask would be a different strategy variant.

### Performance And Memory

Circular unwrapped crops can be very wide. The `eval_dataset_v2` run contains
unwrapped widths around 6600 pixels. Local mask projection and contour extraction
must avoid per-pixel work on full page images where possible. Start with component
rectangles and local canvases bounded to each line.

### Existing Artifacts

Old generated PAGE XML, line images, and active-learning revisions will not change
until regenerated. Any comparison must regenerate prepared pages under the new
strategy before drawing conclusions.

## Proposed Acceptance Checks

Minimum checks before treating `local_polygons_v1` as a viable proposed strategy:

- baseline normalization unit tests for `A-B-C-B`, `A-B-C-D-C-B`, branched
  repeated walks, and true closed circular paths
- synthetic horizontal parity test against `legacy_axis_bound_v1`
- synthetic vertical and curved tests showing local top/bottom cleanup rejects
  adjacent-line components
- `eval_dataset` full-pipeline ablation
- `eval_dataset` OCR fine-tune ablation with strict attention to horizontal regressions
- `eval_dataset_v2` circular OCR ablation
- manifest-level crop diagnostics showing a meaningful median-background fraction
  for circular crops compared with `local_tangent_band_v1`

## Implementation Status

Status as of 2026-05-19:

- Baseline normalization has been hardened first in the shared topology layer, so
  both the benchmark strategy and proposed strategies read the same normalized
  baselines before computing local geometry.
- `local_polygons_v1` is registered as the proposed research strategy. Production
  remains pinned to `legacy_axis_bound_v1`.
- PAGE-XML preparation is implemented as a separate strategy step in
  `app/recognition/line_segmentation/local_polygons.py`. It projects heatmap
  component rectangles into baseline-local `(s, n)` coordinates, remaps the
  original page image into each local component crop, applies the legacy-style
  Otsu connected-component cleanup that trims small components touching local
  top/bottom boundaries, builds a local mask/polygon from the cleaned component
  crops, maps that polygon back to PAGE-space `Coords`, and writes per-line
  metadata with `crop_model = "local_polygon_unwrap"` and
  `local_cleanup_model = "legacy_remap_top_bottom_cc"`.
- OCR crop preparation remains a second step. `prepare_page_line_dataset(...)`
  reloads the generated PAGE XML and the strategy metadata, then the shared cropper
  unwraps `Coords + Baseline` only when the line metadata requests
  `local_polygon_unwrap`.
- Small open-baseline lines now preserve foreground that extends before or after the
  raw baseline endpoints. PAGE-XML generation extrapolates the baseline-local
  station outside the endpoint range when assigning component geometry, records
  `local_s_min`/`local_s_max`, and OCR crop preparation passes those limits into
  the unwrap step so the final text-line image is not clipped back to the original
  baseline length.
- One-point baselines are treated as degenerate local horizontal frames rather than
  as unwrappable failures. The strategy projects heatmap component extents into
  `s = page_x - point_x`, `n = page_y - point_y`, maps nonzero-width PAGE
  `Coords`, and unwraps with the recorded local `s`/`n` bounds instead of writing a
  1x1 fallback crop.
- The default research knobs for this strategy live in
  `app/recognition/line_segmentation/local_polygons.py` as
  `DEFAULT_LOCAL_POLYGON_CONFIG`. Harness runs can override them through
  per-call `strategy_config` / `line_segmentation_args` without changing
  production adoption state.
- The three slow final pre-commit success checks have not been run yet. Manual OCR
  crop review on `eval_dataset` and `eval_dataset_v2` is the next checkpoint.

## Non-Goals For The First Version

- production adoption
- migration of existing PAGE XML or active-learning checkpoint lineage
- OCR-confidence-based orientation search
- ink-only foreground masking
- replacing the legacy strategy

`legacy_axis_bound_v1` should remain available for benchmark comparison and
production rollback regardless of whether `local_polygons_v1` succeeds.
