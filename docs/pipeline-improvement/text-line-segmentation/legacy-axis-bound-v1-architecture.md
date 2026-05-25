# legacy_axis_bound_v1 Architecture

This document describes the current `legacy_axis_bound_v1` text-line segmentation strategy in enough detail to reimplement it without reading the source code.

It also places the strategy in the context of the behavior-preserving production refactor implemented through `docs/exec-plans/proposed/production-strategy-aware-ocr-crops.md`. The important point for that refactor is that a text-line segmentation strategy is not only "how PAGE-XML `Coords` are generated". For OCR, the strategy also includes "how PAGE-XML geometry is converted into OCR line images".

## Current Role

`legacy_axis_bound_v1` is the historical axis-aligned strategy for the text-line segmentation harness. It is no longer the production default after the 2026-05-23 `local_polygons_v1` adoption, but it remains registered for rollback, historical comparison, and masked-crop fallback behavior.

The strategy generates PAGE `TextLine/Coords` polygons from:

- PAGE `TextLine/Baseline` polylines
- the page image
- a CRAFT-style heatmap image
- fixed axis-aligned bounding-box and connected-component heuristics

It does not use existing PAGE `TextLine/Coords` as an input to generate new `Coords`. Existing line `Coords` are removed and replaced only when the strategy can generate a polygon for the corresponding text line.

The strategy is "legacy" because most of its geometry comes from the older `src/gnn_inference/segment_from_point_clusters.py` helper. The strategy wrapper in `app/recognition/line_segmentation/legacy_axis_bound.py` adapts PAGE baselines and heatmap boxes into the old helper's expected point-cluster format.

## Production Refactor Context

The production refactor first preserved behavior for `legacy_axis_bound_v1` while making the production path modular enough for later strategies such as `local_tangent_band_v1` and `local_polygons_v1`.

The production behavior is explicitly two-phase:

- PAGE `Coords` generation can already start from PAGE `Baseline` geometry through the registered text-line strategy.
- Production OCR inference, app line-image export, and active-learning training consume saved PAGE `Coords` through `app/recognition/line_segmentation/ocr_crops.py`.

For `legacy_axis_bound_v1`, that crop layer keeps the final OCR crop as an axis-aligned masked crop of the generated line polygon. The generated polygon itself already came from axis-aligned heatmap boxes, padding, connected-component cleanup, rectangular masks, and contour extraction.

For a strategy with a different crop model, such as a local tangent band or local-polygon unwrap, changing only PAGE `Coords` generation would be incomplete. The PAGE polygon may change, and the shared crop layer has one place to honor per-line crop metadata. The legacy crop remains identical to the historical masked behavior when legacy or fallback metadata is selected.

## Production GUI Save Context

In the production GUI, user layout edits are applied to graph data first. For example, when a user adds or deletes a node, the browser posts the edited graph nodes, graph edges, text-line labels, text-region labels, and text content to the save endpoint.

For normal layout saves, production then calls the page generation path. Text-only saves are different: they update text content in the existing PAGE XML and do not regenerate layout geometry.

For a layout save, the production page generation path does this:

1. Writes an intermediate baseline PAGE XML under:

   ```text
   layout_analysis_output/_baseline_page_xml/<page_id>.xml
   ```

2. Builds that intermediate XML from the current edited graph.

3. Finds graph connected components from the saved graph edges.

4. Creates one PAGE `TextLine` per connected component.

5. Stores the graph-derived line label in the PAGE `TextLine` custom attribute:

   ```text
   custom="structure_line_id_<line_label>"
   ```

6. Traces the ordered component path through the graph topology.

7. Converts each ordered graph point into a PAGE baseline point.

The current baseline coordinate formula is:

```text
baseline_x = int(point_x * 2)
baseline_y = int((point_y + point_radius / 2) * 2)
```

The page dimensions in this intermediate XML are also doubled from the heatmap dimensions:

```text
page_width = heatmap_width * 2
page_height = heatmap_height * 2
```

The intermediate call intentionally passes no line polygon data, so this baseline PAGE XML should be treated as the editable layout skeleton. It may have region `Coords`, but its text-line `Coords` are not the final strategy output.

Production then applies the registered production text-line segmentation strategy to that intermediate baseline XML:

```text
source PAGE Baseline + image + heatmap
    -> production strategy
    -> final PAGE TextLine/Coords
```

The final PAGE XML is written under:

```text
layout_analysis_output/page-xml-format/<page_id>.xml
```

For the current production default, the strategy is `local_polygons_v1`, its runtime config keeps `BINARIZE_THRESHOLD=0.45`, and `include_empty_text_lines` is passed as `true`.

After the final PAGE XML is written, production regenerates app line images from that final PAGE XML. The crop path reads the generated `TextLine/Coords`, loads the sibling line-segmentation metadata when present, and calls the shared crop layer. For new `local_polygons_v1` saves, the selected crop is `local_polygon_unwrap`; for legacy or missing metadata, the selected crop remains the generic masked polygon crop.

This means a user layout change affects OCR geometry in this order:

```text
edited graph nodes/edges
    -> graph-derived PAGE Baseline
    -> strategy-derived PAGE Coords
    -> Coords-derived OCR crop
```

So if a user adds a node and that node changes the connected component path, the saved PAGE baseline changes. Because the production strategy starts from the current PAGE baseline, the generated `Coords` may also change. The OCR crop then changes because it is prepared from the new final `Coords` and, for `local_polygons_v1`, from the new strategy metadata.

## Source Files

The current implementation is spread across these files:

- `app/recognition/line_segmentation/legacy_axis_bound.py`
  - strategy wrapper
  - PAGE baseline loading orchestration
  - heatmap box assignment to baselines
  - scratch point-cluster file generation
  - PAGE `Coords` replacement
  - metadata emission

- `app/recognition/line_segmentation/pagexml.py`
  - PAGE namespace handling
  - baseline record loading
  - text-line `Coords` removal
  - text-line `Coords` insertion by numeric line id

- `src/gnn_inference/segment_from_point_clusters.py`
  - heatmap thresholding
  - bounding-box generation
  - old point-cluster line assignment
  - dynamic rectangular padding
  - connected-component cleanup
  - rectangle-to-polygon contour extraction
  - scratch line-image writing

- `app/recognition/pagexml_line_dataset.py`
  - OCR dataset loading from PAGE XML

- `app/recognition/line_segmentation/ocr_crops.py`
  - app-style masked line crop generation from final PAGE `Coords`
  - strategy metadata loading and crop-model selection
  - local-tangent unwrap delegation when metadata requests it

The scratch line images written by `segment_from_point_clusters.py` are not the canonical production OCR crops. In the app and research OCR harness, final OCR crops are prepared later from the resulting PAGE XML through `ocr_crops.py`.

## Strategy Name

The registered strategy name is:

```text
legacy_axis_bound_v1
```

The strategy class exposes this exact name. Promotion, adoption, metadata, and gate evidence should use the exact string.

## Inputs

The strategy requires these inputs:

- `xml_path`
  - PAGE XML containing `TextLine` elements and `Baseline` children.
  - Existing `TextLine/Coords` may be present but are not trusted as source geometry.

- `image_path`
  - Page image corresponding to the PAGE XML.
  - The page image coordinate system is the output coordinate system for generated `Coords`.

- `heatmap_path`
  - CRAFT-style heatmap image.
  - May have a different width and height from the page image.
  - If the heatmap has multiple channels, the first channel is used.

- `output_xml_path`
  - Destination PAGE XML path.

- `output_root`
  - Working/output root used for temporary legacy helper inputs and outputs.

- `metadata_path`, optional
  - Destination JSON metadata path.

- strategy config, optional
  - Partial config is allowed. Missing values are filled from the defaults below.

## Configuration

The default `legacy_axis_bound_v1` configuration is:

```text
BINARIZE_THRESHOLD = 0.5098
BBOX_PAD_V = 0.7
BBOX_PAD_H = 0.5
CC_SIZE_THRESHOLD_RATIO = 0.4
include_empty_text_lines = false
```

Config normalization rules:

- `BINARIZE_THRESHOLD` is converted to float.
- `BBOX_PAD_V` is converted to float.
- `BBOX_PAD_H` is converted to float.
- `CC_SIZE_THRESHOLD_RATIO` is converted to float.
- `include_empty_text_lines` is converted to bool.
- `INCLUDE_EMPTY_TEXT_LINES` is accepted as an alias when `include_empty_text_lines` is not present.
- If neither empty-line key is present, empty text lines are excluded.

The canonical legacy helper has its own older default threshold of `0.30`, but the strategy always passes the normalized strategy threshold when calling it.

## Outputs

The main output is PAGE XML with regenerated `TextLine/Coords` polygons.

The strategy:

- parses the input XML
- removes all existing `Coords` children under `TextLine` elements
- inserts one new `Coords` child for each text line that has a generated polygon
- places the new `Coords` before the line's `Baseline` element when a baseline exists
- preserves other PAGE content
- preserves existing `TextRegion/Coords`; it does not recompute region polygons
- writes XML with an XML declaration and UTF-8 encoding

The strategy result metadata includes:

- `strategy_name`
- `output_xml_path`
- `metadata_path`
- `line_metadata`
- `geometry_summary`

The `geometry_summary` includes:

- `geometry_source = "baseline_heatmap"`
- `line_segmentation_strategy_name = "legacy_axis_bound_v1"`
- `prepared_line_count`
- `source_text_line_count`
- `source_line_coverage`
- `heatmap_box_count`
- `assigned_box_count`
- `heatmap_box_assignment_rate`
- `baseline_line_count`
- `max_assignment_distance`
- `mean_assignment_distance`

`source_line_coverage` is:

```text
prepared_line_count / source_text_line_count
```

If `source_text_line_count` is zero, coverage is `null`.

`heatmap_box_assignment_rate` is:

```text
assigned_box_count / heatmap_box_count
```

If `heatmap_box_count` is zero, assignment rate is `null`.

## Coordinate Systems

There are two coordinate systems in the strategy:

- page image coordinates
- heatmap coordinates

PAGE baselines and final PAGE `Coords` are in page image coordinates.

Heatmap connected components are first detected after resizing the heatmap to page image size. Their centers and dimensions are then converted back to heatmap coordinates to create synthetic point-cluster files for the legacy helper. The legacy helper later scales those synthetic points back to page image coordinates when `upscale_heatmap` is true.

The round trip is intentional because the old helper expects:

- a heatmap file in its original heatmap coordinate system
- point-cluster node coordinates in that same heatmap coordinate system
- an image file in page image coordinates

The wrapper therefore creates synthetic old-format inputs rather than rewriting the old helper.

Coordinate conversion from page image to heatmap space uses:

```text
x_to_heatmap = heatmap_width / image_width
y_to_heatmap = heatmap_height / image_height
```

For a heatmap bounding box detected in page image coordinates:

```text
center_x_page = x + width / 2
center_y_page = y + height / 2
node_x_heatmap = center_x_page * x_to_heatmap
node_y_heatmap = center_y_page * y_to_heatmap
node_radius = max(width * x_to_heatmap, height * y_to_heatmap)
```

The generated synthetic node is:

```text
[node_x_heatmap, node_y_heatmap, node_radius]
```

The old helper reads these values as integers, so fractional values written by the wrapper are truncated by the helper during load.

## PAGE Baseline Loading

The strategy loads baseline records from the PAGE XML.

For each `TextLine` descendant:

1. Find its `Baseline` child.
2. Parse the `points` attribute into integer `(x, y)` pairs.
3. Determine the text-line id from the `id` attribute if present.
4. Determine a line custom value from the PAGE `custom` attribute if present.
5. Determine the numeric line id.
6. Determine the line text from `TextEquiv/Unicode`.
7. Optionally skip the line if it has empty text.

Empty text-line handling depends on `include_empty_text_lines`.

When `include_empty_text_lines` is false, lines with no OCR text are skipped by the strategy. This is the default for research and OCR-preparation use because empty ground-truth lines are not useful for recognition evaluation.

When `include_empty_text_lines` is true, the strategy also prepares geometry for empty text lines. Production save/regeneration paths can use this mode because layout geometry exists independently of OCR transcription text.

The numeric line id is the stable key used to map generated polygons back into the PAGE document. In the current PAGE helpers, numeric id extraction prefers the `structure_line_id_` pattern in the PAGE `custom` attribute and otherwise falls back to digits in line identifiers or traversal order.

## Baseline Distance Function

Heatmap boxes are assigned to the nearest PAGE baseline.

Distance from a point to a baseline polyline is computed as the minimum Euclidean distance from the point to any segment in the baseline.

For a segment from `start` to `end` and a point `p`:

```text
segment = end - start
length_squared = dot(segment, segment)
```

If `length_squared` is zero, the segment is treated as a point and distance is:

```text
norm(p - start)
```

Otherwise:

```text
t = dot(p - start, segment) / length_squared
t_clamped = min(1, max(0, t))
projection = start + t_clamped * segment
distance = norm(p - projection)
```

For a baseline with one point, distance is the Euclidean distance to that point. For a missing or empty baseline, distance is infinite.

## Heatmap Box Extraction

The strategy loads the page image and heatmap through the canonical legacy image loader.

Image loader behavior:

- Read image with `skimage.io.imread`.
- If the loaded array's first dimension is `2`, keep only the first slice.
- If grayscale, convert to RGB.
- If RGBA, drop the alpha channel.

Heatmap handling:

- If the heatmap is 3D, use the first channel.
- Resize the heatmap to page image width and height using bilinear interpolation.

Bounding-box extraction then runs on the resized heatmap:

1. Convert `BINARIZE_THRESHOLD` to an 8-bit threshold:

   ```text
   threshold_value = int(BINARIZE_THRESHOLD * 255)
   ```

2. Apply OpenCV binary thresholding:

   ```text
   det > threshold_value -> foreground
   ```

3. Find external contours only:

   ```text
   cv2.RETR_EXTERNAL
   cv2.CHAIN_APPROX_SIMPLE
   ```

4. Convert each contour to an axis-aligned bounding rectangle:

   ```text
   (x, y, width, height) = cv2.boundingRect(contour)
   ```

These boxes are in page image coordinates because extraction happens after resizing the heatmap to the page image size.

## Heatmap Box Assignment To Baselines

For each heatmap bounding box:

1. Compute the page-coordinate box center:

   ```text
   center = (x + width / 2, y + height / 2)
   ```

2. Compute distance from the center to every loaded baseline.

3. Choose the baseline with the smallest distance.

4. Reject the box if the best distance is greater than:

   ```text
   max(20.0, height * 2.5)
   ```

5. If accepted, emit one synthetic node in heatmap coordinates using the conversion described above.

6. Emit the selected baseline's numeric line id as the synthetic label.

The distance gate is deliberately simple. It is not local tangent aware and does not model curved-line normal bands. A component close enough to a baseline centerline is treated as belonging to that line.

## Synthetic Legacy Helper Inputs

The wrapper creates a temporary old-format workspace under:

```text
<output_root>/_legacy_axis_bound_v1_geometry/
```

Before writing, it deletes any previous workspace at that exact path.

It then creates:

```text
images_resized/
heatmaps/
layout_analysis_output/gnn-format/
```

The page image is copied to:

```text
images_resized/<page_stem>.jpg
```

The heatmap is copied to:

```text
heatmaps/<page_stem>.jpg
```

The synthetic node file is written to:

```text
layout_analysis_output/gnn-format/<page_stem>_inputs_unnormalized.txt
```

Each node row is written with six decimal places:

```text
x y radius
```

The synthetic label file is written to:

```text
layout_analysis_output/gnn-format/<page_stem>_labels_textline.txt
```

Each label row contains one integer numeric line id.

These files imitate old GNN prediction outputs. No GNN inference is run by this strategy at this stage.

## Canonical Helper Invocation

The strategy calls the canonical helper:

```text
segmentLinesFromPointClusters(
    BASE_PATH = work_root,
    page = xml_path.stem,
    BINARIZE_THRESHOLD = normalized BINARIZE_THRESHOLD,
    BBOX_PAD_V = normalized BBOX_PAD_V,
    BBOX_PAD_H = normalized BBOX_PAD_H,
    CC_SIZE_THRESHOLD_RATIO = normalized CC_SIZE_THRESHOLD_RATIO,
    GNN_PRED_PATH = work_root / "layout_analysis_output"
)
```

Other helper arguments use defaults:

- `upscale_heatmap = true`
- `debug_mode = false`

Because `upscale_heatmap` is true, the helper:

- loads the page image and heatmap
- resizes the heatmap to page image size
- converts the page image to grayscale for processing
- scales synthetic point coordinates from heatmap coordinates back to page image coordinates
- generates heatmap bounding boxes at page image size

The helper returns a dictionary:

```text
line_numeric_id -> polygon_points
```

The strategy normalizes returned keys to integers and normalizes points to integer coordinate pairs.

## Helper Point And Label Loading

The helper loads the synthetic point file and label file.

Point behavior:

- Load with `numpy.loadtxt`.
- Force at least 2D shape.
- Convert to integer type.

Label behavior:

- Read one line at a time.
- Skip labels whose text is `none`, case-insensitive.
- Convert remaining labels to integers.

If the helper scales points from heatmap to page image size, it multiplies `x` by:

```text
image_width / heatmap_width
```

and `y` by:

```text
image_height / heatmap_height
```

The scaled coordinates are then converted to integers.

## Helper Box Labeling

The helper independently regenerates heatmap bounding boxes from the resized heatmap. It then labels those boxes using the synthetic points.

For each heatmap bounding box:

```text
x_min = x
x_max = x + width
y_min = y
y_max = y + height
```

Find synthetic points inside the box:

```text
x_min <= point_x <= x_max
y_min <= point_y <= y_max
```

If there are no points inside the box, the box is omitted.

If all contained points have the same label, the full bounding box is assigned that label:

```text
(x, y, width, height, label)
```

If contained points have multiple labels:

1. Sort contained points by `y`.
2. Start vertical boundaries with `y_min`.
3. For each adjacent point pair where the label changes, add the midpoint between their y coordinates.
4. End boundaries with `y_max`.
5. For each vertical segment, assign the label of the first point whose y coordinate lies inside that segment.
6. Emit one labeled sub-box per segment.

This multi-label split is axis-aligned and y-ordered. It assumes conflicts between lines can be separated by horizontal cuts, which is one reason the strategy is not appropriate as a curved-text model.

## Helper Line Type Detection

For each line label, the helper detects an approximate line type from the centers of the labeled boxes.

If there are fewer than two boxes:

```text
line_type = horizontal
model = null
```

Otherwise:

1. Sort centers by x coordinate.
2. Compute:

   ```text
   x_range = max_x - min_x
   y_range = max_y - min_y
   ```

3. If:

   ```text
   x_range < y_range * 0.3
   ```

   classify as `vertical`.

4. Else if:

   ```text
   y_range < x_range * 0.3
   ```

   classify as `horizontal`.

5. Else fit a RANSAC line model with random state `42`.

6. If the RANSAC score is greater than `0.85`, classify as `slanted`.

7. Otherwise classify as `curved` and fit a univariate spline with:

   ```text
   smoothing = number_of_centers * 2
   ```

If model fitting fails, the helper falls back to `horizontal`.

The current polygon generation path does not use the slanted or curved model to unwrap text. The detected type only affects padding orientation and connected-component cleanup axis.

## Dynamic Box Padding

For each line label, the helper processes that line's labeled heatmap boxes.

Padding ratios depend on the detected line type:

If the line is horizontal:

```text
padding_ratio_v = BBOX_PAD_V
padding_ratio_h = BBOX_PAD_H
```

Otherwise:

```text
padding_ratio_v = BBOX_PAD_H
padding_ratio_h = BBOX_PAD_V
```

For each original box:

```text
dynamic_pad_v = int(box_height * padding_ratio_v)
dynamic_pad_h = int(box_width * padding_ratio_h)
```

Initial crop bounds are:

```text
x1 = max(0, x - dynamic_pad_h)
y1 = max(0, y - dynamic_pad_v)
x2 = x + width + dynamic_pad_h
y2 = y + height + dynamic_pad_v
```

The crop is taken from the grayscale page image:

```text
blob = processing_image[y1:y2, x1:x2]
```

Upper bounds are effectively clipped by NumPy slicing if they exceed image size.

## Connected-Component Cleanup

Each padded box crop is cleaned with connected-component heuristics.

If the crop is empty, the helper returns an empty cleaned crop and a zero-size crop coordinate adjustment.

Otherwise:

1. Threshold the crop using:

   ```text
   cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
   ```

2. Run connected components with stats using 8-connectivity.

3. Initialize retained crop bounds in local crop coordinates:

   ```text
   top = 0
   bottom = crop_height
   left = 0
   right = crop_width
   ```

4. For each foreground component, compute:

   ```text
   component_x
   component_y
   component_width
   component_height
   component_area
   ```

5. Determine whether the component touches the relevant crop boundary.

   For non-vertical lines:

   ```text
   touches_boundary = component_y == 0
                      or component_y + component_height == crop_height
   ```

   For vertical lines:

   ```text
   touches_boundary = component_x == 0
                      or component_x + component_width == crop_width
   ```

6. Determine whether the component is small enough to be treated as boundary noise.

   For non-vertical lines:

   ```text
   size_constrained = component_height <= CC_SIZE_THRESHOLD_RATIO * crop_height
   ```

   For vertical lines:

   ```text
   size_constrained = component_width <= CC_SIZE_THRESHOLD_RATIO * crop_width
   ```

7. If both conditions are true, remove the boundary side by tightening the retained crop bounds.

   For non-vertical lines:

   ```text
   if component_y == 0:
       top = max(top, component_y + component_height)
   if component_y + component_height == crop_height:
       bottom = min(bottom, component_y)
   ```

   For vertical lines:

   ```text
   if component_x == 0:
       left = max(left, component_x + component_width)
   if component_x + component_width == crop_width:
       right = min(right, component_x)
   ```

8. If the retained bounds are invalid, return an empty cleaned crop.

9. Otherwise return:

   ```text
   cleaned_blob = blob[top:bottom, left:right]
   crop_coords = [top, bottom, left, right]
   ```

The helper then converts the local cleaned crop adjustment back to page coordinates:

```text
final_box_x = x1 + left
final_box_y = y1 + top
final_box_width = cleaned_blob_width
final_box_height = cleaned_blob_height
```

Only positive-width and positive-height cleaned boxes are retained.

## Rectangle Mask To Polygon

For each line label, the helper converts cleaned boxes into one line polygon.

1. Create a zero-valued binary mask with the same width and height as the page image.

2. Draw every cleaned box for that line as a filled white rectangle.

3. Find external contours in the mask.

4. If there are multiple contours, attempt to bridge disconnected groups.

The bridge logic is:

1. Compute the average height of the cleaned boxes:

   ```text
   avg_line_height = mean(box_height)
   ```

2. Group cleaned boxes by the contour containing their center point.

3. If grouping produces more than one group:

   - choose the first group as connected
   - keep the other groups unconnected
   - repeatedly find the closest pair of boxes between any connected group and any unconnected group
   - distance is Euclidean distance between box centers
   - choose left and right boxes by x coordinate
   - bridge from the right edge of the left box to the left edge of the right box
   - vertical bridge center is the mean of the two box center y values
   - bridge height is approximately `avg_line_height`
   - draw the bridge as a filled rectangle
   - mark the unconnected group as connected

Bridge rectangle coordinates:

```text
bridge_y_center = (left_box_center_y + right_box_center_y) / 2
bridge_y1 = bridge_y_center - avg_line_height / 2
bridge_y2 = bridge_y_center + avg_line_height / 2
bridge_x1 = left_box_x + left_box_width
bridge_x2 = right_box_x
```

After bridging, contours are recomputed.

If at least one contour exists, the helper selects the largest contour by area. The selected contour's points become the generated line polygon. Points are returned in OpenCV contour order.

The resulting polygon is therefore the contour of a union of axis-aligned rectangles and rectangular bridges. It is not a true baseline-parallel band and it is not an unwrapped curved-line representation.

## Scratch Line Images Written By The Helper

The helper also writes one cropped line image per generated polygon under:

```text
layout_analysis_output/image-format/<page_stem>/
```

The file name is:

```text
line<label + 1 padded to three digits>.jpg
```

For each polygon, the helper:

1. Computes the polygon bounding rectangle.
2. Crops the grayscale page image to that rectangle.
3. Creates a new image filled with the median page color.
4. Draws a shifted polygon mask.
5. Copies foreground pixels inside the polygon mask.
6. Writes the result as JPEG.

These scratch line images are a side effect of the legacy helper. The current strategy wrapper uses the returned polygons, not these images, as the canonical output.

Production OCR and the research OCR dataset preparation should use the final PAGE XML geometry and the configured OCR crop preparation module. In today's legacy behavior, that crop preparation is the app-style masked polygon crop described below.

## PAGE XML Replacement

After polygon generation, the strategy modifies the input PAGE XML.

It removes all existing `Coords` children under every `TextLine`.

It then walks the `TextLine` elements again and computes each line's numeric id with the same PAGE helper logic used during baseline loading.

For each line:

- If there is a generated polygon for that numeric id, insert a new `Coords`.
- If there is no generated polygon, leave the line without `TextLine/Coords`.

The inserted `Coords` has a `points` attribute formatted as:

```text
x1,y1 x2,y2 x3,y3 ...
```

The new `Coords` is inserted before `Baseline` when `Baseline` exists. If no `Baseline` child exists, it is inserted as the first child.

The strategy does not alter the baseline points themselves.

## OCR Crop Behavior For Legacy

The OCR crop behavior currently used after PAGE XML preparation is the generic PAGE polygon masked crop in `app/recognition/line_segmentation/ocr_crops.py`.

For each prepared line record:

1. Load the page image in grayscale.
2. Read the final PAGE `TextLine/Coords` polygon.
3. Convert the polygon to an OpenCV int32 contour.
4. Compute its axis-aligned bounding rectangle:

   ```text
   x, y, width, height = cv2.boundingRect(polygon)
   ```

5. Crop the page image to that rectangle.
6. Fill a new image with the median background value from the page image.
7. Shift polygon points by subtracting `(x, y)`.
8. Draw the shifted polygon as a filled mask.
9. Copy pixels from the crop into the new image where the mask is nonzero.
10. Encode to JPEG with quality `95`, then decode back to grayscale for app-compatible image behavior.

This crop is axis-aligned at the outer bounding rectangle level. It preserves pixels inside the generated polygon and replaces pixels outside the polygon with the median page background.

For `legacy_axis_bound_v1`, this is consistent with the generated geometry because the polygon is itself built from axis-aligned rectangles and rectangular bridges.

This exact crop behavior is the default legacy OCR crop implementation and the fallback when line-segmentation metadata is missing, malformed, unsupported, or delegated to legacy behavior.

## End-To-End Data Flow

The full strategy flow is:

```text
PAGE XML Baseline
    + page image
    + heatmap
        |
        v
load baseline records
        |
        v
resize heatmap to page image size
        |
        v
threshold heatmap and find external component boxes
        |
        v
assign each box center to nearest baseline, with distance rejection
        |
        v
write synthetic old-format point and label files in heatmap coordinates
        |
        v
call segmentLinesFromPointClusters
        |
        v
helper regenerates heatmap boxes and labels boxes by synthetic points
        |
        v
helper pads boxes, removes small boundary components, draws box masks
        |
        v
helper bridges disconnected masks and extracts largest contour
        |
        v
strategy receives line_numeric_id -> polygon
        |
        v
remove existing TextLine/Coords
        |
        v
insert generated TextLine/Coords by numeric line id
        |
        v
write PAGE XML and metadata
        |
        v
OCR dataset/inference crop preparation reads final PAGE Coords and optional metadata
        |
        v
legacy masked polygon crop creates OCR line image
```

## Reimplementation Contract

A compatible reimplementation must preserve these behaviors:

- Use PAGE `Baseline` as the source text-line centerline geometry.
- Ignore existing PAGE `TextLine/Coords` during strategy generation.
- Respect `include_empty_text_lines`.
- Use the default config values listed above.
- Use first heatmap channel when heatmap is multi-channel.
- Resize heatmap to page image size before initial component detection.
- Threshold heatmap using `int(BINARIZE_THRESHOLD * 255)`.
- Use external contours and axis-aligned bounding rectangles.
- Assign heatmap boxes to nearest baseline by point-to-polyline distance.
- Reject assignments farther than `max(20.0, heatmap_box_height * 2.5)`.
- Convert accepted box centers to synthetic heatmap-space nodes.
- Use the assigned baseline numeric id as the synthetic label.
- Feed synthetic nodes and labels through the old point-cluster helper semantics.
- In the helper, label boxes by contained synthetic points.
- Split multi-label boxes with y-axis midpoint cuts.
- Detect line type with the same horizontal, vertical, RANSAC, and spline fallback logic.
- Use dynamic padding ratios, swapping horizontal and vertical padding for non-horizontal line types.
- Remove small boundary-touching connected components using `CC_SIZE_THRESHOLD_RATIO`.
- Build final polygons from masks of cleaned axis-aligned rectangles.
- Bridge disconnected groups with rectangular bridges before selecting the largest contour.
- Replace only `TextLine/Coords` in PAGE XML.
- Preserve `Baseline` and `TextRegion/Coords`.
- Insert new `Coords` before `Baseline`.
- Write metadata with strategy name, geometry source, counts, coverage, and assignment rates.
- Keep the default OCR crop for this strategy as the masked polygon crop from final PAGE `Coords`.

## Known Limitations

This strategy is not a curved-text or circular-text strategy.

It can classify a set of boxes as `curved`, but that classification does not create a local tangent band, does not unwrap the line, and does not change the final OCR crop into a baseline-following crop. The final polygon is still the contour of axis-aligned boxes and rectangular bridges.

The strategy assumes heatmap connected components are meaningful local text blobs. If the heatmap misses a character or joins neighboring lines, the baseline assignment and contour generation inherit that error.

The box-to-baseline assignment uses a single nearest-distance rule. It does not consider reading order, local baseline tangent, interline spacing, character orientation, or text-region membership.

The conflict split for multi-label boxes is y-axis based. It is suited to roughly horizontal line separation and can be wrong for vertical, highly slanted, circular, or densely curved writing.

The strategy can omit lines. A line is omitted when no heatmap boxes are assigned to its numeric id or when later cleanup produces no polygon. Omitted lines remain without `TextLine/Coords` after replacement.

The strategy updates line polygons only. It does not recompute text-region polygons, reading order, baseline geometry, OCR text, or active-learning checkpoint metadata.

The baseline-derived research harness cannot perfectly reconstruct every manual graph edit from PAGE baseline alone. PAGE baselines do not preserve every added or deleted graph node. That is why research gates track geometry coverage metrics such as `source_line_coverage` and `heatmap_box_assignment_rate`.

## Behavior-Preserving Refactor Contract

The production strategy-aware OCR crop refactor makes `legacy_axis_bound_v1` the default compatibility strategy for both geometry and crop preparation.

The explicit production crop strategy boundary has behavior equivalent to:

```text
crop_strategy = masked_pagexml_coords
geometry_strategy = legacy_axis_bound_v1
```

For legacy behavior, OCR line images should remain byte-level or near-byte-level equivalent subject to existing JPEG encoding and OpenCV version differences.

The crop layer makes this contract visible in metadata:

- production geometry strategy name
- OCR crop strategy name
- whether the crop came from PAGE `Coords`
- whether any strategy-specific unwrapping was applied
- fallback behavior when strategy metadata is missing

For existing manuscripts without new metadata, fallback should be:

```text
geometry_strategy = legacy_axis_bound_v1 or unknown_legacy
crop_strategy = masked_pagexml_coords
unwrapped = false
```

The refactor does not silently change production OCR crop behavior while adding modularity. `local_tangent_band_v1` or any future curved-line strategy still requires explicit production adoption before newly saved production pages carry metadata that asks OCR to use unwrapped crops.

## Verification Guidance

A behavior-preserving implementation should be checked at several levels:

- Unit test baseline record loading and numeric id mapping.
- Unit test point-to-baseline distance for horizontal, vertical, slanted, single-point, and empty baselines.
- Unit test heatmap box assignment, including the `max(20.0, height * 2.5)` rejection rule.
- Golden test generated PAGE `Coords` for a small fixture page.
- Golden test masked OCR crop shape and image content for a fixture polygon.
- Regression test that existing `TextLine/Coords` are ignored as input and replaced.
- Regression test that `TextRegion/Coords` and `Baseline` are preserved.
- Regression test both empty-line modes.
- Research gate check for `source_line_coverage` and `heatmap_box_assignment_rate`.

For production refactor verification, compare OCR crops before and after the refactor while using the legacy strategy. Any differences should be explained by file encoding, dependency version, or explicitly approved metadata-only changes.
