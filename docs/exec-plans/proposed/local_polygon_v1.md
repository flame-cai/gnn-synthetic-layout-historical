can you think of a strategy (local_polygons_v1) which will generalize segmentLinesFromPointClusters(...) (which currently generates axis-aligned-ish PAGE polygons), which will generate similar axis-aligned PAGE polygons by treating each curve as "locally as a curve". We want this because segmentLinesFromPointClusters(...) really has good heuristics for not including unnecessary text coming from adjacent line from the "top" or "bottom". this notion of top and bottom also perhaps can be generalized. Once these heuristics are applied to the text-line in a generalized way, we want to move to next step of unwrapping. We want this generalized version to ideally work exactly the same for horizonal lines, and also work with the same spirit for curved lines (unlike the band strategy, which does not apply any heuristics to not include unnecessary text from adjacent lines)

We want to unwrap without delegating horizontal lines to the legacy strategy. We just want one general strategy which unwraps the PAGE-polygons  into a horizonal strip


#### proposed new strategy
The clean strategy is not “band then unwrap”; it is legacy polygon heuristics in a local coordinate system, then unwrap from those tighter polygons.

I’d call it something like local_polygons_v1.

Core Idea
For each line, build a local coordinate frame around the baseline:

s = distance along baseline
n = signed distance normal to baseline
Then convert the line’s heatmap components into this (s, n) space. In that space, every line is treated as locally horizontal. Now the old legacy idea of “top” and “bottom” becomes:

top    = negative normal side
bottom = positive normal side
For horizontal lines, this local frame is essentially the original image frame, so the strategy can behave like segmentLinesFromPointClusters(...) without delegating to it.

Proposed local_polygons_v1 Pipeline

Normalize baseline
Use the same topology cleanup we discussed: remove graph-walk backtracking, handle circular paths, enforce reading direction. This is mandatory before any local geometry.

Assign heatmap components to lines
Threshold heatmap and extract connected-component boxes, as legacy does. But assignment uses nearest point on the normalized baseline, not axis-aligned containment.

Project each component into local line coordinates
For each heatmap component, sample its corners or mask pixels into (s, n). It becomes a local rectangle/blob in unwrapped line space.

Run legacy-style cleanup in local space
This is the key part. For each component/local blob:

pad along n using the legacy vertical padding idea
pad along s using the legacy horizontal padding idea
binarize the corresponding local image patch
remove small connected components touching the local top/bottom boundary
for vertical/curved/circular text, “top/bottom” means normal-side boundaries, not page y-boundaries
Build a tight local polygon
Union the cleaned local rectangles/masks for that line. Bridge nearby disconnected groups in local s order, using local line height, same spirit as legacy. Then contour the union mask in (s, n) space.

Map polygon back to PAGE space
Convert the local polygon boundary back through the baseline transform:

page_point = baseline_point_at_s + n * local_normal_at_s
This gives PAGE Coords. The polygon is no longer globally axis-aligned for curved lines, but it is “axis-aligned in local line coordinates,” which is the right generalization.

Unwrap OCR crop from this tight polygon
Use the same baseline-local remap as current local_tangent_band_v1, but mask with the new tight local polygon, not the broad ribbon. Pixels outside the cleaned local polygon become page median color.

Why This Fixes The Median Background Issue
local_tangent_band_v1 masks with a broad ribbon around the whole baseline, so almost every pixel survives.

local_polygons_v1 would mask with the cleaned component-derived polygon. That means the unwrapped crop should look like:

ink / real text pixels inside cleaned local polygons
median-color background outside them
This is much closer to legacy OCR crops.

Why Horizontal Lines Can Match Legacy
For a normal horizontal line:

s ~= page x
n ~= page y - baseline_y
So local top/bottom are the same as image top/bottom. If we keep the same heatmap thresholding, dynamic padding, boundary-touching component cleanup, bridge heuristic, and contour extraction, the horizontal output can be made effectively identical to legacy without calling the legacy strategy.

The main risks are:

baseline normalization must be fixed first
circular lines need careful seam selection so components do not split badly at s=0
PAGE polygons from local masks may have many points and may need simplification
exact legacy parity for horizontal lines should be protected with tests before expanding to curves


#### WEIRD LINE FIX: INVESTAGTE BASELINE NORMALIZATION ####


TODO: investigate all Baselines in all the PAGE-XMLS and try to find such issues in app/tests/eval_dataset, app/tests/eval_dataset_v2 and all manuscripts in app/input_manuscripts

Yes. The root cause is likely the baseline topology, not the median-fill code.
For word_0020.png / line 19, the source baseline is:
text

756,816 -> 806,814 -> 856,818 -> 806,814

That is a short retraced graph walk: A -> B -> C -> B.
Current normalize_baseline_topology(...) only detects out-and-back paths when there are at least min_mirror_pairs=3. For A-B-C-B, there is only one mirror pair, so it is not normalized. It stays curved_open, gets crop_model="local_tangent_band", then _band_polygon(...) builds a self-crossing-ish band from the folded baseline. OCR unwrapping then remaps that folded band, producing the strange crop.
The source of these baselines is also important: trace_component_with_backtracking (line 362) creates a DFS traversal that explicitly appends parent nodes while backtracking. That is fine as a graph traversal, but it is not always a clean PAGE baseline.
Best Fix Direction
Do not patch the cropper first. Fix baseline normalization, because both strategy geometry and OCR unwrapping call it.
I’d recommend a two-part normalization hardening:
Handle short retraced tails
Detect short patterns like A-B-C-B and A-B-C-D-C-B even when they have fewer than 3 mirror pairs. For line 19, this would normalize to:
text



A -> B -> C

Then it classifies as horizontal_straight, and local_tangent_band_v1 delegates it to legacy masked cropping instead of unwrapping.

Handle small branched retraced walks
I found other examples like A-B-C-A-C-B. A simple “allow one mirror pair” rule can misclassify that as a fake closed loop. A safer repair is: when a baseline has repeated/near-repeated points and the unique points form a straight horizontal/vertical line, collapse to unique points ordered along the dominant axis.

This should be guarded by tests so true circular paths still remain closed_circular.
Concrete tests I’d add before changing behavior:
A-B-C-B normalizes to A-B-C, line_kind="horizontal_straight".
A-B-C-A-C-B normalizes to ordered unique straight points, not closed_circular.
Existing circular out-and-back test still passes.
A local strategy prep for short retraced horizontal baseline produces crop_model="legacy_axis_bound_delegate" and does not unwrap.
After this fix, old generated crops will not change until regenerated, but new research harness prep should stop producing that folded local-tangent crop for line 19.









