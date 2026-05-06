# Circular OCR Idea Pool

Keep one testable idea per id. Remove ideas only after the human decides they are no longer useful.

## Active Ideas

- `LOCAL-TANGENT-BAND-V1`: Use PAGE-XML baselines as open curves, cut circular baselines at the topmost point, and build page-space polygons from local normal bands.
- `OCR-CONFIDENCE-ORIENTATION`: Generate forward, reversed, forward vertical flip, and reversed vertical flip candidates, then select by OCR confidence without reading ground truth.

## Deferred Ideas

- `ADAPTIVE-BAND-WIDTH`: Estimate per-line band width from heatmap and gnn-labeled foreground instead of a fixed half width.
- `DIAGONAL-SPILLOVER-PENALTY`: Add tangent-angle-aware filtering near diagonal regions where current global padding fails.

