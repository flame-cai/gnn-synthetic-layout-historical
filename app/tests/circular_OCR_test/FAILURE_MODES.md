# Failure Modes

## Diagonal Tangent Over/Under-Cropping

- Symptom: crops include too much or too little foreground near roughly 45, 135, 225, and 315 degree tangent regions.
- Evidence: observed current behavior in `eval_dataset_v2`.
- Likely cause: global page x/y padding and rectangle bridging instead of local tangent/normal padding.
- Priority: high.
- Related ideas: `LOCAL-TANGENT-BAND-V1`, `DIAGONAL-SPILLOVER-PENALTY`.
- Status: active.

## Clockwise/Counterclockwise Ambiguity

- Symptom: circular crops may read in the wrong direction.
- Evidence: topological circular cut alone does not define reading order.
- Likely cause: direction choice is ambiguous after unwrapping.
- Priority: high.
- Related ideas: `OCR-CONFIDENCE-ORIENTATION`.
- Status: active.

## Glyph Upright Ambiguity

- Symptom: unwrapped text may be upside down even when reading order is correct.
- Evidence: local normals can choose either side of the baseline.
- Likely cause: baseline orientation and glyph upright direction are not separately encoded.
- Priority: high.
- Related ideas: `OCR-CONFIDENCE-ORIENTATION`.
- Status: active.

## Topmost-Cut Mismatch

- Symptom: circular text is split at a visually awkward or wrong point.
- Evidence: current convention is dataset-specific.
- Likely cause: future data may not follow the topmost annotation convention.
- Priority: medium.
- Related ideas: `LOCAL-TANGENT-BAND-V1`.
- Status: monitored.

## OCR-Confidence Misselection

- Symptom: selector chooses a high-confidence wrong orientation.
- Evidence: confidence is not a calibrated correctness guarantee.
- Likely cause: OCR model can be confidently wrong on unusual line crops.
- Priority: medium.
- Related ideas: `OCR-CONFIDENCE-ORIENTATION`.
- Status: monitored.

## Page-Space Coords Replaced By Crop Geometry

- Symptom: PAGE-XML `Coords` no longer describe manuscript layout.
- Evidence: would break PAGE-XML consumers and GUI assumptions.
- Likely cause: mixing segmentation polygons with unwrapped OCR rectangles.
- Priority: blocking.
- Related ideas: all.
- Status: guarded by unit tests and `STRATEGY_CONTRACT.md`.

