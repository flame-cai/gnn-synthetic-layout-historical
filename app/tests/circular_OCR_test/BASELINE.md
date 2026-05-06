# Baseline

`baseline.json` is the machine-readable accepted baseline consumed by the circular gate. The gate reads `primary_blocking_metric_name`, `metric_direction`, `minimum_improvement_delta`, and `metrics`.

The intended loop is:

1. Run the current accepted strategy.
2. Inspect `circular_ocr_latest.json`, per-page CSV, per-line CSV, segmentation metadata, and orientation metadata.
3. If the result is manually accepted, update `baseline.json` with the accepted metrics and strategy.
4. The next iteration must beat that baseline by at least `minimum_improvement_delta`.

The initial checked-in baseline is a seed control record for enabling the harness. Replace it with a real current-control run when you want strict numeric gating against the known-bad control.

