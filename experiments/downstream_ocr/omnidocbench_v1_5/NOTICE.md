# OmniDocBench v1.5 source notice

This directory contains the smallest source boundary required to calculate the
downstream OCR TextEdit metric with OmniDocBench v1.5.

- Upstream repository: <https://github.com/opendatalab/OmniDocBench>
- Upstream branch: `v1_5`
- Pinned Git commit: `59b103c4b47d3a01fada83491585d6512a40c0bc`
- Upstream license: Apache License 2.0 (see `LICENSE`)

The following definitions were extracted from the pinned upstream source
without changing their calculation:

- `utils.data_preprocess.textblock2unicode`
- `utils.data_preprocess.clean_string`
- `utils.match.get_gt_pred_lines`
- `utils.match.compute_edit_distance_matrix_new`
- `utils.match.match_gt2pred_simple`
- `metrics.cal_metric.call_Edit_dist`

Unrelated OmniDocBench table, formula, Markdown, CDM, BLEU, and METEOR code was
not copied. Imports were made package-relative for this repository.

One upstream diagnostic-only issue was patched in `utils/match.py`: prediction
index zero is tested with `pred_idx != ""` rather than Python truth-value
checks. This changes only `pred_category_type` and `pred_position` metadata for
prediction index zero; the cost matrix, Hungarian assignment, matched text, and
numeric TextEdit score are unchanged.

