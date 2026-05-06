# Shared Generalized Segmentation Ablation Gates

## Summary
Create two new blocking pre-commit gates that mirror the existing `Full Pipeline Gate` and `Recognition Fine-Tune Gate`, but replace only the text-line segmentation method with the generalized strategy path. The original gates remain unchanged for baseline comparison. The two new ablation gates and the existing `Circular Layout OCR Gate` must use the exact same shared generalized segmentation strategy code, so every future strategy iteration is evaluated consistently across all three generalized-strategy checks.

## Key Changes
- Extract the circular harness segmentation logic into one shared generalized segmentation package under `app/`.
- Replace circular-harness-local strategy code with calls into this shared strategy registry.
- Add a single active strategy selector, defaulting to `local_tangent_band_v1`, used by:
  - generalized full-pipeline gate
  - generalized recognition fine-tune gate
  - circular layout OCR gate
- Keep GUI/default behavior unchanged. Generalized segmentation remains opt-in unless explicitly selected by a test/gate.
- Do not add OCR unwrapping or orientation-selection changes to the two new ablation gates now. However, structure the shared strategy boundary so future strategy iterations can expand beyond segmentation without duplicating code.

## New Gates
- Add `Full Pipeline Generalized Segmentation Gate`:
  - mirrors `test_ci_e2e.py`
  - same `eval_dataset`, upload flow, GNN inference, save, local OCR, evaluation, and thresholds
  - only difference: save/generation uses the shared generalized segmentation strategy instead of the legacy axis-bound segmenter.
- Add `Recognition Fine-Tune Generalized Segmentation Gate`:
  - mirrors `test_recognition_finetuning_precommit_e2e.py`
  - same OCR recipe, page split, thresholds, active-learning policy, and metrics
  - rewrites temporary PAGE-XML under `app/tests/logs/...`
  - generates new `TextLine/Coords` from PAGE-XML `Baseline`, explicitly ignoring original `Coords`.
- Add both gates as default blocking phases in `scripts/run_precommit_eval.py`.
- Add skip flags:
  - `SKIP_PIPELINE_GENERALIZED_SEGMENTATION_HOOK=1`
  - `SKIP_RECOGNITION_GENERALIZED_SEGMENTATION_HOOK=1`

## Interfaces And Artifacts
- Shared segmentation API returns the existing `polygons_data` contract used by `create_page_xml`: `{line_label: {"points": page_space_polygon, "image": masked_crop}}`.
- Shared strategy inputs support both modes:
  - predicted pipeline mode: page image, heatmap, GNN points, predicted labels/edges/baselines
  - ground-truth recognition mode: page image, PAGE-XML `Baseline`, text, and optional GNN-format labels when available
- Circular OCR keeps its separate OCR unwrapping/orientation-selection steps, but its segmentation step must call the same shared strategy registry.
- Latest artifact pointers:
  - `ci_eval_generalized_segmentation_latest.txt/json`
  - `recognition_finetune_generalized_segmentation_latest.md/json/txt`
- Metadata must record active strategy name, strategy config, input source, and proof that recognition ablation did not use original PAGE-XML `Coords`.

## Test Plan
- Unit tests:
  - shared strategy produces page-space polygons from straight, vertical, curved, and circular baselines.
  - recognition ablation ignores bad original `Coords` and uses `Baseline`.
  - circular harness and both new ablation gates resolve the same active strategy name/config.
  - shared API output remains compatible with `create_page_xml`.
- Gate tests:
  - `python -m unittest tests.test_ci_e2e_generalized_segmentation -v`
  - `python -m unittest tests.test_recognition_finetuning_precommit_generalized_segmentation_e2e -v`
  - `python -m unittest tests.test_circular_ocr_precommit_e2e -v`
  - `python scripts/run_precommit_eval.py`
- Existing original gates must still pass unchanged:
  - Full Pipeline Gate
  - Recognition Fine-Tune Gate

## Assumptions
- The new ablation gates are blocking default pre-commit checks.
- `eval_dataset` has no checked-in GNN-format ground-truth labels, so the recognition ablation uses PAGE-XML `Baseline` as ground-truth line geometry and explicitly ignores `Coords`.
- For now, these two new gates are line-segmentation ablations only. Future unwrapping/orientation strategy work should still route through the same shared strategy/config mechanism when it becomes part of the generalized experiment.
