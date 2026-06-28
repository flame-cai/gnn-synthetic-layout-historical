# Make Production OCR Cropping Strategy-Aware

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This document is maintained in accordance with `PLANS.md` from the repository root.

## Purpose / Big Picture

After this change, adopting a text-line segmentation strategy for production will adopt both halves of the strategy: how the app writes PAGE-XML `TextLine/Coords`, and how the app turns saved PAGE-XML lines into OCR-ready images. At the time of this refactor, the production strategy was `legacy_axis_bound_v1`, so the visible behavior needed to remain effectively the same before and after the refactor: horizontal manuscript pages still produced the same masked line crops, local OCR still read saved PAGE XML, and active-learning jobs still trained from saved page revisions.

The important gain is modularity. Today the research harness has a strategy-aware OCR crop path, but production local OCR and active-learning training mostly crop directly from saved `Coords`. After this refactor, production code will call a shared strategy-aware crop preparation layer. With `legacy_axis_bound_v1`, that layer will still choose the existing axis-aligned masked crop. If a future operator explicitly adopts `local_tangent_band_v1` for production, the same production layer will be able to use local-tangent unwrapping for lines whose saved strategy metadata says they were produced by local-tangent geometry.

This plan intentionally keeps OCR preparation as two phases:

    durable page geometry:
        Baseline + image + heatmap -> PAGE TextLine/Coords + strategy metadata

    derived OCR representation:
        PAGE TextLine/Coords + Baseline + strategy metadata -> OCR-ready crop image

The two phases should stay separate because PAGE XML is the durable, editable, auditable layout representation. OCR crops are model inputs and may be unwrapped, padded, normalized, or otherwise transformed in ways that are not true page-space geometry. A local-tangent unwrapped strip must never be written back as PAGE `Coords`.

This refactor should make the second phase strategy-aware without collapsing it into the first phase. Promotion to production becomes easier because the app will have one shared place to honor strategy crop behavior, but promotion is still not automatic. Research promotion says a strategy passed harness gates. Production adoption still requires an explicit adoption step and production validation.

## Progress

- [x] (2026-05-15 13:53 IST) Created this proposed ExecPlan after confirming that production layout save already regenerates PAGE `Coords` from a fresh baseline PAGE XML, while production OCR inference and active-learning training still crop from saved `Coords`.
- [x] (2026-05-15 14:22 IST) Hardened the plan with the two-phase architecture rationale, guardrails that keep PAGE geometry separate from OCR crop images, and phased implementation constraints for production adoption.
- [x] (2026-05-15 14:50 IST) Implemented shared strategy-aware OCR crop module at `app/recognition/line_segmentation/ocr_crops.py`.
- [x] (2026-05-15 14:50 IST) Refactored research dataset preparation to use the shared crop module while keeping `baseline_heatmap` strategy generation behavior.
- [x] (2026-05-15 14:50 IST) Refactored production app line-image export to use the shared cropper and the metadata sidecar written during layout save.
- [x] (2026-05-15 14:50 IST) Refactored production local OCR inference to use shared crop extraction with sibling metadata discovery and masked-crop fallback.
- [x] (2026-05-15 14:50 IST) Refactored active-learning snapshots and revision preparation to preserve and consume line-segmentation metadata without requiring heatmaps.
- [x] (2026-05-15 14:50 IST) Added tests for missing/malformed/legacy/delegated/local-tangent crop metadata, PAGE XML immutability, local OCR fallback, and active-learning metadata snapshot plumbing.
- [x] (2026-05-15 14:50 IST) Updated repository docs and text-line strategy docs to describe strategy-aware production OCR crops and checked-in promotion record generation.
- [x] (2026-05-15 14:57 IST) Ran targeted unit tests, OCR pre-commit unit gate, and fast full-pipeline gate; all passed. Slow OCR e2e ablation gates were not run in this pass.

## Surprises & Discoveries

- Observation: production PAGE `Coords` generation already starts from `Baseline`.
  Evidence: `app/gnn_inference.py::generate_xml_and_images_for_page(...)` writes `layout_analysis_output/_baseline_page_xml/<page>.xml`, then calls `apply_text_line_segmentation_strategy(...)` with that baseline XML as `source_pagexml_path` and writes the final XML under `layout_analysis_output/page-xml-format/`.

- Observation: production local OCR inference still performs its own direct `Coords` crop.
  Evidence: `app/recognition/recognize_manuscript_text_v2_pretrained.py::process_page_xml(...)` reads each `TextLine/Coords`, computes `cv2.boundingRect(...)`, masks the polygon inside that rectangle, and feeds the resulting image to OCR. It does not call the strategy-aware crop/unwrapping code from `app/recognition/pagexml_line_dataset.py`.

- Observation: GUI active-learning training snapshots final PAGE XML and image files, then prepares OCR data from saved `Coords`.
  Evidence: `app/ocr_active_learning_runtime.py::_snapshot_page_revision(...)` copies final PAGE XML and resized image into the revision snapshot. `_prepare_revision_pages(...)` calls `prepare_page_datasets(...)` without a strategy name or heatmap directory, so the default `geometry_source="pagexml_coords"` path is used.

- Observation: `app/tests/logs/` evidence is ignored locally, so durable promotion summaries need a checked-in home.
  Evidence: `scripts/promote_text_line_strategy.py::write_strategy_promotion_evidence(...)` now refreshes `docs/pipeline-improvement/text-line-segmentation/strategy-promotion-record.md` in addition to the ignored latest log files.

## Decision Log

- Decision: treat OCR crop preparation as part of a production strategy adoption, but keep PAGE XML geometry and OCR crop images as separate representations.
  Rationale: PAGE `Coords` are page-space manuscript geometry. OCR unwrapping is a derived image representation for recognition. Writing unwrapped rectangles back into PAGE XML would corrupt the page-space layout model.
  Date/Author: 2026-05-15 / Codex

- Decision: keep production strategy adoption explicit even after research promotion succeeds.
  Rationale: research promotion proves a strategy passed harness gates against a benchmark. Production adoption changes GUI save behavior, local OCR input images, and active-learning training data. Those are related but not the same operational decision.
  Date/Author: 2026-05-15 / Codex

- Decision: choose OCR crop behavior from per-line metadata when possible, not from strategy name alone.
  Rationale: `local_tangent_band_v1` can delegate simple horizontal lines to legacy behavior while using local-tangent crops for curved, vertical, or circular lines. A page-level strategy name is not enough to know what each line needs.
  Date/Author: 2026-05-15 / Codex

- Decision: refactor production to use a shared crop module rather than making `local_tangent_band_v1` a special case in each caller.
  Rationale: the goal is smooth future promotion/adoption. Production line-image export, local OCR inference, and active-learning training should all ask the same code how to crop a line for OCR.
  Date/Author: 2026-05-15 / Codex

- Decision: preserve then-current behavior for `legacy_axis_bound_v1`.
  Rationale: this was a modularization plan, not a production rollout of a new recognition behavior. With the 2026-05-15 production pin, outputs needed to remain axis-aligned masked crops.
  Date/Author: 2026-05-15 / Codex

- Decision: use saved strategy metadata when available, and fall back to the existing masked crop when metadata is missing.
  Rationale: existing manuscripts and snapshots may not have strategy metadata sidecars. Missing metadata must not break OCR or trigger unexpected migration.
  Date/Author: 2026-05-15 / Codex

## Outcomes & Retrospective

Implemented on 2026-05-15. The production app behavior initially remained masked-crop compatible for `legacy_axis_bound_v1`, while app line-image export, local OCR inference, research dataset preparation, and active-learning revision training shared one crop decision layer that could honor local-tangent metadata after explicit production adoption.

Follow-up implemented on 2026-05-23: production adopted `local_polygons_v1` for future layout saves/regenerations. New saves now write strategy metadata for `crop_model="local_polygon_unwrap"` and optional reading-direction metadata. Existing pages remain unmigrated and continue to use the masked PAGE `Coords` fallback when metadata is missing or unsupported.

Follow-up implemented on 2026-05-30: production adopted `local_polygons_stable_unwrap_v1` after the strategy became the research benchmark and gained explicit production runtime config. New saves now write strategy metadata for `crop_model="local_polygon_stable_unwrap"`. The strategy preserves `local_polygons_v1` PAGE `Coords` geometry for the same inputs while changing the derived OCR line image to the stable unwrap path.

Validation run on 2026-05-15:

    conda run -n gnn_layout python -m unittest app.tests.test_strategy_aware_ocr_crops_unit -v
    conda run -n gnn_layout python -m unittest app.tests.test_strategy_promotion_unit -v
    conda run -n gnn_layout python -m unittest app.tests.test_line_segmentation_strategy_unit -v
    conda run -n gnn_layout python -m unittest app.tests.test_recognition_active_learning_unit -v
    conda run -n gnn_layout python -m unittest app.tests.test_recognition_active_learning_backend_unit -v
    conda run -n gnn_layout python -m unittest app.tests.test_recognition_finetuning_precommit_unit -v
    conda run -n gnn_layout python -m unittest discover -s app/tests -p "test_ci_e2e.py" -v

The slow surrogate OCR e2e gates were not run in this implementation pass.

## Context and Orientation

This repository has a semi-automatic manuscript annotation app under `app/`. The app writes PAGE XML, a document layout format where each `TextLine` may contain a `Baseline` and `Coords`. In this plan, `Baseline` means a polyline that follows the center path of a text line. `Coords` means a page-space polygon around that text line.

The current strategy registry is in `app/recognition/line_segmentation/registry.py`. It registers `legacy_axis_bound_v1`, `local_tangent_band_v1`, `local_polygons_v1`, `local_polygons_hstraight_smooth_unwrap_v1`, and `local_polygons_stable_unwrap_v1`.

`legacy_axis_bound_v1` is the current production strategy. It reads PAGE `Baseline`, the page image, and a heatmap, then writes page-space `Coords` using the historical axis-bound polygon method. Its OCR crop behavior is the old masked crop: take a bounding rectangle around `Coords`, fill a new image with the page median color, and copy pixels inside the polygon mask.

When this refactor started, `local_tangent_band_v1` was the proposed research strategy. It also writes page-space `Coords`, but for vertical, curved, and circular lines it can later unwrap the line into a horizontal OCR crop using both `Coords` and `Baseline`. The unwrapped OCR crop is a derived image; it must not be written back as PAGE `Coords`. Current research roles live in `app/recognition/line_segmentation/strategy_config.py`.

The research harness now reaches strategy-aware crop preparation through `app/recognition/line_segmentation/ocr_crops.py`, called from `app/recognition/pagexml_line_dataset.py::prepare_page_line_dataset(...)`. When a strategy name is supplied, it regenerates `Coords` and uses saved per-line strategy metadata to decide whether to call `unwrap_line_crop_for_ocr(...)` or fall back to `masked_line_crop(...)`.

Before this implementation, production differed in three important places. These are now refactored:

`app/gnn_inference.py::_write_app_line_images_from_pagexml(...)` writes app line images after layout save. It now reads final PAGE XML, loads the sibling metadata sidecar, and calls the shared cropper for every line.

`app/recognition/recognize_manuscript_text_v2_pretrained.py::process_page_xml(...)` runs local OCR inference. It now calls `extract_ocr_line_crops_from_page_xml(...)`, which discovers sibling metadata and falls back to masked crops.

`app/ocr_active_learning_runtime.py::_prepare_revision_pages(...)` prepares active-learning training data from revision snapshots. It keeps `geometry_source="pagexml_coords"` and now passes each snapshot page XML directory as the metadata sidecar source.

The production layout save path itself is already strategy-aware for PAGE XML generation. `app/gnn_inference.py::generate_xml_and_images_for_page(...)` builds a fresh baseline PAGE XML from edited graph nodes and edges, applies `get_production_strategy_name()`, and writes final PAGE `Coords`.

## Architectural Guardrails

Keep the two phases explicit in code, names, tests, and documentation.

The geometry phase owns durable PAGE layout:

    input: graph-derived PAGE Baseline, page image, heatmap, strategy config
    output: final PAGE TextLine/Coords and line segmentation metadata

The OCR crop phase owns model input preparation:

    input: final PAGE TextLine/Coords, PAGE Baseline, page image, strategy metadata, crop config
    output: OCR-ready crop image and crop metadata

The OCR crop phase may unwrap or normalize a line image. It must not mutate PAGE geometry. PAGE `Coords` remain page-space polygons even when the OCR model receives a horizontal unwrapped strip.

Do not let callers infer crop behavior only from `production_strategy_name`. The cropper must look at per-line metadata when it exists. This matters because `local_tangent_band_v1` can intentionally preserve legacy behavior for simple horizontal lines while using local-tangent behavior for curved or circular lines.

Do not require heatmaps during normal production OCR inference or active-learning training from saved revisions. Production snapshots already contain final PAGE `Coords`. Heatmaps are required for strategy geometry regeneration, not for reading an already-saved production revision.

Do not migrate old manuscripts as a side effect of this refactor. Existing PAGE XML without strategy metadata remains valid and must fall back to the masked `Coords` crop.

Do not silently treat research promotion as production adoption. The research workflow may promote a benchmark strategy for future harness comparisons. Production adoption must remain a separate script/config change that changes future app saves and OCR crops only after explicit operator intent.

Do not change the research harness or pre-commit ablation gates as part of production adoption. In particular, adopting the current research benchmark for the app must not change benchmark/proposed comparison semantics, add benchmark-only skip behavior, or alter research gate runners under `app/tests/`. Any such change is a separate research-harness maintenance task and needs separate justification.

The fallback behavior must be boring and predictable:

    missing metadata -> masked PAGE Coords crop
    malformed metadata -> log warning, masked PAGE Coords crop
    unsupported crop_model -> log warning, masked PAGE Coords crop
    legacy_axis_bound_v1 -> masked PAGE Coords crop
    local_tangent_band_v1 + crop_model=legacy_axis_bound_delegate -> masked PAGE Coords crop
    local_tangent_band_v1 + crop_model=local_tangent_band -> local-tangent unwrap

## Plan of Work

Start by adding a shared crop module under `app/recognition/line_segmentation/`. A good name is `ocr_crops.py`. This module owns the decision "given one PAGE line, what image should OCR see?" It should not generate or mutate PAGE XML. It should consume final PAGE XML records, a page image, an optional strategy name, optional per-line strategy metadata, and optional crop config, then return a crop image plus small metadata.

The new module should move or wrap the existing behavior from `app/recognition/pagexml_line_dataset.py` rather than retyping it. Move the old masked-crop operation into the new module as a public function such as `masked_line_crop(...)`. Move the local-tangent branch into a function such as `crop_line_record_for_ocr(...)`. This function should have this behavior:

    If strategy_name is "local_tangent_band_v1" and the line metadata has crop_model == "local_tangent_band", call unwrap_line_crop_for_ocr(...).
    Otherwise, call masked_line_crop(...).

This preserved production behavior for `legacy_axis_bound_v1`, because legacy lines fall through to the masked crop.

Add a helper that loads line segmentation metadata from the sidecar JSON written next to final PAGE XML. Production layout save currently writes metadata to `layout_analysis_output/page-xml-format/<page>_line_segmentation_metadata.json`. The loader should return a mapping from integer `line_numeric_id` to that line's metadata. If the file is missing, invalid, or does not contain a line, the caller should get an empty mapping and the cropper should use the masked crop.

Update `app/recognition/pagexml_line_dataset.py` to import and use the new crop module. This keeps the research harness on the same crop code that production will use. `prepare_page_line_dataset(...)` should continue to support both modes: strategy generation from `Baseline` plus heatmap, and existing `pagexml_coords` input. When strategy metadata is available without rerunning strategy generation, it should still be able to choose the appropriate OCR crop.

Update `app/gnn_inference.py::_write_app_line_images_from_pagexml(...)`. Change its signature so it can accept `strategy_name`, `strategy_metadata_path`, and `strategy_config`. In `generate_xml_and_images_for_page(...)`, pass the `strategy_result.strategy_name` and `strategy_result.metadata_path` returned by `apply_text_line_segmentation_strategy(...)`. The function should still write the same folder layout under `layout_analysis_output/image-format/<page>/<textbox_label>/line_<id>.jpg`. With `legacy_axis_bound_v1`, it should still write masked crops.

Update production local OCR inference in `app/recognition/recognize_manuscript_text_v2_pretrained.py`. Avoid duplicating crop logic inside `process_page_xml(...)`. Add or use a helper from the shared module that reads `load_pagexml_lines(...)`, loads optional strategy metadata, and returns `(PIL.Image, line_context)` pairs. `process_page_xml(...)` should keep its public behavior of modifying the PAGE XML with recognized text. It should accept optional keyword arguments for `line_segmentation_strategy_name`, `line_segmentation_metadata_path`, and `crop_config`, defaulting to automatic discovery of the sibling metadata file. If no metadata is found, behavior remains the old masked crop.

Update `app/ocr_model_manager.py::ManuscriptAwareOcrModelManager.recognize_page(...)` so it passes the page's sibling line segmentation metadata path into `process_page_xml(...)` when the file exists. Keep a fallback that works with the current positional call so tests and scripts that do not know about strategy metadata still run.

Update active-learning revision snapshots in `app/ocr_active_learning_runtime.py`. `_snapshot_page_revision(...)` should copy the line segmentation metadata sidecar for the page when it exists. The snapshot directory should contain:

    page-xml-format/<page>.xml
    page-xml-format/<page>_line_segmentation_metadata.json
    images_resized/<page>.jpg

Update `_prepare_revision_pages(...)` or `prepare_page_datasets(...)` so revision preparation can pass each page's metadata sidecar into `prepare_page_line_dataset(...)`. Keep `geometry_source="pagexml_coords"` for production snapshots; the final PAGE XML already has `Coords`, so active-learning training should not require heatmaps or regenerate geometry during normal production use. It should only use metadata to decide crop shape.

Add tests before changing behavior where practical. The tests should prove that current legacy production behavior remains the same through the new abstraction. Use synthetic images and PAGE XML where possible so the tests are fast and deterministic.

Update documentation after the code refactor. This is not optional because future agents and operators use the docs as the source of truth for what production adoption means.

At minimum, update these repository-root docs:

    EVAL.md
    README.md
    VISION.md
    AGENTS.md

`EVAL.md` should state that production adoption means both future PAGE `Coords` generation and strategy-aware OCR crop preparation. It should remove or revise any wording that implies production must be changed to start `Coords` generation from `Baseline`, because production already does that during layout saves. The remaining production gap is OCR crop preparation.

`README.md` should briefly explain that the app has a production strategy pin and that production OCR crops now go through the shared strategy-aware crop layer. Keep this operator-facing and concise.

`VISION.md` should describe the intended long-term loop: research harness strategies are evaluated and promoted as benchmarks, then an explicit production adoption can bring both geometry generation and OCR crop behavior into the app.

`AGENTS.md` should be updated so future agents know that a strategy is not only PAGE `Coords` generation. For this repository, a production text-line segmentation strategy also includes the way PAGE lines are converted into OCR-ready crops.

Also update these text-line-specific docs:

    docs/pipeline-improvement/text-line-segmentation/strategy-promotion-workflow.md
    docs/pipeline-improvement/text-line-segmentation/local-tangent-band-v1-architecture.md

`strategy-promotion-workflow.md` should distinguish research promotion from production adoption and should say that production adoption is only smooth after production code consumes the shared crop layer.

`local-tangent-band-v1-architecture.md` should keep the rule that PAGE `Coords` remain page-space polygons, while OCR unwrapping remains a derived crop representation. It should mention which production callers now use that crop representation after this plan is implemented.

If `docs/pipeline-improvement/text-line-segmentation/legacy-axis-bound-v1-architecture.md` exists by the time this plan is implemented, update it to state that legacy production crop behavior is the masked axis-aligned crop and that this remains the fallback behavior for missing or non-local-tangent metadata.

Review current proposed ExecPlans that mention production crop behavior. Do not rewrite completed historical evidence casually, but add a short note to any active proposed plan if its guidance would mislead a future implementer after this refactor.

## Implementation Phases

Implement this as a staged refactor. Do not mix the mechanical crop extraction with production adoption of new local-tangent behavior.

Phase 1: Extract legacy crop behavior.

Move the existing masked PAGE `Coords` crop into the shared module and point research dataset preparation at that function. The output for `legacy_axis_bound_v1` should remain the same. This phase should not introduce unwrapping into production callers.

Phase 2: Add metadata loading and crop selection.

Add the sidecar metadata loader and `crop_line_record_for_ocr(...)`. The cropper should support local-tangent unwrapping, but production callers do not need to use it yet. Tests should cover missing, malformed, legacy, local-tangent, and delegated metadata.

Phase 3: Refactor production app line-image export.

Change `_write_app_line_images_from_pagexml(...)` to call the shared cropper and pass the metadata written by the just-run production strategy. With current production config, this should still produce masked crops.

Phase 4: Refactor production local OCR inference.

Remove the duplicate direct `Coords` masking from local OCR inference and route it through the shared cropper. Keep automatic fallback to sibling metadata discovery and masked crop behavior when metadata is absent.

Phase 5: Refactor active-learning revision snapshots and preparation.

Copy the strategy metadata sidecar into revision snapshots when present. Prepare revision training crops from saved PAGE `Coords` plus metadata. Do not require heatmaps for normal production revision training.

Phase 6: Validate production adoption readiness.

Only after the previous phases are passing should `local_tangent_band_v1` be considered for production adoption. At that point, adoption should be an explicit config/script action and should be validated as a production behavior change, not as part of this refactor.

Each phase should preserve the invariant that missing metadata behaves like the old production app: saved PAGE `Coords` are cropped with the masked polygon crop.

## Concrete Steps

Work from the repository root:

    cd C:\Users\intro\Documents\Projects\gnn-synthetic-layout-historical

Add the shared crop module:

    app/recognition/line_segmentation/ocr_crops.py

The module should expose stable functions with names close to:

    masked_line_crop(processing_image, polygon_points) -> numpy.ndarray
    load_line_segmentation_metadata_by_numeric_id(metadata_path) -> dict[int, dict]
    crop_line_record_for_ocr(processing_image, record, strategy_name=None, strategy_line_metadata=None, crop_config=None) -> OcrCropResult

`OcrCropResult` can be a small dataclass with fields:

    image
    metadata

The metadata should be small but explicit. Include at least:

    crop_model
    crop_source
    line_segmentation_strategy_name
    line_numeric_id
    used_unwrap
    fallback_reason

For the legacy fallback, a reasonable metadata shape is:

    crop_model = "axis_aligned_masked_crop"
    crop_source = "pagexml_coords"
    used_unwrap = false

For local-tangent unwrapping, preserve the existing unwrap metadata and add enough outer metadata to make the crop decision auditable.

Keep the first implementation narrow. It only needs to preserve the current masked crop and delegate to the existing `unwrap_line_crop_for_ocr(...)` when metadata calls for local-tangent unwrapping.

Refactor these files to use the new module:

    app/recognition/pagexml_line_dataset.py
    app/gnn_inference.py
    app/recognition/recognize_manuscript_text_v2_pretrained.py
    app/ocr_model_manager.py
    app/ocr_active_learning_runtime.py
    app/recognition/active_learning.py

The exact edit in `app/recognition/active_learning.py` should be small. Add optional metadata path plumbing to `prepare_page_datasets(...)` only if needed. A reasonable shape is an optional `line_segmentation_metadata_dir` or a callable that resolves one metadata path per page. Do not require heatmaps for production revision snapshots.

Add or update tests:

    app/tests/test_strategy_aware_ocr_crops_unit.py
    app/tests/test_recognition_active_learning_backend_unit.py
    app/tests/test_recognition_active_learning_unit.py
    app/tests/test_line_segmentation_strategy_unit.py

The new crop unit tests should cover:

- missing metadata uses the masked crop
- malformed metadata uses the masked crop and records/logs a fallback
- `legacy_axis_bound_v1` metadata uses the masked crop
- `local_tangent_band_v1` metadata with `crop_model="local_tangent_band"` uses unwrapping
- `local_tangent_band_v1` metadata with `crop_model="legacy_axis_bound_delegate"` uses the masked crop
- crop preparation does not mutate PAGE XML or write unwrapped geometry back into PAGE `Coords`
- crop preparation from saved production revisions does not require heatmaps

The production tests should cover:

- app line-image export still writes the same path structure for legacy pages
- app line-image export for legacy pages matches the old masked crop output on a synthetic fixture
- local OCR crop extraction can discover sibling metadata but falls back safely if it is absent
- active-learning snapshots copy the metadata sidecar when present
- active-learning revision preparation uses saved `Coords` plus metadata, not heatmap regeneration, for production snapshots

When implementation is complete, run targeted tests:

    $env:CONDA_NO_PLUGINS='true'
    conda run -n gnn_layout python -m unittest app.tests.test_strategy_aware_ocr_crops_unit -v
    conda run -n gnn_layout python -m unittest app.tests.test_line_segmentation_strategy_unit -v
    conda run -n gnn_layout python -m unittest app.tests.test_recognition_active_learning_unit -v
    conda run -n gnn_layout python -m unittest app.tests.test_recognition_active_learning_backend_unit -v

Then run the hybrid OCR unit gate:

    $env:CONDA_NO_PLUGINS='true'
    conda run -n gnn_layout python -m unittest app.tests.test_recognition_finetuning_precommit_unit -v

If those pass and a proposed research strategy is configured, run the fast full-pipeline gate:

    $env:CONDA_NO_PLUGINS='true'
    conda run -n gnn_layout python -m unittest discover -s app/tests -p "test_ci_e2e.py" -v

If the change is a research strategy comparison or a production adoption that explicitly needs fresh verifier evidence with both benchmark and proposed roles configured, run the two slow OCR ablation gates as well:

    $env:CONDA_NO_PLUGINS='true'
    conda run -n gnn_layout python -m unittest app.tests.test_recognition_finetuning_precommit_e2e -v
    conda run -n gnn_layout python -m unittest app.tests.test_circular_recognition_finetuning_precommit_e2e -v

## Validation and Acceptance

Acceptance requires behavior that a human can observe.

Acceptance for this refactor is not the same as acceptance for `local_tangent_band_v1` production adoption. This refactor is accepted when production can route OCR crop preparation through the shared cropper while preserving current legacy behavior. A later production adoption is accepted only after proving the adopted strategy's OCR crops and active-learning behavior are acceptable in production-like workflows.

At the time of the initial refactor, `production_strategy_name` remained `legacy_axis_bound_v1`. After the 2026-05-30 adoption, `production_strategy_name` is `local_polygons_stable_unwrap_v1`; future GUI layout saves should still produce:

    layout_analysis_output/page-xml-format/<page>.xml
    layout_analysis_output/page-xml-format/<page>_line_segmentation_metadata.json
    layout_analysis_output/image-format/<page>/<textbox_label>/line_<id>.jpg

Legacy lines and pages without usable metadata should still be axis-aligned masked crops. New `local_polygons_stable_unwrap_v1` lines with valid metadata should use stable local-polygon unwrapping. Unit tests should demonstrate both the masked fallback and strategy-aware unwrap paths.

Local OCR inference should still work when no metadata sidecar exists. A test should call the extraction path on a synthetic PAGE XML with `Coords` and confirm it returns at least one crop using the masked-crop fallback.

When a synthetic metadata sidecar says a line was produced by `local_tangent_band_v1` with `crop_model="local_tangent_band"`, the shared cropper should produce an unwrapped crop and metadata containing `unwrap_strategy="baseline_local_tangent"`. This proves production can honor the research strategy's crop behavior after production adoption, without changing current legacy behavior.

Active-learning revision snapshots should include the metadata sidecar when present. A unit test should create a fake manuscript page with final XML, image, and metadata, call the post-save or snapshot helper, and assert that the revision snapshot contains `<page>_line_segmentation_metadata.json`.

No test should require existing manuscripts to be migrated. Missing metadata must remain a valid input and must choose the masked crop.

## Idempotence and Recovery

All changes are additive refactors around existing behavior. If a page lacks metadata, production should behave as it did before: read saved `Coords` and produce masked crops.

The refactor must not rewrite existing PAGE XML except through existing save and OCR prediction flows. It must not regenerate existing OCR line images unless a user saves a page or explicitly runs an existing generation command.

If a local-tangent metadata file is malformed, log a warning and fall back to masked crops for that page. Do not crash GUI OCR or active-learning training for a metadata parsing failure.

If tests reveal output changes for `legacy_axis_bound_v1`, prefer fixing the shared cropper to match the old masked crop rather than changing test expectations. This plan's purpose is to make behavior modular while preserving current production behavior.

## Artifacts and Notes

Relevant current code excerpts:

    app/gnn_inference.py::generate_xml_and_images_for_page(...)
        create_page_xml(..., baseline_xml_path, polygons_data={})
        apply_text_line_segmentation_strategy(..., source_pagexml_path=baseline_xml_path, output_pagexml_path=final_xml_path)
        _write_app_line_images_from_pagexml(final_xml_path, ...)

    app/recognition/pagexml_line_dataset.py::prepare_page_line_dataset(...)
        calls crop_line_record_for_ocr(...), which unwraps only when metadata says crop_model == "local_tangent_band"

    app/recognition/recognize_manuscript_text_v2_pretrained.py::process_page_xml(...)
        uses extract_ocr_line_crops_from_page_xml(...) and no longer reimplements TextLine/Coords masking internally.

Expected short test transcript after implementation:

    python -m unittest app.tests.test_strategy_aware_ocr_crops_unit -v
    test_legacy_metadata_uses_masked_crop ... ok
    test_local_tangent_metadata_unwraps ... ok
    test_missing_metadata_falls_back_to_masked_crop ... ok

## Interfaces and Dependencies

Use only existing dependencies already present in the repository: `cv2`, `numpy`, `PIL`, and the existing PAGE XML helpers. Do not add a new package.

The shared crop interface should be independent of Flask. It should live under `app/recognition/line_segmentation/` because it is part of the line segmentation strategy lifecycle, but it should import only pure recognition helpers and not import `app.py`.

Required new or stabilized interfaces:

    app.recognition.line_segmentation.ocr_crops.masked_line_crop(processing_image, polygon_points)

This function preserves the old crop behavior: compute `cv2.boundingRect(...)`, create a median-color background, and copy pixels inside the polygon mask.

    app.recognition.line_segmentation.ocr_crops.crop_line_record_for_ocr(processing_image, record, strategy_name=None, strategy_line_metadata=None, crop_config=None)

This function chooses unwrapping only for `local_tangent_band_v1` lines whose metadata says `crop_model` is `local_tangent_band`. Every other case uses `masked_line_crop(...)`.

    app.recognition.line_segmentation.ocr_crops.load_line_segmentation_metadata_by_numeric_id(metadata_path)

This function reads the strategy metadata JSON sidecar produced by `apply_text_line_segmentation_strategy(...)` and returns a dictionary keyed by integer `line_numeric_id`. It returns an empty dictionary for missing paths.

Optional but useful helper:

    app.recognition.line_segmentation.ocr_crops.default_line_segmentation_metadata_path(pagexml_path)

This function maps `page-xml-format/<page>.xml` to `page-xml-format/<page>_line_segmentation_metadata.json`.

Do not make the shared crop module responsible for recognizing text, mutating PAGE XML, queueing active-learning jobs, or generating strategy `Coords`. Those concerns stay in their current modules.

## Change Note

Initial plan created on 2026-05-15 after clarifying that production already regenerates PAGE `Coords` from `Baseline` during layout save, but production OCR and active-learning data preparation do not yet use the same strategy-aware crop/unwrapping path as the research harness. The plan intentionally preserves current `legacy_axis_bound_v1` behavior while making future production adoption smoother.

Plan update on 2026-05-15: Expanded the documentation requirements so implementation must update all relevant repository guidance, including `EVAL.md`, `README.md`, `VISION.md`, `AGENTS.md`, and text-line strategy docs, not only the evaluation and promotion workflow documents.
