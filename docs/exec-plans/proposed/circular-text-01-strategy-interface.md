# Extract A Shared Text-Line Segmentation Strategy Interface

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This document is maintained in accordance with `PLANS.md` from the repository root.

This is the first plan in the circular text support series. It uses `docs/exec-plans/proposed/circular-text-support.md` as the research source and must not overwrite that file.

## Purpose / Big Picture

After this change, the repository will have one shared implementation path for converting PAGE `Baseline` polylines plus checked-in or generated heatmaps into PAGE `Coords` polygons. A user or researcher will be able to run the app, the pretrained full-pipeline gate, and the OCR fine-tuning gates with the same named text-line segmentation strategy instead of three near-copies of the old crop code. This makes later ablation work meaningful: when `legacy_axis_bound_v1` and `local_tangent_band_v1` are compared, the only intentional difference is the named strategy, not different plumbing around it.

This plan only extracts the strategy interface and ports the current behavior into the benchmark strategy named `legacy_axis_bound_v1`. It does not implement the new curved/circular strategy and it does not overhaul every gate yet; those are covered by `docs/exec-plans/proposed/circular-text-02-ablation-gates.md` and `docs/exec-plans/proposed/circular-text-03-generalized-cropping-unwrapping.md`.

## Progress

- [x] (2026-05-09 22:31 IST) Read `PLANS.md`, the research source `docs/exec-plans/proposed/circular-text-support.md`, and the current OCR crop and gate code.
- [x] (2026-05-09 22:31 IST) Identified the current benchmark path: `app/recognition/pagexml_line_dataset.py` rebuilds polygons from PAGE `Baseline` plus heatmap through `segmentLinesFromPointClusters(...)`, while `app/gnn_inference.py` also calls `segmentLinesFromPointClusters(...)` during app saves.
- [ ] Create the shared strategy types, registry, and deterministic apply function under `app/recognition/line_segmentation/`.
- [ ] Move the current baseline-plus-heatmap polygon generation into `legacy_axis_bound_v1` without changing its geometry output.
- [ ] Rework app save OCR preparation, PAGE baseline OCR preparation, and existing gates to call the shared strategy layer.
- [ ] Add focused unit tests that prove the legacy strategy preserves current behavior and that all call sites select strategy names through the registry.
- [ ] Run the validation commands in this plan and record results here before marking this plan complete.

## Surprises & Discoveries

- Observation: the current code already has a baseline-derived OCR geometry path, but it is embedded inside OCR dataset preparation instead of being a shared strategy.
  Evidence: `app/recognition/pagexml_line_dataset.py` defines `GEOMETRY_SOURCE_BASELINE_HEATMAP`, `_build_baseline_component_nodes(...)`, `_generate_polygons_from_baselines(...)`, and `prepare_page_line_dataset(...)`.

- Observation: the app and the test harness do not call exactly the same segmentation helper module by default.
  Evidence: `app/gnn_inference.py` imports `segmentLinesFromPointClusters` from `app/segment_from_point_clusters.py`, while `app/recognition/pagexml_line_dataset.py` inserts `src/gnn_inference` on `sys.path` and imports `segment_from_point_clusters` from there. `app/tests/test_ci_e2e.py` then monkeypatches `pagexml_line_dataset.segmentLinesFromPointClusters` back to the `src` implementation.

- Observation: generated logs under `app/tests/logs/` already mention names such as `legacy_axis_bound_v1` and `local_tangent_band_v1`, but there is no checked-in source implementation for those names yet.
  Evidence: `rg "local_tangent|legacy_axis" app scripts docs` finds only generated log files and the research document, not production Python modules.

## Decision Log

- Decision: call the current benchmark strategy `legacy_axis_bound_v1`.
  Rationale: the name is explicit about the old global-axis and bounding-box style of the algorithm, and it gives future plans a stable baseline name to compare against.
  Date/Author: 2026-05-09 / Codex

- Decision: make the strategy API produce copied PAGE-XML with updated page-space `TextLine/Coords`, not OCR-ready crop images.
  Rationale: PAGE `Coords` are manuscript page geometry. OCR crop orientation and unwrapping are a separate downstream representation and must not be written back as PAGE geometry.
  Date/Author: 2026-05-09 / Codex

- Decision: keep `PreparedPageDataset` compatibility in `app/recognition/pagexml_line_dataset.py`, but have it consume strategy output instead of owning the strategy implementation.
  Rationale: OCR fine-tuning code already expects `PreparedPageDataset`. Moving the strategy boundary below that data class preserves existing OCR code while making geometry generation reusable.
  Date/Author: 2026-05-09 / Codex

## Outcomes & Retrospective

Not yet implemented. At completion, summarize whether the legacy strategy output matched the previous `baseline_heatmap` path, which call sites were migrated, and whether any path-specific behavior had to remain temporarily.

## Context and Orientation

This repository has two connected products. The `src/` tree contains the graph neural network text-line segmentation core. The `app/` tree contains the Flask annotation and OCR application plus the test harnesses used as pre-commit gates.

Several terms are used precisely in this plan.

PAGE-XML is the XML format this application writes for manuscript layout and OCR text. A `TextLine` element can contain a `Baseline` and `Coords`. A `Baseline` is a polyline of points such as `x1,y1 x2,y2`; it describes the center or writing path of one text line. `Coords` is a polygon in page-image coordinates; it describes the region occupied by the text line on the manuscript page.

A text-line segmentation strategy is a deterministic algorithm that takes a page image, a heatmap, PAGE `Baseline` and text content, and a strategy config, then writes a copied PAGE-XML file whose `TextLine/Coords` polygons were updated by that strategy. Deterministic means the same inputs and config produce the same output XML and metadata.

The current legacy behavior has two main paths:

`app/gnn_inference.py` is used by the Flask app save route in `app/app.py`. `save_correction(...)` calls `generate_xml_and_images_for_page(...)`, which saves graph labels, runs `segmentLinesFromPointClusters(...)`, then calls `create_page_xml(...)`. This writes PAGE-XML plus app-style line images under `layout_analysis_output/page-xml-format/` and `layout_analysis_output/image-format/`.

`app/recognition/pagexml_line_dataset.py` is used by the OCR fine-tuning harness. When `prepare_page_line_dataset(..., geometry_source="baseline_heatmap")` is called, it parses ground-truth PAGE `Baseline` polylines, assigns heatmap connected components to the nearest baseline, calls `segmentLinesFromPointClusters(...)`, and then masks axis-aligned crop rectangles into OCR line images. The geometry guard checks `source_line_coverage >= 0.90` and `heatmap_box_assignment_rate >= 0.90`.

The old helper `segmentLinesFromPointClusters(...)` uses heatmap bounding boxes, global x/y padding, connected-component cleanup, and rectangle bridging to return text-line polygons. This works best for horizontal text and is the benchmark to preserve as `legacy_axis_bound_v1`.

## Plan of Work

Create a new package `app/recognition/line_segmentation/`. The package should be small and explicit:

- `app/recognition/line_segmentation/__init__.py` exports the public registry and data classes.
- `app/recognition/line_segmentation/types.py` defines the request, result, config, and metadata data classes.
- `app/recognition/line_segmentation/pagexml.py` contains PAGE-XML parsing and writing helpers for `Baseline`, `Coords`, `TextEquiv`, page size, and line ids.
- `app/recognition/line_segmentation/legacy_axis_bound.py` implements `legacy_axis_bound_v1`.
- `app/recognition/line_segmentation/registry.py` maps stable strategy names to implementations and validates config.

The public call should be deterministic and shaped like this:

    apply_text_line_segmentation_strategy(
        page_image_path: Path,
        heatmap_path: Path,
        source_pagexml_path: Path,
        output_pagexml_path: Path,
        strategy_name: str,
        strategy_config: Mapping[str, object] | None = None,
        metadata_path: Path | None = None,
    ) -> TextLineSegmentationResult

`TextLineSegmentationResult` should include:

    strategy_name: str
    source_pagexml_path: str
    output_pagexml_path: str
    page_image_path: str
    heatmap_path: str
    metadata_path: str | None
    line_count: int
    prepared_line_count: int
    line_metadata: list[dict]
    geometry_summary: dict

`line_metadata` must include at least `line_id`, `line_custom`, `line_numeric_id`, `baseline_points`, `coords_points`, and any strategy-specific source information. `geometry_summary` must preserve the existing guard fields when the legacy strategy is used: `source_line_coverage`, `heatmap_box_count`, `assigned_box_count`, and `heatmap_box_assignment_rate`.

Port the existing `baseline_heatmap` implementation from `app/recognition/pagexml_line_dataset.py` into `legacy_axis_bound_v1`. Keep the default settings:

    BINARIZE_THRESHOLD=0.5098
    BBOX_PAD_V=0.7
    BBOX_PAD_H=0.5
    CC_SIZE_THRESHOLD_RATIO=0.4

The port must not read PAGE `Coords` when generating polygons. It may read `TextEquiv/Unicode` to skip empty text lines in OCR ground-truth preparation, matching the existing behavior. It must continue to support manual node-add history imperfectly but pragmatically by assigning heatmap boxes to the nearest PAGE baseline and measuring assignment coverage, just as the current guard does.

Rework `app/recognition/pagexml_line_dataset.py` so crop preparation has two separate steps:

1. If a strategy is requested, call `apply_text_line_segmentation_strategy(...)` to write an intermediate copied PAGE-XML under the page output root.
2. Load line records from that copied PAGE-XML and create `PreparedPageDataset` crops from its `Coords`.

Keep backward compatibility for existing callers. `geometry_source="pagexml_coords"` should still read the input XML `Coords` directly. `geometry_source="baseline_heatmap"` should become an alias for `strategy_name="legacy_axis_bound_v1"` until later plans remove or rename the old option. New code should prefer an explicit `line_segmentation_strategy_name` or `strategy_name` field.

Rework `app/gnn_inference.py` with the smallest safe migration. The app currently creates baselines and polygons in one pass. Split that into two conceptual steps:

1. Write a baseline PAGE-XML file from graph connected components, text-region labels, `Baseline`, and `TextEquiv`, but without relying on the old polygon helper for final `Coords`.
2. Call the shared strategy to create the final PAGE-XML with `Coords`, then use the OCR crop preparation code or a shared crop writer to write app-style line images.

If splitting `create_page_xml(...)` completely is too large for this plan, add an internal baseline-only writer and leave `create_page_xml(...)` as a compatibility wrapper that calls the new writer plus `legacy_axis_bound_v1`.

The OCR crop preparation must separately consume copied PAGE-XML plus page image plus `Coords` plus `Baseline` plus unwrap config and return `PreparedPageDataset`-compatible images and manifests. For this plan, the unwrap config can keep the current axis-aligned masked crop behavior. The new generalized unwrapping is part of plan 03.

Do not write artifacts outside the repository. Intermediate XML, manifests, and run evidence should stay under the caller's existing output root, usually `app/tests/logs/` for gates or `app/input_manuscripts/...` for app runs.

## Concrete Steps

Work from the repository root:

    cd c:\Users\intro\OneDrive\Documents\MEGA\CAI-FLAME\gnn-synthetic-layout-historical

Create `app/recognition/line_segmentation/` and add the files described above. Keep the code importable both from the app root and from the unittest harness, because existing tests add `app/` and the repository root to `sys.path`.

Move or wrap code from these existing functions:

    app/recognition/pagexml_line_dataset.py::_load_pagexml_baseline_records
    app/recognition/pagexml_line_dataset.py::_count_text_lines_with_text_and_baseline
    app/recognition/pagexml_line_dataset.py::_build_baseline_component_nodes
    app/recognition/pagexml_line_dataset.py::_generate_polygons_from_baselines
    app/recognition/pagexml_line_dataset.py::load_pagexml_lines
    app/recognition/pagexml_line_dataset.py::_masked_line_crop

Leave thin compatibility wrappers in `pagexml_line_dataset.py` if existing tests import private helpers. Do not delete public names during this plan unless no checked-in code imports them.

Add or update unit tests in:

    app/tests/test_line_segmentation_strategy_unit.py
    app/tests/test_recognition_finetuning_precommit_unit.py
    app/tests/test_recognition_active_learning_backend_unit.py

The focused tests should prove:

- `get_text_line_segmentation_strategy("legacy_axis_bound_v1")` returns the legacy implementation.
- requesting an unknown strategy raises a clear `ValueError` or `KeyError` that includes the unknown name.
- a synthetic horizontal page with one PAGE `Baseline`, one heatmap component, and no PAGE `Coords` produces a copied PAGE-XML with one `TextLine/Coords`.
- the copied PAGE-XML preserves `Baseline`, `TextEquiv`, `TextRegion`, page attributes, and line ids.
- `geometry_source="baseline_heatmap"` and `strategy_name="legacy_axis_bound_v1"` produce equivalent line counts and guard metrics for the same synthetic page.
- the app save path can be configured to use the registry instead of importing `segmentLinesFromPointClusters(...)` directly.

When editing `app/tests/test_ci_e2e.py`, remove the monkeypatch that forces `pagexml_line_dataset.segmentLinesFromPointClusters` to the `src` implementation only after the shared strategy imports one canonical implementation itself. If keeping the monkeypatch is temporarily required, document why in this plan's `Surprises & Discoveries`.

## Validation and Acceptance

Use the `gnn_layout` conda environment. From the repository root, run:

    $env:CONDA_NO_PLUGINS='true'; conda run -n gnn_layout python -m unittest app.tests.test_line_segmentation_strategy_unit -v

Expected result: all new strategy unit tests pass. The synthetic test should fail before this plan because there is no registry or named strategy, and pass after implementation.

Run the existing OCR pre-commit unit tests:

    $env:CONDA_NO_PLUGINS='true'; conda run -n gnn_layout python -m unittest app.tests.test_recognition_finetuning_precommit_unit -v

Expected result: tests still confirm the hybrid recipe and geometry guard. If assertions change, they should change from `line_geometry_source="baseline_heatmap"` to the explicit `legacy_axis_bound_v1` strategy while preserving the same default thresholds.

Run the pretrained full-pipeline gate:

    $env:CONDA_NO_PLUGINS='true'; conda run -n gnn_layout python -m unittest discover -s app/tests -p "test_ci_e2e.py" -v

Expected result: the gate still uploads `app/tests/eval_dataset/images/`, produces PAGE-XML and OCR text, writes `app/tests/logs/ci_eval_results_latest.*`, and stays within the existing thresholds from `app/tests/precommit_gate_config.py`.

Run the OCR fine-tuning surrogate gate:

    $env:CONDA_NO_PLUGINS='true'; conda run -n gnn_layout python -m unittest app.tests.test_recognition_finetuning_precommit_e2e -v

Expected result: the gate still writes `app/tests/logs/recognition_finetune_precommit_latest.*`, reports `study_mode="recognition_precommit_gate"`, and passes the checked-in OCR thresholds.

Run the two-phase launcher:

    $env:CONDA_NO_PLUGINS='true'; conda run -n gnn_layout python scripts/run_precommit_eval.py

Expected result: the launcher finds `gnn_layout`, runs the full-pipeline phase and the OCR fine-tuning phase, and prints artifact paths under `app/tests/logs/`.

For long OCR runs on Windows, if `conda run` fails after writing artifacts because of console encoding, retry with the direct interpreter:

    C:\Users\intro\miniconda3\envs\gnn_layout\python.exe -m unittest app.tests.test_recognition_finetuning_precommit_e2e -v
    C:\Users\intro\miniconda3\envs\gnn_layout\python.exe scripts/run_precommit_eval.py

The acceptance criterion is observable: all existing behavior stays green while both app and test code can name `legacy_axis_bound_v1` through one shared registry.

## Idempotence and Recovery

All code changes in this plan should be additive or thin-wrapper refactors. Re-running strategy application on the same output directory may replace the copied PAGE-XML, metadata JSON, and prepared crops for that page. That is acceptable and should be deterministic.

Do not delete `docs/exec-plans/proposed/circular-text-support.md`. It is the research source and has existing user changes. Do not clean generated logs unless a test explicitly owns its temporary run directory. If a test creates a temporary directory under `app/tests/_tmp_*`, its teardown may remove only that directory.

If a refactor breaks the full app path, first switch the app save route back to `legacy_axis_bound_v1` through the compatibility wrapper and keep the new registry in use for tests. Then record the temporary compromise in `Decision Log` and finish the migration in a follow-up edit before closing the plan.

## Artifacts and Notes

Important current source paths:

    app/app.py
    app/gnn_inference.py
    app/segment_from_point_clusters.py
    src/gnn_inference/segment_from_point_clusters.py
    app/recognition/pagexml_line_dataset.py
    app/recognition/active_learning.py
    app/tests/test_ci_e2e.py
    app/tests/test_recognition_finetuning_precommit_unit.py
    app/tests/test_recognition_finetuning_precommit_e2e.py
    scripts/run_precommit_eval.py

Generated logs such as `app/tests/logs/recognition_finetune_proposed_latest.md` are evidence only. Do not treat them as source of truth for available strategies.

## Interfaces and Dependencies

The final strategy API must expose:

    apply_text_line_segmentation_strategy(...)
    get_text_line_segmentation_strategy(strategy_name: str)
    list_text_line_segmentation_strategies() -> tuple[str, ...]

`legacy_axis_bound_v1` must remain available with this config shape:

    {
        "BINARIZE_THRESHOLD": 0.5098,
        "BBOX_PAD_V": 0.7,
        "BBOX_PAD_H": 0.5,
        "CC_SIZE_THRESHOLD_RATIO": 0.4
    }

`PreparedPageDataset` remains the OCR consumer contract. Its manifest should gain strategy metadata without removing current fields:

    geometry_source
    geometry_summary
    line_segmentation_strategy_name
    line_segmentation_metadata_path

Use existing dependencies already present in the codebase: `cv2`, `numpy`, `skimage.io`, `shapely`, and `xml.etree.ElementTree`. Do not add a new geometry or image-processing dependency for this extraction.

## Change Note

Initial split plan created on 2026-05-09. The reason for the split is to make the circular text research implementable in small, verifiable stages while preserving the original research document unchanged.
