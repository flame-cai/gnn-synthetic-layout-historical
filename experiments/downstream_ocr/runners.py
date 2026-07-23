from __future__ import annotations

import csv
import concurrent.futures
import json
import math
import os
import shutil
import sys
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

from dotenv import load_dotenv

from .adapter import AdapterError, VLM_END_TO_END_PROMPT, vlm_json_to_pagexml
from .dataset.pagexml2pagexml_dataset import (
    PageXmlPair,
    TEXTEDIT_AGGREGATE_KEYS,
    evaluate_pagexml_pairs,
    textedit_reproducibility_metadata,
)
from .devanagari_textedit import (
    DEVANAGARI_TEXTEDIT_AGGREGATE_KEYS,
    devanagari_textedit_reproducibility_metadata,
)
from .diagnostics import write_page_diagnostics
from .metrics import (
    FAILURE_STATUSES,
    aggregate_page_records,
    evaluate_page,
)
from .pagexml import (
    empty_page_like,
    extract_structure_line_id,
    local_name,
    load_pagexml,
    qualified,
    tag_namespace,
    write_pagexml,
)
from .reporting import summarize_usage_metadata, usage_metadata_to_dict
from .reproducibility import write_reproducibility_manifest
from .splits import (
    DEFAULT_SPLIT_SEED,
    Fold,
    ManuscriptPaths,
    default_manuscript_paths,
    discover_page_ids,
    load_or_create_folds_json,
)
from .vlm_cache import materialize_cached_fold, validate_vlm_cache
from .vlm_providers import VLM_PROVIDER_SPECS, is_vlm_method


REPO_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = REPO_ROOT / "app"
BASE_OCR_CHECKPOINT = APP_ROOT / "recognition" / "pretrained_model" / "vadakautuhala.pth"
PRETRAINED_GNN_MODEL = APP_ROOT / "pretrained_gnn" / "v2.pt"
PRETRAINED_GNN_CONFIG = APP_ROOT / "pretrained_gnn" / "gnn_preprocessing_v2.yaml"
DEFAULT_GNN_FINETUNING_CONFIG = (
    REPO_ROOT / "experiments" / "downstream_ocr" / "configs" / "gnn_finetuning.yaml"
)
DEFAULT_GEMINI_TIMEOUT_SECONDS = 45.0
DEFAULT_GEMINI_PAGE_WORKERS = 4
DEFAULT_GEMINI_REQUEST_SPACING_SECONDS = 0.25
DEFAULT_GEMINI_MAX_RETRIES = 3
DEFAULT_GEMINI_RETRY_BASE_DELAY_SECONDS = 1.0


def _write_experiment_reproducibility(
    output_root: Path,
    *,
    gnn_finetuning_config: str | Path = DEFAULT_GNN_FINETUNING_CONFIG,
) -> Path:
    gnn_finetuning_config = Path(gnn_finetuning_config)
    return write_reproducibility_manifest(
        output_root,
        repo_root=REPO_ROOT,
        artifact_paths=(
            BASE_OCR_CHECKPOINT,
            PRETRAINED_GNN_MODEL,
            PRETRAINED_GNN_CONFIG,
            gnn_finetuning_config,
            REPO_ROOT / "src" / "configs" / "augment.yaml",
        ),
    )


def _write_report_artifacts(
    output_root: Path,
    *,
    input_usd_per_1m_tokens: float | None = None,
    output_usd_per_1m_tokens: float | None = None,
) -> None:
    from .reporting import write_experiment_report

    write_experiment_report(
        output_root,
        input_usd_per_1m_tokens=input_usd_per_1m_tokens,
        output_usd_per_1m_tokens=output_usd_per_1m_tokens,
    )


def _load_gui_runtime_ocr_recipe():
    _ensure_app_import_path()
    from ocr_active_learning_runtime import _runtime_recipe

    return _runtime_recipe()


def _ocr_recipe_metadata_for_method(method: "MethodSpec") -> dict | None:
    if not method.uses_finetuning:
        return None
    recipe = _load_gui_runtime_ocr_recipe()
    return {
        "source": "app.ocr_active_learning_runtime._runtime_recipe",
        "recipe": recipe.to_dict(),
        "sibling_checkpoint_strategy": recipe.sibling_checkpoint_strategy,
    }


def _gnn_recipe_metadata_for_method(
    method: "MethodSpec",
    gnn_finetuning_config: str | Path = DEFAULT_GNN_FINETUNING_CONFIG,
) -> dict | None:
    if not method.uses_finetuning or method.uses_gt_layout:
        return None
    from .gnn_finetuning import load_gnn_finetuning_recipe

    gnn_finetuning_config = Path(gnn_finetuning_config)
    recipe = load_gnn_finetuning_recipe(gnn_finetuning_config)
    return {
        "source": str(gnn_finetuning_config.resolve()),
        "recipe": recipe.metadata(),
        "test_layout_checkpoint_condition": "fold_local_finetuned_gnn",
    }


@dataclass(frozen=True)
class MethodSpec:
    method_id: str
    display_name: str
    uses_gt_layout: bool
    uses_finetuning: bool = False
    finetune_page_count: int = 0
    provider_id: str | None = None
    model_id: str | None = None
    provides_layout: bool = True
    page_cer_uses_output_order: bool = False


DISABLED_METHOD_IDS = {
    "gemini_gt_layout": "Gemini + GT Layout is disabled for the current experiment.",
    "vlm_e2e": (
        "The provider-ambiguous vlm_e2e method was removed. Use gemini_e2e, "
        "openai_e2e, claude_e2e, or sarvam_e2e."
    ),
}

METHODS: tuple[MethodSpec, ...] = (
    *(
        MethodSpec(
            spec.method_id,
            spec.display_name,
            uses_gt_layout=False,
            provider_id=spec.provider_id,
            model_id=spec.model_id,
            provides_layout=spec.provides_layout,
            page_cer_uses_output_order=spec.page_cer_uses_output_order,
        )
        for spec in VLM_PROVIDER_SPECS
    ),
    MethodSpec("annotation_tool_e2e", "Annotation tool (End-to-End)", uses_gt_layout=False),
    MethodSpec("annotation_tool_gt_layout", "Annotation tool (End-to-End with Graph Layout Grounding)", uses_gt_layout=True),
    MethodSpec("annotation_tool_pred_layout_ft_1", "Annotation tool (Predicted test layout, 1-page joint GNN+OCR fine-tuning)", uses_gt_layout=False, uses_finetuning=True, finetune_page_count=1),
    MethodSpec("annotation_tool_gt_layout_ft_1", "Annotation tool (GT Layout, 1-page fine-tuning)", uses_gt_layout=True, uses_finetuning=True, finetune_page_count=1),
    MethodSpec("annotation_tool_pred_layout_ft_2", "Annotation tool (Predicted test layout, 2-page joint GNN+OCR fine-tuning)", uses_gt_layout=False, uses_finetuning=True, finetune_page_count=2),
    MethodSpec("annotation_tool_gt_layout_ft_2", "Annotation tool (GT Layout, 2-page fine-tuning)", uses_gt_layout=True, uses_finetuning=True, finetune_page_count=2),
    MethodSpec("annotation_tool_pred_layout_ft_3", "Annotation tool (Predicted test layout, 3-page joint GNN+OCR fine-tuning)", uses_gt_layout=False, uses_finetuning=True, finetune_page_count=3),
    MethodSpec("annotation_tool_gt_layout_ft_3", "Annotation tool (GT Layout, 3-page fine-tuning)", uses_gt_layout=True, uses_finetuning=True, finetune_page_count=3),
)


def _ensure_app_import_path() -> None:
    for path in (APP_ROOT, REPO_ROOT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def method_by_id(method_id: str) -> MethodSpec:
    if method_id in DISABLED_METHOD_IDS:
        raise ValueError(DISABLED_METHOD_IDS[method_id])
    for method in METHODS:
        if method.method_id == method_id:
            return method
    raise KeyError(f"Unknown method id: {method_id}")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _find_page_image(page_id: str, directories: Iterable[Path]) -> Path:
    extensions = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG", ".TIF", ".TIFF", ".jp2")
    for directory in directories:
        for extension in extensions:
            candidate = Path(directory) / f"{page_id}{extension}"
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f"No page image found for {page_id}.")


def _copy_gt_subset(pagexml_dir: Path, page_ids: Iterable[str], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    for page_id in page_ids:
        shutil.copy2(pagexml_dir / f"{page_id}.xml", output_dir / f"{page_id}.xml")
    return output_dir


def _write_placeholder_text_pagexml(source_xml_path: Path, target_xml_path: Path) -> None:
    tree = ET.parse(source_xml_path)
    root = tree.getroot()
    text_line_count = 0
    for textline in root.iter():
        if local_name(textline.tag) != "TextLine":
            continue
        namespace = tag_namespace(textline.tag)
        for child in list(textline):
            if local_name(child.tag) == "TextEquiv":
                textline.remove(child)
        text_equiv = ET.SubElement(textline, qualified("TextEquiv", namespace))
        unicode_elem = ET.SubElement(text_equiv, qualified("Unicode", namespace))
        unicode_elem.text = "x"
        text_line_count += 1
    if text_line_count <= 0:
        raise ValueError(f"No TextLine elements found in {source_xml_path}.")
    target_xml_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(target_xml_path, encoding="UTF-8", xml_declaration=True)


def _prepare_gt_layout_pages(
    paths: ManuscriptPaths,
    page_ids: Iterable[str],
    output_root: Path,
):
    _ensure_app_import_path()
    from recognition.line_segmentation.reading_direction import (
        default_reading_direction_metadata_path,
        load_reading_direction_annotations_by_line_id,
    )
    from recognition.line_segmentation.runtime_config import get_strategy_runtime_config
    from recognition.line_segmentation.strategy_config import get_production_strategy_name
    from recognition.pagexml_line_dataset import prepare_page_line_dataset

    if paths.heatmaps_dir is None or not paths.heatmaps_dir.exists():
        raise FileNotFoundError(
            f"Production-style baseline-to-Coords preparation requires heatmaps: {paths.heatmaps_dir}"
        )

    strategy_name = get_production_strategy_name()
    manuscript_overrides = _load_manuscript_line_segmentation_args(paths.root)
    prepared = {}
    for page_id in page_ids:
        xml_path = paths.pagexml_dir / f"{page_id}.xml"
        image_path = _find_page_image(page_id, [paths.images_dir])
        heatmap_path = _find_page_image(page_id, [paths.heatmaps_dir])
        strategy_config = get_strategy_runtime_config(
            strategy_name,
            overrides=manuscript_overrides,
            include_empty_text_lines=True,
        )
        reading_annotations = load_reading_direction_annotations_by_line_id(
            default_reading_direction_metadata_path(xml_path)
        )
        strategy_config["reading_direction_annotations_by_line_id"] = reading_annotations
        prepared[page_id] = prepare_page_line_dataset(
            xml_path,
            image_path,
            output_root / page_id,
            heatmap_path=heatmap_path,
            geometry_source="baseline_heatmap",
            segmentation_args=strategy_config,
            line_segmentation_strategy_name=strategy_name,
        )
    return prepared


def _prepare_generated_layout_page(
    *,
    manuscript_root: Path,
    page_id: str,
    source_xml_dir: Path,
    prepared_root: Path,
):
    _ensure_app_import_path()
    from recognition.pagexml_line_dataset import prepare_page_line_dataset
    from recognition.line_segmentation.ocr_crops import default_line_segmentation_metadata_path

    xml_path = manuscript_root / "layout_analysis_output" / "page-xml-format" / f"{page_id}.xml"
    source_xml_path = source_xml_dir / f"{page_id}.xml"
    _write_placeholder_text_pagexml(xml_path, source_xml_path)
    image_path = _find_page_image(
        page_id,
        [
            manuscript_root / "layout_analysis_output" / "images_resized",
            manuscript_root / "images_resized",
        ],
    )
    metadata_path = default_line_segmentation_metadata_path(xml_path)
    return prepare_page_line_dataset(
        source_xml_path,
        image_path,
        prepared_root / page_id,
        geometry_source="pagexml_coords",
        line_segmentation_metadata_path=metadata_path if metadata_path.exists() else None,
    )


def _copy_raw_manuscript_for_auto_layout(paths: ManuscriptPaths, target_root: Path) -> Path:
    if target_root.exists():
        shutil.rmtree(target_root)
    target_root.mkdir(parents=True, exist_ok=True)
    for name in ("gnn-dataset", "images", "images_resized", "heatmaps"):
        source = paths.root / name
        if source.exists():
            shutil.copytree(source, target_root / name)
    settings_path = paths.root / "processing_settings.json"
    if settings_path.exists():
        shutil.copy2(settings_path, target_root / settings_path.name)
    return target_root


def _load_manuscript_line_segmentation_args(manuscript_root: Path) -> dict:
    settings_path = manuscript_root / "processing_settings.json"
    if not settings_path.exists():
        return {}
    try:
        payload = json.loads(settings_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    raw_args = payload.get("line_segmentation_args", {})
    return dict(raw_args) if isinstance(raw_args, dict) else {}


def _coerce_float_or_none(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _coerce_int_or_none(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _load_layout_effort_by_page(manuscript_root: Path) -> dict[str, dict]:
    effort_path = manuscript_root / "layout_analysis_output" / "layout_effort.json"
    if not effort_path.exists():
        return {}
    try:
        payload = json.loads(effort_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    pages = payload.get("pages") if isinstance(payload, dict) else None
    if not isinstance(pages, dict):
        return {}

    effort_by_page: dict[str, dict] = {}
    for page_key, page_payload in pages.items():
        if not isinstance(page_payload, dict):
            continue
        page_id = str(page_payload.get("page_id") or page_key)
        totals = page_payload.get("totals")
        if not isinstance(totals, dict):
            totals = page_payload
        revision_count = _coerce_int_or_none(page_payload.get("revision_count"))
        revisions = page_payload.get("layout_revisions")
        if revision_count is None and isinstance(revisions, list):
            revision_count = len(revisions)
        effort_by_page[page_id] = {
            "layout_effort_available": True,
            "layout_effort_edit_count": _coerce_int_or_none(totals.get("edit_count")),
            "layout_effort_active_edit_time_seconds": _coerce_float_or_none(
                totals.get("active_edit_time_seconds")
            ),
            "layout_effort_revision_count": revision_count,
            "layout_effort_source_path": str(effort_path),
        }
    return effort_by_page


def _layout_effort_fields(effort_by_page: dict[str, dict], page_id: str) -> dict:
    fields = {
        "layout_effort_available": False,
        "layout_effort_edit_count": None,
        "layout_effort_active_edit_time_seconds": None,
        "layout_effort_revision_count": None,
        "layout_effort_source_path": "",
    }
    fields.update(effort_by_page.get(page_id) or {})
    return fields


def _prepare_annotation_tool_predicted_layout_pages(
    *,
    paths: ManuscriptPaths,
    fold: Fold,
    preparation_dir: Path,
    gnn_model_path: Path = PRETRAINED_GNN_MODEL,
    gnn_config_path: Path = PRETRAINED_GNN_CONFIG,
) -> dict:
    _ensure_app_import_path()
    from gnn_inference import generate_xml_and_images_for_page, run_gnn_prediction_for_page

    manuscript_root = _copy_raw_manuscript_for_auto_layout(paths, preparation_dir / "manuscript")
    line_segmentation_args = _load_manuscript_line_segmentation_args(manuscript_root)

    for page_id in fold.test_page_ids:
        graph = run_gnn_prediction_for_page(
            str(manuscript_root),
            page_id,
            str(gnn_model_path),
            str(gnn_config_path),
        )
        node_count = len(graph.get("nodes", []))
        textbox_labels = graph.get("textbox_labels") or [0] * node_count
        generate_xml_and_images_for_page(
            str(manuscript_root),
            page_id,
            graph.get("textline_labels", [-1] * node_count),
            graph.get("edges", []),
            line_segmentation_args,
            textbox_labels=textbox_labels,
            nodes=graph.get("nodes", []),
            text_content={},
            reading_direction_annotations=[],
        )

    prepared_pages = {}
    for page_id in fold.test_page_ids:
        prepared_pages[page_id] = _prepare_generated_layout_page(
            manuscript_root=manuscript_root,
            page_id=page_id,
            source_xml_dir=preparation_dir / "ocr_source_page_xml",
            prepared_root=preparation_dir / "ocr_prepared_pages",
        )
    return prepared_pages


def run_annotation_tool_auto_layout(
    *,
    paths: ManuscriptPaths,
    fold: Fold,
    method: MethodSpec,
    run_dir: Path,
    prepared_pages: dict | None = None,
) -> Path:
    _ensure_app_import_path()
    from recognition.active_learning import generate_prediction_pagexmls

    if prepared_pages is None:
        prepared_pages = _prepare_annotation_tool_predicted_layout_pages(
            paths=paths,
            fold=fold,
            preparation_dir=run_dir,
        )

    prediction = generate_prediction_pagexmls(
        BASE_OCR_CHECKPOINT,
        prepared_pages,
        run_dir / "prediction_page_xml",
        width_policy="batch_max_pad",
    )
    return Path(prediction.prediction_folder)


def run_local_ocr_with_gt_layout(
    *,
    paths: ManuscriptPaths,
    fold: Fold,
    method: MethodSpec,
    run_dir: Path,
) -> Path:
    _ensure_app_import_path()
    from recognition.active_learning import (
        fine_tune_checkpoint_on_pages,
        generate_prediction_pagexmls,
    )

    all_needed_pages = (
        tuple(dict.fromkeys((*fold.train_page_ids, *fold.test_page_ids)))
        if method.uses_finetuning
        else tuple(fold.test_page_ids)
    )
    prepared_pages = _prepare_gt_layout_pages(paths, all_needed_pages, run_dir / "prepared_pages")
    checkpoint = BASE_OCR_CHECKPOINT
    inference_width_policy = "batch_max_pad"
    if method.uses_finetuning:
        selected_train = fold.train_page_ids[: method.finetune_page_count]
        recipe = _load_gui_runtime_ocr_recipe()
        inference_width_policy = recipe.width_policy
        _write_json(
            run_dir / "ocr_active_learning_recipe.json",
            {
                "source": "app.ocr_active_learning_runtime._runtime_recipe",
                "recipe": recipe.to_dict(),
                "sibling_checkpoint_strategy": recipe.sibling_checkpoint_strategy,
            },
        )
        current_checkpoint = checkpoint
        history_pages = []
        for step_index, page_id in enumerate(selected_train, start=1):
            result = fine_tune_checkpoint_on_pages(
                [prepared_pages[page_id]],
                current_checkpoint,
                run_dir / "finetune" / f"step_{step_index:02d}_{page_id}",
                step_index=step_index,
                validation_ratio=0.0,
                split_seed=42,
                oversampling_policy=recipe.oversampling_policy,
                augmentation_policy=recipe.augmentation_policy,
                history_source_pages=history_pages,
                history_sample_line_count=recipe.history_sample_line_count,
                sibling_checkpoint_strategy=recipe.sibling_checkpoint_strategy,
                width_policy=recipe.width_policy,
                lr_scheduler=recipe.lr_scheduler,
                optimizer_name=recipe.optimizer,
                background_plus_rotation_variant_count=recipe.background_plus_rotation_variant_count,
                shuffle_train_each_epoch=recipe.shuffle_train_each_epoch,
                lr=recipe.lr,
                num_iter=recipe.num_iter,
                adam=recipe.optimizer == "adam",
                batch_size=1,
                workers=0,
                valInterval=5,
            )
            current_checkpoint = Path(result.output_checkpoint)
            history_pages.append(prepared_pages[page_id])
        checkpoint = current_checkpoint

    test_pages = {page_id: prepared_pages[page_id] for page_id in fold.test_page_ids}
    prediction = generate_prediction_pagexmls(
        checkpoint,
        test_pages,
        run_dir / "prediction_page_xml",
        width_policy=inference_width_policy,
    )
    return Path(prediction.prediction_folder)


def run_local_finetuning_ladder(
    *,
    paths: ManuscriptPaths,
    fold: Fold,
    methods: Iterable[MethodSpec],
    output_root: Path,
    fine_tune_fn: Callable | None = None,
    predict_fn: Callable | None = None,
    predicted_layout_test_pages_by_count: dict[int, dict] | None = None,
    gnn_ladder_fn: Callable | None = None,
    predicted_layout_prepare_fn: Callable | None = None,
    gnn_finetuning_config: str | Path = DEFAULT_GNN_FINETUNING_CONFIG,
) -> dict[str, Path]:
    """Train one corrected-layout OCR ladder and fold-local GNN ladder.

    Every fine-tuning page supplies corrected PAGE layout and Unicode text to
    the OCR ladder and corrected graph-format labels to the GNN ladder.
    Human-corrected held-out layouts bypass GNN inference. Predicted held-out
    layouts are regenerated with the GNN checkpoint at the same page depth.
    """
    selected_methods = tuple(
        sorted(
            (method for method in methods if method.uses_finetuning),
            key=lambda method: (method.finetune_page_count, method.method_id),
        )
    )
    if not selected_methods:
        return {}

    _ensure_app_import_path()
    if fine_tune_fn is None or predict_fn is None:
        from recognition.active_learning import (
            fine_tune_checkpoint_on_pages,
            generate_prediction_pagexmls,
        )

        fine_tune_fn = fine_tune_checkpoint_on_pages
        predict_fn = generate_prediction_pagexmls

    max_finetune_pages = max(method.finetune_page_count for method in selected_methods)
    selected_train = tuple(fold.train_page_ids[:max_finetune_pages])
    if len(selected_train) < max_finetune_pages:
        raise ValueError(
            f"{fold.fold_id}: requested {max_finetune_pages} fine-tuning pages, "
            f"but only {len(fold.train_page_ids)} train pages are available."
        )

    only_gt_test_layout = all(method.uses_gt_layout for method in selected_methods)
    ladder_name = (
        "annotation_tool_gt_layout_ft_ladder"
        if only_gt_test_layout
        else "annotation_tool_finetuning_ladder"
    )
    ladder_dir = output_root / "runs" / ladder_name / fold.fold_id
    gt_test_page_ids = fold.test_page_ids if any(method.uses_gt_layout for method in selected_methods) else ()
    gt_needed_pages = tuple(dict.fromkeys((*selected_train, *gt_test_page_ids)))
    gt_layout_pages = _prepare_gt_layout_pages(
        paths,
        gt_needed_pages,
        ladder_dir / "corrected_layout_pages",
    )

    predicted_layout_methods = tuple(method for method in selected_methods if not method.uses_gt_layout)
    if predicted_layout_methods:
        if predicted_layout_test_pages_by_count is None:
            if gnn_ladder_fn is None:
                from .gnn_finetuning import run_gnn_finetuning_ladder

                gnn_ladder_fn = run_gnn_finetuning_ladder
            if predicted_layout_prepare_fn is None:
                predicted_layout_prepare_fn = _prepare_annotation_tool_predicted_layout_pages
            gnn_ladder_result = gnn_ladder_fn(
                manuscript_root=paths.root,
                train_page_ids=selected_train,
                base_checkpoint=PRETRAINED_GNN_MODEL,
                output_root=ladder_dir / "gnn_finetune",
                config_path=gnn_finetuning_config,
            )
            predicted_layout_test_pages_by_count = {}
            requested_counts = sorted(
                {method.finetune_page_count for method in predicted_layout_methods}
            )
            for finetune_page_count in requested_counts:
                try:
                    gnn_checkpoint = gnn_ladder_result.checkpoint_by_count[
                        finetune_page_count
                    ]
                except KeyError as exc:
                    raise ValueError(
                        f"{fold.fold_id}: GNN ladder did not produce a "
                        f"{finetune_page_count}-page checkpoint."
                    ) from exc
                predicted_layout_test_pages_by_count[finetune_page_count] = (
                    predicted_layout_prepare_fn(
                        paths=paths,
                        fold=fold,
                        preparation_dir=(
                            ladder_dir
                            / "predicted_layout_test_pages"
                            / f"step_{finetune_page_count:02d}"
                        ),
                        gnn_model_path=Path(gnn_checkpoint),
                        gnn_config_path=PRETRAINED_GNN_CONFIG,
                    )
                )
        else:
            gnn_ladder_result = None

        for method in predicted_layout_methods:
            prepared_at_count = predicted_layout_test_pages_by_count.get(
                method.finetune_page_count,
                {},
            )
            missing_predicted_pages = sorted(
                set(fold.test_page_ids) - set(prepared_at_count)
            )
            if missing_predicted_pages:
                raise ValueError(
                    f"{fold.fold_id}: {method.finetune_page_count}-page "
                    "predicted-layout preparation is missing test pages: "
                    f"{missing_predicted_pages}"
                )
    else:
        gnn_ladder_result = None

    recipe = _load_gui_runtime_ocr_recipe()
    recipe_payload = {
        "source": "app.ocr_active_learning_runtime._runtime_recipe",
        "recipe": recipe.to_dict(),
        "sibling_checkpoint_strategy": recipe.sibling_checkpoint_strategy,
        "ladder_train_page_ids": selected_train,
    }
    _write_json(ladder_dir / "ocr_active_learning_recipe.json", recipe_payload)

    checkpoint_by_count: dict[int, Path] = {}
    current_checkpoint = BASE_OCR_CHECKPOINT
    history_pages = []
    for step_index, page_id in enumerate(selected_train, start=1):
        result = fine_tune_fn(
            [gt_layout_pages[page_id]],
            current_checkpoint,
            ladder_dir / "finetune" / f"step_{step_index:02d}_{page_id}",
            step_index=step_index,
            validation_ratio=0.0,
            split_seed=42,
            oversampling_policy=recipe.oversampling_policy,
            augmentation_policy=recipe.augmentation_policy,
            history_source_pages=history_pages,
            history_sample_line_count=recipe.history_sample_line_count,
            sibling_checkpoint_strategy=recipe.sibling_checkpoint_strategy,
            width_policy=recipe.width_policy,
            lr_scheduler=recipe.lr_scheduler,
            optimizer_name=recipe.optimizer,
            background_plus_rotation_variant_count=recipe.background_plus_rotation_variant_count,
            shuffle_train_each_epoch=recipe.shuffle_train_each_epoch,
            lr=recipe.lr,
            num_iter=recipe.num_iter,
            adam=recipe.optimizer == "adam",
            batch_size=1,
            workers=0,
            valInterval=5,
        )
        current_checkpoint = Path(result.output_checkpoint)
        checkpoint_by_count[step_index] = current_checkpoint
        history_pages.append(gt_layout_pages[page_id])

    prediction_dirs: dict[str, Path] = {}
    for method in selected_methods:
        checkpoint = checkpoint_by_count[method.finetune_page_count]
        source_test_pages = (
            gt_layout_pages
            if method.uses_gt_layout
            else predicted_layout_test_pages_by_count[method.finetune_page_count]
        )
        test_pages = {page_id: source_test_pages[page_id] for page_id in fold.test_page_ids}
        method_run_dir = output_root / "runs" / method.method_id / fold.fold_id
        gnn_checkpoint = None
        if not method.uses_gt_layout and gnn_ladder_result is not None:
            gnn_checkpoint = gnn_ladder_result.checkpoint_by_count[
                method.finetune_page_count
            ]
        _write_json(
            method_run_dir / "ocr_active_learning_recipe.json",
            {
                **recipe_payload,
                "ladder_run_dir": str(ladder_dir.resolve()),
                "finetune_page_count": method.finetune_page_count,
                "checkpoint_path": str(checkpoint.resolve()),
                "training_layout_condition": "human_corrected_gt_layout",
                "test_layout_condition": (
                    "human_corrected_gt_layout" if method.uses_gt_layout else "predicted_layout"
                ),
                "gnn_finetuning_condition": (
                    "not_applied_to_human_corrected_test_layout"
                    if method.uses_gt_layout
                    else "fold_local_joint_finetuning"
                ),
                "gnn_checkpoint_path": (
                    str(Path(gnn_checkpoint).resolve())
                    if gnn_checkpoint is not None
                    else None
                ),
            },
        )
        prediction = predict_fn(
            checkpoint,
            test_pages,
            method_run_dir / "prediction_page_xml",
            width_policy=recipe.width_policy,
        )
        prediction_dirs[method.method_id] = Path(prediction.prediction_folder)

    _write_json(
        ladder_dir / "ladder_summary.json",
        {
            "fold_id": fold.fold_id,
            "train_page_ids": list(fold.train_page_ids),
            "test_page_ids": list(fold.test_page_ids),
            "ladder_train_page_ids": list(selected_train),
            "checkpoint_by_finetune_page_count": {
                str(count): str(path.resolve()) for count, path in checkpoint_by_count.items()
            },
            "gnn_checkpoint_by_finetune_page_count": (
                {
                    str(count): str(path.resolve())
                    for count, path in gnn_ladder_result.checkpoint_by_count.items()
                }
                if gnn_ladder_result is not None
                else {}
            ),
            "gnn_ladder_summary_path": (
                str(gnn_ladder_result.summary_path.resolve())
                if gnn_ladder_result is not None
                else None
            ),
            "training_layout_condition": "human_corrected_gt_layout",
            "method_test_layout_conditions": {
                method.method_id: (
                    "human_corrected_gt_layout" if method.uses_gt_layout else "predicted_layout"
                )
                for method in selected_methods
            },
            "prediction_dirs": {method_id: str(path.resolve()) for method_id, path in prediction_dirs.items()},
        },
    )
    return prediction_dirs


def run_local_gt_layout_finetuning_ladder(
    *,
    paths: ManuscriptPaths,
    fold: Fold,
    methods: Iterable[MethodSpec],
    output_root: Path,
    fine_tune_fn: Callable | None = None,
    predict_fn: Callable | None = None,
) -> dict[str, Path]:
    """Backward-compatible entry point for the original corrected-layout ladder."""
    return run_local_finetuning_ladder(
        paths=paths,
        fold=fold,
        methods=methods,
        output_root=output_root,
        fine_tune_fn=fine_tune_fn,
        predict_fn=predict_fn,
    )


def _gemini_timeout_seconds() -> float:
    raw_value = os.getenv("GEMINI_OCR_TIMEOUT_SECONDS")
    if raw_value is None or str(raw_value).strip() == "":
        return DEFAULT_GEMINI_TIMEOUT_SECONDS
    try:
        value = float(raw_value)
    except ValueError:
        return DEFAULT_GEMINI_TIMEOUT_SECONDS
    if not math.isfinite(value) or value <= 0:
        return DEFAULT_GEMINI_TIMEOUT_SECONDS
    return max(5.0, min(value, 300.0))


def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None or str(raw_value).strip() == "":
        return default
    try:
        value = int(raw_value)
    except ValueError:
        return default
    return max(minimum, min(value, maximum))


def _env_float(name: str, default: float, *, minimum: float, maximum: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None or str(raw_value).strip() == "":
        return default
    try:
        value = float(raw_value)
    except ValueError:
        return default
    if not math.isfinite(value):
        return default
    return max(minimum, min(value, maximum))


def _gemini_page_workers() -> int:
    return _env_int("GEMINI_OCR_PAGE_WORKERS", DEFAULT_GEMINI_PAGE_WORKERS, minimum=1, maximum=16)


def _gemini_max_retries() -> int:
    return _env_int("GEMINI_OCR_MAX_RETRIES", DEFAULT_GEMINI_MAX_RETRIES, minimum=0, maximum=10)


def _gemini_request_spacing_seconds() -> float:
    return _env_float(
        "GEMINI_OCR_REQUEST_SPACING_SECONDS",
        DEFAULT_GEMINI_REQUEST_SPACING_SECONDS,
        minimum=0.0,
        maximum=60.0,
    )


def _gemini_retry_base_delay_seconds() -> float:
    return _env_float(
        "GEMINI_OCR_RETRY_BASE_DELAY_SECONDS",
        DEFAULT_GEMINI_RETRY_BASE_DELAY_SECONDS,
        minimum=0.0,
        maximum=60.0,
    )


class _RequestRateLimiter:
    def __init__(self, min_interval_seconds: float):
        self._min_interval_seconds = max(0.0, float(min_interval_seconds))
        self._lock = threading.Lock()
        self._next_start_time = 0.0

    def wait(self) -> None:
        if self._min_interval_seconds <= 0:
            return
        with self._lock:
            now = time.perf_counter()
            wait_seconds = max(0.0, self._next_start_time - now)
            self._next_start_time = max(now, self._next_start_time) + self._min_interval_seconds
        if wait_seconds > 0:
            time.sleep(wait_seconds)


def _retry_delay_seconds(attempt_number: int) -> float:
    base_delay = _gemini_retry_base_delay_seconds()
    if base_delay <= 0:
        return 0.0
    return min(base_delay * (2 ** max(0, attempt_number - 1)), 60.0)


def _failure_status_from_exception(exc: Exception) -> str:
    name = exc.__class__.__name__.lower()
    message = str(exc).lower()
    if "timeout" in name or "timed out" in message or "deadline exceeded" in message:
        return "api_timeout"
    if isinstance(exc, AdapterError):
        value = str(exc)
        return value if value in {"empty_response", "json_parse_error", "json_schema_error", "adapter_error", "other_output_error"} else "adapter_error"
    return "api_error"


def _failure_status_from_app_gemini_result(result: dict) -> str:
    error_code = str((result or {}).get("errorCode") or "")
    if error_code == "gemini_timeout":
        return "api_timeout"
    if error_code == "gemini_empty_response":
        return "empty_response"
    if error_code == "gemini_invalid_response":
        return "other_output_error"
    return "api_error"


def _usage_records_from_attempts(attempts: Iterable[dict]) -> list[dict]:
    records = []
    for attempt in attempts:
        for record in attempt.get("usage_records") or []:
            if isinstance(record, dict):
                records.append(record)
        metadata = usage_metadata_to_dict(attempt.get("usage_metadata"))
        if metadata:
            records.append({"usage_metadata": metadata})
    return records


def _gemini_usage_payload(
    *,
    page_id: str,
    status: str,
    model: str,
    elapsed_seconds: float,
    attempts: list[dict],
    error: str | None = None,
    max_retries: int | None = None,
) -> dict:
    usage_records = _usage_records_from_attempts(attempts)
    attempt_count = len(attempts)
    effective_max_retries = _gemini_max_retries() if max_retries is None else int(max_retries)
    payload = {
        "page_id": page_id,
        "elapsed_seconds": elapsed_seconds,
        "status": status,
        "model": model,
        "attempt_count": attempt_count,
        "retry_count": max(0, attempt_count - 1),
        "max_retries": effective_max_retries,
        "request_count": max(attempt_count, len(usage_records)),
        "attempts": attempts,
        "usage_records": usage_records,
        "usage_metadata": summarize_usage_metadata(usage_records),
    }
    if error:
        payload["error"] = error
    return payload


def _write_empty_prediction_for_status(paths: ManuscriptPaths, page_id: str, prediction_dir: Path) -> None:
    adapt_failed_prediction(paths.pagexml_dir / f"{page_id}.xml", prediction_dir / f"{page_id}.xml")


def run_vlm_end_to_end_gemini(
    *,
    paths: ManuscriptPaths,
    fold: Fold,
    method: MethodSpec,
    run_dir: Path,
) -> tuple[Path, dict[str, str]]:
    api_key = load_gemini_api_key()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured in app/.env or the environment.")

    raw_dir = run_dir / "raw_gemini_json"
    prediction_dir = run_dir / "prediction_page_xml"
    usage_dir = run_dir / "gemini_usage"
    raw_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    statuses: dict[str, str] = {}
    page_ids = tuple(fold.test_page_ids)
    if not page_ids:
        return prediction_dir, statuses

    max_retries = _gemini_max_retries()
    rate_limiter = _RequestRateLimiter(_gemini_request_spacing_seconds())
    max_workers = min(_gemini_page_workers(), len(page_ids))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _run_vlm_end_to_end_gemini_page,
                paths=paths,
                page_id=page_id,
                api_key=api_key,
                raw_dir=raw_dir,
                prediction_dir=prediction_dir,
                usage_dir=usage_dir,
                rate_limiter=rate_limiter,
            ): page_id
            for page_id in page_ids
        }
        for future in concurrent.futures.as_completed(futures):
            page_id = futures[future]
            try:
                result = future.result()
                statuses[result["page_id"]] = result["status"]
            except Exception as exc:
                status = _failure_status_from_exception(exc)
                statuses[page_id] = status
                _write_empty_prediction_for_status(paths, page_id, prediction_dir)
                _write_json(
                    raw_dir / f"{page_id}.json",
                    {
                        "status": "failure",
                        "failure_status": status,
                        "error": str(exc),
                    },
                )
                _write_json(
                    usage_dir / f"{page_id}.json",
                    _gemini_usage_payload(
                        page_id=page_id,
                        status=status,
                        model="gemini-3.5-flash",
                        elapsed_seconds=0.0,
                        attempts=[
                            {
                                "attempt": 1,
                                "status": status,
                                "elapsed_seconds": 0.0,
                                "error": str(exc),
                            }
                        ],
                        error=str(exc),
                        max_retries=max_retries,
                    ),
                )
    return prediction_dir, statuses


def _run_vlm_end_to_end_gemini_page(
    *,
    paths: ManuscriptPaths,
    page_id: str,
    api_key: str,
    raw_dir: Path,
    prediction_dir: Path,
    usage_dir: Path,
    rate_limiter: _RequestRateLimiter,
) -> dict:
    from google import genai
    from google.genai import types
    from PIL import Image

    model_name = "gemini-3.5-flash"
    timeout_ms = int(_gemini_timeout_seconds() * 1000)
    max_retries = _gemini_max_retries()
    image_path = _find_page_image(page_id, [paths.images_dir])
    template_page = load_pagexml(paths.pagexml_dir / f"{page_id}.xml", repair_geometry=True)
    started = time.perf_counter()
    attempts: list[dict] = []
    final_status = "api_error"
    final_error = None

    for attempt_number in range(1, max_retries + 2):
        attempt_started = time.perf_counter()
        attempt_usage_metadata = {}
        try:
            client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(
                    timeout=timeout_ms,
                    retryOptions=types.HttpRetryOptions(attempts=1),
                ),
            )
            rate_limiter.wait()
            with Image.open(image_path) as image:
                response = client.models.generate_content(
                    model=model_name,
                    contents=[image, VLM_END_TO_END_PROMPT],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.2,
                    ),
                )
            raw_text = response.text or ""
            (raw_dir / f"{page_id}.json").write_text(raw_text, encoding="utf-8")
            attempt_usage_metadata = usage_metadata_to_dict(getattr(response, "usage_metadata", None))
            vlm_json_to_pagexml(
                raw_text,
                template_page=template_page,
                output_path=prediction_dir / f"{page_id}.xml",
            )
            final_status = "success"
            attempts.append(
                {
                    "attempt": attempt_number,
                    "status": final_status,
                    "elapsed_seconds": time.perf_counter() - attempt_started,
                    "usage_metadata": attempt_usage_metadata,
                }
            )
            final_error = None
            break
        except Exception as exc:
            final_status = _failure_status_from_exception(exc)
            final_error = str(exc)
            attempts.append(
                {
                    "attempt": attempt_number,
                    "status": final_status,
                    "elapsed_seconds": time.perf_counter() - attempt_started,
                    "error": final_error,
                    "usage_metadata": attempt_usage_metadata,
                }
            )
            if attempt_number <= max_retries:
                delay = _retry_delay_seconds(attempt_number)
                if delay > 0:
                    time.sleep(delay)

    if final_status != "success":
        _write_empty_prediction_for_status(paths, page_id, prediction_dir)
        raw_path = raw_dir / f"{page_id}.json"
        if not raw_path.exists():
            _write_json(
                raw_path,
                {
                    "status": "failure",
                    "failure_status": final_status,
                    "error": final_error,
                },
            )

    _write_json(
        usage_dir / f"{page_id}.json",
        _gemini_usage_payload(
            page_id=page_id,
            status=final_status,
            model=model_name,
            elapsed_seconds=time.perf_counter() - started,
            attempts=attempts,
            error=final_error,
            max_retries=max_retries,
        ),
    )
    return {"page_id": page_id, "status": final_status}


def _recording_gemini_client_factory(real_client_cls, usage_records: list[dict], lock: threading.Lock):
    class RecordingModels:
        def __init__(self, models):
            self._models = models

        def generate_content(self, *args, **kwargs):
            response = self._models.generate_content(*args, **kwargs)
            usage = usage_metadata_to_dict(getattr(response, "usage_metadata", None))
            with lock:
                usage_records.append({"usage_metadata": usage})
            return response

        def __getattr__(self, name):
            return getattr(self._models, name)

    class RecordingClient:
        def __init__(self, *args, **kwargs):
            self._client = real_client_cls(*args, **kwargs)
            self.models = RecordingModels(self._client.models)

        def __getattr__(self, name):
            return getattr(self._client, name)

    return RecordingClient


def _copy_gt_layout_manuscript_for_gemini(paths: ManuscriptPaths, fold: Fold, target_root: Path) -> Path:
    if target_root.exists():
        shutil.rmtree(target_root)
    (target_root / "images_resized").mkdir(parents=True, exist_ok=True)
    pagexml_target = target_root / "layout_analysis_output" / "page-xml-format"
    pagexml_target.mkdir(parents=True, exist_ok=True)
    for page_id in fold.test_page_ids:
        shutil.copy2(_find_page_image(page_id, [paths.images_dir]), target_root / "images_resized" / f"{page_id}.jpg")
        for suffix in (".xml", "_line_segmentation_metadata.json", "_reading_direction_metadata.json"):
            source = paths.pagexml_dir / f"{page_id}{suffix}"
            if source.exists():
                shutil.copy2(source, pagexml_target / source.name)
    return target_root


def _write_layout_grounded_gemini_prediction_pagexml(
    *,
    layout_xml_path: Path,
    predictions_by_structure_line_id: dict,
    output_path: Path,
) -> None:
    tree = ET.parse(layout_xml_path)
    root = tree.getroot()
    root_namespace = tag_namespace(root.tag)
    if root_namespace:
        ET.register_namespace("", root_namespace)

    for textline in root.iter():
        if local_name(textline.tag) != "TextLine":
            continue

        for child in list(textline):
            if local_name(child.tag) == "TextEquiv":
                textline.remove(child)

        line_key = extract_structure_line_id(textline.get("custom"))
        if line_key is None:
            line_key = textline.get("id")
        predicted_text = str(predictions_by_structure_line_id.get(str(line_key), "") or "").strip()
        if not predicted_text:
            continue

        namespace = tag_namespace(textline.tag) or root_namespace
        text_equiv = ET.SubElement(textline, qualified("TextEquiv", namespace))
        unicode_elem = ET.SubElement(text_equiv, qualified("Unicode", namespace))
        unicode_elem.text = predicted_text

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(ET, "indent"):
        ET.indent(tree, space="\t", level=0)
    tree.write(output_path, encoding="UTF-8", xml_declaration=True)


def _run_layout_grounded_gemini_page_worker(payload: dict) -> dict:
    page_id = str(payload["page_id"])
    manuscript_id = str(payload["manuscript_id"])
    upload_root = Path(payload["upload_root"])
    source_pagexml_dir = Path(payload["source_pagexml_dir"])
    copied_pagexml_dir = Path(payload["copied_pagexml_dir"])
    prediction_dir = Path(payload["prediction_dir"])
    usage_dir = Path(payload["usage_dir"])
    raw_dir = Path(payload["raw_dir"])
    max_retries = int(payload["max_retries"])
    model_name = "gemini-3.5-flash"
    started = time.perf_counter()
    attempts: list[dict] = []
    final_status = "api_error"
    final_error = None
    final_result: dict | None = None

    load_gemini_api_key()
    _ensure_app_import_path()
    import app as backend_app_module

    previous_upload_folder = backend_app_module.UPLOAD_FOLDER
    real_gemini_client_cls = backend_app_module.genai.Client
    layout_xml_path = source_pagexml_dir / f"{page_id}.xml"
    copied_xml_path = copied_pagexml_dir / f"{page_id}.xml"

    try:
        backend_app_module.UPLOAD_FOLDER = str(upload_root)
        for attempt_number in range(1, max_retries + 2):
            attempt_started = time.perf_counter()
            usage_records: list[dict] = []
            usage_lock = threading.Lock()
            backend_app_module.genai.Client = _recording_gemini_client_factory(
                real_gemini_client_cls,
                usage_records,
                usage_lock,
            )
            try:
                shutil.copy2(layout_xml_path, copied_xml_path)
                result = backend_app_module._run_gemini_recognition_internal(manuscript_id, page_id)
                final_result = dict(result or {})
                if result.get("error"):
                    final_status = _failure_status_from_app_gemini_result(result)
                    final_error = str(result.get("error") or final_status)
                    attempts.append(
                        {
                            "attempt": attempt_number,
                            "status": final_status,
                            "elapsed_seconds": time.perf_counter() - attempt_started,
                            "error": final_error,
                            "usage_records": usage_records,
                        }
                    )
                else:
                    final_status = "success"
                    final_error = None
                    _write_layout_grounded_gemini_prediction_pagexml(
                        layout_xml_path=layout_xml_path,
                        predictions_by_structure_line_id=result.get("text", {}),
                        output_path=prediction_dir / f"{page_id}.xml",
                    )
                    attempts.append(
                        {
                            "attempt": attempt_number,
                            "status": final_status,
                            "elapsed_seconds": time.perf_counter() - attempt_started,
                            "usage_records": usage_records,
                        }
                    )
                    break
            except Exception as exc:
                final_status = _failure_status_from_exception(exc)
                final_error = str(exc)
                attempts.append(
                    {
                        "attempt": attempt_number,
                        "status": final_status,
                        "elapsed_seconds": time.perf_counter() - attempt_started,
                        "error": final_error,
                        "usage_records": usage_records,
                    }
                )
            finally:
                backend_app_module.genai.Client = real_gemini_client_cls

            if attempt_number <= max_retries:
                delay = _retry_delay_seconds(attempt_number)
                if delay > 0:
                    time.sleep(delay)
    finally:
        backend_app_module.genai.Client = real_gemini_client_cls
        backend_app_module.UPLOAD_FOLDER = previous_upload_folder

    if final_status != "success":
        adapt_failed_prediction(layout_xml_path, prediction_dir / f"{page_id}.xml")

    if final_result is None:
        final_result = {"error": final_error, "status": final_status}
    _write_json(raw_dir / f"{page_id}.json", final_result)
    _write_json(
        usage_dir / f"{page_id}.json",
        _gemini_usage_payload(
            page_id=page_id,
            status=final_status,
            model=model_name,
            elapsed_seconds=time.perf_counter() - started,
            attempts=attempts,
            error=final_error,
            max_retries=max_retries,
        ),
    )
    return {"page_id": page_id, "status": final_status}


def run_layout_grounded_gemini_with_app_copy(
    *,
    paths: ManuscriptPaths,
    fold: Fold,
    method: MethodSpec,
    run_dir: Path,
) -> tuple[Path, dict[str, str]]:
    load_gemini_api_key()
    upload_root = run_dir / "upload_root"
    manuscript_root = _copy_gt_layout_manuscript_for_gemini(paths, fold, upload_root / paths.manuscript_id)
    statuses: dict[str, str] = {}
    usage_dir = run_dir / "gemini_usage"
    raw_dir = run_dir / "raw_gemini_json"
    prediction_dir = run_dir / "prediction_page_xml"
    usage_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    page_ids = tuple(fold.test_page_ids)
    if not page_ids:
        return prediction_dir, statuses

    max_retries = _gemini_max_retries()
    rate_limiter = _RequestRateLimiter(_gemini_request_spacing_seconds())
    max_workers = min(_gemini_page_workers(), len(page_ids))
    copied_pagexml_dir = manuscript_root / "layout_analysis_output" / "page-xml-format"
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for page_id in page_ids:
            rate_limiter.wait()
            futures[
                executor.submit(
                    _run_layout_grounded_gemini_page_worker,
                    {
                        "page_id": page_id,
                        "manuscript_id": paths.manuscript_id,
                        "upload_root": str(upload_root),
                        "source_pagexml_dir": str(paths.pagexml_dir),
                        "copied_pagexml_dir": str(copied_pagexml_dir),
                        "prediction_dir": str(prediction_dir),
                        "usage_dir": str(usage_dir),
                        "raw_dir": str(raw_dir),
                        "max_retries": max_retries,
                    },
                )
            ] = page_id
        for future in concurrent.futures.as_completed(futures):
            page_id = futures[future]
            try:
                result = future.result()
                statuses[result["page_id"]] = result["status"]
            except Exception as exc:
                status = _failure_status_from_exception(exc)
                statuses[page_id] = status
                _write_empty_prediction_for_status(paths, page_id, prediction_dir)
                _write_json(
                    raw_dir / f"{page_id}.json",
                    {
                        "status": "failure",
                        "failure_status": status,
                        "error": str(exc),
                    },
                )
                _write_json(
                    usage_dir / f"{page_id}.json",
                    _gemini_usage_payload(
                        page_id=page_id,
                        status=status,
                        model="gemini-3.5-flash",
                        elapsed_seconds=0.0,
                        attempts=[
                            {
                                "attempt": 1,
                                "status": status,
                                "elapsed_seconds": 0.0,
                                "error": str(exc),
                            }
                        ],
                        error=str(exc),
                        max_retries=max_retries,
                    ),
                )
    return prediction_dir, statuses


def adapt_failed_prediction(gt_page_path: Path, output_path: Path) -> None:
    gt_page = load_pagexml(gt_page_path, repair_geometry=True)
    empty_page = empty_page_like(gt_page)
    write_pagexml(empty_page, output_path, lines=())


def run_vlm_json_file_adapter(
    *,
    paths: ManuscriptPaths,
    fold: Fold,
    json_dir: Path,
    output_dir: Path,
) -> dict[str, str]:
    statuses: dict[str, str] = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    for page_id in fold.test_page_ids:
        gt_xml = paths.pagexml_dir / f"{page_id}.xml"
        template_page = load_pagexml(gt_xml, repair_geometry=True)
        json_path = json_dir / f"{page_id}.json"
        pred_xml = output_dir / f"{page_id}.xml"
        if not json_path.exists():
            adapt_failed_prediction(gt_xml, pred_xml)
            statuses[page_id] = "empty_response"
            continue
        try:
            vlm_json_to_pagexml(json_path.read_text(encoding="utf-8"), template_page=template_page, output_path=pred_xml)
            statuses[page_id] = "success"
        except AdapterError as exc:
            adapt_failed_prediction(gt_xml, pred_xml)
            statuses[page_id] = str(exc) if str(exc) else "adapter_error"
        except Exception:
            adapt_failed_prediction(gt_xml, pred_xml)
            statuses[page_id] = "adapter_error"
    return statuses


def evaluate_prediction_folder(
    *,
    paths: ManuscriptPaths,
    fold: Fold,
    method: MethodSpec,
    prediction_dir: Path,
    statuses: dict[str, str] | None = None,
    diagnostics_dir: Path | None = None,
) -> list[dict]:
    records = []
    textedit_pairs: list[PageXmlPair] = []
    records_by_textedit_key: dict[str, dict] = {}
    layout_effort_by_page = _load_layout_effort_by_page(paths.root)
    for page_id in fold.test_page_ids:
        gt_xml_path = paths.pagexml_dir / f"{page_id}.xml"
        gt_page = load_pagexml(gt_xml_path, repair_geometry=True)
        pred_path = prediction_dir / f"{page_id}.xml"
        status = (statuses or {}).get(page_id, "success")
        if not pred_path.exists():
            pred_page = empty_page_like(gt_page)
            status = "empty_response"
        else:
            pred_page = load_pagexml(
                pred_path,
                strict=True,
                repair_geometry=True,
                allow_empty_geometry=not method.provides_layout,
            )
        record = evaluate_page(
            manuscript_id=paths.manuscript_id,
            fold_id=fold.fold_id,
            page_id=page_id,
            method_id=method.method_id,
            gt_page=gt_page,
            pred_page=pred_page,
            status=status,
            calculate_textedit=False,
            calculate_layout_metrics=method.provides_layout,
            calculate_page_cer=(
                method.provides_layout or method.page_cer_uses_output_order
            ),
            page_cer_predicted_lines_in_output_order=(
                method.page_cer_uses_output_order
            ),
        )
        textedit_key = f"{fold.fold_id}:{page_id}"
        textedit_pairs.append(
            PageXmlPair(
                key=textedit_key,
                gt_xml_path=gt_xml_path,
                pred_xml_path=(
                    None
                    if status in FAILURE_STATUSES or not pred_path.exists()
                    else pred_path
                ),
                image_name=gt_page.image_filename,
            )
        )
        records_by_textedit_key[textedit_key] = record
        record.update(_layout_effort_fields(layout_effort_by_page, page_id))
        if diagnostics_dir is not None:
            record.update(
                write_page_diagnostics(
                    gt_page=gt_page,
                    pred_page=pred_page,
                    output_dir=diagnostics_dir / fold.fold_id / method.method_id,
                )
            )
        records.append(record)
    textedit_evaluation = evaluate_pagexml_pairs(textedit_pairs)
    for textedit_key, payload in textedit_evaluation.page_metrics.items():
        records_by_textedit_key[textedit_key].update(payload)
    return records


def evaluate_existing_prediction_tree(
    *,
    manuscript_root: str | Path,
    predictions_root: str | Path,
    method_id: str,
    output_root: str | Path,
    write_diagnostics: bool = False,
    split_seed: int = DEFAULT_SPLIT_SEED,
) -> dict:
    paths = default_manuscript_paths(manuscript_root)
    method = method_by_id(method_id)
    all_records = []
    predictions_root = Path(predictions_root)
    output_root = Path(output_root)
    folds = _load_or_create_run_folds(paths, output_root, split_seed=split_seed)
    _write_experiment_reproducibility(output_root)
    for fold in folds:
        prediction_dir = predictions_root / fold.fold_id / method_id
        all_records.extend(
            evaluate_prediction_folder(
                paths=paths,
                fold=fold,
                method=method,
                prediction_dir=prediction_dir,
                diagnostics_dir=(output_root / "diagnostics") if write_diagnostics else None,
            )
        )
    aggregate = aggregate_page_records(all_records)
    payload = {
        "method": asdict(method),
        "manuscript_id": paths.manuscript_id,
        "folds": [asdict(fold) for fold in folds],
        "textedit_metric": textedit_reproducibility_metadata(),
        "devanagari_textedit_metric": (
            devanagari_textedit_reproducibility_metadata()
        ),
        "aggregate": aggregate,
        "page_records": all_records,
    }
    _write_json(output_root / method_id / "metrics.json", payload)
    _write_csv(output_root / method_id / "per_page.csv", all_records)
    _write_report_artifacts(output_root)
    return payload


def _retained_prediction_path(
    output_root: Path,
    *,
    method_id: str,
    fold_id: str,
    page_id: str,
) -> Path | None:
    fold_root = output_root / "runs" / method_id / fold_id
    candidates = (
        fold_root / "prediction_page_xml" / f"{page_id}.xml",
        fold_root / "predictions" / f"{page_id}.xml",
    )
    return next((candidate for candidate in candidates if candidate.exists()), None)


def refresh_textedit_for_existing_run(
    output_root: str | Path,
    *,
    manuscript_root: str | Path | None = None,
) -> dict:
    """Refresh only TextEdit fields from retained PAGE-XML predictions."""
    root = Path(output_root)
    metrics_paths = sorted((root / "metrics").glob("*/metrics.json"))
    if not metrics_paths:
        raise ValueError(f"No method metrics found under {root / 'metrics'}")

    first_payload = json.loads(metrics_paths[0].read_text(encoding="utf-8"))
    manuscript_id = str(first_payload.get("manuscript_id") or root.name)
    paths = default_manuscript_paths(
        manuscript_root or (APP_ROOT / "input_manuscripts" / manuscript_id)
    )
    refreshed_methods = []
    refreshed_page_count = 0

    for metrics_path in metrics_paths:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        method = dict(payload.get("method") or {})
        method_id = str(method.get("method_id") or metrics_path.parent.name)
        page_records = list(payload.get("page_records") or [])
        pairs = []
        records_by_key = {}
        for record in page_records:
            fold_id = str(record["fold_id"])
            page_id = str(record["page_id"])
            key = f"{fold_id}:{page_id}"
            pred_path = _retained_prediction_path(
                root,
                method_id=method_id,
                fold_id=fold_id,
                page_id=page_id,
            )
            if record.get("status") in FAILURE_STATUSES:
                pred_path = None
            gt_path = paths.pagexml_dir / f"{page_id}.xml"
            pairs.append(
                PageXmlPair(
                    key=key,
                    gt_xml_path=gt_path,
                    pred_xml_path=pred_path,
                )
            )
            records_by_key[key] = record

        evaluation = evaluate_pagexml_pairs(pairs)
        for key, textedit_payload in evaluation.page_metrics.items():
            records_by_key[key].update(textedit_payload)

        refreshed_aggregate = aggregate_page_records(page_records)
        official_edit_dist = evaluation.official_result["Edit_dist"]
        refreshed_aggregate["textedit_all_page_avg"] = float(
            official_edit_dist["ALL_page_avg"]
        )
        refreshed_aggregate["textedit_edit_whole"] = float(
            official_edit_dist["edit_whole"]
        )
        refreshed_aggregate["textedit_edit_sample_avg"] = float(
            official_edit_dist["edit_sample_avg"]
        )
        refreshed_aggregate["mean_textedit"] = refreshed_aggregate[
            "textedit_all_page_avg"
        ]
        refreshed_aggregate["micro_textedit"] = refreshed_aggregate[
            "textedit_edit_whole"
        ]
        aggregate = dict(payload.get("aggregate") or {})
        for key in TEXTEDIT_AGGREGATE_KEYS:
            aggregate[key] = refreshed_aggregate[key]
        for key in DEVANAGARI_TEXTEDIT_AGGREGATE_KEYS:
            aggregate[key] = refreshed_aggregate[key]
        aggregate["devanagari_textedit_page_count"] = refreshed_aggregate[
            "devanagari_textedit_page_count"
        ]
        payload["aggregate"] = aggregate
        payload["page_records"] = page_records
        payload["textedit_metric"] = textedit_reproducibility_metadata()
        payload["devanagari_textedit_metric"] = (
            devanagari_textedit_reproducibility_metadata()
        )
        _write_json(metrics_path, payload)
        _write_csv(metrics_path.parent / "per_page.csv", page_records)
        refreshed_methods.append(method_id)
        refreshed_page_count += len(page_records)

    _write_report_artifacts(root)
    return {
        "output_root": str(root.resolve()),
        "manuscript_id": manuscript_id,
        "method_ids": refreshed_methods,
        "method_count": len(refreshed_methods),
        "page_record_count": refreshed_page_count,
        "textedit_metric": textedit_reproducibility_metadata(),
        "devanagari_textedit_metric": (
            devanagari_textedit_reproducibility_metadata()
        ),
    }


def refresh_textedit_results(output_root: str | Path) -> dict:
    """Refresh one manuscript run or every manuscript child of a combined run."""
    root = Path(output_root)
    if (root / "folds.json").exists():
        return {
            "runs": [refresh_textedit_for_existing_run(root)],
            "combined_report": None,
        }

    run_roots = sorted(
        child
        for child in root.iterdir()
        if child.is_dir() and (child / "folds.json").exists()
    )
    if not run_roots:
        raise ValueError(f"No manuscript run roots found under {root}")
    runs = [refresh_textedit_for_existing_run(run_root) for run_root in run_roots]

    from .reporting import write_combined_table_report

    combined = write_combined_table_report(run_roots, root)
    return {
        "runs": runs,
        "combined_report": str(combined.markdown_path.resolve()),
    }


def _select_json_dir(json_root: Path, *, fold: Fold, method_id: str) -> Path:
    candidates = (
        json_root / fold.fold_id / method_id,
        json_root / fold.fold_id,
        json_root / method_id,
        json_root,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return json_root


def adapt_vlm_json_and_evaluate(
    *,
    manuscript_root: str | Path,
    json_root: str | Path,
    output_root: str | Path,
    method_id: str = "gemini_e2e",
    write_diagnostics: bool = False,
    fold_ids: Iterable[str] | None = None,
    max_test_pages: int | None = None,
    split_seed: int = DEFAULT_SPLIT_SEED,
) -> dict:
    paths = default_manuscript_paths(manuscript_root)
    method = method_by_id(method_id)
    json_root = Path(json_root)
    output_root = Path(output_root)
    folds = select_folds(
        _load_or_create_run_folds(paths, output_root, split_seed=split_seed),
        fold_ids=fold_ids,
        max_test_pages=max_test_pages,
    )
    _write_experiment_reproducibility(output_root)
    all_records = []
    status_root = output_root / "statuses"
    for fold in folds:
        json_dir = _select_json_dir(json_root, fold=fold, method_id=method_id)
        prediction_dir = output_root / "predictions" / fold.fold_id / method_id
        statuses = run_vlm_json_file_adapter(
            paths=paths,
            fold=fold,
            json_dir=json_dir,
            output_dir=prediction_dir,
        )
        _write_json(
            status_root / fold.fold_id / f"{method_id}.json",
            {
                "fold_id": fold.fold_id,
                "method_id": method_id,
                "json_dir": str(json_dir),
                "statuses": statuses,
            },
        )
        all_records.extend(
            evaluate_prediction_folder(
                paths=paths,
                fold=fold,
                method=method,
                prediction_dir=prediction_dir,
                statuses=statuses,
                diagnostics_dir=(output_root / "diagnostics") if write_diagnostics else None,
            )
        )
    payload = {
        "method": asdict(method),
        "manuscript_id": paths.manuscript_id,
        "json_root": str(json_root),
        "folds": [asdict(fold) for fold in folds],
        "textedit_metric": textedit_reproducibility_metadata(),
        "devanagari_textedit_metric": (
            devanagari_textedit_reproducibility_metadata()
        ),
        "aggregate": aggregate_page_records(all_records),
        "page_records": all_records,
    }
    _write_json(output_root / "metrics" / method.method_id / "metrics.json", payload)
    _write_csv(output_root / "metrics" / method.method_id / "per_page.csv", all_records)
    _write_report_artifacts(output_root)
    return payload


def run_local_gt_layout_experiment(
    *,
    manuscript_root: str | Path,
    output_root: str | Path,
    method_ids: Iterable[str] = (
        "annotation_tool_gt_layout",
        "annotation_tool_gt_layout_ft_1",
        "annotation_tool_gt_layout_ft_2",
        "annotation_tool_gt_layout_ft_3",
    ),
    write_diagnostics: bool = False,
    fold_ids: Iterable[str] | None = None,
    max_test_pages: int | None = None,
    split_seed: int = DEFAULT_SPLIT_SEED,
) -> dict:
    paths = default_manuscript_paths(manuscript_root)
    output_root = Path(output_root)
    folds = select_folds(
        _load_or_create_run_folds(paths, output_root, split_seed=split_seed),
        fold_ids=fold_ids,
        max_test_pages=max_test_pages,
    )
    _write_experiment_reproducibility(output_root)
    results = {}
    methods = tuple(method_by_id(method_id) for method_id in method_ids)
    records_by_method = _run_methods_with_finetuning_ladder(
        paths=paths,
        folds=folds,
        methods=methods,
        output_root=output_root,
        write_diagnostics=write_diagnostics,
    )
    for method in methods:
        all_records = records_by_method[method.method_id]
        payload = {
            "method": asdict(method),
            "ocr_active_learning_recipe": _ocr_recipe_metadata_for_method(method),
            "gnn_finetuning_recipe": _gnn_recipe_metadata_for_method(method),
            "manuscript_id": paths.manuscript_id,
            "folds": [asdict(fold) for fold in folds],
            "textedit_metric": textedit_reproducibility_metadata(),
            "devanagari_textedit_metric": (
                devanagari_textedit_reproducibility_metadata()
            ),
            "aggregate": aggregate_page_records(all_records),
            "page_records": all_records,
        }
        _write_json(output_root / "metrics" / method.method_id / "metrics.json", payload)
        _write_csv(output_root / "metrics" / method.method_id / "per_page.csv", all_records)
        results[method.method_id] = payload
    _write_report_artifacts(output_root)
    return results


def prepare_gt_layout_pages_only(
    *,
    manuscript_root: str | Path,
    output_root: str | Path,
    page_ids: Iterable[str] | None = None,
) -> dict:
    paths = default_manuscript_paths(manuscript_root)
    selected_page_ids = tuple(page_ids or discover_page_ids(paths))
    if not selected_page_ids:
        raise ValueError(f"No pages selected for manuscript: {paths.root}")
    output_root = Path(output_root)
    _write_experiment_reproducibility(output_root)
    prepared_pages = _prepare_gt_layout_pages(paths, selected_page_ids, output_root / "prepared_pages")
    rows = []
    for page_id in selected_page_ids:
        prepared_page = prepared_pages[page_id]
        rows.append(
            {
                "page_id": page_id,
                "prepared_line_count": len(prepared_page.records),
                "source_xml_path": prepared_page.source_xml_path,
                "gt_path": prepared_page.gt_path,
                "manifest_path": prepared_page.manifest_path,
                "line_segmentation_strategy_name": prepared_page.line_segmentation_strategy_name,
                "geometry_summary": prepared_page.geometry_summary,
                "finetune_filter": prepared_page.finetune_filter,
            }
        )
    payload = {
        "manuscript_id": paths.manuscript_id,
        "page_count": len(rows),
        "pages": rows,
    }
    _write_json(output_root / "prepare_gt_layout_summary.json", payload)
    return payload


def run_method(
    *,
    paths: ManuscriptPaths,
    fold: Fold,
    method: MethodSpec,
    run_dir: Path,
    vlm_predictions_root: Path | None = None,
    allow_vlm_pagexml_drift: bool = False,
) -> tuple[Path, dict[str, str] | None]:
    if method.method_id in DISABLED_METHOD_IDS:
        raise ValueError(DISABLED_METHOD_IDS[method.method_id])
    if is_vlm_method(method.method_id):
        if vlm_predictions_root is None:
            raise ValueError(
                f"--vlm-predictions-root is required for cached VLM method {method.method_id}."
            )
        prediction_dir, statuses, _ = materialize_cached_fold(
            paths=paths,
            fold=fold,
            cache_root=vlm_predictions_root,
            method_id=method.method_id,
            output_dir=run_dir / "predictions",
            allow_pagexml_drift=allow_vlm_pagexml_drift,
        )
        return prediction_dir, statuses
    if method.method_id == "annotation_tool_e2e":
        return run_annotation_tool_auto_layout(paths=paths, fold=fold, method=method, run_dir=run_dir), None
    if method.uses_gt_layout:
        return run_local_ocr_with_gt_layout(paths=paths, fold=fold, method=method, run_dir=run_dir), None
    raise ValueError(f"Method is not runnable by this harness yet: {method.method_id}")


def run_methods_experiment(
    *,
    manuscript_root: str | Path,
    output_root: str | Path,
    method_ids: Iterable[str],
    write_diagnostics: bool = False,
    fold_ids: Iterable[str] | None = None,
    max_test_pages: int | None = None,
    split_seed: int = DEFAULT_SPLIT_SEED,
    vlm_predictions_root: str | Path | None = None,
    allow_vlm_pagexml_drift: bool = False,
    input_usd_per_1m_tokens: float | None = None,
    output_usd_per_1m_tokens: float | None = None,
    gnn_finetuning_config: str | Path = DEFAULT_GNN_FINETUNING_CONFIG,
) -> dict:
    paths = default_manuscript_paths(manuscript_root)
    output_root = Path(output_root)
    gnn_finetuning_config = Path(gnn_finetuning_config)
    methods = tuple(method_by_id(method_id) for method_id in method_ids)
    vlm_methods = tuple(method for method in methods if is_vlm_method(method.method_id))
    if vlm_methods and vlm_predictions_root is None:
        raise ValueError("--vlm-predictions-root is required when any cached VLM method is selected.")
    cache_manifests = {
        method.method_id: validate_vlm_cache(
            paths=paths,
            cache_root=Path(vlm_predictions_root),
            method_id=method.method_id,
            allow_pagexml_drift=allow_vlm_pagexml_drift,
        )
        for method in vlm_methods
    }
    folds = select_folds(
        _load_or_create_run_folds(paths, output_root, split_seed=split_seed),
        fold_ids=fold_ids,
        max_test_pages=max_test_pages,
    )
    _write_experiment_reproducibility(
        output_root,
        gnn_finetuning_config=gnn_finetuning_config,
    )
    results = {}
    records_by_method = _run_methods_with_finetuning_ladder(
        paths=paths,
        folds=folds,
        methods=methods,
        output_root=output_root,
        write_diagnostics=write_diagnostics,
        vlm_predictions_root=Path(vlm_predictions_root) if vlm_predictions_root is not None else None,
        allow_vlm_pagexml_drift=allow_vlm_pagexml_drift,
        gnn_finetuning_config=gnn_finetuning_config,
    )
    for method in methods:
        all_records = records_by_method[method.method_id]
        payload = {
            "method": asdict(method),
            "ocr_active_learning_recipe": _ocr_recipe_metadata_for_method(method),
            "gnn_finetuning_recipe": _gnn_recipe_metadata_for_method(
                method,
                gnn_finetuning_config,
            ),
            "manuscript_id": paths.manuscript_id,
            "folds": [asdict(fold) for fold in folds],
            "textedit_metric": textedit_reproducibility_metadata(),
            "devanagari_textedit_metric": (
                devanagari_textedit_reproducibility_metadata()
            ),
            "aggregate": aggregate_page_records(all_records),
            "page_records": all_records,
        }
        if method.method_id in cache_manifests:
            cache_dir = (
                Path(vlm_predictions_root)
                / paths.manuscript_id
                / method.method_id
            )
            payload["preprediction_cache"] = {
                "cache_root": str(Path(vlm_predictions_root).resolve()),
                "manifest_path": str((cache_dir / "manifest.json").resolve()),
                "provider": cache_manifests[method.method_id]["provider"],
                "prompt_sha256": cache_manifests[method.method_id]["prompt_sha256"],
                "page_count": cache_manifests[method.method_id]["page_count"],
                "pagexml_drift_allowed": allow_vlm_pagexml_drift,
            }
        _write_json(output_root / "metrics" / method.method_id / "metrics.json", payload)
        _write_csv(output_root / "metrics" / method.method_id / "per_page.csv", all_records)
        results[method.method_id] = payload
    _write_report_artifacts(
        output_root,
        input_usd_per_1m_tokens=input_usd_per_1m_tokens,
        output_usd_per_1m_tokens=output_usd_per_1m_tokens,
    )
    return results


def _run_methods_with_finetuning_ladder(
    *,
    paths: ManuscriptPaths,
    folds: Iterable[Fold],
    methods: Iterable[MethodSpec],
    output_root: Path,
    write_diagnostics: bool = False,
    vlm_predictions_root: Path | None = None,
    allow_vlm_pagexml_drift: bool = False,
    gnn_finetuning_config: str | Path = DEFAULT_GNN_FINETUNING_CONFIG,
) -> dict[str, list[dict]]:
    method_list = tuple(methods)
    records_by_method: dict[str, list[dict]] = {method.method_id: [] for method in method_list}
    finetune_methods = tuple(method for method in method_list if method.uses_finetuning)
    ordinary_methods = tuple(method for method in method_list if not method.uses_finetuning)
    diagnostics_dir = output_root / "diagnostics" if write_diagnostics else None

    for fold in folds:
        needs_predicted_layout_pages = any(
            method.method_id == "annotation_tool_e2e"
            for method in method_list
        )
        predicted_layout_test_pages = None
        if needs_predicted_layout_pages:
            predicted_layout_test_pages = _prepare_annotation_tool_predicted_layout_pages(
                paths=paths,
                fold=fold,
                preparation_dir=(
                    output_root
                    / "runs"
                    / "annotation_tool_predicted_layout_shared"
                    / fold.fold_id
                ),
            )

        if finetune_methods:
            prediction_dirs = run_local_finetuning_ladder(
                paths=paths,
                fold=fold,
                methods=finetune_methods,
                output_root=output_root,
                gnn_finetuning_config=gnn_finetuning_config,
            )
            for method in finetune_methods:
                records_by_method[method.method_id].extend(
                    evaluate_prediction_folder(
                        paths=paths,
                        fold=fold,
                        method=method,
                        prediction_dir=prediction_dirs[method.method_id],
                        diagnostics_dir=diagnostics_dir,
                    )
                )

        for method in ordinary_methods:
            run_dir = output_root / "runs" / method.method_id / fold.fold_id
            if method.method_id == "annotation_tool_e2e":
                prediction_dir = run_annotation_tool_auto_layout(
                    paths=paths,
                    fold=fold,
                    method=method,
                    run_dir=run_dir,
                    prepared_pages=predicted_layout_test_pages,
                )
                statuses = None
            else:
                prediction_dir, statuses = run_method(
                    paths=paths,
                    fold=fold,
                    method=method,
                    run_dir=run_dir,
                    vlm_predictions_root=vlm_predictions_root,
                    allow_vlm_pagexml_drift=allow_vlm_pagexml_drift,
                )
            records_by_method[method.method_id].extend(
                evaluate_prediction_folder(
                    paths=paths,
                    fold=fold,
                    method=method,
                    prediction_dir=prediction_dir,
                    statuses=statuses,
                    diagnostics_dir=diagnostics_dir,
                )
            )
    return records_by_method


def _load_or_create_run_folds(
    paths: ManuscriptPaths,
    output_root: Path,
    *,
    split_seed: int = DEFAULT_SPLIT_SEED,
) -> tuple[Fold, ...]:
    page_ids = discover_page_ids(paths)
    return load_or_create_folds_json(
        output_root / "folds.json",
        manuscript_id=paths.manuscript_id,
        page_ids=page_ids,
        split_seed=split_seed,
    )


def select_folds(
    folds: Iterable[Fold],
    *,
    fold_ids: Iterable[str] | None = None,
    max_test_pages: int | None = None,
) -> tuple[Fold, ...]:
    requested = set(fold_ids or [])
    selected = [fold for fold in folds if not requested or fold.fold_id in requested]
    if requested:
        found = {fold.fold_id for fold in selected}
        missing = sorted(requested - found)
        if missing:
            raise ValueError(f"Unknown fold ids: {missing}")
    if max_test_pages is not None:
        limit = max(0, int(max_test_pages))
        selected = [
            Fold(
                fold_id=fold.fold_id,
                train_page_ids=fold.train_page_ids,
                test_page_ids=fold.test_page_ids[:limit],
            )
            for fold in selected
        ]
    return tuple(selected)


def load_gemini_api_key() -> str:
    load_dotenv(APP_ROOT / ".env")
    return (os.getenv("GEMINI_API_KEY") or "").strip()
