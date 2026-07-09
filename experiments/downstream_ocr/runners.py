from __future__ import annotations

import csv
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
from typing import Iterable

from dotenv import load_dotenv

from .adapter import AdapterError, VLM_END_TO_END_PROMPT, vlm_json_to_pagexml
from .diagnostics import write_page_diagnostics
from .metrics import aggregate_page_records, evaluate_page
from .pagexml import empty_page_like, local_name, load_pagexml, qualified, tag_namespace, write_pagexml
from .reporting import summarize_usage_metadata, usage_metadata_to_dict
from .reproducibility import write_reproducibility_manifest
from .splits import Fold, ManuscriptPaths, default_manuscript_paths, discover_page_ids, make_three_folds


REPO_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = REPO_ROOT / "app"
BASE_OCR_CHECKPOINT = APP_ROOT / "recognition" / "pretrained_model" / "vadakautuhala.pth"
PRETRAINED_GNN_MODEL = APP_ROOT / "pretrained_gnn" / "v2.pt"
PRETRAINED_GNN_CONFIG = APP_ROOT / "pretrained_gnn" / "gnn_preprocessing_v2.yaml"
DEFAULT_GEMINI_TIMEOUT_SECONDS = 45.0


def _write_experiment_reproducibility(output_root: Path) -> Path:
    return write_reproducibility_manifest(
        output_root,
        repo_root=REPO_ROOT,
        artifact_paths=(
            BASE_OCR_CHECKPOINT,
            PRETRAINED_GNN_MODEL,
            PRETRAINED_GNN_CONFIG,
        ),
    )


def _write_report_artifacts(output_root: Path) -> None:
    from .reporting import write_experiment_report

    write_experiment_report(output_root)


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


@dataclass(frozen=True)
class MethodSpec:
    method_id: str
    display_name: str
    uses_gt_layout: bool
    uses_finetuning: bool = False
    finetune_page_count: int = 0
    uses_gemini: bool = False


METHODS: tuple[MethodSpec, ...] = (
    MethodSpec("vlm_e2e", "VLM (End-to-End)", uses_gt_layout=False, uses_gemini=True),
    MethodSpec("gemini_gt_layout", "VLM (End-to-End with Graph Layout Grounding)", uses_gt_layout=True, uses_gemini=True),
    MethodSpec("annotation_tool_e2e", "Annotation tool (End-to-End)", uses_gt_layout=False),
    MethodSpec("annotation_tool_gt_layout", "Annotation tool (End-to-End with Graph Layout Grounding)", uses_gt_layout=True),
    MethodSpec("annotation_tool_gt_layout_ft_1", "Annotation tool (GT Layout, 1-page fine-tuning)", uses_gt_layout=True, uses_finetuning=True, finetune_page_count=1),
    MethodSpec("annotation_tool_gt_layout_ft_2", "Annotation tool (GT Layout, 2-page fine-tuning)", uses_gt_layout=True, uses_finetuning=True, finetune_page_count=2),
    MethodSpec("annotation_tool_gt_layout_ft_3", "Annotation tool (GT Layout, 3-page fine-tuning)", uses_gt_layout=True, uses_finetuning=True, finetune_page_count=3),
)


def _ensure_app_import_path() -> None:
    for path in (APP_ROOT, REPO_ROOT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def method_by_id(method_id: str) -> MethodSpec:
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


def run_annotation_tool_auto_layout(
    *,
    paths: ManuscriptPaths,
    fold: Fold,
    method: MethodSpec,
    run_dir: Path,
) -> Path:
    _ensure_app_import_path()
    from gnn_inference import generate_xml_and_images_for_page, run_gnn_prediction_for_page
    from recognition.active_learning import generate_prediction_pagexmls

    manuscript_root = _copy_raw_manuscript_for_auto_layout(paths, run_dir / "manuscript")
    line_segmentation_args = _load_manuscript_line_segmentation_args(manuscript_root)

    for page_id in fold.test_page_ids:
        graph = run_gnn_prediction_for_page(
            str(manuscript_root),
            page_id,
            str(PRETRAINED_GNN_MODEL),
            str(PRETRAINED_GNN_CONFIG),
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
            source_xml_dir=run_dir / "ocr_source_page_xml",
            prepared_root=run_dir / "ocr_prepared_pages",
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


def _failure_status_from_exception(exc: Exception) -> str:
    name = exc.__class__.__name__.lower()
    message = str(exc).lower()
    if "timeout" in name or "timed out" in message or "deadline exceeded" in message:
        return "api_timeout"
    if isinstance(exc, AdapterError):
        value = str(exc)
        return value if value in {"empty_response", "json_parse_error", "json_schema_error", "adapter_error", "other_output_error"} else "adapter_error"
    return "api_error"


def _write_empty_prediction_for_status(paths: ManuscriptPaths, page_id: str, prediction_dir: Path) -> None:
    adapt_failed_prediction(paths.pagexml_dir / f"{page_id}.xml", prediction_dir / f"{page_id}.xml")


def run_vlm_end_to_end_gemini(
    *,
    paths: ManuscriptPaths,
    fold: Fold,
    method: MethodSpec,
    run_dir: Path,
) -> tuple[Path, dict[str, str]]:
    from google import genai
    from google.genai import types

    api_key = load_gemini_api_key()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured in app/.env or the environment.")

    timeout_ms = int(_gemini_timeout_seconds() * 1000)
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=timeout_ms,
            retryOptions=types.HttpRetryOptions(attempts=1),
        ),
    )
    raw_dir = run_dir / "raw_gemini_json"
    prediction_dir = run_dir / "prediction_page_xml"
    usage_dir = run_dir / "gemini_usage"
    raw_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    statuses: dict[str, str] = {}

    for page_id in fold.test_page_ids:
        image_path = _find_page_image(page_id, [paths.images_dir])
        template_page = load_pagexml(paths.pagexml_dir / f"{page_id}.xml", repair_geometry=True)
        started = time.perf_counter()
        try:
            from PIL import Image

            with Image.open(image_path) as image:
                response = client.models.generate_content(
                    model="gemini-3.5-flash",
                    contents=[image, VLM_END_TO_END_PROMPT],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.2,
                    ),
                )
            raw_text = response.text or ""
            (raw_dir / f"{page_id}.json").write_text(raw_text, encoding="utf-8")
            vlm_json_to_pagexml(
                raw_text,
                template_page=template_page,
                output_path=prediction_dir / f"{page_id}.xml",
            )
            statuses[page_id] = "success"
            usage = getattr(response, "usage_metadata", None)
            _write_json(
                usage_dir / f"{page_id}.json",
                {
                    "page_id": page_id,
                    "status": statuses[page_id],
                    "model": "gemini-3.5-flash",
                    "elapsed_seconds": time.perf_counter() - started,
                    "usage_metadata": usage_metadata_to_dict(usage),
                },
            )
        except Exception as exc:
            statuses[page_id] = _failure_status_from_exception(exc)
            _write_empty_prediction_for_status(paths, page_id, prediction_dir)
            _write_json(
                usage_dir / f"{page_id}.json",
                {
                    "page_id": page_id,
                    "elapsed_seconds": time.perf_counter() - started,
                    "status": statuses[page_id],
                    "error": str(exc),
                },
            )
    return prediction_dir, statuses


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


def run_layout_grounded_gemini_with_app_copy(
    *,
    paths: ManuscriptPaths,
    fold: Fold,
    method: MethodSpec,
    run_dir: Path,
) -> tuple[Path, dict[str, str]]:
    load_gemini_api_key()
    _ensure_app_import_path()
    import app as backend_app_module

    upload_root = run_dir / "upload_root"
    manuscript_root = _copy_gt_layout_manuscript_for_gemini(paths, fold, upload_root / paths.manuscript_id)
    previous_upload_folder = backend_app_module.UPLOAD_FOLDER
    real_gemini_client_cls = backend_app_module.genai.Client
    backend_app_module.UPLOAD_FOLDER = str(upload_root)
    statuses: dict[str, str] = {}
    usage_dir = run_dir / "gemini_usage"
    try:
        for page_id in fold.test_page_ids:
            started = time.perf_counter()
            usage_records: list[dict] = []
            usage_lock = threading.Lock()
            backend_app_module.genai.Client = _recording_gemini_client_factory(
                real_gemini_client_cls,
                usage_records,
                usage_lock,
            )
            try:
                result = backend_app_module._run_gemini_recognition_internal(paths.manuscript_id, page_id)
                if result.get("error"):
                    status = "api_timeout" if result.get("errorCode") == "gemini_timeout" else "api_error"
                    statuses[page_id] = status
                    _write_empty_prediction_for_status(
                        ManuscriptPaths(
                            manuscript_id=paths.manuscript_id,
                            root=manuscript_root,
                            images_dir=manuscript_root / "images_resized",
                            pagexml_dir=manuscript_root / "layout_analysis_output" / "page-xml-format",
                            line_images_dir=manuscript_root / "layout_analysis_output" / "image-format",
                            heatmaps_dir=None,
                        ),
                        page_id,
                        manuscript_root / "layout_analysis_output" / "page-xml-format",
                    )
                else:
                    statuses[page_id] = "success"
                _write_json(
                    usage_dir / f"{page_id}.json",
                    {
                        "page_id": page_id,
                        "elapsed_seconds": time.perf_counter() - started,
                        "status": statuses[page_id],
                        "model": "gemini-3.5-flash",
                        "usage_records": usage_records,
                        "usage_metadata": summarize_usage_metadata(usage_records),
                    },
                )
            except Exception as exc:
                statuses[page_id] = _failure_status_from_exception(exc)
                _write_empty_prediction_for_status(
                    ManuscriptPaths(
                        manuscript_id=paths.manuscript_id,
                        root=manuscript_root,
                        images_dir=manuscript_root / "images_resized",
                        pagexml_dir=manuscript_root / "layout_analysis_output" / "page-xml-format",
                        line_images_dir=manuscript_root / "layout_analysis_output" / "image-format",
                        heatmaps_dir=None,
                    ),
                    page_id,
                    manuscript_root / "layout_analysis_output" / "page-xml-format",
                )
                _write_json(
                    usage_dir / f"{page_id}.json",
                    {
                        "page_id": page_id,
                        "elapsed_seconds": time.perf_counter() - started,
                        "status": statuses[page_id],
                        "error": str(exc),
                        "model": "gemini-3.5-flash",
                        "usage_records": usage_records,
                        "usage_metadata": summarize_usage_metadata(usage_records),
                    },
                )
            finally:
                backend_app_module.genai.Client = real_gemini_client_cls
    finally:
        backend_app_module.genai.Client = real_gemini_client_cls
        backend_app_module.UPLOAD_FOLDER = previous_upload_folder
    return manuscript_root / "layout_analysis_output" / "page-xml-format", statuses


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
    layout_effort_by_page = _load_layout_effort_by_page(paths.root)
    for page_id in fold.test_page_ids:
        gt_page = load_pagexml(paths.pagexml_dir / f"{page_id}.xml", repair_geometry=True)
        pred_path = prediction_dir / f"{page_id}.xml"
        status = (statuses or {}).get(page_id, "success")
        if not pred_path.exists():
            pred_page = empty_page_like(gt_page)
            status = "empty_response"
        else:
            pred_page = load_pagexml(pred_path, strict=True, repair_geometry=True)
        record = evaluate_page(
            manuscript_id=paths.manuscript_id,
            fold_id=fold.fold_id,
            page_id=page_id,
            method_id=method.method_id,
            gt_page=gt_page,
            pred_page=pred_page,
            status=status,
        )
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
    return records


def evaluate_existing_prediction_tree(
    *,
    manuscript_root: str | Path,
    predictions_root: str | Path,
    method_id: str,
    output_root: str | Path,
    write_diagnostics: bool = False,
) -> dict:
    paths = default_manuscript_paths(manuscript_root)
    method = method_by_id(method_id)
    page_ids = discover_page_ids(paths)
    folds = make_three_folds(page_ids)
    all_records = []
    predictions_root = Path(predictions_root)
    output_root = Path(output_root)
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
        "aggregate": aggregate,
        "page_records": all_records,
    }
    _write_json(output_root / method_id / "metrics.json", payload)
    _write_csv(output_root / method_id / "per_page.csv", all_records)
    _write_report_artifacts(output_root)
    return payload


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
    method_id: str = "vlm_e2e",
    write_diagnostics: bool = False,
    fold_ids: Iterable[str] | None = None,
    max_test_pages: int | None = None,
) -> dict:
    paths = default_manuscript_paths(manuscript_root)
    method = method_by_id(method_id)
    page_ids = discover_page_ids(paths)
    folds = select_folds(make_three_folds(page_ids), fold_ids=fold_ids, max_test_pages=max_test_pages)
    json_root = Path(json_root)
    output_root = Path(output_root)
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
) -> dict:
    paths = default_manuscript_paths(manuscript_root)
    page_ids = discover_page_ids(paths)
    folds = select_folds(make_three_folds(page_ids), fold_ids=fold_ids, max_test_pages=max_test_pages)
    output_root = Path(output_root)
    _write_experiment_reproducibility(output_root)
    results = {}
    for method_id in method_ids:
        method = method_by_id(method_id)
        all_records = []
        for fold in folds:
            run_dir = output_root / "runs" / method.method_id / fold.fold_id
            prediction_dir = run_local_ocr_with_gt_layout(
                paths=paths,
                fold=fold,
                method=method,
                run_dir=run_dir,
            )
            fold_records = evaluate_prediction_folder(
                paths=paths,
                fold=fold,
                method=method,
                prediction_dir=prediction_dir,
                diagnostics_dir=(output_root / "diagnostics") if write_diagnostics else None,
            )
            all_records.extend(fold_records)
        payload = {
            "method": asdict(method),
            "ocr_active_learning_recipe": _ocr_recipe_metadata_for_method(method),
            "manuscript_id": paths.manuscript_id,
            "folds": [asdict(fold) for fold in folds],
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
) -> tuple[Path, dict[str, str] | None]:
    if method.method_id == "vlm_e2e":
        return run_vlm_end_to_end_gemini(paths=paths, fold=fold, method=method, run_dir=run_dir)
    if method.method_id == "gemini_gt_layout":
        return run_layout_grounded_gemini_with_app_copy(paths=paths, fold=fold, method=method, run_dir=run_dir)
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
) -> dict:
    paths = default_manuscript_paths(manuscript_root)
    page_ids = discover_page_ids(paths)
    folds = select_folds(make_three_folds(page_ids), fold_ids=fold_ids, max_test_pages=max_test_pages)
    output_root = Path(output_root)
    _write_experiment_reproducibility(output_root)
    results = {}
    for method_id in method_ids:
        method = method_by_id(method_id)
        all_records = []
        for fold in folds:
            run_dir = output_root / "runs" / method.method_id / fold.fold_id
            prediction_dir, statuses = run_method(
                paths=paths,
                fold=fold,
                method=method,
                run_dir=run_dir,
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
            "ocr_active_learning_recipe": _ocr_recipe_metadata_for_method(method),
            "manuscript_id": paths.manuscript_id,
            "folds": [asdict(fold) for fold in folds],
            "aggregate": aggregate_page_records(all_records),
            "page_records": all_records,
        }
        _write_json(output_root / "metrics" / method.method_id / "metrics.json", payload)
        _write_csv(output_root / "metrics" / method.method_id / "per_page.csv", all_records)
        results[method.method_id] = payload
    _write_report_artifacts(output_root)
    return results


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
