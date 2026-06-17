from __future__ import annotations

import json
import os
import shutil
import sys
import unittest
import xml.etree.ElementTree as ET
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path


TESTS_ROOT = Path(__file__).resolve().parent
APP_ROOT = TESTS_ROOT.parent
REPO_ROOT = APP_ROOT.parent
LOGS_ROOT = TESTS_ROOT / "logs"

if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recognition.active_learning import generate_prediction_pagexmls
from recognition.pagexml_line_dataset import GEOMETRY_SOURCE_BASELINE_HEATMAP, prepare_page_line_dataset
from tests.backend_app_import import backend_app_module
from tests.evaluate import evaluate_dataset, write_report_files
from tests.precommit_gate_config import get_pipeline_precommit_dataset


PIPELINE_OCR_WIDTH_POLICY = "batch_max_pad"
PIPELINE_OCR_SEGMENTATION_ARGS = {
    "BINARIZE_THRESHOLD": 0.5098,
    "BBOX_PAD_V": 0.7,
    "BBOX_PAD_H": 0.5,
    "CC_SIZE_THRESHOLD_RATIO": 0.4,
}


def _timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _count_pagexml_text_lines_with_text(xml_path: Path) -> int:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    def strip_namespace(tag):
        return tag.split("}", 1)[-1] if "}" in tag else tag

    count = 0
    for textline in root.iter():
        if strip_namespace(textline.tag) != "TextLine":
            continue
        for text_equiv in textline:
            if strip_namespace(text_equiv.tag) != "TextEquiv":
                continue
            for child in text_equiv:
                if strip_namespace(child.tag) == "Unicode" and child.text and child.text.strip():
                    count += 1
                    break
    return count


def _write_placeholder_text_pagexml(source_xml_path: Path, target_xml_path: Path) -> None:
    tree = ET.parse(source_xml_path)
    root = tree.getroot()

    def strip_namespace(tag):
        return tag.split("}", 1)[-1] if "}" in tag else tag

    def tag_namespace(tag):
        return tag.split("}", 1)[0].strip("{") if tag.startswith("{") else ""

    def qualified(tag, namespace):
        return f"{{{namespace}}}{tag}" if namespace else tag

    text_line_count = 0
    for textline in root.iter():
        if strip_namespace(textline.tag) != "TextLine":
            continue

        textline_namespace = tag_namespace(textline.tag)
        for child in list(textline):
            if strip_namespace(child.tag) == "TextEquiv":
                textline.remove(child)

        text_equiv = ET.SubElement(textline, qualified("TextEquiv", textline_namespace))
        unicode_elem = ET.SubElement(text_equiv, qualified("Unicode", textline_namespace))
        unicode_elem.text = "x"
        text_line_count += 1

    if text_line_count <= 0:
        raise AssertionError(f"No TextLine elements found in {source_xml_path}")
    target_xml_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(target_xml_path, encoding="UTF-8", xml_declaration=True)


def _find_page_image(page: str, candidate_dirs) -> Path | None:
    extensions = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG", ".TIF", ".TIFF")
    for directory in candidate_dirs:
        for extension in extensions:
            candidate = Path(directory) / f"{page}{extension}"
            if candidate.exists():
                return candidate
    return None


def _upload_dataset(client, dataset_config, manuscript_name: str):
    image_paths = sorted(dataset_config.images_dir.glob("*.jpg"))

    with ExitStack() as stack:
        data = {
            "manuscriptName": manuscript_name,
            "longestSide": str(dataset_config.longest_side),
            "minDistance": str(dataset_config.min_distance),
            "images": [(stack.enter_context(open(path, "rb")), path.name) for path in image_paths],
        }
        return client.post("/upload", data=data, content_type="multipart/form-data")


def _prepare_pretrained_gate_ocr_page(
    *,
    manuscript_root: Path,
    page: str,
    source_xml_dir: Path,
    prepared_root: Path,
    strategy_name: str,
    strategy_config: dict,
):
    xml_path = manuscript_root / "layout_analysis_output" / "page-xml-format" / f"{page}.xml"
    image_path = _find_page_image(
        page,
        [
            manuscript_root / "layout_analysis_output" / "images_resized",
            manuscript_root / "images_resized",
        ],
    )
    heatmap_path = _find_page_image(page, [manuscript_root / "heatmaps"])
    source_xml_path = source_xml_dir / f"{page}.xml"

    if not xml_path.exists():
        raise AssertionError(f"Missing generated PAGE-XML for OCR prep: {xml_path}")
    if image_path is None:
        raise AssertionError(f"Missing resized page image for OCR prep: {page}")
    if heatmap_path is None:
        raise AssertionError(f"Missing heatmap for OCR prep: {page}")

    _write_placeholder_text_pagexml(xml_path, source_xml_path)
    return prepare_page_line_dataset(
        source_xml_path,
        image_path,
        prepared_root / page,
        heatmap_path=heatmap_path,
        geometry_source=GEOMETRY_SOURCE_BASELINE_HEATMAP,
        segmentation_args=dict(strategy_config or PIPELINE_OCR_SEGMENTATION_ARGS),
        line_segmentation_strategy_name=strategy_name,
    )


def _threshold_result(metric_name: str, observed, operator: str, threshold: float) -> dict:
    if operator == "<=":
        passed = observed is not None and observed <= threshold
    else:
        raise ValueError(f"Unsupported threshold operator: {operator}")
    return {
        "metric_name": metric_name,
        "observed": observed,
        "operator": operator,
        "threshold": threshold,
        "passed": passed,
    }


def _build_absolute_thresholds(dataset_config, aggregate: dict) -> dict:
    return {
        "page_cer": _threshold_result("page_cer", aggregate["page_cer"], "<=", dataset_config.max_page_cer),
    }


def _run_pipeline_role(client, dataset_config, role_config, upload_root: Path) -> dict:
    if not role_config.strategy_name:
        raise ValueError(
            f"Pipeline ablation role {role_config.role!r} requires proposed_strategy_name to be configured."
        )
    expected_pages = dataset_config.ordered_page_ids()
    timestamp = _timestamp_slug()
    manuscript_name = f"{dataset_config.manuscript_name}_{role_config.role}"
    manuscript_root = upload_root / manuscript_name
    role_run_dir = LOGS_ROOT / f"{timestamp}_pipeline_{role_config.role}_{dataset_config.name}"
    pred_folder = manuscript_root / "layout_analysis_output" / "page-xml-format"
    ocr_source_xml_dir = role_run_dir / "ocr_source_page_xml"
    ocr_prepared_dir = role_run_dir / "ocr_prepared_pages"
    ocr_prediction_dir = role_run_dir / "ocr_prediction_page_xml"

    if manuscript_root.exists():
        shutil.rmtree(manuscript_root)
    role_run_dir.mkdir(parents=True, exist_ok=True)
    ocr_source_xml_dir.mkdir(parents=True, exist_ok=True)

    upload_response = _upload_dataset(client, dataset_config, manuscript_name)
    upload_json = upload_response.get_json()
    if upload_response.status_code != 200:
        raise AssertionError(upload_json)
    if sorted(upload_json["pages"]) != expected_pages:
        raise AssertionError(f"Uploaded pages did not match expected pages for {dataset_config.name}.")

    pages_response = client.get(f"/manuscript/{manuscript_name}/pages")
    pages_json = pages_response.get_json()
    if pages_response.status_code != 200:
        raise AssertionError(pages_json)
    if sorted(pages_json["pages"]) != expected_pages:
        raise AssertionError(f"Listed pages did not match expected pages for {dataset_config.name}.")

    for page in upload_json["pages"]:
        graph_response = client.get(f"/semi-segment/{manuscript_name}/{page}")
        graph_json = graph_response.get_json()
        if graph_response.status_code != 200:
            raise AssertionError(graph_json)
        if len(graph_json["graph"]["nodes"]) <= 0 or len(graph_json["graph"]["edges"]) <= 0:
            raise AssertionError(f"No graph nodes or edges returned for page {page}")

        node_count = len(graph_json["graph"]["nodes"])
        save_payload = {
            "graph": graph_json["graph"],
            "modifications": [],
            "textlineLabels": [-1] * node_count,
            "textboxLabels": [0] * node_count,
            "textContent": {},
            "runRecognition": False,
            "recognitionEngine": "local",
        }
        save_response = client.post(f"/semi-segment/{manuscript_name}/{page}", json=save_payload)
        save_json = save_response.get_json()
        if save_response.status_code != 200 or save_json["status"] != "success":
            raise AssertionError(save_json)
        if save_json["lines"] <= 0:
            raise AssertionError(f"No text lines generated for page {page}")

    prepared_pages = {}
    for page in upload_json["pages"]:
        prepared_page = _prepare_pretrained_gate_ocr_page(
            manuscript_root=manuscript_root,
            page=page,
            source_xml_dir=ocr_source_xml_dir,
            prepared_root=ocr_prepared_dir,
            strategy_name=role_config.strategy_name,
            strategy_config=role_config.strategy_config,
        )
        prepared_pages[page] = prepared_page
        if len(prepared_page.records) <= 0:
            raise AssertionError(f"No OCR crop records prepared for page {page}")

    prediction_output = generate_prediction_pagexmls(
        backend_app_module.OCR_MODEL_PATH,
        prepared_pages,
        ocr_prediction_dir,
        width_policy=PIPELINE_OCR_WIDTH_POLICY,
    )
    prediction_folder = Path(prediction_output.prediction_folder)
    if len(list(prediction_folder.glob("*.xml"))) != len(expected_pages):
        raise AssertionError("OCR prediction page count did not match expected page count.")

    for predicted_xml in sorted(prediction_folder.glob("*.xml")):
        shutil.copy2(predicted_xml, pred_folder / predicted_xml.name)
        if _count_pagexml_text_lines_with_text(pred_folder / predicted_xml.name) <= 0:
            raise AssertionError(f"No OCR text returned for page {predicted_xml.stem}")

    result = evaluate_dataset(
        pred_folder=pred_folder,
        gt_folder=dataset_config.pagexml_dir,
        method_name=f"Pipeline ablation {role_config.role} ({dataset_config.name})",
        layout_type=dataset_config.layout_type,
    )
    aggregate = result["aggregate_metrics"]
    threshold_results = _build_absolute_thresholds(dataset_config, aggregate)
    passed = (
        result["files_processed"] == len(expected_pages)
        and all(page["prediction_found"] for page in result["per_page"])
        and all(item["passed"] for item in threshold_results.values())
    )
    failure_message = ""
    if not passed:
        failed = [
            f"{item['metric_name']}={item['observed']} must be {item['operator']} {item['threshold']}"
            for item in threshold_results.values()
            if not item["passed"]
        ]
        failure_message = "Pipeline role failed: " + "; ".join(failed)

    summary_path = role_run_dir / "summary.txt"
    metrics_path = role_run_dir / "metrics.json"
    write_report_files(result, text_path=summary_path, json_path=metrics_path)
    return {
        "role": role_config.role,
        "strategy_name": role_config.strategy_name,
        "strategy_config": dict(role_config.strategy_config),
        "run_dir": role_run_dir,
        "status": "passed" if passed else "failed",
        "passed": passed,
        "failure_message": failure_message,
        "metrics": aggregate,
        "threshold_results": threshold_results,
        "summary_path": summary_path,
        "metrics_path": metrics_path,
    }


def _build_pipeline_comparison(dataset_config, benchmark_result: dict, proposed_result: dict) -> dict:
    allowed = float(dataset_config.strategy_ablation.max_allowed_regression_abs)
    comparisons = []
    metric_name = "page_cer"
    benchmark_value = benchmark_result["metrics"].get(metric_name)
    proposed_value = proposed_result["metrics"].get(metric_name)
    allowed_value = None if benchmark_value is None else benchmark_value + allowed
    passed_page_cer = proposed_value is not None and allowed_value is not None and proposed_value <= allowed_value
    comparisons.append(
        {
            "metric_name": metric_name,
            "benchmark_value": benchmark_value,
            "proposed_value": proposed_value,
            "operator": "<=",
            "allowed_regression_abs": allowed,
            "allowed_value": allowed_value,
            "passed": passed_page_cer,
        }
    )

    passed = (
        benchmark_result["passed"]
        and proposed_result["passed"]
        and all(comparison["passed"] for comparison in comparisons)
    )
    failure_message = ""
    if not benchmark_result["passed"]:
        failure_message = f"Benchmark role failed: {benchmark_result['failure_message']}"
    elif not proposed_result["passed"]:
        failure_message = f"Proposed role failed: {proposed_result['failure_message']}"
    elif not passed:
        failed = [
            f"{item['metric_name']}: benchmark={item['benchmark_value']} proposed={item['proposed_value']} "
            f"must be <= {item['allowed_value']}"
            for item in comparisons
            if not item["passed"]
        ]
        failure_message = "Pipeline strategy ablation failed: " + "; ".join(failed)

    return {
        "benchmark_role": benchmark_result["role"],
        "proposed_role": proposed_result["role"],
        "primary_metric_name": "page_cer",
        "benchmark_value": benchmark_result["metrics"].get("page_cer"),
        "proposed_value": proposed_result["metrics"].get("page_cer"),
        "operator": "<=",
        "allowed_regression_abs": allowed,
        "passed": passed,
        "failure_message": failure_message,
        "metric_comparisons": comparisons,
    }


def _jsonable_role_result(role_result: dict) -> dict:
    return {
        **{key: value for key, value in role_result.items() if key not in {"run_dir", "summary_path", "metrics_path"}},
        "run_dir": str(role_result["run_dir"].resolve()),
        "summary_path": str(role_result["summary_path"].resolve()),
        "metrics_path": str(role_result["metrics_path"].resolve()),
    }


def _write_pipeline_ablation_summary(path: Path, dataset_result: dict) -> None:
    lines = [
        f"# Pipeline Strategy Ablation Gate: {dataset_result['dataset_name']}",
        "",
        f"Status: **{dataset_result['status'].upper()}**",
        "",
        "## Strategy Roles",
        "",
    ]
    for role_name in ("benchmark", "proposed"):
        role_result = dataset_result["strategy_results"][role_name]
        metrics = role_result["metrics"]
        lines.append(
            f"- {role_name}: strategy={role_result['strategy_name']}, "
            f"status={role_result['status']}, "
            f"page_cer={metrics.get('page_cer')}, "
            f"line_cer_50={metrics.get('line_cer_50')}, "
            f"line_cer_75={metrics.get('line_cer_75')}, "
            f"line_cer_range={metrics.get('line_cer_range')}"
        )

    lines.extend(["", "## Comparison", ""])
    for item in dataset_result["comparison"]["metric_comparisons"]:
        lines.append(
            f"- {item['metric_name']}: benchmark={item['benchmark_value']}, "
            f"proposed={item['proposed_value']}, required proposed <= {item['allowed_value']}, "
            f"passed={item['passed']}"
        )
    if dataset_result["failure_message"]:
        lines.extend(["", "## Failure", "", dataset_result["failure_message"]])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_pipeline_strategy_ablation_gate(dataset_name: str = "eval_dataset") -> dict:
    dataset_config = get_pipeline_precommit_dataset(dataset_name)
    expected_pages = dataset_config.ordered_page_ids()
    if not dataset_config.images_dir.exists():
        raise AssertionError(f"Missing eval images directory: {dataset_config.images_dir}")
    if not dataset_config.pagexml_dir.exists():
        raise AssertionError(f"Missing eval PAGE-XML directory: {dataset_config.pagexml_dir}")
    if len(expected_pages) != dataset_config.expected_page_count:
        raise AssertionError(f"Expected {dataset_config.expected_page_count} images for {dataset_config.name}.")
    original_upload_folder = backend_app_module.UPLOAD_FOLDER
    original_model_checkpoint = backend_app_module.MODEL_CHECKPOINT
    original_dataset_config = backend_app_module.DATASET_CONFIG
    original_ocr_model_path = backend_app_module.OCR_MODEL_PATH
    original_ocr_global_context = backend_app_module.OCR_GLOBAL_CONTEXT

    upload_root = APP_ROOT / "input_manuscripts" / "_ci_root"
    if upload_root.exists():
        shutil.rmtree(upload_root)
    upload_root.mkdir(parents=True, exist_ok=True)

    backend_app_module.UPLOAD_FOLDER = str(upload_root)
    backend_app_module.MODEL_CHECKPOINT = str(APP_ROOT / "pretrained_gnn" / "v2.pt")
    backend_app_module.DATASET_CONFIG = str(APP_ROOT / "pretrained_gnn" / "gnn_preprocessing_v2.yaml")
    backend_app_module.OCR_MODEL_PATH = str(APP_ROOT / "recognition" / "pretrained_model" / "vadakautuhala.pth")
    backend_app_module.OCR_GLOBAL_CONTEXT = None
    backend_app_module.app.config["TESTING"] = True

    if not Path(backend_app_module.MODEL_CHECKPOINT).exists():
        raise AssertionError("Missing pretrained GNN checkpoint.")
    if not Path(backend_app_module.DATASET_CONFIG).exists():
        raise AssertionError("Missing GNN preprocessing config.")
    if not Path(backend_app_module.OCR_MODEL_PATH).exists():
        raise AssertionError("Missing local OCR model.")

    try:
        client = backend_app_module.app.test_client()
        strategy_results = {}
        for role_config in dataset_config.strategy_ablation.roles():
            strategy_results[role_config.role] = _run_pipeline_role(client, dataset_config, role_config, upload_root)

        comparison = _build_pipeline_comparison(
            dataset_config,
            strategy_results["benchmark"],
            strategy_results["proposed"],
        )
        status = "passed" if comparison["passed"] else "failed"
        run_dir = LOGS_ROOT / f"{_timestamp_slug()}_pipeline_ablation_{dataset_name}_summary"
        summary_path = run_dir / "summary.md"
        metrics_path = run_dir / "metrics.json"
        run_dir.mkdir(parents=True, exist_ok=True)
        dataset_result = {
            "study_mode": "pipeline_strategy_ablation_gate",
            "dataset_name": dataset_name,
            "dataset_config": dataset_config.to_dict(),
            "run_dir": run_dir,
            "summary_path": summary_path,
            "metrics_path": metrics_path,
            "status": status,
            "passed": comparison["passed"],
            "failure_message": comparison["failure_message"],
            "strategy_results": strategy_results,
            "comparison": comparison,
        }
        _write_pipeline_ablation_summary(summary_path, dataset_result)
        jsonable = {
            **{key: value for key, value in dataset_result.items() if key not in {"run_dir", "summary_path", "metrics_path", "strategy_results"}},
            "run_dir": str(run_dir.resolve()),
            "summary_path": str(summary_path.resolve()),
            "metrics_path": str(metrics_path.resolve()),
            "strategy_results": {
                role: _jsonable_role_result(role_result)
                for role, role_result in strategy_results.items()
            },
        }
        _write_json(
            metrics_path,
            {
                "study_mode": "pipeline_strategy_ablation_gate",
                "run_dir": str(run_dir.resolve()),
                "dataset_results": {dataset_name: jsonable},
                "failed_datasets": [] if comparison["passed"] else [dataset_name],
                "passed_dataset_count": 1 if comparison["passed"] else 0,
            },
        )
        shutil.copy2(summary_path, LOGS_ROOT / f"{dataset_config.latest_artifact_basename}.md")
        shutil.copy2(metrics_path, LOGS_ROOT / f"{dataset_config.latest_artifact_basename}.json")
        return dataset_result
    finally:
        backend_app_module.UPLOAD_FOLDER = original_upload_folder
        backend_app_module.MODEL_CHECKPOINT = original_model_checkpoint
        backend_app_module.DATASET_CONFIG = original_dataset_config
        backend_app_module.OCR_MODEL_PATH = original_ocr_model_path
        backend_app_module.OCR_GLOBAL_CONTEXT = original_ocr_global_context
        if os.getenv("KEEP_CI_ARTIFACTS") != "1" and upload_root.exists():
            shutil.rmtree(upload_root)


__all__ = ["run_pipeline_strategy_ablation_gate"]
