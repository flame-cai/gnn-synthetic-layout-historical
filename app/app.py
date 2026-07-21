# app.py
import os
import sys
import torch
import hashlib

# --- NEW IMPORTS FOR LOCAL OCR ---
# Ensure we can import from the recognition folder
sys.path.append(os.path.join(os.path.dirname(__file__), 'recognition'))

try:
    from recognition.recognize_manuscript_text_v2_pretrained import (
        process_page_xml, 
        load_ocr_model, 
        get_model_config
    )
except ImportError:
    print("Warning: Could not import local recognition modules. Ensure 'recognition' folder exists.")

# Global variable to hold the loaded model so we don't reload it every request
OCR_GLOBAL_CONTEXT = None
OCR_MODEL_PATH = "./recognition/pretrained_model/vadakautuhala.pth" # Adjust path if necessary


import threading 
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

import shutil
from pathlib import Path
import base64
import json
import zipfile
import io
import time
from google import genai
import glob
import re

import xml.etree.ElementTree as ET
import numpy as np
from os.path import isdir, join
import collections
import math
import difflib
from dotenv import load_dotenv
import concurrent.futures
load_dotenv() 
import traceback
from PIL import Image, ImageDraw, ImageOps
from urllib.parse import quote

from google.api_core import retry
from google.genai import types


# from recognition.recognize_manuscript_text import recognize_manuscript_text
# cd recognition
# python recognize_manuscript_text.py complex_layout



# Import your existing pipelines
from inference import process_new_manuscript
from gnn_inference import run_gnn_prediction_for_page, generate_xml_and_images_for_page
from recognition.line_segmentation.reading_direction import (
    default_reading_direction_metadata_path,
    load_reading_direction_annotations_by_line_id,
    load_reading_direction_metadata,
)
from recognition.line_segmentation.ocr_crops import (
    default_line_segmentation_metadata_path,
    load_line_segmentation_metadata_by_numeric_id,
)
from recognition.line_segmentation.geometry import normalize_baseline_topology
from recognition.pagexml_line_dataset import load_pagexml_lines
from recognition.auto_orientation import (
    ROTATE_180_TRANSFORM,
    auto_orientation_custom_metadata,
    auto_orientation_transform_from_custom,
    has_explicit_reading_direction_annotation,
)
from segmentation.utils import load_images_from_folder
from job_orchestrator import JobOrchestrator
from ocr_active_learning_runtime import (
    configure_runtime,
    handle_post_save,
    prepare_for_interactive_ocr,
    record_prediction,
    summarize_page_active_learning,
    summarize_manuscript_active_learning,
)
from ocr_model_manager import ManuscriptAwareOcrModelManager
from layout_effort_logging import record_layout_effort_save, utc_now_iso as layout_effort_utc_now_iso
from text_recovery import (
    backup_page_xml_for_text_recovery,
    build_latest_text_recovery_plan,
    build_text_recovery_state,
)
from pipeline_visualization import (
    default_enabled as pipeline_visualization_default_enabled,
    enqueue as enqueue_pipeline_visualization,
    layout_artifacts as save_layout_visualizations,
    ocr_prediction_artifacts as save_ocr_prediction_visualizations,
    read_mode_pagexml_artifact as save_read_mode_pagexml_artifact,
    upload_artifacts as save_upload_visualizations,
)

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = './input_manuscripts'
MODEL_CHECKPOINT = "./pretrained_gnn/v2.pt"
DATASET_CONFIG = "./pretrained_gnn/gnn_preprocessing_v2.yaml"
OCR_MODEL_MANAGER = ManuscriptAwareOcrModelManager()
JOB_ORCHESTRATOR = JobOrchestrator()
SUPPORTED_RECOGNITION_ENGINES = {"local", "gemini"}
MANUSCRIPT_PROCESSING_SETTINGS_FILENAME = "processing_settings.json"
BINARIZE_THRESHOLD_OVERRIDE_KEY = "BINARIZE_THRESHOLD"
DEFAULT_GEMINI_OCR_TIMEOUT_SECONDS = 45.0


def _coerce_optional_binarize_threshold(value, field_name="binarizationThreshold"):
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return None
    try:
        threshold = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field_name} must be a number between 0 and 1.")
    if not math.isfinite(threshold) or threshold < 0.0 or threshold > 1.0:
        raise ValueError(f"{field_name} must be a number between 0 and 1.")
    return threshold


def _manuscript_processing_settings_path(manuscript_path):
    return Path(manuscript_path) / MANUSCRIPT_PROCESSING_SETTINGS_FILENAME


def _write_manuscript_processing_settings(
    manuscript_path,
    *,
    target_longest_side,
    min_distance,
    binarize_threshold=None,
    pipeline_visualization_enabled=None,
):
    line_segmentation_args = {}
    if binarize_threshold is not None:
        line_segmentation_args[BINARIZE_THRESHOLD_OVERRIDE_KEY] = binarize_threshold

    settings = {
        "target_longest_side": int(target_longest_side),
        "min_distance": int(min_distance),
        "line_segmentation_args": line_segmentation_args,
        "pipeline_visualization": {
            "enabled": (
                pipeline_visualization_default_enabled()
                if pipeline_visualization_enabled is None
                else str(pipeline_visualization_enabled).strip().lower()
                in {"1", "true", "yes", "on", "enabled"}
            ),
            "max_line_previews": 24,
        },
    }
    settings_path = _manuscript_processing_settings_path(manuscript_path)
    settings_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    return settings


def _load_manuscript_line_segmentation_args(manuscript_path):
    settings_path = _manuscript_processing_settings_path(manuscript_path)
    if not settings_path.exists():
        return {}
    try:
        payload = json.loads(settings_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Warning: Could not read manuscript processing settings at {settings_path}: {exc}")
        return {}

    raw_args = payload.get("line_segmentation_args", {})
    if not isinstance(raw_args, dict):
        return {}

    try:
        threshold = _coerce_optional_binarize_threshold(
            raw_args.get(BINARIZE_THRESHOLD_OVERRIDE_KEY),
            field_name=f"stored {BINARIZE_THRESHOLD_OVERRIDE_KEY}",
        )
    except ValueError as exc:
        print(f"Warning: Ignoring invalid manuscript binarization override: {exc}")
        return {}

    if threshold is None:
        return {}
    return {BINARIZE_THRESHOLD_OVERRIDE_KEY: threshold}


def _normalize_recognition_engine(value):
    engine = str(value or "local").strip().lower()
    return engine if engine in SUPPORTED_RECOGNITION_ENGINES else "local"


def _server_gemini_api_key():
    return (os.getenv("GEMINI_API_KEY") or "").strip()


def _gemini_ocr_timeout_seconds():
    raw_value = os.getenv("GEMINI_OCR_TIMEOUT_SECONDS")
    if raw_value is None or str(raw_value).strip() == "":
        return DEFAULT_GEMINI_OCR_TIMEOUT_SECONDS

    try:
        timeout_seconds = float(raw_value)
    except (TypeError, ValueError):
        print(
            f"Warning: Ignoring invalid GEMINI_OCR_TIMEOUT_SECONDS={raw_value!r}; "
            f"using {DEFAULT_GEMINI_OCR_TIMEOUT_SECONDS:g}s."
        )
        return DEFAULT_GEMINI_OCR_TIMEOUT_SECONDS

    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        print(
            f"Warning: Ignoring invalid GEMINI_OCR_TIMEOUT_SECONDS={raw_value!r}; "
            f"using {DEFAULT_GEMINI_OCR_TIMEOUT_SECONDS:g}s."
        )
        return DEFAULT_GEMINI_OCR_TIMEOUT_SECONDS

    return max(5.0, min(timeout_seconds, 300.0))


def _gemini_ocr_http_options():
    timeout_ms = int(_gemini_ocr_timeout_seconds() * 1000)
    return types.HttpOptions(
        timeout=timeout_ms,
        retryOptions=types.HttpRetryOptions(attempts=1),
    )


def _is_timeout_exception(exc):
    if isinstance(exc, TimeoutError):
        return True
    exc_name = exc.__class__.__name__.lower()
    message = str(exc).lower()
    return (
        "timeout" in exc_name
        or "timed out" in message
        or "deadline exceeded" in message
    )


def _recognition_failure_payload(result, engine):
    payload = dict(result or {})
    payload.setdefault("error", "Could not read the page.")
    payload.setdefault("recognitionEngine", engine)
    payload.setdefault("failedEngine", engine)
    if engine == "gemini":
        payload.setdefault("retryable", True)
        payload.setdefault("fallbackEngines", ["local"])
    return payload


def _recognition_reader_capabilities():
    gemini_configured = bool(_server_gemini_api_key())
    return {
        "defaultEngine": "local",
        "readers": {
            "local": {
                "available": True,
                "label": "Built-in Reader",
                "serverConfigured": True,
            },
            "gemini": {
                "available": gemini_configured,
                "label": "Gemini",
                "serverConfigured": gemini_configured,
                "requestTimeoutSeconds": _gemini_ocr_timeout_seconds(),
                "unavailableReason": None if gemini_configured else "Gemini is not configured on this server.",
            },
        },
    }


def _configure_active_learning_runtime():
    configure_runtime(OCR_MODEL_PATH, JOB_ORCHESTRATOR)


def _get_manuscript_active_learning_state(manuscript):
    _configure_active_learning_runtime()
    manuscript_root = Path(UPLOAD_FOLDER) / manuscript
    return summarize_manuscript_active_learning(
        manuscript_root,
        base_checkpoint_path=OCR_MODEL_PATH,
        orchestrator=JOB_ORCHESTRATOR,
    )


def _get_manuscript_local_checkpoint(manuscript):
    active_learning = _get_manuscript_active_learning_state(manuscript)
    return active_learning["active_checkpoint_path"], active_learning["active_checkpoint_id"], active_learning

def parse_page_xml_polygons(xml_path):
    polygons = {}
    if not os.path.exists(xml_path):
        return polygons

    try:
        ns = {'p': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}
        tree = ET.parse(xml_path)
        root = tree.getroot()

        for textline in root.findall(".//p:TextLine", ns):
            custom_attr = textline.get('custom', '')
            if 'structure_line_id_' not in custom_attr:
                continue
            
            try:
                line_id = str(custom_attr.split('structure_line_id_')[1])
            except IndexError:
                continue

            coords_elem = textline.find('p:Coords', ns)
            if coords_elem is not None:
                points_str = coords_elem.get('points', '')
                if points_str:
                    points = [list(map(int, p.split(','))) for p in points_str.strip().split(' ')]
                    polygons[line_id] = points
            
    except Exception as e:
        print(f"Error parsing XML polygons: {e}")
        
    return polygons


def get_existing_text_content(xml_path):
    text_content = {}
    confidences = {}
    
    if not os.path.exists(xml_path):
        return {"text": {}, "confidences": {}}
        
    try:
        ns = {'p': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        for textline in root.findall(".//p:TextLine", ns):
            custom_attr = textline.get('custom', '')
            if 'structure_line_id_' in custom_attr:
                try:
                    line_id = str(custom_attr.split('structure_line_id_')[1])
                except IndexError:
                    continue
                
                text_equiv = textline.find('p:TextEquiv', ns)
                if text_equiv is not None:
                    uni = text_equiv.find('p:Unicode', ns)
                    if uni is not None and uni.text:
                        text_content[line_id] = uni.text
                        
                        te_custom = text_equiv.get('custom', '')
                        if 'confidences:' in te_custom:
                            try:
                                raw_conf = te_custom.split('confidences:')[1].split(';')[0]
                                if raw_conf.strip():
                                    confidences[line_id] = [float(x) for x in raw_conf.split(',')]
                            except Exception:
                                pass
    except Exception as e:
        print(f"Error parsing existing text: {e}")
    
    return {"text": text_content, "confidences": confidences}


def get_existing_reading_direction_annotations(xml_path):
    metadata_path = default_reading_direction_metadata_path(xml_path)
    payload = load_reading_direction_metadata(metadata_path)
    return {
        "lineAnnotations": payload.get("line_annotations", []),
        "staleAnnotations": payload.get("stale_annotations", []),
    }


def _path_is_within(child, parent):
    try:
        Path(child).resolve().relative_to(Path(parent).resolve())
        return True
    except Exception:
        return False


def _safe_manuscript_root(manuscript):
    upload_root = Path(UPLOAD_FOLDER).resolve()
    manuscript_root = (upload_root / manuscript).resolve()
    if not _path_is_within(manuscript_root, upload_root):
        return None
    return manuscript_root


def _safe_existing_file(candidate_path, root_path):
    try:
        resolved_path = Path(candidate_path).resolve()
        resolved_root = Path(root_path).resolve()
    except Exception:
        return None
    if not _path_is_within(resolved_path, resolved_root):
        return None
    return resolved_path if resolved_path.is_file() else None


def _processed_line_image_root(manuscript_root, page):
    return manuscript_root / "layout_analysis_output" / "image-format" / page


def _find_processed_line_image(manuscript_root, page, line_numeric_id, region_custom=None):
    image_root = _processed_line_image_root(manuscript_root, page)
    if not image_root.exists():
        return None

    filename = f"line_{int(line_numeric_id)}.jpg"
    if region_custom:
        candidate = _safe_existing_file(image_root / str(region_custom) / filename, image_root)
        if candidate is not None:
            return candidate

    for candidate in sorted(image_root.glob(f"*/{filename}")):
        safe_candidate = _safe_existing_file(candidate, image_root)
        if safe_candidate is not None:
            return safe_candidate
    return None


def _line_kind_from_metadata(line_metadata):
    if not isinstance(line_metadata, dict):
        return None
    line_kind = line_metadata.get("line_kind")
    if line_kind:
        return str(line_kind)
    topology = line_metadata.get("topology")
    if isinstance(topology, dict) and topology.get("line_kind"):
        return str(topology["line_kind"])
    return None


def _auto_orientation_transforms_by_line_numeric_id(xml_path):
    try:
        _, records = load_pagexml_lines(xml_path, include_empty_text_lines=True)
    except Exception:
        return {}
    return {
        int(record.line_numeric_id): transform
        for record in records
        if (transform := auto_orientation_transform_from_custom(record.text_equiv_custom))
        is not None
    }


def get_existing_line_image_previews(manuscript, page, xml_path):
    previews = {}
    xml_path = Path(xml_path)
    if not xml_path.exists():
        return previews

    manuscript_root = _safe_manuscript_root(manuscript)
    if manuscript_root is None or not manuscript_root.exists():
        return previews

    metadata_path = default_line_segmentation_metadata_path(xml_path)
    metadata_by_numeric_id = load_line_segmentation_metadata_by_numeric_id(metadata_path)
    reading_annotations_by_line_id = load_reading_direction_annotations_by_line_id(
        default_reading_direction_metadata_path(xml_path)
    )
    auto_orientation_by_line_id = _auto_orientation_transforms_by_line_numeric_id(xml_path)

    try:
        _, records = load_pagexml_lines(xml_path, include_empty_text_lines=True)
    except Exception as exc:
        print(f"[{page}] Warning: could not load PAGE lines for image previews: {exc}")
        return previews

    for record in records:
        line_numeric_id = int(record.line_numeric_id)
        line_metadata = metadata_by_numeric_id.get(line_numeric_id, {})
        line_kind = _line_kind_from_metadata(line_metadata)
        has_reading_annotation = bool(
            reading_annotations_by_line_id.get(line_numeric_id)
            or has_explicit_reading_direction_annotation(
                {"strategy_line_metadata": line_metadata}
            )
        )
        should_show_preview = has_reading_annotation or (
            line_kind is not None and line_kind != "horizontal_straight"
        )
        if not should_show_preview:
            continue

        image_path = _find_processed_line_image(
            manuscript_root,
            page,
            line_numeric_id,
            region_custom=record.region_custom,
        )
        if image_path is None:
            continue

        try:
            with Image.open(image_path) as line_image:
                image_width, image_height = line_image.size
        except Exception as exc:
            print(f"[{page}] Warning: could not read line preview image {image_path}: {exc}")
            continue

        previews[str(line_numeric_id)] = {
            "lineKind": line_kind,
            "hasReadingDirectionAnnotation": has_reading_annotation,
            "imageWidth": image_width,
            "imageHeight": image_height,
            "autoOrientationTransform": (
                None
                if has_reading_annotation
                else auto_orientation_by_line_id.get(line_numeric_id)
            ),
            "imageUrl": (
                f"/line-image/{quote(str(manuscript), safe='')}/"
                f"{quote(str(page), safe='')}/{line_numeric_id}"
                f"?v={image_path.stat().st_mtime_ns}-{xml_path.stat().st_mtime_ns}"
            ),
        }
    return previews


def update_page_text_content(xml_path, text_content=None, confidences=None):
    xml_path = Path(xml_path)
    if not xml_path.exists():
        raise FileNotFoundError(f"PAGE XML not found: {xml_path}")

    PAGE_XML_NAMESPACE = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
    ns = {'p': PAGE_XML_NAMESPACE}
    ET.register_namespace('', PAGE_XML_NAMESPACE)

    normalized_text = {
        str(line_id): ("" if value is None else str(value))
        for line_id, value in dict(text_content or {}).items()
    }
    normalized_confidences = {
        str(line_id): list(values or [])
        for line_id, values in dict(confidences or {}).items()
    }

    tree = ET.parse(xml_path)
    root = tree.getroot()
    saved_line_count = 0
    annotated_line_ids = set(
        load_reading_direction_annotations_by_line_id(
            default_reading_direction_metadata_path(xml_path)
        )
    )
    line_metadata_by_numeric_id = load_line_segmentation_metadata_by_numeric_id(
        default_line_segmentation_metadata_path(xml_path)
    )

    for textline in root.findall(".//p:TextLine", ns):
        custom_attr = textline.get('custom', '')
        if 'structure_line_id_' not in custom_attr:
            continue

        try:
            line_id = str(custom_attr.split('structure_line_id_')[1])
        except IndexError:
            continue
        try:
            line_numeric_id = int(line_id)
            has_reading_annotation = bool(
                line_numeric_id in annotated_line_ids
                or has_explicit_reading_direction_annotation(
                    {
                        "strategy_line_metadata": line_metadata_by_numeric_id.get(
                            line_numeric_id, {}
                        )
                    }
                )
            )
        except (TypeError, ValueError):
            has_reading_annotation = False

        existing_equivs = textline.findall('./p:TextEquiv', ns)
        preserved_auto_orientation = ""
        for existing_equiv in existing_equivs:
            preserved_auto_orientation = (
                auto_orientation_custom_metadata(existing_equiv.get("custom"))
                or preserved_auto_orientation
            )
        for existing_equiv in existing_equivs:
            textline.remove(existing_equiv)

        line_text = normalized_text.get(line_id, "")
        if not line_text:
            continue

        text_equiv = ET.SubElement(textline, f"{{{PAGE_XML_NAMESPACE}}}TextEquiv")
        line_confidences = normalized_confidences.get(line_id, [])
        custom_fields = []
        if preserved_auto_orientation and not has_reading_annotation:
            custom_fields.append(preserved_auto_orientation)
        if line_confidences:
            custom_fields.append(f"confidences:{','.join(map(str, line_confidences))}")
        if custom_fields:
            text_equiv.set('custom', ";".join(custom_fields))

        unicode_elem = ET.SubElement(text_equiv, f"{{{PAGE_XML_NAMESPACE}}}Unicode")
        unicode_elem.text = line_text
        saved_line_count += 1

    if hasattr(ET, 'indent'):
        ET.indent(tree, space="\t", level=0)
    tree.write(xml_path, encoding='UTF-8', xml_declaration=True)

    return {"status": "success", "lines": saved_line_count}


def compute_page_layout_fingerprint(xml_path):
    if not os.path.exists(xml_path):
        return None

    try:
        def normalize_point_value(raw_value):
            value = str(raw_value or "").strip()
            try:
                numeric_value = float(value)
            except ValueError:
                return value
            if numeric_value.is_integer():
                return int(numeric_value)
            return round(numeric_value, 6)

        def parse_points(points_str):
            points = []
            for raw_point in str(points_str or "").split():
                if "," not in raw_point:
                    continue
                x_raw, y_raw = raw_point.split(",", 1)
                points.append((normalize_point_value(x_raw), normalize_point_value(y_raw)))
            return points

        def canonicalize_polyline(points_str):
            points = parse_points(points_str)
            if not points:
                return []
            forward = tuple(points)
            backward = tuple(reversed(points))
            return list(min(forward, backward))

        def canonicalize_polygon(points_str):
            points = parse_points(points_str)
            if not points:
                return []

            candidates = []
            for point_order in (points, list(reversed(points))):
                for start_index in range(len(point_order)):
                    rotated = point_order[start_index:] + point_order[:start_index]
                    candidates.append(tuple(rotated))

            return list(min(candidates))

        ns = {'p': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}
        tree = ET.parse(xml_path)
        root = tree.getroot()
        line_records = []
        reading_direction_payload = load_reading_direction_metadata(
            default_reading_direction_metadata_path(xml_path)
        )

        for textline in root.findall(".//p:TextLine", ns):
            custom_attr = textline.get('custom', '')
            if 'structure_line_id_' not in custom_attr:
                continue
            try:
                line_id = str(custom_attr.split('structure_line_id_')[1])
            except IndexError:
                continue

            coords_elem = textline.find('p:Coords', ns)
            baseline_elem = textline.find('p:Baseline', ns)
            line_records.append(
                {
                    "line_id": line_id,
                    "coords": canonicalize_polygon(coords_elem.get('points', '') if coords_elem is not None else ''),
                    "baseline": canonicalize_polyline(
                        baseline_elem.get('points', '') if baseline_elem is not None else ''
                    ),
                }
            )

        serialized = json.dumps(
            {
                "lines": sorted(line_records, key=lambda record: record["line_id"]),
                "reading_direction": reading_direction_payload,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha1(serialized.encode("utf-8")).hexdigest()
    except Exception as e:
        print(f"Error computing page layout fingerprint for {xml_path}: {e}")
        return None


def _normalize_textbox_labels_payload(textbox_labels, node_count):
    try:
        safe_node_count = max(0, int(node_count or 0))
    except (TypeError, ValueError):
        safe_node_count = 0

    normalized = [-1] * safe_node_count
    if not isinstance(textbox_labels, list):
        return normalized

    for index, raw_label in enumerate(textbox_labels[:safe_node_count]):
        try:
            label = int(raw_label)
        except (TypeError, ValueError):
            label = -1
        normalized[index] = label if label >= 0 else -1
    return normalized


def _components_from_graph_payload(graph_payload, node_count):
    try:
        safe_node_count = max(0, int(node_count or 0))
    except (TypeError, ValueError):
        safe_node_count = 0
    if safe_node_count <= 0:
        return []

    parent = list(range(safe_node_count))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for edge in (graph_payload or {}).get("edges") or []:
        try:
            source = int(edge.get("source"))
            target = int(edge.get("target"))
        except (AttributeError, TypeError, ValueError):
            continue
        if 0 <= source < safe_node_count and 0 <= target < safe_node_count:
            union(source, target)

    grouped = collections.defaultdict(list)
    for node_index in range(safe_node_count):
        grouped[find(node_index)].append(node_index)
    return sorted(grouped.values(), key=lambda component: component[0])


def _majority_nonnegative_label(labels, component):
    values = [
        int(labels[node_index])
        for node_index in component
        if 0 <= node_index < len(labels) and int(labels[node_index]) >= 0
    ]
    if not values:
        return None
    counts = collections.Counter(values)
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _resolve_textbox_labels_for_layout(graph_payload, textbox_labels, node_count):
    normalized = _normalize_textbox_labels_payload(textbox_labels, node_count)
    components = _components_from_graph_payload(graph_payload, len(normalized))
    used_labels = {label for label in normalized if label >= 0}
    resolved = list(normalized)
    next_default_label = 0

    for line_index, component in enumerate(components):
        label = _majority_nonnegative_label(normalized, component)
        if label is None:
            while next_default_label in used_labels:
                next_default_label += 1
            label = next_default_label
        used_labels.add(int(label))
        for node_index in component:
            resolved[node_index] = int(label)

    return resolved


def _load_saved_textbox_labels(manuscript_path, page, node_count):
    labels_path = Path(manuscript_path) / "layout_analysis_output" / "gnn-format" / f"{page}_labels_textbox.txt"
    if not labels_path.exists():
        return None
    try:
        labels = np.loadtxt(labels_path, dtype=int, ndmin=1).reshape(-1)
    except Exception as exc:
        print(f"[{page}] Warning: could not read prior textbox labels for telemetry: {exc}")
        return None
    try:
        safe_node_count = max(0, int(node_count or 0))
    except (TypeError, ValueError):
        safe_node_count = 0
    if labels.size != safe_node_count:
        return None
    return [int(value) for value in labels.tolist()]


def _build_page_workflow(
    manuscript_path,
    page,
    text_payload=None,
    active_learning=None,
    graph_payload=None,
    textbox_labels=None,
):
    xml_path = manuscript_path / "layout_analysis_output" / "page-xml-format" / f"{page}.xml"
    layout_fingerprint = compute_page_layout_fingerprint(str(xml_path))
    if text_payload is None:
        text_payload = get_existing_text_content(str(xml_path)).get("text", {}) if xml_path.exists() else {}

    workflow = summarize_page_active_learning(
        manuscript_path,
        page,
        current_text_payload=text_payload,
        current_layout_fingerprint=layout_fingerprint,
        current_graph_payload=graph_payload,
        current_textbox_labels=textbox_labels,
        base_checkpoint_path=OCR_MODEL_PATH,
    )
    if active_learning:
        workflow["active_checkpoint_id"] = active_learning.get("active_checkpoint_id")
    workflow["text_recovery"] = build_text_recovery_state(
        manuscript_path,
        page,
        current_xml_path=xml_path,
    )
    return workflow


def get_ocr_context():
    """
    Singleton to load the OCR model and config only once.
    """
    _configure_active_learning_runtime()
    global OCR_GLOBAL_CONTEXT
    if OCR_GLOBAL_CONTEXT is not None:
        return OCR_GLOBAL_CONTEXT

    if not os.path.exists(OCR_MODEL_PATH):
        print(f"Error: OCR Model not found at {OCR_MODEL_PATH}")
        return None

    try:
        print("Loading Local OCR Model...")
        OCR_GLOBAL_CONTEXT = OCR_MODEL_MANAGER.get_context(OCR_MODEL_PATH)
        print("Local OCR Model Loaded Successfully.")
        return OCR_GLOBAL_CONTEXT
    except Exception as e:
        print(f"Failed to load OCR model: {e}")
        import traceback
        traceback.print_exc()
        return None

def _run_local_recognition_internal(manuscript, page, checkpoint_path=None, checkpoint_id=None, interactive=False):
    """
    Drop-in replacement for Gemini OCR using local EasyOCR/PyTorch.
    1. Loads model (if not loaded).
    2. Runs process_page_xml (crops, infers, updates XML).
    3. Reads updated XML and returns text/confidences.
    """
    print(f"[{page}] Starting Local Recognition...")
    
    # 1. Path Setup
    base_path = Path(UPLOAD_FOLDER) / manuscript
    xml_path = base_path / "layout_analysis_output" / "page-xml-format" / f"{page}.xml"
    
    # PAGE XML coordinates are generated against resized images; avoid loading original large images here.
    image_dirs = [
        str(base_path / "layout_analysis_output" / "images_resized"),
        str(base_path / "images_resized"),
    ]

    if not xml_path.exists():
        print(f"[{page}] XML file not found: {xml_path}")
        return {}

    # 2. Get Model Context
    if interactive:
        prepare_for_interactive_ocr(base_path, orchestrator=JOB_ORCHESTRATOR)

    if checkpoint_path is None:
        checkpoint_path, checkpoint_id, _ = _get_manuscript_local_checkpoint(manuscript)

    checkpoint_path = str(Path(checkpoint_path or OCR_MODEL_PATH).resolve())
    effective_checkpoint_id = checkpoint_id or ("base" if checkpoint_path == str(Path(OCR_MODEL_PATH).resolve()) else None)
    print(f"[{page}] Using Local OCR checkpoint: {effective_checkpoint_id or 'unknown'} @ {checkpoint_path}")
    ctx = get_ocr_context() if checkpoint_path == str(Path(OCR_MODEL_PATH).resolve()) else OCR_MODEL_MANAGER.get_context(checkpoint_path)
    if not ctx:
        return {"error": "OCR Model could not be loaded"}

    try:
        # 3. Run Inference (Modifies XML in-place)
        # We assume single-threaded access to the model for inference is 'safe enough' 
        # via Flask, or process_page_xml handles data loading internally.
        process_page_xml(
            str(xml_path), 
            image_dirs, 
            ctx['model'], 
            ctx['converter'], 
            ctx['config'], 
            ctx['device']
        )
        
        # 4. Read back the results from the updated XML
        # We reuse your existing helper function
        result = get_existing_text_content(str(xml_path))
        layout_fingerprint = compute_page_layout_fingerprint(str(xml_path))
        record_prediction(
            manuscript_root=base_path,
            page_id=page,
            predicted_lines=result.get("text", {}),
            recognition_engine="local",
            checkpoint_id=effective_checkpoint_id,
            checkpoint_path=checkpoint_path,
            confidences=result.get("confidences", {}),
            layout_fingerprint=layout_fingerprint,
            base_checkpoint_path=OCR_MODEL_PATH,
        )
        result["checkpointId"] = effective_checkpoint_id
        result["checkpointPath"] = checkpoint_path
        result["activeLearning"] = _get_manuscript_active_learning_state(manuscript)
        result["pageWorkflow"] = _build_page_workflow(
            base_path,
            page,
            text_payload=result.get("text", {}),
            active_learning=result["activeLearning"],
        )
        enqueue_pipeline_visualization(
            save_ocr_prediction_visualizations,
            base_path,
            page,
            result.get("text", {}),
        )
        return result

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Local Recognition Error: {e}")
        return {}

@app.route('/upload', methods=['POST'])
def upload_manuscript():
    manuscript_name = request.form.get('manuscriptName', 'default_manuscript')
    try:
        longest_side = int(request.form.get('longestSide', 3500))
        min_distance = int(request.form.get('minDistance', 20))
        binarize_threshold = _coerce_optional_binarize_threshold(
            request.form.get('binarizationThreshold')
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    
    manuscript_path = os.path.join(UPLOAD_FOLDER, manuscript_name)
    images_path = os.path.join(manuscript_path, "images")
    
    if os.path.exists(manuscript_path):
        shutil.rmtree(manuscript_path)
    os.makedirs(images_path)
    _write_manuscript_processing_settings(
        manuscript_path,
        target_longest_side=longest_side,
        min_distance=min_distance,
        binarize_threshold=binarize_threshold,
        pipeline_visualization_enabled=request.form.get('pipelineVisualizationEnabled'),
    )

    files = request.files.getlist('images')
    if not files:
        return jsonify({"error": "No files uploaded"}), 400

    for file in files:
        if file.filename:
            file.save(os.path.join(images_path, file.filename))

    try:
        process_new_manuscript(manuscript_path, target_longest_side=longest_side, min_distance=min_distance) 
        processed_pages = []
        for f in sorted(Path(manuscript_path).glob("gnn-dataset/*_dims.txt")):
            processed_pages.append(f.name.replace("_dims.txt", ""))
        for page in processed_pages:
            enqueue_pipeline_visualization(
                save_upload_visualizations,
                manuscript_path,
                page,
            )
            
        return jsonify({"message": "Processed successfully", "pages": processed_pages})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/manuscript/<name>/pages', methods=['GET'])
def get_pages(name):
    """
    Returns list of pages and the ID of the most recently edited page based on XML mtime.
    """
    manuscript_path = Path(UPLOAD_FOLDER) / name
    dataset_path = manuscript_path / "gnn-dataset"
    if not dataset_path.exists():
        return jsonify({"pages": [], "last_edited": None}), 404
    
    pages = sorted([f.name.replace("_dims.txt", "") for f in dataset_path.glob("*_dims.txt")])
    
    # Determine last edited page
    xml_dir = manuscript_path / "layout_analysis_output" / "page-xml-format"
    last_edited = None
    latest_time = 0
    
    if xml_dir.exists():
        for page in pages:
            xml_file = xml_dir / f"{page}.xml"
            if xml_file.exists():
                mtime = xml_file.stat().st_mtime
                if mtime > latest_time:
                    latest_time = mtime
                    last_edited = page

    return jsonify({"pages": pages, "last_edited": last_edited})


@app.route('/manuscript/<name>/active-learning', methods=['GET'])
def get_manuscript_active_learning(name):
    manuscript_path = Path(UPLOAD_FOLDER) / name
    if not manuscript_path.exists():
        return jsonify({"error": "Manuscript not found"}), 404
    try:
        return jsonify(_get_manuscript_active_learning_state(name))
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/recognition/readers', methods=['GET'])
def get_recognition_readers():
    return jsonify(_recognition_reader_capabilities())

@app.route('/semi-segment/<manuscript>/<page>', methods=['GET'])
def get_page_prediction(manuscript, page):
    manuscript_path = Path(UPLOAD_FOLDER) / manuscript
    try:
        graph_data = run_gnn_prediction_for_page(
            str(manuscript_path), 
            page, 
            MODEL_CHECKPOINT, 
            DATASET_CONFIG
        )
        
        img_path = manuscript_path / "images_resized" / f"{page}.jpg"
        encoded_string = ""
        if img_path.exists():
            with open(img_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

        xml_path = manuscript_path / "layout_analysis_output" / "page-xml-format" / f"{page}.xml"
        polygons = {}
        existing_data = {"text": {}, "confidences": {}}
        reading_direction_annotations = {"lineAnnotations": [], "staleAnnotations": []}
        line_image_previews = {}
        
        if xml_path.exists():
            polygons = parse_page_xml_polygons(str(xml_path))
            existing_data = get_existing_text_content(str(xml_path))
            reading_direction_annotations = get_existing_reading_direction_annotations(str(xml_path))
            line_image_previews = get_existing_line_image_previews(manuscript, page, xml_path)

        active_learning = _get_manuscript_active_learning_state(manuscript)
        response = {
            "image": encoded_string,
            "dimensions": graph_data['dimensions'],
            "points": [[n['x'], n['y']] for n in graph_data['nodes']],
            "graph": graph_data,
            "textline_labels": graph_data.get('textline_labels', []),
            "textbox_labels": graph_data.get('textbox_labels', []),
            "polygons": polygons, 
            "textContent": existing_data["text"],
            "textConfidences": existing_data["confidences"],
            "readingDirectionAnnotations": reading_direction_annotations,
            "lineImagePreviews": line_image_previews,
            "activeLearning": active_learning,
            "pageWorkflow": _build_page_workflow(
                manuscript_path,
                page,
                text_payload=existing_data["text"],
                active_learning=active_learning,
                graph_payload=graph_data,
                textbox_labels=graph_data.get('textbox_labels', []),
            ),
        }
        return jsonify(response)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/line-image/<manuscript>/<page>/<int:line_numeric_id>', methods=['GET'])
def get_processed_line_image(manuscript, page, line_numeric_id):
    manuscript_root = _safe_manuscript_root(manuscript)
    if manuscript_root is None or not manuscript_root.exists():
        return jsonify({"error": "Manuscript not found"}), 404

    image_path = _find_processed_line_image(manuscript_root, page, line_numeric_id)
    if image_path is None:
        return jsonify({"error": "Line image not found"}), 404

    xml_root = manuscript_root / "layout_analysis_output" / "page-xml-format"
    xml_path = _safe_existing_file(xml_root / f"{page}.xml", xml_root)
    transform = (
        _auto_orientation_transforms_by_line_numeric_id(xml_path).get(line_numeric_id)
        if xml_path is not None
        else None
    )
    if xml_path is not None:
        reading_annotations = load_reading_direction_annotations_by_line_id(
            default_reading_direction_metadata_path(xml_path)
        )
        line_metadata = load_line_segmentation_metadata_by_numeric_id(
            default_line_segmentation_metadata_path(xml_path)
        ).get(line_numeric_id, {})
        if (
            line_numeric_id in reading_annotations
            or has_explicit_reading_direction_annotation(
                {"strategy_line_metadata": line_metadata}
            )
        ):
            transform = None
    if transform == ROTATE_180_TRANSFORM:
        try:
            with Image.open(image_path) as line_image:
                oriented_image = line_image.transpose(Image.Transpose.ROTATE_180)
                image_buffer = io.BytesIO()
                oriented_image.save(image_buffer, format="JPEG", quality=95)
            image_buffer.seek(0)
            return send_file(image_buffer, mimetype="image/jpeg", download_name=image_path.name)
        except Exception as exc:
            print(f"[{page}] Warning: could not orient line preview {image_path}: {exc}")

    return send_file(image_path, mimetype="image/jpeg")


def ensemble_text_samples(samples):
    valid_samples = [s for s in samples if s and s.strip()]
    if not valid_samples:
        return "", []
    if len(valid_samples) == 1:
        return valid_samples[0], [1.0] * len(valid_samples[0])

    valid_samples.sort(key=len)
    pivot_idx = len(valid_samples) // 2
    pivot = valid_samples[pivot_idx]
    
    GAP_TOKEN = "__GAP__"
    total_samples = len(valid_samples) 
    
    grid = [collections.Counter({char: 1}) for char in pivot]
    insertions = collections.defaultdict(collections.Counter)
    
    others = valid_samples[:pivot_idx] + valid_samples[pivot_idx+1:]
    
    for sample in others:
        matcher = difflib.SequenceMatcher(None, pivot, sample)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                for k in range(i2 - i1):
                    grid[i1 + k][pivot[i1 + k]] += 1
            elif tag == 'replace':
                len_pivot_seg = i2 - i1
                len_sample_seg = j2 - j1
                min_len = min(len_pivot_seg, len_sample_seg)
                for k in range(min_len):
                    grid[i1 + k][sample[j1 + k]] += 1
                for k in range(min_len, len_pivot_seg):
                    grid[i1 + k][GAP_TOKEN] += 1
                if len_sample_seg > len_pivot_seg:
                    inserted_chunk = sample[j1 + min_len : j2]
                    insertions[i2 - 1][inserted_chunk] += 1
            elif tag == 'delete':
                for k in range(i2 - i1):
                    grid[i1 + k][GAP_TOKEN] += 1
            elif tag == 'insert':
                inserted_chunk = sample[j1:j2]
                target_idx = i1 - 1
                insertions[target_idx][inserted_chunk] += 1

    result_chars = []
    result_confidences = []

    def append_result(char_str, vote_count):
        conf = round(vote_count / total_samples, 2)
        for c in char_str:
            result_chars.append(c)
            result_confidences.append(conf)

    if -1 in insertions:
        best_ins, count = insertions[-1].most_common(1)[0]
        append_result(best_ins, count)

    for i in range(len(pivot)):
        best_char, count = grid[i].most_common(1)[0]
        if best_char != GAP_TOKEN:
            append_result(best_char, count)
        if i in insertions:
            best_ins, count = insertions[i].most_common(1)[0]
            append_result(best_ins, count)

    return "".join(result_chars), result_confidences


def _json_from_gemini_text(raw_text):
    text = str(raw_text or "").strip()
    if not text:
        raise ValueError("Gemini returned an empty response.")

    parse_candidates = [text]
    fenced_match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced_match:
        parse_candidates.insert(0, fenced_match.group(1).strip())

    array_start = text.find("[")
    array_end = text.rfind("]")
    if 0 <= array_start < array_end:
        parse_candidates.append(text[array_start : array_end + 1])

    object_start = text.find("{")
    object_end = text.rfind("}")
    if 0 <= object_start < object_end:
        parse_candidates.append(text[object_start : object_end + 1])

    last_error = None
    for candidate in parse_candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc

    raise ValueError(f"Gemini returned invalid JSON: {last_error}")


def _parse_gemini_transcriptions(raw_text):
    data = _json_from_gemini_text(raw_text)

    if isinstance(data, dict):
        for key in ("transcriptions", "lines", "results"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
        else:
            if "id" in data and "text" in data:
                data = [data]
            else:
                raise ValueError("Gemini JSON did not contain a transcription list.")

    if not isinstance(data, list):
        raise ValueError("Gemini JSON must be a transcription list.")

    parsed = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        if "id" not in item or "text" not in item:
            continue
        line_id = str(item.get("id", "")).strip()
        if not line_id:
            continue
        raw_line_text = item.get("text")
        line_text = "" if raw_line_text is None else str(raw_line_text).strip()
        if line_text:
            parsed[line_id] = line_text

    if not parsed:
        raise ValueError("Gemini returned no line-level text for this page.")

    return parsed


def _run_gemini_recognition_internal(
    manuscript,
    page,
    api_key=None,
    N=1,
    num_trace_points=4,
    preserve_auto_orientation_metadata=False,
):
    started_at = time.monotonic()
    print(f"[{page}] Starting parallel recognition with N={N}, points={num_trace_points}...")
    api_key = _server_gemini_api_key()
    if not api_key:
        return {"error": "Gemini is not configured on this server."}
    
    base_path = Path(UPLOAD_FOLDER) / manuscript
    xml_path = base_path / "layout_analysis_output" / "page-xml-format" / f"{page}.xml"
    img_path = base_path / "images_resized" / f"{page}.jpg"

    if not xml_path.exists() or not img_path.exists():
        return {"error": "Page XML or image is missing."}

    try:
        pil_img = Image.open(img_path)
        img_w, img_h = pil_img.size
        
        ns = {'p': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}
        tree = ET.parse(xml_path)
        root = tree.getroot()

        def get_equidistant_points(pts, m):
            if len(pts) < 2: return pts * m
            dists = [0.0]
            for i in range(len(pts)-1):
                dists.append(dists[-1] + ((pts[i+1][0]-pts[i][0])**2 + (pts[i+1][1]-pts[i][1])**2)**0.5)
            
            total_dist = dists[-1]
            if total_dist == 0: return [pts[0]] * m
            
            new_pts = []
            for i in range(m):
                target = (i / (m - 1)) * total_dist
                for j in range(len(dists)-1):
                    if dists[j] <= target <= dists[j+1]:
                        segment_dist = dists[j+1] - dists[j]
                        rat = (target - dists[j]) / segment_dist if segment_dist > 0 else 0
                        nx = pts[j][0] + rat * (pts[j+1][0] - pts[j][0])
                        ny = pts[j][1] + rat * (pts[j+1][1] - pts[j][1])
                        new_pts.append([int(nx), int(ny)])
                        break
            return new_pts

        reading_annotations_by_line_id = load_reading_direction_annotations_by_line_id(
            default_reading_direction_metadata_path(xml_path)
        )

        def normalize_trace_baseline(line_id, pts):
            try:
                line_numeric_id = int(line_id)
            except Exception:
                line_numeric_id = None
            reading_annotation = (
                reading_annotations_by_line_id.get(line_numeric_id)
                if line_numeric_id is not None
                else None
            )
            try:
                topology = normalize_baseline_topology(
                    pts,
                    reading_direction=(reading_annotation or {}).get("reading_direction"),
                    reading_cut_point=(reading_annotation or {}).get("cut_midpoint"),
                )
                if topology.normalized_points:
                    return [
                        [int(round(point[0])), int(round(point[1]))]
                        for point in topology.normalized_points
                    ]
            except Exception as exc:
                print(f"[{page}] Warning: Gemini trace normalization failed for line {line_id}: {exc}")
            return pts

        lines_geometry = [] 
        for textline in root.findall(".//p:TextLine", ns):
            custom_attr = textline.get('custom', '')
            if 'structure_line_id_' not in custom_attr: continue
            line_id = str(custom_attr.split('structure_line_id_')[1])

            base_elem = textline.find('p:Baseline', ns)
            if base_elem is not None and base_elem.get('points'):
                pts = [list(map(int, p.split(','))) for p in base_elem.get('points').strip().split(' ')]
            else: continue
            pts = normalize_trace_baseline(line_id, pts)

            coords_elem = textline.find('p:Coords', ns)
            poly_pts = [list(map(int, p.split(','))) for p in coords_elem.get('points').strip().split(' ')] if coords_elem is not None else []
            
            if poly_pts:
                pxs, pys = [p[0] for p in poly_pts], [p[1] for p in poly_pts]
                width_px, height_px = max(pxs)-min(pxs), max(pys)-min(pys)
                is_vert = height_px > (width_px * 1.2)
                thickness = width_px if is_vert else height_px
            else:
                is_vert, thickness = False, 30

            lines_geometry.append({
                "id": line_id, "baseline": pts, 
                "thickness": max(10, min(thickness, 20)), "is_vertical": is_vert
            })

        if not lines_geometry:
            return {"error": "No text-line baselines are available for Gemini recognition."}

        timeout_seconds = _gemini_ocr_timeout_seconds()
        client = genai.Client(api_key=api_key, http_options=_gemini_ocr_http_options())
        sample_errors = []
        sample_errors_lock = threading.Lock()

        # model = genai.GenerativeModel('gemini-3.5-flash')

        def normalize(x, y):
            return max(0, min(1000, int((y / img_h) * 1000))), max(0, min(1000, int((x / img_w) * 1000)))

        def sample_worker(sample_idx):
            # No path shifting logic here anymore.
            
            regions_payload = []
            for line in lines_geometry:
                trace_raw = get_equidistant_points(line['baseline'], num_trace_points)
                # Use the exact baseline trace without shifting
                shifted = trace_raw
                
                gemini_trace = []
                for px, py in shifted:
                    ny, nx = normalize(px, py)
                    gemini_trace.extend([ny, nx])
                
                regions_payload.append({"id": line['id'], "trace": gemini_trace, "y": trace_raw[0][1]})

            regions_payload.sort(key=lambda k: k['y'])

            # Improved Prompt: Aligning with Autoregressive Spatial Grounding
            prompt_text = (
                "You are an expert Indologist and Paleographer specializing in handwritten Sanskrit manuscripts."
                "Your Task: Perform a diplomatic transcription (OCR) of the attached manuscript image.\n"
                "CRITICAL INSTRUCTIONS:\n"
                "Transcribe the Sanskrit text from the image at the text-line level, where locations of the handwritten text-lines are defined using 'Path Traces'. Each 'Path Trace' refers to one text-line.\n"
                "The coordinates of the Path Traces are normalized on a 0-1000 scale (where [0,0] is top-left and [1000,1000] is bottom-right) "
                "to precisely map the text line locations on the image.\n"
                "For each path trace points, transcribe the text that sits along this curve.\n"
                "Focus strictly on the visual line indicated by the trace; ignore text from lines above or below.\n"
                "The path trace can be curved and even circular. If a path trace is circular, transcribe the text that sits along the entire circle, starting from the top.\n"
                "Transcribe in Unicode Devanagari. Preserve original spelling (Sandhi).\n"
                "Output a JSON array of objects with 'id' and 'text'.\n\n"
                "REGIONS:\n"
            )
            for item in regions_payload:
                prompt_text += f"ID: {item['id']} | Trace: {item['trace']}\n"

            try:
                # Use higher temperature for ensemble diversity if N > 1, else greedy (0.2)
                run_temperature = 0.7 if N > 1 else 0.2
                
                response = client.models.generate_content(
                    model='gemini-3.5-flash', # Or whichever model version you want to use
                    contents=[pil_img, prompt_text],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=run_temperature
                    )
                )

                return _parse_gemini_transcriptions(response.text)
            except Exception as e:
                with sample_errors_lock:
                    sample_errors.append(e)
                print(f"Sample {sample_idx} error: {e.__class__.__name__}: {e}")
                return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=N) as executor:
            future_to_idx = {executor.submit(sample_worker, i): i for i in range(N)}
            all_samples_results = []
            for future in concurrent.futures.as_completed(future_to_idx):
                sample_result = future.result()
                if sample_result:
                    all_samples_results.append(sample_result)

        if not all_samples_results:
            if any(_is_timeout_exception(error) for error in sample_errors):
                return {
                    "error": (
                        f"Gemini did not finish within {timeout_seconds:g} seconds. "
                        "You can retry Gemini or use the built-in reader."
                    ),
                    "errorCode": "gemini_timeout",
                    "retryable": True,
                }
            if any("empty response" in str(error).lower() for error in sample_errors):
                return {
                    "error": "Gemini returned an empty response.",
                    "errorCode": "gemini_empty_response",
                    "retryable": True,
                }
            return {
                "error": "Gemini did not return a usable transcription.",
                "errorCode": "gemini_invalid_response",
                "retryable": True,
            }

        # --- 3. CHARACTER-LEVEL ENSEMBLE ---
        final_map = {}
        final_confidences = {}
        
        texts_by_id = collections.defaultdict(list)
        for res_map in all_samples_results:
            for lid, txt in res_map.items():
                texts_by_id[lid].append(txt)

        for lid, candidates in texts_by_id.items():
            consensus_text, scores = ensemble_text_samples(candidates)
            if consensus_text:
                final_map[lid] = consensus_text
                final_confidences[lid] = scores
                if len(set(candidates)) > 1 and N > 1:
                    print(f"[{page}] Line {lid}: Merged {len(candidates)} samples. " 
                          f"Result: {consensus_text[:15]}... (Variants: {len(set(candidates))})")

        if final_map:
            changed = False
            for textline in root.findall(".//p:TextLine", ns):
                custom_attr = textline.get('custom', '')
                if 'structure_line_id_' in custom_attr:
                    lid = str(custom_attr.split('structure_line_id_')[1])
                    if lid in final_map:
                        te = textline.find("p:TextEquiv", ns)
                        if te is None: te = ET.SubElement(textline, "TextEquiv")
                        uni = te.find("p:Unicode", ns)
                        if uni is None: uni = ET.SubElement(te, "Unicode")
                        uni.text = final_map[lid]

                        if preserve_auto_orientation_metadata:
                            current_custom = te.get('custom', '')
                            preserved_auto_orientation = auto_orientation_custom_metadata(current_custom)
                            try:
                                has_reading_annotation = int(lid) in reading_annotations_by_line_id
                            except (TypeError, ValueError):
                                has_reading_annotation = False
                            custom_fields = []
                            if preserved_auto_orientation and not has_reading_annotation:
                                custom_fields.append(preserved_auto_orientation)
                            if lid in final_confidences:
                                conf_str = ",".join(map(str, final_confidences[lid]))
                                custom_fields.append(f"confidences:{conf_str}")
                            if custom_fields:
                                te.set('custom', ";".join(custom_fields))
                            else:
                                te.attrib.pop('custom', None)
                        elif lid in final_confidences:
                            conf_str = ",".join(map(str, final_confidences[lid]))
                            te.set('custom', f"confidences:{conf_str}")
                        changed = True
            
            if changed:
                tree.write(xml_path, encoding='UTF-8', xml_declaration=True)
                print(f"[{page}] XML updated with robust ensemble text.")

        if not final_map:
            return {
                "error": "Gemini returned no line-level text for this page.",
                "errorCode": "gemini_empty_response",
                "retryable": True,
            }

        elapsed_seconds = time.monotonic() - started_at
        print(f"[{page}] Gemini recognition completed in {elapsed_seconds:.1f}s for {len(final_map)} lines.")
        return { "text": final_map, "confidences": final_confidences }

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Internal Recognition Error: {e}")
        return {"error": str(e)}





@app.route('/existing-manuscripts', methods=['GET'])
def list_existing_manuscripts():
    if not os.path.exists(UPLOAD_FOLDER):
        return jsonify([])
    
    manuscripts = [
        d for d in os.listdir(UPLOAD_FOLDER) 
        if isdir(join(UPLOAD_FOLDER, d)) and not d.startswith('.')
    ]
    return jsonify(sorted(manuscripts))


@app.route('/semi-segment/<manuscript>/<page>', methods=['POST'])
def save_correction(manuscript, page):
    data = request.json
    manuscript_path = Path(UPLOAD_FOLDER) / manuscript
    
    # --- START OF NODE CORRECTION LOGGING ---
    try:
        modifications = data.get('modifications', [])
        nodes_data = data.get('graph', {}).get('nodes', [])
        
        nodes_added = sum(1 for m in modifications if m.get('type') == 'node_add')
        nodes_removed = sum(1 for m in modifications if m.get('type') == 'node_delete')
        final_nodes_count = len(nodes_data) if nodes_data else 0
        
        corrections_dir = manuscript_path / "node_corrections"
        corrections_dir.mkdir(parents=True, exist_ok=True)
        correction_file = corrections_dir / f"{page}.json"
        
        # If file exists, we cumulatively update the counts (useful for multiple saves / auto-saves)
        if correction_file.exists():
            with open(correction_file, 'r') as f:
                prev_data = json.load(f)
            original_nodes_count = prev_data.get('original_nodes', final_nodes_count - nodes_added + nodes_removed)
            total_added = prev_data.get('nodes_added', 0) + nodes_added
            total_removed = prev_data.get('nodes_removed', 0) + nodes_removed
        else:
            original_nodes_count = final_nodes_count - nodes_added + nodes_removed
            total_added = nodes_added
            total_removed = nodes_removed
            
        correction_data = {
            "original_nodes": original_nodes_count,
            "nodes_added": total_added,
            "nodes_removed": total_removed,
            "final_nodes": final_nodes_count
        }
        
        with open(correction_file, 'w') as f:
            json.dump(correction_data, f, indent=4)
            
        print(f"[{page}] Node corrections logged: Original: {original_nodes_count}, Added: {total_added}, Removed: {total_removed}, Final: {final_nodes_count}")
    except Exception as e:
        print(f"[{page}] Error saving node corrections: {e}")
        traceback.print_exc()
    # --- END OF NODE CORRECTION LOGGING ---
    
    textline_labels = data.get('textlineLabels')
    graph_data = data.get('graph') or {}
    baseline_graph = data.get('baselineGraph') or None
    nodes_data = graph_data.get('nodes') or []
    textbox_labels = _resolve_textbox_labels_for_layout(
        graph_data,
        data.get('textboxLabels'),
        len(nodes_data),
    )
    text_content = data.get('textContent') 
    reading_direction_annotations = data.get('readingDirectionAnnotations') or []
    
    run_recognition = data.get('runRecognition', False)
    recognition_engine = _normalize_recognition_engine(data.get('recognitionEngine', 'local'))
    save_intent = data.get('saveIntent', 'commit')
    save_scope = str(data.get('saveScope') or 'layout')
    active_learning_enabled = bool(data.get('activeLearningEnabled', False))
    layout_effort_payload = data.get('layoutEffort') or None
    layout_effort_logging_request_enabled = (
        data.get('layoutEffortLoggingEnabled')
        if 'layoutEffortLoggingEnabled' in data
        else None
    )

    if textline_labels is None or not graph_data:
        return jsonify({"error": "Missing labels or graph data"}), 400

    try:
        xml_path = manuscript_path / "layout_analysis_output" / "page-xml-format" / f"{page}.xml"
        previous_textbox_labels = None
        previous_reading_direction_annotations = None
        if save_scope == 'layout':
            previous_textbox_labels = _load_saved_textbox_labels(manuscript_path, page, len(nodes_data))
            if xml_path.exists():
                previous_reading_direction_annotations = get_existing_reading_direction_annotations(str(xml_path))

        layout_processing_metrics = None
        layout_artifacts_regenerated = False
        if save_scope == 'text_only' and xml_path.exists():
            snapshot_read_mode_pagexml = save_intent == 'commit'
            if snapshot_read_mode_pagexml:
                save_read_mode_pagexml_artifact(
                    manuscript_path, page, xml_path, ground_truth=False
                )
            result = update_page_text_content(xml_path, text_content=text_content)
            if snapshot_read_mode_pagexml:
                save_read_mode_pagexml_artifact(
                    manuscript_path, page, xml_path, ground_truth=True
                )
        else:
            if save_scope == 'layout' and xml_path.exists():
                try:
                    backup_page_xml_for_text_recovery(
                        manuscript_path,
                        page,
                        xml_path=xml_path,
                        layout_fingerprint=compute_page_layout_fingerprint(str(xml_path)),
                    )
                except Exception as backup_error:
                    print(f"[{page}] Error backing up PAGE XML for text recovery: {backup_error}")
                    traceback.print_exc()
            line_segmentation_args = _load_manuscript_line_segmentation_args(manuscript_path)
            layout_artifacts_regenerated = True
            layout_processing_started_at = layout_effort_utc_now_iso()
            layout_processing_start = time.perf_counter()
            result = generate_xml_and_images_for_page(
                str(manuscript_path),
                page,
                textline_labels,
                graph_data.get('edges', []),
                line_segmentation_args,
                textbox_labels=textbox_labels,
                nodes=nodes_data,
                text_content=text_content,
                reading_direction_annotations=reading_direction_annotations,
            )
            layout_processing_metrics = {
                "started_at": layout_processing_started_at,
                "finished_at": layout_effort_utc_now_iso(),
                "duration_seconds": time.perf_counter() - layout_processing_start,
                "status": result.get("status", "success") if isinstance(result, dict) else "success",
                "line_count": (result or {}).get("lines", 0) if isinstance(result, dict) else 0,
                "layout_artifacts_regenerated": True,
            }

        active_learning_result = handle_post_save(
            manuscript=manuscript,
            page=page,
            save_intent=save_intent,
            active_learning_enabled=active_learning_enabled,
            recognition_engine=recognition_engine,
            text_payload=text_content,
            manuscript_root=manuscript_path,
            base_checkpoint_path=OCR_MODEL_PATH,
            graph_payload=graph_data,
            textbox_labels=textbox_labels,
            modifications=modifications,
            previous_textbox_labels=previous_textbox_labels,
            reading_direction_annotations=reading_direction_annotations,
            previous_reading_direction_annotations=previous_reading_direction_annotations,
            save_scope=save_scope,
            orchestrator=JOB_ORCHESTRATOR,
        )
        result['activeLearning'] = active_learning_result['active_learning']
        result['activeLearningRevision'] = active_learning_result['revision']
        result['activeLearningQueuedJobIds'] = active_learning_result['queued_job_ids']
        result['pageWorkflow'] = _build_page_workflow(
            manuscript_path,
            page,
            active_learning=active_learning_result['active_learning'],
            graph_payload=graph_data,
            textbox_labels=textbox_labels,
        )
        if layout_artifacts_regenerated:
            try:
                layout_effort_log_config = None
                if str(layout_effort_logging_request_enabled).strip().lower() in {
                    "0",
                    "false",
                    "no",
                    "off",
                    "disabled",
                }:
                    layout_effort_log_config = {"layout_effort_logging_enabled": False}
                record_layout_effort_save(
                    manuscript_root=manuscript_path,
                    page_id=page,
                    save_scope=save_scope,
                    save_intent=save_intent,
                    layout_metrics=active_learning_result.get("layout_metrics"),
                    layout_effort=layout_effort_payload,
                    processing_metrics=layout_processing_metrics,
                    active_learning_revision=active_learning_result.get("revision"),
                    config=layout_effort_log_config,
                )
            except Exception as logging_error:
                print(f"[{page}] Error saving layout effort log: {logging_error}")
                traceback.print_exc()

            enqueue_pipeline_visualization(
                save_layout_visualizations,
                manuscript_path,
                page,
                baseline_graph=baseline_graph,
                corrected_graph=graph_data,
            )
        elif save_scope == 'text_only':
            enqueue_pipeline_visualization(
                save_layout_visualizations,
                manuscript_path,
                page,
            )

        if run_recognition: 
            # --- MODIFIED: Robust background task with engine switch & logging ---
            checkpoint_path, checkpoint_id, _ = _get_manuscript_local_checkpoint(manuscript)

            def background_task(m, p, engine, local_checkpoint_path, local_checkpoint_id):
                print(f"[{p}] Starting background auto-recognition. Engine: {engine}")
                try:
                    if engine == 'gemini':
                        if not _server_gemini_api_key():
                            print(f"[{p}] ERROR: Gemini is not configured on this server. Aborting recognition.")
                            return
                        gemini_result = _run_gemini_recognition_internal(
                            m,
                            p,
                            preserve_auto_orientation_metadata=True,
                        )
                        if gemini_result.get("error"):
                            print(f"[{p}] ERROR in Gemini recognition: {gemini_result['error']}")
                            return
                        record_prediction(
                            manuscript_root=Path(UPLOAD_FOLDER) / m,
                            page_id=p,
                            predicted_lines=gemini_result.get("text", {}),
                            recognition_engine="gemini",
                            checkpoint_id=None,
                            checkpoint_path=None,
                            confidences=gemini_result.get("confidences", {}),
                            layout_fingerprint=compute_page_layout_fingerprint(
                                str(Path(UPLOAD_FOLDER) / m / "layout_analysis_output" / "page-xml-format" / f"{p}.xml")
                            ),
                            base_checkpoint_path=OCR_MODEL_PATH,
                        )
                        enqueue_pipeline_visualization(
                            save_ocr_prediction_visualizations,
                            Path(UPLOAD_FOLDER) / m,
                            p,
                            gemini_result.get("text", {}),
                        )
                    else:
                        _run_local_recognition_internal(
                            m,
                            p,
                            checkpoint_path=local_checkpoint_path,
                            checkpoint_id=local_checkpoint_id,
                            interactive=False,
                        )
                    print(f"[{p}] Background auto-recognition completed successfully.")
                except Exception as e:
                    print(f"[{p}] ERROR in background auto-recognition: {e}")
                    traceback.print_exc()

            thread = threading.Thread(
                target=background_task,
                args=(manuscript, page, recognition_engine, checkpoint_path, checkpoint_id),
                daemon=True,
            )
            thread.start()
            
            result['autoRecognitionStatus'] = f"processing_in_background_with_{recognition_engine}"

        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/recognize-text', methods=['POST'])
def recognize_text():
    data = request.json
    manuscript = data.get('manuscript')
    page = data.get('page')
    recognition_engine = _normalize_recognition_engine(data.get('recognitionEngine', 'local'))
    
    print(f"[{page}] Manual recognition requested using engine: {recognition_engine}")

    if recognition_engine == 'gemini':
        if not _server_gemini_api_key():
            return jsonify({"error": "Gemini is not configured on this server."}), 400
        result = _run_gemini_recognition_internal(
            manuscript,
            page,
            preserve_auto_orientation_metadata=True,
        )
        if result.get("error"):
            return jsonify(_recognition_failure_payload(result, "gemini")), 502
        record_prediction(
            manuscript_root=Path(UPLOAD_FOLDER) / manuscript,
            page_id=page,
            predicted_lines=result.get("text", {}),
            recognition_engine="gemini",
            checkpoint_id=None,
            checkpoint_path=None,
            confidences=result.get("confidences", {}),
            layout_fingerprint=compute_page_layout_fingerprint(
                str(Path(UPLOAD_FOLDER) / manuscript / "layout_analysis_output" / "page-xml-format" / f"{page}.xml")
            ),
            base_checkpoint_path=OCR_MODEL_PATH,
        )
        enqueue_pipeline_visualization(
            save_ocr_prediction_visualizations,
            Path(UPLOAD_FOLDER) / manuscript,
            page,
            result.get("text", {}),
        )
        result["activeLearning"] = _get_manuscript_active_learning_state(manuscript)
        result["pageWorkflow"] = _build_page_workflow(
            Path(UPLOAD_FOLDER) / manuscript,
            page,
            text_payload=result.get("text", {}),
            active_learning=result["activeLearning"],
        )
    else:
        result = _run_local_recognition_internal(
            manuscript,
            page,
            checkpoint_path=None,
            checkpoint_id=None,
            interactive=True,
        )

    return jsonify(result)


@app.route('/recover-text/<manuscript>/<page>', methods=['POST'])
def recover_text_from_layout_backup(manuscript, page):
    manuscript_root = _safe_manuscript_root(manuscript)
    if manuscript_root is None or not manuscript_root.exists():
        return jsonify({"error": "Manuscript not found"}), 404

    xml_path = manuscript_root / "layout_analysis_output" / "page-xml-format" / f"{page}.xml"
    if not xml_path.exists():
        return jsonify({"error": "Current PAGE XML not found"}), 404

    try:
        plan = build_latest_text_recovery_plan(manuscript_root, page, xml_path)
        if not plan.get("available"):
            return jsonify({"error": "No text recovery backup is available", "textRecovery": plan}), 404

        existing_data = get_existing_text_content(str(xml_path))
        recovered_text = dict(existing_data.get("text", {}))
        recovered_confidences = dict(existing_data.get("confidences", {}))

        for match in plan.get("matches", []):
            line_id = str(match.get("current_line_id"))
            recovered_text[line_id] = str(match.get("recovered_text") or "")
            recovered_confidences.pop(line_id, None)

        update_page_text_content(
            xml_path,
            text_content=recovered_text,
            confidences=recovered_confidences,
        )
        updated_data = get_existing_text_content(str(xml_path))
        active_learning = _get_manuscript_active_learning_state(manuscript)
        workflow = _build_page_workflow(
            manuscript_root,
            page,
            text_payload=updated_data["text"],
            active_learning=active_learning,
        )
        return jsonify(
            {
                "status": "success",
                "text": updated_data["text"],
                "confidences": updated_data["confidences"],
                "textRecovery": plan,
                "activeLearning": active_learning,
                "pageWorkflow": workflow,
            }
        )
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    
@app.route('/save-graph/<manuscript>/<page>', methods=['POST'])
def save_generated_graph(manuscript, page):
    return jsonify({"status": "ok"})


def _zip_directory(zf, source_dir, archive_root):
    source_dir = Path(source_dir)
    if not source_dir.exists():
        return

    for root, dirs, files in os.walk(source_dir):
        dirs.sort()
        for file in sorted(files):
            file_path = Path(root) / file
            rel_path = file_path.relative_to(source_dir).as_posix()
            zf.write(file_path, f"{archive_root}/{rel_path}")


def _find_resized_page_image(manuscript_root, layout_output_root, page_id):
    search_dirs = [
        layout_output_root / "images_resized",
        manuscript_root / "images_resized",
    ]
    extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG", ".TIF", ".TIFF",".jp2"]
    for image_dir in search_dirs:
        for ext in extensions:
            candidate = image_dir / f"{page_id}{ext}"
            if candidate.exists():
                return candidate
    return None


def _write_resized_images_to_zip(zf, manuscript_root, layout_output_root, annotated_page_ids):
    annotated_page_ids = set(annotated_page_ids)
    written = set()
    layout_resized_dir = layout_output_root / "images_resized"
    if layout_resized_dir.exists():
        for file_path in sorted(path for path in layout_resized_dir.iterdir() if path.is_file()):
            if file_path.stem not in annotated_page_ids:
                continue
            arcname = f"images_resized/{file_path.name}"
            zf.write(file_path, arcname)
            written.add(file_path.stem)

    for page_id in sorted(annotated_page_ids - written):
        image_path = _find_resized_page_image(manuscript_root, layout_output_root, page_id)
        if image_path is not None:
            zf.write(image_path, f"images_resized/{image_path.name}")


def _extract_structure_line_id(textline):
    custom_attr = textline.get("custom", "")
    match = re.search(r"structure_line_id_([^;\s]+)", custom_attr)
    if match:
        return match.group(1)
    return None


def _extract_unicode_text(textline, ns):
    text_equiv = textline.find("./p:TextEquiv", ns)
    unicode_elem = text_equiv.find("./p:Unicode", ns) if text_equiv is not None else None
    if unicode_elem is None or unicode_elem.text is None:
        return ""
    return unicode_elem.text.strip()


def _extract_auto_orientation_transform(textline, ns):
    text_equiv = textline.find("./p:TextEquiv", ns)
    return auto_orientation_transform_from_custom(
        text_equiv.get("custom") if text_equiv is not None else None
    )


def _write_ocr_training_format_to_zip(zf, xml_files, image_format_dir):
    page_xml_namespace = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
    ns = {"p": page_xml_namespace}

    for xml_path in sorted(xml_files):
        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError as exc:
            print(f"Error parsing PAGE-XML for OCR training export {xml_path}: {exc}")
            continue

        page_id = xml_path.stem
        annotated_line_ids = set(
            load_reading_direction_annotations_by_line_id(
                default_reading_direction_metadata_path(xml_path)
            )
        )
        line_metadata_by_numeric_id = load_line_segmentation_metadata_by_numeric_id(
            default_line_segmentation_metadata_path(xml_path)
        )
        for region_index, region in enumerate(root.findall(".//p:TextRegion", ns)):
            textbox_label = region.get("custom") or f"textbox_label_{region_index}"
            gt_rows = []

            for textline in region.findall("./p:TextLine", ns):
                label = _extract_unicode_text(textline, ns)
                if not label:
                    continue

                structure_line_id = _extract_structure_line_id(textline)
                if structure_line_id is None:
                    continue

                image_name = f"line_{structure_line_id}.jpg"
                source_image = image_format_dir / page_id / textbox_label / image_name
                if not source_image.exists():
                    print(f"Skipping OCR training label without image: {source_image}")
                    continue

                image_rel_path = f"text-line-images/{image_name}"
                image_arcname = f"ocr-training-format/{page_id}/{textbox_label}/{image_rel_path}"
                try:
                    line_numeric_id = int(structure_line_id)
                    has_reading_annotation = bool(
                        line_numeric_id in annotated_line_ids
                        or has_explicit_reading_direction_annotation(
                            {
                                "strategy_line_metadata": line_metadata_by_numeric_id.get(
                                    line_numeric_id, {}
                                )
                            }
                        )
                    )
                except (TypeError, ValueError):
                    has_reading_annotation = False
                if (
                    not has_reading_annotation
                    and _extract_auto_orientation_transform(textline, ns) == ROTATE_180_TRANSFORM
                ):
                    with Image.open(source_image) as line_image:
                        oriented_image = line_image.transpose(Image.Transpose.ROTATE_180)
                        image_buffer = io.BytesIO()
                        oriented_image.save(image_buffer, format="JPEG", quality=95)
                    zf.writestr(image_arcname, image_buffer.getvalue())
                else:
                    zf.write(source_image, image_arcname)
                gt_rows.append(f"{image_rel_path}\t{label}")

            if gt_rows:
                gt_arcname = f"ocr-training-format/{page_id}/{textbox_label}/gt.txt"
                zf.writestr(gt_arcname, "\n".join(gt_rows) + "\n")

@app.route('/download-results/<manuscript>', methods=['GET'])
def download_results(manuscript):
    manuscript_root = Path(UPLOAD_FOLDER) / manuscript
    manuscript_path = manuscript_root / "layout_analysis_output"
    if not manuscript_path.exists():
         return jsonify({"error": "No output found for this manuscript"}), 404
    
    xml_dir = manuscript_path / "page-xml-format"
    img_dir = manuscript_path / "image-format"
    corrections_dir = Path(UPLOAD_FOLDER) / manuscript / "node_corrections"
    xml_files = sorted(xml_dir.glob("*.xml")) if xml_dir.exists() else []
    if not xml_files:
        return jsonify({"error": "No annotated page layouts found for this manuscript"}), 404
    
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        _zip_directory(zf, xml_dir, "page-xml-format")
        _zip_directory(zf, img_dir, "image-format")
        _write_ocr_training_format_to_zip(zf, xml_files, img_dir)
        _write_resized_images_to_zip(zf, manuscript_root, manuscript_path, [path.stem for path in xml_files])
                    
        # --- START METRICS CALCULATION ---
        total_original = 0
        total_added = 0
        total_removed = 0
        total_final = 0
        total_pages_corrected = 0
        
        if corrections_dir.exists():
            json_files = list(corrections_dir.glob("*.json"))
            total_pages_corrected = len(json_files)
            for f in json_files:
                try:
                    with open(f, 'r') as jf:
                        cdata = json.load(jf)
                        total_original += cdata.get("original_nodes", 0)
                        total_added += cdata.get("nodes_added", 0)
                        total_removed += cdata.get("nodes_removed", 0)
                        total_final += cdata.get("final_nodes", 0)
                except Exception as e:
                    print(f"Error reading metrics file {f}: {e}")
                    
        # Standard Definitions for object detection task via user correction:
        # TP = Original nodes that were kept (Original - Removed)
        # FP = Original nodes that user removed
        # FN = Missed nodes that user had to manually add
        tp = max(0, total_original - total_removed)
        fp = total_removed
        fn = total_added
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics_data = {
            "manuscript": manuscript,
            "total_pages_corrected": total_pages_corrected,
            "total_original_nodes": total_original,
            "total_nodes_added_by_user": total_added,
            "total_nodes_removed_by_user": total_removed,
            "total_final_nodes": total_final,
            "metrics": {
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4)
            }
        }
        
        # Write directly to the ZIP
        zf.writestr('node_metrics.json', json.dumps(metrics_data, indent=4))
        # --- END METRICS CALCULATION ---

    memory_file.seek(0)
    return send_file(
        memory_file, 
        mimetype='application/zip', 
        as_attachment=True, 
        download_name=f'{manuscript}_results.zip'
    )

@app.route('/save-overlay/<manuscript>/<page>', methods=['POST'])
def save_overlay(manuscript, page):
    try:
        data = request.json
        manuscript_path = Path(UPLOAD_FOLDER) / manuscript
        
        dimensions = data.get('dimensions', None)
        
        # 1. Load Original Image
        original_dir = manuscript_path / "images"
        orig_img_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.JPG', '.JPEG', '.PNG', '.AVIF', '.avif', '.jp2']:
            candidate = original_dir / f"{page}{ext}"
            if candidate.exists():
                orig_img_path = candidate
                break
                
        if not orig_img_path:
            orig_img_path = manuscript_path / "images_resized" / f"{page}.jpg"
            if not orig_img_path.exists():
                return jsonify({"error": "Image not found"}), 404
                
        # Open as RGBA for proper blending
        img = Image.open(orig_img_path).convert("RGBA")
        
        # 2. Resize to match graph coordinates
        if dimensions and len(dimensions) == 2:
            target_size = (int(dimensions[0]), int(dimensions[1]))
            img = img.resize(target_size, Image.Resampling.LANCZOS)
        
        # 3. Simulate Frontend CSS: opacity: 0.7 over background-color: #121212
        # #121212 in RGB is (18, 18, 18)
        background = Image.new("RGBA", img.size, (18, 18, 18, 255))
        
        # Set image opacity to 0.7 (255 * 0.7 = 178)
        img_alpha = img.copy()
        img_alpha.putalpha(178)
        
        # Alpha composite to blend the dark background and the image
        img = Image.alpha_composite(background, img_alpha).convert("RGB")
        draw = ImageDraw.Draw(img)
        
        graph = data.get('graph', {})
        nodes = graph.get('nodes', [])
        edges = graph.get('edges', [])
        
        # Compensation for scaleFactor = 0.7 in the UI
        # UI edge width = 4  --> 4 / 0.7 = 5.7 (round to 6)
        # UI node radius = 7 --> 7 / 0.7 = 10
        line_width = 6
        r = 10
        
        # 4. Draw edges (using UI colors: #f44336 or #FF0000)
        for edge in edges:
            source_idx = edge.get('source')
            target_idx = edge.get('target')
            
            if source_idx < len(nodes) and target_idx < len(nodes):
                n1 = nodes[source_idx]
                n2 = nodes[target_idx]
                color = "#f44336" if edge.get('modified') else "#FF0000"
                draw.line([(n1['x'], n1['y']), (n2['x'], n2['y'])], fill=color, width=line_width)
                
        # 5. Draw nodes (using UI color: #000000 / Black)
        for node in nodes:
            x, y = node['x'], node['y']
            draw.ellipse([(x-r, y-r), (x+r, y+r)], fill="#000000")
            
        export_dir = manuscript_path / "overlay_exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        
        save_path = export_dir / f"{page}_overlay.jpg"
        img.save(save_path, "JPEG", quality=95)
        
        print(f"[{page}] Overlay successfully saved to {save_path}")
        return jsonify({"message": "Overlay saved successfully", "path": str(save_path)})
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
        
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
