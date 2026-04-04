# app.py
import os
import sys
import torch
import uuid

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
import google.generativeai as genai
import glob

import xml.etree.ElementTree as ET
import numpy as np
import cv2
from os.path import isdir, join
import collections
import math
import difflib
import re
from dotenv import load_dotenv
import concurrent.futures
load_dotenv() 
import traceback
from PIL import Image, ImageDraw, ImageOps

from google.api_core import retry
from google.generativeai.types import HarmCategory, HarmBlockThreshold


# from recognition.recognize_manuscript_text import recognize_manuscript_text
# cd recognition
# python recognize_manuscript_text.py complex_layout



# Import your existing pipelines
from inference import process_new_manuscript
from gnn_inference import run_gnn_prediction_for_page, generate_xml_and_images_for_page
from scale_config import load_scale_config, restore_x, restore_y, save_scale_config
from segment_from_point_clusters import extract_line_images_with_local_fill, prepare_image_for_line_extraction
from segmentation.utils import load_images_from_folder

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = './input_manuscripts'
MODEL_CHECKPOINT = "./pretrained_gnn/v2.pt"
DATASET_CONFIG = "./pretrained_gnn/gnn_preprocessing_v2.yaml"
UPLOAD_JOBS = {}
UPLOAD_JOBS_LOCK = threading.Lock()
WORD_EXPORT_HEIGHT = 101
MAX_WORD_EXPORT_WORKERS = min(8, max(1, (os.cpu_count() or 1)))
PREPARED_PAGE_CACHE = {}
MAX_PREPARED_PAGE_CACHE_ITEMS = 8


def set_upload_job(job_id, **updates):
    with UPLOAD_JOBS_LOCK:
        job = UPLOAD_JOBS.setdefault(job_id, {})
        job.update(updates)
        return dict(job)


def get_upload_job(job_id):
    with UPLOAD_JOBS_LOCK:
        job = UPLOAD_JOBS.get(job_id)
        return dict(job) if job else None


def progress_payload(status, stage, percent, message, completed=None, total=None, **extra):
    payload = {
        "status": status,
        "stage": stage,
        "progressPercent": int(max(0, min(100, round(percent)))),
        "message": message,
    }
    if completed is not None:
        payload["completed"] = completed
    if total is not None:
        payload["total"] = total
    payload.update(extra)
    return payload

def parse_page_xml_polygons(xml_path):
    polygons = {}
    if not os.path.exists(xml_path):
        return polygons

    try:
        manuscript_path = Path(xml_path).parents[2]
        scale_config = load_scale_config(manuscript_path)
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
                    points = []
                    for p in points_str.strip().split(' '):
                        x, y = map(int, p.split(','))
                        points.append([
                            int(round(restore_x(x, scale_config))),
                            int(round(restore_y(y, scale_config))),
                        ])
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


def normalize_word_cuts(word_cuts):
    normalized = {}
    if not isinstance(word_cuts, dict):
        return normalized

    for line_id, cuts in word_cuts.items():
        if not isinstance(cuts, list):
            continue

        clean_cuts = []
        for cut in cuts:
            try:
                clean_cuts.append(int(round(float(cut))))
            except (TypeError, ValueError):
                continue

        normalized[str(line_id)] = sorted(set(x for x in clean_cuts if x >= 0))

    return normalized


def normalize_word_text_content(word_text_content):
    if not isinstance(word_text_content, dict):
        return {}
    return {str(line_id): '' if value is None else str(value) for line_id, value in word_text_content.items()}


def has_word_mode_annotations(word_cuts, word_text_content):
    normalized_cuts = normalize_word_cuts(word_cuts)
    normalized_text = normalize_word_text_content(word_text_content)
    return any(len(cuts) > 0 for cuts in normalized_cuts.values()) or any('_' in value for value in normalized_text.values())


def sort_line_ids(line_ids):
    def line_key(line_id):
        try:
            return (0, int(line_id))
        except (TypeError, ValueError):
            return (1, str(line_id))
    return sorted((str(line_id) for line_id in line_ids), key=line_key)


def build_line_entries(text_by_line):
    normalized = {str(line_id): '' if value is None else str(value) for line_id, value in (text_by_line or {}).items()}
    if not normalized:
        return []

    if all(str(line_id).isdigit() for line_id in normalized.keys()):
        max_index = max(int(line_id) for line_id in normalized.keys())
        entries = [''] * (max_index + 1)
        for line_id, value in normalized.items():
            entries[int(line_id)] = value
        return entries

    return [normalized[line_id] for line_id in sort_line_ids(normalized.keys())]


def natural_page_key(value):
    parts = re.split(r'(\d+)', str(value))
    key = []
    for part in parts:
        if not part:
            continue
        if part.isdigit():
            key.append((0, int(part)))
        else:
            key.append((1, part.lower()))
    return key


def save_text_annotations(manuscript_path, page, text_by_line):
    text_dir = Path(manuscript_path) / "text"
    text_dir.mkdir(parents=True, exist_ok=True)
    text_path = text_dir / f"{page}_annotated.txt"
    text_path.write_text("\n===\n".join(build_line_entries(text_by_line)), encoding="utf-8")
    return text_path


def update_page_xml_text_content(xml_path, text_content):
    xml_path = Path(xml_path)
    normalized_text = normalize_word_text_content(text_content)

    if not xml_path.exists():
        return normalized_text

    tree = ET.parse(xml_path)
    root = tree.getroot()
    namespace_uri = root.tag.split('}')[0].strip('{') if root.tag.startswith('{') else ''
    ns = {'p': namespace_uri} if namespace_uri else {}

    def qname(tag_name):
        return f"{{{namespace_uri}}}{tag_name}" if namespace_uri else tag_name

    query = ".//p:TextLine" if namespace_uri else ".//TextLine"
    changed = False

    for textline in root.findall(query, ns):
        custom_attr = textline.get('custom', '')
        if 'structure_line_id_' not in custom_attr:
            continue

        line_id = str(custom_attr.split('structure_line_id_')[1])
        if line_id not in normalized_text:
            continue

        text_equiv = textline.find('p:TextEquiv', ns) if namespace_uri else textline.find('TextEquiv')
        if text_equiv is None:
            text_equiv = ET.SubElement(textline, qname('TextEquiv'))

        unicode_elem = text_equiv.find('p:Unicode', ns) if namespace_uri else text_equiv.find('Unicode')
        if unicode_elem is None:
            unicode_elem = ET.SubElement(text_equiv, qname('Unicode'))

        next_text = normalized_text[line_id]
        if (unicode_elem.text or '') != next_text:
            unicode_elem.text = next_text
            changed = True

    if changed:
        tree.write(xml_path, encoding='UTF-8', xml_declaration=True)

    return get_existing_text_content(str(xml_path))["text"]


def resize_binary_word_image(word_image, target_height=WORD_EXPORT_HEIGHT):
    if word_image is None or word_image.size == 0:
        return None

    if len(word_image.shape) == 3:
        word_image = cv2.cvtColor(word_image, cv2.COLOR_BGR2GRAY)

    original_height, original_width = word_image.shape[:2]
    if original_height <= 0 or original_width <= 0:
        return None

    target_width = max(1, int(round(original_width * (target_height / original_height))))
    resized = cv2.resize(word_image, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
    resized = np.where(resized <= 127, 0, 255).astype(np.uint8)
    return resized


def remove_if_exists(path):
    path = Path(path)
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def ensure_word_mode_segments(value, cut_count):
    segments = ('' if value is None else str(value)).split('_')
    desired_segments = max(int(cut_count) + 1, 1)

    if len(segments) < desired_segments:
        segments.extend([''] * (desired_segments - len(segments)))
    elif len(segments) > desired_segments:
        preserved_segments = segments[:desired_segments - 1]
        merged_tail = ''.join(segments[desired_segments - 1:])
        segments = [*preserved_segments, merged_tail]

    return '_'.join(segments)


def save_word_mode_exports(manuscript_path, page, word_cuts, word_text_content, fallback_text_content):
    manuscript_path = Path(manuscript_path)
    word_images_root = manuscript_path / "word_images"
    images_root = word_images_root / "images"
    text_root = word_images_root / "text"
    page_images_dir = images_root / page
    labels_path = text_root / f"{page}_word_labels.txt"
    splits_path = text_root / f"{page}_splits.json"

    normalized_cuts = normalize_word_cuts(word_cuts)
    normalized_text = normalize_word_text_content(word_text_content)

    if not has_word_mode_annotations(normalized_cuts, normalized_text):
        remove_if_exists(page_images_dir)
        remove_if_exists(labels_path)
        remove_if_exists(splits_path)
        return {"saved": False, "wordCount": 0}

    images_root.mkdir(parents=True, exist_ok=True)
    text_root.mkdir(parents=True, exist_ok=True)
    remove_if_exists(page_images_dir)
    page_images_dir.mkdir(parents=True, exist_ok=True)

    line_ids = set(normalized_text.keys()) | set(normalized_cuts.keys())
    line_entries = {}
    for line_id in sort_line_ids(line_ids):
        line_entries[line_id] = ensure_word_mode_segments(
            normalized_text.get(line_id, ''),
            len(normalized_cuts.get(str(line_id), [])),
        )

    flat_words = []
    for line_id in sort_line_ids(line_entries.keys()):
        flat_words.extend(line_entries[line_id].split('_'))
    labels_path.write_text("\n".join(flat_words), encoding="utf-8")
    splits_path.write_text(json.dumps(normalized_cuts, ensure_ascii=False, indent=2), encoding="utf-8")

    xml_path = manuscript_path / "layout_analysis_output" / "page-xml-format" / f"{page}.xml"
    page_image_path = manuscript_path / "images_resized" / f"{page}.jpg"
    polygons = parse_page_xml_polygons(str(xml_path))
    page_image = cv2.imread(str(page_image_path), cv2.IMREAD_COLOR)

    if page_image is None:
        print(f"[{page}] Could not load resized page image for word export: {page_image_path}")
        return {"saved": True, "wordCount": 0}

    page_image = prepare_image_for_line_extraction(page_image)

    line_ids_sorted = sort_line_ids(line_entries.keys())

    def export_line(line_id):
        polygon = polygons.get(str(line_id))
        if not polygon:
            return 0

        polygon_np = np.array(polygon, dtype=np.int32)
        line_images = extract_line_images_with_local_fill(page_image, polygon_np)
        binary_line_image = line_images.get("binary")
        if binary_line_image is None or binary_line_image.size == 0:
            return 0

        if len(binary_line_image.shape) == 3:
            binary_line_image = cv2.cvtColor(binary_line_image, cv2.COLOR_BGR2GRAY)

        x_coords = [point[0] for point in polygon]
        min_x = min(x_coords)
        max_x = max(x_coords)
        coord_width = max(max_x - min_x, 1)
        line_width = binary_line_image.shape[1]

        scaled_cuts = []
        for cut in normalized_cuts.get(str(line_id), []):
            if not (min_x < cut < max_x):
                continue
            scaled_cut = int(round((cut - min_x) * (line_width / coord_width)))
            scaled_cut = max(0, min(line_width, scaled_cut))
            scaled_cuts.append(scaled_cut)
        scaled_cuts = sorted(set(scaled_cuts))

        boundaries = [0, *scaled_cuts, line_width]
        try:
            line_index_value = int(line_id)
            line_prefix = f"{line_index_value:03d}"
        except ValueError:
            line_prefix = str(line_id)

        line_word_count = 0
        for word_index in range(len(boundaries) - 1):
            slice_width = boundaries[word_index + 1] - boundaries[word_index]
            if slice_width <= 0:
                continue

            word_image = binary_line_image[:, boundaries[word_index]:boundaries[word_index + 1]]
            resized_word_image = resize_binary_word_image(word_image, WORD_EXPORT_HEIGHT)
            if resized_word_image is None:
                continue

            word_filename = f"{line_prefix}_{word_index:03d}.png"
            cv2.imwrite(str(page_images_dir / word_filename), resized_word_image)
            line_word_count += 1

        return line_word_count

    total_words = 0
    if len(line_ids_sorted) <= 1:
        for line_id in line_ids_sorted:
            total_words += export_line(line_id)
    else:
        worker_count = min(MAX_WORD_EXPORT_WORKERS, len(line_ids_sorted))
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(export_line, line_id) for line_id in line_ids_sorted]
            for future in concurrent.futures.as_completed(futures):
                total_words += future.result()

    return {"saved": True, "wordCount": total_words}


def get_prepared_page_image(page_image_path):
    page_image_path = Path(page_image_path)
    if not page_image_path.exists():
        return None

    cache_key = (str(page_image_path), page_image_path.stat().st_mtime_ns)
    cached_image = PREPARED_PAGE_CACHE.get(cache_key)
    if cached_image is not None:
        return cached_image

    page_image = cv2.imread(str(page_image_path), cv2.IMREAD_COLOR)
    if page_image is None:
        return None

    prepared_image = prepare_image_for_line_extraction(page_image)
    PREPARED_PAGE_CACHE[cache_key] = prepared_image

    while len(PREPARED_PAGE_CACHE) > MAX_PREPARED_PAGE_CACHE_ITEMS:
        PREPARED_PAGE_CACHE.pop(next(iter(PREPARED_PAGE_CACHE)))

    return prepared_image


def load_word_annotation_data(manuscript_path, page):
    annotation_path = Path(manuscript_path) / "word_annotations" / f"{page}.json"
    default_payload = {"wordCuts": {}, "wordTextContent": {}}

    if not annotation_path.exists():
        return default_payload

    try:
        with open(annotation_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        if isinstance(data, dict) and "wordCuts" in data:
            return {
                "wordCuts": normalize_word_cuts(data.get("wordCuts", {})),
                "wordTextContent": normalize_word_text_content(data.get("wordTextContent", {})),
            }

        return {
            "wordCuts": normalize_word_cuts(data if isinstance(data, dict) else {}),
            "wordTextContent": {},
        }
    except Exception as exc:
        print(f"Error loading word annotation data for {page}: {exc}")
        return default_payload


def save_word_annotation_data(manuscript_path, page, word_cuts, word_text_content, update=False):
    if not update:
        return load_word_annotation_data(manuscript_path, page)

    annotation_dir = Path(manuscript_path) / "word_annotations"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    annotation_path = annotation_dir / f"{page}.json"
    normalized_cuts = normalize_word_cuts(word_cuts)
    normalized_text = normalize_word_text_content(word_text_content)
    if not has_word_mode_annotations(normalized_cuts, normalized_text):
        normalized_text = {}

    payload = {
        "wordCuts": normalized_cuts,
        "wordTextContent": normalized_text,
    }

    with open(annotation_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    return payload


def get_ocr_context():
    """
    Singleton to load the OCR model and config only once.
    """
    global OCR_GLOBAL_CONTEXT
    if OCR_GLOBAL_CONTEXT is not None:
        return OCR_GLOBAL_CONTEXT

    if not os.path.exists(OCR_MODEL_PATH):
        print(f"Error: OCR Model not found at {OCR_MODEL_PATH}")
        return None

    try:
        print("Loading Local OCR Model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        config = get_model_config(OCR_MODEL_PATH)
        model, converter = load_ocr_model(config, device)
        
        OCR_GLOBAL_CONTEXT = {
            'model': model,
            'converter': converter,
            'config': config,
            'device': device
        }
        print("Local OCR Model Loaded Successfully.")
        return OCR_GLOBAL_CONTEXT
    except Exception as e:
        print(f"Failed to load OCR model: {e}")
        import traceback
        traceback.print_exc()
        return None

def _run_local_recognition_internal(manuscript, page):
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
    
    # Image search paths: Look in original images and resized images
    image_dirs = [
        str(base_path / "images"),
        str(base_path / "images_resized")
    ]

    if not xml_path.exists():
        print(f"[{page}] XML file not found: {xml_path}")
        return {}

    # 2. Get Model Context
    ctx = get_ocr_context()
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
        return get_existing_text_content(str(xml_path))

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Local Recognition Error: {e}")
        return {}


def run_upload_job(job_id, manuscript_name, manuscript_path, longest_side, min_distance, x_scale, y_scale, custom_scale_enabled):
    try:
        set_upload_job(
            job_id,
            **progress_payload(
                status="running",
                stage="starting",
                percent=0,
                message="Starting manuscript processing...",
            ),
        )

        scale_config = save_scale_config(
            manuscript_path,
            enabled=custom_scale_enabled,
            x_scale=x_scale,
            y_scale=y_scale,
        )

        def on_progress(stage, completed, total, message):
            total = max(total or 1, 1)
            percent = (completed / total) * 100
            set_upload_job(
                job_id,
                **progress_payload(
                    status="running",
                    stage=stage,
                    percent=percent,
                    message=message,
                    completed=completed,
                    total=total,
                    manuscriptName=manuscript_name,
                ),
            )

        process_new_manuscript(
            manuscript_path,
            target_longest_side=longest_side,
            min_distance=min_distance,
            scale_config=scale_config,
            progress_callback=on_progress,
        )

        processed_pages = [
            f.name.replace("_dims.txt", "")
            for f in sorted(Path(manuscript_path).glob("gnn-dataset/*_dims.txt"))
        ]

        set_upload_job(
            job_id,
            **progress_payload(
                status="completed",
                stage="completed",
                percent=100,
                message="Processing complete.",
                completed=len(processed_pages),
                total=len(processed_pages) or 1,
                pages=processed_pages,
                scaleConfig=scale_config,
                manuscriptName=manuscript_name,
            ),
        )
    except Exception as e:
        traceback.print_exc()
        set_upload_job(
            job_id,
            **progress_payload(
                status="failed",
                stage="failed",
                percent=100,
                message=str(e),
                manuscriptName=manuscript_name,
            ),
            error=str(e),
        )

@app.route('/upload', methods=['POST'])
def upload_manuscript():
    manuscript_name = request.form.get('manuscriptName', 'default_manuscript')
    longest_side = int(request.form.get('longestSide', 2500))
    min_distance = int(request.form.get('minDistance', 20)) 
    custom_scale_enabled = request.form.get('charSpacingMoreThanLineSpacing', 'false').lower() == 'true'
    x_scale = request.form.get('xScale', 0.25)
    y_scale = request.form.get('yScale', 0.5)
    
    manuscript_path = os.path.join(UPLOAD_FOLDER, manuscript_name)
    images_path = os.path.join(manuscript_path, "images")
    
    files = request.files.getlist('images')
    if not files:
        return jsonify({"error": "No files uploaded"}), 400

    if os.path.exists(manuscript_path):
        shutil.rmtree(manuscript_path)
    os.makedirs(images_path)

    for file in files:
        if file.filename:
            file.save(os.path.join(images_path, file.filename))

    try:
        job_id = str(uuid.uuid4())
        set_upload_job(
            job_id,
            **progress_payload(
                status="queued",
                stage="queued",
                percent=0,
                message=f"Upload complete. Queued {len(files)} file(s) for processing.",
                completed=0,
                total=len(files),
                manuscriptName=manuscript_name,
            ),
        )

        thread = threading.Thread(
            target=run_upload_job,
            args=(job_id, manuscript_name, manuscript_path, longest_side, min_distance, x_scale, y_scale, custom_scale_enabled),
            daemon=True,
        )
        thread.start()

        return jsonify({
            "message": "Upload received. Processing started.",
            "jobId": job_id,
            "manuscriptName": manuscript_name,
        }), 202
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/upload-progress/<job_id>', methods=['GET'])
def get_upload_progress(job_id):
    job = get_upload_job(job_id)
    if not job:
        return jsonify({"error": "Upload job not found"}), 404
    return jsonify(job)

@app.route('/manuscript/<name>/pages', methods=['GET'])
def get_pages(name):
    """
    Returns list of pages and the ID of the most recently edited page based on XML mtime.
    """
    manuscript_path = Path(UPLOAD_FOLDER) / name
    dataset_path = manuscript_path / "gnn-dataset"
    if not dataset_path.exists():
        return jsonify({"pages": [], "last_edited": None}), 404
    
    pages = sorted(
        [f.name.replace("_dims.txt", "") for f in dataset_path.glob("*_dims.txt")],
        key=natural_page_key,
    )
    
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
        word_annotation_data = {"wordCuts": {}, "wordTextContent": {}}
        
        if xml_path.exists():
            polygons = parse_page_xml_polygons(str(xml_path))
            existing_data = get_existing_text_content(str(xml_path))
        word_annotation_data = load_word_annotation_data(manuscript_path, page)

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
            "wordCuts": word_annotation_data["wordCuts"],
            "wordTextContent": word_annotation_data["wordTextContent"],
        }
        return jsonify(response)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/binary-overlay/<manuscript>/<page>/<line_id>', methods=['GET'])
def get_binary_overlay(manuscript, page, line_id):
    manuscript_path = Path(UPLOAD_FOLDER) / manuscript
    xml_path = manuscript_path / "layout_analysis_output" / "page-xml-format" / f"{page}.xml"
    page_image_path = manuscript_path / "images_resized" / f"{page}.jpg"

    if not xml_path.exists():
        return jsonify({"error": f"XML not found for page {page}"}), 404

    polygons = parse_page_xml_polygons(str(xml_path))
    polygon = polygons.get(str(line_id))
    if not polygon:
        return jsonify({"error": f"Polygon not found for line {line_id}"}), 404

    prepared_page_image = get_prepared_page_image(page_image_path)
    if prepared_page_image is None:
        return jsonify({"error": f"Image not found for page {page}"}), 404

    polygon_np = np.array(polygon, dtype=np.int32)
    line_images = extract_line_images_with_local_fill(prepared_page_image, polygon_np)
    binary_line_image = line_images.get("binary")
    if binary_line_image is None or binary_line_image.size == 0:
        return jsonify({"error": "Could not generate binary overlay"}), 500

    x, y, w, h = cv2.boundingRect(polygon_np)
    polygon_shifted = polygon_np - [x, y]
    polygon_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(polygon_mask, [polygon_shifted], -1, 255, -1)

    if len(binary_line_image.shape) == 3:
        binary_line_image = cv2.cvtColor(binary_line_image, cv2.COLOR_BGR2GRAY)

    rgba_image = np.zeros((h, w, 4), dtype=np.uint8)
    rgba_image[:, :, 0] = binary_line_image
    rgba_image[:, :, 1] = binary_line_image
    rgba_image[:, :, 2] = binary_line_image
    rgba_image[:, :, 3] = polygon_mask

    success, encoded = cv2.imencode('.png', rgba_image)
    if not success:
        return jsonify({"error": "Could not encode binary overlay"}), 500

    return jsonify({"image": base64.b64encode(encoded.tobytes()).decode('utf-8')})


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


def _run_gemini_recognition_internal(manuscript, page, api_key, N=1, num_trace_points=4):
    print(f"[{page}] Starting parallel recognition with N={N}, points={num_trace_points}...")
    
    base_path = Path(UPLOAD_FOLDER) / manuscript
    xml_path = base_path / "layout_analysis_output" / "page-xml-format" / f"{page}.xml"
    img_path = base_path / "images_resized" / f"{page}.jpg"

    if not xml_path.exists() or not img_path.exists():
        return {}

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

        lines_geometry = [] 
        for textline in root.findall(".//p:TextLine", ns):
            custom_attr = textline.get('custom', '')
            if 'structure_line_id_' not in custom_attr: continue
            line_id = str(custom_attr.split('structure_line_id_')[1])

            base_elem = textline.find('p:Baseline', ns)
            if base_elem is not None and base_elem.get('points'):
                pts = [list(map(int, p.split(','))) for p in base_elem.get('points').strip().split(' ')]
                pts.sort(key=lambda k: k[0])
            else: continue

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

        if not lines_geometry: return {}

        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel('gemini-2.5-flash')

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
                "Transcribe in Unicode Devanagari. Preserve original spelling (Sandhi).\n"
                "Output a JSON array of objects with 'id' and 'text'.\n\n"
                "REGIONS:\n"
            )
            for item in regions_payload:
                prompt_text += f"ID: {item['id']} | Trace: {item['trace']}\n"

            try:
                # Use higher temperature for ensemble diversity if N > 1, else greedy (0.2)
                run_temperature = 0.7 if N > 1 else 0.2
                
                response = model.generate_content(
                    [pil_img, prompt_text],
                    generation_config={"response_mime_type": "application/json", "temperature": run_temperature}
                )
                data = json.loads(response.text)
                if isinstance(data, dict) and "transcriptions" in data: data = data["transcriptions"]
                return {str(i['id']): str(i['text']).strip() for i in data if 'id' in i}
            except Exception as e:
                print(f"Sample {sample_idx} error: {e}")
                return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=N) as executor:
            future_to_idx = {executor.submit(sample_worker, i): i for i in range(N)}
            all_samples_results = [f.result() for f in concurrent.futures.as_completed(future_to_idx) if f.result()]

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

                        if lid in final_confidences:
                            conf_str = ",".join(map(str, final_confidences[lid]))
                            current_custom = te.get('custom', '')
                            new_custom = f"confidences:{conf_str}" 
                            te.set('custom', new_custom)
                        changed = True
            
            if changed:
                tree.write(xml_path, encoding='UTF-8', xml_declaration=True)
                print(f"[{page}] XML updated with robust ensemble text.")

        return { "text": final_map, "confidences": final_confidences }

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Internal Recognition Error: {e}")
        return {}





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
    modifications = data.get('modifications', [])
    
    # --- START OF NODE CORRECTION LOGGING ---
    try:
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
    textbox_labels = data.get('textboxLabels')
    nodes_data = graph_data.get('nodes', [])
    text_content = data.get('textContent') 
    word_text_content = data.get('wordTextContent', {})
    word_cuts = data.get('wordCuts', {})
    word_mode_active = bool(data.get('wordModeActive', False))
    recognition_mode_active = bool(data.get('recognitionModeActive', False))
    
    run_recognition = data.get('runRecognition', False)
    api_key = data.get('apiKey', None)
    # --- NEW: Get engine choice ---
    recognition_engine = data.get('recognitionEngine', 'local') 

    if not textline_labels or not graph_data:
        return jsonify({"error": "Missing labels or graph data"}), 400

    try:
        xml_path = manuscript_path / "layout_analysis_output" / "page-xml-format" / f"{page}.xml"
        use_fast_recognition_save = recognition_mode_active and not modifications and xml_path.exists()

        if use_fast_recognition_save:
            saved_text_data = update_page_xml_text_content(xml_path, text_content)
            result = {"status": "success", "lines": len(saved_text_data), "fastSave": True}
        else:
            result = generate_xml_and_images_for_page(
                str(manuscript_path),
                page,
                textline_labels,
                graph_data['edges'],
                { 
                    'BINARIZE_THRESHOLD': 0.5098,
                    'BBOX_PAD_V': 0.7,
                    'BBOX_PAD_H': 0.5,
                    'CC_SIZE_THRESHOLD_RATIO': 0.4
                },
                textbox_labels=textbox_labels,
                nodes=nodes_data,
                text_content=text_content 
            )
            saved_text_data = get_existing_text_content(str(xml_path))["text"] if xml_path.exists() else normalize_word_text_content(text_content)

        saved_word_annotation_payload = save_word_annotation_data(
            manuscript_path,
            page,
            word_cuts,
            word_text_content,
            update=word_mode_active,
        )

        save_text_annotations(manuscript_path, page, saved_text_data)
        save_word_mode_exports(
            manuscript_path,
            page,
            saved_word_annotation_payload["wordCuts"],
            saved_word_annotation_payload["wordTextContent"],
            saved_text_data,
        )

        if run_recognition: 
            # --- MODIFIED: Robust background task with engine switch & logging ---
            def background_task(m, p, k, engine):
                print(f"[{p}] Starting background auto-recognition. Engine: {engine}")
                try:
                    if engine == 'gemini':
                        if not k:
                            print(f"[{p}] ERROR: Gemini API key is missing. Aborting recognition.")
                            return
                        _run_gemini_recognition_internal(m, p, k)
                    else:
                        _run_local_recognition_internal(m, p)
                    print(f"[{p}] Background auto-recognition completed successfully.")
                except Exception as e:
                    print(f"[{p}] ERROR in background auto-recognition: {e}")
                    traceback.print_exc()

            thread = threading.Thread(target=background_task, args=(manuscript, page, api_key, recognition_engine), daemon=True)
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
    api_key = data.get('apiKey')
    # --- NEW: Get engine choice ---
    recognition_engine = data.get('recognitionEngine', 'local')
    
    print(f"[{page}] Manual recognition requested using engine: {recognition_engine}")

    if recognition_engine == 'gemini':
        if not api_key:
            return jsonify({"error": "API Key required for Gemini"}), 400
        result = _run_gemini_recognition_internal(manuscript, page, api_key)
    else:
        result = _run_local_recognition_internal(manuscript, page)

    return jsonify(result)
    
@app.route('/save-graph/<manuscript>/<page>', methods=['POST'])
def save_generated_graph(manuscript, page):
    return jsonify({"status": "ok"})

@app.route('/download-results/<manuscript>', methods=['GET'])
def download_results(manuscript):
    manuscript_path = Path(UPLOAD_FOLDER) / manuscript / "layout_analysis_output"
    if not manuscript_path.exists():
         return jsonify({"error": "No output found for this manuscript"}), 404
    
    xml_dir = manuscript_path / "page-xml-format"
    img_dir = manuscript_path / "image-format"
    flat_img_dir = manuscript_path / "line_images_flat"
    flat_gray_img_dir = manuscript_path / "line_images_flat_grayscale_mode"
    corrections_dir = Path(UPLOAD_FOLDER) / manuscript / "node_corrections"
    
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        if xml_dir.exists():
            for root, dirs, files in os.walk(xml_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.join('page-xml-format', os.path.relpath(file_path, xml_dir))
                    zf.write(file_path, arcname)
        if img_dir.exists():
            for root, dirs, files in os.walk(img_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.join('image-format', os.path.relpath(file_path, img_dir))
                    zf.write(file_path, arcname)
        if flat_img_dir.exists():
            for root, dirs, files in os.walk(flat_img_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.join('line_images_flat', os.path.relpath(file_path, flat_img_dir))
                    zf.write(file_path, arcname)
        if flat_gray_img_dir.exists():
            for root, dirs, files in os.walk(flat_gray_img_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.join('line_images_flat_grayscale_mode', os.path.relpath(file_path, flat_gray_img_dir))
                    zf.write(file_path, arcname)
                    
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
        for ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.JPG', '.JPEG', '.PNG']:
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
