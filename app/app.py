# app.py
import threading 
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
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
from os.path import isdir, join
import collections
import math
import difflib
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
from segmentation.utils import load_images_from_folder

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = './input_manuscripts'
MODEL_CHECKPOINT = "./pretrained_gnn/v2.pt"
DATASET_CONFIG = "./pretrained_gnn/gnn_preprocessing_v2.yaml"

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


@app.route('/upload', methods=['POST'])
def upload_manuscript():
    manuscript_name = request.form.get('manuscriptName', 'default_manuscript')
    longest_side = int(request.form.get('longestSide', 2500))
    min_distance = int(request.form.get('minDistance', 20)) 
    
    manuscript_path = os.path.join(UPLOAD_FOLDER, manuscript_name)
    images_path = os.path.join(manuscript_path, "images")
    
    if os.path.exists(manuscript_path):
        shutil.rmtree(manuscript_path)
    os.makedirs(images_path)

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
        
        if xml_path.exists():
            polygons = parse_page_xml_polygons(str(xml_path))
            existing_data = get_existing_text_content(str(xml_path))

        response = {
            "image": encoded_string,
            "dimensions": graph_data['dimensions'],
            "points": [[n['x'], n['y']] for n in graph_data['nodes']],
            "graph": graph_data,
            "textline_labels": graph_data.get('textline_labels', []),
            "textbox_labels": graph_data.get('textbox_labels', []),
            "polygons": polygons, 
            "textContent": existing_data["text"],
            "textConfidences": existing_data["confidences"]
        }
        return jsonify(response)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


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
                "For each path trace [y_start, x_start, y_mid1, x_mid1, y_mid2, x_mid2, y_end, x_end], transcribe the text that sits along this curve.\n"
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

# def _run_gemini_recognition_internal(manuscript, page, api_key, N=1, num_trace_points=4):
#     """
#     Drop-in replacement for Gemini OCR.
    
#     STRATEGY:
#     1. Polygon Masking: Precise cropping of text lines using PAGE-XML coords.
#     2. Dynamic Padding: Small/Tiny crops are NOT skipped; they are padded with 
#        extra white background to meet a minimum resolution for the Vision Encoder.
#     3. Safety Override: Explicitly disables safety filters to prevent false positives 
#        on historical/religious Sanskrit content.
#     4. Robust Sampling: Handles N=1 (Low Temp) vs N>1 (High Temp) and catches 
#        API blocking errors gracefully.
#     """
#     print(f"[{page}] Starting chipped recognition with N={N}...")
    
#     base_path = Path(UPLOAD_FOLDER) / manuscript
#     xml_path = base_path / "layout_analysis_output" / "page-xml-format" / f"{page}.xml"
#     img_path = base_path / "images_resized" / f"{page}.jpg"

#     if not xml_path.exists() or not img_path.exists():
#         print(f"[{page}] Missing XML or Image file.")
#         return {}

#     try:
#         # --- 1. CONFIGURATION ---
#         genai.configure(api_key=api_key)
        
#         # CRITICAL: Disable all safety filters. 
#         # "Block None" is required because historical manuscripts often trigger 
#         # false positives for "Hate Speech" (Swastikas) or "Violence" (Depictions of War).
#         safety_settings = {
#             HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
#             HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
#             HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
#             HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
#         }

#         # Using Flash 2.0 as requested/implied (Cost effective, fast)
#         model = genai.GenerativeModel('gemini-2.0-flash') 

#         # --- 2. IMAGE & XML PROCESSING ---
#         pil_img = Image.open(img_path).convert("RGB")
#         img_w, img_h = pil_img.size
        
#         ns = {'p': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}
#         tree = ET.parse(xml_path)
#         root = tree.getroot()
        
#         # Calculate scaling between XML coords (high-res) and Input Image (resized)
#         page_elem = root.find(".//p:Page", ns)
#         xml_w = int(page_elem.get('imageWidth')) if page_elem is not None and page_elem.get('imageWidth') else img_w
#         xml_h = int(page_elem.get('imageHeight')) if page_elem is not None and page_elem.get('imageHeight') else img_h
        
#         scale_x = img_w / xml_w if xml_w > 0 else 1.0
#         scale_y = img_h / xml_h if xml_h > 0 else 1.0

#         lines_data = [] # Stores (line_id, cropped_image_object)

#         for textline in root.findall(".//p:TextLine", ns):
#             custom_attr = textline.get('custom', '')
#             if 'structure_line_id_' not in custom_attr: continue
#             line_id = str(custom_attr.split('structure_line_id_')[1])

#             # Extract Polygon Coordinates
#             coords_elem = textline.find('p:Coords', ns)
#             if coords_elem is not None and coords_elem.get('points'):
#                 points_str = coords_elem.get('points').strip()
#                 poly_pts = []
#                 for p in points_str.split(' '):
#                     try:
#                         px, py = map(int, p.split(','))
#                         poly_pts.append((int(px * scale_x), int(py * scale_y)))
#                     except: continue
                
#                 if len(poly_pts) < 3: continue

#                 # A. MASKING: Isolate text from background noise
#                 # Create a black mask, draw white polygon, composite with white background
#                 mask = Image.new('L', (img_w, img_h), 0)
#                 draw = ImageDraw.Draw(mask)
#                 draw.polygon(poly_pts, outline=255, fill=255)
                
#                 white_bg = Image.new('RGB', (img_w, img_h), (255, 255, 255))
#                 masked_img = Image.composite(pil_img, white_bg, mask)
                
#                 # B. CROPPING & PADDING LOGIC
#                 bbox = mask.getbbox()
#                 if bbox:
#                     left, upper, right, lower = bbox
                    
#                     # 1. Base Padding (Standard context)
#                     pad_l, pad_t, pad_r, pad_b = 5, 5, 5, 5
                    
#                     # 2. Minimum Size Enforcement (Handle "Small Chips")
#                     # If a chip is tiny (e.g. noise or punctuation), the model fails.
#                     # We increase padding to ensure the chip is at least 60x60px.
#                     curr_w = right - left
#                     curr_h = lower - upper
#                     MIN_DIM = 60
                    
#                     if curr_w < MIN_DIM:
#                         needed = MIN_DIM - curr_w
#                         pad_l += needed // 2
#                         pad_r += needed // 2
                    
#                     if curr_h < MIN_DIM:
#                         needed = MIN_DIM - curr_h
#                         pad_t += needed // 2
#                         pad_b += needed // 2

#                     # 3. Apply Padding (clamped to image boundaries)
#                     # Note: If clamped at edge, we might still be small, but we handle that next.
#                     crop_left = max(0, left - pad_l)
#                     crop_upper = max(0, upper - pad_t)
#                     crop_right = min(img_w, right + pad_r)
#                     crop_lower = min(img_h, lower + pad_b)
                    
#                     cropped_chip = masked_img.crop((crop_left, crop_upper, crop_right, crop_lower))
                    
#                     # 4. Final Safety Pad (Canvas Expansion)
#                     # If we hit the image edge and couldn't pad enough, the chip is still small.
#                     # We paste it onto a white canvas of MIN_DIM size.
#                     cw, ch = cropped_chip.size
#                     if cw < MIN_DIM or ch < MIN_DIM:
#                         final_w = max(cw, MIN_DIM)
#                         final_h = max(ch, MIN_DIM)
#                         new_canvas = Image.new('RGB', (final_w, final_h), (255, 255, 255))
#                         # Paste in center
#                         paste_x = (final_w - cw) // 2
#                         paste_y = (final_h - ch) // 2
#                         new_canvas.paste(cropped_chip, (paste_x, paste_y))
#                         cropped_chip = new_canvas

#                     lines_data.append((line_id, cropped_chip))

#         if not lines_data:
#             print(f"[{page}] No lines found with Coords.")
#             return {}

#         print(f"[{page}] Prepared {len(lines_data)} image chips.")

#         # --- 3. WORKER FUNCTION ---
#         line_results = collections.defaultdict(list)

#         def process_line(args):
#             l_id, img_chip = args
#             results = []
            
#             prompt = (
#                 "You are an expert Indologist and Paleographer specializing in historical handwritten Sanskrit manuscripts. "
#                 "TASK: Transcribe the Sanskrit text visible in this image fragment in Unicode Devanagari.\n"
#                 "CONTEXT: This image is a cropped strip containing exactly one line of text. "
#                 "STRICTLY IGNORE partial artifacts from lines above/below. Focus only on the central, dominant line.\n"
#                 "INSTRUCTIONS:\n"
#                 "1. Transcribe strictly in Devanagari script.\n"
#                 "2. Preserve original spelling, including Sandhi.\n"
#                 "3. Do not expand abbreviations.\n"
#                 "4. Output ONLY the transcription text string.\n"
#             )

#             for i in range(N):
#                 # Temperature: Low for precision if N=1, High for diversity if N>1
#                 temp = 0.1 if N == 1 else 0.7 
#                 try:
#                     response = model.generate_content(
#                         [img_chip, prompt],
#                         generation_config={
#                             "temperature": temp,
#                             "max_output_tokens": 1024,
#                         },
#                         safety_settings=safety_settings # APPLY SAFETY FIX
#                     )
                    
#                     # Robust Response Parsing
#                     # Checks for candidates and parts to avoid 'AttributeError' or 'ValueError'
#                     if response.candidates and response.candidates[0].content.parts:
#                         text = response.candidates[0].content.parts[0].text.strip()
#                         # Cleanup formatting
#                         text = text.replace("```json", "").replace("```", "").replace("\n", "").strip()
#                         results.append(text)
#                     else:
#                         # If blocked or empty, log specific reason
#                         reason = "UNKNOWN"
#                         if response.candidates:
#                             reason = response.candidates[0].finish_reason
#                         print(f"[{page}] Line {l_id} Sample {i} Empty. Reason: {reason}")
#                         results.append("")

#                 except Exception as e:
#                     # Log but continue (don't crash the thread)
#                     print(f"[{page}] Error Line {l_id} Sample {i}: {str(e)[:100]}")
#                     results.append("")
            
#             return l_id, results

#         # --- 4. EXECUTION ---
#         # Using ThreadPoolExecutor. 
#         # Note: Workers reduced to 5 to avoid Rate Limit (429) errors on Gemini Flash, 
#         # which can happen if sending too many images simultaneously.
#         with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
#             future_to_line = {executor.submit(process_line, item): item[0] for item in lines_data}
            
#             for future in concurrent.futures.as_completed(future_to_line):
#                 l_id = future_to_line[future]
#                 try:
#                     _, samples = future.result()
#                     line_results[l_id] = samples
#                 except Exception as e:
#                     print(f"[{page}] Critical failure on line {l_id}: {e}")

#         # --- 5. ENSEMBLE PREPARATION ---
#         all_samples_results = []
#         for i in range(N):
#             sample_map = {}
#             for l_id, samples in line_results.items():
#                 if i < len(samples) and samples[i]:
#                     sample_map[l_id] = samples[i]
#             all_samples_results.append(sample_map)

#         # --- 6. AGGREGATION & XML UPDATE ---
#         final_map = {}
#         final_confidences = {}
        
#         texts_by_id = collections.defaultdict(list)
#         for res_map in all_samples_results:
#             for lid, txt in res_map.items():
#                 texts_by_id[lid].append(txt)

#         for lid, candidates in texts_by_id.items():
#             # ensemble_text_samples is assumed to be defined globally as per original snippet
#             consensus_text, scores = ensemble_text_samples(candidates)
#             if consensus_text:
#                 final_map[lid] = consensus_text
#                 final_confidences[lid] = scores
#                 if len(set(candidates)) > 1 and N > 1:
#                     print(f"[{page}] Line {lid}: Merged {len(candidates)} samples.")

#         if final_map:
#             changed = False
#             for textline in root.findall(".//p:TextLine", ns):
#                 custom_attr = textline.get('custom', '')
#                 if 'structure_line_id_' in custom_attr:
#                     lid = str(custom_attr.split('structure_line_id_')[1])
#                     if lid in final_map:
#                         te = textline.find("p:TextEquiv", ns)
#                         if te is None: te = ET.SubElement(textline, "TextEquiv")
#                         uni = te.find("p:Unicode", ns)
#                         if uni is None: uni = ET.SubElement(te, "Unicode")
#                         uni.text = final_map[lid]

#                         if lid in final_confidences:
#                             conf_str = ",".join(map(str, final_confidences[lid]))
#                             current_custom = te.get('custom', '')
#                             # Appending confidence to custom attr
#                             new_conf_str = f"confidences:{conf_str}"
#                             if "confidences:" in current_custom:
#                                 # simplistic replace to avoid duplicate keys if rerunning
#                                 import re
#                                 te.set('custom', re.sub(r'confidences:[0-9.,]+', new_conf_str, current_custom))
#                             else:
#                                 te.set('custom', f"{current_custom} {new_conf_str}".strip())
#                         changed = True
            
#             if changed:
#                 tree.write(xml_path, encoding='UTF-8', xml_declaration=True)
#                 print(f"[{page}] XML updated with robust chipped ensemble text.")

#         return { "text": final_map, "confidences": final_confidences }

#     except Exception as e:
#         traceback.print_exc()
#         print(f"Internal Recognition Error: {e}")
#         return {}

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
    
    textline_labels = data.get('textlineLabels')
    graph_data = data.get('graph')
    textbox_labels = data.get('textboxLabels')
    nodes_data = graph_data.get('nodes')
    text_content = data.get('textContent') 
    
    run_recognition = data.get('runRecognition', False)
    api_key = data.get('apiKey', None) #not used, unsafe

    if not textline_labels or not graph_data:
        return jsonify({"error": "Missing labels or graph data"}), 400

    try:
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

        if run_recognition: 
            def background_task(m, p, k):
                _run_gemini_recognition_internal(m, p, k)

            thread = threading.Thread(target=background_task, args=(manuscript, page, None), daemon=True)
            thread.start()
            
            result['autoRecognitionStatus'] = "processing_in_background"

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
    
    if not api_key:
        return jsonify({"error": "API Key required"}), 400

    result = _run_gemini_recognition_internal(manuscript, page, api_key)
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

    memory_file.seek(0)
    return send_file(
        memory_file, 
        mimetype='application/zip', 
        as_attachment=True, 
        download_name=f'{manuscript}_results.zip'
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)