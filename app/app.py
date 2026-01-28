# app.py
import threading  # <--- CRITICAL IMPORT
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
from PIL import Image
import xml.etree.ElementTree as ET
import numpy as np
from os.path import isdir, join
import collections
import math
import difflib
from dotenv import load_dotenv
import concurrent.futures
load_dotenv() 




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
    """
    Parses PAGE-XML to extract polygon coordinates for each textline.
    Returns: Dict { structure_line_id: [[x,y], [x,y], ...] }
    """
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
                    # Convert "x,y x,y" -> [[x,y], [x,y]]
                    points = [list(map(int, p.split(','))) for p in points_str.strip().split(' ')]
                    polygons[line_id] = points
                    
            # Also try to grab existing text if any
            text_equiv = textline.find('p:TextEquiv/p:Unicode', ns)
            # We can optionally return this, but the main logic relies on the separate dict
            
    except Exception as e:
        print(f"Error parsing XML polygons: {e}")
        
    return polygons


def get_existing_text_content(xml_path):
    """
    Parses PAGE-XML to extract Unicode text AND confidence scores.
    Returns a dict with 'text' and 'confidences' maps.
    """
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
                
                # Extract Text
                text_equiv = textline.find('p:TextEquiv', ns)
                if text_equiv is not None:
                    uni = text_equiv.find('p:Unicode', ns)
                    if uni is not None and uni.text:
                        text_content[line_id] = uni.text
                        
                        # Extract Confidence from 'custom' attribute of TextEquiv
                        # Format expected: "confidences:0.9,0.5,..."
                        te_custom = text_equiv.get('custom', '')
                        if 'confidences:' in te_custom:
                            try:
                                # Split by 'confidences:' and take the part after it
                                # Then split by ';' in case there is other data
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
    manuscript_path = Path(UPLOAD_FOLDER) / name / "gnn-dataset"
    if not manuscript_path.exists():
        return jsonify([]), 404
    pages = sorted([f.name.replace("_dims.txt", "") for f in manuscript_path.glob("*_dims.txt")])
    return jsonify(pages)

# ----------------------------------------------------------------
# 2. UPDATE: Endpoint to return the new data fields
# ----------------------------------------------------------------
@app.route('/semi-segment/<manuscript>/<page>', methods=['GET'])
def get_page_prediction(manuscript, page):
    # print("Received request for manuscript:", manuscript, "page:", page)
    manuscript_path = Path(UPLOAD_FOLDER) / manuscript
    try:
        # 1. Run GNN Inference / Get Graph Data
        graph_data = run_gnn_prediction_for_page(
            str(manuscript_path), 
            page, 
            MODEL_CHECKPOINT, 
            DATASET_CONFIG
        )
        
        # 2. Get Image
        img_path = manuscript_path / "images_resized" / f"{page}.jpg"
        encoded_string = ""
        if img_path.exists():
            with open(img_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

        # 3. CRITICAL: Load Existing XML Data
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
            
            # Data for Recognition Mode
            "polygons": polygons, 
            "textContent": existing_data["text"],           # <--- Must match frontend expectation
            "textConfidences": existing_data["confidences"] # <--- Must match frontend expectation
        }
        return jsonify(response)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


def ensemble_text_samples(samples):
    """
    Performs character-level ensemble.
    Returns: (consensus_string, confidence_scores_list)
    """
    valid_samples = [s for s in samples if s and s.strip()]
    if not valid_samples:
        return "", []
    if len(valid_samples) == 1:
        # Single sample = 100% confidence for all chars
        return valid_samples[0], [1.0] * len(valid_samples[0])

    valid_samples.sort(key=len)
    pivot_idx = len(valid_samples) // 2
    pivot = valid_samples[pivot_idx]
    
    GAP_TOKEN = "__GAP__"
    total_samples = len(valid_samples) # N
    
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
        # For multi-char chunks (insertions), append conf for each char
        for c in char_str:
            result_chars.append(c)
            result_confidences.append(conf)

    # Handle start insertions
    if -1 in insertions:
        best_ins, count = insertions[-1].most_common(1)[0]
        # Heuristic: Add +1 to count for the pivot's implied "nothing here" if needed, 
        # but generally insertions are voted by 'others'. Let's trust the count.
        # We assume the pivot voted "nothing", so count is strictly from others.
        append_result(best_ins, count)

    for i in range(len(pivot)):
        # Pivot position
        best_char, count = grid[i].most_common(1)[0]
        if best_char != GAP_TOKEN:
            append_result(best_char, count)
            
        # Insertions after
        if i in insertions:
            best_ins, count = insertions[i].most_common(1)[0]
            append_result(best_ins, count)

    return "".join(result_chars), result_confidences



def _run_gemini_recognition_internal(manuscript, page, api_key, N=5):
    """
    IMPORTANT: api_key arg is unused; we load GEMINI_API_KEY locally from environment.
    Internal helper to run recognition on a specific page.
    Performs N sampling calls to Gemini IN PARALLEL with varied traces.
    Uses Character-Level Ensemble (CER-inspired) to merge results.
    """
    print(f"[{page}] Starting background recognition with N={N} samples (Parallel)...")
    base_path = Path(UPLOAD_FOLDER) / manuscript
    xml_path = base_path / "layout_analysis_output" / "page-xml-format" / f"{page}.xml"
    img_path = base_path / "images_resized" / f"{page}.jpg"

    if not xml_path.exists() or not img_path.exists():
        print(f"[{page}] Skipping: XML or Image missing.")
        return {}

    try:
        pil_img = Image.open(img_path)
        img_w, img_h = pil_img.size
        
        ns = {'p': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}
        ET.register_namespace('', ns['p']) 
        
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # --- 1. GEOMETRY EXTRACTION ---
        lines_geometry = [] 
        for textline in root.findall(".//p:TextLine", ns):
            custom_attr = textline.get('custom', '')
            if 'structure_line_id_' not in custom_attr: continue
            line_id = str(custom_attr.split('structure_line_id_')[1])

            # Get Baseline (Orientation Truth)
            baseline_pts = []
            base_elem = textline.find('p:Baseline', ns)
            if base_elem is not None and base_elem.get('points'):
                try:
                    baseline_pts = [list(map(int, p.split(','))) for p in base_elem.get('points').strip().split(' ')]
                    baseline_pts.sort(key=lambda k: k[0]) 
                except ValueError: pass

            # Get Polygon (Height Truth)
            coords_elem = textline.find('p:Coords', ns)
            poly_pts = []
            if coords_elem is not None and coords_elem.get('points'):
                try:
                    poly_pts = [list(map(int, p.split(','))) for p in coords_elem.get('points').strip().split(' ')]
                except ValueError: pass
            
            if not poly_pts and not baseline_pts: continue

            # Fallback Spine
            if not baseline_pts and poly_pts:
                xs, ys = [p[0] for p in poly_pts], [p[1] for p in poly_pts]
                sorted_poly = sorted(zip(xs, ys), key=lambda k: k[0])
                p_mid = [int(sum(xs)/len(xs)), int(sum(ys)/len(ys))]
                baseline_pts = [[sorted_poly[0][0], sorted_poly[0][1]], p_mid, [sorted_poly[-1][0], sorted_poly[-1][1]]]

            # Estimate Height
            poly_ys = [p[1] for p in poly_pts] if poly_pts else [p[1] for p in baseline_pts]
            height_px = max(10, min(max(poly_ys) - min(poly_ys), 200)) if poly_ys else 20

            lines_geometry.append({ "id": line_id, "baseline": baseline_pts, "height": height_px })

        if not lines_geometry: return {}

        # --- 2. PARALLEL SAMPLING & API CALLS ---
        
        local_api_key = os.getenv("GEMINI_API_KEY")
        if not local_api_key:
            print(f"[{page}] No GEMINI_API_KEY found in environment variables.")
            return {}

        genai.configure(api_key=local_api_key)
        # We can share the model instance across threads usually, or create new ones.
        # Sharing is generally thread-safe for generate_content.
        model = genai.GenerativeModel('gemini-2.5-flash') 

        def normalize(x, y):
            n_y = int((y / img_h) * 1000)
            n_x = int((x / img_w) * 1000)
            return max(0, min(1000, n_y)), max(0, min(1000, n_x))

        # Helper function to run in a thread
        def process_single_sample(sample_idx):
            try:
                regions_payload = []
                
                # Vertical Shift Logic: N=1 -> 0.3 (Baseline+30%), N>1 -> 0.0 to 0.7
                shift_ratios = [0.3] if N == 1 else [i * (0.7 / (N - 1)) for i in range(N)]
                current_shift_ratio = shift_ratios[sample_idx]

                for line in lines_geometry:
                    pts = line['baseline']
                    h = line['height']
                    
                    # Interpolate 3 points
                    if len(pts) >= 3:
                        trace_raw = [pts[0], pts[len(pts)//2], pts[-1]]
                    elif len(pts) == 2:
                        mid_x, mid_y = (pts[0][0] + pts[1][0]) // 2, (pts[0][1] + pts[1][1]) // 2
                        trace_raw = [pts[0], [mid_x, mid_y], pts[-1]]
                    else:
                        trace_raw = [pts[0], pts[0], pts[0]]

                    # Apply Shift (Upwards relative to page)
                    shift_px = int(h * current_shift_ratio)
                    shifted_trace = [[px, py - shift_px] for px, py in trace_raw]

                    gemini_trace = []
                    for px, py in shifted_trace:
                        ny, nx = normalize(px, py)
                        gemini_trace.extend([ny, nx])
                    
                    regions_payload.append({
                        "id": line['id'],
                        "trace": gemini_trace,
                        "sort_y": trace_raw[0][1]
                    })

                regions_payload.sort(key=lambda k: k['sort_y'])

                prompt_text = (
                    "You are an expert paleographer and OCR engine specialized in historical Sanskrit manuscripts.\n"
                    "I have provided an image of a manuscript page. Your task is to perform visual grounding OCR: "
                    "transcribe the handwritten Devanagari text found at specific spatial locations defined by 'Path Traces'.\n"
                    "The coordinates are normalized on a 0-1000 scale (where [0,0] is top-left and [1000,1000] is bottom-right) "
                    "to precisely map the text line locations on the image.\n"
                    "For each path trace [y_start, x_start, y_mid, x_mid, y_end, x_end], transcribe the text that sits along this curve.\n"
                    "Focus strictly on the visual line indicated by the trace; ignore text from lines above or below.\n"
                    "Output a JSON array of objects with 'id' and 'text'.\n\n"
                    "REGIONS:\n"
                )
                for item in regions_payload:
                    prompt_text += f"ID: {item['id']} | Trace: {item['trace']}\n"

                # API Call
                # print(f"[{page}] Thread {sample_idx+1}/{N} sending request...")
                response = model.generate_content(
                    [pil_img, prompt_text],
                    generation_config={"response_mime_type": "application/json", "temperature": 0.1}
                )
                
                result_list = json.loads(response.text.replace("```json", "").replace("```", ""))
                
                if isinstance(result_list, dict) and "transcriptions" in result_list:
                    result_list = result_list["transcriptions"]
                elif isinstance(result_list, dict):
                    result_list = [{"id": k, "text": v} for k, v in result_list.items()]

                sample_map = {str(i['id']): str(i['text']).strip() for i in result_list if 'id' in i and 'text' in i}
                return sample_map

            except Exception as e:
                print(f"[{page}] Thread {sample_idx+1} failed: {e}")
                return None

        # execute in parallel
        all_samples_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=N) as executor:
            # Submit all tasks
            futures = [executor.submit(process_single_sample, i) for i in range(N)]
            
            # Wait maximum 15 seconds for the batch. 
            # Any task not finished by then is left in 'not_done' and ignored.
            done, not_done = concurrent.futures.wait(futures, timeout=50)
            
            # Process only the ones that finished in time
            for future in done:
                try:
                    res = future.result()
                    if res:
                        all_samples_results.append(res)
                except Exception as exc:
                    print(f"[{page}] Thread exception: {exc}")
            
            if not_done:
                print(f"[{page}] {len(not_done)} samples timed out (>10s) and were dropped.")

        # --- 3. CHARACTER-LEVEL ENSEMBLE ---
        final_map = {}
        final_confidences = {} # Map: line_id -> list of floats
        
        texts_by_id = collections.defaultdict(list)
        for res_map in all_samples_results:
            for lid, txt in res_map.items():
                texts_by_id[lid].append(txt)

        for lid, candidates in texts_by_id.items():
            # Get text AND scores
            consensus_text, scores = ensemble_text_samples(candidates)
            if consensus_text:
                final_map[lid] = consensus_text
                final_confidences[lid] = scores
                # Log if significant divergence occurred
                unique_variants = set(candidates)
                if len(unique_variants) > 1 and N > 1:
                    print(f"[{page}] Line {lid}: Merged {len(candidates)} samples. " 
                          f"Result: {consensus_text[:15]}... (Variants: {len(unique_variants)})")

        # --- 4. UPDATE XML ---
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
                            # Preserve existing custom data if any, simply append/replace conf
                            current_custom = te.get('custom', '')
                            # Simple replacement strategy for robustness
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
    
    textline_labels = data.get('textlineLabels')
    graph_data = data.get('graph')
    textbox_labels = data.get('textboxLabels')
    nodes_data = graph_data.get('nodes')
    text_content = data.get('textContent') 
    
    # Flags for background processing
    run_recognition = data.get('runRecognition', False)
    api_key = data.get('apiKey', None) #not used, unsafe

    if not textline_labels or not graph_data:
        return jsonify({"error": "Missing labels or graph data"}), 400

    try:
        # 1. SAVE LAYOUT (Synchronous)
        # We must wait for this to finish so the XML/Images exist for the thread to read.
        # This is usually very fast (<200ms).
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

        # 2. TRIGGER RECOGNITION (Asynchronous)
        if run_recognition and api_key: #this api_key is placeholder (it is ununsed in the actual function)
            
            # Wrapper to log start/finish in backend console
            def background_task(m, p, k):
                _run_gemini_recognition_internal(m, p, k)

            # Spawn the thread
            # daemon=True ensures the thread doesn't block server shutdown
            thread = threading.Thread(target=background_task, args=(manuscript, page, api_key), daemon=True)
            thread.start()
            
            # Let the frontend know we started it, but don't wait for the result
            result['autoRecognitionStatus'] = "processing_in_background"

        # Return immediately to unblock the UI
        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ----------------------------------------------------------------
# MODIFIED: Existing Recognize Endpoint (Wrapper)
# ----------------------------------------------------------------
@app.route('/recognize-text', methods=['POST'])
def recognize_text():
    data = request.json
    manuscript = data.get('manuscript')
    page = data.get('page')
    api_key = data.get('apiKey')
    
    if not api_key:
        return jsonify({"error": "API Key required"}), 400

    # Use the shared internal function
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

# backend
# ssh -N -L 5001:localhost:5000 kartik@192.168.8.12

# frontend
# ssh -L 8000:localhost:5173 kartik@192.168.8.12