# app.py
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
    """Parses PAGE-XML to extract existing Unicode text content."""
    text_content = {}
    if not os.path.exists(xml_path):
        return text_content
        
    try:
        ns = {'p': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for textline in root.findall(".//p:TextLine", ns):
            custom_attr = textline.get('custom', '')
            if 'structure_line_id_' in custom_attr:
                line_id = str(custom_attr.split('structure_line_id_')[1])
                text_equiv = textline.find('p:TextEquiv/p:Unicode', ns)
                if text_equiv is not None and text_equiv.text:
                    text_content[line_id] = text_equiv.text
    except Exception:
        pass
    return text_content

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

@app.route('/semi-segment/<manuscript>/<page>', methods=['GET'])
def get_page_prediction(manuscript, page):
    print("Received request for manuscript:", manuscript, "page:", page)
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
        if not img_path.exists():
            return jsonify({"error": "Image not found"}), 404

        with open(img_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

        # 3. Check for Existing XML to get Polygons & Text
        # This is critical for the Recognition Mode UI
        xml_path = manuscript_path / "layout_analysis_output" / "page-xml-format" / f"{page}.xml"
        polygons = {}
        existing_text = {}
        
        if xml_path.exists():
            polygons = parse_page_xml_polygons(str(xml_path))
            existing_text = get_existing_text_content(str(xml_path))

        response = {
            "image": encoded_string,
            "dimensions": graph_data['dimensions'],
            "points": [[n['x'], n['y']] for n in graph_data['nodes']],
            "graph": graph_data,
            "textline_labels": graph_data.get('textline_labels', []),
            "textbox_labels": graph_data.get('textbox_labels', []),
            "polygons": polygons,   # <--- NEW: Send specific polygons
            "textContent": existing_text # <--- NEW: Send existing text
        }
        return jsonify(response)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/semi-segment/<manuscript>/<page>', methods=['POST'])
def save_correction(manuscript, page):
    data = request.json
    manuscript_path = Path(UPLOAD_FOLDER) / manuscript
    
    textline_labels = data.get('textlineLabels')
    graph_data = data.get('graph')
    textbox_labels = data.get('textboxLabels')
    nodes_data = graph_data.get('nodes')
    text_content = data.get('textContent') 
    
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

    base_path = Path(UPLOAD_FOLDER) / manuscript
    xml_path = base_path / "layout_analysis_output" / "page-xml-format" / f"{page}.xml"
    img_path = base_path / "images_resized" / f"{page}.jpg"
    
    if not xml_path.exists() or not img_path.exists():
        return jsonify({"error": "Page XML or Image not found. Please save layout first."}), 404

    try:
        pil_img = Image.open(img_path)
        img_w, img_h = pil_img.size
    except Exception as e:
        return jsonify({"error": f"Failed to load image: {str(e)}"}), 500

    regions_to_process = []
    ns = {'p': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}
    try:
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
            if coords_elem is None: continue
            points_str = coords_elem.get('points', '')
            if not points_str: continue

            try:
                points = [list(map(int, p.split(','))) for p in points_str.strip().split(' ')]
            except ValueError: continue
            
            if not points: continue

            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            
            # Normalize 0-1000
            n_ymin = int((min(ys) / img_h) * 1000)
            n_xmin = int((min(xs) / img_w) * 1000)
            n_ymax = int((max(ys) / img_h) * 1000)
            n_xmax = int((max(xs) / img_w) * 1000)

            n_ymin, n_ymax = sorted([max(0, min(1000, n_ymin)), max(0, min(1000, n_ymax))])
            n_xmin, n_xmax = sorted([max(0, min(1000, n_xmin)), max(0, min(1000, n_xmax))])

            regions_to_process.append({
                "id": line_id,
                "box_2d": [n_ymin, n_xmin, n_ymax, n_xmax]
            })

    except Exception as e:
        return jsonify({"error": f"XML Parsing Error: {str(e)}"}), 500

    if not regions_to_process:
         return jsonify({"transcriptions": {}})

    prompt_text = (
        "You are an expert paleographer analyzing a historical manuscript.\n"
        "Your task is to transcribe the handwritten text found inside specific bounding boxes.\n\n"
        "INPUT CONTEXT:\n"
        "The coordinates are in the format [ymin, xmin, ymax, xmax] on a scale of 0 to 1000.\n\n"
        "REGIONS TO TRANSCRIBE:\n"
    )
    for item in regions_to_process:
        prompt_text += f"- Region ID '{item['id']}' at Box: {item['box_2d']}\n"

    prompt_text += (
        "\nOUTPUT INSTRUCTIONS:\n"
        "1. Return a JSON List of objects.\n"
        "2. Each object must have two keys: 'id' (string) and 'text' (string).\n"
        "3. Do not modify the Region ID.\n"
        "4. If the text is illegible, set 'text' to an empty string.\n"
    )

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(
            [pil_img, prompt_text],
            generation_config={"response_mime_type": "application/json"}
        )
        
        raw_result = json.loads(response.text)
        result_list = []
        if isinstance(raw_result, list):
            result_list = raw_result
        elif isinstance(raw_result, dict):
            for val in raw_result.values():
                if isinstance(val, list):
                    result_list = val
                    break

        final_map = {}
        for item in result_list:
            if 'id' in item and 'text' in item:
                final_map[str(item['id'])] = item['text']

        return jsonify({"transcriptions": final_map})

    except Exception as e:
        print(f"Gemini Error: {e}")
        return jsonify({"error": str(e)}), 500

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