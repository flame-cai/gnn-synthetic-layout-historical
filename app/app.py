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
import google.generativeai as genai # NEW IMPORT
import glob # NEW IMPORT
from PIL import Image

# Import your existing pipelines
from inference import process_new_manuscript
from gnn_inference import run_gnn_prediction_for_page, generate_xml_and_images_for_page
from segmentation.utils import load_images_from_folder
import xml.etree.ElementTree as ET # Ensure this is imported

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = './input_manuscripts'
MODEL_CHECKPOINT = "./pretrained_gnn/v2.pt"
DATASET_CONFIG = "./pretrained_gnn/gnn_preprocessing_v2.yaml"

@app.route('/upload', methods=['POST'])
def upload_manuscript():
    """
    Step 1 & 2: Upload images, resize (inference.py), Generate Heatmaps & GNN inputs.
    """
    manuscript_name = request.form.get('manuscriptName', 'default_manuscript')
    longest_side = int(request.form.get('longestSide', 2500))
    # --- MODIFIED: Parse min_distance ---
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
        # Run Step 1-3: Resize and Generate Heatmaps/Points
        # --- MODIFIED: Pass min_distance ---
        process_new_manuscript(manuscript_path, target_longest_side=longest_side, min_distance=min_distance) 
        
        # Get list of processed pages
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
    """
    Step 4 Inference: Run GNN, get graph, return to frontend.
    """
    print("Received request for manuscript:", manuscript, "page:", page)
    manuscript_path = Path(UPLOAD_FOLDER) / manuscript
    try:
        # Run GNN Inference
        graph_data = run_gnn_prediction_for_page(
            str(manuscript_path), 
            page, 
            MODEL_CHECKPOINT, 
            DATASET_CONFIG
        )
        
        # Load Image to send to frontend
        img_path = manuscript_path / "images_resized" / f"{page}.jpg"
        
        if not img_path.exists():
            return jsonify({"error": "Image not found"}), 404

        with open(img_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            
        response = {
            "image": encoded_string,
            "dimensions": graph_data['dimensions'],
            "points": [[n['x'], n['y']] for n in graph_data['nodes']],
            "graph": graph_data,
            "textline_labels": graph_data.get('textline_labels', []),
            "textbox_labels": graph_data.get('textbox_labels', []) # Return textbox labels if they exist
        }
        return jsonify(response)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# 1. Update save_correction to receive text content
@app.route('/semi-segment/<manuscript>/<page>', methods=['POST'])
def save_correction(manuscript, page):
    data = request.json
    manuscript_path = Path(UPLOAD_FOLDER) / manuscript
    
    textline_labels = data.get('textlineLabels')
    graph_data = data.get('graph')
    textbox_labels = data.get('textboxLabels')
    nodes_data = graph_data.get('nodes')
    text_content = data.get('textContent') # <--- NEW: Get text from frontend
    
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
            text_content=text_content # <--- PASS TO LOGIC
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

    # Paths
    base_path = Path(UPLOAD_FOLDER) / manuscript
    xml_path = base_path / "layout_analysis_output" / "page-xml-format" / f"{page}.xml"
    # Use the resized image used for display/inference
    img_path = base_path / "images_resized" / f"{page}.jpg"
    
    if not xml_path.exists() or not img_path.exists():
        return jsonify({"error": "Page XML or Image not found. Please save layout first."}), 404

    # 1. Load Image to get dimensions for normalization
    try:
        pil_img = Image.open(img_path)
        img_w, img_h = pil_img.size
    except Exception as e:
        return jsonify({"error": f"Failed to load image: {str(e)}"}), 500

    # 2. Parse XML to extract Line Coordinates
    # Namespace handling is required for PAGE XML
    ns = {'p': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}
    
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        return jsonify({"error": f"Failed to parse XML: {str(e)}"}), 500

    # Data structure to hold regions: { line_id_int: [ymin, xmin, ymax, xmax] }
    regions_to_process = {}

    # Find all TextLines
    # We look for the 'custom' attribute we added in gnn_inference.py
    for textline in root.findall(".//p:TextLine", ns):
        custom_attr = textline.get('custom', '')
        
        # Extract the integer ID (format: "structure_line_id_{int}")
        if 'structure_line_id_' not in custom_attr:
            continue
            
        try:
            line_id = int(custom_attr.split('structure_line_id_')[1])
        except ValueError:
            continue

        # Get Coords
        coords_elem = textline.find('p:Coords', ns)
        if coords_elem is None:
            continue
            
        points_str = coords_elem.get('points', '')
        if not points_str:
            continue

        # Parse "x,y x,y ..." into list of tuples
        try:
            points = [list(map(int, p.split(','))) for p in points_str.strip().split(' ')]
        except ValueError:
            continue
            
        if not points:
            continue

        # 3. Calculate Bounding Box & Normalize to 0-1000
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        
        min_x, max_x = max(0, min(xs)), min(img_w, max(xs))
        min_y, max_y = max(0, min(ys)), min(img_h, max(ys))

        # Normalize logic: (val / dimension) * 1000, clipped to 0-1000
        n_ymin = int((min_y / img_h) * 1000)
        n_xmin = int((min_x / img_w) * 1000)
        n_ymax = int((max_y / img_h) * 1000)
        n_xmax = int((max_x / img_w) * 1000)

        # Clamp values
        n_ymin = max(0, min(1000, n_ymin))
        n_xmin = max(0, min(1000, n_xmin))
        n_ymax = max(0, min(1000, n_ymax))
        n_xmax = max(0, min(1000, n_xmax))

        # Store as [ymin, xmin, ymax, xmax]
        regions_to_process[line_id] = [n_ymin, n_xmin, n_ymax, n_xmax]

    if not regions_to_process:
         return jsonify({"transcriptions": {}})

    # 4. Construct Gemini Prompt
    genai.configure(api_key=api_key)
    # Using 1.5 Flash as it is optimized for high-volume multimodal tasks
    model = genai.GenerativeModel('gemini-1.5-flash') 

    # We batch all lines into one request context
    prompt_text = (
        "You are an expert OCR engine capable of spatial reasoning. "
        "I provide an image of a manuscript and a list of regions to transcribe.\n\n"
        "**Task**:\n"
        "1. Look at the specific regions defined by the bounding boxes below.\n"
        "2. The bounding boxes are in [ymin, xmin, ymax, xmax] format on a 0-1000 scale.\n"
        "3. Transcribe the handwritten text inside each region exactly.\n"
        "4. Return a raw JSON object where keys are the Region IDs provided and values are the transcriptions.\n\n"
        "**Regions**:\n"
    )

    for lid, bbox in regions_to_process.items():
        prompt_text += f"- Region ID '{lid}': {bbox}\n"

    prompt_text += "\n\n**Output JSON**:"

    try:
        # Pass Image + Prompt
        response = model.generate_content(
            [pil_img, prompt_text], 
            generation_config={"response_mime_type": "application/json"}
        )
        
        text_response = response.text
        
        # Parse JSON
        import json
        result_json = json.loads(text_response)
        
        # Ensure keys match format expected by frontend (strings of ints)
        final_results = {}
        for k, v in result_json.items():
            final_results[str(k)] = v
            
        return jsonify({"transcriptions": final_results})

    except Exception as e:
        print(f"Gemini Spatial Error: {e}")
        return jsonify({"error": str(e)}), 500



@app.route('/save-graph/<manuscript>/<page>', methods=['POST'])
def save_generated_graph(manuscript, page):
    return jsonify({"status": "ok"})

# --- NEW: Endpoint to download results ---
@app.route('/download-results/<manuscript>', methods=['GET'])
def download_results(manuscript):
    manuscript_path = Path(UPLOAD_FOLDER) / manuscript / "layout_analysis_output"
    
    if not manuscript_path.exists():
         return jsonify({"error": "No output found for this manuscript"}), 404
         
    # Directories to zip
    xml_dir = manuscript_path / "page-xml-format"
    img_dir = manuscript_path / "image-format"
    
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Add XMLs
        if xml_dir.exists():
            for root, dirs, files in os.walk(xml_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.join('page-xml-format', os.path.relpath(file_path, xml_dir))
                    zf.write(file_path, arcname)
                    
        # Add Line Images
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


# ssh -N -L 5001:localhost:5000 kartik@192.168.8.12