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

# 2. Add Recognition Endpoint
@app.route('/recognize-text', methods=['POST'])
def recognize_text():
    data = request.json
    manuscript = data.get('manuscript')
    page = data.get('page')
    api_key = data.get('apiKey')
    
    if not api_key:
        return jsonify({"error": "API Key required"}), 400

    manuscript_path = Path(UPLOAD_FOLDER) / manuscript / "layout_analysis_output"
    img_dir = manuscript_path / "image-format" / page
    
    if not img_dir.exists():
        return jsonify({"error": "No line images found. Please save the layout first."}), 404

    # Configure Gemini
    genai.configure(api_key=api_key)
    
    # We use Flash for speed and cost. 
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    results = {}
    
    # Iterate through textbox folders to maintain logical grouping
    textbox_folders = sorted(list(img_dir.glob("textbox_label_*")))
    
    for tb_folder in textbox_folders:
        image_files = sorted(list(tb_folder.glob("*.jpg")))
        if not image_files:
            continue
            
        # We process one textbox at a time to give the model context 
        # but separate lines to ensure robust parsing.
        
        # Prepare inputs: List of [Image, Prompt, Image, Prompt...] is token heavy.
        # Efficient approach: List of [Image1, Image2, ... , Text Prompt]
        
        inputs = []
        file_map = [] # To map index back to filename/line_id
        
        for img_path in image_files:
            try:
                # Extract line ID from filename "line_{id}.jpg"
                line_id = img_path.stem.split('_')[1]
                
                # Load image for Gemini
                pil_img = Image.open(img_path)
                inputs.append(pil_img)
                file_map.append(line_id)
            except Exception as e:
                print(f"Skipping bad image {img_path}: {e}")

        if not inputs:
            continue

        # PROMPT ENGINEERING
        # 1. Role: Paleographer.
        # 2. Task: Transcribe.
        # 3. Output Format: Strict JSON. 
        # We ask for a list corresponding to the provided images in order.
        
        prompt = (
            "You are an expert Sanskrit paleographer. "
            "Transcribe the handwritten text in the provided textline images exactly as it appears. "
            "The images are provided in reading order (line by line). "
            "Return a raw JSON object (no markdown formatting) where keys are the indices (0, 1, 2...) "
            "corresponding to the order of images passed, and values are the transcriptions. "
            "Example: {\"0\": \"The first line text\", \"1\": \"The second line text\"}"
        )
        inputs.append(prompt)

        try:
            response = model.generate_content(inputs, generation_config={"response_mime_type": "application/json"})
            text_response = response.text
            
            # Parse JSON
            import json
            batch_result = json.loads(text_response)
            
            # Map back to line IDs
            for idx_str, text in batch_result.items():
                idx = int(idx_str)
                if idx < len(file_map):
                    real_line_id = file_map[idx]
                    results[real_line_id] = text
                    
        except Exception as e:
            print(f"Gemini Error for {tb_folder}: {e}")
            # Continue to next textbox even if one fails
            
    return jsonify({"transcriptions": results})



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