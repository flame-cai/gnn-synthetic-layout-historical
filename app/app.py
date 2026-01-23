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
    """
    Refined based on Google AI docs:
    1. Uses gemini-1.5-flash (optimized for multimodal speed/cost).
    2. Uses native JSON Mode for robust output.
    3. Normalizes coordinates to 0-1000 (Gemini native scale).
    """
    data = request.json
    manuscript = data.get('manuscript')
    page = data.get('page')
    api_key = data.get('apiKey')
    
    if not api_key:
        return jsonify({"error": "API Key required"}), 400

    # 1. Setup Paths
    base_path = Path(UPLOAD_FOLDER) / manuscript
    xml_path = base_path / "layout_analysis_output" / "page-xml-format" / f"{page}.xml"
    img_path = base_path / "images_resized" / f"{page}.jpg"
    
    if not xml_path.exists() or not img_path.exists():
        return jsonify({"error": "Page XML or Image not found. Please save layout first."}), 404

    # 2. Load Image & Dimensions
    try:
        pil_img = Image.open(img_path)
        img_w, img_h = pil_img.size
    except Exception as e:
        return jsonify({"error": f"Failed to load image: {str(e)}"}), 500

    # 3. Parse XML & Prepare Regions
    # We map "structure_line_id" -> [ymin, xmin, ymax, xmax]
    regions_to_process = []
    
    ns = {'p': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        for textline in root.findall(".//p:TextLine", ns):
            custom_attr = textline.get('custom', '')
            
            # Extract ID
            if 'structure_line_id_' not in custom_attr:
                continue
            try:
                line_id = str(custom_attr.split('structure_line_id_')[1])
            except IndexError:
                continue

            # Extract Coords
            coords_elem = textline.find('p:Coords', ns)
            if coords_elem is None: continue
            points_str = coords_elem.get('points', '')
            if not points_str: continue

            try:
                points = [list(map(int, p.split(','))) for p in points_str.strip().split(' ')]
            except ValueError: continue
            
            if not points: continue

            # Convert Polygon -> Bounding Box
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            
            # Normalize to 0-1000 (Gemini Native Scale)
            # Formula: int(val / dimension * 1000)
            n_ymin = int((min(ys) / img_h) * 1000)
            n_xmin = int((min(xs) / img_w) * 1000)
            n_ymax = int((max(ys) / img_h) * 1000)
            n_xmax = int((max(xs) / img_w) * 1000)

            # Clamp & Sort (Safety)
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

    # 4. Construct Prompt
    # We ask for a list of objects, which is more robust for JSON mode than dynamic keys.
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

    # 5. Call Gemini API
    try:
        genai.configure(api_key=api_key)
        
        # Use 1.5-flash (Best for OCR speed/cost)
        model = genai.GenerativeModel('gemini-2.5-flash')

        response = model.generate_content(
            [pil_img, prompt_text],
            generation_config={
                "response_mime_type": "application/json",
                # We expect a structure like: [{"id": "1", "text": "abc"}, ...]
            }
        )
        
        # 6. Process Response
        # Because we used response_mime_type, .text is guaranteed to be JSON (no markdown backticks)
        raw_result = json.loads(response.text)
        
        # Convert List back to Map for Frontend: { "1": "abc", "2": "def" }
        # Handle cases where Gemini might wrap the list in a root key like {"result": [...]}
        result_list = []
        if isinstance(raw_result, list):
            result_list = raw_result
        elif isinstance(raw_result, dict):
            # Try to find the first list value
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
        # Detailed error for debugging
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

# backend
# ssh -N -L 5001:localhost:5000 kartik@192.168.8.12

# frontend
# ssh -L 8000:localhost:5173 kartik@192.168.8.12


# inference.py
import os
import argparse
import gc
from PIL import Image
import torch

from segmentation.segment_graph import images2points
from gnn_inference import run_gnn_inference



def process_new_manuscript(manuscript_path="./input_manuscripts/sample_manuscript_1"):
    source_images_path = os.path.join(manuscript_path, "images")
    # We will save processed (and potentially resized) images here
    # to avoid modifying source files while iterating over them.
    resized_images_path = os.path.join(manuscript_path, "images_resized")

    try:
        # Create the target folder
        os.makedirs(resized_images_path, exist_ok=True)
        
        # Verify source exists
        if not os.path.exists(source_images_path):
            print(f"Error: Source directory {source_images_path} not found.")
            return

    except Exception as e:
        print(f"An error occurred setting up directories: {e}")
        return

    # Valid image extensions to look for
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}

    # Get list of files in the directory
    files = [f for f in os.listdir(source_images_path) if os.path.isfile(os.path.join(source_images_path, f))]

    print(f"Found {len(files)} files in {source_images_path}...")

    for filename in files:
        # Skip non-image files based on extension
        ext = os.path.splitext(filename)[1].lower()
        if ext not in valid_extensions:
            continue

        base_filename = os.path.splitext(filename)[0]
        file_path = os.path.join(source_images_path, filename)

        try:
            # Open the image from the folder
            with Image.open(file_path) as image:
                
                width, height = image.size
                
                # 1. VALIDATION: Check if image is too small for CV tasks
                # If both dimensions are smaller than 600, we reject the image.
                if width < 600 and height < 600:
                    raise ValueError(f"Image resolution too low ({width}x{height}). Both dimensions are < 600px.")

                # 2. RESIZING: Downscale only if too large
                target_longest_side = 2500
                
                # Check if the longest side exceeds the target
                if max(width, height) > target_longest_side:
                    
                    # Calculate scaling factor
                    scale_factor = target_longest_side / max(width, height)
                    
                    # Calculate new dimensions
                    new_width = int(width * scale_factor)
                    new_height = int(height * scale_factor)
                    
                    # Handle Resampling filter compatibility
                    try:
                        resampling_filter = Image.Resampling.LANCZOS
                    except AttributeError:
                        resampling_filter = Image.LANCZOS

                    print(f"Downscaling '{filename}': ({width}x{height}) -> ({new_width}x{new_height})")
                    image = image.resize((new_width, new_height), resampling_filter)
                    
                else:
                    print(f"Image '{filename}' is within limits ({width}x{height}). Keeping original size.")
                    

                # Standardize Color Mode
                if image.mode in ("RGBA", "P", "LA"):
                    image = image.convert("RGB")

                # Save processed image to the NEW folder
                new_filename = f"{base_filename}.jpg"
                save_path = os.path.join(resized_images_path, new_filename)
                
                image.save(save_path, "JPEG")
                print(f"Processed: {new_filename}")

        except Exception as img_err:
            # This block catches the ValueError raised above and prints the message
            print(f"Failed to process image {filename}: {img_err}")
            continue

    # Point the inference function to the new resized/processed folder
    print("Running images2points on processed folder...")
    images2points(resized_images_path) 
    
    # Cleanup resources
    torch.cuda.empty_cache()
    gc.collect()

    print("Processing complete.")






if __name__ == "__main__":
    # 1. Parse standard CLI arguments4
    parser = argparse.ArgumentParser(description="GNN Layout Analysis Inference")
    parser.add_argument("--manuscript_path", type=str, default="./input_manuscripts/sample_manuscript_1", help="Path to the manuscript directory")
    args = parser.parse_args()

    # the data preparation.yaml is tied to the model_checkpoint used.
    args.model_checkpoint = "./pretrained_gnn/v2.pt"
    args.dataset_config_path = "./pretrained_gnn/gnn_preprocessing_v2.yaml"

    # -- Hyperparameters
    args.visualize = True
    args.BINARIZE_THRESHOLD = 0.5098
    args.BBOX_PAD_V = 0.7
    args.BBOX_PAD_H = 0.5
    args.CC_SIZE_THRESHOLD_RATIO = 0.4

    process_new_manuscript(args.manuscript_path)
    run_gnn_inference(args)



