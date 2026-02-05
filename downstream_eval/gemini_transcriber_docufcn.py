import os
import sys
import json
import argparse
import collections
import concurrent.futures
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# ---------------- CONFIGURATION ---------------- #

API_KEY = os.getenv("GEMINI_API_KEY")

# --- Configuration for polyfit ---
POLY_DEGREE = 3  # Degree of polynomial to fit through polygon points (allows for curvature)
VIZ_FOLDER = "trace_visualizations"

def get_equidistant_points(pts, m):
    """
    Interpolates a list of points to return exactly 'm' equidistant points along the path.
    """
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

def ensemble_text_samples(candidates):
    """
    Simple majority vote or longest string selection for ensemble logic.
    """
    if not candidates: return "", 0.0
    counter = collections.Counter(candidates)
    most_common = counter.most_common(1)[0]
    text = most_common[0]
    score = most_common[1] / len(candidates)
    return text, [score]

def generate_trace_from_polygon(poly_pts, is_vertical, resolution=50):
    """
    Calculates a centerline trace from a bounding polygon using Polynomial Regression.
    This works for both box-like and tight-contour polygons.
    """
    if not poly_pts or len(poly_pts) < 2:
        return []

    pts_array = np.array(poly_pts)
    x = pts_array[:, 0]
    y = pts_array[:, 1]

    try:
        if is_vertical:
            # Fit X as a function of Y for vertical lines
            # Use lower degree if fewer points available
            deg = min(POLY_DEGREE, len(y) - 1)
            if deg < 1: return poly_pts
            
            p = np.poly1d(np.polyfit(y, x, deg))
            
            # Generate smooth Y range
            min_y, max_y = np.min(y), np.max(y)
            y_line = np.linspace(min_y, max_y, resolution)
            x_line = p(y_line)
            
            # Combine into list of [x, y]
            trace = np.column_stack((x_line, y_line)).astype(int).tolist()
        else:
            # Fit Y as a function of X for horizontal lines (standard)
            deg = min(POLY_DEGREE, len(x) - 1)
            if deg < 1: return poly_pts
            
            p = np.poly1d(np.polyfit(x, y, deg))
            
            # Generate smooth X range
            min_x, max_x = np.min(x), np.max(x)
            x_line = np.linspace(min_x, max_x, resolution)
            y_line = p(x_line)
            
            trace = np.column_stack((x_line, y_line)).astype(int).tolist()
            
        return trace
    except Exception as e:
        # Fallback: simple average of points if regression fails
        print(f"Warning: Trace generation failed ({e}), using raw points.")
        return poly_pts

def _run_gemini_recognition_internal(image_path, xml_path, output_xml_path, N=1, num_trace_points=4):
    page_name = xml_path.stem
    print(f"[{page_name}] Processing with N={N}, trace_points={num_trace_points}...")

    if not xml_path.exists() or not image_path.exists():
        print(f"Error: Missing files for {page_name}")
        return {}

    try:
        # Load Image
        pil_img = Image.open(image_path)
        img_w, img_h = pil_img.size
        
        # Prepare visualization image (OpenCV format)
        viz_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        ns = {'p': 'https://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}
        tree = ET.parse(xml_path)
        root = tree.getroot()

        lines_geometry = [] 
        
        # Parse PageXML
        for textline in root.findall(".//p:TextLine", ns):
            line_id = textline.get('id') # Fallback if custom struct id missing
            custom_attr = textline.get('custom', '')
            if 'structure_line_id_' in custom_attr:
                line_id = str(custom_attr.split('structure_line_id_')[1])

            # --- MODIFIED SECTION: Extract Coords instead of Baseline ---
            coords_elem = textline.find('p:Coords', ns)
            poly_pts = []
            if coords_elem is not None and coords_elem.get('points'):
                poly_pts = [list(map(int, p.split(','))) for p in coords_elem.get('points').strip().split(' ')]
                
            
            if not poly_pts:
                print("poly_pts detected")
                continue

            # Calculate orientation
            pxs, pys = [p[0] for p in poly_pts], [p[1] for p in poly_pts]
            width_px, height_px = max(pxs)-min(pxs), max(pys)-min(pys)
            is_vert = height_px > (width_px * 1.5) # Threshold for vertical
            
            # Generate faithful centerline trace from Polygon
            # We generate a high-res trace first (e.g. 50 pts) to capture curvature
            high_res_trace = generate_trace_from_polygon(poly_pts, is_vert, resolution=50)
            
            # --- VISUALIZATION BLOCK ---
            # Draw Polygon (Green)
            pts_np = np.array(poly_pts, np.int32).reshape((-1, 1, 2))
            cv2.polylines(viz_img, [pts_np], True, (0, 255, 0), 1)
            
            # Draw Generated Trace (Red)
            trace_np = np.array(high_res_trace, np.int32).reshape((-1, 1, 2))
            cv2.polylines(viz_img, [trace_np], False, (0, 0, 255), 2)
            # ---------------------------

            lines_geometry.append({
                "id": line_id, 
                "trace_raw": high_res_trace, # Use this calculated trace
                "is_vertical": is_vert
            })

        # Save Visualization
        viz_dir = Path("output_visualizations_docufcn")
        viz_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(viz_dir / f"{page_name}_trace_viz.jpg"), viz_img)

        if not lines_geometry: 
            print(f"[{page_name}] No text lines found.")
            return {}

        # Setup Gemini
        genai.configure(api_key=API_KEY)
        model = genai.GenerativeModel('gemini-2.5-flash') # Updated model name in case 2.5 isn't public yet, adjust as needed

        def normalize(x, y):
            return max(0, min(1000, int((y / img_h) * 1000))), max(0, min(1000, int((x / img_w) * 1000)))

        def sample_worker(sample_idx):
            regions_payload = []
            for line in lines_geometry:
                # Downsample the high-res trace to the specific number of points Gemini wants (e.g. 4)
                final_trace_pts = get_equidistant_points(line['trace_raw'], num_trace_points)
                
                gemini_trace = []
                for px, py in final_trace_pts:
                    ny, nx = normalize(px, py)
                    gemini_trace.extend([ny, nx])
                
                # Sorting key (y for horizontal, x for vertical)
                sort_key = final_trace_pts[0][1] if not line['is_vertical'] else final_trace_pts[0][0]
                regions_payload.append({"id": line['id'], "trace": gemini_trace, "sort_k": sort_key})

            # Sort lines top-to-bottom (or left-to-right) for logical reading order in prompt
            regions_payload.sort(key=lambda k: k['sort_k'])

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

        # Execute Parallel Calls
        all_samples_results = []
        if N > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=N) as executor:
                future_to_idx = {executor.submit(sample_worker, i): i for i in range(N)}
                all_samples_results = [f.result() for f in concurrent.futures.as_completed(future_to_idx) if f.result()]
        else:
            res = sample_worker(0)
            if res: all_samples_results.append(res)

        # Consensus Logic
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

        # Update XML
        if final_map:
            changed = False
            for textline in root.findall(".//p:TextLine", ns):
                # Try finding ID via custom attribute or direct ID
                lid = None
                custom_attr = textline.get('custom', '')
                if 'structure_line_id_' in custom_attr:
                    lid = str(custom_attr.split('structure_line_id_')[1])
                else:
                    lid = textline.get('id')

                if lid and lid in final_map:
                    te = textline.find("p:TextEquiv", ns)
                    if te is None: te = ET.SubElement(textline, "TextEquiv")
                    uni = te.find("p:Unicode", ns)
                    if uni is None: uni = ET.SubElement(te, "Unicode")
                    uni.text = final_map[lid]

                    if lid in final_confidences:
                        conf_str = ",".join(map(str, final_confidences[lid]))
                        # Append confidence to existing custom attr
                        existing_custom = te.get('custom', '')
                        if existing_custom:
                            new_custom = f"{existing_custom} confidences:{conf_str}"
                        else:
                            new_custom = f"confidences:{conf_str}"
                        te.set('custom', new_custom)
                    changed = True
            
            if changed:
                output_xml_path.parent.mkdir(parents=True, exist_ok=True)
                tree.write(output_xml_path, encoding='UTF-8', xml_declaration=True)
                print(f"[{page_name}] XML updated and saved.")
        else:
            print(f"[{page_name}] No transcriptions generated to save.")

    except Exception as e:
        traceback.print_exc()
        print(f"Internal Recognition Error for {page_name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Gemini PAGE-XML Transcriber")
    parser.add_argument("--images", required=True, help="Folder containing images")
    parser.add_argument("--xml", required=True, help="Folder containing PAGE-XML files")
    parser.add_argument("--N", type=int, default=1, help="Number of ensemble samples")
    parser.add_argument("--points", type=int, default=4, help="Number of trace points for Gemini")
    
    args = parser.parse_args()

    img_folder = Path(args.images)
    xml_folder = Path(args.xml)
    out_folder = Path("page-xml-transcripts-docufcn")
    out_folder.mkdir(exist_ok=True)

    xml_files = list(xml_folder.glob("*.xml"))
    print(f"Found {len(xml_files)} XML files.")

    for xml_file in xml_files:
        # Assuming image has same stem name; adjust extension logic if needed (.jpg/.png)
        img_name = xml_file.stem + ".jpg" 
        # Check if custom image filename is in XML metadata (optional refinement)
        # For this script we assume filename match or simple extension swap
        
        img_path = img_folder / img_name
        if not img_path.exists():
            # Try png
            img_path = img_folder / (xml_file.stem + ".png")
        
        output_xml = out_folder / xml_file.name
        
        _run_gemini_recognition_internal(
            img_path, 
            xml_file, 
            output_xml, 
            N=args.N, 
            num_trace_points=args.points
        )

if __name__ == "__main__":
    main()