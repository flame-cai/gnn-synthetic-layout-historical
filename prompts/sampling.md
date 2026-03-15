Okay great. Now as an expert in Large Language Models, Visual grounding, and sampling from LLMs, I want you to focus on backend call to gemini, which asks gemini to recognize text content from the manuscript image, using the visual grounding of text line spatial location traces. The trace format is [y_start, x_start, y_mid, x_mid, y_end, x_end] on a 0-1000 scale. I want you update the below function as follows:

Right now, the function makes one call to gemini, to recognize the text contents. What we want to do is make N number of calls to gemini. 
So the function definition will change from 

def _run_gemini_recognition_internal(manuscript, page, api_key):

to 

def _run_gemini_recognition_internal(manuscript, page, api_key, N=1):

This means we want to "sample" Gemini, to get different recognized text for each textline for each page. The main intuition is that if we sample many times, we can get a confidence score of which part of the recognized string gemini is confident about, and which part it is not confident about. We want to record and use these confidence scores.
One more intution is that the "trace" which we pass to gemini has no cannonical true value. It can vary a little bit inside the bounding polygon of the text line. In other words, we can sample different traces, to get different gemini outputs, with the hope that the combined ensemble results will be better. While sampling the trace, please consider if the orientation of the text line (horizontal, vertical, slant etc..), and sample the trace accordingly. The intuition is that we want to sample all traces "between" the topline, throughline, and baseline. 

Please think carefully when implementing this sampling robustly, and writing the ensemble code. Only make slight changes to the actual prompt if required.
Also make sure the function does not break and downstream, or upsteam code.


Ensure the textline id match to the respective recognized text, and the grounding is aligned well with Gemini internal spatial grounding co-ordinate token.
Please write robust code, with good debugging and assert statements where require. Write robust code to handle API failures, and handle possible edge cases.


# ----------------------------------------------------------------
# NEW HELPER: Internal function to run Gemini logic (Refactored)
# ----------------------------------------------------------------
def _run_gemini_recognition_internal(manuscript, page, api_key):
    """
    Internal helper to run recognition on a specific page.
    Reads the existing XML, runs Gemini, updates the XML with text, and returns the text dict.
    """
    base_path = Path(UPLOAD_FOLDER) / manuscript
    xml_path = base_path / "layout_analysis_output" / "page-xml-format" / f"{page}.xml"
    img_path = base_path / "images_resized" / f"{page}.jpg"

    if not xml_path.exists() or not img_path.exists():
        print(f"Skipping recognition for {page}: XML or Image missing.")
        return {}

    try:
        pil_img = Image.open(img_path)
        img_w, img_h = pil_img.size
        
        # Standard PAGE-XML namespace
        ns = {'p': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}
        ET.register_namespace('', ns['p']) # Register to avoid ns0 prefixes
        
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        regions_to_process = []

        # Helper to normalize pixel coords to 0-1000 scale
        def normalize(x, y):
            n_y = int((y / img_h) * 1000)
            n_x = int((x / img_w) * 1000)
            return max(0, min(1000, n_y)), max(0, min(1000, n_x))

        # 1. Extract Trace data for Gemini
        for textline in root.findall(".//p:TextLine", ns):
            custom_attr = textline.get('custom', '')
            if 'structure_line_id_' not in custom_attr: continue
            
            line_id = str(custom_attr.split('structure_line_id_')[1])
            
            trace_points = []
            
            # Prefer Baseline
            baseline_elem = textline.find('p:Baseline', ns)
            if baseline_elem is not None:
                points_str = baseline_elem.get('points', '')
                if points_str:
                    try:
                        pts = [list(map(int, p.split(','))) for p in points_str.strip().split(' ')]
                        pts.sort(key=lambda k: k[0])
                        if len(pts) >= 2:
                            trace_points = [pts[0], pts[len(pts) // 2], pts[-1]]
                    except ValueError: pass

            # Fallback Coords
            if not trace_points:
                coords_elem = textline.find('p:Coords', ns)
                if coords_elem is not None:
                    points_str = coords_elem.get('points', '')
                    if points_str:
                        try:
                            pts = [list(map(int, p.split(','))) for p in points_str.strip().split(' ')]
                            if pts:
                                xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                                sorted_pts = sorted(zip(xs, ys), key=lambda k: k[0])
                                trace_points = [
                                    [sorted_pts[0][0], sorted_pts[0][1]], 
                                    [int(sum(xs)/len(xs)), int(sum(ys)/len(ys))], 
                                    [sorted_pts[-1][0], sorted_pts[-1][1]]
                                ]
                        except ValueError: continue

            if trace_points:
                gemini_trace = []
                for px, py in trace_points:
                    ny, nx = normalize(px, py)
                    gemini_trace.extend([ny, nx])
                
                regions_to_process.append({
                    "id": line_id,
                    "trace": gemini_trace,
                    "sort_y": trace_points[0][1]
                })

        if not regions_to_process:
            return {}

        regions_to_process.sort(key=lambda k: k['sort_y'])

        # 2. Build Prompt
        prompt_text = (
            "You are an expert paleographer transcribing a Sanskrit manuscript.\n"
            "I will provide a list of Region IDs and a 'Path Trace' for each text-line.\n"
            "The trace format is [y_start, x_start, y_mid, x_mid, y_end, x_end] on a 0-1000 scale.\n"
            "Precisely transcribe the handwritten Devanagari text that sits on this curve.\n"
            "Ignore text from lines above or below this specific path.\n"
            "Output a JSON array of objects with 'id' and 'text'.\n\n"
            "REGIONS:\n"
        )
        for item in regions_to_process:
            prompt_text += f"ID: {item['id']} | Trace: {item['trace']}\n"

        # 3. Call API
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-pro') # Using stable pro model
        
        response = model.generate_content(
            [pil_img, prompt_text],
            generation_config={"response_mime_type": "application/json"}
        )
        
        try:
            result_list = json.loads(response.text)
        except:
            # Fallback if model returns list wrapped in dict or markdown
            cleaned_text = response.text.replace("```json", "").replace("```", "")
            result_list = json.loads(cleaned_text)

        if isinstance(result_list, dict) and "transcriptions" in result_list:
            result_list = result_list["transcriptions"] # Handle variance

        final_map = {}
        for item in result_list:
            if 'id' in item and 'text' in item:
                final_map[str(item['id'])] = item['text']

        # 4. Update XML with Results immediately
        if final_map:
            changed = False
            for textline in root.findall(".//p:TextLine", ns):
                custom_attr = textline.get('custom', '')
                if 'structure_line_id_' in custom_attr:
                    lid = str(custom_attr.split('structure_line_id_')[1])
                    if lid in final_map and final_map[lid]:
                        # Check/Create TextEquiv/Unicode
                        te = textline.find("p:TextEquiv", ns)
                        if te is None:
                            te = ET.SubElement(textline, "TextEquiv")
                        uni = te.find("p:Unicode", ns)
                        if uni is None:
                            uni = ET.SubElement(te, "Unicode")
                        
                        uni.text = final_map[lid]
                        changed = True
            
            if changed:
                tree.write(xml_path, encoding='UTF-8', xml_declaration=True)
                print(f"Auto-recognition updated XML for {page}")

        return final_map

    except Exception as e:
        print(f"Internal Recognition Error: {e}")
        return {}