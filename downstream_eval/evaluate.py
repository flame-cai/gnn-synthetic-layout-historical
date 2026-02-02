import os
import json
import glob
import xml.etree.ElementTree as ET
import unicodedata
import statistics

# ==========================================
# 1. CORE METRIC UTILITIES
# ==========================================

def levenshtein_distance(s1, s2):
    """
    Calculates the Levenshtein distance between two strings.
    Optimized for memory usage (only stores two rows).
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def calculate_cer_ratio(distance, ref_length):
    """ Safe division for CER """
    if ref_length == 0:
        return 1.0 if distance > 0 else 0.0
    return distance / ref_length

# ==========================================
# 2. FILE PARSERS (Updated for Coordinates)
# ==========================================

def get_avg_y(points_str):
    """
    Parses a point string "x1,y1 x2,y2 ..." and returns average Y.
    """
    try:
        points = points_str.strip().split()
        y_coords = [int(p.split(',')[1]) for p in points]
        return statistics.mean(y_coords) if y_coords else 0
    except:
        return 0

def parse_pagexml(filepath):
    """
    Parses PageXML to extract Unicode text lines AND their Y-coordinates.
    Returns: list of dicts [{'text': str, 'y': float}, ...]
    """
    extracted_lines = []
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()
        
        # Handle XML Namespaces
        ns = {'ns': root.tag.split('}')[0].strip('{')} if '}' in root.tag else {}
        
        if ns:
            lines = root.findall('.//ns:TextLine', ns)
        else:
            lines = root.findall('.//TextLine')

        for line in lines:
            # 1. Extract Text
            if ns:
                equiv = line.find('./ns:TextEquiv/ns:Unicode', ns)
                baseline = line.find('./ns:Baseline', ns)
                coords = line.find('./ns:Coords', ns)
            else:
                equiv = line.find('./TextEquiv/Unicode')
                baseline = line.find('./Baseline')
                coords = line.find('./Coords')
                
            if equiv is not None and equiv.text:
                clean_text = unicodedata.normalize('NFC', equiv.text).strip()
                if clean_text:
                    # 2. Extract Y-Coordinate (prefer Baseline, fallback to Coords)
                    y_pos = 0
                    if baseline is not None:
                        y_pos = get_avg_y(baseline.get('points'))
                    elif coords is not None:
                        y_pos = get_avg_y(coords.get('points'))
                    
                    extracted_lines.append({'text': clean_text, 'y': y_pos})
                    
    except Exception as e:
        print(f"[Error] Failed to parse XML {filepath}: {e}")
        
    return extracted_lines

def parse_json(filepath):
    """
    Parses JSON. Since JSON has no coordinates in this schema, 
    we assign a dummy 'y' index based on reading order.
    Returns: list of dicts [{'text': str, 'y': int_index}, ...]
    """
    extracted_lines = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        regions = data.get('regions', [])
        row_counter = 0
        
        for region in regions:
            lines = region.get('lines', [])
            for line in lines:
                if isinstance(line, str):
                    clean_text = unicodedata.normalize('NFC', line).strip()
                    if clean_text:
                        # Use counter as Y to maintain file order
                        extracted_lines.append({'text': clean_text, 'y': row_counter})
                        row_counter += 1
                        
    except Exception as e:
        print(f"[Error] Failed to parse JSON {filepath}: {e}")
        
    return extracted_lines

# ==========================================
# 3. METRIC CALCULATIONS
# ==========================================

def calculate_matched_cer(gt_objs, pred_objs, threshold=0.25):
    """
    Performs the AP-style matching and calculates stats + Post-Match CER.
    """
    gt_lines = [x['text'] for x in gt_objs]
    pred_lines = [x['text'] for x in pred_objs]
    
    # --- 1. The Matching Logic (Greedy by CER) ---
    potential_matches = []
    
    for p_idx, pred in enumerate(pred_lines):
        for g_idx, gt in enumerate(gt_lines):
            dist = levenshtein_distance(pred, gt)
            cer = calculate_cer_ratio(dist, len(gt))
            
            # Record match if it meets threshold
            if cer <= threshold:
                potential_matches.append({
                    'cer': cer,
                    'dist': dist,
                    'p_idx': p_idx,
                    'g_idx': g_idx
                })
    
    # Sort best matches first
    potential_matches.sort(key=lambda x: x['cer'])
    
    matched_gt = set()
    matched_pred = set()
    
    # Stats for F1/AP
    tp_count = 0
    
    # Accumulators for "Matched Page CER"
    matched_edit_distance = 0
    
    for match in potential_matches:
        if match['p_idx'] not in matched_pred and match['g_idx'] not in matched_gt:
            # Accept Match
            tp_count += 1
            matched_pred.add(match['p_idx'])
            matched_gt.add(match['g_idx'])
            
            # Add the edit distance of the matched pair
            matched_edit_distance += match['dist']

    fp_count = len(pred_lines) - len(matched_pred)
    fn_count = len(gt_lines) - len(matched_gt)
    
    # --- 2. Calculate "Matched Page CER" ---
    # Cost = (Edits in TPs) + (Length of FNs/Deletions) + (Length of FPs/Insertions)
    
    cost_fn = sum(len(gt_lines[i]) for i in range(len(gt_lines)) if i not in matched_gt)
    cost_fp = sum(len(pred_lines[i]) for i in range(len(pred_lines)) if i not in matched_pred)
    
    total_edit_cost = matched_edit_distance + cost_fn + cost_fp
    
    # Ground truth total length (for denominator)
    total_gt_chars = sum(len(s) for s in gt_lines)
    
    return {
        'tp': tp_count,
        'fp': fp_count,
        'fn': fn_count,
        'edit_cost': total_edit_cost,
        'gt_char_len': total_gt_chars
    }

def calculate_crude_cer(gt_objs, pred_objs):
    """
    Sorts lines by Y-coordinate, joins them into one string, 
    and calculates one massive CER.
    """
    # Sort by Y coordinate
    gt_sorted = sorted(gt_objs, key=lambda k: k['y'])
    pred_sorted = sorted(pred_objs, key=lambda k: k['y'])
    
    # Join with newline to represent page structure
    gt_blob = "\n".join([x['text'] for x in gt_sorted])
    pred_blob = "\n".join([x['text'] for x in pred_sorted])
    
    dist = levenshtein_distance(pred_blob, gt_blob)
    return dist, len(gt_blob)

# ==========================================
# 4. MAIN EXECUTION
# ==========================================

def evaluate_method(pred_folder, gt_folder, parser_func, method_name):
    print(f"\n==================================================")
    print(f"EVALUATING: {method_name}")
    print(f"==================================================")
    
    gt_files = sorted(glob.glob(os.path.join(gt_folder, "*.xml")))
    if not gt_files:
        print("No GT files found!")
        return

    # Accumulators for Object-Level Metrics (F1, Precision, Recall)
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    # Accumulators for Character-Level Metrics (CERs)
    sum_matched_cer_dist = 0
    sum_matched_gt_len = 0
    
    sum_crude_cer_dist = 0
    sum_crude_gt_len = 0
    
    file_count = 0
    
    for gt_path in gt_files:
        filename_base = os.path.splitext(os.path.basename(gt_path))[0]
        
        # Determine prediction file path
        if "json" in method_name.lower():
            pred_ext = ".json"
        else:
            pred_ext = ".xml"
            
        pred_path = os.path.join(pred_folder, filename_base + pred_ext)
        
        # Parse GT
        gt_objs = parse_pagexml(gt_path)
        
        # Handle missing predictions
        if not os.path.exists(pred_path):
            print(f"  [Miss] Pred file not found: {filename_base}")
            # All GT lines are FN
            total_fn += len(gt_objs)
            # Add full length of GT to error costs (Deletion)
            gt_len_total = sum(len(x['text']) for x in gt_objs)
            sum_matched_cer_dist += gt_len_total
            sum_matched_gt_len += gt_len_total
            # Crude CER logic: Pred is empty string, distance = gt length
            # Note: For crude, we join with \n, so length might differ slightly
            gt_blob_len = len("\n".join([x['text'] for x in gt_objs]))
            sum_crude_cer_dist += gt_blob_len
            sum_crude_gt_len += gt_blob_len
            continue
            
        # Parse Pred
        pred_objs = parser_func(pred_path)
        
        # --- Metric 1 & Base: Matching Logic + Matched CER ---
        # Using threshold 0.25 as requested in prompt description (though user code had 0.20)
        # We will use 0.25 to align with the function default and prompt text.
        res = calculate_matched_cer(gt_objs, pred_objs, threshold=0.50)
        
        total_tp += res['tp']
        total_fp += res['fp']
        total_fn += res['fn']
        
        sum_matched_cer_dist += res['edit_cost']
        sum_matched_gt_len += res['gt_char_len']
        
        # --- Metric 2: Crude CER (Top-to-Bottom) ---
        crude_dist, crude_len = calculate_crude_cer(gt_objs, pred_objs)
        sum_crude_cer_dist += crude_dist
        sum_crude_gt_len += crude_len
        
        file_count += 1

    # --- Final Calculations ---
    
    # 1. Object Detection Metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # 2. Matched CER (Micro-average)
    matched_cer_final = sum_matched_cer_dist / sum_matched_gt_len if sum_matched_gt_len > 0 else 0
    
    # 3. Crude CER (Micro-average)
    crude_cer_final = sum_crude_cer_dist / sum_crude_gt_len if sum_crude_gt_len > 0 else 0
    
    print(f"Files Processed: {file_count}")
    print(f"-" * 40)
    print(f"OBJECT LEVEL METRICS (Threshold: CER <= 0.25)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  (TP: {total_tp}, FP: {total_fp}, FN: {total_fn})")
    print(f"-" * 40)
    print(f"CHARACTER LEVEL METRICS")
    print(f"  1. Matched CER: {matched_cer_final:.4f}")
    print(f"     (Sum of edits in matches + insertions + deletions)")
    print(f"  2. Crude CER:   {crude_cer_final:.4f}")
    print(f"     (Concatenated page sorted Top-to-Bottom)")
    print(f"==================================================\n")

document_layout_type="complex"
# Define directories
DIR_JSON_PRED = f"{document_layout_type}/gemini"
DIR_XML_PRED = f"{document_layout_type}/gnn_gemini"
DIR_XML_PRED_EASY = f"{document_layout_type}/gnn_easyocr"
DIR_GT = f"{document_layout_type}/ground_truth"

if os.path.exists(DIR_GT):
    evaluate_method(DIR_JSON_PRED, DIR_GT, parse_pagexml, "GEMINI")
    evaluate_method(DIR_XML_PRED, DIR_GT, parse_pagexml, "GNN+GEMINI")
    evaluate_method(DIR_XML_PRED_EASY, DIR_GT, parse_pagexml, "GNN+EASYOCR")
else:
    print(f"Please ensure the directory '{DIR_GT}' exists and contains the dataset.")