import os
import json
import glob
import xml.etree.ElementTree as ET
import unicodedata

# ==========================================
# 1. CORE METRIC UTILITIES
# ==========================================

def levenshtein_distance(s1, s2):
    """
    Calculates the Levenshtein distance between two strings.
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

def calculate_cer(pred, gt):
    """
    Calculates Character Error Rate (CER).
    Formula: Distance / Length of Ground Truth
    """
    if not gt:
        return 1.0 if pred else 0.0
    
    dist = levenshtein_distance(pred, gt)
    return dist / len(gt)

# ==========================================
# 2. FILE PARSERS
# ==========================================

def parse_pagexml(filepath):
    """
    Parses PageXML to extract Unicode text lines.
    Ignores Region/Structure, flattens all lines into a list.
    """
    text_lines = []
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()
        
        # Handle XML Namespaces usually present in PAGE format
        # e.g. {http://schema.primaresearch.org/...}TextEquiv
        ns = {'ns': root.tag.split('}')[0].strip('{')} if '}' in root.tag else {}
        
        # Find all TextLines, then their TextEquiv -> Unicode
        # Using specific namespace search if it exists, else generic
        if ns:
            lines = root.findall('.//ns:TextLine', ns)
        else:
            lines = root.findall('.//TextLine')

        for line in lines:
            # Get TextEquiv/Unicode
            if ns:
                equiv = line.find('./ns:TextEquiv/ns:Unicode', ns)
            else:
                equiv = line.find('./TextEquiv/Unicode')
                
            if equiv is not None and equiv.text:
                # Normalize text to handle diacritics consistently
                clean_text = unicodedata.normalize('NFC', equiv.text).strip()
                if clean_text:
                    text_lines.append(clean_text)
                    
    except Exception as e:
        print(f"[Error] Failed to parse XML {filepath}: {e}")
        
    return text_lines

def parse_json(filepath):
    """
    Parses custom JSON format to extract text lines.
    Navigates regions -> lines list.
    """
    text_lines = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        regions = data.get('regions', [])
        for region in regions:
            lines = region.get('lines', [])
            for line in lines:
                # JSON lines are strings directly
                if isinstance(line, str):
                    clean_text = unicodedata.normalize('NFC', line).strip()
                    if clean_text:
                        text_lines.append(clean_text)
                        
    except Exception as e:
        print(f"[Error] Failed to parse JSON {filepath}: {e}")
        
    return text_lines

# ==========================================
# 3. MATCHING LOGIC (The AP Adaptation)
# ==========================================

def match_lines_for_ap(gt_lines, pred_lines, cer_threshold=0.5):
    """
    Matches predictions to ground truth based on CER threshold.
    Returns counts of TP, FP, FN.
    
    Algorithm:
    1. Calculate CER between EVERY prediction and EVERY GT.
    2. Filter pairs where CER <= Threshold.
    3. Sort pairs by lowest CER (best matches first).
    4. Greedily assign matches (claim GTs).
    """
    
    # Store all potential valid matches: (cer, pred_index, gt_index)
    potential_matches = []
    
    for p_idx, pred in enumerate(pred_lines):
        for g_idx, gt in enumerate(gt_lines):
            cer = calculate_cer(pred, gt)
            if cer <= cer_threshold:
                potential_matches.append((cer, p_idx, g_idx))
    
    # Sort by CER ascending (Best matches first)
    # This acts as our "Confidence" sorting mechanism for the geometric match
    potential_matches.sort(key=lambda x: x[0])
    
    matched_gt_indices = set()
    matched_pred_indices = set()
    
    tp_count = 0
    
    for cer, p_idx, g_idx in potential_matches:
        # If neither this Pred nor this GT has been used yet
        if p_idx not in matched_pred_indices and g_idx not in matched_gt_indices:
            # It's a MATCH (True Positive)
            tp_count += 1
            matched_pred_indices.add(p_idx)
            matched_gt_indices.add(g_idx)
            
    # False Positives: Predictions that didn't match any GT (or CER was too high)
    fp_count = len(pred_lines) - len(matched_pred_indices)
    
    # False Negatives: GT lines that were never matched
    fn_count = len(gt_lines) - len(matched_gt_indices)
    
    return tp_count, fp_count, fn_count

# ==========================================
# 4. MAIN EXECUTION
# ==========================================

def evaluate_method(pred_folder, gt_folder, parser_func, method_name):
    print(f"\n--- Evaluating {method_name} ---")
    
    # Get list of GT files
    gt_files = sorted(glob.glob(os.path.join(gt_folder, "*.xml")))
    
    if not gt_files:
        print("No GT files found!")
        return

    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    file_count = 0
    
    for gt_path in gt_files:
        filename_base = os.path.splitext(os.path.basename(gt_path))[0]
        
        # Determine prediction file path (handle json vs xml extension)
        if "json" in method_name.lower():
            pred_ext = ".json"
        else:
            pred_ext = ".xml"
            
        pred_path = os.path.join(pred_folder, filename_base + pred_ext)
        
        if not os.path.exists(pred_path):
            print(f"[Warning] Prediction file missing: {pred_path}")
            # If missing, all GTs are FN
            gt_lines = parse_pagexml(gt_path)
            total_fn += len(gt_lines)
            continue
            
        # Parse content
        gt_lines = parse_pagexml(gt_path)
        pred_lines = parser_func(pred_path)
        
        # Run Matcher
        tp, fp, fn = match_lines_for_ap(gt_lines, pred_lines, cer_threshold=0.50)
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        file_count += 1

    # Final Metrics Calculation
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Evaluated {file_count} files.")
    print(f"Total Lines (GT): {total_tp + total_fn}")
    print(f"Total Lines (Pred): {total_tp + total_fp}")
    print("-" * 30)
    print(f"Matches (TP): {total_tp} (CER <= 0.5)")
    print(f"Misses (FP):  {total_fp}")
    print(f"Missed GT (FN): {total_fn}")
    print("-" * 30)
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

# Define directories
DIR_JSON_PRED = "json-format-pred"
DIR_XML_PRED = "page-xml-format-pred"
DIR_XML_PRED_EASY = "page-xml-format-pred-easy"

DIR_GT = "page-xml-format"

# Ensure directories exist (for testing safety)
if os.path.exists(DIR_GT):
    # 1. Evaluate Method 1 (JSON)
    evaluate_method(DIR_JSON_PRED, DIR_GT, parse_json, "NO STRUCTURE: GEMINI (JSON)")

    # 2. Evaluate Method 2 (PageXML)
    evaluate_method(DIR_XML_PRED, DIR_GT, parse_pagexml, "STRUCTURE: GEMINI (PageXML)")

    # 2. Evaluate Method 3 (PageXML)
    evaluate_method(DIR_XML_PRED_EASY, DIR_GT, parse_pagexml, "STRUCTURE: EASYOCR (PageXML)")
else:
    print(f"Please ensure the directory '{DIR_GT}' exists and contains the dataset.")