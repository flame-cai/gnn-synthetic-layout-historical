import os
import glob
import xml.etree.ElementTree as ET
import unicodedata
import sys

# Try to import Shapely for robust Polygon math
try:
    from shapely.geometry import Polygon
except ImportError:
    print("\n[CRITICAL ERROR] The 'shapely' library is required for Polygon IoU.")
    print("Please install it using: pip install shapely\n")
    sys.exit(1)

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

def parse_polygon_string(points_str):
    """
    Parses "x1,y1 x2,y2 ..." into a list of tuples [(x1,y1), (x2,y2)...]
    and creates a Shapely Polygon object.
    """
    try:
        points = points_str.strip().split()
        if len(points) < 3:
            return None # Not a valid polygon
        
        coords = []
        for p in points:
            x, y = map(int, p.split(','))
            coords.append((x, y))
            
        poly = Polygon(coords)
        
        # Fix invalid geometry (e.g., self-intersection)
        if not poly.is_valid:
            poly = poly.buffer(0)
            
        return poly
    except Exception:
        return None

def calculate_iou_polygon(poly1, poly2):
    """
    Calculates Intersection over Union (IoU) of two Shapely Polygons.
    """
    if poly1 is None or poly2 is None:
        return 0.0
    
    try:
        # Calculate intersection area
        inter_area = poly1.intersection(poly2).area
        
        if inter_area == 0:
            return 0.0
            
        # Calculate union area
        union_area = poly1.union(poly2).area
        
        if union_area == 0:
            return 0.0
            
        return inter_area / union_area
    except Exception:
        # Fallback for topological errors
        return 0.0

# ==========================================
# 2. FILE PARSER & FILTERS
# ==========================================

def parse_pagexml(filepath):
    extracted_lines = []
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()

        # Namespace-agnostic tags
        def strip_ns(tag):
            return tag.split('}', 1)[-1]

        for textline in root.iter():
            if strip_ns(textline.tag) != "TextLine":
                continue

            # Find Coords and Unicode inside this <TextLine>
            coords = None
            text = None
            for child in textline:
                tag = strip_ns(child.tag)
                if tag == "Coords":
                    coords = child.get("points")
                elif tag == "TextEquiv":
                    for sub in child:
                        if strip_ns(sub.tag) == "Unicode":
                            text = sub.text

            if text and coords:
                clean_text = unicodedata.normalize("NFC", text).strip()
                poly_obj = parse_polygon_string(coords)

                if clean_text and poly_obj:
                    centroid = poly_obj.centroid
                    min_x, min_y, max_x, max_y = poly_obj.bounds

                    extracted_lines.append({
                        'text': clean_text,
                        'poly': poly_obj,
                        'y_center': centroid.y,
                        'x_min': min_x,
                        'width': max_x - min_x,
                    })

    except Exception as e:
        print(f"[Error] Failed to parse XML {filepath}: {e}")

    return extracted_lines




def filter_simple_layout_lines(line_objs):
    """
    Filters lines for 'simple' layouts. 
    Assumption: Main text lines are the longest. Marginalia/page numbers are short.
    Method: 1D K-Means clustering (k=2) on line widths.
    """
    if not line_objs:
        return []
        
    # Extract widths
    widths = [obj['width'] for obj in line_objs]
    
    # If there are very few lines, clustering is unstable/unnecessary. Keep all.
    if len(widths) < 3:
        return line_objs

    # --- Simple 2-Means Clustering ---
    # 1. Initialize Centroids (Min and Max)
    c_short = min(widths)
    c_long = max(widths)
    
    # 2. Iterate (5 iterations is usually sufficient for 1D)
    for _ in range(5):
        cluster_short = []
        cluster_long = []
        
        for w in widths:
            if abs(w - c_short) < abs(w - c_long):
                cluster_short.append(w)
            else:
                cluster_long.append(w)
        
        # Update centroids
        if cluster_short: c_short = sum(cluster_short) / len(cluster_short)
        if cluster_long: c_long = sum(cluster_long) / len(cluster_long)
        
    # 3. Determine threshold logic
    # If the centroids are very close, it means there isn't a clear distinction 
    # (e.g., the page is ALL main text). In this case, keep everything.
    # We define "close" as within 20% of the max width.
    if abs(c_long - c_short) < (max(widths) * 0.2):
        return line_objs

    # 4. Filter: Keep lines belonging to the "Long" cluster
    filtered_lines = []
    for obj in line_objs:
        w = obj['width']
        dist_to_long = abs(w - c_long)
        dist_to_short = abs(w - c_short)
        
        # If closer to long centroid, keep it
        if dist_to_long <= dist_to_short:
            filtered_lines.append(obj)
            
    return filtered_lines

# ==========================================
# 3. METRIC CALCULATIONS
# ==========================================

def calculate_page_level_stats(gt_objs, pred_objs):
    """
    Sorts all lines Top-Left to Bottom-Right, concatenates, and computes distance.
    Using Centroid Y for robust sorting.
    """
    # Sort logic: Primary key Y (top to bottom), Secondary key X (left to right)
    gt_sorted = sorted(gt_objs, key=lambda k: (k['y_center'], k['x_min']))
    pred_sorted = sorted(pred_objs, key=lambda k: (k['y_center'], k['x_min']))
    
    gt_blob = "\n".join([x['text'] for x in gt_sorted])
    pred_blob = "\n".join([x['text'] for x in pred_sorted])
    
    dist = levenshtein_distance(pred_blob, gt_blob)
    return dist, len(gt_blob)

def pair_lines_by_polygon_iou(gt_objs, pred_objs):
    """
    Matches predictions to GT based on highest Polygon IoU (Greedy 1-to-1).
    """
    potential_matches = []
    
    # 1. Calculate all IoUs
    for g_idx, gt in enumerate(gt_objs):
        for p_idx, pred in enumerate(pred_objs):
            # POLYGON IOU CALCULATION
            iou = calculate_iou_polygon(gt['poly'], pred['poly'])
            
            if iou > 0:
                potential_matches.append({'g_idx': g_idx, 'p_idx': p_idx, 'iou': iou})
    
    # 2. Sort by IoU descending (Greedy approach)
    potential_matches.sort(key=lambda x: x['iou'], reverse=True)
    
    final_matches = []
    matched_gt = set()
    matched_pred = set()
    
    # 3. Assign matches
    for m in potential_matches:
        if m['g_idx'] not in matched_gt and m['p_idx'] not in matched_pred:
            final_matches.append(m)
            matched_gt.add(m['g_idx'])
            matched_pred.add(m['p_idx'])
            
    unmatched_gt = set(range(len(gt_objs))) - matched_gt
    unmatched_pred = set(range(len(pred_objs))) - matched_pred
    
    return final_matches, unmatched_gt, unmatched_pred

def calculate_line_level_cost(matches, unmatched_gt, unmatched_pred, gt_objs, pred_objs, iou_threshold):
    """
    Calculates total edit distance given specific IoU threshold.
    """
    total_dist = 0
    
    # 1. Process Matches
    for m in matches:
        gt_text = gt_objs[m['g_idx']]['text']
        pred_text = pred_objs[m['p_idx']]['text']
        
        if m['iou'] >= iou_threshold:
            # Valid geometric match: Calculate character distance
            total_dist += levenshtein_distance(pred_text, gt_text)
        else:
            # IoU too low: Treat as Missed GT + False Positive Pred
            total_dist += len(gt_text) # Deletion cost
            total_dist += len(pred_text) # Insertion cost

    # 2. Process Completely Unmatched GT (Deletions)
    for idx in unmatched_gt:
        total_dist += len(gt_objs[idx]['text'])
        
    # 3. Process Completely Unmatched Preds (Insertions)
    for idx in unmatched_pred:
        total_dist += len(pred_objs[idx]['text'])
        
    return total_dist

# ==========================================
# 4. MAIN EXECUTION
# ==========================================

def safe_div(n, d):
    return n / d if d > 0 else 0.0

def evaluate_dataset(pred_folder, gt_folder, method_name, layout_type="complex"):
    print(f"\n==================================================")
    print(f"EVALUATING: {method_name}")
    print(f"LAYOUT TYPE: {layout_type}")
    print(f"==================================================")
    
    gt_files = sorted(glob.glob(os.path.join(gt_folder, "*.xml")))
    if not gt_files:
        print("No GT files found!")
        return

    # Accumulators for Global Averages
    sum_page_dist = 0
    sum_page_gt_len = 0
    sum_line_dist_50 = 0
    sum_line_dist_75 = 0
    sum_line_dist_range = 0 
    total_gt_len_all_files = 0
    
    file_count = 0
    
    # NEW: List to store per-page metrics
    per_page_results = []
    
    for gt_path in gt_files:
        filename_base = os.path.splitext(os.path.basename(gt_path))[0]
        
        # Look for XML prediction
        pred_path = os.path.join(pred_folder, filename_base + ".xml")
        
        # Parse Data
        gt_objs = parse_pagexml(gt_path)
        
        if not os.path.exists(pred_path):
            # print(f"  [Miss] Pred file not found: {filename_base}") # Optional: silence this if verbose
            pred_objs = []
        else:
            pred_objs = parse_pagexml(pred_path)
            
        # --- Filter for Simple Layouts ---
        if layout_type == "simple":
            gt_objs = filter_simple_layout_lines(gt_objs)
            pred_objs = filter_simple_layout_lines(pred_objs)
            
        # Global GT Length for this file
        current_gt_len = sum(len(x['text']) for x in gt_objs)
        total_gt_len_all_files += current_gt_len
        
        # --- 1. Page Level CER ---
        p_dist, p_len = calculate_page_level_stats(gt_objs, pred_objs)
        sum_page_dist += p_dist
        sum_page_gt_len += p_len 

        # --- 2. Line Level Matching ---
        matches, unmatched_gt, unmatched_pred = pair_lines_by_polygon_iou(gt_objs, pred_objs)
        
        # Calculate costs for this specific file
        cost_50 = calculate_line_level_cost(matches, unmatched_gt, unmatched_pred, gt_objs, pred_objs, 0.5)
        cost_75 = calculate_line_level_cost(matches, unmatched_gt, unmatched_pred, gt_objs, pred_objs, 0.75)
        
        # Add to global accumulators
        sum_line_dist_50 += cost_50
        sum_line_dist_75 += cost_75
        
        # Range calculation
        thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        file_range_dist = 0
        for t in thresholds:
            file_range_dist += calculate_line_level_cost(matches, unmatched_gt, unmatched_pred, gt_objs, pred_objs, t)
        sum_line_dist_range += file_range_dist
        
        # --- NEW: Store Per-Page Metrics ---
        # We use safe_div to handle cases where a page might be empty (length 0)
        file_page_cer = safe_div(p_dist, p_len)
        file_line_cer_50 = safe_div(cost_50, current_gt_len)
        
        per_page_results.append({
            'filename': filename_base,
            'page_cer': file_page_cer,
            'line_cer_50': file_line_cer_50,
            'gt_len': current_gt_len
        })

        file_count += 1

    # --- Final Aggregate Calculation ---
    cer_page = safe_div(sum_page_dist, sum_page_gt_len)
    cer_line_50 = safe_div(sum_line_dist_50, total_gt_len_all_files)
    cer_line_75 = safe_div(sum_line_dist_75, total_gt_len_all_files)
    cer_line_range = safe_div(sum_line_dist_range, total_gt_len_all_files) / 10.0

    print(f"Files Processed: {file_count}")
    print(f"-" * 60)
    print(f"AGGREGATE METRICS:")
    print(f"1. PAGE-LEVEL CER:        {cer_page:.4f}")
    print(f"2. LINE-LEVEL CER (IoU 0.50): {cer_line_50:.4f}")
    print(f"3. LINE-LEVEL CER (IoU 0.75): {cer_line_75:.4f}")
    print(f"4. LINE-LEVEL CER (Range):    {cer_line_range:.4f}")
    
    # --- NEW: Print Per-Page Breakdown ---
    print(f"-" * 60)
    print(f"PER-PAGE BREAKDOWN:")
    print(f"{'Filename':<35} | {'Page CER':<10} | {'Line CER (0.5)':<15} | {'GT Len':<6}")
    print(f"-" * 75)
    
    for res in per_page_results:
        print(f"{res['filename']:<35} | {res['page_cer']:.4f}     | {res['line_cer_50']:.4f}          | {res['gt_len']:<6}")
        
    print(f"==================================================\n")


# ==========================================
# 5. CONFIGURATION
# ==========================================

# NOTE: Change this to "simple" or "complex" as needed
document_layout_type="simple" 

# Define directories
DIR_XML_PRED_NO_STRUCTURE = f"{document_layout_type}/gemini"
DIR_XML_PRED_FULL = f"{document_layout_type}/gnn_gemini_fullimage"
# DIR_XML_PRED_SUB = f"{document_layout_type}/gnn_gemini_subimages"
DIR_XML_PRED_DOCUFCN = f"{document_layout_type}/docufcn_gemini_v3"
DIR_XML_PRED_SEAMFORMER = f"{document_layout_type}/seamformer_gemini_v3"
DIR_XML_PRED_GEMINI_PERFECTLAYOUT = f"{document_layout_type}/gnn_gemini_perfectlayout"


DIR_XML_PRED_EASY = f"{document_layout_type}/gnn_easyocr"
DIR_XML_PRED_EASY_DOC = f"{document_layout_type}/docufcn_easyocr_v2"
DIR_XML_PRED_EASY_SEAM = f"{document_layout_type}/seamformer_easyocr_v2"
DIR_XML_PRED_EASY_PERFECTLAYOUT = f"{document_layout_type}/gnn_easyocr_perfectlayout"


PRE_DIR_XML_PRED_EASY = f"{document_layout_type}/pretrained_gnn_easyocr"
PRE_DIR_XML_PRED_EASY_DOC = f"{document_layout_type}/pretrained_docufcn_easyocr_v2"
PRE_DIR_XML_PRED_EASY_SEAM = f"{document_layout_type}/pretrained_seamformer_easyocr_v2"
PRE_DIR_XML_PRED_EASY_PERFECTLAYOUT = f"{document_layout_type}/pretrained_gnn_easyocr_perfectlayout"

if document_layout_type=="simple":
    DIR_GT = f"{document_layout_type}/ground_truth"
else:
    DIR_GT = f"{document_layout_type}/ground_truth_modified"


if os.path.exists(DIR_GT):
    # evaluate_dataset(DIR_XML_PRED_NO_STRUCTURE, DIR_GT, "GEMINI (FULL IMAGE)", layout_type=document_layout_type)


    # evaluate_dataset(DIR_XML_PRED_SEAMFORMER, DIR_GT, "SEAMFORMER+GEMINI (FULL IMAGE)", layout_type=document_layout_type)
    # evaluate_dataset(DIR_XML_PRED_DOCUFCN, DIR_GT, "DOCUFCN+GEMINI (FULL IMAGE)", layout_type=document_layout_type)
    # evaluate_dataset(DIR_XML_PRED_FULL, DIR_GT, "GNN+GEMINI (FULL IMAGE)", layout_type=document_layout_type)
    # evaluate_dataset(DIR_XML_PRED_GEMINI_PERFECTLAYOUT, DIR_GT, "GNN+GEMINI (FULL IMAGE) PERFECT LAYOUT", layout_type=document_layout_type)

    # evaluate_dataset(DIR_XML_PRED_EASY_DOC, DIR_GT, "DOCUFCN+EASYOCR", layout_type=document_layout_type)
    # evaluate_dataset(DIR_XML_PRED_EASY_SEAM, DIR_GT, "SEAMFORMER+EASYOCR", layout_type=document_layout_type)
    # evaluate_dataset(DIR_XML_PRED_EASY, DIR_GT, "GNN+EASYOCR", layout_type=document_layout_type)
    # evaluate_dataset(DIR_XML_PRED_EASY_PERFECTLAYOUT, DIR_GT, "GNN+EASYOCR PERFECT LAYOUT", layout_type=document_layout_type)

    evaluate_dataset(PRE_DIR_XML_PRED_EASY_DOC, DIR_GT, "DOCUFCN+EASYOCR", layout_type=document_layout_type)
    evaluate_dataset(PRE_DIR_XML_PRED_EASY_SEAM, DIR_GT, "SEAMFORMER+EASYOCR", layout_type=document_layout_type)
    evaluate_dataset(PRE_DIR_XML_PRED_EASY, DIR_GT, "GNN+EASYOCR", layout_type=document_layout_type)
    evaluate_dataset(PRE_DIR_XML_PRED_EASY_PERFECTLAYOUT, DIR_GT, "GNN+EASYOCR PERFECT LAYOUT", layout_type=document_layout_type)
    
else:
    print(f"Please ensure the directory '{DIR_GT}' exists and contains the dataset.")