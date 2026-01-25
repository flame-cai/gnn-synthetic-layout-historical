import os
import json
import glob
import re
import logging
import argparse
import xml.etree.ElementTree as ET
import jiwer
from typing import List, Tuple, Dict, Optional

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("structure_evaluation.log", mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def extract_page_number(filename: str) -> int:
    """Extracts page number from filenames like '3976_0002.xml'."""
    base = os.path.basename(filename)
    name_without_ext = os.path.splitext(base)[0]
    numbers = re.findall(r'\d+', name_without_ext)
    return int(numbers[-1]) if numbers else -1

def calculate_polygon_area(points_str: str) -> float:
    """
    Calculates the area of a polygon using the Shoelace formula.
    Input format: "x1,y1 x2,y2 x3,y3 ..."
    """
    try:
        # Convert string "x1,y1 x2,y2" into list of tuples [(x1,y1), (x2,y2)]
        points = []
        for pair in points_str.strip().split():
            x, y = map(int, pair.split(','))
            points.append((x, y))
        
        # Shoelace formula
        n = len(points)
        if n < 3: return 0.0
        
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += points[i][0] * points[j][1]
            area -= points[j][0] * points[i][1]
            
        return abs(area) / 2.0
    except Exception as e:
        logger.warning(f"Failed to calculate area for points snippet '{points_str[:20]}...': {e}")
        return 0.0

def parse_page_xml(file_path: str) -> str:
    """
    Parses a PAGE-XML file to extract text.
    1. Identifies the biggest TextRegion by area.
    2. Extracts all TextLines from that region.
    3. Concatenates and removes whitespace.
    """
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # Handle XML Namespaces (PAGE-XML usually has a default namespace)
        # We strip the namespace to make searching easier, or extract it.
        # Here we extract it dynamically.
        ns_match = re.match(r'\{(.*)\}PcGts', root.tag)
        ns = {'p': ns_match.group(1)} if ns_match else {}
        
        # Helper to find tags with namespace
        def find_all(element, tag):
            return element.findall(f'p:{tag}', ns) if ns else element.findall(tag)
        
        def find(element, tag):
            return element.find(f'p:{tag}', ns) if ns else element.find(tag)

        # 1. Find all TextRegions and calculate their areas
        page_elem = find(root, 'Page')
        if page_elem is None:
            logger.error(f"No <Page> element found in {file_path}")
            return ""

        text_regions = find_all(page_elem, 'TextRegion')
        if not text_regions:
            return ""

        biggest_region = None
        max_area = -1.0

        for region in text_regions:
            coords = find(region, 'Coords')
            if coords is not None:
                points_str = coords.get('points')
                area = calculate_polygon_area(points_str)
                if area > max_area:
                    max_area = area
                    biggest_region = region
        
        if biggest_region is None:
            return ""

        # 2. Extract TextLines from the biggest region
        text_lines = find_all(biggest_region, 'TextLine')
        
        full_text = []
        for line in text_lines:
            # Look for TextEquiv/Unicode
            text_equiv = find(line, 'TextEquiv')
            if text_equiv is not None:
                unicode_text = find(text_equiv, 'Unicode')
                if unicode_text is not None and unicode_text.text:
                    full_text.append(unicode_text.text)

        # 3. Concatenate and clean (remove ALL whitespace)
        raw_string = "".join(full_text)
        clean_string = "".join(raw_string.split())
        
        return clean_string

    except Exception as e:
        logger.error(f"Error parsing XML {file_path}: {e}")
        return ""

def main():
    parser = argparse.ArgumentParser(description="Evaluate PAGE-XML predictions against Ground Truth.")
    parser.add_argument('xml_dir', type=str, help='Directory containing .xml files')
    parser.add_argument('ground_truth', type=str, help='Path to the Ground Truth JSON file')
    parser.add_argument('--output', type=str, default='xml_evaluation_results.json', help='Output JSON file')
    
    args = parser.parse_args()

    # 1. Load Ground Truth
    logger.info(f"Loading Ground Truth from {args.ground_truth}...")
    if not os.path.exists(args.ground_truth):
        logger.error("Ground truth file does not exist.")
        return

    with open(args.ground_truth, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
        # Create map { page_number: content }
        gt_map = {item['page_number']: item['content'] for item in gt_data}

    # 2. Process XML Files
    xml_files = sorted(glob.glob(os.path.join(args.xml_dir, "*.xml")))
    logger.info(f"Found {len(xml_files)} XML files in {args.xml_dir}")

    results = []
    cer_scores = []

    for xml_file in xml_files:
        page_num = extract_page_number(xml_file)
        
        if page_num == -1:
            logger.warning(f"Could not extract page number from {xml_file}. Skipping.")
            continue
            
        if page_num not in gt_map:
            logger.warning(f"Page {page_num} found in XML but NOT in ground truth. Skipping.")
            continue

        # Parse XML
        pred_text = parse_page_xml(xml_file)
        gt_text = gt_map[page_num]

        # Calculate Metric (CER)
        if not gt_text:
            logger.warning(f"Page {page_num}: Ground Truth is empty.")
            continue

        # Handle empty predictions (100% error)
        if not pred_text:
            cer = 1.0
        else:
            cer = jiwer.cer(gt_text, pred_text)

        cer_scores.append(cer)
        logger.info(f"Page {page_num} | CER: {cer:.4f} | Pred Len: {len(pred_text)} | GT Len: {len(gt_text)}")

        results.append({
            "page_number": page_num,
            "filename": os.path.basename(xml_file),
            "cer": cer,
            "ground_truth_len": len(gt_text),
            "prediction_len": len(pred_text),
            "prediction_snippet": pred_text[:50] + "..." if pred_text else ""
        })

    # 3. Summary
    if cer_scores:
        avg_cer = sum(cer_scores) / len(cer_scores)
        logger.info("-" * 40)
        logger.info(f"Evaluation Complete.")
        logger.info(f"Total Pages Evaluated: {len(cer_scores)}")
        logger.info(f"Average CER: {avg_cer:.4f}")
        logger.info("-" * 40)
    else:
        avg_cer = 0.0
        logger.warning("No pages evaluated.")

    # 4. Save Output
    output_data = {
        "summary": {
            "total_pages": len(cer_scores),
            "average_cer": avg_cer
        },
        "details": results
    }

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
    
    logger.info(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()


# python evaluate_page_xml.py ./page_xml_format ground_truth.json --output structure_results.json