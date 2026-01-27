import os
import json
import glob
import re
import time
import logging
import argparse
from typing import List, Dict, Tuple

# Third-party imports
import google.generativeai as genai
from PIL import Image
import jiwer

# --- Configuration ---
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("no_structure_evaluation.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_gemini(api_key: str):
    """Configures the Gemini API."""
    if not api_key:
        logger.error("API Key not found. Please provide a valid Google API Key.")
        raise ValueError("API Key is missing.")
    genai.configure(api_key=api_key)

def extract_page_number_from_filename(filename: str) -> int:
    """
    Extracts page number from filenames like '3976_0002.jpg'.
    Assumes the last sequence of digits before the extension is the page number.
    """
    base = os.path.basename(filename)
    name_without_ext = os.path.splitext(base)[0]
    # Find all digit sequences
    numbers = re.findall(r'\d+', name_without_ext)
    if numbers:
        # Return the last number found (e.g., 0002 -> 2)
        return int(numbers[-1])
    return -1

def clean_text(text: str) -> str:
    """Removes all whitespace to match the ground truth format."""
    if not text:
        return ""
    # Remove markdown code blocks if Gemini accidentally includes them
    text = text.replace("```json", "").replace("```", "")
    return "".join(text.split())

def perform_ocr_with_gemini(image_path: str, model) -> str:
    """
    Sends image to Gemini for Sanskrit OCR.
    """
    try:
        img = Image.open(image_path)
        
        # Specialized prompt for strict transcription
        prompt = (
            "You are an expert paleographer analyzing a historical Sanskrit manuscript. "
            "Please transcribe the Sanskrit text from this image exactly as it appears. "
            "1. Output ONLY the Sanskrit text. "
            "2. Do not include page numbers, headers, translations, or explanations. "
            "3. Do not add markdown formatting."
        )
        
        response = model.generate_content([prompt, img])
        return response.text
    except Exception as e:
        logger.error(f"Failed to OCR image {image_path}: {e}")
        return ""

def calculate_metrics(ground_truth: str, ocr_output: str) -> float:
    """
    Calculates Character Error Rate (CER).
    """
    if not ground_truth:
        logger.warning("Ground truth is empty. Skipping metric calculation.")
        return 1.0 # 100% error if ground truth is missing but we tried to predict
        
    if not ocr_output and ground_truth:
        return 1.0 # 100% error if OCR failed
        
    # Jiwer expects strings with spaces for word error rate, but handles chars fine for CER
    # We pass the raw strings.
    error_rate = jiwer.cer(ground_truth, ocr_output)
    return error_rate

def main():
    parser = argparse.ArgumentParser(description="OCR Sanskrit images and evaluate against Ground Truth.")
    parser.add_argument('--api_key', type=str, required=True, help='Google Gemini API Key')
    parser.add_argument('--images_dir', type=str, required=True, help='Folder containing .jpg images')
    parser.add_argument('--ground_truth', type=str, required=True, help='Path to the JSON list created in step 1')
    parser.add_argument('--output', type=str, default='ocr_results.json', help='Output JSON file for OCR results')
    
    args = parser.parse_args()

    # 1. Setup
    setup_gemini(args.api_key)
    model = genai.GenerativeModel('gemini-2.5-flash') # Flash is fast and usually sufficient for clear OCR
    
    # 2. Load Ground Truth
    logger.info(f"Loading ground truth from {args.ground_truth}...")
    try:
        with open(args.ground_truth, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
            # Convert list to dict for faster lookup: {page_num: content}
            gt_map = {item['page_number']: item['content'] for item in gt_data}
    except FileNotFoundError:
        logger.error("Ground truth file not found.")
        return

    # 3. Find Images
    image_pattern = os.path.join(args.images_dir, "*.jpg")
    image_files = sorted(glob.glob(image_pattern))
    
    if not image_files:
        logger.error(f"No images found in {args.images_dir}")
        return

    logger.info(f"Found {len(image_files)} images.")

    results_list = []
    cer_scores = []
    
    # 4. Process Loop
    for img_path in image_files:
        page_num = extract_page_number_from_filename(img_path)
        
        if page_num == -1:
            logger.warning(f"Could not extract page number from {img_path}. Skipping.")
            continue
            
        if page_num not in gt_map:
            logger.warning(f"Page {page_num} found in images but NOT in ground truth. Skipping evaluation for this page.")
            continue

        logger.info(f"Processing Page {page_num}...")
        
        # A. Perform OCR
        raw_ocr_text = perform_ocr_with_gemini(img_path, model)
        
        # B. Clean Text (Remove whitespace)
        clean_ocr_text = clean_text(raw_ocr_text)
        gt_text = gt_map[page_num] # Already cleaned in previous script
        
        # C. Calculate Metric (CER)
        cer = calculate_metrics(gt_text, clean_ocr_text)
        cer_scores.append(cer)
        
        logger.info(f"Page {page_num} CER: {cer:.4f}")

        # D. Store Result
        results_list.append({
            "page_number": page_num,
            "ground_truth_length": len(gt_text),
            "ocr_length": len(clean_ocr_text),
            "cer": cer,
            "ocr_content": clean_ocr_text,
            "ground_truth_content_snippet": gt_text[:30] + "..." # Log snippet only
        })
        
        # Sleep briefly to avoid aggressive rate limiting
        time.sleep(1)

    # 5. Final Statistics
    if cer_scores:
        avg_cer = sum(cer_scores) / len(cer_scores)
        logger.info("="*30)
        logger.info(f"Total Pages Processed: {len(cer_scores)}")
        logger.info(f"Average CER: {avg_cer:.4f}")
        logger.info("="*30)
    else:
        logger.warning("No pages were processed successfully.")

    # 6. Save Results
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump({
            "summary": {
                "average_cer": avg_cer if cer_scores else 0,
                "total_pages": len(cer_scores)
            },
            "details": results_list
        }, f, indent=4, ensure_ascii=False)
    
    logger.info(f"Detailed results saved to {args.output}")

if __name__ == "__main__":
    main()






