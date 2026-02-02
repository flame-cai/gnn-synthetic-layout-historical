import os
import sys
import json
import time
import datetime
import typing
import logging
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
from PIL import Image, UnidentifiedImageError
from dotenv import load_dotenv

load_dotenv()

# ---------------- CONFIGURATION ---------------- #

API_KEY = os.getenv("GEMINI_API_KEY")

# NOTE: Ensure you are using a valid model ID. 
MODEL_NAME = "gemini-2.5-flash" 

# Retry settings
MAX_RETRIES = 3
INITIAL_BACKOFF = 2

# Concurrency settings
MAX_WORKERS = 5  # Adjust based on your API rate limits (RPM/TPM)

# ---------------- PROMPT DEFINITION ---------------- #

SYSTEM_PROMPT = """
You are an expert Indologist and Paleographer specializing in handwritten Sanskrit manuscripts.
Your Task: Perform a diplomatic transcription (OCR) of the manuscript image and provide bounding boxes.

CRITICAL INSTRUCTIONS:
1. Output Format: Output ONLY raw valid JSON. No Markdown.
2. Coordinates: You MUST provide bounding boxes for every text line and region.
   - Format: [ymin, xmin, ymax, xmax]
   - Scale: Normalized coordinates from 0 to 1000 (where 1000 is the full width/height).
3. Granularity: Transcribe at the VISUAL TEXT-LINE level.
4. Script: Unicode Devanagari.

JSON SCHEMA:
{
  "status": "success",
  "regions": [
    {
      "id": "region_0",
      "type": "main_text",
      "box_2d": [ymin, xmin, ymax, xmax], 
      "lines": [
        {
          "id": "line_0",
          "box_2d": [ymin, xmin, ymax, xmax],
          "text": "Transcribed text here"
        }
      ]
    }
  ]
}
"""

# ---------------- LOGGING SETUP ---------------- #

def setup_logging(output_dir: Path):
    log_file = output_dir / "processing.log"
    # Reset handlers
    logging.getLogger().handlers = []
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info(f"Logging initialized. Log file: {log_file}")

# ---------------- XML HELPERS ---------------- #

def coords_to_string(box_2d: list, width: int, height: int) -> str:
    """
    Converts normalized [ymin, xmin, ymax, xmax] (0-1000) to PAGE-XML points string.
    PAGE-XML expects 'x,y x,y ...' (polygon style). We convert the box to 4 points.
    """
    if not box_2d or len(box_2d) != 4:
        return ""
    
    ymin, xmin, ymax, xmax = box_2d
    
    # Normalize 0-1000 to Absolute Pixels
    abs_ymin = int((ymin / 1000) * height)
    abs_xmin = int((xmin / 1000) * width)
    abs_ymax = int((ymax / 1000) * height)
    abs_xmax = int((xmax / 1000) * width)
    
    # Create points: Top-Left -> Top-Right -> Bottom-Right -> Bottom-Left
    points = [
        f"{abs_xmin},{abs_ymin}",
        f"{abs_xmax},{abs_ymin}",
        f"{abs_xmax},{abs_ymax}",
        f"{abs_xmin},{abs_ymax}"
    ]
    return " ".join(points)

def json_to_page_xml(json_data: dict, filename: str, width: int, height: int) -> str:
    """Converts the Gemini JSON output to PAGE-XML format."""
    
    # Namespaces
    ns = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
    ET.register_namespace("", ns)
    
    root = ET.Element(f"{{{ns}}}PcGts")
    
    # Metadata
    metadata = ET.SubElement(root, "Metadata")
    ET.SubElement(metadata, "Creator").text = "Gemini-OCR-Wrapper"
    ET.SubElement(metadata, "Created").text = datetime.datetime.now().isoformat()
    
    # Page
    page = ET.SubElement(root, "Page")
    page.set("imageFilename", filename)
    page.set("imageWidth", str(width))
    page.set("imageHeight", str(height))
    
    # Regions and Lines
    regions = json_data.get("regions", [])
    for r_idx, region in enumerate(regions):
        text_region = ET.SubElement(page, "TextRegion")
        r_id = region.get("id", f"region_{r_idx}")
        text_region.set("id", r_id)
        text_region.set("custom", f"type:{region.get('type', 'unknown')}")
        
        # Region Coords
        r_coords = ET.SubElement(text_region, "Coords")
        r_coords.set("points", coords_to_string(region.get("box_2d"), width, height))
        
        # Lines
        for l_idx, line in enumerate(region.get("lines", [])):
            text_line = ET.SubElement(text_region, "TextLine")
            l_id = line.get("id", f"{r_id}_line_{l_idx}")
            text_line.set("id", l_id)
            
            # Line Coords
            l_coords = ET.SubElement(text_line, "Coords")
            l_coords.set("points", coords_to_string(line.get("box_2d"), width, height))
            
            # Text Equiv
            text_equiv = ET.SubElement(text_line, "TextEquiv")
            ET.SubElement(text_equiv, "Unicode").text = line.get("text", "")

    # Pretty print
    xml_str = ET.tostring(root, encoding='utf-8')
    parsed = minidom.parseString(xml_str)
    return parsed.toprettyxml(indent="  ")

# ---------------- CORE LOGIC ---------------- #

def call_gemini_with_retry(model, prompt, image):
    delay = INITIAL_BACKOFF
    last_exception = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = model.generate_content([prompt, image])
            return response
        except (google_exceptions.InternalServerError, 
                google_exceptions.ServiceUnavailable, 
                google_exceptions.ResourceExhausted) as e:
            
            logging.warning(f"API Error (Attempt {attempt}/{MAX_RETRIES}): {e}. Retrying in {delay}s...")
            time.sleep(delay)
            delay *= 2
            last_exception = e
        except Exception as e:
            raise e

    raise last_exception

def process_single_image(model, img_path: Path, output_path: Path):
    """
    Worker function for parallel processing.
    """
    # Change extension to .xml
    output_file = output_path / f"{img_path.stem}.xml"
    
    if output_file.exists():
        logging.info(f"Skipping {img_path.name} (already exists).")
        return

    logging.info(f"Processing {img_path.name}...")

    try:
        # 1. Load Image & Get Dimensions
        try:
            with Image.open(img_path) as img:
                img.load() # Force load
                width, height = img.size
                image_copy = img.copy() # Copy for API call to avoid file lock issues
        except (UnidentifiedImageError, OSError) as e:
            raise ValueError(f"Corrupt image: {e}")

        # 2. API Call
        response = call_gemini_with_retry(model, SYSTEM_PROMPT, image_copy)

        # 3. Validation & Parsing
        if not response.parts:
            if response.prompt_feedback:
                raise ValueError(f"Safety Filter Block: {response.prompt_feedback}")
            raise ValueError("Empty response from API.")

        raw_text = response.text.strip()
        
        # Strip Markdown code blocks
        if raw_text.startswith("```json"):
            raw_text = raw_text.split("```json")[1].split("```")[0]
        elif raw_text.startswith("```"):
            raw_text = raw_text.strip("`")

        try:
            json_content = json.loads(raw_text)
        except json.JSONDecodeError:
            raise ValueError("Failed to parse JSON response.")

        # 4. Convert to PAGE-XML
        xml_content = json_to_page_xml(json_content, img_path.name, width, height)

        # 5. Save XML
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(xml_content)
        
        logging.info(f"SUCCESS: {output_file.name}")

    except Exception as e:
        logging.error(f"FAILURE on {img_path.name}: {str(e)}")
        # Save an error log file instead of corrupt XML
        err_file = output_path / f"{img_path.stem}.err"
        with open(err_file, 'w', encoding='utf-8') as f:
            f.write(f"Timestamp: {datetime.datetime.now()}\nError: {str(e)}\n")

def main(input_folder: str):
    if not API_KEY:
        print("CRITICAL ERROR: API Key not set.")
        sys.exit(1)

    input_path = Path(input_folder)
    if not input_path.exists():
        print(f"Error: Input folder {input_folder} not found.")
        sys.exit(1)
    
    output_path = input_path.parent / "gemini"
    output_path.mkdir(exist_ok=True)

    setup_logging(output_path)
    
    genai.configure(api_key=API_KEY)
    
    # Using low temperature for deterministic coordinates
    model = genai.GenerativeModel(
        MODEL_NAME,
        generation_config={"response_mime_type": "application/json", "temperature": 0.0}
    )

    extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.tiff', '*.bmp']
    image_files = []
    for ext in extensions:
        image_files.extend(list(input_path.glob(ext)))
        image_files.extend(list(input_path.glob(ext.upper())))

    if not image_files:
        logging.warning("No images found.")
        return

    logging.info(f"Found {len(image_files)} images. Starting parallel processing with {MAX_WORKERS} workers.")

    # ---------------- PARALLEL EXECUTION ---------------- #
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Map futures to filenames for tracking
        future_to_file = {
            executor.submit(process_single_image, model, img_file, output_path): img_file 
            for img_file in sorted(image_files)
        }
        
        # Process as they complete
        for future in as_completed(future_to_file):
            img_file = future_to_file[future]
            try:
                future.result() # Exceptions are caught inside process_single_image, but just in case
            except Exception as exc:
                logging.error(f"Unhandled exception for {img_file.name}: {exc}")

    logging.info("Batch processing complete.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <path_to_images_folder>")
    else:
        main(sys.argv[1])