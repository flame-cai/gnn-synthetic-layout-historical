import os
import sys
import json
import glob
import time
import typing
from pathlib import Path
import google.generativeai as genai
from PIL import Image
import logging
from dotenv import load_dotenv
load_dotenv() 

# ---------------- CONFIGURATION ---------------- #

# REPLACE WITH YOUR ACTUAL API KEY
API_KEY = os.getenv("GEMINI_API_KEY")

# Model Selection: 'gemini-1.5-flash' is faster/cheaper, 'gemini-1.5-pro' is better for complex handwriting.
MODEL_NAME = "gemini-2.5-flash" 
# Retry settings
MAX_RETRIES = 3
INITIAL_BACKOFF = 2  # seconds

# ---------------- PROMPT DEFINITION ---------------- #

SYSTEM_PROMPT = """
You are an expert Indologist and Paleographer specializing in handwritten Sanskrit manuscripts.
Your Task: Perform a diplomatic transcription (OCR) of the attached manuscript image.

CRITICAL INSTRUCTIONS:
1. Output Format: Output ONLY raw valid JSON. Do not use Markdown formatting (no ```json blocks).
2. Granularity: You must transcribe at the VISUAL TEXT-LINE level. Do not merge lines even if a word is broken. 
3. Layout Handling: Categorize every line into regions. Common Sanskrit manuscript regions:
   - "main_text": The central root text.
   - "top_margin" / "bottom_margin": Headers/Footers.
   - "left_margin" / "right_margin": Marginalia.
   - "interlinear": Small commentary written between main lines.
4. Script: Transcribe in Unicode Devanagari. Preserve original spelling (Sandhi). Use "[?]" for illegible characters.

JSON SCHEMA:
{
  "status": "success",
  "regions": [
    {
      "region_type": "main_text",
      "lines": [
        "Transcribed text of line 1",
        "Transcribed text of line 2"
      ]
    },
    {
      "region_type": "right_margin",
      "lines": [
        "Transcribed text of marginalia"
      ]
    }
  ]
}
"""

# ---------------- LOGGING SETUP ---------------- #

def setup_logging(output_dir: Path):
    """Sets up logging to console and a file in the output directory."""
    log_file = output_dir / "processing.log"
    
    # Reset any existing handlers
    logging.getLogger().handlers = []
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info(f"Logging initialized. Log file: {log_file}")

# ---------------- CORE LOGIC ---------------- #

def generate_failure_json(filename: str, error_msg: str) -> dict:
    """Creates a standardized JSON structure for failed items."""
    return {
        "file_name": filename,
        "status": "failed",
        "error_message": str(error_msg),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "regions": []
    }

def call_gemini_with_retry(model, prompt, image):
    """Wraps the API call with exponential backoff for transient errors."""
    delay = INITIAL_BACKOFF
    last_exception = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # Generate content
            response = model.generate_content([prompt, image])
            return response
        except (google_exceptions.InternalServerError, 
                google_exceptions.ServiceUnavailable, 
                google_exceptions.ResourceExhausted) as e:
            
            logging.warning(f"API Error (Attempt {attempt}/{MAX_RETRIES}): {e}. Retrying in {delay}s...")
            time.sleep(delay)
            delay *= 2  # Exponential backoff
            last_exception = e
        except Exception as e:
            # For non-retriable errors (like auth errors), fail immediately
            raise e

    raise last_exception

def process_image(model, img_path: Path, output_path: Path):
    """
    Processes a single image: Load -> OCR -> Validate -> Save.
    Handles all internal exceptions to ensure the main loop doesn't crash.
    """
    output_file = output_path / f"{img_path.stem}.json"
    
    if output_file.exists():
        logging.info(f"Skipping {img_path.name} (already exists).")
        return

    logging.info(f"Processing {img_path.name}...")

    try:
        # 1. Image Validation
        try:
            img = Image.open(img_path)
            # Ensure image is loaded fully
            img.load() 
        except (UnidentifiedImageError, OSError) as e:
            raise ValueError(f"Corrupt or unsupported image file: {e}")

        # 2. API Call with Retry
        response = call_gemini_with_retry(model, SYSTEM_PROMPT, img)

        # 3. Response Validation
        if not response.parts:
            # Check for safety blocks
            if response.prompt_feedback:
                raise ValueError(f"Blocked by safety filters: {response.prompt_feedback}")
            raise ValueError("API returned empty response.")

        raw_text = response.text.strip()

        # 4. JSON Parsing & Sanitization
        # Strip markdown code blocks if Gemini ignores the system instruction
        if raw_text.startswith("```json"):
            raw_text = raw_text.split("```json")[1]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]
        elif raw_text.startswith("```"):
            raw_text = raw_text.strip("`")

        try:
            json_content = json.loads(raw_text)
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON received for {img_path.name}. Raw text: {raw_text[:100]}...")
            raise ValueError(f"JSON Parsing Failed: {e}")

        # 5. Structure Assertion
        # We add metadata and ensure the 'status' is success
        json_content["file_name"] = img_path.name
        json_content["status"] = "success"
        
        # Write Success
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(json_content, f, ensure_ascii=False, indent=2)
        
        logging.info(f"SUCCESS: {output_file.name}")

    except Exception as e:
        logging.error(f"FAILURE on {img_path.name}: {str(e)}")
        
        # Create the 'Failed' JSON file
        failure_data = generate_failure_json(img_path.name, str(e))
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(failure_data, f, ensure_ascii=False, indent=2)

def main(input_folder: str):
    # 1. Assert API Key
    if "YOUR_GEMINI_API_KEY_HERE" in API_KEY or not API_KEY:
        print("CRITICAL ERROR: API Key not set. Please edit the script or set GEMINI_API_KEY env var.")
        sys.exit(1)

    # 2. Setup Paths
    input_path = Path(input_folder)
    assert input_path.exists(), f"Input folder does not exist: {input_folder}"
    
    # Create parallel output folder
    output_path = input_path.parent / "gemini-no-structure-pred"
    output_path.mkdir(exist_ok=True)

    # 3. Setup Logging
    setup_logging(output_path)
    
    # 4. Configure Gemini
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel(
        MODEL_NAME,
        generation_config={
            "response_mime_type": "application/json", 
            "temperature": 0.0 # Zero temp for maximum determinism
        }
    )

    # 5. Gather Images
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.tiff', '*.bmp']
    image_files = []
    for ext in extensions:
        image_files.extend(list(input_path.glob(ext)))
        image_files.extend(list(input_path.glob(ext.upper()))) # Handle case sensitivity

    if not image_files:
        logging.warning("No images found in the provided directory.")
        return

    logging.info(f"Found {len(image_files)} images to process.")

    # 6. Execution Loop
    for img_file in sorted(image_files):
        process_image(model, img_file, output_path)
        
        # Respect Rate Limits (Add small delay between files)
        time.sleep(1)

    logging.info("Batch processing complete.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ocr_sanskrit.py <path_to_images_folder>")
    else:
        main(sys.argv[1])