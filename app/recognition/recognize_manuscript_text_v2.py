import argparse # Add this to imports at the top
import os
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
import logging
from lxml import etree as ET
from collections import OrderedDict
from PIL import Image

# Import necessary modules from existing codebase
from dataset import AlignCollate
from model import Model
from utils import CTCLabelConverter
import skimage.io as io

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("manuscript_recognition.log")
    ]
)
logger = logging.getLogger(__name__)

# Constants
SANSKRIT_CHARACTERS="""`0123456789~!@#$%^&*()-_+=[]\\{}|;':",./<>? abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.ँंःअअंअःआइईउऊऋएऐऑओऔकखगघङचछजझञटठडढणतथदधनऩपफबभमयरऱलळवशषसह़ािीुूृॅेैॉोौ्ॐ॒क़ख़ग़ज़ड़ढ़फ़ॠ।०१२३४५६७८९॰""" 
# Configuration Object
class OCRConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class InMemoryDataset(torch.utils.data.Dataset):
    """
    Custom Dataset to handle images generated in memory (cropped from full page)
    instead of reading from disk.
    """
    def __init__(self, image_list, opt):
        self.image_list = image_list # List of (PIL_Image, metadata_tuple)
        self.opt = opt

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        img, metadata = self.image_list[index]
        
        # Convert to grayscale if model expects 1 channel
        if self.opt.rgb:
            img = img.convert('RGB')
        else:
            img = img.convert('L')
            
        return img, metadata

def get_model_config(saved_model_path):
    return OCRConfig(
        saved_model=saved_model_path,
        Transformation=None,
        FeatureExtraction="ResNet",
        SequenceModeling="BiLSTM",
        Prediction="CTC",
        batch_size=1, 
        workers=0,
        batch_max_length=250,
        imgH=50,
        imgW=2000,
        rgb=False,
        character=SANSKRIT_CHARACTERS,
        sensitive=False,
        PAD=True,
        num_fiducial=20,
        input_channel=1,
        output_channel=512,
        hidden_size=512,
        num_class=len(SANSKRIT_CHARACTERS) + 1
    )

def load_ocr_model(config, device):
    logger.info(f"Loading model from {config.saved_model}...")
    converter = CTCLabelConverter(config.character)
    config.num_class = len(converter.character)

    model = Model(config)
    model = model.to(device)
    
    if not os.path.exists(config.saved_model):
        raise FileNotFoundError(f"Model file not found at: {config.saved_model}")

    state_dict = torch.load(config.saved_model, map_location=device)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.'):
            name = name[7:]
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model.eval()
    logger.info("Model loaded successfully.")
    return model, converter

def parse_coords(coords_str):
    """Parses PAGE-XML coords string 'x1,y1 x2,y2 ...' into a numpy array."""
    try:
        points = []
        for pair in coords_str.strip().split(' '):
            x, y = map(int, pair.split(','))
            points.append([x, y])
        return np.array(points, dtype=np.int32)
    except Exception as e:
        logger.error(f"Error parsing coordinates '{coords_str}': {e}")
        return None

def process_page_xml(xml_path, image_root_dirs, model, converter, config, device):
    """
    Parses PAGE-XML, loads full page image, extracts lines, runs inference, and updates XML.
    """
    try:
        parser = ET.XMLParser(remove_blank_text=True)
        tree = ET.parse(xml_path, parser)
        root = tree.getroot()

        # Robust Namespace Handling
        # Extract namespace from the root tag (e.g., {http://schema...}PcGts)
        ns_url = root.tag.split('}')[0].strip('{') if '}' in root.tag else ''
        ns = {'pc': ns_url} if ns_url else {}
        
        # Helper for finding elements with or without namespace
        def find_all(element, tag):
            if ns_url:
                return element.findall(f".//pc:{tag}", ns)
            return element.findall(f".//{tag}")

        def find(element, tag):
            if ns_url:
                return element.find(f"pc:{tag}", ns)
            return element.find(tag)

        # 1. Locate and Load Full Page Image
        page_elem = find(root, 'Page')
        if page_elem is None:
            logger.error(f"No Page element found in {xml_path}")
            return

        image_filename = page_elem.get('imageFilename')
        
        # Search for image in possible directories
        full_image_path = None
        for img_dir in image_root_dirs:
            potential_path = os.path.join(img_dir, image_filename)
            if os.path.exists(potential_path):
                full_image_path = potential_path
                break
        
        if not full_image_path:
            logger.error(f"Could not find image file '{image_filename}' in provided directories for XML: {xml_path}")
            return

        # --- USER REQUESTED LOAD LOGIC ---
        try:
            image = io.imread(full_image_path) # Loads as RGB
        except Exception as e:
            logger.error(f"Failed to read image: {full_image_path} | Error: {e}")
            return

        if image.shape[0] == 2: image = image[0]
        if len(image.shape) == 2: image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        if image.shape[2] == 4: image = image[:,:,:3]
        image = np.array(image)
        # ---------------------------------

        # Preprocess: Convert RGB (from io.imread) to Grayscale 
        # We do this because the subsequent crop logic and model expectation (input_channel=1) requires 2D array
        processing_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # 2. Extract Lines and Crop Images
        batch_data = [] 
        
        # Calculate Median Color for the page
        page_median_color = int(np.median(processing_image))
        
        text_regions = find_all(page_elem, 'TextRegion')
        
        for region in text_regions:
            region_id = region.get('id')
            
            text_lines = find_all(region, 'TextLine')
            for line in text_lines:
                line_id = line.get('id')
                coords_elem = find(line, 'Coords')
                
                if coords_elem is None:
                    continue
                
                coords_str = coords_elem.get('points')
                polygon = parse_coords(coords_str)
                
                if polygon is None or len(polygon) < 3:
                    continue

                try:
                    # --- Step A: Pad the Polygon Vertically (4px) ---
                    PAD_Y = 12
                    if len(polygon) > 0:
                        # Find vertical center of the polygon
                        y_center = (np.min(polygon[:, 1]) + np.max(polygon[:, 1])) / 2
                        # Expand: Top points go up (-), Bottom points go down (+)
                        polygon[:, 1] = np.where(polygon[:, 1] < y_center, 
                                                 polygon[:, 1] - PAD_Y, 
                                                 polygon[:, 1] + PAD_Y)

                    # --- Step B: Generate Image Content (with Safe Cropping) ---
                    x, y, w, h = cv2.boundingRect(polygon)
                    
                    # Clamp coordinates to image boundaries (essential after padding)
                    x_start = max(0, x)
                    y_start = max(0, y)
                    x_end = min(processing_image.shape[1], x + w)
                    y_end = min(processing_image.shape[0], y + h)

                    # Validate crop dimensions
                    if x_end <= x_start or y_end <= y_start: 
                        continue
                    
                    cropped_line_image = processing_image[y_start:y_end, x_start:x_end]
                    
                    # Create canvas (median color)
                    new_img = np.ones(cropped_line_image.shape, dtype=np.uint8) * page_median_color
                    mask_polygon = np.zeros(cropped_line_image.shape[:2], dtype=np.uint8)
                    
                    # Shift polygon relative to the *actual* crop origin
                    polygon_shifted = polygon - [x_start, y_start]
                    
                    cv2.drawContours(mask_polygon, [polygon_shifted], -1, 255, -1)
                    new_img[mask_polygon == 255] = cropped_line_image[mask_polygon == 255]
                    
                    # --- Step C: Simulate JPEG Artifacts ---
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
                    is_success, buffer = cv2.imencode(".jpg", new_img, encode_param)
                    if is_success:
                        new_img = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)

                    # Convert to PIL Image for Dataset
                    pil_img = Image.fromarray(new_img)
                    batch_data.append((pil_img, (line, region_id, line_id)))

                except Exception as e:
                    logger.warning(f"Error processing crop for line {line_id} in {xml_path}: {e}")
                    continue
                # --- END USER SNIPPET ---

        if not batch_data:
            logger.info(f"No valid text lines found in {xml_path}")
            return

        # 3. Create DataLoader and Run Inference
        dataset = InMemoryDataset([ (item[0], item[1]) for item in batch_data ], config)
        align_collate = AlignCollate(imgH=config.imgH, imgW=config.imgW, keep_ratio_with_pad=config.PAD)
        
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=config.batch_size, 
            shuffle=False, num_workers=config.workers, 
            collate_fn=align_collate, pin_memory=True
        )

        updated_count = 0

        with torch.no_grad():
            for image_tensors, metadata_list in data_loader:
                batch_size = image_tensors.size(0)
                image = image_tensors.to(device)
                
                # Dummy text input for CTC
                text_for_pred = torch.LongTensor(batch_size, config.batch_max_length + 1).fill_(0).to(device)
                
                # Inference
                preds = model(image, text_for_pred, is_train=False)
                
                # Decode
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, preds_size)

                # Update XML Objects in memory
                for i, pred_text in enumerate(preds_str):
                    line_elem, r_id, l_id = metadata_list[i]
                    
                    # Check/Remove existing TextEquiv
                    existing_equivs = find_all(line_elem, 'TextEquiv')
                    for eq in existing_equivs:
                        line_elem.remove(eq)
                    
                    # Create new TextEquiv
                    # Use proper namespace for creating elements
                    qname_equiv = f"{{{ns_url}}}TextEquiv" if ns_url else "TextEquiv"
                    qname_unicode = f"{{{ns_url}}}Unicode" if ns_url else "Unicode"
                    
                    text_equiv = ET.SubElement(line_elem, qname_equiv)
                    unicode_elem = ET.SubElement(text_equiv, qname_unicode)
                    unicode_elem.text = pred_text
                    
                    updated_count += 1

        # 4. Save the Updated XML
        tree.write(xml_path, pretty_print=True, encoding='UTF-8', xml_declaration=True)
        logger.info(f"Updated {xml_path}: {updated_count} lines recognized.")

    except Exception as e:
        logger.error(f"Failed to process file {xml_path}: {e}", exc_info=True)

def recognize_manuscript_text(xml_folder, image_folder, model_path="pretrained_model/vadakautuhala.pth"):
    """
    Main driver function to recognize text and update PAGE-XMLs.
    Args:
        xml_folder: Path to directory containing .xml files.
        image_folder: Path to directory containing original full-page images.
        model_path: Path to the .pth model file.
    """
    # 1. Validate Inputs
    if not os.path.exists(xml_folder):
        logger.error(f"XML directory not found: {xml_folder}")
        return
    
    if not os.path.exists(image_folder):
        logger.error(f"Image directory not found: {image_folder}")
        return

    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return

    # 2. Setup Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    config = get_model_config(model_path)
    try:
        model, converter = load_ocr_model(config, device)
    except Exception as e:
        logger.error(f"Critical error loading model: {e}")
        return

    # 3. Iterate over XML files
    # Filter for .xml files only
    xml_files = sorted([f for f in os.listdir(xml_folder) if f.endswith('.xml')])
    
    if not xml_files:
        logger.warning(f"No XML files found in {xml_folder}")
        return

    logger.info(f"Found {len(xml_files)} XML files. Starting processing...")
    logger.info(f"Looking for images in: {image_folder}")

    # Prepare directory list for process_page_xml (it expects a list)
    image_search_dirs = [image_folder]

    for xml_file in xml_files:
        xml_path = os.path.join(xml_folder, xml_file)
        # process_page_xml handles reading the XML, finding the specific image 
        # inside image_folder based on the XML's imageFilename attribute, 
        # cropping, predicting, and updating.
        process_page_xml(xml_path, image_search_dirs, model, converter, config, device)

    logger.info("Process completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OCR on PAGE-XML files using full-page images.")
    
    parser.add_argument(
        '--xml_folder', 
        type=str, 
        required=True, 
        help="Path to the folder containing PAGE-XML files."
    )
    parser.add_argument(
        '--image_folder', 
        type=str, 
        required=True, 
        help="Path to the folder containing original full-page images."
    )
    parser.add_argument(
        '--model', 
        type=str, 
        default="pretrained_model/vadakautuhala.pth",
        help="Path to the trained .pth model file."
    )

    args = parser.parse_args()

    recognize_manuscript_text(args.xml_folder, args.image_folder, args.model)

# python recognize_manuscript_text_v2.py  --xml_folder ../../downstream_eval/complex/docufcn_easyocr --image_folder ../../downstream_eval/complex/images

# python recognize_manuscript_text_v2.py  --xml_folder ../../downstream_eval/complex/gnn_easyocr_v2 --image_folder ../../downstream_eval/complex/images