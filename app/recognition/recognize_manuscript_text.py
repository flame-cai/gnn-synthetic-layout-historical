import os
import sys
import torch
import torch.nn.functional as F
import torch.utils.data
import logging
from lxml import etree as ET
from collections import OrderedDict  # <--- ADD THIS

# lxml

# Import necessary modules from existing codebase
from dataset import RawDataset, AlignCollate
from model import Model
from utils import CTCLabelConverter

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

# Constants based on the provided instructions and recognition.py
SANSKRIT_CHARACTERS="""`0123456789~!@#$%^&*()-_+=[]\\{}|;':",./<>? abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.ँंःअअंअःआइईउऊऋएऐऑओऔकखगघङचछजझञटठडढणतथदधनऩपफबभमयरऱलळवशषसह़ािीुूृॅेैॉोौ्ॐ॒क़ख़ग़ज़ड़ढ़फ़ॠ।०१२३४५६७८९॰"""

# Configuration Object (mimics argparse)
class OCRConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def get_model_config(saved_model_path):
    """
    Returns the configuration object required to initialize the Model.
    Parameters match those found in recognition.py calls.
    """
    return OCRConfig(
        saved_model=saved_model_path,
        Transformation=None,
        FeatureExtraction="ResNet",
        SequenceModeling="BiLSTM",
        Prediction="CTC",
        batch_size=1,           # Adjusted for inference speed/memory balance
        workers=0,
        batch_max_length=250,    # As per recognition.py
        imgH=50,                 # As per recognition.py
        imgW=2000,               # As per recognition.py
        rgb=False,
        character=SANSKRIT_CHARACTERS,
        sensitive=False,         # Although SANSKRIT_CHARACTERS implies case sensitivity, typical settings use False for standardizing
        PAD=True,                # As per recognition.py
        num_fiducial=20,
        input_channel=1,
        output_channel=512,
        hidden_size=512,
        num_class=len(SANSKRIT_CHARACTERS) + 1 # +1 for CTC blank
    )

def load_ocr_model(config, device):
    """Loads the model into memory once without DataParallel."""
    logger.info(f"Loading model from {config.saved_model}...")
    
    # Calculate num_class
    converter = CTCLabelConverter(config.character)
    config.num_class = len(converter.character)

    # Initialize Model (Raw, no DataParallel)
    model = Model(config)
    model = model.to(device)
    
    if not os.path.exists(config.saved_model):
        raise FileNotFoundError(f"Model file not found at: {config.saved_model}")

    # Load State Dict and Fix Keys
    # The checkpoint has 'module.' prefixes because it was trained with DataParallel.
    # We must strip them to run on a single instance.
    state_dict = torch.load(config.saved_model, map_location=device)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.'):
            name = name[7:] # remove 'module.'
        new_state_dict[name] = v

    # Load fixed state dict
    model.load_state_dict(new_state_dict)
    
    model.eval()
    logger.info("Model loaded successfully (DataParallel prefix stripped).")
    return model, converter


def run_inference(model, converter, image_folder, config, device):
    """
    Runs inference on all images found recursively with batch_size=1.
    """
    logger.info(f"Starting inference on images in: {image_folder}")
    
    dataset = RawDataset(root=image_folder, opt=config)
    align_collate = AlignCollate(imgH=config.imgH, imgW=config.imgW, keep_ratio_with_pad=config.PAD)
    
    # workers=0 is set in config now
    data_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=config.batch_size, # This is 1
        shuffle=False,
        num_workers=config.workers,
        collate_fn=align_collate,
        pin_memory=True
    )

    results = {}
    
    with torch.no_grad():
        for i, (image_tensors, image_path_list) in enumerate(data_loader):
            batch_size = image_tensors.size(0) # Will be 1
            image = image_tensors.to(device)
            
            # Create dummy text input required by Model.forward() signature
            # Shape: [Batch_Size, Max_Len + 1]
            text_for_pred = torch.LongTensor(batch_size, config.batch_max_length + 1).fill_(0).to(device)
            
            # Predict
            # Pass text_for_pred explicitly
            preds = model(image, text_for_pred, is_train=False)
            
            # Decode CTC
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, preds_size)

            for img_path, pred_text in zip(image_path_list, preds_str):
                results[img_path] = pred_text
                
            if i % 50 == 0:
                logger.info(f"Processed {i} lines...")

    logger.info(f"Inference completed. Processed {len(results)} images.")
    return results

def parse_metadata_from_path(image_path, manuscript_name):
    """
    Parses the file path to extract Page ID, Textbox Label, and Line ID.
    
    Expected Structure:
    .../{manuscript_name}/layout_analysis_output/image-format/{page_id}/{textbox_label}/{line_filename}.jpg
    
    Example:
    .../233_0002/textbox_label_6/line_0.jpg
    -> Page: 233_0002
    -> Region Custom: textbox_label_6
    -> Line Custom: structure_line_id_0 (derived from line_0)
    """
    try:
        # Normalize path separators
        path_parts = os.path.normpath(image_path).split(os.sep)
        
        # Locate the 'image-format' directory index to find relative structure
        if 'image-format' in path_parts:
            idx = path_parts.index('image-format')
            # Check if path depth is sufficient
            if len(path_parts) > idx + 3:
                page_id = path_parts[idx + 1]
                textbox_label = path_parts[idx + 2]
                filename = path_parts[idx + 3]
                
                name_no_ext = os.path.splitext(filename)[0] # e.g., 'line_0'
                
                # Logic to convert 'line_X' to 'structure_line_id_X'
                # Assumption: The suffix number matches.
                if "line_" in name_no_ext:
                    line_num = name_no_ext.split('_')[-1]
                    line_custom_id = f"structure_line_id_{line_num}"
                else:
                    # Fallback if naming convention differs
                    line_custom_id = name_no_ext

                return {
                    'page_id': page_id,
                    'region_custom': textbox_label,
                    'line_custom': line_custom_id
                }
    except Exception as e:
        logger.error(f"Error parsing path {image_path}: {e}")
        
    return None

def update_page_xml(xml_path, text_data, ns_map):
    """
    Updates the XML file with recognized text.
    
    text_data: dict -> { (region_custom, line_custom): text_content }
    """
    if not os.path.exists(xml_path):
        logger.warning(f"XML file not found: {xml_path}")
        return

    try:
        parser = ET.XMLParser(remove_blank_text=True)
        tree = ET.parse(xml_path, parser)
        root = tree.getroot()
        
        # Define Namespace
        # The default namespace needs to be handled carefully in lxml XPath
        ns = {'pc': ns_map[None]} if None in ns_map else {'pc': list(ns_map.values())[0]}

        updated_count = 0
        
        # Iterate over TextRegions
        # Finding regions that match our data keys
        for region in root.findall(".//pc:TextRegion", ns):
            region_custom = region.get('custom')
            
            # Optimization: check if we have any data for this region
            region_has_updates = any(k[0] == region_custom for k in text_data.keys())
            if not region_has_updates:
                continue

            for line in region.findall(".//pc:TextLine", ns):
                line_custom = line.get('custom')
                
                key = (region_custom, line_custom)
                
                if key in text_data:
                    rec_text = text_data[key]
                    
                    # Remove existing TextEquiv if present to avoid duplicates
                    for existing_equiv in line.findall("pc:TextEquiv", ns):
                        line.remove(existing_equiv)
                    
                    # Create new TextEquiv
                    text_equiv = ET.SubElement(line, f"{{{ns['pc']}}}TextEquiv")
                    unicode_elem = ET.SubElement(text_equiv, f"{{{ns['pc']}}}Unicode")
                    unicode_elem.text = rec_text
                    
                    updated_count += 1

        if updated_count > 0:
            tree.write(xml_path, pretty_print=True, encoding='UTF-8', xml_declaration=True)
            logger.info(f"Updated {xml_path} - {updated_count} lines written.")
        else:
            logger.info(f"No matching lines found to update in {xml_path}")

    except Exception as e:
        logger.error(f"Failed to update XML {xml_path}: {e}")

def recognize_manuscript_text(manuscript_name):
    """
    Main driver function to recognize text and update PAGE-XMLs.
    """
    # 1. Setup Directories
    base_dir = "../input_manuscripts" # Assumed relative to current script, adjust if needed
    manuscript_dir = os.path.join(base_dir, manuscript_name, "layout_analysis_output")
    image_root = os.path.join(manuscript_dir, "image-format")
    xml_root = os.path.join(manuscript_dir, "page-xml-format")
    model_path = "pretrained_model/vadakautuhala.pth" 

    # Validate paths
    if not os.path.exists(image_root):
        logger.error(f"Image directory not found: {image_root}")
        return
    if not os.path.exists(xml_root):
        logger.error(f"XML directory not found: {xml_root}")
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

    # 3. Run Inference on ALL images
    # We pass the root 'image-format' folder. RawDataset recursively finds all images 
    # inside 233_0002/textbox_label_X/line_Y.jpg
    inference_results = run_inference(model, converter, image_root, config, device)

    if not inference_results:
        logger.warning("No text recognized or no images found.")
        return

    # 4. Organize Results by Page
    # Structure: structured_results[page_id] = { (region_custom, line_custom): text }
    structured_results = {}

    logger.info("Organizing recognized text...")
    for img_path, text in inference_results.items():
        meta = parse_metadata_from_path(img_path, manuscript_name)
        if meta:
            page_id = meta['page_id']
            if page_id not in structured_results:
                structured_results[page_id] = {}
            
            key = (meta['region_custom'], meta['line_custom'])
            structured_results[page_id][key] = text

    # 5. Update XML Files
    logger.info("Updating PAGE-XML files...")
    
    # We use a dummy parser to get the namespace map from the first available XML
    # assuming all XMLs share the same schema.
    ns_map = {None: "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}
    
    # Iterate through pages identified in images
    for page_id, text_data in structured_results.items():
        xml_filename = f"{page_id}.xml"
        xml_path = os.path.join(xml_root, xml_filename)
        
        if os.path.exists(xml_path):
            # Dynamic Namespace detection (safer)
            try:
                events = ET.iterparse(xml_path, events=('start-ns',))
                for event, elem in events:
                    if event == 'start-ns':
                        ns_map = {elem[0]: elem[1]} # Usually default ns
                        break
            except:
                pass # Fallback to default
            
            update_page_xml(xml_path, text_data, ns_map)
        else:
            logger.error(f"Corresponding XML file not found for page {page_id} at {xml_path}")

    logger.info(f"Process completed for manuscript: {manuscript_name}")

if __name__ == "__main__":
    # Example usage
    # Ensure this script is run from the 'recognition' directory or paths are adjusted accordingly
    
    # Change this to the actual manuscript folder name you want to process
    MANUSCRIPT_NAME = "test_manuscript_01" 
    
    if len(sys.argv) > 1:
        MANUSCRIPT_NAME = sys.argv[1]

    recognize_manuscript_text(MANUSCRIPT_NAME)

