from __future__ import annotations

import argparse
import logging
import os
import sys

import cv2
import numpy as np
import skimage.io as io
import torch
import torch.utils.data
from lxml import etree as ET
from PIL import Image

try:
    from .dataset import AlignCollate
    from .ocr_defaults import build_label_converter, build_ocr_config, create_model, load_state_dict_compat
except ImportError:  # pragma: no cover - script execution fallback
    from dataset import AlignCollate
    from ocr_defaults import build_label_converter, build_ocr_config, create_model, load_state_dict_compat


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("manuscript_recognition.log")],
)
logger = logging.getLogger(__name__)


class InMemoryDataset(torch.utils.data.Dataset):
    def __init__(self, image_list, opt):
        self.image_list = image_list
        self.opt = opt

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image, metadata = self.image_list[index]
        image = image.convert("RGB" if self.opt.rgb else "L")
        return image, metadata


def get_model_config(saved_model_path):
    return build_ocr_config(saved_model_path=saved_model_path)


def load_ocr_model(config, device):
    logger.info(f"Loading model from {config.saved_model}...")
    converter = build_label_converter(config)
    config.num_class = len(converter.character)
    model = create_model(config, device=device, data_parallel=False)

    if not os.path.exists(config.saved_model):
        raise FileNotFoundError(f"Model file not found at: {config.saved_model}")

    load_state_dict_compat(model, config.saved_model, map_location=device, strict=True)
    model.eval()
    logger.info("Model loaded successfully.")
    return model, converter


def parse_coords(coords_str):
    try:
        points = []
        for pair in coords_str.strip().split(" "):
            x_val, y_val = map(int, pair.split(","))
            points.append([x_val, y_val])
        return np.array(points, dtype=np.int32)
    except Exception as exc:
        logger.error(f"Error parsing coordinates '{coords_str}': {exc}")
        return None


def process_page_xml(xml_path, image_root_dirs, model, converter, config, device):
    try:
        parser = ET.XMLParser(remove_blank_text=True)
        tree = ET.parse(xml_path, parser)
        root = tree.getroot()

        ns_url = root.tag.split("}")[0].strip("{") if "}" in root.tag else ""
        ns = {"pc": ns_url} if ns_url else {}

        def find_all(element, tag):
            if ns_url:
                return element.findall(f".//pc:{tag}", ns)
            return element.findall(f".//{tag}")

        def find(element, tag):
            if ns_url:
                return element.find(f"pc:{tag}", ns)
            return element.find(tag)

        page_elem = find(root, "Page")
        if page_elem is None:
            logger.error(f"No Page element found in {xml_path}")
            return

        image_filename = page_elem.get("imageFilename")
        full_image_path = None
        for img_dir in image_root_dirs:
            potential_path = os.path.join(img_dir, image_filename)
            if os.path.exists(potential_path):
                full_image_path = potential_path
                break

        if not full_image_path:
            logger.error(f"Could not find image file '{image_filename}' in provided directories for XML: {xml_path}")
            return

        try:
            image = io.imread(full_image_path)
        except Exception as exc:
            logger.error(f"Failed to read image: {full_image_path} | Error: {exc}")
            return

        if image.shape[0] == 2:
            image = image[0]
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        if image.shape[2] == 4:
            image = image[:, :, :3]
        image = np.array(image)
        processing_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        batch_data = []
        page_median_color = int(np.median(processing_image))

        text_regions = find_all(page_elem, "TextRegion")
        for region in text_regions:
            region_id = region.get("id")
            text_lines = find_all(region, "TextLine")
            for line in text_lines:
                line_id = line.get("id")
                coords_elem = find(line, "Coords")
                if coords_elem is None:
                    continue

                polygon = parse_coords(coords_elem.get("points"))
                if polygon is None or len(polygon) < 3:
                    continue

                try:
                    x_val, y_val, width, height = cv2.boundingRect(polygon)
                    x_start = max(0, x_val)
                    y_start = max(0, y_val)
                    x_end = min(processing_image.shape[1], x_val + width)
                    y_end = min(processing_image.shape[0], y_val + height)
                    if x_end <= x_start or y_end <= y_start:
                        continue

                    cropped_line_image = processing_image[y_start:y_end, x_start:x_end]
                    new_img = np.ones(cropped_line_image.shape, dtype=np.uint8) * page_median_color
                    mask_polygon = np.zeros(cropped_line_image.shape[:2], dtype=np.uint8)
                    polygon_shifted = polygon - [x_start, y_start]
                    cv2.drawContours(mask_polygon, [polygon_shifted], -1, 255, -1)
                    new_img[mask_polygon == 255] = cropped_line_image[mask_polygon == 255]

                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
                    is_success, buffer = cv2.imencode(".jpg", new_img, encode_param)
                    if is_success:
                        new_img = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)

                    batch_data.append((Image.fromarray(new_img), (line, region_id, line_id)))
                except Exception as exc:
                    logger.warning(f"Error processing crop for line {line_id} in {xml_path}: {exc}")
                    continue

        if not batch_data:
            logger.info(f"No valid text lines found in {xml_path}")
            return

        dataset = InMemoryDataset([(item[0], item[1]) for item in batch_data], config)
        align_collate = AlignCollate(imgH=config.imgH, imgW=config.imgW, keep_ratio_with_pad=config.PAD)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.workers,
            collate_fn=align_collate,
            pin_memory=True,
        )

        updated_count = 0
        with torch.no_grad():
            for image_tensors, metadata_list in data_loader:
                batch_size = image_tensors.size(0)
                image = image_tensors.to(device)
                text_for_pred = torch.LongTensor(batch_size, config.batch_max_length + 1).fill_(0).to(device)
                preds = model(image, text_for_pred, is_train=False)

                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, preds_size)

                for index, pred_text in enumerate(preds_str):
                    line_elem, _, _ = metadata_list[index]
                    for existing_equiv in find_all(line_elem, "TextEquiv"):
                        line_elem.remove(existing_equiv)

                    qname_equiv = f"{{{ns_url}}}TextEquiv" if ns_url else "TextEquiv"
                    qname_unicode = f"{{{ns_url}}}Unicode" if ns_url else "Unicode"
                    text_equiv = ET.SubElement(line_elem, qname_equiv)
                    unicode_elem = ET.SubElement(text_equiv, qname_unicode)
                    unicode_elem.text = pred_text
                    updated_count += 1

        tree.write(xml_path, pretty_print=True, encoding="UTF-8", xml_declaration=True)
        logger.info(f"Updated {xml_path}: {updated_count} lines recognized.")
    except Exception as exc:
        logger.error(f"Failed to process file {xml_path}: {exc}", exc_info=True)


def recognize_manuscript_text(xml_folder, image_folder, model_path="pretrained_model/vadakautuhala.pth"):
    if not os.path.exists(xml_folder):
        logger.error(f"XML directory not found: {xml_folder}")
        return
    if not os.path.exists(image_folder):
        logger.error(f"Image directory not found: {image_folder}")
        return
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    config = get_model_config(model_path)
    try:
        model, converter = load_ocr_model(config, device)
    except Exception as exc:
        logger.error(f"Critical error loading model: {exc}")
        return

    xml_files = sorted(file_name for file_name in os.listdir(xml_folder) if file_name.endswith(".xml"))
    if not xml_files:
        logger.warning(f"No XML files found in {xml_folder}")
        return

    logger.info(f"Found {len(xml_files)} XML files. Starting processing...")
    logger.info(f"Looking for images in: {image_folder}")
    image_search_dirs = [image_folder]

    for xml_file in xml_files:
        process_page_xml(os.path.join(xml_folder, xml_file), image_search_dirs, model, converter, config, device)

    logger.info("Process completed.")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run OCR on PAGE-XML files using full-page images.")
    parser.add_argument("--xml_folder", type=str, required=True, help="Path to the folder containing PAGE-XML files.")
    parser.add_argument("--image_folder", type=str, required=True, help="Path to the folder containing original full-page images.")
    parser.add_argument(
        "--model",
        type=str,
        default="pretrained_model/vadakautuhala.pth",
        help="Path to the trained .pth model file.",
    )
    args = parser.parse_args(argv)
    recognize_manuscript_text(args.xml_folder, args.image_folder, args.model)


if __name__ == "__main__":
    main()
