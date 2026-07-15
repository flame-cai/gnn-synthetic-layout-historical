from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import skimage.io as io
import torch
import torch.utils.data
from lxml import etree as ET
from PIL import Image

try:
    from .auto_orientation import (
        IDENTITY_TRANSFORM,
        ROTATE_180_TRANSFORM,
        select_orientation_from_predictions,
        should_auto_orient_from_predictions,
    )
    from .dataset import AlignCollate
    from .line_segmentation.ocr_crops import (
        crop_line_record_for_ocr,
        default_line_segmentation_metadata_path,
        load_line_segmentation_metadata_by_numeric_id,
        load_line_segmentation_strategy_name,
    )
    from .ocr_defaults import build_label_converter, build_ocr_config, create_model, load_state_dict_compat
    from .pagexml_line_dataset import _encode_like_app_jpg, _load_processing_image, load_pagexml_lines
except ImportError:  # pragma: no cover - script execution fallback
    from auto_orientation import (
        IDENTITY_TRANSFORM,
        ROTATE_180_TRANSFORM,
        select_orientation_from_predictions,
        should_auto_orient_from_predictions,
    )
    from dataset import AlignCollate
    from line_segmentation.ocr_crops import (
        crop_line_record_for_ocr,
        default_line_segmentation_metadata_path,
        load_line_segmentation_metadata_by_numeric_id,
        load_line_segmentation_strategy_name,
    )
    from ocr_defaults import build_label_converter, build_ocr_config, create_model, load_state_dict_compat
    from pagexml_line_dataset import _encode_like_app_jpg, _load_processing_image, load_pagexml_lines


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


def _pagexml_namespace_helpers(root):
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

    return ns_url, find_all, find


def _line_contexts_by_id_and_custom(page_elem, find_all):
    by_line_id = {}
    by_line_custom = {}
    for region in find_all(page_elem, "TextRegion"):
        region_id = region.get("id")
        for line in find_all(region, "TextLine"):
            context = (line, region_id, line.get("id"))
            line_id = line.get("id")
            line_custom = line.get("custom")
            if line_id:
                by_line_id[line_id] = context
            if line_custom:
                by_line_custom[line_custom] = context
    return by_line_id, by_line_custom


def _extract_ocr_line_crops_with_tree(
    xml_path,
    image_root_dirs,
    *,
    line_segmentation_strategy_name=None,
    line_segmentation_metadata_path=None,
    crop_config=None,
):
    parser = ET.XMLParser(remove_blank_text=True)
    tree = ET.parse(xml_path, parser)
    root = tree.getroot()
    ns_url, find_all, find = _pagexml_namespace_helpers(root)

    page_elem = find(root, "Page")
    if page_elem is None:
        logger.error(f"No Page element found in {xml_path}")
        return tree, root, ns_url, []

    image_filename = page_elem.get("imageFilename")
    if not image_filename:
        logger.error(f"No Page imageFilename found in {xml_path}")
        return tree, root, ns_url, []
    full_image_path = None
    for img_dir in image_root_dirs:
        potential_path = os.path.join(img_dir, image_filename)
        if os.path.exists(potential_path):
            full_image_path = potential_path
            break

    if not full_image_path:
        logger.error(f"Could not find image file '{image_filename}' in provided directories for XML: {xml_path}")
        return tree, root, ns_url, []

    try:
        processing_image = _load_processing_image(full_image_path)
    except Exception as exc:
        logger.error(f"Failed to read image: {full_image_path} | Error: {exc}")
        return tree, root, ns_url, []

    metadata_path = (
        Path(line_segmentation_metadata_path)
        if line_segmentation_metadata_path is not None
        else default_line_segmentation_metadata_path(xml_path)
    )
    metadata_by_numeric_id = load_line_segmentation_metadata_by_numeric_id(metadata_path)
    effective_strategy_name = line_segmentation_strategy_name or load_line_segmentation_strategy_name(metadata_path)

    try:
        _, records = load_pagexml_lines(xml_path, include_empty_text_lines=True)
    except Exception as exc:
        logger.error(f"Failed to parse PAGE XML lines from {xml_path}: {exc}")
        return tree, root, ns_url, []

    contexts_by_line_id, contexts_by_line_custom = _line_contexts_by_id_and_custom(page_elem, find_all)
    batch_data = []
    for record in records:
        context = contexts_by_line_id.get(record.line_id) or contexts_by_line_custom.get(record.line_custom)
        if context is None:
            logger.warning(f"Could not match OCR crop record to XML line {record.line_id} in {xml_path}")
            continue
        try:
            crop_result = crop_line_record_for_ocr(
                processing_image,
                record,
                strategy_name=effective_strategy_name,
                strategy_line_metadata=metadata_by_numeric_id.get(int(record.line_numeric_id)),
                crop_config=crop_config or {},
            )
            _, decoded_jpg = _encode_like_app_jpg(crop_result.image)
            line_elem, region_id, line_id = context
            batch_data.append((Image.fromarray(decoded_jpg), (line_elem, region_id, line_id, crop_result.metadata)))
        except Exception as exc:
            logger.warning(f"Error processing crop for line {record.line_id} in {xml_path}: {exc}")
            continue

    return tree, root, ns_url, batch_data


def extract_ocr_line_crops_from_page_xml(
    xml_path,
    image_root_dirs,
    *,
    line_segmentation_strategy_name=None,
    line_segmentation_metadata_path=None,
    crop_config=None,
):
    _, _, _, batch_data = _extract_ocr_line_crops_with_tree(
        xml_path,
        image_root_dirs,
        line_segmentation_strategy_name=line_segmentation_strategy_name,
        line_segmentation_metadata_path=line_segmentation_metadata_path,
        crop_config=crop_config,
    )
    return batch_data


def process_page_xml(
    xml_path,
    image_root_dirs,
    model,
    converter,
    config,
    device,
    *,
    line_segmentation_strategy_name=None,
    line_segmentation_metadata_path=None,
    crop_config=None,
):
    try:
        tree, root, ns_url, batch_data = _extract_ocr_line_crops_with_tree(
            xml_path,
            image_root_dirs,
            line_segmentation_strategy_name=line_segmentation_strategy_name,
            line_segmentation_metadata_path=line_segmentation_metadata_path,
            crop_config=crop_config,
        )
        _, find_all, _ = _pagexml_namespace_helpers(root)

        if not batch_data:
            logger.info(f"No valid text lines found in {xml_path}")
            return

        candidate_batch_data = []
        auto_orientation_group_ids = set()
        for group_id, (line_image, context) in enumerate(batch_data):
            crop_metadata = context[3]
            candidate_batch_data.append((line_image, (group_id, IDENTITY_TRANSFORM)))
            if should_auto_orient_from_predictions(crop_metadata):
                auto_orientation_group_ids.add(group_id)
                candidate_batch_data.append(
                    (line_image.rotate(180, expand=False), (group_id, ROTATE_180_TRANSFORM))
                )

        dataset = InMemoryDataset(candidate_batch_data, config)
        align_collate = AlignCollate(imgH=config.imgH, imgW=config.imgW, keep_ratio_with_pad=config.PAD)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.workers,
            collate_fn=align_collate,
            pin_memory=True,
        )

        predictions_by_group = {}
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
                    group_id, transform = metadata_list[index]
                    predictions_by_group.setdefault(int(group_id), {})[str(transform)] = pred_text

        updated_count = 0
        rotated_line_count = 0
        auto_orientation_selections = []
        for group_id, (_, context) in enumerate(batch_data):
            line_elem, _, line_id, crop_metadata = context
            candidate_predictions = predictions_by_group.get(group_id, {})
            selection = None
            if group_id in auto_orientation_group_ids:
                selection = select_orientation_from_predictions(candidate_predictions)
                pred_text = selection.selected_text
                crop_metadata["auto_orientation"] = selection.to_metadata()
                topology = crop_metadata.get("topology") or {}
                auto_orientation_selections.append(
                    {
                        "line_id": str(line_id),
                        "line_numeric_id": crop_metadata.get("line_numeric_id"),
                        "line_kind": topology.get("line_kind"),
                        **selection.to_metadata(),
                    }
                )
                if selection.selected_transform == ROTATE_180_TRANSFORM:
                    rotated_line_count += 1
                logger.info(
                    "Auto-oriented curved OCR line=%s transform=%s identity=%r rotate_180=%r",
                    line_id,
                    selection.selected_transform,
                    candidate_predictions.get(IDENTITY_TRANSFORM, ""),
                    candidate_predictions.get(ROTATE_180_TRANSFORM, ""),
                )
            else:
                pred_text = candidate_predictions.get(IDENTITY_TRANSFORM, "")

            for existing_equiv in find_all(line_elem, "TextEquiv"):
                line_elem.remove(existing_equiv)

            qname_equiv = f"{{{ns_url}}}TextEquiv" if ns_url else "TextEquiv"
            qname_unicode = f"{{{ns_url}}}Unicode" if ns_url else "Unicode"
            text_equiv = ET.SubElement(line_elem, qname_equiv)
            if selection is not None:
                text_equiv.set(
                    "custom",
                    (
                        "auto_orientation_model:decoded_text_devanagari_evidence_v1;"
                        f"auto_orientation_transform:{selection.selected_transform};"
                        f"auto_orientation_reason:{selection.reason}"
                    ),
                )
            unicode_elem = ET.SubElement(text_equiv, qname_unicode)
            unicode_elem.text = pred_text
            updated_count += 1

        tree.write(xml_path, pretty_print=True, encoding="UTF-8", xml_declaration=True)
        logger.info(
            "Updated %s: %d lines recognized; %d unannotated curved lines evaluated; %d rotated.",
            xml_path,
            updated_count,
            len(auto_orientation_group_ids),
            rotated_line_count,
        )
        return {
            "updated_line_count": updated_count,
            "auto_orientation_line_count": len(auto_orientation_group_ids),
            "auto_orientation_rotated_line_count": rotated_line_count,
            "auto_orientation_selections": auto_orientation_selections,
        }
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
