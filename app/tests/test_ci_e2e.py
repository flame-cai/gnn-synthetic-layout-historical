import os
import shutil
import sys
import unittest
import xml.etree.ElementTree as ET
from contextlib import ExitStack
from pathlib import Path


TESTS_ROOT = Path(__file__).resolve().parent
APP_ROOT = TESTS_ROOT.parent
REPO_ROOT = APP_ROOT.parent
LOGS_ROOT = TESTS_ROOT / "logs"

if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ORIGINAL_CWD = Path.cwd()
os.chdir(TESTS_ROOT)
import app as backend_app_module
os.chdir(APP_ROOT)
from recognition.active_learning import generate_prediction_pagexmls
from recognition.pagexml_line_dataset import (
    GEOMETRY_SOURCE_BASELINE_HEATMAP,
    prepare_page_line_dataset,
)
from tests.evaluate import evaluate_dataset, write_report_files
from tests.precommit_gate_config import get_pipeline_precommit_datasets


PIPELINE_PRECOMMIT_DATASETS = get_pipeline_precommit_datasets()
PIPELINE_OCR_WIDTH_POLICY = "batch_max_pad"
PIPELINE_OCR_SEGMENTATION_ARGS = {
    "BINARIZE_THRESHOLD": 0.5098,
    "BBOX_PAD_V": 0.7,
    "BBOX_PAD_H": 0.5,
    "CC_SIZE_THRESHOLD_RATIO": 0.4,
}


class EndToEndEvalDatasetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._original_upload_folder = backend_app_module.UPLOAD_FOLDER
        cls._original_model_checkpoint = backend_app_module.MODEL_CHECKPOINT
        cls._original_dataset_config = backend_app_module.DATASET_CONFIG
        cls._original_ocr_model_path = backend_app_module.OCR_MODEL_PATH
        cls._original_ocr_global_context = backend_app_module.OCR_GLOBAL_CONTEXT

        cls.upload_root = APP_ROOT / "input_manuscripts" / "_ci_root"
        if cls.upload_root.exists():
            shutil.rmtree(cls.upload_root)
        cls.upload_root.mkdir(parents=True, exist_ok=True)

        backend_app_module.UPLOAD_FOLDER = str(cls.upload_root)
        backend_app_module.MODEL_CHECKPOINT = str(APP_ROOT / "pretrained_gnn" / "v2.pt")
        backend_app_module.DATASET_CONFIG = str(APP_ROOT / "pretrained_gnn" / "gnn_preprocessing_v2.yaml")
        backend_app_module.OCR_MODEL_PATH = str(APP_ROOT / "recognition" / "pretrained_model" / "vadakautuhala.pth")
        backend_app_module.OCR_GLOBAL_CONTEXT = None
        backend_app_module.app.config["TESTING"] = True

        cls.client = backend_app_module.app.test_client()

    @classmethod
    def tearDownClass(cls):
        backend_app_module.UPLOAD_FOLDER = cls._original_upload_folder
        backend_app_module.MODEL_CHECKPOINT = cls._original_model_checkpoint
        backend_app_module.DATASET_CONFIG = cls._original_dataset_config
        backend_app_module.OCR_MODEL_PATH = cls._original_ocr_model_path
        backend_app_module.OCR_GLOBAL_CONTEXT = cls._original_ocr_global_context

        if os.getenv("KEEP_CI_ARTIFACTS") != "1" and cls.upload_root.exists():
            shutil.rmtree(cls.upload_root)

    def test_precommit_pipeline_datasets_end_to_end(self):
        self.assertTrue(Path(backend_app_module.MODEL_CHECKPOINT).exists(), "Missing pretrained GNN checkpoint.")
        self.assertTrue(Path(backend_app_module.DATASET_CONFIG).exists(), "Missing GNN preprocessing config.")
        self.assertTrue(Path(backend_app_module.OCR_MODEL_PATH).exists(), "Missing local OCR model.")

        for dataset_config in PIPELINE_PRECOMMIT_DATASETS:
            with self.subTest(dataset=dataset_config.name):
                self._assert_dataset_gate_passes(dataset_config)

    def _assert_dataset_gate_passes(self, dataset_config):
        expected_pages = dataset_config.ordered_page_ids()
        manuscript_root = self.upload_root / dataset_config.manuscript_name
        pred_folder = manuscript_root / "layout_analysis_output" / "page-xml-format"
        ocr_source_xml_dir = manuscript_root / "layout_analysis_output" / "_ci_ocr_source_page_xml"
        ocr_prepared_dir = manuscript_root / "layout_analysis_output" / "_ci_ocr_prepared_pages"
        ocr_prediction_dir = manuscript_root / "layout_analysis_output" / "_ci_ocr_prediction_page_xml"

        self.assertTrue(dataset_config.images_dir.exists(), f"Missing eval images directory: {dataset_config.images_dir}")
        self.assertTrue(dataset_config.pagexml_dir.exists(), f"Missing eval PAGE-XML directory: {dataset_config.pagexml_dir}")
        self.assertEqual(
            len(expected_pages),
            dataset_config.expected_page_count,
            f"Expected {dataset_config.expected_page_count} images for dataset {dataset_config.name}.",
        )

        if manuscript_root.exists():
            shutil.rmtree(manuscript_root)

        upload_response = self._upload_dataset(dataset_config)
        upload_json = upload_response.get_json()

        self.assertEqual(upload_response.status_code, 200, upload_json)
        self.assertEqual(sorted(upload_json["pages"]), expected_pages)

        pages_response = self.client.get(f"/manuscript/{dataset_config.manuscript_name}/pages")
        pages_json = pages_response.get_json()
        self.assertEqual(pages_response.status_code, 200, pages_json)
        self.assertEqual(sorted(pages_json["pages"]), expected_pages)

        for page in upload_json["pages"]:
            graph_response = self.client.get(f"/semi-segment/{dataset_config.manuscript_name}/{page}")
            graph_json = graph_response.get_json()
            self.assertEqual(graph_response.status_code, 200, graph_json)
            self.assertIn("graph", graph_json)
            self.assertGreater(len(graph_json["graph"]["nodes"]), 0, f"No nodes returned for page {page}")
            self.assertGreater(len(graph_json["graph"]["edges"]), 0, f"No edges returned for page {page}")

            node_count = len(graph_json["graph"]["nodes"])
            save_payload = {
                "graph": graph_json["graph"],
                "modifications": [],
                "textlineLabels": [-1] * node_count,
                "textboxLabels": [0] * node_count,
                "textContent": {},
                "runRecognition": False,
                "recognitionEngine": "local",
            }
            save_response = self.client.post(
                f"/semi-segment/{dataset_config.manuscript_name}/{page}",
                json=save_payload,
            )
            save_json = save_response.get_json()
            self.assertEqual(save_response.status_code, 200, save_json)
            self.assertEqual(save_json["status"], "success")
            self.assertGreater(save_json["lines"], 0, f"No text lines generated for page {page}")

        prepared_pages = {}
        if ocr_source_xml_dir.exists():
            shutil.rmtree(ocr_source_xml_dir)
        if ocr_prepared_dir.exists():
            shutil.rmtree(ocr_prepared_dir)
        ocr_source_xml_dir.mkdir(parents=True, exist_ok=True)

        for page in upload_json["pages"]:
            prepared_page = self._prepare_pretrained_gate_ocr_page(
                manuscript_root=manuscript_root,
                page=page,
                source_xml_dir=ocr_source_xml_dir,
                prepared_root=ocr_prepared_dir,
            )
            prepared_pages[page] = prepared_page
            self.assertGreater(len(prepared_page.records), 0, f"No OCR crop records prepared for page {page}")

        prediction_output = generate_prediction_pagexmls(
            backend_app_module.OCR_MODEL_PATH,
            prepared_pages,
            ocr_prediction_dir,
            width_policy=PIPELINE_OCR_WIDTH_POLICY,
        )
        prediction_folder = Path(prediction_output.prediction_folder)
        self.assertEqual(len(list(prediction_folder.glob("*.xml"))), len(expected_pages))

        for predicted_xml in sorted(prediction_folder.glob("*.xml")):
            shutil.copy2(predicted_xml, pred_folder / predicted_xml.name)
            self.assertGreater(
                self._count_pagexml_text_lines_with_text(pred_folder / predicted_xml.name),
                0,
                f"No OCR text returned for page {predicted_xml.stem}",
            )

        self.assertTrue(pred_folder.exists(), f"Prediction folder missing: {pred_folder}")
        self.assertEqual(len(list(pred_folder.glob("*.xml"))), len(expected_pages))

        result = evaluate_dataset(
            pred_folder=pred_folder,
            gt_folder=dataset_config.pagexml_dir,
            method_name=f"CI eval dataset ({dataset_config.name})",
            layout_type=dataset_config.layout_type,
        )
        LOGS_ROOT.mkdir(parents=True, exist_ok=True)
        write_report_files(
            result,
            text_path=LOGS_ROOT / "ci_eval_results_latest.txt",
            json_path=LOGS_ROOT / "ci_eval_results_latest.json",
        )

        aggregate = result["aggregate_metrics"]
        worst_page_line_cer = max(page["line_cer_50"] for page in result["per_page"])

        self.assertEqual(result["files_processed"], len(expected_pages))
        self.assertTrue(all(page["prediction_found"] for page in result["per_page"]))
        self.assertLessEqual(aggregate["page_cer"], dataset_config.max_page_cer)
        self.assertLessEqual(aggregate["line_cer_50"], dataset_config.max_line_cer_50)
        self.assertLessEqual(aggregate["line_cer_75"], dataset_config.max_line_cer_75)
        self.assertLessEqual(aggregate["line_cer_range"], dataset_config.max_line_cer_range)
        self.assertLessEqual(worst_page_line_cer, dataset_config.max_worst_page_line_cer_50)

    def _upload_dataset(self, dataset_config):
        image_paths = sorted(dataset_config.images_dir.glob("*.jpg"))

        with ExitStack() as stack:
            data = {
                "manuscriptName": dataset_config.manuscript_name,
                "longestSide": str(dataset_config.longest_side),
                "minDistance": str(dataset_config.min_distance),
                "images": [(stack.enter_context(open(path, "rb")), path.name) for path in image_paths],
            }
            return self.client.post("/upload", data=data, content_type="multipart/form-data")

    def _prepare_pretrained_gate_ocr_page(self, manuscript_root, page, source_xml_dir, prepared_root):
        xml_path = manuscript_root / "layout_analysis_output" / "page-xml-format" / f"{page}.xml"
        image_path = self._find_page_image(
            page,
            [
                manuscript_root / "layout_analysis_output" / "images_resized",
                manuscript_root / "images_resized",
            ],
        )
        heatmap_path = self._find_page_image(page, [manuscript_root / "heatmaps"])
        source_xml_path = source_xml_dir / f"{page}.xml"

        self.assertTrue(xml_path.exists(), f"Missing generated PAGE-XML for OCR prep: {xml_path}")
        self.assertIsNotNone(image_path, f"Missing resized page image for OCR prep: {page}")
        self.assertIsNotNone(heatmap_path, f"Missing heatmap for OCR prep: {page}")

        self._write_placeholder_text_pagexml(xml_path, source_xml_path)
        return prepare_page_line_dataset(
            source_xml_path,
            image_path,
            prepared_root / page,
            heatmap_path=heatmap_path,
            geometry_source=GEOMETRY_SOURCE_BASELINE_HEATMAP,
            segmentation_args=PIPELINE_OCR_SEGMENTATION_ARGS,
        )

    def _find_page_image(self, page, candidate_dirs):
        extensions = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG", ".TIF", ".TIFF")
        for directory in candidate_dirs:
            for extension in extensions:
                candidate = Path(directory) / f"{page}{extension}"
                if candidate.exists():
                    return candidate
        return None

    def _write_placeholder_text_pagexml(self, source_xml_path, target_xml_path):
        tree = ET.parse(source_xml_path)
        root = tree.getroot()

        def strip_namespace(tag):
            return tag.split("}", 1)[-1] if "}" in tag else tag

        def tag_namespace(tag):
            return tag.split("}", 1)[0].strip("{") if tag.startswith("{") else ""

        def qualified(tag, namespace):
            return f"{{{namespace}}}{tag}" if namespace else tag

        text_line_count = 0
        for textline in root.iter():
            if strip_namespace(textline.tag) != "TextLine":
                continue

            textline_namespace = tag_namespace(textline.tag)
            for child in list(textline):
                if strip_namespace(child.tag) == "TextEquiv":
                    textline.remove(child)

            text_equiv = ET.SubElement(textline, qualified("TextEquiv", textline_namespace))
            unicode_elem = ET.SubElement(text_equiv, qualified("Unicode", textline_namespace))
            unicode_elem.text = "x"
            text_line_count += 1

        self.assertGreater(text_line_count, 0, f"No TextLine elements found in {source_xml_path}")
        target_xml_path.parent.mkdir(parents=True, exist_ok=True)
        tree.write(target_xml_path, encoding="UTF-8", xml_declaration=True)

    def _count_pagexml_text_lines_with_text(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        def strip_namespace(tag):
            return tag.split("}", 1)[-1] if "}" in tag else tag

        count = 0
        for textline in root.iter():
            if strip_namespace(textline.tag) != "TextLine":
                continue
            for text_equiv in textline:
                if strip_namespace(text_equiv.tag) != "TextEquiv":
                    continue
                for child in text_equiv:
                    if strip_namespace(child.tag) == "Unicode" and child.text and child.text.strip():
                        count += 1
                        break
        return count


if __name__ == "__main__":
    unittest.main()
