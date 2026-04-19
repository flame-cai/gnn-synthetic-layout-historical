import os
import shutil
import sys
import unittest
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
from tests.evaluate import evaluate_dataset, write_report_files
from tests.precommit_gate_config import get_pipeline_precommit_datasets


PIPELINE_PRECOMMIT_DATASETS = get_pipeline_precommit_datasets()


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

            recog_response = self.client.post(
                "/recognize-text",
                json={
                    "manuscript": dataset_config.manuscript_name,
                    "page": page,
                    "recognitionEngine": "local",
                },
            )
            recog_json = recog_response.get_json()
            self.assertEqual(recog_response.status_code, 200, recog_json)
            self.assertIn("text", recog_json)
            self.assertGreater(len(recog_json["text"]), 0, f"No OCR text returned for page {page}")

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


if __name__ == "__main__":
    unittest.main()
