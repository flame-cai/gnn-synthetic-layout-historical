import importlib
import io
import shutil
import sys
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from PIL import Image


TESTS_ROOT = Path(__file__).resolve().parent
APP_ROOT = TESTS_ROOT.parent
REPO_ROOT = APP_ROOT.parent

if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

backend_app_module = importlib.import_module("app.app")


class DownloadResultsExportUnitTest(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        tmp_root = TESTS_ROOT / "_tmp_download_results_unit"
        if tmp_root.exists():
            shutil.rmtree(tmp_root)

    def _make_root(self, name: str) -> Path:
        root = TESTS_ROOT / "_tmp_download_results_unit" / name
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)
        return root

    def test_download_exports_labeled_ocr_training_pairs_and_resized_images(self):
        upload_root = self._make_root("labeled_pairs")
        manuscript = "manuscript_a"
        manuscript_root = upload_root / manuscript
        output_root = manuscript_root / "layout_analysis_output"
        xml_dir = output_root / "page-xml-format"
        image_region_dir = output_root / "image-format" / "233_0001" / "textbox_label_3"
        resized_dir = output_root / "images_resized"
        xml_dir.mkdir(parents=True)
        image_region_dir.mkdir(parents=True)
        resized_dir.mkdir(parents=True)

        Image.new("L", (12, 6), color=210).save(image_region_dir / "line_7.jpg")
        Image.new("L", (12, 6), color=220).save(image_region_dir / "line_8.jpg")
        Image.new("RGB", (20, 10), color=(255, 255, 255)).save(resized_dir / "233_0001.jpg")
        Image.new("RGB", (20, 10), color=(240, 240, 240)).save(resized_dir / "233_0002.jpg")
        saved_label = "\u0930\u093e\u092e"

        (xml_dir / "233_0001.xml").write_text(
            f"""<?xml version="1.0" encoding="UTF-8"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">
  <Page imageFilename="233_0001.jpg" imageWidth="20" imageHeight="10">
    <TextRegion id="region_0" custom="textbox_label_3">
      <Coords points="0,0 10,0 10,10 0,10" />
      <TextLine id="line_0" custom="structure_line_id_7">
        <Coords points="0,0 10,0 10,5 0,5" />
        <TextEquiv><Unicode>{saved_label}</Unicode></TextEquiv>
      </TextLine>
      <TextLine id="line_1" custom="structure_line_id_8">
        <Coords points="0,5 10,5 10,10 0,10" />
      </TextLine>
      <TextLine id="line_2" custom="structure_line_id_9">
        <Coords points="0,10 10,10 10,15 0,15" />
        <TextEquiv><Unicode>missing image</Unicode></TextEquiv>
      </TextLine>
    </TextRegion>
  </Page>
</PcGts>
""",
            encoding="utf-8",
        )

        client = backend_app_module.app.test_client()
        with mock.patch.object(backend_app_module, "UPLOAD_FOLDER", str(upload_root)):
            response = client.get(f"/download-results/{manuscript}")

        self.assertEqual(response.status_code, 200)
        with zipfile.ZipFile(io.BytesIO(response.data)) as zf:
            names = set(zf.namelist())
            self.assertIn("page-xml-format/233_0001.xml", names)
            self.assertIn("image-format/233_0001/textbox_label_3/line_7.jpg", names)
            self.assertIn("image-format/233_0001/textbox_label_3/line_8.jpg", names)
            self.assertIn("images_resized/233_0001.jpg", names)
            self.assertNotIn("images_resized/233_0002.jpg", names)
            self.assertIn(
                "ocr-training-format/233_0001/textbox_label_3/text-line-images/line_7.jpg",
                names,
            )
            self.assertNotIn(
                "ocr-training-format/233_0001/textbox_label_3/text-line-images/line_8.jpg",
                names,
            )
            self.assertNotIn(
                "ocr-training-format/233_0001/textbox_label_3/text-line-images/line_9.jpg",
                names,
            )
            gt_text = zf.read("ocr-training-format/233_0001/textbox_label_3/gt.txt").decode("utf-8")

        self.assertEqual(gt_text, f"text-line-images/line_7.jpg\t{saved_label}\n")

    def test_download_requires_at_least_one_annotated_page_layout(self):
        upload_root = self._make_root("no_layouts")
        manuscript = "manuscript_b"
        (upload_root / manuscript / "layout_analysis_output" / "page-xml-format").mkdir(parents=True)

        client = backend_app_module.app.test_client()
        with mock.patch.object(backend_app_module, "UPLOAD_FOLDER", str(upload_root)):
            response = client.get(f"/download-results/{manuscript}")

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.get_json()["error"], "No annotated page layouts found for this manuscript")


if __name__ == "__main__":
    unittest.main()
