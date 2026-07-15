import json
import io
import shutil
import sys
import unittest
from pathlib import Path

from PIL import Image


TESTS_ROOT = Path(__file__).resolve().parent
APP_ROOT = TESTS_ROOT.parent
REPO_ROOT = APP_ROOT.parent

if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.backend_app_import import backend_app_module


class ReadModeLineImagePreviewsUnitTest(unittest.TestCase):
    def setUp(self):
        self.tmp_root = TESTS_ROOT / "_tmp_read_mode_line_previews"
        if self.tmp_root.exists():
            shutil.rmtree(self.tmp_root)
        self.upload_root = self.tmp_root / "input_manuscripts"
        self.manuscript = "preview_manuscript"
        self.page = "233_0001"
        self.manuscript_root = self.upload_root / self.manuscript
        self.xml_dir = self.manuscript_root / "layout_analysis_output" / "page-xml-format"
        self.image_dir = self.manuscript_root / "layout_analysis_output" / "image-format" / self.page / "textbox_label_0"
        self.xml_dir.mkdir(parents=True, exist_ok=True)
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.old_upload_folder = backend_app_module.UPLOAD_FOLDER
        backend_app_module.UPLOAD_FOLDER = str(self.upload_root)

    def tearDown(self):
        backend_app_module.UPLOAD_FOLDER = self.old_upload_folder
        if self.tmp_root.exists():
            shutil.rmtree(self.tmp_root)

    def _write_page_xml(self):
        xml_path = self.xml_dir / f"{self.page}.xml"
        xml_path.write_text(
            """<?xml version="1.0" encoding="UTF-8"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">
  <Page imageFilename="233_0001.jpg" imageWidth="240" imageHeight="120">
    <TextRegion id="region_0" custom="textbox_label_0">
      <TextLine id="region_0_line_0" custom="structure_line_id_1">
        <Coords points="10,10 110,10 110,30 10,30"/>
        <Baseline points="10,20 110,20"/>
      </TextLine>
      <TextLine id="region_0_line_1" custom="structure_line_id_2">
        <Coords points="10,40 120,35 130,60 20,65"/>
        <Baseline points="10,55 60,40 120,55"/>
        <TextEquiv custom="auto_orientation_model:decoded_text_devanagari_evidence_v1;auto_orientation_transform:rotate_180;auto_orientation_reason:test">
          <Unicode>राम</Unicode>
        </TextEquiv>
      </TextLine>
      <TextLine id="region_0_line_2" custom="structure_line_id_3">
        <Coords points="10,80 110,80 110,100 10,100"/>
        <Baseline points="110,90 10,90"/>
        <TextEquiv custom="auto_orientation_transform:rotate_180">
          <Unicode>annotated</Unicode>
        </TextEquiv>
      </TextLine>
    </TextRegion>
  </Page>
</PcGts>
""",
            encoding="utf-8",
        )
        return xml_path

    def _write_line_images(self):
        for line_id in (1, 2, 3):
            image = Image.new("L", (100 + line_id, 24), color=240)
            if line_id in {2, 3}:
                image.paste(20, (0, 0, 36, 24))
            image.save(self.image_dir / f"line_{line_id}.jpg")

    def _write_metadata(self):
        (self.xml_dir / f"{self.page}_line_segmentation_metadata.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "strategy_name": "local_polygons_stable_unwrap_v1",
                    "line_metadata": [
                        {"line_numeric_id": 1, "line_kind": "horizontal_straight"},
                        {"line_numeric_id": 2, "line_kind": "curved_open"},
                        {
                            "line_numeric_id": 3,
                            "line_kind": "horizontal_straight",
                            "reading_direction_annotation": {
                                "reading_direction": [1, 0],
                                "cut_midpoint": [60, 90],
                            },
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        (self.xml_dir / f"{self.page}_reading_direction_metadata.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "line_annotations": [
                        {
                            "resolved_line_numeric_id": 3,
                            "reading_direction": [1, 0],
                            "cut_midpoint": [60, 90],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

    def test_preview_payload_selects_only_non_straight_or_annotated_lines(self):
        xml_path = self._write_page_xml()
        self._write_line_images()
        self._write_metadata()

        previews = backend_app_module.get_existing_line_image_previews(
            self.manuscript,
            self.page,
            xml_path,
        )

        self.assertNotIn("1", previews)
        self.assertEqual(previews["2"]["lineKind"], "curved_open")
        self.assertFalse(previews["2"]["hasReadingDirectionAnnotation"])
        self.assertEqual(previews["2"]["imageWidth"], 102)
        self.assertEqual(previews["2"]["autoOrientationTransform"], "rotate_180")
        self.assertEqual(previews["3"]["lineKind"], "horizontal_straight")
        self.assertTrue(previews["3"]["hasReadingDirectionAnnotation"])
        self.assertIsNone(previews["3"]["autoOrientationTransform"])
        self.assertIn(f"/line-image/{self.manuscript}/{self.page}/2", previews["2"]["imageUrl"])

    def test_line_image_route_serves_processed_crop_by_line_id(self):
        self._write_page_xml()
        self._write_line_images()
        self._write_metadata()

        client = backend_app_module.app.test_client()
        response = client.get(f"/line-image/{self.manuscript}/{self.page}/2")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "image/jpeg")
        self.assertGreater(len(response.data), 0)
        with Image.open(io.BytesIO(response.data)) as served_image:
            grayscale = served_image.convert("L")
            left_pixel = grayscale.getpixel((10, 12))
            right_pixel = grayscale.getpixel((90, 12))
        self.assertGreater(left_pixel, right_pixel)
        response.close()

    def test_explicit_annotation_overrides_stale_auto_orientation_in_preview_route(self):
        xml_path = self._write_page_xml()
        self._write_line_images()
        self._write_metadata()
        (self.xml_dir / f"{self.page}_reading_direction_metadata.json").unlink()

        previews = backend_app_module.get_existing_line_image_previews(
            self.manuscript,
            self.page,
            xml_path,
        )
        self.assertTrue(previews["3"]["hasReadingDirectionAnnotation"])
        self.assertIsNone(previews["3"]["autoOrientationTransform"])

        client = backend_app_module.app.test_client()
        response = client.get(f"/line-image/{self.manuscript}/{self.page}/3")

        self.assertEqual(response.status_code, 200)
        with Image.open(io.BytesIO(response.data)) as served_image:
            grayscale = served_image.convert("L")
            left_pixel = grayscale.getpixel((10, 12))
            right_pixel = grayscale.getpixel((90, 12))
        self.assertLess(left_pixel, right_pixel)
        response.close()


if __name__ == "__main__":
    unittest.main()
