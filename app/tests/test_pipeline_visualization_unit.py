from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from app.pipeline_visualization import (
    layout_artifacts,
    read_mode_pagexml_artifact,
    upload_artifacts,
)


class PipelineVisualizationUnitTests(unittest.TestCase):
    def test_sidecar_writes_all_available_pipeline_stages(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            page = "page_001"
            (root / "images_resized").mkdir()
            (root / "heatmaps").mkdir()
            (root / "gnn-dataset").mkdir()
            crop_root = root / "layout_analysis_output" / "image-format" / page / "textbox_label_0"
            crop_root.mkdir(parents=True)
            (root / "processing_settings.json").write_text(
                json.dumps({"pipeline_visualization": {"enabled": True, "max_line_previews": 4}}),
                encoding="utf-8",
            )
            image = np.full((80, 120, 3), 220, dtype=np.uint8)
            cv2.imwrite(str(root / "images_resized" / f"{page}.jpg"), image)
            cv2.imwrite(str(root / "heatmaps" / f"{page}.jpg"), np.full((40, 60), 127, dtype=np.uint8))
            np.savetxt(
                root / "gnn-dataset" / f"{page}_inputs_normalized.txt",
                [[0.2, 0.3, 0], [0.7, 0.6, 0], [0.45, 0.5, 0]],
            )
            np.savetxt(root / "gnn-dataset" / f"{page}_dims.txt", [60, 40])
            cv2.imwrite(str(crop_root / "line_0.jpg"), np.full((30, 100), 200, dtype=np.uint8))

            upload_artifacts(root, page)
            layout_artifacts(
                root,
                page,
                baseline_graph={
                    "nodes": [{"x": 24, "y": 24}, {"x": 84, "y": 48}],
                    "edges": [{"source": 0, "target": 1}],
                },
                corrected_graph={
                    "nodes": [{"x": 24, "y": 24}, {"x": 104, "y": 18}],
                    "edges": [{"source": 0, "target": 1}],
                },
            )
            xml_path = root / "layout_analysis_output" / "page-xml-format" / f"{page}.xml"
            xml_path.parent.mkdir(parents=True)
            xml_path.write_text(
                '<PcGts><TextLine id="line_0" custom="structure_line_id_0"><TextEquiv><Unicode>राम</Unicode></TextEquiv></TextLine></PcGts>',
                encoding="utf-8",
            )
            read_mode_pagexml_artifact(root, page, xml_path, ground_truth=False)
            xml_path.write_text(
                '<PcGts><TextLine id="line_0" custom="structure_line_id_0"><TextEquiv><Unicode>रमा</Unicode></TextEquiv></TextLine></PcGts>',
                encoding="utf-8",
            )
            read_mode_pagexml_artifact(root, page, xml_path, ground_truth=True)

            output = root / "visualizations" / page
            expected = {
                "01_original_image.jpg",
                "02_craft_heatmap.jpg",
                "03_gnn_preprocessing.jpg",
                "04_05_layout_graph_correction_diff.jpg",
                "06_processed_line_images.jpg",
                "06_processed_line_images/textbox_label_0/line_0.jpg",
                "07_ocr_predictions_page.xml",
                "08_ocr_ground_truth_page.xml",
                "07_08_ocr_text_correction_diff.jpg",
                "07_08_ocr_text_correction_diff.json",
                "07_08_ocr_text_correction_diff/line_0.png",
                "manifest.json",
            }
            self.assertTrue(all((output / filename).is_file() for filename in expected))
            preprocessing = cv2.imread(str(output / "03_gnn_preprocessing.jpg"))
            correction_diff = cv2.imread(str(output / "04_05_layout_graph_correction_diff.jpg"))
            text_diff = cv2.imread(str(output / "07_08_ocr_text_correction_diff.jpg"))
            self.assertGreater(preprocessing.shape[1], image.shape[1])  # feature legend panel
            self.assertTrue(np.any((correction_diff[:, :, 0] > 120) & (correction_diff[:, :, 2] < 90)))  # blue human addition
            self.assertTrue(np.any((correction_diff[:, :, 2] > 160) & (correction_diff[:, :, 0] < 90)))  # vermilion human deletion
            self.assertTrue(np.any((text_diff[:, :, 0] > 120) & (text_diff[:, :, 2] < 90)))  # same blue semantic in Unicode diff
            self.assertTrue(np.any((text_diff[:, :, 2] > 160) & (text_diff[:, :, 0] < 90)))  # same vermilion semantic in Unicode diff
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["stages"]["layout_graph_diff"]["status"], "available")
            self.assertEqual(manifest["stages"]["ocr_text_comparison"]["status"], "available")
            self.assertIn("राम", (output / "07_ocr_predictions_page.xml").read_text(encoding="utf-8"))
            self.assertIn("रमा", (output / "08_ocr_ground_truth_page.xml").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
