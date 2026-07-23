from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from shapely.geometry import GeometryCollection, box

from experiments.downstream_ocr.adapter import (
    extract_sarvam_html_text_lines,
    sarvam_html_to_pagexml,
)
from experiments.downstream_ocr.metrics import aggregate_page_records, evaluate_page
from experiments.downstream_ocr.pagexml import (
    PageXmlPage,
    TextLine,
    first_child,
    iter_descendants,
    load_pagexml,
    write_pagexml,
)
from experiments.downstream_ocr.reporting import write_experiment_report
from experiments.downstream_ocr.vlm_cache import acquire_manuscript_provider
from experiments.downstream_ocr.vlm_providers import (
    VlmProviderResponse,
    _invoke_sarvam,
    provider_by_id,
)


SARVAM_HTML = """<!DOCTYPE html>
<html lang="sa-IN">
<head>
  <meta charset="UTF-8">
  <style>p { color: red; }</style>
</head>
<body>
  <p class="paragraph">प्रथमा पङ्क्तिः<br/>
  द्वितीया पङ्क्तिः</p>
  <p class="folio">तृतीया <span>पङ्क्तिः</span></p>
  <h2 class="section-title">शीर्षकम्<br>अन्तिमा पङ्क्तिः</h2>
</body>
</html>
"""

SARVAM_SEMANTIC_HTML = """<!DOCTYPE html>
<html lang="sa-IN">
<head><title>यह पाठ नहीं आना चाहिए</title></head>
<body>
  <header class="header">शीर्षरेखा<br/>द्वितीयशीर्षरेखा</header>
  <aside class="sidebar">शास्रीकविवाह<br/>तासुर</aside>
  <div class="formula">सूत्रम्<br/>व्याख्या</div>
  <div class="unlisted-wrapper">अवर्गीकृतपङ्क्तिः<br/>द्वितीयावर्गीकृतपङ्क्तिः</div>
  <footer class="footer">=प्रश्नोत्तररूपवेद्वाक्य१=बृहत्पुरा<br/>
  एकादशः</footer>
  <footer class="footer"><p>अन्तःस्थपङ्क्तिः</p></footer>
</body>
</html>
"""

SARVAM_TABLE_HTML = """<body>
<div class="page-body-container">
<div class="table"><table>
<thead>
<tr>
<th class="nowrap">किया चूअकमे ५</th>
<th>ब्रह्मचारीइतिकेषः ४<br/>ब्रह्मचारीइतिकेषः ४</th>
</tr>
</thead>
<tbody>
<tr>
<td class="nowrap">M-233</td>
<td></td>
</tr>
<tr>
<td class="nowrap">याज्ञव०३</td>
<td>मनं घेवारलॄतये॥आदिमर्ध्यवसानेपुनबैच्छन्दोपलक्षिता॥ब्राह्मणप्रियविशे।जैत्रचर्यायथा<br/>क्रम॥कृताभिनकाकैन्जुजीतवाग्यतोगुर्चनुतया॥आपोशानकिंयापूर्वसल्लसानुमकुदाप<br/>न॥ब्रह्मणवेसिथतोएकमन्नमघारनापदि॥ब्राह्मणकाममश्रीयात्यादेव्रतमपीडयुत्तम<br/>धुमासांजनोदित्यसक्तस्त्रीप्राणिहंसने॥नास्करालेखनाप्रीलापरवादादिवर्जयेत्सःगुरुर्वःअरुन्धनाषरा॥४</td>
</tr>
<tr>
<td class="nowrap">गायत्रीव्यतिरिक्तः</td>
<td>क्रियारूलावैदमस्सौत्रइति॥उपनीयदददेद्माचार्यःसगुदाहृतः॥एकेदेशमुध्यायज्ञविद्युत<br/>रुदुच्यते॥एतेमान्यायथापूर्वमेन्यामातागरीयसी॥प्रतिवेदंब्रहत्कार्येद्वान्शाब्दानपंच<br/>पाणग्रहणपर्यन्तवा॥ग्रहणांतिकामत्यन्येकेषांत्तश्चैवषोडशो॥आषोउशाद्रादविशाश्वतुर्विशाञ्चवस राम ३</td>
</tr>
</tbody>
</table></div>
</div>
</body>
"""


def _template_page(page_id: str = "p1") -> PageXmlPage:
    return PageXmlPage(
        page_id=page_id,
        image_filename=f"{page_id}.jpg",
        width=100,
        height=80,
        lines=(),
    )


def _make_manuscript(root: Path) -> Path:
    manuscript = root / "manuscript"
    images = manuscript / "images_resized"
    pagexml = manuscript / "layout_analysis_output" / "page-xml-format"
    images.mkdir(parents=True)
    pagexml.mkdir(parents=True)
    (images / "p1.jpg").write_bytes(b"image")
    write_pagexml(_template_page(), pagexml / "p1.xml", lines=())
    return manuscript


class SarvamAdapterTests(unittest.TestCase):
    def test_html_blocks_and_breaks_become_atomic_text_lines(self):
        self.assertEqual(
            extract_sarvam_html_text_lines(SARVAM_HTML),
            (
                "प्रथमा पङ्क्तिः",
                "द्वितीया पङ्क्तिः",
                "तृतीया पङ्क्तिः",
                "शीर्षकम्",
                "अन्तिमा पङ्क्तिः",
            ),
        )

    def test_header_footer_aside_and_text_classes_are_extracted_once(self):
        self.assertEqual(
            extract_sarvam_html_text_lines(SARVAM_SEMANTIC_HTML),
            (
                "शीर्षरेखा",
                "द्वितीयशीर्षरेखा",
                "शास्रीकविवाह",
                "तासुर",
                "सूत्रम्",
                "व्याख्या",
                "अवर्गीकृतपङ्क्तिः",
                "द्वितीयावर्गीकृतपङ्क्तिः",
                "=प्रश्नोत्तररूपवेद्वाक्य१=बृहत्पुरा",
                "एकादशः",
                "अन्तःस्थपङ्क्तिः",
            ),
        )

    def test_standard_html_table_cells_and_breaks_are_text_lines(self):
        lines = extract_sarvam_html_text_lines(SARVAM_TABLE_HTML)

        self.assertEqual(len(lines), 13)
        self.assertEqual(
            lines[:6],
            (
                "किया चूअकमे ५",
                "ब्रह्मचारीइतिकेषः ४",
                "ब्रह्मचारीइतिकेषः ४",
                "M-233",
                "याज्ञव०३",
                "मनं घेवारलॄतये॥आदिमर्ध्यवसानेपुनबैच्छन्दोपलक्षिता॥ब्राह्मणप्रियविशे।जैत्रचर्यायथा",
            ),
        )
        self.assertNotIn("", lines)
        self.assertEqual(
            lines[-1],
            "पाणग्रहणपर्यन्तवा॥ग्रहणांतिकामत्यन्येकेषांत्तश्चैवषोडशो॥आषोउशाद्रादविशाश्वतुर्विशाञ्चवस राम ३",
        )

    def test_pagexml_has_one_region_and_explicitly_empty_geometry(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "p1.xml"
            sarvam_html_to_pagexml(
                SARVAM_HTML,
                template_page=_template_page(),
                output_path=output_path,
            )

            root = ET.parse(output_path).getroot()
            regions = list(iter_descendants(root, "TextRegion"))
            text_lines = list(iter_descendants(root, "TextLine"))
            self.assertEqual(len(regions), 1)
            self.assertEqual(len(text_lines), 5)
            for text_line in text_lines:
                self.assertEqual(first_child(text_line, "Coords").get("points"), "")
                self.assertEqual(first_child(text_line, "Baseline").get("points"), "")

            with self.assertRaisesRegex(ValueError, "missing TextLine/Coords"):
                load_pagexml(output_path)
            parsed = load_pagexml(output_path, allow_empty_geometry=True)
            self.assertTrue(all(line.polygon.is_empty for line in parsed.lines))
            self.assertEqual(
                [line.text for line in parsed.lines],
                list(extract_sarvam_html_text_lines(SARVAM_HTML)),
            )

    def test_text_only_metrics_leave_cer_and_layout_unavailable(self):
        gt_line = TextLine(
            page_id="p1",
            line_id="gt_1",
            points=((0, 0), (20, 0), (20, 10), (0, 10)),
            polygon=box(0, 0, 20, 10),
            text="समानम्",
        )
        pred_line = TextLine(
            page_id="p1",
            line_id="line_0",
            points=(),
            polygon=GeometryCollection(),
            text="समानम्",
            region_id="region_0",
        )
        record = evaluate_page(
            manuscript_id="m",
            fold_id="fold_1",
            page_id="p1",
            method_id="sarvam_e2e",
            gt_page=PageXmlPage("p1", "p1.jpg", 100, 80, (gt_line,)),
            pred_page=PageXmlPage("p1", "p1.jpg", 100, 80, (pred_line,)),
            calculate_layout_metrics=False,
            calculate_page_cer=False,
        )
        aggregate = aggregate_page_records([record])

        self.assertIsNone(record["page_cer"])
        self.assertIsNone(record["object_g_f1_50"])
        self.assertEqual(record["textedit_all_page_avg"], 0.0)
        self.assertEqual(aggregate["page_cer_page_count"], 0)
        self.assertEqual(aggregate["layout_metric_page_count"], 0)
        self.assertIsNone(aggregate["micro_page_cer"])
        self.assertIsNone(aggregate["pixel_f1"])
        self.assertEqual(aggregate["textedit_all_page_avg"], 0.0)

    def test_sarvam_sdk_job_contract_downloads_html(self):
        captured: dict = {}

        class FakeJob:
            job_id = "job_123"

            def upload_files(self, *, file_paths):
                captured["file_paths"] = file_paths

            def start(self):
                captured["started"] = True

            def wait_until_complete(self):
                return SimpleNamespace(job_state="Completed")

            def download_outputs(self, *, output_dir):
                Path(output_dir, "page_output.html").write_text(
                    SARVAM_HTML,
                    encoding="utf-8",
                )

        class FakeDocumentIntelligence:
            def create_job(self, *, job_parameters):
                captured["job_parameters"] = job_parameters
                return FakeJob()

        class FakeSarvamAI:
            def __init__(self, *, api_subscription_key):
                captured["api_subscription_key"] = api_subscription_key
                self.document_intelligence = FakeDocumentIntelligence()

        fake_module = types.ModuleType("sarvamai")
        fake_module.SarvamAI = FakeSarvamAI
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "p1.jpg"
            image_path.write_bytes(b"image")
            with patch.dict(sys.modules, {"sarvamai": fake_module}):
                response = _invoke_sarvam(
                    provider_by_id("sarvam"),
                    api_key="test-key",
                    image_path=image_path,
                    prompt=None,
                    timeout_seconds=600,
                )

        self.assertEqual(captured["api_subscription_key"], "test-key")
        self.assertEqual(
            captured["job_parameters"],
            {"language": "sa-IN", "output_format": "html"},
        )
        self.assertEqual(captured["file_paths"], [str(image_path)])
        self.assertTrue(captured["started"])
        self.assertEqual(response.raw_text, SARVAM_HTML)
        self.assertEqual(response.response_id, "job_123")

    def test_sarvam_current_sdk_contract_is_also_supported(self):
        captured: dict = {}

        class FakeJob:
            job_id = "job_current"

            def upload_file(self, file_path):
                captured["file_path"] = file_path

            def start(self):
                pass

            def wait_until_complete(self, timeout_seconds):
                captured["timeout_seconds"] = timeout_seconds
                return SimpleNamespace(job_state="JobState.Completed")

            def download_output(self, output_path):
                with zipfile.ZipFile(output_path, "w") as archive:
                    archive.writestr("p1_output.html", SARVAM_HTML)

        class FakeDocumentIntelligence:
            def create_job(self, *, language, output_format):
                captured["language"] = language
                captured["output_format"] = output_format
                return FakeJob()

        class FakeSarvamAI:
            def __init__(self, *, api_subscription_key):
                self.document_intelligence = FakeDocumentIntelligence()

        fake_module = types.ModuleType("sarvamai")
        fake_module.SarvamAI = FakeSarvamAI
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "p1.jpg"
            image_path.write_bytes(b"image")
            with patch.dict(sys.modules, {"sarvamai": fake_module}):
                response = _invoke_sarvam(
                    provider_by_id("sarvam"),
                    api_key="test-key",
                    image_path=image_path,
                    prompt=None,
                    timeout_seconds=600,
                )

        self.assertEqual(captured["language"], "sa-IN")
        self.assertEqual(captured["output_format"], "html")
        self.assertEqual(captured["file_path"], str(image_path))
        self.assertEqual(captured["timeout_seconds"], 600)
        self.assertEqual(response.raw_text, SARVAM_HTML)

    def test_cache_fingerprints_sarvam_contract_and_adapts_html(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manuscript = _make_manuscript(root)
            cache_root = root / "cache"

            def fake_invoke(spec, **kwargs):
                self.assertEqual(spec.provider_id, "sarvam")
                self.assertIsNone(kwargs["prompt"])
                return VlmProviderResponse(raw_text=SARVAM_HTML, response_id="job_1")

            with patch.dict("os.environ", {"SARVAM_API_KEY": "test-key"}):
                manifest = acquire_manuscript_provider(
                    manuscript_root=manuscript,
                    cache_root=cache_root,
                    provider_id="sarvam",
                    env_path=root / ".env",
                    page_workers=1,
                    request_spacing_seconds=0,
                    max_retries=0,
                    invoke=fake_invoke,
                )

            page_dir = cache_root / "manuscript" / "sarvam_e2e" / "pages" / "p1"
            request = json.loads((page_dir / "request.json").read_text(encoding="utf-8"))
            self.assertNotIn("prompt", request)
            self.assertEqual(
                request["request_parameters"],
                {"language": "sa-IN", "output_format": "html"},
            )
            self.assertEqual(manifest["success_count"], 1)
            prediction = load_pagexml(
                page_dir / "prediction.xml",
                allow_empty_geometry=True,
            )
            self.assertEqual(len(prediction.lines), 5)

    def test_report_uses_sarvam_output_order_for_cer(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            gt_line = TextLine(
                page_id="p1",
                line_id="gt_1",
                points=((0, 0), (20, 0), (20, 10), (0, 10)),
                polygon=box(0, 0, 20, 10),
                text="समानम्",
            )
            pred_line = TextLine(
                page_id="p1",
                line_id="line_0",
                points=(),
                polygon=GeometryCollection(),
                text="समानम्",
            )
            record = evaluate_page(
                manuscript_id="m",
                fold_id="fold_1",
                page_id="p1",
                method_id="sarvam_e2e",
                gt_page=PageXmlPage("p1", "p1.jpg", 100, 80, (gt_line,)),
                pred_page=PageXmlPage("p1", "p1.jpg", 100, 80, (pred_line,)),
                calculate_layout_metrics=False,
                calculate_page_cer=True,
                page_cer_predicted_lines_in_output_order=True,
            )
            metrics_path = root / "metrics" / "sarvam_e2e" / "metrics.json"
            metrics_path.parent.mkdir(parents=True)
            metrics_path.write_text(
                json.dumps(
                    {
                        "method": {
                            "method_id": "sarvam_e2e",
                            "display_name": "Sarvam (End-to-End)",
                            "uses_gt_layout": False,
                            "uses_finetuning": False,
                            "finetune_page_count": 0,
                            "provider_id": "sarvam",
                            "model_id": "sarvam-vision",
                            "provides_layout": False,
                        },
                        "manuscript_id": "m",
                        "aggregate": aggregate_page_records([record]),
                        "page_records": [record],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            artifacts = write_experiment_report(root)
            table = json.loads(
                artifacts.off_the_shelf_table_json_path.read_text(encoding="utf-8")
            )
            self.assertEqual(table["rows"][0]["method_id"], "sarvam_e2e")
            self.assertEqual(table["rows"][0]["micro_page_cer"], 0.0)
            self.assertEqual(table["rows"][0]["textedit_all_page_avg"], 0.0)
            self.assertIn("0.0000", artifacts.markdown_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
