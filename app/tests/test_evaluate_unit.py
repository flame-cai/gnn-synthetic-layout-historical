import shutil
import sys
import unittest
from pathlib import Path


TESTS_ROOT = Path(__file__).resolve().parent
APP_ROOT = TESTS_ROOT.parent

if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tests.evaluate import evaluate_dataset


PAGE_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"


def _line_xml(line_id, text, coords, baseline):
    return f"""
      <TextLine id="{line_id}" custom="{line_id}">
        <Coords points="{coords}" />
        <Baseline points="{baseline}" />
        <TextEquiv><Unicode>{text}</Unicode></TextEquiv>
      </TextLine>"""


def _write_page(path, lines):
    path.write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<PcGts xmlns="{PAGE_NS}">
  <Page imageFilename="{path.stem}.jpg" imageWidth="200" imageHeight="200">
    <TextRegion id="region_0" custom="textbox_label_0">
{''.join(lines)}
    </TextRegion>
  </Page>
</PcGts>
""",
        encoding="utf-8",
    )


class EvaluateDatasetUnitTest(unittest.TestCase):
    def setUp(self):
        self.tmp_root = TESTS_ROOT / "_tmp_evaluate_unit"
        if self.tmp_root.exists():
            shutil.rmtree(self.tmp_root)
        self.gt_dir = self.tmp_root / "gt"
        self.pred_dir = self.tmp_root / "pred"
        self.gt_dir.mkdir(parents=True)
        self.pred_dir.mkdir(parents=True)

    def tearDown(self):
        if self.tmp_root.exists():
            shutil.rmtree(self.tmp_root)

    def test_page_cer_orders_lines_by_baseline_not_coords(self):
        _write_page(
            self.gt_dir / "page_1.xml",
            [
                _line_xml("line_a", "A", "0,0 100,0 100,20 0,20", "0,10 100,10"),
                _line_xml("line_b", "B", "0,100 100,100 100,120 0,120", "0,110 100,110"),
            ],
        )
        _write_page(
            self.pred_dir / "page_1.xml",
            [
                _line_xml("line_a", "A", "0,100 100,100 100,120 0,120", "0,10 100,10"),
                _line_xml("line_b", "B", "0,0 100,0 100,20 0,20", "0,110 100,110"),
            ],
        )

        result = evaluate_dataset(self.pred_dir, self.gt_dir, "unit", layout_type="simple")

        self.assertEqual(result["per_page"][0]["page_cer"], 0.0)
        self.assertEqual(result["aggregate_metrics"]["page_cer"], 0.0)

    def test_simple_layout_keeps_short_width_lines_for_page_cer(self):
        _write_page(
            self.gt_dir / "page_1.xml",
            [
                _line_xml("line_1", "LONGONE", "0,0 100,0 100,20 0,20", "0,10 100,10"),
                _line_xml("line_2", "LONGTWO", "0,30 100,30 100,50 0,50", "0,40 100,40"),
                _line_xml("line_3", "S", "0,60 10,60 10,80 0,80", "0,70 10,70"),
            ],
        )
        _write_page(
            self.pred_dir / "page_1.xml",
            [
                _line_xml("line_1", "LONGONE", "0,0 100,0 100,20 0,20", "0,10 100,10"),
                _line_xml("line_2", "LONGTWO", "0,30 100,30 100,50 0,50", "0,40 100,40"),
            ],
        )

        result = evaluate_dataset(self.pred_dir, self.gt_dir, "unit", layout_type="simple")

        self.assertGreater(result["per_page"][0]["page_cer"], 0.0)
        self.assertEqual(result["per_page"][0]["gt_len"], len("LONGONE") + len("LONGTWO") + len("S"))


if __name__ == "__main__":
    unittest.main()
