import shutil
import sys
import unittest
from pathlib import Path


TESTS_ROOT = Path(__file__).resolve().parent
APP_ROOT = TESTS_ROOT.parent
REPO_ROOT = APP_ROOT.parent

if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from text_recovery import (  # noqa: E402
    backup_page_xml_for_text_recovery,
    build_text_recovery_plan,
    build_text_recovery_state,
    latest_text_recovery_backup,
)


class TextRecoveryUnitTest(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        tmp_root = TESTS_ROOT / "_tmp_text_recovery_unit"
        if tmp_root.exists():
            shutil.rmtree(tmp_root)

    def setUp(self):
        self.tmp_root = TESTS_ROOT / "_tmp_text_recovery_unit" / self._testMethodName
        if self.tmp_root.exists():
            shutil.rmtree(self.tmp_root)
        self.xml_dir = self.tmp_root / "layout_analysis_output" / "page-xml-format"
        self.xml_dir.mkdir(parents=True, exist_ok=True)

    def _write_xml(self, path: Path, lines: list[dict]):
        line_xml = []
        for line in lines:
            coords = " ".join(f"{x},{y}" for x, y in line.get("coords", []))
            text = line.get("text", "")
            text_equiv = f"<TextEquiv><Unicode>{text}</Unicode></TextEquiv>" if text else ""
            baseline_points = line.get("baseline", [])
            baseline = (
                f'<Baseline points="{" ".join(f"{x},{y}" for x, y in baseline_points)}"/>'
                if baseline_points
                else ""
            )
            line_xml.append(
                f'<TextLine id="line_{line["id"]}" custom="structure_line_id_{line["id"]}">'
                f'<Coords points="{coords}"/>'
                f"{baseline}"
                f"{text_equiv}"
                f"</TextLine>"
            )
        path.write_text(
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">'
            "<Page><TextRegion>"
            + "".join(line_xml)
            + "</TextRegion></Page></PcGts>",
            encoding="utf-8",
        )

    def test_layout_backup_is_versioned_and_reported_available_after_current_ocr_exists(self):
        current_xml = self.xml_dir / "233_0001.xml"
        self._write_xml(
            current_xml,
            [
                {"id": "0", "text": "rama", "coords": [(0, 0), (100, 0), (100, 20), (0, 20)]},
            ],
        )

        backup = backup_page_xml_for_text_recovery(self.tmp_root, "233_0001", xml_path=current_xml)
        latest = latest_text_recovery_backup(self.tmp_root, "233_0001")
        state = build_text_recovery_state(self.tmp_root, "233_0001", current_xml_path=current_xml)

        self.assertIsNotNone(backup)
        self.assertEqual(latest["backup_id"], backup["backup_id"])
        self.assertTrue(Path(latest["backup_xml"]).exists())
        self.assertTrue(state["available"])
        self.assertEqual(state["backup_text_line_count"], 1)
        self.assertEqual(state["current_text_line_count"], 1)

    def test_backup_records_whether_page_had_read_mode_annotations(self):
        current_xml = self.xml_dir / "233_0001.xml"
        self._write_xml(
            current_xml,
            [{"id": "0", "text": "rama", "coords": [(0, 0), (100, 0), (100, 20), (0, 20)]}],
        )

        backup_page_xml_for_text_recovery(
            self.tmp_root, "233_0001", xml_path=current_xml, had_read_mode_annotations=True
        )
        annotated_state = build_text_recovery_state(
            self.tmp_root, "233_0001", current_xml_path=current_xml
        )
        self.assertTrue(annotated_state["had_read_mode_annotations"])

        backup_page_xml_for_text_recovery(
            self.tmp_root, "233_0001", xml_path=current_xml, had_read_mode_annotations=False
        )
        prediction_only_state = build_text_recovery_state(
            self.tmp_root, "233_0001", current_xml_path=current_xml
        )
        self.assertFalse(prediction_only_state["had_read_mode_annotations"])

    def test_backup_without_recorded_annotation_flag_reports_unknown(self):
        current_xml = self.xml_dir / "233_0001.xml"
        self._write_xml(
            current_xml,
            [{"id": "0", "text": "rama", "coords": [(0, 0), (100, 0), (100, 20), (0, 20)]}],
        )

        backup_page_xml_for_text_recovery(self.tmp_root, "233_0001", xml_path=current_xml)
        state = build_text_recovery_state(self.tmp_root, "233_0001", current_xml_path=current_xml)

        # Unknown, not False: the GUI must still warn for legacy backups.
        self.assertIsNone(state["had_read_mode_annotations"])

    def test_recovery_plan_matches_by_text_similarity_and_coords_overlap(self):
        backup_xml = self.xml_dir / "backup.xml"
        current_xml = self.xml_dir / "current.xml"
        self._write_xml(
            backup_xml,
            [
                {"id": "0", "text": "ramena", "coords": [(0, 0), (100, 0), (100, 20), (0, 20)]},
                {"id": "1", "text": "sita", "coords": [(0, 40), (100, 40), (100, 60), (0, 60)]},
            ],
        )
        self._write_xml(
            current_xml,
            [
                {"id": "10", "text": "ramena", "coords": [(2, 1), (102, 1), (102, 21), (2, 21)]},
                {"id": "11", "text": "laksmana", "coords": [(0, 80), (100, 80), (100, 100), (0, 100)]},
            ],
        )

        plan = build_text_recovery_plan(backup_xml, current_xml)

        self.assertEqual(plan["matched_line_count"], 1)
        self.assertEqual(plan["matches"][0]["backup_line_id"], "0")
        self.assertEqual(plan["matches"][0]["current_line_id"], "10")
        self.assertEqual(plan["matches"][0]["recovered_text"], "ramena")
        self.assertEqual(plan["unrecovered_line_ids"], ["11"])

    def test_recovery_plan_skips_ambiguous_repeated_text_without_coords_disambiguation(self):
        backup_xml = self.xml_dir / "ambiguous_backup.xml"
        current_xml = self.xml_dir / "ambiguous_current.xml"
        self._write_xml(
            backup_xml,
            [
                {"id": "0", "text": "rama", "coords": []},
                {"id": "1", "text": "rama", "coords": []},
            ],
        )
        self._write_xml(
            current_xml,
            [
                {"id": "10", "text": "rama", "coords": []},
            ],
        )

        plan = build_text_recovery_plan(backup_xml, current_xml)

        self.assertEqual(plan["matched_line_count"], 0)
        self.assertEqual(plan["unrecovered_line_ids"], ["10"])
        self.assertEqual(plan["skipped"][0]["reason"], "ambiguous")


    # ---- geometry pass -------------------------------------------------

    @staticmethod
    def _line(line_id, text, top, *, baseline_top=None, x0=0, x1=200, height=20):
        """A horizontal line at a given y, with a matching baseline."""
        base_y = top + height if baseline_top is None else baseline_top + height
        return {
            "id": line_id,
            "text": text,
            "coords": [(x0, top), (x1, top), (x1, top + height), (x0, top + height)],
            "baseline": [(x0, base_y), ((x0 + x1) // 2, base_y), (x1, base_y)],
        }

    def _plan_for(self, backup_lines, current_lines):
        backup_xml = self.xml_dir / "backup.xml"
        current_xml = self.xml_dir / "current.xml"
        self._write_xml(backup_xml, backup_lines)
        self._write_xml(current_xml, current_lines)
        return build_text_recovery_plan(backup_xml, current_xml)

    def test_geometry_pass_recovers_a_line_whose_new_reading_is_wrong(self):
        plan = self._plan_for(
            [self._line("0", "rama", 0), self._line("1", "sita", 100)],
            # line 1's fresh reading is garbage: no text evidence at all
            [self._line("7", "rama", 0), self._line("8", "xq", 100)],
        )
        by_current = {m["current_line_id"]: m for m in plan["matches"]}
        self.assertEqual(by_current["7"]["match_method"], "text")
        self.assertEqual(by_current["8"]["match_method"], "baseline_geometry")
        self.assertEqual(by_current["8"]["recovered_text"], "sita")
        self.assertEqual(plan["geometry_matched_line_count"], 1)
        self.assertEqual(plan["geometry_matched_line_ids"], ["8"])
        self.assertEqual(plan["unrecovered_line_ids"], [])

    def test_geometry_pass_recovers_a_line_the_reader_left_empty(self):
        # An empty current line is invisible to the text pass, and is the safest
        # line to restore because there is nothing to overwrite.
        plan = self._plan_for(
            [self._line("0", "rama", 0), self._line("1", "sita", 100)],
            [self._line("7", "rama", 0), self._line("8", "", 100)],
        )
        by_current = {m["current_line_id"]: m for m in plan["matches"]}
        self.assertEqual(by_current["8"]["match_method"], "baseline_geometry")
        self.assertEqual(by_current["8"]["recovered_text"], "sita")

    def test_geometry_pass_declines_when_the_baseline_moved(self):
        plan = self._plan_for(
            [self._line("0", "rama", 0), self._line("1", "sita", 100)],
            # the human re-cut this line, so its baseline is somewhere else
            [self._line("7", "rama", 0), self._line("8", "xq", 400)],
        )
        self.assertEqual(plan["geometry_matched_line_count"], 0)
        self.assertEqual(plan["unrecovered_line_ids"], ["8"])

    def test_geometry_pass_never_blanks_a_line_from_an_empty_backup_line(self):
        plan = self._plan_for(
            [self._line("0", "rama", 0), self._line("1", "", 100)],
            [self._line("7", "rama", 0), self._line("8", "xq", 100)],
        )
        self.assertEqual(plan["geometry_matched_line_count"], 0)
        self.assertEqual(plan["unrecovered_line_ids"], ["8"])

    def test_geometry_pass_declines_when_two_backup_lines_sit_on_the_same_baseline(self):
        plan = self._plan_for(
            [self._line("0", "rama", 100), self._line("1", "sita", 100)],
            [self._line("7", "xq", 100)],
        )
        self.assertEqual(plan["geometry_matched_line_count"], 0)
        self.assertEqual(plan["unrecovered_line_ids"], ["7"])

    def test_geometry_pass_cannot_steal_a_backup_line_from_the_text_pass(self):
        plan = self._plan_for(
            [self._line("0", "rama", 0)],
            # both current lines sit near the backup line; the text pass owns one
            [self._line("7", "rama", 0), self._line("8", "", 0)],
        )
        self.assertEqual(plan["matched_line_count"], 1)
        self.assertEqual(plan["matches"][0]["match_method"], "text")
        self.assertEqual(plan["geometry_matched_line_count"], 0)

    def test_pass_one_results_are_unchanged_by_the_geometry_pass(self):
        plan = self._plan_for(
            [self._line("0", "rama", 0), self._line("1", "sita", 100)],
            [self._line("7", "rama", 0), self._line("8", "sita", 100)],
        )
        self.assertEqual(plan["text_matched_line_count"], 2)
        self.assertEqual(plan["geometry_matched_line_count"], 0)
        self.assertTrue(all(m["match_method"] == "text" for m in plan["matches"]))

    def test_geometry_pass_is_skipped_when_lines_have_no_baseline(self):
        backup = [{"id": "0", "text": "rama", "coords": [(0, 0), (200, 0), (200, 20), (0, 20)]}]
        current = [{"id": "7", "text": "xq", "coords": [(0, 0), (200, 0), (200, 20), (0, 20)]}]
        plan = self._plan_for(backup, current)
        self.assertEqual(plan["geometry_matched_line_count"], 0)


if __name__ == "__main__":
    unittest.main()

