import json
import os
import shutil
import sys
import unittest
from pathlib import Path
from unittest import mock


TESTS_ROOT = Path(__file__).resolve().parent
APP_ROOT = TESTS_ROOT.parent
REPO_ROOT = APP_ROOT.parent

if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from layout_effort_logging import record_layout_effort_save


class LayoutEffortLoggingUnitTest(unittest.TestCase):
    def setUp(self):
        self.tmp_root = TESTS_ROOT / "_tmp_layout_effort_logging_unit"
        self.manuscript_root = self.tmp_root / "manuscript_a"
        self.manuscript_root.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmp_root, ignore_errors=True)

    def _log_payload(self):
        return json.loads(
            (self.manuscript_root / "layout_analysis_output" / "layout_effort.json").read_text(
                encoding="utf-8"
            )
        )

    def test_page_effort_accumulates_edits_but_keeps_latest_processing_time(self):
        record_layout_effort_save(
            manuscript_root=self.manuscript_root,
            page_id="page_1",
            save_scope="layout",
            save_intent="commit",
            active_learning_revision={"revision_number": 1},
            layout_metrics={
                "original_nodes": 10,
                "final_nodes": 12,
                "nodes_added": 2,
                "nodes_deleted": 0,
                "original_edges": 8,
                "final_edges": 9,
                "edges_added": 1,
                "edges_deleted": 0,
                "text_region_annotations_changed": 3,
                "reading_direction_metrics": {
                    "reading_direction_annotation_count": 1,
                    "reading_direction_annotations_added": 1,
                },
            },
            layout_effort={
                "edit_count": 2,
                "active_edit_time_ms": 5000,
                "left_click_add_node_edits": 1,
                "key_hold_edit_count": 1,
                "key_hold_edit_counts": {"a": 1},
                "key_hold_duration_ms": {"a": 1200},
            },
            processing_metrics={"duration_seconds": 4.0, "status": "success", "line_count": 5},
        )

        record_layout_effort_save(
            manuscript_root=self.manuscript_root,
            page_id="page_1",
            save_scope="layout",
            save_intent="commit",
            active_learning_revision={"revision_number": 2},
            layout_metrics={
                "original_nodes": 12,
                "final_nodes": 11,
                "nodes_added": 0,
                "nodes_deleted": 1,
                "original_edges": 9,
                "final_edges": 7,
                "edges_added": 0,
                "edges_deleted": 2,
                "text_region_annotations_changed": 1,
                "reading_direction_metrics": {
                    "reading_direction_annotation_count": 2,
                    "reading_direction_annotations_changed": 1,
                },
            },
            layout_effort={
                "edit_count": 1,
                "active_edit_time_ms": 0,
                "key_hold_edit_count": 1,
                "key_hold_edit_counts": {"d": 1},
                "key_hold_duration_ms": {"d": 900},
            },
            processing_metrics={"duration_seconds": 1.5, "status": "success", "line_count": 6},
        )

        payload = self._log_payload()
        page = payload["pages"]["page_1"]
        self.assertEqual(page["revision_count"], 2)
        self.assertEqual(page["original_nodes"], 10)
        self.assertEqual(page["final_nodes"], 11)
        self.assertEqual(page["original_edges"], 8)
        self.assertEqual(page["final_edges"], 7)
        self.assertEqual(page["totals"]["edit_count"], 3)
        self.assertEqual(page["totals"]["active_edit_time_seconds"], 7.0)
        self.assertEqual(page["totals"]["nodes_added"], 2)
        self.assertEqual(page["totals"]["nodes_deleted"], 1)
        self.assertEqual(page["totals"]["edges_added"], 1)
        self.assertEqual(page["totals"]["edges_deleted"], 2)
        self.assertEqual(page["totals"]["text_lines_region_labeled"], 4)
        self.assertEqual(page["totals"]["manual_text_line_orientations_labeled_current"], 2)
        self.assertEqual(page["totals"]["manual_text_line_orientation_annotation_edits"], 2)
        self.assertEqual(page["totals"]["latest_processing_time_seconds"], 1.5)

        summary = payload["manuscript_summary"]
        self.assertEqual(summary["page_count"], 1)
        self.assertEqual(summary["layout_revision_count"], 2)
        self.assertEqual(summary["edit_count"], 3)
        self.assertEqual(summary["active_edit_time_seconds"], 7.0)
        self.assertEqual(summary["latest_layout_processing_time_seconds"], 1.5)

    def test_environment_flag_can_disable_layout_effort_logging(self):
        with mock.patch.dict(os.environ, {"LAYOUT_EFFORT_LOGGING_ENABLED": "false"}):
            result = record_layout_effort_save(
                manuscript_root=self.manuscript_root,
                page_id="page_1",
                save_scope="layout",
                save_intent="commit",
                layout_metrics={},
                layout_effort={"edit_count": 1},
                processing_metrics={"duration_seconds": 1.0},
            )

        self.assertIsNone(result)
        self.assertFalse((self.manuscript_root / "layout_analysis_output" / "layout_effort.json").exists())


if __name__ == "__main__":
    unittest.main()
