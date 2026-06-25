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

from telemetry import (
    compute_layout_edit_metrics,
    compute_reading_direction_edit_metrics,
    compute_text_edit_metrics,
    compute_text_region_edit_metrics,
)


class RecognitionTelemetryUnitTest(unittest.TestCase):
    def test_compute_text_edit_metrics_tracks_total_distance_and_changed_lines(self):
        metrics = compute_text_edit_metrics(
            predicted_lines={"1": "abc", "2": "ram"},
            saved_lines={"1": "adc", "2": "rama", "3": "new"},
        )

        self.assertEqual(metrics["changed_line_count"], 3)
        self.assertEqual(metrics["total_edit_distance"], 5)
        self.assertAlmostEqual(metrics["normalized_edit_distance"], 5 / 10)
        self.assertAlmostEqual(metrics["page_cer"], 5 / 10)
        self.assertEqual([row["line_id"] for row in metrics["per_line_diffs"]], ["1", "2", "3"])
        self.assertEqual(metrics["per_line_diffs"][1]["line_cer"], 1 / 4)

    def test_compute_layout_edit_metrics_counts_nodes_and_edges(self):
        metrics = compute_layout_edit_metrics(
            [
                {"type": "node_add"},
                {"type": "node_delete"},
                {"type": "add"},
                {"type": "delete"},
                {"type": "reset_heuristic"},
            ]
        )

        self.assertEqual(metrics["nodes_added"], 1)
        self.assertEqual(metrics["nodes_deleted"], 1)
        self.assertEqual(metrics["edges_added"], 1)
        self.assertEqual(metrics["edges_deleted"], 1)
        self.assertEqual(metrics["reset_heuristic_count"], 1)
        self.assertEqual(metrics["modification_count"], 5)
        self.assertEqual(metrics["original_nodes"], 0)
        self.assertEqual(metrics["final_nodes"], 0)
        self.assertEqual(metrics["original_edges"], 0)
        self.assertEqual(metrics["final_edges"], 0)

    def test_compute_text_region_edit_metrics_compares_against_prior_labels(self):
        graph_payload = {
            "nodes": [{}, {}, {}, {}],
            "edges": [{"source": 0, "target": 1}, {"source": 2, "target": 3}],
        }

        metrics = compute_text_region_edit_metrics(
            graph_payload,
            textbox_labels=[5, 5, 6, 6],
            previous_textbox_labels=[5, 5, 7, 7],
        )

        self.assertEqual(metrics["text_line_count"], 2)
        self.assertEqual(metrics["current_region_count"], 2)
        self.assertEqual(metrics["text_region_annotations_changed"], 1)
        self.assertEqual(metrics["changed_text_lines"][0]["line_id"], "1")

    def test_compute_reading_direction_edit_metrics_counts_added_changed_deleted(self):
        previous = {
            "lineAnnotations": [
                {"annotation_id": "0", "reading_direction": [1, 0], "cut_midpoint": [1, 1]},
                {"annotation_id": "1", "reading_direction": [0, 1], "cut_midpoint": [2, 2]},
            ]
        }
        current = [
            {"annotation_id": "0", "reading_direction": [1, 0], "cut_midpoint": [1, 1]},
            {"annotation_id": "2", "reading_direction": [0, -1], "cut_midpoint": [3, 3]},
        ]

        metrics = compute_reading_direction_edit_metrics(current, previous)

        self.assertEqual(metrics["reading_direction_annotation_count"], 2)
        self.assertEqual(metrics["reading_direction_annotations_added"], 1)
        self.assertEqual(metrics["reading_direction_annotations_deleted"], 1)
        self.assertEqual(metrics["reading_direction_annotations_changed"], 0)

    def test_compute_layout_edit_metrics_includes_annotation_deltas(self):
        graph_payload = {
            "nodes": [{}, {}, {}, {}],
            "edges": [{"source": 0, "target": 1}, {"source": 2, "target": 3}],
        }

        metrics = compute_layout_edit_metrics(
            [{"type": "node_add"}, {"type": "reading_direction"}],
            graph_payload=graph_payload,
            textbox_labels=[0, 0, 0, 0],
            previous_textbox_labels=[0, 0, 1, 1],
            reading_direction_annotations=[{"annotation_id": "0", "reading_direction": [1, 0]}],
            previous_reading_direction_annotations=[],
            include_annotation_deltas=True,
        )

        self.assertEqual(metrics["original_nodes"], 3)
        self.assertEqual(metrics["final_nodes"], 4)
        self.assertEqual(metrics["original_edges"], 2)
        self.assertEqual(metrics["final_edges"], 2)
        self.assertEqual(metrics["nodes_added"], 1)
        self.assertEqual(metrics["reading_direction_modification_count"], 1)
        self.assertEqual(metrics["text_region_annotations_changed"], 1)
        self.assertEqual(metrics["reading_direction_annotations_added"], 1)
        self.assertEqual(metrics["total_layout_interventions"], 3)


if __name__ == "__main__":
    unittest.main()
