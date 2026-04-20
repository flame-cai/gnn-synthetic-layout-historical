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

from telemetry import compute_layout_edit_metrics, compute_text_edit_metrics


class RecognitionTelemetryUnitTest(unittest.TestCase):
    def test_compute_text_edit_metrics_tracks_total_distance_and_changed_lines(self):
        metrics = compute_text_edit_metrics(
            predicted_lines={"1": "abc", "2": "ram"},
            saved_lines={"1": "adc", "2": "rama", "3": "new"},
        )

        self.assertEqual(metrics["changed_line_count"], 3)
        self.assertEqual(metrics["total_edit_distance"], 5)
        self.assertAlmostEqual(metrics["normalized_edit_distance"], 5 / 10)
        self.assertEqual([row["line_id"] for row in metrics["per_line_diffs"]], ["1", "2", "3"])

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


if __name__ == "__main__":
    unittest.main()
