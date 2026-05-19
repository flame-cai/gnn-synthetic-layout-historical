from __future__ import annotations

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

from recognition.line_segmentation.geometry import normalize_baseline_topology


class BaselineNormalizationUnitTest(unittest.TestCase):
    def _assert_points_equal(self, actual, expected):
        self.assertEqual(
            [[round(point[0], 3), round(point[1], 3)] for point in actual],
            [[float(point[0]), float(point[1])] for point in expected],
        )

    def test_short_retraced_tail_is_trimmed(self):
        topology = normalize_baseline_topology(
            [(756, 816), (806, 814), (856, 818), (806, 814)],
        )

        self._assert_points_equal(
            topology.normalized_points,
            [(756, 816), (806, 814), (856, 818)],
        )
        self.assertTrue(topology.was_out_and_back)
        self.assertTrue(topology.short_tail_trimmed)
        self.assertEqual(topology.out_and_back_detection, "short_tail")
        self.assertIn("short_tail_trimmed", topology.normalization_actions)
        self.assertFalse(topology.is_closed)
        self.assertEqual(topology.line_kind, "horizontal_straight")

    def test_longer_short_retraced_tail_is_trimmed(self):
        topology = normalize_baseline_topology(
            [(10, 20), (30, 20), (50, 20), (70, 20), (50, 20), (30, 20)],
        )

        self._assert_points_equal(topology.normalized_points, [(10, 20), (30, 20), (50, 20), (70, 20)])
        self.assertTrue(topology.short_tail_trimmed)
        self.assertEqual(topology.mirror_pair_count, 2)
        self.assertEqual(topology.dominant_axis, "horizontal")

    def test_branched_repeat_does_not_become_fake_closed_loop(self):
        topology = normalize_baseline_topology(
            [(10, 10), (40, 10), (25, 35), (10, 10), (25, 35), (40, 10)],
        )

        self.assertFalse(topology.short_tail_trimmed)
        self.assertFalse(topology.is_closed)
        self.assertNotEqual(topology.line_kind, "closed_circular")
        self.assertNotEqual(topology.normalized_points, [[10.0, 10.0], [40.0, 10.0], [25.0, 35.0], [10.0, 10.0]])

    def test_straight_repeated_walk_is_sorted_along_dominant_axis(self):
        topology = normalize_baseline_topology(
            [(50, 20), (10, 20), (30, 20), (10, 20), (50, 20)],
        )

        self._assert_points_equal(topology.normalized_points, [(10, 20), (30, 20), (50, 20)])
        self.assertTrue(topology.dominant_axis_deduped)
        self.assertEqual(topology.dominant_axis, "horizontal")
        self.assertEqual(topology.repeated_near_point_count, 2)
        self.assertEqual(topology.line_kind, "horizontal_straight")

    def test_closed_circular_path_keeps_closed_topology_and_stable_seam(self):
        topology = normalize_baseline_topology(
            [(48, 16), (78, 48), (48, 80), (18, 48), (48, 16), (18, 48), (48, 80), (78, 48)],
        )

        self.assertTrue(topology.was_out_and_back)
        self.assertFalse(topology.short_tail_trimmed)
        self.assertTrue(topology.is_closed)
        self.assertEqual(topology.line_kind, "closed_circular")
        self.assertEqual(topology.normalized_points[0], topology.normalized_points[-1])
        self.assertEqual(topology.normalized_points[0], [48.0, 16.0])
        self.assertIn("cut_at_top", topology.normalization_actions)


if __name__ == "__main__":
    unittest.main()
