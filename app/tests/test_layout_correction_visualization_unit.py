from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from app.visualize_layout_corrections import (
    CORRECT_BGR,
    EXTRA_BGR,
    MISSING_BGR,
    GraphSnapshot,
    _validate_output_path,
    compare_graphs,
    render_correction_overlay,
)


class LayoutCorrectionVisualizationUnitTests(unittest.TestCase):
    def setUp(self):
        self.baseline = GraphSnapshot(
            nodes=((10.0, 10.0), (25.0, 25.0), (40.0, 10.0)),
            edges=frozenset({(0, 1), (0, 2)}),
        )
        self.corrected = GraphSnapshot(
            nodes=((10.02, 10.0), (40.0, 10.0), (60.0, 40.0)),
            edges=frozenset({(0, 1), (1, 2)}),
        )

    def test_compare_graphs_uses_coordinates_after_node_reindexing(self):
        difference = compare_graphs(
            self.baseline,
            self.corrected,
            node_match_tolerance=0.25,
        )

        self.assertEqual(len(difference.correct_nodes), 2)
        self.assertEqual(difference.missing_nodes, ((60.0, 40.0),))
        self.assertEqual(difference.extra_nodes, ((25.0, 25.0),))
        self.assertEqual(len(difference.correct_edges), 1)
        self.assertEqual(len(difference.missing_edges), 1)
        self.assertEqual(len(difference.extra_edges), 1)

    def test_renderer_contains_requested_colors_and_black_outlines(self):
        difference = compare_graphs(self.baseline, self.corrected)
        image = np.full((80, 100, 3), (40, 120, 220), dtype=np.uint8)

        rendered = render_correction_overlay(
            image,
            difference,
            (100.0, 80.0),
            edge_width=2,
            node_radius=5,
            outline_width=1,
        )

        self.assertEqual(rendered.shape, image.shape)
        colors = {
            tuple(int(channel) for channel in pixel)
            for pixel in rendered.reshape(-1, 3)
        }
        self.assertIn(CORRECT_BGR, colors)
        self.assertIn(MISSING_BGR, colors)
        self.assertIn(EXTRA_BGR, colors)
        self.assertGreater(
            np.count_nonzero(np.all(rendered == np.asarray(CORRECT_BGR), axis=2)),
            0,
        )
        untouched_background_pixel = rendered[70, 90]
        self.assertEqual(
            int(untouched_background_pixel[0]),
            int(untouched_background_pixel[1]),
        )
        self.assertEqual(
            int(untouched_background_pixel[1]),
            int(untouched_background_pixel[2]),
        )

    def test_node_radius_must_exceed_edge_width(self):
        difference = compare_graphs(self.baseline, self.corrected)
        with self.assertRaisesRegex(ValueError, "Node radius"):
            render_correction_overlay(
                np.full((80, 100, 3), 255, dtype=np.uint8),
                difference,
                (100.0, 80.0),
                edge_width=4,
                node_radius=4,
                include_legend=False,
            )

    def test_manuscript_output_is_restricted_to_visualizations(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manuscript_root = Path(temp_dir)
            allowed = manuscript_root / "visualizations" / "page.png"
            forbidden = manuscript_root / "images" / "page.png"

            _validate_output_path(allowed, manuscript_root)
            with self.assertRaisesRegex(ValueError, "visualizations"):
                _validate_output_path(forbidden, manuscript_root)


if __name__ == "__main__":
    unittest.main()
