"""Regression tests for complex-script annotation rendering."""

from pathlib import Path
import unittest

from experiments.annotation_viz.visualize_page_annotations import ShapedTextRenderer


MANGAL_FONT = Path(r"C:\Windows\Fonts\mangal.ttf")


@unittest.skipUnless(MANGAL_FONT.is_file(), "Mangal is not installed")
class ShapedTextRendererTests(unittest.TestCase):
    def test_devanagari_conjunct_is_shaped_into_positioned_glyphs(self) -> None:
        text = "क्षेत्र"
        renderer = ShapedTextRenderer(MANGAL_FONT, pixel_size=25)

        glyphs, _ = renderer._glyph_layout(text)
        mask = renderer.render_mask(text)

        self.assertLess(len(glyphs), len(text))
        self.assertIsNotNone(mask.getbbox())


if __name__ == "__main__":
    unittest.main()
