"""Render Sarvam JSON layout blocks over the source manuscript image."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
IMAGE_PATH = ROOT / "app" / "input_manuscripts" / "dense" / "images" / "17.jpg"
JSON_PATH = EXPERIMENT_DIR / "output" / "extracted" / "metadata" / "page_001.json"
OUTPUT_PATH = EXPERIMENT_DIR / "output" / "sarvam_blocks_overlay.png"

COLORS = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
    (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
]


def main() -> None:
    result = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    image = Image.open(IMAGE_PATH).convert("RGBA")
    if (result["image_width"], result["image_height"]) != image.size:
        raise ValueError("Sarvam JSON dimensions do not match the source image.")

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()
    for index, block in enumerate(result["blocks"], start=1):
        coords = block["coordinates"]
        box = (coords["x1"], coords["y1"], coords["x2"], coords["y2"])
        color = COLORS[(index - 1) % len(COLORS)]
        draw.rectangle(box, fill=(*color, 45), outline=(*color, 255), width=5)
        label = f"{index}: {block['layout_tag']}"
        label_box = draw.textbbox((box[0], box[1]), label, font=font)
        draw.rectangle((label_box[0] - 3, label_box[1] - 3, label_box[2] + 3, label_box[3] + 3), fill=(*color, 235))
        draw.text((box[0], box[1]), label, fill=(0, 0, 0, 255), font=font)

    Image.alpha_composite(image, overlay).convert("RGB").save(OUTPUT_PATH, quality=95)
    summary = {
        "coordinate_space": {"width": result["image_width"], "height": result["image_height"]},
        "blocks": len(result["blocks"]),
        "line_level_boxes": False,
        "reason": "Each returned box is a layout block; several blocks contain multiple newline-delimited text lines.",
    }
    (EXPERIMENT_DIR / "output" / "bbox_assessment.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
