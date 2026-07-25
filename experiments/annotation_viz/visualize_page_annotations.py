"""Render cropped views of the saved page-10 annotations.

The script reads only the already-saved layout-analysis output.  It creates
four equal-size bottom-left crops: text-line labels, text-region labels,
PAGE-XML Unicode transcriptions, and the unannotated manuscript image.
"""

from __future__ import annotations

import argparse
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE = REPOSITORY_ROOT / "app" / "input_manuscripts" / "dense" / "images" / "10.jpg"
DEFAULT_GRAPH_ROOT = (
    REPOSITORY_ROOT
    / "app"
    / "input_manuscripts"
    / "dense"
    / "layout_analysis_output"
    / "gnn-format"
)
DEFAULT_XML = (
    REPOSITORY_ROOT
    / "app"
    / "input_manuscripts"
    / "dense"
    / "layout_analysis_output"
    / "page-xml-format"
    / "10.xml"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DEFAULT_FONT = Path(r"C:\Windows\Fonts\mangal.ttf")

# A deliberately portrait crop anchored to the page's lower-left corner.
CROP_WIDTH = 900
CROP_HEIGHT = 1_200

BLACK_BGR = (0, 0, 0)

# User-specified colour-blind-friendly palette, stored as BGR for OpenCV.
# White and black remain reserved for annotation panels and graph outlines.
ANNOTATION_BGR = (
    (75, 25, 230),   # #e6194B
    (75, 180, 60),   # #3cb44b
    (25, 225, 255),  # #ffe119
    (216, 99, 67),   # #4363d8
    (49, 130, 245),  # #f58231
    (244, 212, 66),  # #42d4f4
    (230, 50, 240),  # #f032e6
    (212, 190, 250), # #fabed4
    (144, 153, 70),  # #469990
    (255, 190, 220), # #dcbeff
    (36, 99, 154),   # #9A6324
    (200, 250, 255), # #fffac8
    (0, 0, 128),     # #800000
    (195, 255, 170), # #aaffc3
    (117, 0, 0),     # #000075
    (169, 169, 169), # #a9a9a9
)


def read_image(path: Path) -> np.ndarray:
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode image: {path}")
    return image


def read_vector(path: Path) -> np.ndarray:
    values = np.loadtxt(path, dtype=int, ndmin=1)
    return np.asarray(values, dtype=int).reshape(-1)


def read_points(path: Path) -> np.ndarray:
    points = np.loadtxt(path, dtype=float, ndmin=2)
    if points.shape[1] < 2:
        raise ValueError(f"Expected x/y coordinates in {path}")
    return points[:, :2]


def read_edges(path: Path) -> np.ndarray:
    edges = np.loadtxt(path, dtype=int, ndmin=2)
    if edges.shape[1] < 2:
        raise ValueError(f"Expected source/target columns in {path}")
    return edges[:, :2]


def read_dimensions(path: Path) -> tuple[float, float]:
    dimensions = np.loadtxt(path, dtype=float).reshape(-1)
    if len(dimensions) < 2 or min(dimensions[:2]) <= 0:
        raise ValueError(f"Expected positive graph dimensions in {path}")
    return float(dimensions[0]), float(dimensions[1])


def label_color(label: int) -> tuple[int, int, int]:
    """Return a stable palette colour for a saved annotation ID."""

    return ANNOTATION_BGR[label % len(ANNOTATION_BGR)]


def draw_edge(
    canvas: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
    *,
    width: int,
    outline_width: int,
) -> None:
    cv2.line(canvas, start, end, BLACK_BGR, width + 2 * outline_width, cv2.LINE_AA)
    cv2.line(canvas, start, end, color, width, cv2.LINE_AA)


def draw_node(
    canvas: np.ndarray,
    point: tuple[int, int],
    color: tuple[int, int, int],
    *,
    radius: int,
    outline_width: int,
) -> None:
    cv2.circle(canvas, point, radius + outline_width, BLACK_BGR, -1, cv2.LINE_AA)
    cv2.circle(canvas, point, radius, color, -1, cv2.LINE_AA)


def crop_bounds(image: np.ndarray) -> tuple[int, int, int, int]:
    height, width = image.shape[:2]
    if width < CROP_WIDTH or height < CROP_HEIGHT:
        raise ValueError(f"Image is smaller than the requested {CROP_WIDTH}x{CROP_HEIGHT} crop")
    return 0, height - CROP_HEIGHT, CROP_WIDTH, height


def render_graph(
    image: np.ndarray,
    points: np.ndarray,
    edges: np.ndarray,
    labels: np.ndarray,
    *,
    edge_width: int,
    node_radius: int,
    outline_width: int,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    if len(points) != len(labels):
        raise ValueError("Each graph node must have one annotation label")

    canvas = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    for source, target in edges:
        if not (0 <= source < len(points) and 0 <= target < len(points)):
            raise ValueError(f"Invalid edge ({source}, {target})")
        # Same-line edges naturally receive their shared line/region colour.
        color = label_color(int(labels[source]))
        draw_edge(
            canvas,
            tuple(np.rint(points[source]).astype(int)),
            tuple(np.rint(points[target]).astype(int)),
            color,
            width=edge_width,
            outline_width=outline_width,
        )
    for point, label in zip(points, labels, strict=True):
        draw_node(
            canvas,
            tuple(np.rint(point).astype(int)),
            label_color(int(label)),
            radius=node_radius,
            outline_width=outline_width,
        )
    x0, y0, x1, y1 = crop_bounds(canvas)
    return canvas[y0:y1, x0:x1].copy(), (x0, y0, x1, y1)


def unicode_text_by_line(xml_path: Path) -> dict[int, str]:
    root = ET.parse(xml_path).getroot()
    result: dict[int, str] = {}
    for line in root.iter():
        if line.tag.rsplit("}", 1)[-1] != "TextLine":
            continue
        match = re.search(r"structure_line_id_(\d+)", line.get("custom", ""))
        if match is None:
            continue
        text = next((child.text or "" for child in line.iter() if child.tag.rsplit("}", 1)[-1] == "Unicode"), "")
        if text.strip():
            result[int(match.group(1))] = text.strip()
    return result


def truncate_to_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, width: int) -> str:
    if draw.textbbox((0, 0), text, font=font)[2] <= width:
        return text
    ellipsis = "…"
    shortened = text
    while shortened and draw.textbbox((0, 0), shortened + ellipsis, font=font)[2] > width:
        shortened = shortened[:-1]
    return shortened + ellipsis if shortened else ellipsis


def render_unicode_overlay(
    image: np.ndarray,
    points: np.ndarray,
    line_labels: np.ndarray,
    texts: dict[int, str],
    crop_box: tuple[int, int, int, int],
) -> np.ndarray:
    x0, y0, x1, y1 = crop_box
    crop = grayscale_bgr(image[y0:y1, x0:x1])
    pil_image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    font = ImageFont.truetype(str(DEFAULT_FONT), size=25)
    for line_id, text in texts.items():
        member_points = points[line_labels == line_id]
        if not len(member_points):
            continue
        visible = member_points[
            (member_points[:, 0] >= x0)
            & (member_points[:, 0] < x0 + crop.shape[1])
            & (member_points[:, 1] >= y0)
            & (member_points[:, 1] < y0 + crop.shape[0])
        ]
        if not len(visible):
            continue
        # Sanskrit lines are read left-to-right, so anchor each transcription
        # at the leftmost graph node visible in the crop.
        local_x = max(8, round(np.min(visible[:, 0]) - x0) + 8)
        local_y = round(np.mean(visible[:, 1]) - y0)
        rendered = truncate_to_width(draw, text, font, crop.shape[1] - local_x - 8)
        # A pale panel lets the Unicode remain legible over both manuscript and graph.
        box = draw.textbbox((local_x, local_y - 34), rendered, font=font, stroke_width=1)
        draw.rectangle((box[0] - 4, box[1] - 2, box[2] + 4, box[3] + 2), fill=(255, 255, 255, 215))
        draw.text((local_x, local_y - 34), rendered, font=font, fill=(0, 0, 0), stroke_width=1, stroke_fill=(255, 255, 255))
    return cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise OSError(f"Could not encode {path}")
    encoded.tofile(path)


def grayscale_bgr(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--graph-root", type=Path, default=DEFAULT_GRAPH_ROOT)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--edge-width", type=int, default=5)
    parser.add_argument("--node-radius", type=int, default=12)
    parser.add_argument("--outline-width", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.edge_width, args.node_radius, args.outline_width) < 1:
        raise ValueError("Graph styling values must be positive")
    image = read_image(args.image)
    points = read_points(args.graph_root / "10_inputs_unnormalized.txt")
    edges = read_edges(args.graph_root / "10_edges.txt")
    graph_width, graph_height = read_dimensions(args.graph_root / "10_dims.txt")
    image_height, image_width = image.shape[:2]
    points *= (image_width / graph_width, image_height / graph_height)
    line_labels = read_vector(args.graph_root / "10_labels_textline.txt")
    region_labels = read_vector(args.graph_root / "10_labels_textbox.txt")
    style = dict(edge_width=args.edge_width, node_radius=args.node_radius, outline_width=args.outline_width)

    line_crop, crop_box = render_graph(image, points, edges, line_labels, **style)
    region_crop, _ = render_graph(image, points, edges, region_labels, **style)
    unicode_crop = render_unicode_overlay(image, points, line_labels, unicode_text_by_line(args.xml), crop_box)
    x0, y0, x1, y1 = crop_box
    plain_crop = grayscale_bgr(image[y0:y1, x0:x1])

    write_image(args.output_dir / "page_10_text_lines.png", line_crop)
    write_image(args.output_dir / "page_10_text_regions.png", region_crop)
    write_image(args.output_dir / "page_10_unicode_text.png", unicode_crop)
    write_image(args.output_dir / "page_10_no_annotations.png", plain_crop)
    print(f"Crop: x={crop_box[0]}..{crop_box[2]}, y={crop_box[1]}..{crop_box[3]}")
    print(f"Wrote: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
