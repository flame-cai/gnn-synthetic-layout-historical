"""Opt-in, best-effort visual artifacts for the production page pipeline.

This module is deliberately a sidecar: callers enqueue it after a production
stage has completed, and any visualization failure is logged rather than
changing the stage result.  Artifacts are manuscript-local at
``visualizations/<page>/`` and can safely be deleted without affecting OCR,
PAGE XML, or graph state.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import threading
import unicodedata
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


LOGGER = logging.getLogger(__name__)
SETTINGS_FILENAME = "processing_settings.json"
ROOT_NAME = "visualizations"
_WRITE_LOCK = threading.Lock()
CB_BLUE = (178, 114, 0)       # Okabe-Ito blue #0072B2, in OpenCV BGR
CB_VERMILION = (0, 94, 213)   # Okabe-Ito vermilion #D55E00, in OpenCV BGR
CB_GREY = (90, 90, 90)


def _bool(value, default=False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on", "enabled"}


def default_enabled() -> bool:
    return _bool(os.getenv("APP_PIPELINE_VISUALIZATION"), False)


def is_enabled(manuscript_root: str | Path) -> bool:
    """Return the manuscript opt-in value, falling back to the env default."""
    settings_path = Path(manuscript_root) / SETTINGS_FILENAME
    try:
        payload = json.loads(settings_path.read_text(encoding="utf-8"))
        config = payload.get("pipeline_visualization", {})
        if isinstance(config, Mapping) and "enabled" in config:
            return _bool(config["enabled"])
    except (OSError, ValueError, TypeError):
        pass
    return default_enabled()


def _config(manuscript_root: Path) -> dict:
    try:
        payload = json.loads((manuscript_root / SETTINGS_FILENAME).read_text(encoding="utf-8"))
        config = payload.get("pipeline_visualization", {})
    except (OSError, ValueError, TypeError):
        config = {}
    config = dict(config) if isinstance(config, Mapping) else {}
    return {
        "max_line_previews": max(1, min(int(config.get("max_line_previews", 24)), 100)),
    }


def _root(manuscript_root: Path, page_id: str) -> Path:
    root = manuscript_root / ROOT_NAME / str(page_id)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _manifest_path(root: Path) -> Path:
    return root / "manifest.json"


def _update_manifest(root: Path, page_id: str, **stages) -> None:
    path = _manifest_path(root)
    try:
        payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, ValueError):
        payload = {}
    payload.setdefault("schema_version", 1)
    payload["page_id"] = str(page_id)
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    payload.setdefault("stages", {}).update(stages)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _stage(filename: str, note: str | None = None) -> dict:
    payload = {"status": "available", "file": filename}
    if note:
        payload["note"] = note
    return payload


def _load_points(manuscript_root: Path, page_id: str):
    dataset = manuscript_root / "gnn-dataset"
    points_path = dataset / f"{page_id}_inputs_normalized.txt"
    dims_path = dataset / f"{page_id}_dims.txt"
    if not points_path.exists() or not dims_path.exists():
        return np.empty((0, 3)), (1.0, 1.0)
    try:
        points = np.loadtxt(points_path, ndmin=2)
        if points.size == 0:
            points = np.empty((0, 3))
        dims = np.loadtxt(dims_path, ndmin=1)
        dimensions = (float(dims[0] * 2), float(dims[1] * 2))
        # gnn-dataset positions are normalized by the longest page dimension;
        # all visualization drawing thereafter uses frontend/page pixels.
        points = np.array(points, copy=True)
        points[:, :2] *= max(dimensions)
        return points, dimensions
    except (OSError, ValueError):
        return np.empty((0, 3)), (1.0, 1.0)


def _page_image(manuscript_root: Path, page_id: str):
    image = cv2.imread(str(manuscript_root / "images_resized" / f"{page_id}.jpg"))
    if image is None:
        raise FileNotFoundError(f"No resized image for visualization: {page_id}")
    return image


def _canvas_point(point, dimensions, canvas):
    width, height = dimensions
    return (
        int(round(float(point[0]) * canvas.shape[1] / max(width, 1.0))),
        int(round(float(point[1]) * canvas.shape[0] / max(height, 1.0))),
    )


def _draw_graph(canvas, points, dimensions, edges=(), *, edge_color=(255, 80, 0), node_color=(0, 0, 255)):
    for edge in edges or ():
        try:
            source, target = int(edge["source"]), int(edge["target"])
            if source < 0 or target < 0:
                continue
            cv2.line(canvas, _canvas_point(points[source], dimensions, canvas), _canvas_point(points[target], dimensions, canvas), edge_color, 2, cv2.LINE_AA)
        except (IndexError, KeyError, TypeError, ValueError):
            continue
    for point in points:
        cv2.circle(canvas, _canvas_point(point, dimensions, canvas), 3, node_color, -1, cv2.LINE_AA)
    return canvas


def _preprocessing_features(points: np.ndarray) -> tuple[list[dict], np.ndarray, dict[tuple[int, int], int]]:
    """Return exact candidate connectivity and categorical GNN input values."""
    if len(points) == 0:
        return [], np.empty(0, dtype=int), {}
    try:
        src_root = Path(__file__).resolve().parents[1] / "src"
        if str(src_root) not in sys.path:
            sys.path.insert(0, str(src_root))
        import yaml
        from gnn_training.gnn_data_preparation.config_models import DatasetCreationConfig
        from gnn_training.gnn_data_preparation.graph_constructor import create_input_graph_edges

        config_path = Path(__file__).resolve().parent / "pretrained_gnn" / "gnn_preprocessing_v2.yaml"
        config = DatasetCreationConfig(**yaml.safe_load(config_path.read_text(encoding="utf-8")))
        result = create_input_graph_edges(points, {"width": 1.0, "height": 1.0}, config.input_graph)
        return (
            [{"source": int(u), "target": int(v)} for u, v in result.get("edges", [])],
            np.asarray(result.get("heuristic_degrees", []), dtype=int),
            {tuple(map(int, edge)): int(count) for edge, count in result.get("heuristic_edge_counts", {}).items()},
        )
    except Exception as exc:  # Visualization is intentionally best effort.
        LOGGER.warning("Could not build GNN preprocessing features: %s", exc)
        return [], np.zeros(len(points), dtype=int), {}


_FEATURE_COLORS = [
    (128, 128, 128), (255, 144, 30), (44, 160, 44), (214, 39, 40),
    (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127),
    (188, 189, 34), (23, 190, 207), (0, 0, 0),
]


def _feature_color(value: int) -> tuple[int, int, int]:
    return _FEATURE_COLORS[min(max(int(value), 0), len(_FEATURE_COLORS) - 1)]


def _append_legend(image: np.ndarray, rows: list[tuple[str, tuple[int, int, int]]], *, title="GNN input features") -> np.ndarray:
    width = 280
    panel = cv2.copyMakeBorder(image, 0, 0, 0, width, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.putText(panel, title, (image.shape[1] + 14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
    for index, (label, color) in enumerate(rows):
        y = 55 + index * 20
        cv2.rectangle(panel, (image.shape[1] + 14, y - 11), (image.shape[1] + 28, y + 3), color, -1)
        cv2.putText(panel, label, (image.shape[1] + 38, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    return panel


def _draw_preprocessing(canvas: np.ndarray, points: np.ndarray, dimensions) -> np.ndarray:
    candidates, degrees, overlap_counts = _preprocessing_features(points)
    for edge in candidates:
        source, target = int(edge["source"]), int(edge["target"])
        overlap = min(overlap_counts.get(tuple(sorted((source, target))), 0), 10)
        cv2.line(canvas, _canvas_point(points[source], dimensions, canvas), _canvas_point(points[target], dimensions, canvas), _feature_color(overlap), 1, cv2.LINE_AA)
    for index, point in enumerate(points):
        degree = min(int(degrees[index]) if index < len(degrees) else 0, 10)
        cv2.circle(canvas, _canvas_point(point, dimensions, canvas), 4, _feature_color(degree), -1, cv2.LINE_AA)

    max_degree = min(int(degrees.max()) if len(degrees) else 0, 10)
    max_overlap = min(max(overlap_counts.values(), default=0), 10)
    rows = [(f"Node one-hot: degree {value}", _feature_color(value)) for value in range(max_degree + 1)]
    rows += [(f"Edge one-hot: overlap {value}", _feature_color(value)) for value in range(max_overlap + 1)]
    return _append_legend(canvas, rows)


def _saved_graph(manuscript_root: Path, page_id: str):
    graph_root = manuscript_root / "layout_analysis_output" / "gnn-format"
    points_path = graph_root / f"{page_id}_inputs_unnormalized.txt"
    dims_path = graph_root / f"{page_id}_dims.txt"
    edges_path = graph_root / f"{page_id}_edges.txt"
    if not points_path.exists() or not dims_path.exists():
        return None
    try:
        raw_points = np.loadtxt(points_path, ndmin=2)
        points = np.array(raw_points, copy=True)
        points[:, :2] *= 2.0
        dims = np.loadtxt(dims_path, ndmin=1)
        edges = []
        if edges_path.exists() and edges_path.stat().st_size:
            for source, target, *_ in np.loadtxt(edges_path, dtype=int, ndmin=2):
                edges.append({"source": int(source), "target": int(target)})
        return points, (float(dims[0] * 2), float(dims[1] * 2)), edges
    except (OSError, ValueError, IndexError):
        return None


def _font(size: int):
    candidates = [
        # Mangal is installed with Windows Indic language support and is a
        # reliable Pillow-readable Devanagari font. Nirmala is normally a TTC
        # on Windows, so retain it as a useful fallback rather than assuming a
        # non-existent .ttf file.
        os.path.join(os.environ.get("WINDIR", r"C:\\Windows"), "Fonts", "mangal.ttf"),
        os.path.join(os.environ.get("WINDIR", r"C:\\Windows"), "Fonts", "Nirmala.ttc"),
        os.path.join(os.environ.get("WINDIR", r"C:\\Windows"), "Fonts", "kokila.ttf"),
        "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _line_images(manuscript_root: Path, page_id: str, max_count: int) -> list[Path]:
    root = manuscript_root / "layout_analysis_output" / "image-format" / str(page_id)
    return sorted(root.glob("**/line_*.jpg"))[:max_count] if root.exists() else []


def _copy_processed_line_images(manuscript_root: Path, page_id: str, output_root: Path) -> int:
    """Copy the exact OCR inputs, preserving their text-region hierarchy."""
    source_root = manuscript_root / "layout_analysis_output" / "image-format" / str(page_id)
    destination_root = output_root / "06_processed_line_images"
    if not source_root.exists():
        return 0
    if destination_root.exists():
        shutil.rmtree(destination_root)
    copied = 0
    for source in sorted(source_root.glob("**/line_*.jpg")):
        destination = destination_root / source.relative_to(source_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied += 1
    return copied


def _contact_sheet(paths: list[Path], output_path: Path, *, text_by_line: Mapping[str, str] | None = None) -> bool:
    if not paths:
        return False
    tile_w, tile_h, cols = 280, 120, 3
    rows = (len(paths) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tile_w, rows * tile_h), "white")
    draw = ImageDraw.Draw(sheet)
    label_font, text_font = _font(15), _font(17)
    for index, path in enumerate(paths):
        try:
            with Image.open(path) as source:
                image = source.convert("RGB")
                image.thumbnail((tile_w - 12, 65), Image.Resampling.LANCZOS)
                x, y = (index % cols) * tile_w, (index // cols) * tile_h
                sheet.paste(image, (x + 6, y + 5))
                line_id = path.stem.replace("line_", "")
                draw.text((x + 6, y + 74), f"Line {line_id}", fill="black", font=label_font)
                text = (text_by_line or {}).get(line_id, "")
                if text:
                    draw.text((x + 6, y + 93), text[:34], fill="black", font=text_font)
        except OSError:
            continue
    sheet.save(output_path, "JPEG", quality=92)
    return True


def _pagexml_text_by_line(xml_path: Path) -> dict[str, str]:
    try:
        root = ET.parse(xml_path).getroot()
    except (OSError, ET.ParseError):
        return {}
    result = {}
    for line in root.iter():
        if line.tag.rsplit("}", 1)[-1] != "TextLine":
            continue
        line_id = str(line.get("custom", "")).split("structure_line_id_")[-1]
        if not line_id or line_id == str(line.get("custom", "")):
            line_id = str(line.get("id") or len(result))
        text = ""
        for descendant in line.iter():
            if descendant.tag.rsplit("}", 1)[-1] == "Unicode":
                text = descendant.text or ""
                break
        result[line_id] = text
    return result


def _graphemes(text: str) -> list[str]:
    """Keep Indic combining marks and virama-linked letters in one render cell."""
    clusters: list[str] = []
    for char in str(text or ""):
        extends_previous = (
            bool(clusters)
            and (
                unicodedata.combining(char) > 0
                or char in {"\u200c", "\u200d"}
                or clusters[-1].endswith("\u094d")
            )
        )
        if extends_previous:
            clusters[-1] += char
        else:
            clusters.append(char)
    return clusters


def _align_graphemes(prediction: str, ground_truth: str) -> list[tuple[str | None, str | None, str]]:
    pred, truth = _graphemes(prediction), _graphemes(ground_truth)
    rows, columns = len(pred), len(truth)
    costs = [[0] * (columns + 1) for _ in range(rows + 1)]
    for row in range(rows + 1):
        costs[row][0] = row
    for column in range(columns + 1):
        costs[0][column] = column
    for row in range(1, rows + 1):
        for column in range(1, columns + 1):
            costs[row][column] = min(
                costs[row - 1][column] + 1,
                costs[row][column - 1] + 1,
                costs[row - 1][column - 1] + (pred[row - 1] != truth[column - 1]),
            )
    aligned = []
    row, column = rows, columns
    while row or column:
        if row and column and costs[row][column] == costs[row - 1][column - 1] + (pred[row - 1] != truth[column - 1]):
            aligned.append((pred[row - 1], truth[column - 1], "same" if pred[row - 1] == truth[column - 1] else "substitution"))
            row, column = row - 1, column - 1
        elif row and costs[row][column] == costs[row - 1][column] + 1:
            aligned.append((pred[row - 1], None, "extra"))
            row -= 1
        else:
            aligned.append((None, truth[column - 1], "missing"))
            column -= 1
    return list(reversed(aligned))


def _rgb(bgr: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple(reversed(bgr))


def _render_ocr_alignment(line_id: str, prediction: str, ground_truth: str, crop_path: Path | None, output_path: Path) -> dict:
    aligned = _align_graphemes(prediction, ground_truth)
    font, label_font = _font(28), _font(16)
    cell_width = max(
        26,
        int(max((font.getlength(token or " ") for pair in aligned for token in pair[:2]), default=18)) + 12,
    )
    crop_width, row_height, header_height = 230, 42, 38
    width = crop_width + max(1, len(aligned)) * cell_width + 24
    height = header_height + row_height * 2 + 18
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    if crop_path is not None and crop_path.is_file():
        with Image.open(crop_path) as crop:
            crop = crop.convert("RGB")
            crop.thumbnail((crop_width - 14, height - 14), Image.Resampling.LANCZOS)
            image.paste(crop, (7, (height - crop.height) // 2))
    x0 = crop_width + 8
    draw.text((x0, 4), f"Line {line_id}  |  CED {sum(kind != 'same' for _, _, kind in aligned)} / {max(len(_graphemes(ground_truth)), 1)}", fill="black", font=label_font)
    draw.text((x0 - 4, header_height + 8), "Pred", fill="black", font=label_font)
    draw.text((x0 - 4, header_height + row_height + 8), "GT", fill="black", font=label_font)
    for index, (pred, truth, kind) in enumerate(aligned):
        x = x0 + 34 + index * cell_width
        draw.rectangle((x, header_height + 2, x + cell_width, header_height + row_height - 2), outline=(225, 225, 225))
        draw.rectangle((x, header_height + row_height + 2, x + cell_width, header_height + row_height * 2 - 2), outline=(225, 225, 225))
        pred_color = _rgb(CB_VERMILION) if kind in {"extra", "substitution"} else _rgb(CB_GREY)
        truth_color = _rgb(CB_BLUE) if kind in {"missing", "substitution"} else _rgb(CB_GREY)
        if pred:
            draw.text((x, header_height + 3), pred, fill=pred_color, font=font)
        if truth:
            draw.text((x, header_height + row_height + 3), truth, fill=truth_color, font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, "PNG")
    edit_distance = sum(kind != "same" for _, _, kind in aligned)
    return {"prediction": prediction, "ground_truth": ground_truth, "character_edit_distance": edit_distance, "cer": edit_distance / max(len(_graphemes(ground_truth)), 1)}


def ocr_text_comparison_artifact(manuscript_root: str | Path, page_id: str) -> None:
    """Render one combined, aligned predicted-versus-GT Unicode comparison."""
    manuscript_root = Path(manuscript_root)
    if not is_enabled(manuscript_root):
        return
    root = _root(manuscript_root, page_id)
    prediction_xml = root / "07_ocr_predictions_page.xml"
    ground_truth_xml = root / "08_ocr_ground_truth_page.xml"
    if not prediction_xml.exists() or not ground_truth_xml.exists():
        return
    predictions, truths = _pagexml_text_by_line(prediction_xml), _pagexml_text_by_line(ground_truth_xml)
    line_ids = sorted(
        set(predictions) | set(truths),
        key=lambda value: (0, int(value)) if str(value).isdigit() else (1, str(value)),
    )
    if not line_ids:
        return
    crops = {path.stem.replace("line_", ""): path for path in _line_images(manuscript_root, page_id, _config(manuscript_root)["max_line_previews"])}
    directory = root / "07_08_ocr_text_correction_diff"
    if directory.exists():
        shutil.rmtree(directory)
    records, rendered = {}, []
    for line_id in line_ids:
        path = directory / f"line_{line_id}.png"
        records[line_id] = _render_ocr_alignment(line_id, predictions.get(line_id, ""), truths.get(line_id, ""), crops.get(line_id), path)
        rendered.append(path)
    rows = [Image.open(path).convert("RGB") for path in rendered]
    try:
        legend_height = 34
        overview = Image.new("RGB", (max(680, max(row.width for row in rows)), sum(row.height for row in rows) + legend_height), "white")
        y = 0
        for row in rows:
            overview.paste(row, (0, y))
            y += row.height
        draw = ImageDraw.Draw(overview)
        label_font = _font(15)
        draw.rectangle((8, y + 10, 24, y + 26), fill=_rgb(CB_BLUE))
        draw.text((30, y + 7), "blue: predicted-missing / human-added", fill="black", font=label_font)
        offset = 290
        draw.rectangle((offset, y + 10, offset + 16, y + 26), fill=_rgb(CB_VERMILION))
        draw.text((offset + 22, y + 7), "vermilion: predicted-extra / human-deleted", fill="black", font=label_font)
        overview.save(root / "07_08_ocr_text_correction_diff.jpg", "JPEG", quality=94)
    finally:
        for row in rows:
            row.close()
    (root / "07_08_ocr_text_correction_diff.json").write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    with _WRITE_LOCK:
        stage = _stage("07_08_ocr_text_correction_diff.jpg", "Aligned Unicode comparison: blue is predicted-missing/human-added; vermilion is predicted-extra/human-deleted.")
        stage["directory"] = "07_08_ocr_text_correction_diff"
        stage["metrics"] = "Character edit distance and CER are recorded per line."
        _update_manifest(root, page_id, ocr_text_comparison=stage)


def read_mode_pagexml_artifact(manuscript_root: str | Path, page_id: str, xml_path: str | Path, *, ground_truth: bool) -> None:
    """Snapshot PAGE XML immediately before/after a committed Read Mode save."""
    manuscript_root, xml_path = Path(manuscript_root), Path(xml_path)
    if not is_enabled(manuscript_root) or not xml_path.is_file():
        return
    filename = "08_ocr_ground_truth_page.xml" if ground_truth else "07_ocr_predictions_page.xml"
    stage_key = "ocr_ground_truth_pagexml" if ground_truth else "ocr_predictions_pagexml"
    # Copy first: this must preserve the pre-update XML even if another
    # diagnostic worker is currently assembling image artifacts.
    root = _root(manuscript_root, page_id)
    shutil.copy2(xml_path, root / filename)
    with _WRITE_LOCK:
        _update_manifest(root, page_id, **{stage_key: _stage(filename, "PAGE XML snapshot from the committed Read Mode save.")})
    if ground_truth:
        ocr_text_comparison_artifact(manuscript_root, page_id)


def upload_artifacts(manuscript_root: str | Path, page_id: str) -> None:
    """Create only stages 1-3 after upload; graph differences need human edits."""
    manuscript_root = Path(manuscript_root)
    if not is_enabled(manuscript_root):
        return
    with _WRITE_LOCK:
        root = _root(manuscript_root, page_id)
        _update_manifest(
            root, page_id,
            layout_graph_diff={"status": "pending", "note": "Created after a human layout correction is saved."},
            processed_line_images={"status": "pending", "note": "Created after a layout save."},
            ocr_text_comparison={"status": "pending", "note": "Created after a committed Read Mode save captures predicted and corrected PAGE XML."},
        )
        image = _page_image(manuscript_root, page_id)
        cv2.imwrite(str(root / "01_original_image.jpg"), image)
        _update_manifest(root, page_id, original_image=_stage("01_original_image.jpg"))

        heatmap = cv2.imread(str(manuscript_root / "heatmaps" / f"{page_id}.jpg"))
        if heatmap is not None:
            heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(str(root / "02_craft_heatmap.jpg"), heatmap)
            _update_manifest(root, page_id, craft_heatmap=_stage("02_craft_heatmap.jpg"))

        points, dimensions = _load_points(manuscript_root, page_id)
        preprocessing = _draw_preprocessing(image.copy(), points, dimensions)
        cv2.imwrite(str(root / "03_gnn_preprocessing.jpg"), preprocessing)
        _update_manifest(root, page_id, gnn_preprocessing=_stage("03_gnn_preprocessing.jpg", "Node colors are heuristic-degree one-hot values; edge colors are heuristic-overlap one-hot values across KNN candidates."))


def _safe_graph(graph: Mapping | None) -> tuple[list[dict], set[tuple[int, int]]]:
    graph = graph if isinstance(graph, Mapping) else {}
    nodes = [dict(node) for node in graph.get("nodes", []) if isinstance(node, Mapping)]
    edges = set()
    for edge in graph.get("edges", []):
        try:
            source, target = sorted((int(edge["source"]), int(edge["target"])))
            if source >= 0 and target >= 0 and source != target and source < len(nodes) and target < len(nodes):
                edges.add((source, target))
        except (KeyError, TypeError, ValueError):
            continue
    return nodes, edges


def _match_nodes(baseline_nodes: list[dict], corrected_nodes: list[dict], tolerance=2.0) -> dict[int, int]:
    matches, unmatched = {}, set(range(len(corrected_nodes)))
    for before_index, before in enumerate(baseline_nodes):
        try:
            bx, by = float(before["x"]), float(before["y"])
        except (KeyError, TypeError, ValueError):
            continue
        nearest = min(
            (
                (np.hypot(float(corrected_nodes[index]["x"]) - bx, float(corrected_nodes[index]["y"]) - by), index)
                for index in unmatched
            ),
            default=(float("inf"), None),
        )
        if nearest[1] is not None and nearest[0] <= tolerance:
            matches[before_index] = nearest[1]
            unmatched.remove(nearest[1])
    return matches


def _dashed_line(canvas, start, end, color, thickness=2, outline_color=(0, 0, 0), outline_thickness=1):
    start, end = np.array(start, dtype=float), np.array(end, dtype=float)
    length = np.linalg.norm(end - start)
    if length == 0:
        return
    direction = (end - start) / length
    for distance in np.arange(0, length, 10):
        pt1 = tuple(np.int32(start + direction * distance))
        pt2 = tuple(np.int32(start + direction * min(distance + 5, length)))
        if outline_thickness > 0:
            cv2.line(canvas, pt1, pt2, outline_color, thickness + outline_thickness * 2, cv2.LINE_AA)
        cv2.line(canvas, pt1, pt2, color, thickness, cv2.LINE_AA)


def _draw_diff_graph(image: np.ndarray, baseline_graph: Mapping, corrected_graph: Mapping, *, corrected_view: bool) -> tuple[np.ndarray, bool]:
    baseline_nodes, baseline_edges = _safe_graph(baseline_graph)
    corrected_nodes, corrected_edges = _safe_graph(corrected_graph)
    mapping = _match_nodes(baseline_nodes, corrected_nodes)
    reverse_mapping = {after: before for before, after in mapping.items()}
    removed_nodes = set(range(len(baseline_nodes))) - set(mapping)
    added_nodes = set(range(len(corrected_nodes))) - set(reverse_mapping)
    mapped_baseline_edges = {
        edge: tuple(sorted((mapping[edge[0]], mapping[edge[1]])))
        for edge in baseline_edges if edge[0] in mapping and edge[1] in mapping
    }
    deleted_edges = {edge for edge, mapped in mapped_baseline_edges.items() if mapped not in corrected_edges}
    added_edges = {
        edge for edge in corrected_edges
        if edge[0] in added_nodes or edge[1] in added_nodes or tuple(sorted((reverse_mapping[edge[0]], reverse_mapping[edge[1]]))) not in baseline_edges
    }
    changed = bool(removed_nodes or added_nodes or deleted_edges or added_edges)
    if not changed:
        return image, False

    canvas = image.copy()
    to_point = lambda node: (int(round(float(node["x"]))), int(round(float(node["y"]))))
    
    # New Colors (in OpenCV BGR)
    COLOR_MISSING = (49, 130, 245)  # Orange (#f58231)
    COLOR_EXTRA = (216, 99, 67)     # Blue (#4363d8)
    COLOR_CORRECT = (0, 0, 0)       # Black
    COLOR_OUTLINE = (0, 0, 0)       # Black

    def draw_edge(pt1, pt2, color, thickness, is_dashed=False):
        if is_dashed:
            _dashed_line(canvas, pt1, pt2, color, thickness, outline_color=COLOR_OUTLINE, outline_thickness=1)
        else:
            cv2.line(canvas, pt1, pt2, COLOR_OUTLINE, thickness + 2, cv2.LINE_AA) # Outline
            cv2.line(canvas, pt1, pt2, color, thickness, cv2.LINE_AA)             # Inner color

    # Edges are now bolder (base thickness 3 instead of 2)
    if corrected_view:
        for source, target in corrected_edges - added_edges:
            draw_edge(to_point(corrected_nodes[source]), to_point(corrected_nodes[target]), COLOR_CORRECT, 3)
        for source, target in added_edges:
            draw_edge(to_point(corrected_nodes[source]), to_point(corrected_nodes[target]), COLOR_MISSING, 3)
        for source, target in deleted_edges:
            draw_edge(to_point(baseline_nodes[source]), to_point(baseline_nodes[target]), COLOR_EXTRA, 3, is_dashed=True)
    else:
        for source, target in baseline_edges - deleted_edges:
            draw_edge(to_point(baseline_nodes[source]), to_point(baseline_nodes[target]), COLOR_CORRECT, 3)
        for source, target in deleted_edges:
            draw_edge(to_point(baseline_nodes[source]), to_point(baseline_nodes[target]), COLOR_EXTRA, 3)
        for source, target in added_edges:
            draw_edge(to_point(corrected_nodes[source]), to_point(corrected_nodes[target]), COLOR_MISSING, 3, is_dashed=True)

    def draw_node(pt, color, radius):
        cv2.circle(canvas, pt, radius + 1, COLOR_OUTLINE, -1, cv2.LINE_AA) # Outline circle
        cv2.circle(canvas, pt, radius, color, -1, cv2.LINE_AA)           # Filled center

    for index, node in enumerate(corrected_nodes):
        if index in added_nodes:
            draw_node(to_point(node), COLOR_MISSING, 4)
        else:
            draw_node(to_point(node), COLOR_CORRECT, 4)
    for index in removed_nodes:
        draw_node(to_point(baseline_nodes[index]), COLOR_EXTRA, 4)

    return canvas, True

def _layout_diff_comparison(predicted: np.ndarray, corrected: np.ndarray) -> np.ndarray:
    header = 36
    left = cv2.copyMakeBorder(predicted, header, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    right = cv2.copyMakeBorder(corrected, header, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.putText(left, "Predicted graph", (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(right, "Human-corrected graph", (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    comparison = np.hstack([left, right])
    
    # Legend colors in OpenCV BGR
    COLOR_MISSING = (49, 130, 245)
    COLOR_EXTRA = (216, 99, 67)
    COLOR_CORRECT = (0, 0, 0)
    
    return _append_legend(
        comparison,
        [
            ("Unchanged node/edge", COLOR_CORRECT),
            ("Missing (human-added / predicted-missing)", COLOR_MISSING),
            ("Extra (human-deleted / predicted-extra)", COLOR_EXTRA),
        ],
        title="Layout correction diff",
    )

def layout_artifacts(
    manuscript_root: str | Path,
    page_id: str,
    baseline_graph: Mapping | None = None,
    corrected_graph: Mapping | None = None,
) -> None:
    """Create layout/OCR artifacts; graph panels are emitted only for real diffs."""
    manuscript_root = Path(manuscript_root)
    if not is_enabled(manuscript_root):
        return
    with _WRITE_LOCK:
        root = _root(manuscript_root, page_id)
        image = _page_image(manuscript_root, page_id)
        if baseline_graph is not None and corrected_graph is not None:
            predicted, has_diff = _draw_diff_graph(image, baseline_graph, corrected_graph, corrected_view=False)
            corrected, _ = _draw_diff_graph(image, baseline_graph, corrected_graph, corrected_view=True)
            if has_diff:
                cv2.imwrite(str(root / "04_05_layout_graph_correction_diff.jpg"), _layout_diff_comparison(predicted, corrected))
                _update_manifest(
                    root,
                    page_id,
                    layout_graph_diff=_stage("04_05_layout_graph_correction_diff.jpg", "Comparison: orange (#f58231) is human-added/predicted-missing; blue (#4363d8) is human-deleted/predicted-extra."),
                )

        crops = _line_images(manuscript_root, page_id, _config(manuscript_root)["max_line_previews"])
        copied_crop_count = _copy_processed_line_images(manuscript_root, page_id, root)
        if _contact_sheet(crops, root / "06_processed_line_images.jpg"):
            stage = _stage("06_processed_line_images.jpg", "Exact app OCR crops, including unwrapped lines where configured.")
            stage["directory"] = "06_processed_line_images"
            stage["line_image_count"] = copied_crop_count
            _update_manifest(root, page_id, processed_line_images=stage)


def ocr_prediction_artifacts(manuscript_root: str | Path, page_id: str, predictions: Mapping[str, str] | None) -> None:
    """Record that OCR completed; the comparison is frozen at Read Mode save."""
    manuscript_root = Path(manuscript_root)
    if not is_enabled(manuscript_root):
        return
    with _WRITE_LOCK:
        root = _root(manuscript_root, page_id)
        _update_manifest(
            root,
            page_id,
            ocr_text_comparison={
                "status": "pending",
                "note": "OCR completed. The aligned prediction-versus-human-text comparison is created on the next committed Read Mode save.",
            },
        )


def enqueue(action: Callable, *args, **kwargs) -> None:
    """Run best-effort visualization work after the response-producing stage."""
    def runner():
        try:
            action(*args, **kwargs)
        except Exception:
            LOGGER.exception("Pipeline visualization failed without affecting production processing.")
    threading.Thread(target=runner, daemon=True, name="pipeline-visualization").start()
