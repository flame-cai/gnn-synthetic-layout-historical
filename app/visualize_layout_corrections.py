"""Render a single, headless overlay of saved production layout corrections.

The initial graph is reconstructed from the immutable ``gnn-dataset`` inputs
with the production GNN.  It is compared with the graph saved under
``layout_analysis_output/gnn-format``.  Manuscript data is read-only; the only
permitted manuscript-local output is an image under ``visualizations/``.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


Point = tuple[float, float]
Edge = tuple[int, int]
DrawnEdge = tuple[Point, Point]

MISSING_BGR = (49, 130, 245)  # RGB #f58231: human-added / prediction-missing
EXTRA_BGR = (216, 99, 67)  # RGB #4363d8: human-deleted / prediction-extra
CORRECT_BGR = (0, 0, 0)
OUTLINE_BGR = (0, 0, 0)

APP_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = APP_ROOT / "pretrained_gnn" / "v2.pt"
DEFAULT_CONFIG_PATH = APP_ROOT / "pretrained_gnn" / "gnn_preprocessing_v2.yaml"


@dataclass(frozen=True)
class GraphSnapshot:
    nodes: tuple[Point, ...]
    edges: frozenset[Edge]


@dataclass(frozen=True)
class GraphDifference:
    correct_nodes: tuple[Point, ...]
    missing_nodes: tuple[Point, ...]
    extra_nodes: tuple[Point, ...]
    correct_edges: tuple[DrawnEdge, ...]
    missing_edges: tuple[DrawnEdge, ...]
    extra_edges: tuple[DrawnEdge, ...]


def _canonical_edge(source: int, target: int) -> Edge:
    return (source, target) if source < target else (target, source)


def load_points(path: str | Path) -> tuple[Point, ...]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Node file not found: {path}")
    if path.stat().st_size == 0:
        return ()

    values = np.loadtxt(path, dtype=float, ndmin=2)
    if values.shape[1] < 2:
        raise ValueError(f"Node file must have at least two columns: {path}")
    if not np.isfinite(values[:, :2]).all():
        raise ValueError(f"Node file contains non-finite coordinates: {path}")
    return tuple((float(row[0]), float(row[1])) for row in values)


def load_edges(path: str | Path, node_count: int) -> frozenset[Edge]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Edge file not found: {path}")
    if path.stat().st_size == 0:
        return frozenset()

    values = np.loadtxt(path, dtype=int, ndmin=2)
    if values.shape[1] < 2:
        raise ValueError(f"Edge file must have at least two columns: {path}")

    edges: set[Edge] = set()
    for row in values:
        source, target = int(row[0]), int(row[1])
        if source == target:
            continue
        if not (0 <= source < node_count and 0 <= target < node_count):
            raise ValueError(
                f"Edge ({source}, {target}) references a node outside "
                f"0..{max(node_count - 1, 0)} in {path}"
            )
        edges.add(_canonical_edge(source, target))
    return frozenset(edges)


def load_dimensions(path: str | Path) -> tuple[float, float]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dimensions file not found: {path}")
    values = np.asarray(np.loadtxt(path, dtype=float)).reshape(-1)
    if values.size < 2 or not np.isfinite(values[:2]).all():
        raise ValueError(f"Invalid graph dimensions: {path}")
    width, height = float(values[0]), float(values[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"Graph dimensions must be positive: {path}")
    return width, height


def match_nodes(
    baseline_nodes: tuple[Point, ...],
    corrected_nodes: tuple[Point, ...],
    tolerance: float,
) -> dict[int, int]:
    """Return a one-to-one baseline-to-corrected mapping by nearby coordinates."""

    if tolerance < 0:
        raise ValueError("Node matching tolerance must be non-negative.")

    tolerance_squared = tolerance * tolerance
    candidates: list[tuple[float, int, int]] = []
    for baseline_index, (baseline_x, baseline_y) in enumerate(baseline_nodes):
        for corrected_index, (corrected_x, corrected_y) in enumerate(corrected_nodes):
            distance_squared = (baseline_x - corrected_x) ** 2 + (
                baseline_y - corrected_y
            ) ** 2
            if distance_squared <= tolerance_squared:
                candidates.append(
                    (distance_squared, baseline_index, corrected_index)
                )

    mapping: dict[int, int] = {}
    used_corrected: set[int] = set()
    for _, baseline_index, corrected_index in sorted(candidates):
        if baseline_index in mapping or corrected_index in used_corrected:
            continue
        mapping[baseline_index] = corrected_index
        used_corrected.add(corrected_index)
    return mapping


def compare_graphs(
    baseline: GraphSnapshot,
    corrected: GraphSnapshot,
    *,
    node_match_tolerance: float = 0.25,
) -> GraphDifference:
    """Classify the net node and edge differences between two graphs."""

    baseline_to_corrected = match_nodes(
        baseline.nodes, corrected.nodes, node_match_tolerance
    )
    corrected_to_baseline = {
        corrected_index: baseline_index
        for baseline_index, corrected_index in baseline_to_corrected.items()
    }

    correct_nodes = tuple(
        corrected.nodes[corrected_index]
        for corrected_index in sorted(corrected_to_baseline)
    )
    missing_nodes = tuple(
        point
        for corrected_index, point in enumerate(corrected.nodes)
        if corrected_index not in corrected_to_baseline
    )
    extra_nodes = tuple(
        point
        for baseline_index, point in enumerate(baseline.nodes)
        if baseline_index not in baseline_to_corrected
    )

    correct_edges: list[DrawnEdge] = []
    missing_edges: list[DrawnEdge] = []
    for source, target in sorted(corrected.edges):
        mapped_source = corrected_to_baseline.get(source)
        mapped_target = corrected_to_baseline.get(target)
        is_correct = (
            mapped_source is not None
            and mapped_target is not None
            and _canonical_edge(mapped_source, mapped_target) in baseline.edges
        )
        drawn_edge = (corrected.nodes[source], corrected.nodes[target])
        (correct_edges if is_correct else missing_edges).append(drawn_edge)

    extra_edges: list[DrawnEdge] = []
    for source, target in sorted(baseline.edges):
        mapped_source = baseline_to_corrected.get(source)
        mapped_target = baseline_to_corrected.get(target)
        remains = (
            mapped_source is not None
            and mapped_target is not None
            and _canonical_edge(mapped_source, mapped_target) in corrected.edges
        )
        if not remains:
            extra_edges.append((baseline.nodes[source], baseline.nodes[target]))

    return GraphDifference(
        correct_nodes=correct_nodes,
        missing_nodes=missing_nodes,
        extra_nodes=extra_nodes,
        correct_edges=tuple(correct_edges),
        missing_edges=tuple(missing_edges),
        extra_edges=tuple(extra_edges),
    )


def _edges_from_payload(edges: Iterable[dict], node_count: int) -> frozenset[Edge]:
    result: set[Edge] = set()
    for edge in edges:
        source, target = int(edge["source"]), int(edge["target"])
        if source == target:
            continue
        if not (0 <= source < node_count and 0 <= target < node_count):
            raise ValueError(
                f"Inferred edge ({source}, {target}) references an invalid node."
            )
        result.add(_canonical_edge(source, target))
    return frozenset(result)


def infer_baseline_edges(
    manuscript_root: str | Path,
    page_id: str,
    *,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> frozenset[Edge]:
    """Re-run the production GNN on raw inputs without touching manuscript data."""

    manuscript_root = Path(manuscript_root)
    raw_root = manuscript_root / "gnn-dataset"
    normalized_path = raw_root / f"{page_id}_inputs_normalized.txt"
    dimensions_path = raw_root / f"{page_id}_dims.txt"
    raw_node_count = len(
        load_points(raw_root / f"{page_id}_inputs_unnormalized.txt")
    )

    for required_path in (normalized_path, dimensions_path, Path(model_path), Path(config_path)):
        if not required_path.is_file():
            raise FileNotFoundError(f"Required inference input not found: {required_path}")

    if str(APP_ROOT) not in sys.path:
        sys.path.insert(0, str(APP_ROOT))
    from gnn_inference import run_gnn_prediction_for_page

    with tempfile.TemporaryDirectory(prefix="layout-correction-baseline-") as temp_dir:
        temporary_root = Path(temp_dir)
        temporary_dataset = temporary_root / "gnn-dataset"
        temporary_dataset.mkdir()
        shutil.copyfile(
            normalized_path,
            temporary_dataset / normalized_path.name,
        )
        shutil.copyfile(
            dimensions_path,
            temporary_dataset / dimensions_path.name,
        )
        payload = run_gnn_prediction_for_page(
            str(temporary_root),
            page_id,
            str(Path(model_path).resolve()),
            str(Path(config_path).resolve()),
        )

    return _edges_from_payload(payload.get("edges", ()), raw_node_count)


def load_graph_comparison(
    manuscript_root: str | Path,
    page_id: str,
    *,
    baseline_edges_path: str | Path | None = None,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> tuple[GraphSnapshot, GraphSnapshot, tuple[float, float]]:
    manuscript_root = Path(manuscript_root)
    raw_root = manuscript_root / "gnn-dataset"
    corrected_root = manuscript_root / "layout_analysis_output" / "gnn-format"

    baseline_nodes = load_points(raw_root / f"{page_id}_inputs_unnormalized.txt")
    corrected_nodes = load_points(
        corrected_root / f"{page_id}_inputs_unnormalized.txt"
    )
    baseline_edges = (
        load_edges(baseline_edges_path, len(baseline_nodes))
        if baseline_edges_path is not None
        else infer_baseline_edges(
            manuscript_root,
            page_id,
            model_path=model_path,
            config_path=config_path,
        )
    )
    corrected_edges = load_edges(
        corrected_root / f"{page_id}_edges.txt", len(corrected_nodes)
    )
    dimensions = load_dimensions(raw_root / f"{page_id}_dims.txt")
    return (
        GraphSnapshot(baseline_nodes, baseline_edges),
        GraphSnapshot(corrected_nodes, corrected_edges),
        dimensions,
    )


def _draw_edge(
    canvas: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
    *,
    width: int,
    outline_width: int,
) -> None:
    cv2.line(
        canvas,
        start,
        end,
        OUTLINE_BGR,
        width + (2 * outline_width),
        cv2.LINE_AA,
    )
    cv2.line(canvas, start, end, color, width, cv2.LINE_AA)


def _draw_node(
    canvas: np.ndarray,
    point: tuple[int, int],
    color: tuple[int, int, int],
    *,
    radius: int,
    outline_width: int,
) -> None:
    cv2.circle(
        canvas,
        point,
        radius + outline_width,
        OUTLINE_BGR,
        -1,
        cv2.LINE_AA,
    )
    cv2.circle(canvas, point, radius, color, -1, cv2.LINE_AA)


def _with_legend(
    image: np.ndarray,
    difference: GraphDifference,
    *,
    edge_width: int,
    node_radius: int,
    outline_width: int,
) -> np.ndarray:
    height, width = image.shape[:2]
    font_scale = max(0.48, min(1.0, width / 2400.0))
    font_thickness = max(1, round(font_scale * 2))
    line_height = max(30, round(44 * font_scale))
    padding = max(12, round(18 * font_scale))
    footer_height = padding * 2 + line_height * 4
    canvas = np.full((height + footer_height, width, 3), 255, dtype=np.uint8)
    canvas[:height] = image

    title_y = height + padding + line_height - 10
    cv2.putText(
        canvas,
        "Initial GNN prediction vs. saved human correction",
        (padding, title_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        CORRECT_BGR,
        font_thickness,
        cv2.LINE_AA,
    )

    entries = (
        (
            CORRECT_BGR,
            f"Correct: {len(difference.correct_nodes)} nodes, "
            f"{len(difference.correct_edges)} edges",
        ),
        (
            MISSING_BGR,
            f"Missing / human-added: {len(difference.missing_nodes)} nodes, "
            f"{len(difference.missing_edges)} edges",
        ),
        (
            EXTRA_BGR,
            f"Extra / human-deleted: {len(difference.extra_nodes)} nodes, "
            f"{len(difference.extra_edges)} edges",
        ),
    )
    marker_start_x = padding
    marker_end_x = padding + max(42, round(62 * font_scale))
    marker_center_x = (marker_start_x + marker_end_x) // 2
    text_x = marker_end_x + padding
    for row, (color, label) in enumerate(entries, start=1):
        center_y = height + padding + line_height * row + line_height // 2
        _draw_edge(
            canvas,
            (marker_start_x, center_y),
            (marker_end_x, center_y),
            color,
            width=edge_width,
            outline_width=outline_width,
        )
        _draw_node(
            canvas,
            (marker_center_x, center_y),
            color,
            radius=node_radius,
            outline_width=outline_width,
        )
        cv2.putText(
            canvas,
            label,
            (text_x, center_y + round(7 * font_scale)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            CORRECT_BGR,
            font_thickness,
            cv2.LINE_AA,
        )
    return canvas


def render_correction_overlay(
    image: np.ndarray,
    difference: GraphDifference,
    graph_dimensions: tuple[float, float],
    *,
    edge_width: int | None = None,
    node_radius: int | None = None,
    outline_width: float | None = None,
    include_legend: bool = False,
) -> np.ndarray:
    """Draw all net graph corrections over a grayscale page image."""

    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a three-channel BGR page image.")

    image_height, image_width = image.shape[:2]
    graph_width, graph_height = graph_dimensions
    if graph_width <= 0 or graph_height <= 0:
        raise ValueError("Graph dimensions must be positive.")

    longest_side = max(image_height, image_width)
    edge_width = (
        max(2, round(longest_side / 1200))
        if edge_width is None
        else int(edge_width)
    )
    node_radius = (
        max(edge_width + 2, round(longest_side / 320))
        if node_radius is None
        else int(node_radius)
    )
    outline_width = (
        max(1, round(longest_side / 3500))
        if outline_width is None
        else max(1, round(float(outline_width)))
    )
    if edge_width < 1:
        raise ValueError("Edge width must be at least 1 pixel.")
    if node_radius <= edge_width:
        raise ValueError("Node radius must be greater than edge width.")
    if outline_width < 1:
        raise ValueError("Outline width must be at least 1 pixel.")

    scale_x = image_width / graph_width
    scale_y = image_height / graph_height

    def to_pixel(point: Point) -> tuple[int, int]:
        return (
            int(round(point[0] * scale_x)),
            int(round(point[1] * scale_y)),
        )

    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    canvas = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)
    for edges, color in (
        (difference.correct_edges, CORRECT_BGR),
        (difference.missing_edges, MISSING_BGR),
        (difference.extra_edges, EXTRA_BGR),
    ):
        for start, end in edges:
            _draw_edge(
                canvas,
                to_pixel(start),
                to_pixel(end),
                color,
                width=edge_width,
                outline_width=outline_width,
            )

    for nodes, color in (
        (difference.correct_nodes, CORRECT_BGR),
        (difference.missing_nodes, MISSING_BGR),
        (difference.extra_nodes, EXTRA_BGR),
    ):
        for point in nodes:
            _draw_node(
                canvas,
                to_pixel(point),
                color,
                radius=node_radius,
                outline_width=outline_width,
            )

    if include_legend:
        canvas = _with_legend(
            canvas,
            difference,
            edge_width=edge_width,
            node_radius=node_radius,
            outline_width=outline_width,
        )
    return canvas


def read_image(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Page image not found: {path}")
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode page image: {path}")
    return image


def write_image(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise ValueError("Output must use a .png, .jpg, .jpeg, or .webp extension.")
    path.parent.mkdir(parents=True, exist_ok=True)
    parameters = [cv2.IMWRITE_JPEG_QUALITY, 95] if suffix in {".jpg", ".jpeg"} else []
    success, encoded = cv2.imencode(suffix, image, parameters)
    if not success:
        raise OSError(f"Could not encode visualization as {suffix}.")
    encoded.tofile(path)


def manuscript_page_from_image(image_path: str | Path) -> tuple[Path, str]:
    image_path = Path(image_path).resolve()
    for candidate in image_path.parents:
        if (candidate / "gnn-dataset").is_dir():
            return candidate, image_path.stem
    raise ValueError(
        "Could not find the manuscript root above the image; expected an "
        "ancestor containing gnn-dataset/."
    )


def _validate_output_path(output_path: Path, manuscript_root: Path) -> None:
    resolved_output = output_path.resolve()
    resolved_root = manuscript_root.resolve()
    try:
        resolved_output.relative_to(resolved_root)
    except ValueError:
        return

    visualization_root = (resolved_root / "visualizations").resolve()
    try:
        resolved_output.relative_to(visualization_root)
    except ValueError as exc:
        raise ValueError(
            "Manuscript-local output is restricted to the visualizations/ "
            "directory so source and correction data cannot be overwritten."
        ) from exc


def build_visualization(
    image_path: str | Path,
    *,
    output_path: str | Path | None = None,
    baseline_edges_path: str | Path | None = None,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    node_match_tolerance: float = 0.25,
    edge_width: int | None = None,
    node_radius: int | None = None,
    outline_width: float | None = None,
    include_legend: bool = False,
) -> tuple[Path, GraphSnapshot, GraphSnapshot, GraphDifference]:
    image_path = Path(image_path).resolve()
    manuscript_root, page_id = manuscript_page_from_image(image_path)
    output_path = (
        Path(output_path).resolve()
        if output_path is not None
        else manuscript_root / "visualizations" / f"{page_id}_layout_corrections.png"
    )
    _validate_output_path(output_path, manuscript_root)

    baseline, corrected, dimensions = load_graph_comparison(
        manuscript_root,
        page_id,
        baseline_edges_path=baseline_edges_path,
        model_path=model_path,
        config_path=config_path,
    )
    difference = compare_graphs(
        baseline,
        corrected,
        node_match_tolerance=node_match_tolerance,
    )
    visualization = render_correction_overlay(
        read_image(image_path),
        difference,
        dimensions,
        edge_width=edge_width,
        node_radius=node_radius,
        outline_width=outline_width,
        include_legend=include_legend,
    )
    write_image(output_path, visualization)
    return output_path, baseline, corrected, difference


def _optional_positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("Value must be a positive integer.")
    return parsed


def _positive_number(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be greater than zero.")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Headlessly overlay the initial GNN graph and the saved human "
            "layout correction on one page image."
        )
    )
    parser.add_argument(
        "image",
        type=Path,
        help="Page image inside a production manuscript (for example images/11.jpg).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Output image. Defaults to "
            "<manuscript>/visualizations/<page>_layout_corrections.png."
        ),
    )
    parser.add_argument(
        "--baseline-edges",
        type=Path,
        help="Optional initial edge file; otherwise the production GNN is re-run.",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--node-match-tolerance",
        type=float,
        default=0.25,
        help="Maximum raw graph-coordinate drift for matching unchanged nodes.",
    )
    parser.add_argument("--edge-width", type=_optional_positive_integer)
    parser.add_argument("--node-radius", type=_optional_positive_integer)
    parser.add_argument("--outline-width", type=_positive_number)
    legend_group = parser.add_mutually_exclusive_group()
    legend_group.add_argument(
        "--legend",
        dest="include_legend",
        action="store_true",
        help="Append the correction-count legend.",
    )
    legend_group.add_argument(
        "--no-legend",
        dest="include_legend",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(include_legend=False)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_path, baseline, corrected, difference = build_visualization(
        args.image,
        output_path=args.output,
        baseline_edges_path=args.baseline_edges,
        model_path=args.model,
        config_path=args.config,
        node_match_tolerance=args.node_match_tolerance,
        edge_width=args.edge_width,
        node_radius=args.node_radius,
        outline_width=args.outline_width,
        include_legend=args.include_legend,
    )
    print(
        f"Baseline graph: {len(baseline.nodes)} nodes, {len(baseline.edges)} edges"
    )
    print(
        f"Corrected graph: {len(corrected.nodes)} nodes, {len(corrected.edges)} edges"
    )
    print(
        "Net corrections: "
        f"missing/human-added={len(difference.missing_nodes)} nodes, "
        f"{len(difference.missing_edges)} edges; "
        f"extra/human-deleted={len(difference.extra_nodes)} nodes, "
        f"{len(difference.extra_edges)} edges"
    )
    print(f"Visualization: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
