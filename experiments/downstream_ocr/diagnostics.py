from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .metrics import compute_iou_matrix, match_objects, polygons_to_mask
from .pagexml import PageXmlPage, iter_polygon_parts


def _draw_geometry_outline(canvas: np.ndarray, geometry, color: tuple[int, int, int], thickness: int) -> None:
    for polygon in iter_polygon_parts(geometry):
        points = np.asarray(
            [[int(round(x)), int(round(y))] for x, y in polygon.exterior.coords[:-1]],
            dtype=np.int32,
        )
        cv2.polylines(canvas, [points], isClosed=True, color=color, thickness=thickness)


def write_polygon_diagnostic(
    *,
    gt_page: PageXmlPage,
    pred_page: PageXmlPage,
    output_path: str | Path,
    iou_matrix: np.ndarray | None = None,
    threshold: float = 0.50,
) -> Path:
    canvas = np.full((gt_page.height, gt_page.width, 3), 255, dtype=np.uint8)
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    if iou_matrix is not None:
        for gt_idx, pred_idx, _ in match_objects(iou_matrix, threshold):
            matched_gt.add(gt_idx)
            matched_pred.add(pred_idx)

    for idx, line in enumerate(gt_page.lines):
        color = (0, 160, 0) if idx in matched_gt else (0, 0, 255)
        _draw_geometry_outline(canvas, line.polygon, color, thickness=2)

    for idx, line in enumerate(pred_page.lines):
        color = (200, 0, 0) if idx in matched_pred else (255, 128, 0)
        _draw_geometry_outline(canvas, line.polygon, color, thickness=1)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), canvas)
    return output


def write_pixel_mask_diagnostic(
    *,
    gt_page: PageXmlPage,
    pred_page: PageXmlPage,
    output_path: str | Path,
) -> Path:
    gt_mask = polygons_to_mask((line.polygon for line in gt_page.lines), gt_page.width, gt_page.height)
    pred_mask = polygons_to_mask((line.polygon for line in pred_page.lines), gt_page.width, gt_page.height)
    canvas = np.full((gt_page.height, gt_page.width, 3), 255, dtype=np.uint8)
    true_positive = np.logical_and(gt_mask, pred_mask)
    false_negative = np.logical_and(gt_mask, ~pred_mask)
    false_positive = np.logical_and(~gt_mask, pred_mask)
    canvas[true_positive] = (80, 180, 80)
    canvas[false_negative] = (40, 40, 220)
    canvas[false_positive] = (230, 150, 40)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), canvas)
    return output


def write_page_diagnostics(
    *,
    gt_page: PageXmlPage,
    pred_page: PageXmlPage,
    output_dir: str | Path,
    threshold: float = 0.50,
) -> dict[str, str]:
    output_root = Path(output_dir)
    iou_matrix = compute_iou_matrix(list(gt_page.lines), list(pred_page.lines))
    polygon_path = write_polygon_diagnostic(
        gt_page=gt_page,
        pred_page=pred_page,
        output_path=output_root / f"{gt_page.page_id}_polygons_iou_{int(threshold * 100):02d}.png",
        iou_matrix=iou_matrix,
        threshold=threshold,
    )
    mask_path = write_pixel_mask_diagnostic(
        gt_page=gt_page,
        pred_page=pred_page,
        output_path=output_root / f"{gt_page.page_id}_pixel_mask_overlap.png",
    )
    return {
        "polygon_diagnostic_path": str(polygon_path.resolve()),
        "pixel_mask_diagnostic_path": str(mask_path.resolve()),
    }
