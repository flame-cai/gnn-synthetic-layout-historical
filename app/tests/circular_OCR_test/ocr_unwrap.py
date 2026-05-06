from __future__ import annotations

import json
import math
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from recognition.pagexml_line_dataset import PreparedLineRecord, PreparedPageDataset, _encode_like_app_jpg, _load_processing_image

from .geometry import PageXmlLine, normalize_open_curve, parse_pagexml_lines


@dataclass(frozen=True)
class OrientationCandidate:
    name: str
    source_path: str


def generate_orientation_candidates(source_path: str) -> list[OrientationCandidate]:
    return [
        OrientationCandidate("forward", source_path),
        OrientationCandidate("reversed", source_path),
        OrientationCandidate("forward_vertical_flip", source_path),
        OrientationCandidate("reversed_vertical_flip", source_path),
    ]


def choose_orientation_by_confidence(candidate_scores: list[dict], ground_truth_text: str | None = None) -> dict:
    if not candidate_scores:
        raise ValueError("At least one orientation candidate score is required.")
    selected = max(
        candidate_scores,
        key=lambda item: (float(item.get("confidence_score", 0.0) or 0.0), str(item.get("candidate_name", ""))),
    )
    selected_name = selected["candidate_name"]
    return {
        "selected_candidate_name": selected_name,
        "selected_candidate_path": selected.get("candidate_image_path"),
        "selected_prediction": selected.get("predicted_text", ""),
        "selected_confidence_score": float(selected.get("confidence_score", 0.0) or 0.0),
        "rejected_candidate_names": [
            item.get("candidate_name") for item in candidate_scores if item.get("candidate_name") != selected_name
        ],
        "selector_reason": "max_confidence",
        "used_ground_truth_text": False,
    }


def _point_to_segment_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    px_val, py_val = point
    sx_val, sy_val = start
    ex_val, ey_val = end
    dx_val = ex_val - sx_val
    dy_val = ey_val - sy_val
    length_sq = dx_val * dx_val + dy_val * dy_val
    if length_sq <= 1e-6:
        return math.hypot(px_val - sx_val, py_val - sy_val)
    ratio = max(0.0, min(1.0, ((px_val - sx_val) * dx_val + (py_val - sy_val) * dy_val) / length_sq))
    proj_x = sx_val + ratio * dx_val
    proj_y = sy_val + ratio * dy_val
    return math.hypot(px_val - proj_x, py_val - proj_y)


def _point_to_curve_distance(point: tuple[float, float], curve_points: list[tuple[float, float]]) -> float:
    if not curve_points:
        return 0.0
    if len(curve_points) == 1:
        return math.hypot(point[0] - curve_points[0][0], point[1] - curve_points[0][1])
    return min(
        _point_to_segment_distance(point, start, end)
        for start, end in zip(curve_points, curve_points[1:])
    )


def derive_half_height_from_page_coords(
    page_space_coords: list[tuple[int, int]] | list[tuple[float, float]],
    baseline_points: list[tuple[int, int]] | list[tuple[float, float]],
    default_half_height: int,
) -> int:
    if not page_space_coords or not baseline_points:
        return int(default_half_height)
    curve = normalize_open_curve(baseline_points)
    distances = [
        _point_to_curve_distance((float(x_val), float(y_val)), curve.points)
        for x_val, y_val in page_space_coords
    ]
    if not distances:
        return int(default_half_height)
    return max(4, int(math.ceil(max(distances))))


def _numeric_suffix(value: str, fallback: int) -> int:
    digits = re.findall(r"\d+", value or "")
    return int(digits[-1]) if digits else fallback


def prepare_unwrapped_line_record(
    page_id: str,
    line_id: str,
    line_custom: str,
    text: str,
    page_space_coords: list[tuple[int, int]] | list[tuple[float, float]],
    baseline_points: list[tuple[int, int]] | list[tuple[float, float]],
    selected_crop_rel_path: str,
    candidate_metadata: list[dict],
    region_id: str = "region_0",
    region_custom: str = "textbox_label_0",
    app_image_rel_path: str | None = None,
) -> PreparedLineRecord:
    polygon_points = [[int(round(x_val)), int(round(y_val))] for x_val, y_val in page_space_coords]
    if polygon_points:
        x_min = min(point[0] for point in polygon_points)
        y_center = sum(point[1] for point in polygon_points) / len(polygon_points)
    else:
        x_min = 0.0
        y_center = 0.0
    return PreparedLineRecord(
        page_id=page_id,
        region_id=region_id,
        region_custom=region_custom,
        line_id=line_id,
        line_custom=line_custom,
        line_numeric_id=_numeric_suffix(line_custom, 0),
        text=text,
        polygon_points=polygon_points,
        y_center=float(y_center),
        x_min=float(x_min),
        app_image_rel_path=app_image_rel_path,
        flat_image_rel_path=selected_crop_rel_path,
    )


def _curve_samples(points: list[tuple[float, float]], sample_step_px: float) -> tuple[np.ndarray, np.ndarray]:
    if len(points) < 2:
        point = points[0] if points else (0.0, 0.0)
        return np.array([point], dtype=np.float32), np.array([[1.0, 0.0]], dtype=np.float32)
    distances = [0.0]
    for start, end in zip(points, points[1:]):
        distances.append(distances[-1] + math.hypot(end[0] - start[0], end[1] - start[1]))
    total = max(distances[-1], 1.0)
    sample_count = max(2, int(math.ceil(total / max(sample_step_px, 1.0))) + 1)
    targets = np.linspace(0.0, total, sample_count)
    samples = []
    tangents = []
    segment_index = 0
    for target in targets:
        while segment_index < len(distances) - 2 and distances[segment_index + 1] < target:
            segment_index += 1
        start = points[segment_index]
        end = points[segment_index + 1]
        segment_length = max(distances[segment_index + 1] - distances[segment_index], 1e-6)
        ratio = (target - distances[segment_index]) / segment_length
        x_val = start[0] + (end[0] - start[0]) * ratio
        y_val = start[1] + (end[1] - start[1]) * ratio
        dx_val = (end[0] - start[0]) / segment_length
        dy_val = (end[1] - start[1]) / segment_length
        samples.append((x_val, y_val))
        tangents.append((dx_val, dy_val))
    return np.array(samples, dtype=np.float32), np.array(tangents, dtype=np.float32)


def _unwrap_line_image(
    processing_image: np.ndarray,
    baseline_points: list[tuple[int, int]],
    half_height_px: int,
    sample_step_px: float,
    min_width_px: int,
) -> np.ndarray:
    curve = normalize_open_curve(baseline_points)
    samples, tangents = _curve_samples(curve.points, sample_step_px)
    width = max(min_width_px, len(samples))
    height = max(2, int(half_height_px) * 2)
    if len(samples) != width:
        x_source = np.linspace(0, len(samples) - 1, width)
        samples = np.column_stack(
            [
                np.interp(x_source, np.arange(len(samples)), samples[:, 0]),
                np.interp(x_source, np.arange(len(samples)), samples[:, 1]),
            ]
        ).astype(np.float32)
        tangents = np.column_stack(
            [
                np.interp(x_source, np.arange(len(tangents)), tangents[:, 0]),
                np.interp(x_source, np.arange(len(tangents)), tangents[:, 1]),
            ]
        ).astype(np.float32)
    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]]).astype(np.float32)
    offsets = np.linspace(-half_height_px, half_height_px, height, dtype=np.float32)
    map_x = samples[:, 0][None, :] + offsets[:, None] * normals[:, 0][None, :]
    map_y = samples[:, 1][None, :] + offsets[:, None] * normals[:, 1][None, :]
    page_median = int(np.median(processing_image))
    return cv2.remap(
        processing_image,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=page_median,
    )


def _candidate_image(base_image: np.ndarray, candidate_name: str) -> np.ndarray:
    image = base_image
    if candidate_name in {"reversed", "reversed_vertical_flip"}:
        image = cv2.flip(image, 1)
    if candidate_name in {"forward_vertical_flip", "reversed_vertical_flip"}:
        image = cv2.flip(image, 0)
    return image


def _write_candidate_images(
    base_image: np.ndarray,
    candidates_root: Path,
    line_index: int,
    line: PageXmlLine,
) -> list[dict]:
    rows = []
    for candidate in generate_orientation_candidates(f"line_{line_index:04d}.png"):
        candidate_rel_path = Path("candidates") / line.line_custom / f"{candidate.name}.png"
        candidate_path = candidates_root.parent / candidate_rel_path
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(candidate_path), _candidate_image(base_image, candidate.name))
        rows.append(
            {
                "page_id": line.page_id,
                "line_id": line.line_id,
                "line_custom": line.line_custom,
                "candidate_name": candidate.name,
                "candidate_image_path": str(candidate_path.resolve()),
                "predicted_text": "",
                "confidence_score": 0.0,
            }
        )
    return rows


def _select_without_model(candidate_rows: list[dict]) -> dict:
    rows = []
    for row in candidate_rows:
        score = 1.0 if row["candidate_name"] == "forward" else 0.0
        rows.append({**row, "confidence_score": score})
    return choose_orientation_by_confidence(rows)


def _rewrite_selected_crop_files(
    output_root: Path,
    record: PreparedLineRecord,
    selected_candidate_path: str | Path,
) -> None:
    selected_image = cv2.imread(str(selected_candidate_path), cv2.IMREAD_GRAYSCALE)
    if selected_image is None:
        raise ValueError(f"Could not read selected orientation candidate: {selected_candidate_path}")

    jpg_bytes, decoded_jpg = _encode_like_app_jpg(selected_image)
    if record.app_image_rel_path:
        app_abs_path = output_root / record.app_image_rel_path
        app_abs_path.parent.mkdir(parents=True, exist_ok=True)
        app_abs_path.write_bytes(jpg_bytes)
    if record.flat_image_rel_path:
        flat_abs_path = output_root / "finetune_dataset" / record.flat_image_rel_path
        flat_abs_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(flat_abs_path), decoded_jpg)


def _apply_orientation_predictions_to_page(
    prepared_page: PreparedPageDataset,
    predictions_by_path: dict[str, dict],
    checkpoint_path: str | Path,
    width_policy: str = "batch_max_pad",
) -> PreparedPageDataset:
    output_root = Path(prepared_page.output_root)
    manifest_path = Path(prepared_page.manifest_path)
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    records_by_line_custom = {record.line_custom: record for record in prepared_page.records}
    updated_orientation_metadata = []
    for line_metadata in manifest_payload.get("orientation_metadata", []):
        candidate_rows = []
        for candidate in line_metadata.get("candidates", []):
            prediction = predictions_by_path.get(str(Path(candidate["candidate_image_path"]).resolve()), {})
            candidate_rows.append(
                {
                    **candidate,
                    "predicted_text": prediction.get("predicted_label", ""),
                    "confidence_score": float(prediction.get("confidence_score", 0.0) or 0.0),
                    "resized_width": int(prediction.get("resized_width", 0) or 0),
                    "pad_fraction": float(prediction.get("pad_fraction", 0.0) or 0.0),
                }
            )

        selection = choose_orientation_by_confidence(candidate_rows)
        record = records_by_line_custom.get(line_metadata.get("line_custom", ""))
        if record is not None and selection.get("selected_candidate_path"):
            _rewrite_selected_crop_files(output_root, record, selection["selected_candidate_path"])

        updated_orientation_metadata.append(
            {
                **line_metadata,
                "selected_candidate_name": selection["selected_candidate_name"],
                "selection": {
                    **selection,
                    "checkpoint_path": str(Path(checkpoint_path).resolve()),
                    "width_policy": width_policy,
                },
                "candidates": candidate_rows,
            }
        )

    manifest_payload["orientation_metadata"] = updated_orientation_metadata
    manifest_payload["orientation_selector"] = {
        "selector": "max_confidence",
        "uses_ground_truth_text": False,
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "width_policy": width_policy,
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return prepared_page


def apply_ocr_confidence_orientation_selection(
    prepared_page: PreparedPageDataset,
    checkpoint_path: str | Path,
    width_policy: str = "batch_max_pad",
) -> PreparedPageDataset:
    """Score one page's orientation candidates and rewrite selected crop files."""

    return apply_ocr_confidence_orientation_selection_to_pages(
        [prepared_page],
        checkpoint_path,
        width_policy=width_policy,
    )[0]


def apply_ocr_confidence_orientation_selection_to_pages(
    prepared_pages: list[PreparedPageDataset],
    checkpoint_path: str | Path,
    width_policy: str = "batch_max_pad",
) -> list[PreparedPageDataset]:
    """Score prepared orientation candidates and rewrite selected crop files.

    PAGE-XML coords and baseline geometry are not changed here. This function
    only updates OCR-ready crop artifacts and the page manifest metadata.
    """

    from recognition.ocr_defaults import get_device, load_inference_model, run_line_image_inference_from_loaded_model

    pages = list(prepared_pages)
    candidate_roots = [
        Path(prepared_page.output_root) / "candidates"
        for prepared_page in pages
        if (Path(prepared_page.output_root) / "candidates").exists()
    ]
    if not candidate_roots:
        return pages

    model, converter, opt, device = load_inference_model(
        checkpoint_path,
        device=get_device(),
        width_policy=width_policy,
        workers=0,
    )
    predictions_by_path: dict[str, dict] = {}
    for candidates_root in candidate_roots:
        for prediction in run_line_image_inference_from_loaded_model(candidates_root, model, converter, opt, device):
            predictions_by_path[str(Path(prediction["image_path"]).resolve())] = prediction

    return [
        _apply_orientation_predictions_to_page(
            prepared_page,
            predictions_by_path,
            checkpoint_path,
            width_policy=width_policy,
        )
        for prepared_page in pages
    ]


def prepare_circular_page_line_dataset(
    xml_path: str | Path,
    image_path: str | Path,
    output_root: str | Path,
    unwrapping_config: dict,
    orientation_scores_by_line: dict[str, list[dict]] | None = None,
) -> PreparedPageDataset:
    xml_path = Path(xml_path)
    image_path = Path(image_path)
    output_root = Path(output_root)
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    lines = parse_pagexml_lines(xml_path)
    processing_image = _load_processing_image(image_path)
    image_format_root = output_root / "image-format" / xml_path.stem
    finetune_dataset_root = output_root / "finetune_dataset"
    test_dir = finetune_dataset_root / "test"
    test_dir.mkdir(parents=True, exist_ok=True)
    candidates_root = output_root / "candidates"
    half_height_px = int(unwrapping_config.get("half_height_px", 42))
    sample_step_px = float(unwrapping_config.get("sample_step_px", 2))
    min_width_px = int(unwrapping_config.get("min_width_px", 12))

    gt_lines = []
    prepared_records = []
    orientation_metadata = []
    for index, line in enumerate(lines, start=1):
        line_half_height_px = derive_half_height_from_page_coords(
            line.coords_points,
            line.baseline_points,
            default_half_height=half_height_px,
        )
        raw_unwrapped = _unwrap_line_image(
            processing_image,
            line.baseline_points,
            half_height_px=line_half_height_px,
            sample_step_px=sample_step_px,
            min_width_px=min_width_px,
        )
        candidate_rows = _write_candidate_images(raw_unwrapped, candidates_root, index, line)
        scores = orientation_scores_by_line.get(line.line_custom) if orientation_scores_by_line else None
        selection = choose_orientation_by_confidence(scores) if scores else _select_without_model(candidate_rows)
        selected_candidate = selection["selected_candidate_name"]
        selected_image = _candidate_image(raw_unwrapped, selected_candidate)
        jpg_bytes, decoded_jpg = _encode_like_app_jpg(selected_image)

        app_rel_path = Path("image-format") / line.page_id / line.region_custom / f"line_{_numeric_suffix(line.line_custom, index)}.jpg"
        app_abs_path = output_root / app_rel_path
        app_abs_path.parent.mkdir(parents=True, exist_ok=True)
        app_abs_path.write_bytes(jpg_bytes)

        flat_rel_path = Path("test") / f"word_{index:04d}.png"
        flat_abs_path = finetune_dataset_root / flat_rel_path
        cv2.imwrite(str(flat_abs_path), decoded_jpg)
        gt_lines.append(f"{flat_rel_path.as_posix()}\t{line.text}")
        record = prepare_unwrapped_line_record(
            page_id=line.page_id,
            line_id=line.line_id,
            line_custom=line.line_custom,
            text=line.text,
            page_space_coords=line.coords_points,
            baseline_points=line.baseline_points,
            selected_crop_rel_path=flat_rel_path.as_posix(),
            candidate_metadata=candidate_rows,
            region_id=line.region_id,
            region_custom=line.region_custom,
            app_image_rel_path=app_rel_path.as_posix(),
        )
        prepared_records.append(record)
        orientation_metadata.append(
            {
                "page_id": line.page_id,
                "line_id": line.line_id,
                "line_custom": line.line_custom,
                "selected_candidate_name": selected_candidate,
                "selected_crop_rel_path": flat_rel_path.as_posix(),
                "selection": selection,
                "candidates": candidate_rows,
                "line_half_height_px": line_half_height_px,
                "page_space_coords": line.coords_points,
                "baseline_points": line.baseline_points,
            }
        )

    gt_path = finetune_dataset_root / "gt.txt"
    gt_path.write_text("\n".join(gt_lines) + ("\n" if gt_lines else ""), encoding="utf-8")
    manifest_path = output_root / "manifest.json"
    manifest_payload = {
        "page_id": xml_path.stem,
        "image_filename": image_path.name,
        "source_xml_path": str(xml_path.resolve()),
        "source_image_path": str(image_path.resolve()),
        "unwrapping_config": unwrapping_config,
        "orientation_metadata": orientation_metadata,
        "records": [asdict(record) for record in prepared_records],
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return PreparedPageDataset(
        page_id=xml_path.stem,
        image_filename=image_path.name,
        source_xml_path=str(xml_path.resolve()),
        source_image_path=str(image_path.resolve()),
        output_root=str(output_root.resolve()),
        image_format_dir=str(image_format_root.resolve()),
        finetune_dataset_dir=str(finetune_dataset_root.resolve()),
        gt_path=str(gt_path.resolve()),
        manifest_path=str(manifest_path.resolve()),
        records=prepared_records,
    )
