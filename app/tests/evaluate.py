import argparse
import json
import os
import glob
import sys
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path

try:
    from shapely.geometry import Polygon
except ImportError:
    print("\n[CRITICAL ERROR] The 'shapely' library is required for Polygon IoU.")
    print("Please install it using: pip install shapely\n")
    sys.exit(1)


def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def parse_polygon_string(points_str):
    try:
        points = points_str.strip().split()
        if len(points) < 3:
            return None

        coords = []
        for point in points:
            x, y = map(int, point.split(","))
            coords.append((x, y))

        poly = Polygon(coords)
        if not poly.is_valid:
            poly = poly.buffer(0)
        return poly
    except Exception:
        return None


def parse_point_string(points_str):
    try:
        points = []
        for point in points_str.strip().split():
            x, y = map(int, point.split(","))
            points.append((x, y))
        return points
    except Exception:
        return []


def line_order_key_from_baseline(points):
    if not points:
        return (float("inf"), float("inf"))

    y_values = [point[1] for point in points]
    x_values = [point[0] for point in points]
    return ((min(y_values) + max(y_values)) / 2.0, min(x_values))


def calculate_iou_polygon(poly1, poly2):
    if poly1 is None or poly2 is None:
        return 0.0

    try:
        inter_area = poly1.intersection(poly2).area
        if inter_area == 0:
            return 0.0

        union_area = poly1.union(poly2).area
        if union_area == 0:
            return 0.0

        return inter_area / union_area
    except Exception:
        return 0.0


def parse_pagexml(filepath):
    extracted_lines = []
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()

        def strip_ns(tag):
            return tag.split("}", 1)[-1]

        for textline in root.iter():
            if strip_ns(textline.tag) != "TextLine":
                continue

            coords = None
            baseline = None
            text = None
            for child in textline:
                tag = strip_ns(child.tag)
                if tag == "Coords":
                    coords = child.get("points")
                elif tag == "Baseline":
                    baseline = child.get("points")
                elif tag == "TextEquiv":
                    for sub in child:
                        if strip_ns(sub.tag) == "Unicode":
                            text = sub.text

            if text and baseline:
                clean_text = unicodedata.normalize("NFC", text).strip()
                baseline_points = parse_point_string(baseline)
                poly_obj = parse_polygon_string(coords) if coords else None

                if clean_text and baseline_points:
                    order_y, order_x = line_order_key_from_baseline(baseline_points)
                    extracted_lines.append(
                        {
                            "text": clean_text,
                            "poly": poly_obj,
                            "order_y": order_y,
                            "order_x": order_x,
                        }
                    )
    except Exception as exc:
        print(f"[Error] Failed to parse XML {filepath}: {exc}")

    return extracted_lines


def calculate_page_level_stats(gt_objs, pred_objs):
    gt_sorted = sorted(gt_objs, key=lambda item: (item["order_y"], item["order_x"]))
    pred_sorted = sorted(pred_objs, key=lambda item: (item["order_y"], item["order_x"]))

    gt_blob = "\n".join(item["text"] for item in gt_sorted)
    pred_blob = "\n".join(item["text"] for item in pred_sorted)

    dist = levenshtein_distance(pred_blob, gt_blob)
    return dist, len(gt_blob)


def pair_lines_by_polygon_iou(gt_objs, pred_objs):
    potential_matches = []

    for gt_idx, gt in enumerate(gt_objs):
        for pred_idx, pred in enumerate(pred_objs):
            iou = calculate_iou_polygon(gt["poly"], pred["poly"])
            if iou > 0:
                potential_matches.append({"g_idx": gt_idx, "p_idx": pred_idx, "iou": iou})

    potential_matches.sort(key=lambda item: item["iou"], reverse=True)

    final_matches = []
    matched_gt = set()
    matched_pred = set()

    for match in potential_matches:
        if match["g_idx"] not in matched_gt and match["p_idx"] not in matched_pred:
            final_matches.append(match)
            matched_gt.add(match["g_idx"])
            matched_pred.add(match["p_idx"])

    unmatched_gt = set(range(len(gt_objs))) - matched_gt
    unmatched_pred = set(range(len(pred_objs))) - matched_pred
    return final_matches, unmatched_gt, unmatched_pred


def calculate_line_level_cost(matches, unmatched_gt, unmatched_pred, gt_objs, pred_objs, iou_threshold):
    total_dist = 0

    for match in matches:
        gt_text = gt_objs[match["g_idx"]]["text"]
        pred_text = pred_objs[match["p_idx"]]["text"]

        if match["iou"] >= iou_threshold:
            total_dist += levenshtein_distance(pred_text, gt_text)
        else:
            total_dist += len(gt_text)
            total_dist += len(pred_text)

    for idx in unmatched_gt:
        total_dist += len(gt_objs[idx]["text"])

    for idx in unmatched_pred:
        total_dist += len(pred_objs[idx]["text"])

    return total_dist


def safe_div(numerator, denominator):
    return numerator / denominator if denominator > 0 else 0.0


def evaluate_dataset(pred_folder, gt_folder, method_name, layout_type="complex"):
    gt_folder_path = Path(gt_folder)
    pred_folder_path = Path(pred_folder)

    if not gt_folder_path.exists():
        raise FileNotFoundError(f"Ground-truth directory not found: {gt_folder_path}")

    gt_files = sorted(glob.glob(os.path.join(str(gt_folder_path), "*.xml")))
    if not gt_files:
        raise FileNotFoundError(f"No PAGE-XML files found in ground-truth directory: {gt_folder_path}")

    sum_page_dist = 0
    sum_page_gt_len = 0
    sum_line_dist_50 = 0
    sum_line_dist_75 = 0
    sum_line_dist_range = 0
    total_gt_len_all_files = 0
    per_page_results = []

    for gt_path in gt_files:
        filename_base = os.path.splitext(os.path.basename(gt_path))[0]
        pred_path = pred_folder_path / f"{filename_base}.xml"

        gt_objs = parse_pagexml(gt_path)
        pred_objs = parse_pagexml(str(pred_path)) if pred_path.exists() else []

        current_gt_len = sum(len(item["text"]) for item in gt_objs)
        total_gt_len_all_files += current_gt_len

        page_dist, page_len = calculate_page_level_stats(gt_objs, pred_objs)
        sum_page_dist += page_dist
        sum_page_gt_len += page_len

        matches, unmatched_gt, unmatched_pred = pair_lines_by_polygon_iou(gt_objs, pred_objs)
        cost_50 = calculate_line_level_cost(matches, unmatched_gt, unmatched_pred, gt_objs, pred_objs, 0.5)
        cost_75 = calculate_line_level_cost(matches, unmatched_gt, unmatched_pred, gt_objs, pred_objs, 0.75)

        sum_line_dist_50 += cost_50
        sum_line_dist_75 += cost_75

        file_range_dist = 0
        for threshold in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
            file_range_dist += calculate_line_level_cost(
                matches,
                unmatched_gt,
                unmatched_pred,
                gt_objs,
                pred_objs,
                threshold,
            )
        sum_line_dist_range += file_range_dist

        per_page_results.append(
            {
                "filename": filename_base,
                "page_cer": safe_div(page_dist, page_len),
                "line_cer_50": safe_div(cost_50, current_gt_len),
                "gt_len": current_gt_len,
                "prediction_found": pred_path.exists(),
            }
        )

    result = {
        "method_name": method_name,
        "layout_type": layout_type,
        "pred_folder": str(pred_folder_path.resolve()),
        "gt_folder": str(gt_folder_path.resolve()),
        "files_processed": len(gt_files),
        "aggregate_metrics": {
            "page_cer": safe_div(sum_page_dist, sum_page_gt_len),
            "line_cer_50": safe_div(sum_line_dist_50, total_gt_len_all_files),
            "line_cer_75": safe_div(sum_line_dist_75, total_gt_len_all_files),
            "line_cer_range": safe_div(sum_line_dist_range, total_gt_len_all_files) / 10.0,
        },
        "per_page": per_page_results,
    }

    return result


def format_report(result):
    aggregate = result["aggregate_metrics"]
    lines = [
        "",
        "==================================================",
        f"EVALUATING: {result['method_name']}",
        f"LAYOUT TYPE: {result['layout_type']}",
        "==================================================",
        f"Files Processed: {result['files_processed']}",
        "------------------------------------------------------------",
        "AGGREGATE METRICS:",
        f"1. PAGE-LEVEL CER:             {aggregate['page_cer']:.4f}",
        f"2. LINE-LEVEL CER (IoU 0.50):  {aggregate['line_cer_50']:.4f}",
        f"3. LINE-LEVEL CER (IoU 0.75):  {aggregate['line_cer_75']:.4f}",
        f"4. LINE-LEVEL CER (Range):     {aggregate['line_cer_range']:.4f}",
        "------------------------------------------------------------",
        "PER-PAGE BREAKDOWN:",
        f"{'Filename':<35} | {'Page CER':<10} | {'Line CER (0.5)':<15} | {'GT Len':<6} | {'Pred?':<5}",
        "---------------------------------------------------------------------------------------",
    ]

    for page in result["per_page"]:
        lines.append(
            f"{page['filename']:<35} | "
            f"{page['page_cer']:.4f}     | "
            f"{page['line_cer_50']:.4f}          | "
            f"{page['gt_len']:<6} | "
            f"{'yes' if page['prediction_found'] else 'no':<5}"
        )

    lines.append("==================================================")
    lines.append("")
    return "\n".join(lines)


def write_report_files(result, text_path=None, json_path=None):
    written_paths = []

    if text_path is not None:
        text_path = Path(text_path)
        text_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.write_text(format_report(result), encoding="utf-8")
        written_paths.append(text_path)

    if json_path is not None:
        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        written_paths.append(json_path)

    return written_paths


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Evaluate predicted PAGE-XML files against PAGE-XML ground truth.")
    parser.add_argument("--pred-folder", required=True, help="Directory containing predicted PAGE-XML files.")
    parser.add_argument("--gt-folder", required=True, help="Directory containing ground-truth PAGE-XML files.")
    parser.add_argument("--method-name", default="evaluation-run", help="Human-readable name shown in the report.")
    parser.add_argument(
        "--layout-type",
        choices=["simple", "complex"],
        default="complex",
        help="Whether to apply the simple-layout width filter before evaluation.",
    )
    parser.add_argument("--output-text", help="Optional path to save the formatted text report.")
    parser.add_argument("--output-json", help="Optional path to save the JSON report.")
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    result = evaluate_dataset(
        pred_folder=args.pred_folder,
        gt_folder=args.gt_folder,
        method_name=args.method_name,
        layout_type=args.layout_type,
    )
    report = format_report(result)
    print(report)
    write_report_files(result, text_path=args.output_text, json_path=args.output_json)


if __name__ == "__main__":
    main()
