from __future__ import annotations

import unicodedata
from typing import Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment

from .text import levenshtein_distance


METRIC_NAME = "Devanagari-relaxed unordered TextLine TextEdit"
METRIC_VERSION = "1.0.0"
SANSKRIT_PUNCTUATION = frozenset({"।", "॥", "॰"})

DEVANAGARI_TEXTEDIT_AGGREGATE_KEYS = (
    "mean_devanagari_textedit",
    "median_devanagari_textedit",
    "micro_devanagari_textedit",
    "devanagari_textedit_all_page_avg",
    "devanagari_textedit_edit_whole",
    "devanagari_textedit_edit_sample_avg",
)


def devanagari_relaxed_normalize(text: str | None) -> str:
    """Keep textual Unicode content while ignoring spacing and most punctuation.

    Unicode letters, combining marks, and numbers are retained for every script,
    so mixed Sanskrit/English text is not silently discarded. Danda characters
    and the Devanagari abbreviation sign are retained because they can carry
    textual meaning in Sanskrit manuscript transcriptions.
    """
    normalized = unicodedata.normalize("NFC", text or "")
    return "".join(
        character
        for character in normalized
        if unicodedata.category(character)[0] in {"L", "M", "N"}
        or character in SANSKRIT_PUNCTUATION
    )


def devanagari_textedit_reproducibility_metadata() -> dict:
    return {
        "metric_name": METRIC_NAME,
        "metric_version": METRIC_VERSION,
        "metric_description": (
            "Unordered PAGE TextLine edit distance with Unicode-aware "
            "normalization intended for Devanagari manuscript transcription."
        ),
        "relationship_to_upstream_textedit": (
            "Additive metric. The existing OmniDocBench v1.5 TextEdit remains "
            "the upstream compatibility metric and is not changed."
        ),
        "unicode_normalization": "NFC",
        "kept_unicode_general_categories": [
            "L* (letters)",
            "M* (combining marks)",
            "N* (numbers)",
        ],
        "additional_kept_characters": sorted(SANSKRIT_PUNCTUATION),
        "ignored_content": (
            "Whitespace, format controls such as ZWJ/ZWNJ, symbols, and "
            "punctuation other than danda, double danda, and the Devanagari "
            "abbreviation sign."
        ),
        "empty_normalized_line_policy": (
            "Lines that become empty after normalization are skipped for both "
            "ground truth and prediction."
        ),
        "line_matching": (
            "One-to-one Hungarian assignment using normalized Levenshtein "
            "distance divided by max(reference length, prediction length). "
            "Unmatched reference lines are deletions; unmatched prediction "
            "lines are concatenated into one insertion record."
        ),
        "aggregation_field": "devanagari_textedit_all_page_avg",
        "aggregation_definition": (
            "For each page, sum character edits over matched/unmatched records "
            "and divide by the sum of max(reference length, prediction length); "
            "then take the unweighted mean over pages."
        ),
        "geometry_and_order": (
            "TextRegion membership, XML order, coordinates, baselines, geometry, "
            "and reading order are ignored."
        ),
        "lower_is_better": True,
    }


def _item_text(item: dict, *, ground_truth: bool) -> str:
    key = "text" if ground_truth else "content"
    return str(item.get(key, "") or "")


def _normalized_lines(
    items: Iterable[dict],
    *,
    ground_truth: bool,
) -> list[dict]:
    lines = []
    for item_index, item in enumerate(items):
        text = _item_text(item, ground_truth=ground_truth)
        normalized = devanagari_relaxed_normalize(text)
        if not normalized:
            continue
        lines.append(
            {
                "item_index": item_index,
                "source_id": str(item.get("source_id", item_index)),
                "text": text,
                "normalized": normalized,
            }
        )
    return lines


def _match_record(
    gt_line: dict | None,
    pred_lines: list[dict],
) -> dict:
    gt_text = "" if gt_line is None else str(gt_line["text"])
    norm_gt = "" if gt_line is None else str(gt_line["normalized"])
    pred_text = "".join(str(line["text"]) for line in pred_lines)
    norm_pred = "".join(str(line["normalized"]) for line in pred_lines)
    distance = levenshtein_distance(norm_gt, norm_pred)
    upper_len = max(len(norm_gt), len(norm_pred))
    return {
        "gt_source_id": None if gt_line is None else gt_line["source_id"],
        "pred_source_ids": [line["source_id"] for line in pred_lines],
        "gt": gt_text,
        "pred": pred_text,
        "norm_gt": norm_gt,
        "norm_pred": norm_pred,
        "distance": distance,
        "upper_len": upper_len,
        "ratio": distance / upper_len if upper_len > 0 else 0.0,
    }


def match_devanagari_textedit_items(
    gt_items: Iterable[dict],
    pred_items: Iterable[dict],
) -> list[dict]:
    """Match normalized TextLines without using layout or reading order."""
    gt_lines = _normalized_lines(gt_items, ground_truth=True)
    pred_lines = _normalized_lines(pred_items, ground_truth=False)

    if not gt_lines:
        return [_match_record(None, pred_lines)] if pred_lines else []
    if not pred_lines:
        return [_match_record(gt_line, []) for gt_line in gt_lines]

    costs = np.zeros((len(gt_lines), len(pred_lines)), dtype=float)
    for gt_index, gt_line in enumerate(gt_lines):
        for pred_index, pred_line in enumerate(pred_lines):
            norm_gt = str(gt_line["normalized"])
            norm_pred = str(pred_line["normalized"])
            costs[gt_index, pred_index] = (
                levenshtein_distance(norm_gt, norm_pred)
                / max(len(norm_gt), len(norm_pred))
            )

    matched_gt, matched_pred = linear_sum_assignment(costs)
    prediction_by_gt = {
        int(gt_index): int(pred_index)
        for gt_index, pred_index in zip(matched_gt, matched_pred)
    }
    matches = [
        _match_record(
            gt_line,
            (
                [pred_lines[prediction_by_gt[gt_index]]]
                if gt_index in prediction_by_gt
                else []
            ),
        )
        for gt_index, gt_line in enumerate(gt_lines)
    ]

    used_predictions = set(prediction_by_gt.values())
    unmatched_predictions = [
        pred_line
        for pred_index, pred_line in enumerate(pred_lines)
        if pred_index not in used_predictions
    ]
    if unmatched_predictions:
        matches.append(_match_record(None, unmatched_predictions))
    return matches


def devanagari_textedit_page_metric(matches: Iterable[dict]) -> dict:
    samples = list(matches)
    edit_sum = sum(int(sample["distance"]) for sample in samples)
    upper_sum = sum(int(sample["upper_len"]) for sample in samples)
    ratio_sum = sum(
        float(sample["ratio"])
        for sample in samples
        if int(sample["upper_len"]) > 0
    )
    sample_count = sum(
        1 for sample in samples if int(sample["upper_len"]) > 0
    )
    page_ratio = edit_sum / upper_sum if upper_sum > 0 else 0.0
    sample_average = ratio_sum / sample_count if sample_count > 0 else 0.0
    return {
        "devanagari_textedit_distance_sum": edit_sum,
        "devanagari_textedit_max_length_sum": upper_sum,
        "devanagari_textedit_sample_ratio_sum": ratio_sum,
        "devanagari_textedit_sample_count": sample_count,
        "devanagari_textedit_page_count": 1,
        "devanagari_textedit": page_ratio,
        "devanagari_textedit_all_page_avg": page_ratio,
        "devanagari_textedit_edit_whole": page_ratio,
        "devanagari_textedit_edit_sample_avg": sample_average,
        "devanagari_textedit_match_count": len(samples),
    }


def evaluate_devanagari_textedit_items(
    gt_items: Iterable[dict],
    pred_items: Iterable[dict],
) -> tuple[dict, tuple[dict, ...]]:
    matches = tuple(match_devanagari_textedit_items(gt_items, pred_items))
    return devanagari_textedit_page_metric(matches), matches
