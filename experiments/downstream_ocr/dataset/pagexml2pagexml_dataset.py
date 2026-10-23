from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
import tempfile
import threading
import xml.etree.ElementTree as ET

from ..omnidocbench_v1_5 import (
    OMNIDOCBENCH_BRANCH,
    OMNIDOCBENCH_COMMIT,
    OMNIDOCBENCH_REPOSITORY,
)
from ..omnidocbench_v1_5.metrics.cal_metric import call_Edit_dist
from ..omnidocbench_v1_5.registry.registry import DATASET_REGISTRY
from ..omnidocbench_v1_5.utils.match import match_gt2pred_simple
from ..devanagari_textedit import evaluate_devanagari_textedit_items


LOGGER = logging.getLogger(__name__)

PAGE_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
NS = {"pc": PAGE_NS}
DATASET_NAME = "pagexml2pagexml_dataset"
ADAPTER_VERSION = "1.0.0"
METRIC_NAME = "Unordered PAGE-XML TextLine TextEdit"
_OFFICIAL_METRIC_LOCK = threading.Lock()

TEXTEDIT_AGGREGATE_KEYS = (
    "mean_textedit",
    "median_textedit",
    "micro_textedit",
    "textedit_all_page_avg",
    "textedit_edit_whole",
    "textedit_edit_sample_avg",
)


def textedit_reproducibility_metadata() -> dict:
    return {
        "metric_name": METRIC_NAME,
        "metric_description": (
            "Unordered PAGE-XML TextLine TextEdit using OmniDocBench v1.5 "
            "simple_match. Each PAGE TextLine/TextEquiv/Unicode transcription "
            "is atomic. TextRegion information, XML order, coordinates, "
            "baselines, geometry, and reading order are ignored."
        ),
        "omnidocbench_repository": OMNIDOCBENCH_REPOSITORY,
        "omnidocbench_git_commit": OMNIDOCBENCH_COMMIT,
        "omnidocbench_branch_or_tag": OMNIDOCBENCH_BRANCH,
        "adapter_version": ADAPTER_VERSION,
        "dataset_name": DATASET_NAME,
        "page_namespace": PAGE_NS,
        "text_equiv_selection_policy": (
            "prefer direct TextEquiv@index='0'; otherwise the smallest numeric "
            "index; otherwise the first direct TextEquiv in document order"
        ),
        "empty_text_policy": (
            "missing Unicode and empty or whitespace-only Unicode values are "
            "skipped identically for ground truth and prediction and logged"
        ),
        "file_pairing_policy": (
            "same relative XML filename under GT and prediction roots; a "
            "missing prediction file supplies an empty prediction item list"
        ),
        "normalization_functions_used": [
            "utils.data_preprocess.textblock2unicode",
            "utils.data_preprocess.clean_string",
        ],
        "pairwise_cost_function": "utils.match.compute_edit_distance_matrix_new",
        "match_method": "simple_match",
        "matcher": "utils.match.match_gt2pred_simple",
        "metric_function": "metrics.cal_metric.call_Edit_dist",
        "aggregation_field": "Edit_dist.ALL_page_avg",
        "supplementary_fields": [
            "Edit_dist.edit_whole",
            "Edit_dist.edit_sample_avg",
        ],
        "granularity_note": (
            "PAGE TextLine granularity; this is not the official paragraph-level "
            "OmniDocBench leaderboard score"
        ),
    }


def _numeric_index(text_equiv: ET.Element) -> int | None:
    raw_index = text_equiv.get("index")
    if raw_index is None:
        return None
    try:
        return int(raw_index)
    except ValueError:
        return None


def extract_unicode(text_line: ET.Element) -> str | None:
    """Select one direct TextLine/TextEquiv/Unicode transcription."""
    text_equivs = list(text_line.findall("./pc:TextEquiv", NS))
    if not text_equivs:
        return None

    indexed = [
        (xml_order, text_equiv, _numeric_index(text_equiv))
        for xml_order, text_equiv in enumerate(text_equivs)
    ]
    zero_indexed = [item for item in indexed if item[2] == 0]
    if zero_indexed:
        selected = zero_indexed[0][1]
    else:
        numeric = [item for item in indexed if item[2] is not None]
        if numeric:
            selected = min(numeric, key=lambda item: (int(item[2]), item[0]))[1]
        else:
            selected = indexed[0][1]

    unicode_element = selected.find("./pc:Unicode", NS)
    if unicode_element is None or unicode_element.text is None:
        return None
    return unicode_element.text


def _item(*, text: str, index: int, source_id: str, ground_truth: bool) -> dict:
    if ground_truth:
        return {
            "category_type": "text_block",
            "text": text,
            "attribute": {},
            "position": [index, index],
            "source_id": source_id,
        }
    return {
        "category_type": "text_all",
        "content": text,
        "position": [index, index],
        "source_id": source_id,
    }


def parse_pagexml(
    path: str | Path,
    *,
    ground_truth: bool,
) -> tuple[str, list[dict]]:
    """Convert PAGE TextLines to OmniDocBench-compatible text items."""
    xml_path = Path(path)
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"Invalid PAGE-XML file: {xml_path}") from exc

    page = root.find("./pc:Page", NS)
    if page is None:
        raise ValueError(
            f"No PAGE Page element in namespace {PAGE_NS!r} found in {xml_path}"
        )

    image_name = page.get("imageFilename") or xml_path.stem
    items: list[dict] = []
    for traversal_index, text_line in enumerate(page.findall(".//pc:TextLine", NS)):
        source_id = text_line.get("id") or f"line_{traversal_index}"
        text = extract_unicode(text_line)
        if text is None or not text.strip():
            LOGGER.warning(
                "Skipping PAGE TextLine with no usable Unicode text: file=%s line_id=%s",
                xml_path,
                source_id,
            )
            continue
        items.append(
            _item(
                text=text,
                index=len(items),
                source_id=source_id,
                ground_truth=ground_truth,
            )
        )
    return image_name, items


def items_from_text_lines(lines, *, ground_truth: bool) -> list[dict]:
    """Create adapter items from already parsed lines for in-memory callers."""
    items = []
    for traversal_index, line in enumerate(lines):
        text = str(getattr(line, "text", "") or "")
        if not text.strip():
            continue
        items.append(
            _item(
                text=text,
                index=len(items),
                source_id=str(getattr(line, "line_id", f"line_{traversal_index}")),
                ground_truth=ground_truth,
            )
        )
    return items


def match_items(
    gt_items: list[dict],
    pred_items: list[dict],
    *,
    image_name: str,
) -> list[dict]:
    if not gt_items and not pred_items:
        return []
    matches, unmatched_table_predictions = match_gt2pred_simple(
        gt_items,
        pred_items,
        "text",
        image_name,
    )
    if unmatched_table_predictions is not None:
        raise AssertionError("Text matching unexpectedly returned table predictions.")
    return matches


def match_pagexml(
    gt_xml_path: str | Path,
    pred_xml_path: str | Path | None,
    *,
    image_name: str | None = None,
) -> list[dict]:
    gt_image_name, gt_items = parse_pagexml(gt_xml_path, ground_truth=True)
    if pred_xml_path is None or not Path(pred_xml_path).exists():
        if pred_xml_path is not None:
            LOGGER.warning(
                "Missing prediction PAGE-XML; evaluating an empty prediction: %s",
                pred_xml_path,
            )
        pred_items: list[dict] = []
    else:
        _, pred_items = parse_pagexml(pred_xml_path, ground_truth=False)
    return match_items(
        gt_items,
        pred_items,
        image_name=image_name or gt_image_name,
    )


@dataclass(frozen=True)
class PageXmlPair:
    key: str
    gt_xml_path: Path
    pred_xml_path: Path | None
    image_name: str | None = None


@dataclass(frozen=True)
class TextEditEvaluation:
    page_metrics: dict[str, dict]
    official_result: dict
    matches_by_key: dict[str, tuple[dict, ...]]
    devanagari_matches_by_key: dict[str, tuple[dict, ...]]


def _metric_image_name(key: str, image_name: str) -> str:
    extension = Path(image_name).suffix.lower()
    if extension not in {".jpg", ".png"}:
        extension = ".jpg"
    safe_key = "".join(character if character.isalnum() else "_" for character in key)
    return f"{safe_key}__{Path(image_name).stem}{extension}"


def _call_official_edit_dist(samples: list[dict]) -> dict:
    if not samples:
        return {
            "Edit_dist": {
                "ALL_page_avg": 0.0,
                "edit_whole": 0.0,
                "edit_sample_avg": 0.0,
            }
        }
    with _OFFICIAL_METRIC_LOCK:
        previous_cwd = Path.cwd()
        with tempfile.TemporaryDirectory(prefix="pagexml_textedit_") as tmp_dir:
            temporary_root = Path(tmp_dir)
            (temporary_root / "result").mkdir()
            try:
                os.chdir(temporary_root)
                _, result = call_Edit_dist(samples).evaluate(
                    save_name="pagexml_textedit"
                )
            finally:
                os.chdir(previous_cwd)
    return result


def _page_metric(samples: list[dict]) -> dict:
    if not samples:
        return {
            "textedit_distance_sum": 0,
            "textedit_max_length_sum": 0,
            "textedit_sample_ratio_sum": 0.0,
            "textedit_sample_count": 0,
            "textedit_page_count": 1,
            "textedit": 0.0,
            "textedit_all_page_avg": 0.0,
            "textedit_edit_whole": 0.0,
            "textedit_edit_sample_avg": 0.0,
            "textedit_match_count": 0,
        }

    edit_sum = sum(int(sample["Edit_num"]) for sample in samples)
    upper_sum = sum(int(sample["upper_len"]) for sample in samples)
    ratio_sum = sum(
        float(sample["Edit_num"]) / float(sample["upper_len"])
        for sample in samples
        if int(sample["upper_len"]) > 0
    )
    sample_count = sum(1 for sample in samples if int(sample["upper_len"]) > 0)
    page_ratio = edit_sum / upper_sum if upper_sum > 0 else 0.0
    sample_average = ratio_sum / sample_count if sample_count > 0 else 0.0
    return {
        "textedit_distance_sum": edit_sum,
        "textedit_max_length_sum": upper_sum,
        "textedit_sample_ratio_sum": ratio_sum,
        "textedit_sample_count": sample_count,
        "textedit_page_count": 1,
        "textedit": page_ratio,
        "textedit_all_page_avg": page_ratio,
        "textedit_edit_whole": page_ratio,
        "textedit_edit_sample_avg": sample_average,
        "textedit_match_count": len(samples),
    }


def evaluate_pagexml_pairs(pairs: list[PageXmlPair]) -> TextEditEvaluation:
    all_matches: list[dict] = []
    matches_by_key: dict[str, tuple[dict, ...]] = {}
    devanagari_matches_by_key: dict[str, tuple[dict, ...]] = {}
    devanagari_page_metrics: dict[str, dict] = {}
    for pair in pairs:
        gt_image_name, gt_items = parse_pagexml(
            pair.gt_xml_path,
            ground_truth=True,
        )
        if pair.pred_xml_path is None or not pair.pred_xml_path.exists():
            if pair.pred_xml_path is not None:
                LOGGER.warning(
                    "Missing prediction PAGE-XML; evaluating an empty prediction: %s",
                    pair.pred_xml_path,
                )
            pred_items: list[dict] = []
        else:
            _, pred_items = parse_pagexml(
                pair.pred_xml_path,
                ground_truth=False,
            )
        metric_image_name = _metric_image_name(
            pair.key,
            pair.image_name or gt_image_name,
        )
        matches = match_items(
            gt_items,
            pred_items,
            image_name=metric_image_name,
        )
        for match in matches:
            match["_pagexml_evaluation_key"] = pair.key
        matches_by_key[pair.key] = tuple(matches)
        all_matches.extend(matches)
        (
            devanagari_page_metrics[pair.key],
            devanagari_matches_by_key[pair.key],
        ) = evaluate_devanagari_textedit_items(gt_items, pred_items)

    official_result = _call_official_edit_dist(all_matches)
    page_metrics = {
        pair.key: {
            **_page_metric(
                [
                    sample
                    for sample in all_matches
                    if sample["_pagexml_evaluation_key"] == pair.key
                ]
            ),
            **devanagari_page_metrics[pair.key],
        }
        for pair in pairs
    }
    return TextEditEvaluation(
        page_metrics=page_metrics,
        official_result=official_result,
        matches_by_key=matches_by_key,
        devanagari_matches_by_key=devanagari_matches_by_key,
    )


def evaluate_text_line_items(
    gt_items: list[dict],
    pred_items: list[dict],
    *,
    image_name: str,
) -> dict:
    matches = match_items(gt_items, pred_items, image_name=image_name)
    _call_official_edit_dist(matches)
    devanagari_metric, _ = evaluate_devanagari_textedit_items(
        gt_items,
        pred_items,
    )
    return {
        **_page_metric(matches),
        **devanagari_metric,
    }


@DATASET_REGISTRY.register(DATASET_NAME)
class PageXml2PageXmlDataset:
    """Official-style dataset wrapper for same-relative-name PAGE-XML pairs."""

    def __init__(self, cfg_task: dict):
        dataset_cfg = cfg_task["dataset"]
        self.match_method = dataset_cfg.get("match_method", "simple_match")
        if self.match_method != "simple_match":
            raise ValueError(
                "pagexml2pagexml_dataset supports only match_method=simple_match"
            )
        gt_root = Path(dataset_cfg["ground_truth"]["data_path"])
        pred_root = Path(dataset_cfg["prediction"]["data_path"])
        self.samples = {
            "text_block": self.get_matched_elements(gt_root, pred_root)
        }

    def __getitem__(self, cat_name, idx):
        return self.samples[cat_name][idx]

    def get_matched_elements(
        self,
        gt_root: Path,
        pred_root: Path,
    ) -> list[dict]:
        matched: list[dict] = []
        gt_paths = sorted(gt_root.rglob("*.xml"))
        if not gt_paths:
            raise ValueError(f"No ground-truth PAGE-XML files found in {gt_root}")
        for gt_path in gt_paths:
            relative_path = gt_path.relative_to(gt_root)
            pred_path = pred_root / relative_path
            image_name, _ = parse_pagexml(gt_path, ground_truth=True)
            matched.extend(
                match_pagexml(
                    gt_path,
                    pred_path,
                    image_name=image_name,
                )
            )
        return matched
