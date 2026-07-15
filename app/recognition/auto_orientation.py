from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import unicodedata
from typing import Any, Mapping


IDENTITY_TRANSFORM = "identity"
ROTATE_180_TRANSFORM = "rotate_180"
SUPPORTED_AUTO_ORIENTATION_TRANSFORMS = frozenset(
    {IDENTITY_TRANSFORM, ROTATE_180_TRANSFORM}
)
AUTO_ORIENTATION_LINE_KINDS = frozenset({"curved_open", "closed_circular"})

# Devanagari, Devanagari Extended, Vedic Extensions, and Devanagari Extended-A.
DEVANAGARI_RANGES = (
    (0x0900, 0x097F),
    (0x1CD0, 0x1CFF),
    (0xA8E0, 0xA8FF),
    (0x11B00, 0x11B5F),
)


def auto_orientation_transform_from_custom(custom: str | None) -> str | None:
    """Read the persisted OCR-selected transform from PAGE TextEquiv/@custom."""
    for raw_part in str(custom or "").split(";"):
        key, separator, value = raw_part.strip().partition(":")
        if separator and key == "auto_orientation_transform":
            normalized = value.strip()
            if normalized in SUPPORTED_AUTO_ORIENTATION_TRANSFORMS:
                return normalized
    return None


def auto_orientation_custom_metadata(custom: str | None) -> str:
    """Return only auto-orientation fields so text edits can preserve them."""
    fields = []
    for raw_part in str(custom or "").split(";"):
        part = raw_part.strip()
        key, separator, _ = part.partition(":")
        if separator and key.startswith("auto_orientation_"):
            fields.append(part)
    return ";".join(fields)


@dataclass(frozen=True)
class DevanagariPredictionEvidence:
    text: str
    devanagari_count: int
    foreign_letter_or_number_count: int
    script_signal_count: int
    devanagari_purity: float
    net_devanagari_evidence: int

    @property
    def rank(self) -> tuple[int, float, int, int]:
        return (
            self.net_devanagari_evidence,
            self.devanagari_purity,
            self.devanagari_count,
            -self.foreign_letter_or_number_count,
        )

    def to_metadata(self) -> dict:
        payload = asdict(self)
        payload["rank"] = list(self.rank)
        return payload


@dataclass(frozen=True)
class AutoOrientationSelection:
    selected_transform: str
    selected_text: str
    reason: str
    candidates: dict[str, DevanagariPredictionEvidence]

    def to_metadata(self) -> dict:
        return {
            "selection_model": "decoded_text_devanagari_evidence_v1",
            "uses_model_confidence": False,
            "selected_transform": self.selected_transform,
            "selected_text": self.selected_text,
            "reason": self.reason,
            "candidates": {
                transform: evidence.to_metadata()
                for transform, evidence in self.candidates.items()
            },
        }


def _is_devanagari_codepoint(character: str) -> bool:
    codepoint = ord(character)
    return any(start <= codepoint <= end for start, end in DEVANAGARI_RANGES)


def devanagari_prediction_evidence(text: str | None) -> DevanagariPredictionEvidence:
    normalized = unicodedata.normalize("NFC", str(text or ""))
    devanagari_count = 0
    foreign_count = 0

    for character in normalized:
        category = unicodedata.category(character)
        if not category or category[0] not in {"L", "M", "N"}:
            continue
        if _is_devanagari_codepoint(character):
            devanagari_count += 1
        else:
            foreign_count += 1

    signal_count = devanagari_count + foreign_count
    purity = devanagari_count / signal_count if signal_count else 0.0
    return DevanagariPredictionEvidence(
        text=normalized,
        devanagari_count=devanagari_count,
        foreign_letter_or_number_count=foreign_count,
        script_signal_count=signal_count,
        devanagari_purity=purity,
        net_devanagari_evidence=devanagari_count - foreign_count,
    )


def _valid_direction(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return False
    try:
        return math.hypot(float(value[0]), float(value[1])) > 1e-6
    except (TypeError, ValueError):
        return False


def _line_kind(crop_metadata: Mapping[str, Any]) -> str | None:
    topology = crop_metadata.get("topology")
    if isinstance(topology, Mapping) and topology.get("line_kind"):
        return str(topology["line_kind"])

    strategy_metadata = crop_metadata.get("strategy_line_metadata")
    if isinstance(strategy_metadata, Mapping):
        if strategy_metadata.get("line_kind"):
            return str(strategy_metadata["line_kind"])
        topology = strategy_metadata.get("topology")
        if isinstance(topology, Mapping) and topology.get("line_kind"):
            return str(topology["line_kind"])
    return None


def has_explicit_reading_direction_annotation(crop_metadata: Mapping[str, Any]) -> bool:
    orientation = crop_metadata.get("orientation")
    if isinstance(orientation, Mapping) and _valid_direction(orientation.get("reading_direction")):
        return True

    topology = crop_metadata.get("topology")
    if isinstance(topology, Mapping) and _valid_direction(topology.get("reading_direction")):
        return True

    strategy_metadata = crop_metadata.get("strategy_line_metadata")
    if isinstance(strategy_metadata, Mapping):
        annotation = strategy_metadata.get("reading_direction_annotation")
        if isinstance(annotation, Mapping) and _valid_direction(annotation.get("reading_direction")):
            return True
    return False


def should_auto_orient_from_predictions(crop_metadata: Mapping[str, Any] | None) -> bool:
    if not isinstance(crop_metadata, Mapping) or not crop_metadata.get("used_unwrap"):
        return False
    if crop_metadata.get("applied_auto_orientation_transform") in SUPPORTED_AUTO_ORIENTATION_TRANSFORMS:
        return False
    if _line_kind(crop_metadata) not in AUTO_ORIENTATION_LINE_KINDS:
        return False
    if has_explicit_reading_direction_annotation(crop_metadata):
        return False

    orientation = crop_metadata.get("orientation")
    candidate_transforms = (
        orientation.get("candidate_transforms")
        if isinstance(orientation, Mapping)
        else None
    )
    return isinstance(candidate_transforms, (list, tuple)) and ROTATE_180_TRANSFORM in candidate_transforms


def select_orientation_from_predictions(
    predictions_by_transform: Mapping[str, str | None],
) -> AutoOrientationSelection:
    identity = devanagari_prediction_evidence(predictions_by_transform.get(IDENTITY_TRANSFORM))
    rotated = devanagari_prediction_evidence(predictions_by_transform.get(ROTATE_180_TRANSFORM))
    candidates = {
        IDENTITY_TRANSFORM: identity,
        ROTATE_180_TRANSFORM: rotated,
    }

    if rotated.script_signal_count == 0:
        reason = (
            "no_script_evidence_preserve_identity"
            if identity.script_signal_count == 0
            else "rotate_180_has_no_script_evidence"
        )
        return AutoOrientationSelection(
            selected_transform=IDENTITY_TRANSFORM,
            selected_text=identity.text,
            reason=reason,
            candidates=candidates,
        )

    if identity.script_signal_count == 0:
        return AutoOrientationSelection(
            selected_transform=ROTATE_180_TRANSFORM,
            selected_text=rotated.text,
            reason="identity_has_no_script_evidence",
            candidates=candidates,
        )

    if rotated.rank > identity.rank:
        return AutoOrientationSelection(
            selected_transform=ROTATE_180_TRANSFORM,
            selected_text=rotated.text,
            reason="rotate_180_has_more_devanagari_evidence",
            candidates=candidates,
        )

    return AutoOrientationSelection(
        selected_transform=IDENTITY_TRANSFORM,
        selected_text=identity.text,
        reason="identity_has_equal_or_more_devanagari_evidence",
        candidates=candidates,
    )
