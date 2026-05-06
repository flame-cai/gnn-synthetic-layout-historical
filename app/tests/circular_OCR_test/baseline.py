from __future__ import annotations

import json
import math
from pathlib import Path


class BaselineComparisonError(ValueError):
    pass


def load_baseline(path: str | Path) -> dict:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BaselineComparisonError(f"Malformed baseline JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise BaselineComparisonError("Baseline JSON must contain an object.")
    return payload


def _finite_number(value) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


def compare_metrics_to_baseline(current_metrics: dict, baseline: dict) -> dict:
    metric_name = baseline.get("primary_blocking_metric_name")
    direction = baseline.get("metric_direction")
    baseline_metrics = baseline.get("metrics")
    if not metric_name or direction not in {"lower_is_better", "higher_is_better"} or not isinstance(baseline_metrics, dict):
        raise BaselineComparisonError("Baseline must define primary metric, metric direction, and metrics object.")
    if metric_name not in baseline_metrics:
        raise BaselineComparisonError(f"Baseline missing metric value for {metric_name}.")

    baseline_value = _finite_number(baseline_metrics.get(metric_name))
    current_value = _finite_number(current_metrics.get(metric_name))
    minimum_delta = _finite_number(baseline.get("minimum_improvement_delta", 0.0))
    if minimum_delta is None:
        raise BaselineComparisonError("minimum_improvement_delta must be numeric.")
    if baseline_value is None:
        raise BaselineComparisonError(f"Baseline metric {metric_name} must be finite.")
    if current_value is None:
        return {
            "passed": False,
            "metric_name": metric_name,
            "direction": direction,
            "baseline_value": baseline_metrics.get(metric_name),
            "current_value": current_metrics.get(metric_name),
            "minimum_improvement_delta": minimum_delta,
            "delta": None,
            "failure_message": f"Current metric {metric_name} is missing or non-finite.",
        }

    if direction == "lower_is_better":
        delta = baseline_value - current_value
    else:
        delta = current_value - baseline_value
    passed = delta >= minimum_delta
    return {
        "passed": passed,
        "metric_name": metric_name,
        "direction": direction,
        "baseline_value": baseline_value,
        "current_value": current_value,
        "minimum_improvement_delta": minimum_delta,
        "delta": delta,
        "failure_message": "" if passed else (
            f"{metric_name} improved by {delta}, below required delta {minimum_delta}."
        ),
    }
