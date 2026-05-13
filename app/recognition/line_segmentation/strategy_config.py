from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

STRATEGY_ROLE_CONFIG_JSON = """{
  "benchmark_strategy_name": "legacy_axis_bound_v1",
  "proposed_strategy_name": "local_tangent_band_v1",
  "production_strategy_name": "legacy_axis_bound_v1",
  "research_promotion_history": [],
  "production_adoption_history": []
}
"""

STRATEGY_ROLE_CONFIG = json.loads(STRATEGY_ROLE_CONFIG_JSON)
STRATEGY_ROLE_CONFIG_PATH = Path(__file__).resolve()
DEFAULT_PRODUCTION_STRATEGY_NAME = "legacy_axis_bound_v1"


def _clone_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(payload))


def _normalize_optional_strategy_name(value: Any) -> str | None:
    if value is None:
        return None
    return str(value).strip() or None


def _normalize_required_strategy_name(payload: dict[str, Any], key: str) -> str:
    strategy_name = str(payload[key]).strip()
    if not strategy_name:
        raise ValueError(f"{key} must be a non-empty string.")
    return strategy_name


def _normalize_history_list(payload: dict[str, Any], key: str, *, legacy_key: str | None = None) -> list[Any]:
    if key in payload:
        history = payload[key]
    elif legacy_key is not None and legacy_key in payload:
        history = payload[legacy_key]
    else:
        history = []
    if not isinstance(history, list):
        raise ValueError(f"{key} must be a list.")
    return _clone_payload({"items": history})["items"]


def normalize_strategy_role_config_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("strategy role config payload must be a dictionary.")

    benchmark_strategy_name = _normalize_required_strategy_name(payload, "benchmark_strategy_name")
    proposed_strategy_name = _normalize_optional_strategy_name(payload.get("proposed_strategy_name"))
    production_strategy_name = _normalize_optional_strategy_name(
        payload.get("production_strategy_name", DEFAULT_PRODUCTION_STRATEGY_NAME)
    )
    if production_strategy_name is None:
        raise ValueError("production_strategy_name must be a non-empty string.")

    return {
        "benchmark_strategy_name": benchmark_strategy_name,
        "proposed_strategy_name": proposed_strategy_name,
        "production_strategy_name": production_strategy_name,
        "research_promotion_history": _normalize_history_list(
            payload,
            "research_promotion_history",
            legacy_key="promotion_history",
        ),
        "production_adoption_history": _normalize_history_list(payload, "production_adoption_history"),
    }


def get_strategy_role_config() -> dict[str, Any]:
    return normalize_strategy_role_config_payload(STRATEGY_ROLE_CONFIG)


def get_benchmark_strategy_name() -> str:
    return get_strategy_role_config()["benchmark_strategy_name"]


def get_proposed_strategy_name() -> str | None:
    return get_strategy_role_config()["proposed_strategy_name"]


def get_production_strategy_name() -> str:
    return get_strategy_role_config()["production_strategy_name"]


def load_strategy_role_config_from_path(path: str | Path) -> dict[str, Any]:
    source = Path(path).read_text(encoding="utf-8")
    match = re.search(r'STRATEGY_ROLE_CONFIG_JSON = """([\s\S]*?)"""', source)
    if match is None:
        match = re.search(r"STRATEGY_ROLE_CONFIG_JSON = '''([\s\S]*?)'''", source)
    if match is None:
        raise ValueError(f"Could not locate STRATEGY_ROLE_CONFIG_JSON in {path}")
    return normalize_strategy_role_config_payload(json.loads(match.group(1)))


def render_strategy_role_config(payload: dict[str, Any]) -> str:
    normalized = normalize_strategy_role_config_payload(payload)
    source = STRATEGY_ROLE_CONFIG_PATH.read_text(encoding="utf-8")
    replacement = 'STRATEGY_ROLE_CONFIG_JSON = """' + json.dumps(
        normalized,
        indent=2,
        ensure_ascii=False,
    ) + '\n"""'
    updated = re.sub(
        r'STRATEGY_ROLE_CONFIG_JSON = """[\s\S]*?"""',
        lambda _: replacement,
        source,
        count=1,
    )
    if 'STRATEGY_ROLE_CONFIG_JSON = """' not in updated:
        raise ValueError("Could not locate STRATEGY_ROLE_CONFIG_JSON block for rewrite.")
    return updated


def write_strategy_role_config(path: str | Path, payload: dict[str, Any]) -> Path:
    destination = Path(path)
    destination.write_text(render_strategy_role_config(payload), encoding="utf-8")
    return destination


__all__ = [
    "DEFAULT_PRODUCTION_STRATEGY_NAME",
    "STRATEGY_ROLE_CONFIG",
    "STRATEGY_ROLE_CONFIG_PATH",
    "get_benchmark_strategy_name",
    "get_proposed_strategy_name",
    "get_production_strategy_name",
    "get_strategy_role_config",
    "load_strategy_role_config_from_path",
    "normalize_strategy_role_config_payload",
    "render_strategy_role_config",
    "write_strategy_role_config",
]
