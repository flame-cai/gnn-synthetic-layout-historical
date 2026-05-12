from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

STRATEGY_ROLE_CONFIG_JSON = """{
  "benchmark_strategy_name": "legacy_axis_bound_v1",
  "proposed_strategy_name": "local_tangent_band_v1",
  "promotion_history": []
}
"""

STRATEGY_ROLE_CONFIG = json.loads(STRATEGY_ROLE_CONFIG_JSON)
STRATEGY_ROLE_CONFIG_PATH = Path(__file__).resolve()


def _clone_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(payload))


def normalize_strategy_role_config_payload(payload: dict[str, Any]) -> dict[str, Any]:
    benchmark_strategy_name = str(payload["benchmark_strategy_name"]).strip()
    if not benchmark_strategy_name:
        raise ValueError("benchmark_strategy_name must be a non-empty string.")

    proposed_strategy_name = payload.get("proposed_strategy_name")
    if proposed_strategy_name is not None:
        proposed_strategy_name = str(proposed_strategy_name).strip() or None

    promotion_history = payload.get("promotion_history", [])
    if not isinstance(promotion_history, list):
        raise ValueError("promotion_history must be a list.")

    return {
        "benchmark_strategy_name": benchmark_strategy_name,
        "proposed_strategy_name": proposed_strategy_name,
        "promotion_history": _clone_payload({"items": promotion_history})["items"],
    }


def get_strategy_role_config() -> dict[str, Any]:
    return normalize_strategy_role_config_payload(STRATEGY_ROLE_CONFIG)


def get_benchmark_strategy_name() -> str:
    return get_strategy_role_config()["benchmark_strategy_name"]


def get_proposed_strategy_name() -> str | None:
    return get_strategy_role_config()["proposed_strategy_name"]


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
    "STRATEGY_ROLE_CONFIG",
    "STRATEGY_ROLE_CONFIG_PATH",
    "get_benchmark_strategy_name",
    "get_proposed_strategy_name",
    "get_strategy_role_config",
    "normalize_strategy_role_config_payload",
    "render_strategy_role_config",
    "write_strategy_role_config",
]
