from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = REPO_ROOT / "app"

if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recognition.line_segmentation.registry import (
    list_text_line_segmentation_strategies,
    validate_production_role_strategy,
)
from recognition.line_segmentation.runtime_config import PRODUCTION_STRATEGY_RUNTIME_CONFIGS
from recognition.line_segmentation.strategy_config import (
    STRATEGY_ROLE_CONFIG_PATH,
    load_strategy_role_config_from_path,
    write_strategy_role_config,
)


LOGGER = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _validate_registered_strategy(strategy_name: str) -> None:
    available = set(list_text_line_segmentation_strategies())
    if strategy_name not in available:
        raise ValueError(
            f"Production strategy {strategy_name!r} is not registered. "
            f"Available strategies: {', '.join(sorted(available))}"
        )
    validate_production_role_strategy(strategy_name)
    if strategy_name not in PRODUCTION_STRATEGY_RUNTIME_CONFIGS:
        raise ValueError(
            f"Production strategy {strategy_name!r} does not have a production runtime config. "
            "Add an explicit config before adopting it for app saves."
        )


def _build_adoption_history_entry(
    *,
    adopted_strategy_name: str,
    previous_production_strategy_name: str,
    author_or_tool: str,
    reason: str | None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "adopted_strategy_name": adopted_strategy_name,
        "previous_production_strategy_name": previous_production_strategy_name,
        "adoption_timestamp_utc": _utc_now_iso(),
        "author_or_tool": author_or_tool,
    }
    if reason:
        entry["reason"] = reason
    return entry


def adopt_text_line_strategy_for_app(
    *,
    strategy: str,
    reason: str | None = None,
    apply: bool = False,
    strategy_config_path: Path = STRATEGY_ROLE_CONFIG_PATH,
    author_or_tool: str = "scripts/adopt_text_line_strategy_for_app.py",
) -> dict[str, Any]:
    strategy = str(strategy).strip()
    if not strategy:
        raise ValueError("strategy must be a non-empty string.")
    normalized_reason = str(reason).strip() if reason is not None else None
    normalized_reason = normalized_reason or None

    _validate_registered_strategy(strategy)

    current_config = load_strategy_role_config_from_path(strategy_config_path)
    current_production_strategy = current_config["production_strategy_name"]
    if current_production_strategy == strategy:
        return {
            "changed": False,
            "applied": False,
            "idempotent": True,
            "message": f"The production app is already pinned to {strategy}.",
            "config_path": str(strategy_config_path.resolve()),
            "config_before": current_config,
            "config_after": current_config,
        }

    history_entry = _build_adoption_history_entry(
        adopted_strategy_name=strategy,
        previous_production_strategy_name=current_production_strategy,
        author_or_tool=author_or_tool,
        reason=normalized_reason,
    )
    updated_config = {
        **current_config,
        "production_strategy_name": strategy,
        "production_adoption_history": list(current_config.get("production_adoption_history", [])) + [history_entry],
    }

    result = {
        "changed": current_config != updated_config,
        "applied": False,
        "idempotent": False,
        "message": f"Adopt {strategy} as the production app text-line segmentation strategy.",
        "config_path": str(strategy_config_path.resolve()),
        "config_before": current_config,
        "config_after": updated_config,
    }
    if apply and result["changed"]:
        LOGGER.info(
            "Applying production strategy adoption config_path=%s previous_production_strategy=%s adopted_strategy=%s",
            strategy_config_path,
            current_production_strategy,
            strategy,
        )
        write_strategy_role_config(strategy_config_path, updated_config)
        result["applied"] = True
        result["message"] = f"Adopted {strategy} as the production app strategy in {strategy_config_path}."
    return result


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Adopt a registered text-line segmentation strategy as the production app default."
    )
    parser.add_argument("--strategy", required=True, help="Registered strategy name to use for future app saves.")
    parser.add_argument("--reason", default=None, help="Optional human-readable reason to record in adoption history.")
    parser.add_argument("--apply", action="store_true", help="Write the checked-in strategy role config.")
    parser.add_argument(
        "--strategy-config-path",
        default=str(STRATEGY_ROLE_CONFIG_PATH),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--author-or-tool",
        default="scripts/adopt_text_line_strategy_for_app.py",
        help=argparse.SUPPRESS,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = _build_arg_parser().parse_args(argv)
    try:
        result = adopt_text_line_strategy_for_app(
            strategy=args.strategy,
            reason=args.reason,
            apply=bool(args.apply),
            strategy_config_path=Path(args.strategy_config_path),
            author_or_tool=args.author_or_tool,
        )
    except ValueError as exc:
        print(f"[production-adoption] Refused: {exc}", file=sys.stderr)
        return 1
    before = result["config_before"]
    after = result["config_after"]
    print(f"[production-adoption] Config: {result['config_path']}")
    print(f"[production-adoption] {result['message']}")
    if not args.apply:
        print(
            "[production-adoption] Dry run only. production app strategy: "
            f"{before.get('production_strategy_name')} -> {after.get('production_strategy_name')}"
        )
        print(
            "[production-adoption] Dry run only. research benchmark: "
            f"{before.get('benchmark_strategy_name')} -> {after.get('benchmark_strategy_name')}"
        )
        print(
            "[production-adoption] Dry run only. proposed: "
            f"{before.get('proposed_strategy_name')} -> {after.get('proposed_strategy_name')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
