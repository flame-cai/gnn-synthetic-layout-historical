from __future__ import annotations

import json
import re
import shutil
import sys
import unittest
from pathlib import Path


TESTS_ROOT = Path(__file__).resolve().parent
APP_ROOT = TESTS_ROOT.parent
REPO_ROOT = APP_ROOT.parent

if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recognition.line_segmentation.strategy_config import write_strategy_role_config
from scripts.adopt_text_line_strategy_for_app import adopt_text_line_strategy_for_app


def _load_config_payload(path: Path) -> dict:
    source = path.read_text(encoding="utf-8")
    match = re.search(r'STRATEGY_ROLE_CONFIG_JSON = """([\s\S]*?)"""', source)
    if match is None:
        match = re.search(r"STRATEGY_ROLE_CONFIG_JSON = '''([\s\S]*?)'''", source)
    if match is None:
        raise AssertionError(f"Could not locate STRATEGY_ROLE_CONFIG_JSON in {path}")
    return json.loads(match.group(1))


class StrategyAdoptionUnitTest(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        tmp_root = TESTS_ROOT / "_tmp_strategy_adoption_unit"
        if tmp_root.exists():
            shutil.rmtree(tmp_root)

    def setUp(self):
        self.tmp_root = TESTS_ROOT / "_tmp_strategy_adoption_unit" / self._testMethodName
        if self.tmp_root.exists():
            shutil.rmtree(self.tmp_root)
        self.tmp_root.mkdir(parents=True, exist_ok=True)
        self.config_path = self.tmp_root / "strategy_config.py"
        write_strategy_role_config(
            self.config_path,
            {
                "benchmark_strategy_name": "legacy_axis_bound_v1",
                "proposed_strategy_name": "local_tangent_band_v1",
                "production_strategy_name": "legacy_axis_bound_v1",
                "research_promotion_history": [],
                "production_adoption_history": [],
            },
        )

    def test_dry_run_changes_only_production_fields(self):
        before = self.config_path.read_text(encoding="utf-8")

        result = adopt_text_line_strategy_for_app(
            strategy="local_tangent_band_v1",
            reason="manual app rollout after review",
            apply=False,
            strategy_config_path=self.config_path,
        )

        self.assertTrue(result["changed"])
        self.assertFalse(result["applied"])
        self.assertEqual(before, self.config_path.read_text(encoding="utf-8"))
        self.assertEqual(result["config_after"]["benchmark_strategy_name"], "legacy_axis_bound_v1")
        self.assertEqual(result["config_after"]["proposed_strategy_name"], "local_tangent_band_v1")
        self.assertEqual(result["config_after"]["production_strategy_name"], "local_tangent_band_v1")
        self.assertEqual(len(result["config_after"]["production_adoption_history"]), 1)
        self.assertEqual(
            result["config_after"]["production_adoption_history"][0]["reason"],
            "manual app rollout after review",
        )
        self.assertEqual(result["config_after"]["research_promotion_history"], [])

    def test_apply_updates_only_production_strategy_and_history(self):
        result = adopt_text_line_strategy_for_app(
            strategy="local_tangent_band_v1",
            reason="operator accepted rollout",
            apply=True,
            strategy_config_path=self.config_path,
        )

        self.assertTrue(result["changed"])
        self.assertTrue(result["applied"])
        payload = _load_config_payload(self.config_path)
        self.assertEqual(payload["benchmark_strategy_name"], "legacy_axis_bound_v1")
        self.assertEqual(payload["proposed_strategy_name"], "local_tangent_band_v1")
        self.assertEqual(payload["research_promotion_history"], [])
        self.assertEqual(payload["production_strategy_name"], "local_tangent_band_v1")
        self.assertEqual(len(payload["production_adoption_history"]), 1)
        history_entry = payload["production_adoption_history"][0]
        self.assertEqual(history_entry["adopted_strategy_name"], "local_tangent_band_v1")
        self.assertEqual(history_entry["previous_production_strategy_name"], "legacy_axis_bound_v1")
        self.assertEqual(history_entry["author_or_tool"], "scripts/adopt_text_line_strategy_for_app.py")
        self.assertEqual(history_entry["reason"], "operator accepted rollout")

    def test_idempotent_when_strategy_already_adopted(self):
        first = adopt_text_line_strategy_for_app(
            strategy="local_tangent_band_v1",
            apply=True,
            strategy_config_path=self.config_path,
        )
        second = adopt_text_line_strategy_for_app(
            strategy="local_tangent_band_v1",
            reason="same strategy again",
            apply=True,
            strategy_config_path=self.config_path,
        )

        self.assertTrue(first["applied"])
        self.assertFalse(second["changed"])
        self.assertTrue(second["idempotent"])
        payload = _load_config_payload(self.config_path)
        self.assertEqual(payload["production_strategy_name"], "local_tangent_band_v1")
        self.assertEqual(len(payload["production_adoption_history"]), 1)

    def test_rejects_unknown_strategy(self):
        with self.assertRaisesRegex(ValueError, "is not registered"):
            adopt_text_line_strategy_for_app(
                strategy="missing_strategy_v1",
                apply=False,
                strategy_config_path=self.config_path,
            )


if __name__ == "__main__":
    unittest.main()
