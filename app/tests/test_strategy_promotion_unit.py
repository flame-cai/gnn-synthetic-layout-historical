from __future__ import annotations

import json
import os
import re
import shutil
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path


TESTS_ROOT = Path(__file__).resolve().parent
APP_ROOT = TESTS_ROOT.parent
REPO_ROOT = APP_ROOT.parent

if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recognition.line_segmentation.strategy_config import write_strategy_role_config
from scripts.promote_text_line_strategy import (
    promote_text_line_strategy,
    write_checked_in_strategy_promotion_record,
)


def _mtime_iso(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00",
        "Z",
    )


def _load_config_payload(path: Path) -> dict:
    source = path.read_text(encoding="utf-8")
    match = re.search(r'STRATEGY_ROLE_CONFIG_JSON = """([\s\S]*?)"""', source)
    if match is None:
        match = re.search(r"STRATEGY_ROLE_CONFIG_JSON = '''([\s\S]*?)'''", source)
    if match is None:
        raise AssertionError(f"Could not locate STRATEGY_ROLE_CONFIG_JSON in {path}")
    return json.loads(match.group(1))


class StrategyPromotionUnitTest(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        tmp_root = TESTS_ROOT / "_tmp_strategy_promotion_unit"
        if tmp_root.exists():
            shutil.rmtree(tmp_root)

    def setUp(self):
        self.tmp_root = TESTS_ROOT / "_tmp_strategy_promotion_unit" / self._testMethodName
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

    def _write_evidence(
        self,
        *,
        benchmark_strategy_name: str = "legacy_axis_bound_v1",
        proposed_strategy_name: str = "local_tangent_band_v1",
        all_passed: bool = True,
    ) -> Path:
        gate_artifacts = self.tmp_root / "gate_artifacts"
        gate_artifacts.mkdir(parents=True, exist_ok=True)

        gate_results = {}
        for gate_key, metric_name, benchmark_value, proposed_value, operator in (
            ("pipeline_eval_dataset", "page_cer", 0.20, 0.19, "<="),
            ("ocr_eval_dataset", "curve_metric_value", 0.24, 0.22, "<="),
            ("circular_ocr_eval_dataset_v2", "curve_metric_value", 0.95, 0.18, "<"),
        ):
            metrics_path = gate_artifacts / f"{gate_key}.json"
            summary_path = gate_artifacts / f"{gate_key}.md"
            run_metrics_path = gate_artifacts / f"{gate_key}_run.json"
            run_summary_path = gate_artifacts / f"{gate_key}_run.md"
            metrics_path.write_text('{"ok": true}\n', encoding="utf-8")
            summary_path.write_text(f"# {gate_key}\n", encoding="utf-8")
            run_metrics_path.write_text('{"ok": true}\n', encoding="utf-8")
            run_summary_path.write_text(f"# {gate_key} run\n", encoding="utf-8")
            passed = all_passed
            gate_results[gate_key] = {
                "passed": passed,
                "failure_message": "" if passed else f"{gate_key} failed",
                "primary_metric_name": metric_name,
                "benchmark_value": benchmark_value,
                "proposed_value": proposed_value,
                "operator": operator,
                "benchmark_strategy_name": benchmark_strategy_name,
                "proposed_strategy_name": proposed_strategy_name,
                "artifact_paths": {
                    "latest_metrics_json": str(metrics_path.resolve()),
                    "latest_summary_md": str(summary_path.resolve()),
                    "run_metrics_json": str(run_metrics_path.resolve()),
                    "run_summary_md": str(run_summary_path.resolve()),
                },
                "artifact_mtime_utc": {
                    "latest_metrics_json": _mtime_iso(metrics_path),
                    "latest_summary_md": _mtime_iso(summary_path),
                    "run_metrics_json": _mtime_iso(run_metrics_path),
                    "run_summary_md": _mtime_iso(run_summary_path),
                },
            }

        evidence_path = self.tmp_root / "strategy_promotion_latest.json"
        evidence = {
            "study_mode": "strategy_promotion_evidence",
            "generated_at_utc": "2026-05-12T00:00:00Z",
            "benchmark_strategy_name": benchmark_strategy_name,
            "proposed_strategy_name": proposed_strategy_name,
            "promotion_recommended": all_passed,
            "promotion_blockers": [] if all_passed else ["one or more gates failed"],
            "gate_results": gate_results,
        }
        evidence_path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
        return evidence_path

    def test_dry_run_does_not_modify_config(self):
        evidence_path = self._write_evidence()
        before = self.config_path.read_text(encoding="utf-8")

        result = promote_text_line_strategy(
            candidate="local_tangent_band_v1",
            previous_benchmark="legacy_axis_bound_v1",
            metrics_path=evidence_path,
            apply=False,
            strategy_config_path=self.config_path,
        )

        after = self.config_path.read_text(encoding="utf-8")
        self.assertTrue(result["changed"])
        self.assertFalse(result["applied"])
        self.assertEqual(before, after)
        self.assertEqual(result["config_after"]["benchmark_strategy_name"], "local_tangent_band_v1")
        self.assertIsNone(result["config_after"]["proposed_strategy_name"])
        self.assertEqual(result["config_after"]["production_strategy_name"], "legacy_axis_bound_v1")
        self.assertEqual(result["config_after"]["production_adoption_history"], [])

    def test_apply_updates_config_when_all_gates_pass(self):
        evidence_path = self._write_evidence()

        result = promote_text_line_strategy(
            candidate="local_tangent_band_v1",
            previous_benchmark="legacy_axis_bound_v1",
            metrics_path=evidence_path,
            apply=True,
            strategy_config_path=self.config_path,
        )

        self.assertTrue(result["changed"])
        self.assertTrue(result["applied"])
        payload = _load_config_payload(self.config_path)
        self.assertEqual(payload["benchmark_strategy_name"], "local_tangent_band_v1")
        self.assertIsNone(payload["proposed_strategy_name"])
        self.assertEqual(payload["production_strategy_name"], "legacy_axis_bound_v1")
        self.assertEqual(payload["production_adoption_history"], [])
        self.assertEqual(len(payload["research_promotion_history"]), 1)
        self.assertEqual(payload["research_promotion_history"][0]["promoted_strategy_name"], "local_tangent_band_v1")

    def test_checked_in_promotion_record_is_rendered_from_evidence(self):
        evidence_path = self._write_evidence()
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        record_path = self.tmp_root / "docs" / "strategy-promotion-record.md"

        written_path = write_checked_in_strategy_promotion_record(evidence, record_path)

        self.assertEqual(written_path, record_path)
        record = record_path.read_text(encoding="utf-8")
        self.assertIn("Text-Line Strategy Promotion Record", record)
        self.assertIn("`legacy_axis_bound_v1`", record)
        self.assertIn("`local_tangent_band_v1`", record)
        self.assertIn("`pipeline_eval_dataset`", record)
        self.assertIn("`ocr_eval_dataset`", record)
        self.assertIn("`circular_ocr_eval_dataset_v2`", record)
        self.assertIn("`app/tests/logs/` is ignored", record)

    def test_apply_is_idempotent_and_does_not_duplicate_history(self):
        evidence_path = self._write_evidence()

        first = promote_text_line_strategy(
            candidate="local_tangent_band_v1",
            previous_benchmark="legacy_axis_bound_v1",
            metrics_path=evidence_path,
            apply=True,
            strategy_config_path=self.config_path,
        )
        second = promote_text_line_strategy(
            candidate="local_tangent_band_v1",
            previous_benchmark="legacy_axis_bound_v1",
            metrics_path=evidence_path,
            apply=True,
            strategy_config_path=self.config_path,
        )

        self.assertTrue(first["applied"])
        self.assertFalse(second["changed"])
        self.assertTrue(second["idempotent"])
        payload = _load_config_payload(self.config_path)
        self.assertEqual(len(payload["research_promotion_history"]), 1)

    def test_apply_preserves_existing_production_adoption_state(self):
        write_strategy_role_config(
            self.config_path,
            {
                "benchmark_strategy_name": "legacy_axis_bound_v1",
                "proposed_strategy_name": "local_tangent_band_v1",
                "production_strategy_name": "legacy_axis_bound_v1",
                "research_promotion_history": [],
                "production_adoption_history": [
                    {
                        "adopted_strategy_name": "legacy_axis_bound_v1",
                        "previous_production_strategy_name": "legacy_axis_bound_v1",
                        "adoption_timestamp_utc": "2026-05-12T00:00:00Z",
                        "author_or_tool": "unit-test",
                        "reason": "existing production state",
                    }
                ],
            },
        )
        evidence_path = self._write_evidence()

        promote_text_line_strategy(
            candidate="local_tangent_band_v1",
            previous_benchmark="legacy_axis_bound_v1",
            metrics_path=evidence_path,
            apply=True,
            strategy_config_path=self.config_path,
        )

        payload = _load_config_payload(self.config_path)
        self.assertEqual(payload["benchmark_strategy_name"], "local_tangent_band_v1")
        self.assertEqual(payload["production_strategy_name"], "legacy_axis_bound_v1")
        self.assertEqual(len(payload["research_promotion_history"]), 1)
        self.assertEqual(len(payload["production_adoption_history"]), 1)
        self.assertEqual(payload["production_adoption_history"][0]["reason"], "existing production state")

    def test_refuses_when_metrics_file_is_missing(self):
        missing_path = self.tmp_root / "missing.json"

        with self.assertRaisesRegex(ValueError, "Missing promotion evidence"):
            promote_text_line_strategy(
                candidate="local_tangent_band_v1",
                previous_benchmark="legacy_axis_bound_v1",
                metrics_path=missing_path,
                apply=False,
                strategy_config_path=self.config_path,
            )

    def test_refuses_when_any_gate_failed(self):
        evidence_path = self._write_evidence(all_passed=False)

        with self.assertRaisesRegex(ValueError, "did not recommend promotion"):
            promote_text_line_strategy(
                candidate="local_tangent_band_v1",
                previous_benchmark="legacy_axis_bound_v1",
                metrics_path=evidence_path,
                apply=False,
                strategy_config_path=self.config_path,
            )

    def test_refuses_when_evidence_is_stale(self):
        evidence_path = self._write_evidence()
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        stale_artifact = Path(
            evidence["gate_results"]["pipeline_eval_dataset"]["artifact_paths"]["latest_metrics_json"]
        )
        stale_mtime = stale_artifact.stat().st_mtime + 5.0
        os.utime(stale_artifact, (stale_mtime, stale_mtime))

        with self.assertRaisesRegex(ValueError, "evidence was stale"):
            promote_text_line_strategy(
                candidate="local_tangent_band_v1",
                previous_benchmark="legacy_axis_bound_v1",
                metrics_path=evidence_path,
                apply=False,
                strategy_config_path=self.config_path,
            )

    def test_refuses_when_metrics_candidate_or_benchmark_mismatch(self):
        wrong_candidate_evidence = self._write_evidence(proposed_strategy_name="legacy_axis_bound_v1")
        with self.assertRaisesRegex(ValueError, "did not match --candidate"):
            promote_text_line_strategy(
                candidate="local_tangent_band_v1",
                previous_benchmark="legacy_axis_bound_v1",
                metrics_path=wrong_candidate_evidence,
                apply=False,
                strategy_config_path=self.config_path,
            )

        wrong_benchmark_evidence = self._write_evidence(benchmark_strategy_name="local_tangent_band_v1")
        with self.assertRaisesRegex(ValueError, "did not match --previous-benchmark"):
            promote_text_line_strategy(
                candidate="local_tangent_band_v1",
                previous_benchmark="legacy_axis_bound_v1",
                metrics_path=wrong_benchmark_evidence,
                apply=False,
                strategy_config_path=self.config_path,
            )

    def test_refuses_when_candidate_is_not_registered(self):
        evidence_path = self._write_evidence(proposed_strategy_name="missing_strategy_v1")

        with self.assertRaisesRegex(ValueError, "is not registered"):
            promote_text_line_strategy(
                candidate="missing_strategy_v1",
                previous_benchmark="legacy_axis_bound_v1",
                metrics_path=evidence_path,
                apply=False,
                strategy_config_path=self.config_path,
            )


if __name__ == "__main__":
    unittest.main()
