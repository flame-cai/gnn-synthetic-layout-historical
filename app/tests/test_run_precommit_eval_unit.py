from __future__ import annotations

import json
import os
import shutil
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


TESTS_ROOT = Path(__file__).resolve().parent
APP_ROOT = TESTS_ROOT.parent
REPO_ROOT = APP_ROOT.parent

if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import run_precommit_eval


class RunPrecommitEvalUnitTest(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        tmp_root = TESTS_ROOT / "_tmp_run_precommit_eval_unit"
        if tmp_root.exists():
            shutil.rmtree(tmp_root)

    def setUp(self):
        self.tmp_root = TESTS_ROOT / "_tmp_run_precommit_eval_unit" / self._testMethodName
        if self.tmp_root.exists():
            shutil.rmtree(self.tmp_root)
        self.logs_root = self.tmp_root / "logs"
        self.logs_root.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.logs_root / "gate_latest.json"
        self.phase = run_precommit_eval.PrecommitPhase(
            name="Unit Gate",
            command=["-m", "unittest", "unit_gate"],
            skip_env_var="SKIP_UNIT_GATE",
            artifact_paths=(self.metrics_path,),
        )

    def _write_latest_metrics(
        self,
        *run_dirs: Path,
        summary_dir: Path | None = None,
        plot_paths: tuple[Path | None, ...] = (),
    ) -> None:
        strategy_results = {
            f"role_{index}": {
                key: value
                for key, value in {
                    "run_dir": str(run_dir.resolve()),
                    "plot_path": str(plot_paths[index].resolve())
                    if index < len(plot_paths) and plot_paths[index] is not None
                    else None,
                }.items()
                if value is not None
            }
            for index, run_dir in enumerate(run_dirs)
        }
        dataset_result = {"strategy_results": strategy_results}
        if summary_dir is not None:
            dataset_result["run_dir"] = str(summary_dir.resolve())
        self.metrics_path.write_text(
            json.dumps({"dataset_results": {"eval_dataset": dataset_result}}),
            encoding="utf-8",
        )

    def test_cleanup_switch_defaults_on_and_allows_disable(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(run_precommit_eval.strategy_gate_cleanup_enabled())
        with patch.dict(os.environ, {"CLEAN_UP": "0"}, clear=True):
            self.assertFalse(run_precommit_eval.strategy_gate_cleanup_enabled())
        with patch.dict(os.environ, {"CLEAN_UP": "off"}, clear=True):
            self.assertFalse(run_precommit_eval.strategy_gate_cleanup_enabled())
        with patch.dict(os.environ, {"CLEAN_UP": "1"}, clear=True):
            self.assertTrue(run_precommit_eval.strategy_gate_cleanup_enabled())

    def test_cleanup_deletes_only_role_run_dirs_inside_logs(self):
        benchmark_root = self.logs_root / "benchmark"
        proposed_root = self.logs_root / "proposed"
        benchmark_dir = benchmark_root / "policy" / "slug"
        proposed_dir = proposed_root / "policy" / "slug"
        outside_dir = self.tmp_root / "outside"
        summary_dir = self.logs_root / "summary"
        for artifact_dir in (benchmark_dir, proposed_dir, outside_dir, summary_dir):
            artifact_dir.mkdir(parents=True, exist_ok=True)
            (artifact_dir / "marker.txt").write_text("keep or delete\n", encoding="utf-8")
        self._write_latest_metrics(benchmark_dir, proposed_dir, outside_dir)

        with patch.object(run_precommit_eval, "LOGS_DIR", self.logs_root):
            deleted = run_precommit_eval.cleanup_passing_phase_role_runs(self.phase)

        self.assertEqual({path.name for path in deleted}, {"benchmark", "proposed"})
        self.assertFalse(benchmark_root.exists())
        self.assertFalse(proposed_root.exists())
        self.assertTrue(outside_dir.exists())
        self.assertTrue(summary_dir.exists())

    def test_cleanup_preserves_role_plots_in_summary_run_dir(self):
        benchmark_root = self.logs_root / "benchmark"
        proposed_root = self.logs_root / "proposed"
        benchmark_dir = benchmark_root / "policy" / "slug"
        proposed_dir = proposed_root / "policy" / "slug"
        summary_dir = self.logs_root / "summary"
        benchmark_plot = benchmark_dir / "plots" / "page_cer_vs_finetune_pages.png"
        proposed_plot = proposed_dir / "plots" / "page_cer_vs_finetune_pages.png"
        for plot_path, content in ((benchmark_plot, "benchmark plot\n"), (proposed_plot, "proposed plot\n")):
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            plot_path.write_text(content, encoding="utf-8")
        summary_dir.mkdir(parents=True, exist_ok=True)
        self._write_latest_metrics(
            benchmark_dir,
            proposed_dir,
            summary_dir=summary_dir,
            plot_paths=(benchmark_plot, proposed_plot),
        )

        with patch.object(run_precommit_eval, "LOGS_DIR", self.logs_root):
            deleted = run_precommit_eval.cleanup_passing_phase_role_runs(self.phase)

        self.assertEqual({path.name for path in deleted}, {"benchmark", "proposed"})
        self.assertFalse(benchmark_root.exists())
        self.assertFalse(proposed_root.exists())
        self.assertEqual(
            (summary_dir / "plots" / "eval_dataset_role_0" / "page_cer_vs_finetune_pages.png").read_text(
                encoding="utf-8"
            ),
            "benchmark plot\n",
        )
        self.assertEqual(
            (summary_dir / "plots" / "eval_dataset_role_1" / "page_cer_vs_finetune_pages.png").read_text(
                encoding="utf-8"
            ),
            "proposed plot\n",
        )

    def test_failed_phase_keeps_role_run_dirs(self):
        role_dir = self.logs_root / "benchmark"
        role_dir.mkdir(parents=True, exist_ok=True)
        self._write_latest_metrics(role_dir)

        failed_result = type("Result", (), {"returncode": 1})()
        with (
            patch.object(run_precommit_eval, "LOGS_DIR", self.logs_root),
            patch.object(run_precommit_eval.subprocess, "run", return_value=failed_result),
        ):
            returncode = run_precommit_eval._run_phase(["python"], self.phase)

        self.assertEqual(returncode, 1)
        self.assertTrue(role_dir.exists())

    def test_passing_phase_keeps_role_run_dirs_when_cleanup_is_disabled(self):
        role_dir = self.logs_root / "benchmark"
        role_dir.mkdir(parents=True, exist_ok=True)
        self._write_latest_metrics(role_dir)

        passed_result = type("Result", (), {"returncode": 0})()
        with (
            patch.dict(os.environ, {"CLEAN_UP": "0"}),
            patch.object(run_precommit_eval, "LOGS_DIR", self.logs_root),
            patch.object(run_precommit_eval.subprocess, "run", return_value=passed_result),
        ):
            returncode = run_precommit_eval._run_phase(["python"], self.phase)

        self.assertEqual(returncode, 0)
        self.assertTrue(role_dir.exists())


if __name__ == "__main__":
    unittest.main()
