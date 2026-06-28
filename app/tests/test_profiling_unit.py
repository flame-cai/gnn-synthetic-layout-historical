import json
import os
import shutil
import sys
import unittest
from pathlib import Path
from unittest import mock


TESTS_ROOT = Path(__file__).resolve().parent
APP_ROOT = TESTS_ROOT.parent

if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from profiling import (
    create_layout_save_timing_recorder,
    layout_save_timing_enabled,
    should_capture_cuda_trace,
    summarize_gpu_job,
)


class ProfilingUnitTest(unittest.TestCase):
    def setUp(self):
        self.tmp_root = TESTS_ROOT / "_tmp_profiling_unit"
        if self.tmp_root.exists():
            shutil.rmtree(self.tmp_root)
        self.tmp_root.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        if self.tmp_root.exists():
            shutil.rmtree(self.tmp_root)

    def test_cuda_trace_capture_is_opt_in_and_sampled_once(self):
        profiling_root = self.tmp_root / "profiling"
        marker = profiling_root / "ocr_fine_tune_trace_seen.marker"

        with mock.patch.dict(os.environ, {"ACTIVE_LEARNING_PROFILE_CUDA": ""}):
            self.assertFalse(should_capture_cuda_trace("ocr_fine_tune", profiling_root))
            self.assertFalse(marker.exists())

        with mock.patch.dict(os.environ, {"ACTIVE_LEARNING_PROFILE_CUDA": "1"}):
            self.assertTrue(should_capture_cuda_trace("ocr_fine_tune", profiling_root))
            self.assertTrue(marker.exists())
            self.assertFalse(should_capture_cuda_trace("ocr_fine_tune", profiling_root))

    def test_gpu_summary_records_start_and_finish_times(self):
        result, summary = summarize_gpu_job("unit", {}, lambda: "ok")

        self.assertEqual(result, "ok")
        self.assertIn("started_at", summary)
        self.assertIn("finished_at", summary)
        self.assertGreaterEqual(summary["wall_time_seconds"], 0.0)

    def test_layout_save_timing_is_disabled_by_default(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(layout_save_timing_enabled())
            recorder = create_layout_save_timing_recorder(self.tmp_root, page_id="page_1")
            with recorder.chunk("noop"):
                pass
            self.assertIsNone(recorder.finish())
            self.assertFalse((self.tmp_root / "layout_analysis_output" / "profiling").exists())

    def test_layout_save_timing_writes_chunked_jsonl_when_enabled(self):
        output_path = self.tmp_root / "custom_timings.jsonl"
        with mock.patch.dict(os.environ, {"LAYOUT_SAVE_TIMING_ENABLED": "1"}):
            recorder = create_layout_save_timing_recorder(
                self.tmp_root,
                page_id="page_1",
                config={"layout_save_timing_log_path": str(output_path)},
                metadata={"pipeline_stage": "layout_save_to_page_xml"},
            )
            with recorder.chunk("write_baseline_page_xml", {"line_count": 2}):
                pass
            recorder.finish("success", {"line_count": 2})

        payload = json.loads(output_path.read_text(encoding="utf-8").strip())
        self.assertEqual(payload["event_type"], "layout_save_timing")
        self.assertEqual(payload["metadata"]["page_id"], "page_1")
        self.assertEqual(payload["metadata"]["line_count"], 2)
        self.assertEqual(payload["chunks"][0]["name"], "write_baseline_page_xml")
        self.assertEqual(payload["chunks"][0]["status"], "success")
        self.assertGreaterEqual(payload["chunks"][0]["duration_seconds"], 0.0)


if __name__ == "__main__":
    unittest.main()
