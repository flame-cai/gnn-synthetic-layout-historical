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

from profiling import should_capture_cuda_trace, summarize_gpu_job


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


if __name__ == "__main__":
    unittest.main()
