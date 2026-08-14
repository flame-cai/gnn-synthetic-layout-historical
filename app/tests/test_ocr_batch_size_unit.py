"""The OCR fine-tuning batch size must not depend on the host's GPU count.

`app/recognition/train.py` used to scale `batch_size` (and `workers`) by
`torch.cuda.device_count()`. That made the effective batch size a property of
the machine rather than of the recipe: the OCR active-learning recipe is
calibrated at `batch_size=1`, and the research ablations ran that way under
`CUDA_VISIBLE_DEVICES=0`, but the GUI server sees every GPU, so the same
manuscript trained at batch 3 on a 3-GPU box and batch 1 on a laptop.

These tests pin the configured value through both the option builder and the
`train()` setup path, at several simulated GPU counts.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


TESTS_ROOT = Path(__file__).resolve().parent
APP_ROOT = TESTS_ROOT.parent
REPO_ROOT = APP_ROOT.parent

for path in (str(APP_ROOT), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from recognition.active_learning import _build_finetune_options
from recognition.train import _seed_everything


class OcrBatchSizeUnitTest(unittest.TestCase):
    def _options(self, **overrides):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            return _build_finetune_options(
                base_checkpoint=root / "base.pth",
                lmdb_root=root / "lmdb",
                experiment_dir=root / "run",
                **overrides,
            )

    def test_default_finetune_batch_size_is_one(self):
        self.assertEqual(self._options().batch_size, 1)

    def test_explicit_batch_size_override_is_respected(self):
        self.assertEqual(self._options(batch_size=4).batch_size, 4)

    def test_batch_size_is_unchanged_by_the_number_of_visible_gpus(self):
        for device_count in (0, 1, 2, 3, 8):
            with self.subTest(num_gpu=device_count):
                opt = self._options()
                self.assertEqual(opt.batch_size, 1)
                with mock.patch("torch.cuda.device_count", return_value=device_count), \
                        mock.patch("torch.cuda.is_available", return_value=device_count > 0), \
                        mock.patch("torch.cuda.manual_seed"):
                    _seed_everything(opt)
                self.assertEqual(
                    opt.batch_size,
                    1,
                    f"batch_size must stay 1 with {device_count} visible GPUs; "
                    "the recipe pins it and scaling it changes the effective step size",
                )
                self.assertEqual(opt.num_gpu, device_count)

    def test_a_non_default_batch_size_also_survives_multi_gpu(self):
        opt = self._options(batch_size=2)
        with mock.patch("torch.cuda.device_count", return_value=4), \
                mock.patch("torch.cuda.is_available", return_value=True), \
                mock.patch("torch.cuda.manual_seed"):
            _seed_everything(opt)
        self.assertEqual(opt.batch_size, 2)


if __name__ == "__main__":
    unittest.main()
