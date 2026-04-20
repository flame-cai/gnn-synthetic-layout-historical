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

from manuscript_ocr_registry import load_registry


class ManuscriptOcrRegistryUnitTest(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        tmp_root = TESTS_ROOT / "_tmp_registry_unit"
        if tmp_root.exists():
            shutil.rmtree(tmp_root)

    def test_record_page_revision_deduplicates_same_hash(self):
        root = TESTS_ROOT / "_tmp_registry_unit" / "dedupe"
        root.mkdir(parents=True, exist_ok=True)
        base_checkpoint = root / "base.pth"
        base_checkpoint.write_text("base", encoding="utf-8")
        registry = load_registry(root / "manuscript", base_checkpoint)

        first = registry.record_page_revision(
            "233_0001",
            {
                "content_hash": "abc123",
                "save_intent": "commit",
                "supervision_present": True,
                "recognition_engine": "local",
                "text_line_count": 2,
                "text_non_empty_line_count": 2,
            },
        )
        second = registry.record_page_revision(
            "233_0001",
            {
                "content_hash": "abc123",
                "save_intent": "commit",
                "supervision_present": True,
                "recognition_engine": "local",
                "text_line_count": 2,
                "text_non_empty_line_count": 2,
            },
        )

        self.assertEqual(first.revision_number, 1)
        self.assertFalse(first.is_duplicate)
        self.assertEqual(second.revision_number, 1)
        self.assertTrue(second.is_duplicate)
        self.assertEqual(len(registry.data["page_revisions"]["233_0001"]), 1)

    def test_active_checkpoint_falls_back_to_previous_when_active_is_missing(self):
        root = TESTS_ROOT / "_tmp_registry_unit" / "fallback"
        root.mkdir(parents=True, exist_ok=True)
        base_checkpoint = root / "base.pth"
        previous_checkpoint = root / "prev.pth"
        base_checkpoint.write_text("base", encoding="utf-8")
        previous_checkpoint.write_text("prev", encoding="utf-8")
        registry = load_registry(root / "manuscript", base_checkpoint)
        registry.ensure_checkpoint_record("prev", previous_checkpoint, status="active")
        registry.ensure_checkpoint_record("missing", root / "missing.pth", status="active")
        registry.data["previous_active_checkpoint_id"] = "prev"
        registry.data["active_checkpoint_id"] = "missing"
        registry.save()

        resolved = registry.active_checkpoint()

        self.assertEqual(resolved, previous_checkpoint.resolve())
        self.assertEqual(registry.active_checkpoint_id(), "prev")
        self.assertEqual(registry.data["last_checkpoint_fallback"]["to_checkpoint_id"], "prev")


if __name__ == "__main__":
    unittest.main()
