import shutil
import sys
import unittest
from unittest import mock
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

    def _registry_for(self, name):
        root = TESTS_ROOT / "_tmp_registry_unit" / name
        root.mkdir(parents=True, exist_ok=True)
        base_checkpoint = root / "base.pth"
        base_checkpoint.write_text("base", encoding="utf-8")
        return load_registry(root / "manuscript", base_checkpoint)

    def test_imported_transcription_without_registry_history_counts_as_annotated(self):
        # dense_layout / moderate_layout / circular_layout: human transcription in
        # PAGE XML, never reviewed in the tool, so no revisions and no predictions.
        registry = self._registry_for("imported_transcription")
        self.assertTrue(
            registry.has_read_mode_annotations("233_0001", {"0": "rama", "1": "sita"})
        )

    def test_unedited_prediction_does_not_count_as_annotated(self):
        registry = self._registry_for("prediction_only")
        registry.remember_prediction(
            "233_0001",
            {"recognition_engine": "local", "predicted_lines": {"0": "rama", "1": "sita"}},
        )

        self.assertFalse(
            registry.has_read_mode_annotations("233_0001", {"0": "rama", "1": "sita"})
        )
        # Empty lines on either side must not create a spurious difference.
        self.assertFalse(
            registry.has_read_mode_annotations("233_0001", {"0": "rama", "1": "sita", "2": "  "})
        )
        # One corrected line is enough to protect the page.
        self.assertTrue(
            registry.has_read_mode_annotations("233_0001", {"0": "rAma", "1": "sita"})
        )

    def test_reviewed_page_counts_even_when_text_matches_the_prediction(self):
        registry = self._registry_for("reviewed_unchanged")
        registry.remember_prediction(
            "233_0001",
            {"recognition_engine": "local", "predicted_lines": {"0": "rama"}},
        )
        registry.record_page_revision(
            "233_0001",
            {
                "content_hash": "gt-1",
                "save_intent": "commit",
                "supervision_present": True,
                "recognition_engine": "local",
                "text_line_count": 1,
                "text_non_empty_line_count": 1,
            },
        )

        self.assertTrue(registry.has_read_mode_annotations("233_0001", {"0": "rama"}))

    def test_page_without_text_is_never_annotated(self):
        registry = self._registry_for("no_text")
        self.assertFalse(registry.has_read_mode_annotations("233_0001", {}))
        self.assertFalse(registry.has_read_mode_annotations("233_0001", {"0": "", "1": "   "}))

    def test_has_text_review_revision_distinguishes_layout_saves_from_text_work(self):
        root = TESTS_ROOT / "_tmp_registry_unit" / "text_review_history"
        root.mkdir(parents=True, exist_ok=True)
        base_checkpoint = root / "base.pth"
        base_checkpoint.write_text("base", encoding="utf-8")
        registry = load_registry(root / "manuscript", base_checkpoint)

        # A Layout Mode save: a commit that carries no OCR supervision.
        registry.record_page_revision(
            "233_0001",
            {
                "content_hash": "layout-1",
                "save_intent": "commit",
                "supervision_present": False,
                "recognition_engine": "local",
                "text_line_count": 0,
                "text_non_empty_line_count": 0,
            },
        )
        self.assertFalse(registry.has_text_review_revision("233_0001"))

        # A Text Review autosave is enough to count as human text work.
        registry.record_page_revision(
            "233_0001",
            {
                "content_hash": "draft-1",
                "save_intent": "draft",
                "supervision_present": False,
                "recognition_engine": "local",
                "text_line_count": 2,
                "text_non_empty_line_count": 1,
            },
        )
        self.assertTrue(registry.has_text_review_revision("233_0001"))

        # A reviewed commit on a different page counts too.
        registry.record_page_revision(
            "233_0002",
            {
                "content_hash": "gt-1",
                "save_intent": "commit",
                "supervision_present": True,
                "recognition_engine": "local",
                "text_line_count": 2,
                "text_non_empty_line_count": 2,
            },
        )
        self.assertTrue(registry.has_text_review_revision("233_0002"))
        self.assertFalse(registry.has_text_review_revision("233_0003"))

    def test_commit_after_same_hash_draft_creates_new_revision(self):
        root = TESTS_ROOT / "_tmp_registry_unit" / "draft_then_commit"
        root.mkdir(parents=True, exist_ok=True)
        base_checkpoint = root / "base.pth"
        base_checkpoint.write_text("base", encoding="utf-8")
        registry = load_registry(root / "manuscript", base_checkpoint)

        draft = registry.record_page_revision(
            "233_0001",
            {
                "content_hash": "abc123",
                "save_intent": "draft",
                "supervision_present": True,
                "recognition_engine": "local",
                "text_line_count": 2,
                "text_non_empty_line_count": 2,
            },
        )
        commit = registry.record_page_revision(
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

        self.assertEqual(draft.revision_number, 1)
        self.assertFalse(draft.is_duplicate)
        self.assertEqual(commit.revision_number, 2)
        self.assertFalse(commit.is_duplicate)
        self.assertEqual(len(registry.data["page_revisions"]["233_0001"]), 2)
        self.assertEqual(registry.data["page_revisions"]["233_0001"][-1]["save_intent"], "commit")

    def test_commit_after_reverted_drafts_deduplicates_against_latest_commit(self):
        root = TESTS_ROOT / "_tmp_registry_unit" / "reverted_drafts"
        root.mkdir(parents=True, exist_ok=True)
        base_checkpoint = root / "base.pth"
        base_checkpoint.write_text("base", encoding="utf-8")
        registry = load_registry(root / "manuscript", base_checkpoint)

        initial_commit = registry.record_page_revision(
            "233_0001",
            {
                "content_hash": "commit_hash",
                "save_intent": "commit",
                "supervision_present": True,
                "recognition_engine": "local",
                "text_line_count": 2,
                "text_non_empty_line_count": 2,
            },
        )
        changed_draft = registry.record_page_revision(
            "233_0001",
            {
                "content_hash": "draft_hash",
                "save_intent": "draft",
                "supervision_present": True,
                "recognition_engine": "local",
                "text_line_count": 2,
                "text_non_empty_line_count": 2,
            },
        )
        reverted_draft = registry.record_page_revision(
            "233_0001",
            {
                "content_hash": "commit_hash",
                "save_intent": "draft",
                "supervision_present": True,
                "recognition_engine": "local",
                "text_line_count": 2,
                "text_non_empty_line_count": 2,
            },
        )
        repeated_commit = registry.record_page_revision(
            "233_0001",
            {
                "content_hash": "commit_hash",
                "save_intent": "commit",
                "supervision_present": True,
                "recognition_engine": "local",
                "text_line_count": 2,
                "text_non_empty_line_count": 2,
            },
        )

        self.assertEqual(initial_commit.revision_number, 1)
        self.assertFalse(changed_draft.is_duplicate)
        self.assertFalse(reverted_draft.is_duplicate)
        self.assertEqual(repeated_commit.revision_number, 1)
        self.assertTrue(repeated_commit.is_duplicate)
        self.assertEqual(len(registry.data["page_revisions"]["233_0001"]), 3)

    def test_supervised_revision_queries_handle_mixed_page_id_prefixes(self):
        root = TESTS_ROOT / "_tmp_registry_unit" / "mixed_page_ids"
        root.mkdir(parents=True, exist_ok=True)
        base_checkpoint = root / "base.pth"
        base_checkpoint.write_text("base", encoding="utf-8")
        registry = load_registry(root / "manuscript", base_checkpoint)

        for page_id in ("2", "10", "a", "b"):
            revision = registry.record_page_revision(
                page_id,
                {
                    "content_hash": f"hash-{page_id}",
                    "save_intent": "commit",
                    "supervision_present": True,
                    "recognition_engine": "local",
                    "text_line_count": 1,
                    "text_non_empty_line_count": 1,
                },
            )
            registry.mark_revision_consumed(page_id, revision.revision_number, f"ckpt-{page_id}")

        latest = registry.latest_supervised_commit_revisions()
        approved = registry.approved_supervised_revisions()

        self.assertEqual([revision["page_id"] for revision in latest], ["2", "10", "a", "b"])
        self.assertEqual([revision["page_id"] for revision in approved], ["2", "10", "a", "b"])

    def test_save_retries_when_atomic_replace_hits_temporary_windows_lock(self):
        root = TESTS_ROOT / "_tmp_registry_unit" / "atomic_retry"
        root.mkdir(parents=True, exist_ok=True)
        base_checkpoint = root / "base.pth"
        base_checkpoint.write_text("base", encoding="utf-8")
        registry = load_registry(root / "manuscript", base_checkpoint)
        registry.set_status("queued", "AL: queued")

        path_type = type(registry.registry_path)
        original_replace = path_type.replace
        replace_attempts = {"count": 0}

        def flaky_replace(path_self, target):
            replace_attempts["count"] += 1
            if replace_attempts["count"] == 1:
                raise PermissionError("[WinError 32] simulated temporary file lock")
            return original_replace(path_self, target)

        registry.data["status"] = {
            "code": "running",
            "label": "AL: running",
            "updated_at": "2026-04-20T00:00:00+00:00",
            "details": {"job_id": "job-1"},
        }
        with mock.patch.object(path_type, "replace", new=flaky_replace):
            with mock.patch("manuscript_ocr_registry.time.sleep", return_value=None):
                registry.save()

        self.assertEqual(replace_attempts["count"], 2)
        reloaded = load_registry(root / "manuscript", base_checkpoint)
        self.assertEqual(reloaded.get_status()["code"], "running")

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
