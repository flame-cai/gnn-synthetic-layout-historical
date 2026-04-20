import shutil
import sys
import unittest
from unittest import mock
from pathlib import Path

from PIL import Image


TESTS_ROOT = Path(__file__).resolve().parent
APP_ROOT = TESTS_ROOT.parent
REPO_ROOT = APP_ROOT.parent

if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from job_orchestrator import JobType
from manuscript_ocr_registry import load_registry
from ocr_active_learning_runtime import (
    configure_runtime,
    handle_post_save,
    record_prediction,
    rebuild_manuscript_lineage,
    run_ocr_finetune_job,
    summarize_manuscript_active_learning,
    summarize_page_active_learning,
)


class _StubOrchestrator:
    def __init__(self):
        self.jobs = []

    def enqueue(self, job):
        self.jobs.append(job)
        return getattr(job, "job_id", f"job-{len(self.jobs)}")

    def get_job_status(self, job_id):
        return {}


class RecognitionActiveLearningBackendUnitTest(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        tmp_root = TESTS_ROOT / "_tmp_backend_al_unit"
        if tmp_root.exists():
            shutil.rmtree(tmp_root)

    def _make_manuscript_root(self, name: str) -> tuple[Path, Path]:
        manuscript_root = TESTS_ROOT / "_tmp_backend_al_unit" / name
        xml_dir = manuscript_root / "layout_analysis_output" / "page-xml-format"
        image_dir = manuscript_root / "layout_analysis_output" / "images_resized"
        xml_dir.mkdir(parents=True, exist_ok=True)
        image_dir.mkdir(parents=True, exist_ok=True)
        (xml_dir / "233_0001.xml").write_text("<PcGts></PcGts>", encoding="utf-8")
        Image.new("L", (32, 16), color=255).save(image_dir / "233_0001.jpg")
        base_checkpoint = manuscript_root / "base.pth"
        base_checkpoint.write_text("base", encoding="utf-8")
        return manuscript_root, base_checkpoint

    def test_commit_save_with_supervision_enqueues_finetune_job(self):
        manuscript_root, base_checkpoint = self._make_manuscript_root("commit")
        orchestrator = _StubOrchestrator()
        configure_runtime(base_checkpoint, orchestrator=None)

        result = handle_post_save(
            manuscript="commit_manuscript",
            page="233_0001",
            save_intent="commit",
            active_learning_enabled=True,
            recognition_engine="local",
            text_payload={"1": "rama"},
            manuscript_root=manuscript_root,
            base_checkpoint_path=base_checkpoint,
            graph_payload={"nodes": [{"x": 1, "y": 2}], "edges": []},
            textbox_labels=[0],
            modifications=[{"type": "node_add"}],
            orchestrator=orchestrator,
        )

        self.assertEqual(len(orchestrator.jobs), 1)
        self.assertEqual(orchestrator.jobs[0].job_type, JobType.OCR_FINE_TUNE.value)
        self.assertEqual(result["revision"]["revision_number"], 1)
        self.assertTrue(result["entered_active_learning"])

    def test_draft_and_layout_only_saves_do_not_enqueue_ocr_training(self):
        manuscript_root, base_checkpoint = self._make_manuscript_root("draft")
        orchestrator = _StubOrchestrator()
        configure_runtime(base_checkpoint, orchestrator=None)

        draft_result = handle_post_save(
            manuscript="draft_manuscript",
            page="233_0001",
            save_intent="draft",
            active_learning_enabled=True,
            recognition_engine="local",
            text_payload={"1": "rama"},
            manuscript_root=manuscript_root,
            base_checkpoint_path=base_checkpoint,
            graph_payload={"nodes": [{"x": 1, "y": 2}], "edges": []},
            textbox_labels=[0],
            modifications=[],
            orchestrator=orchestrator,
        )
        layout_only_result = handle_post_save(
            manuscript="draft_manuscript",
            page="233_0001",
            save_intent="commit",
            active_learning_enabled=True,
            recognition_engine="local",
            text_payload={},
            manuscript_root=manuscript_root,
            base_checkpoint_path=base_checkpoint,
            graph_payload={"nodes": [{"x": 1, "y": 2}], "edges": []},
            textbox_labels=[0],
            modifications=[],
            orchestrator=orchestrator,
        )

        self.assertEqual(orchestrator.jobs, [])
        self.assertFalse(draft_result["entered_active_learning"])
        self.assertFalse(layout_only_result["entered_active_learning"])

    def test_commit_after_same_hash_draft_enqueues_finetune_job(self):
        manuscript_root, base_checkpoint = self._make_manuscript_root("draft_then_commit")
        orchestrator = _StubOrchestrator()
        configure_runtime(base_checkpoint, orchestrator=None)

        draft_result = handle_post_save(
            manuscript="draft_then_commit_manuscript",
            page="233_0001",
            save_intent="draft",
            active_learning_enabled=False,
            recognition_engine="local",
            text_payload={"1": "rama"},
            manuscript_root=manuscript_root,
            base_checkpoint_path=base_checkpoint,
            graph_payload={"nodes": [{"x": 1, "y": 2}], "edges": []},
            textbox_labels=[0],
            modifications=[],
            orchestrator=orchestrator,
        )
        commit_result = handle_post_save(
            manuscript="draft_then_commit_manuscript",
            page="233_0001",
            save_intent="commit",
            active_learning_enabled=True,
            recognition_engine="local",
            text_payload={"1": "rama"},
            manuscript_root=manuscript_root,
            base_checkpoint_path=base_checkpoint,
            graph_payload={"nodes": [{"x": 1, "y": 2}], "edges": []},
            textbox_labels=[0],
            modifications=[],
            orchestrator=orchestrator,
        )

        self.assertFalse(draft_result["entered_active_learning"])
        self.assertEqual(commit_result["revision"]["revision_number"], 2)
        self.assertFalse(commit_result["revision"]["is_duplicate"])
        self.assertTrue(commit_result["entered_active_learning"])
        self.assertEqual(len(orchestrator.jobs), 1)
        self.assertEqual(orchestrator.jobs[0].job_type, JobType.OCR_FINE_TUNE.value)

    def test_page_summary_marks_page_ready_when_prediction_matches_layout(self):
        manuscript_root, base_checkpoint = self._make_manuscript_root("page_ready")
        configure_runtime(base_checkpoint, orchestrator=None)
        record_prediction(
            manuscript_root=manuscript_root,
            page_id="233_0001",
            predicted_lines={"1": "rama"},
            recognition_engine="local",
            checkpoint_id="base",
            checkpoint_path=base_checkpoint,
            confidences={},
            layout_fingerprint="layout-a",
            base_checkpoint_path=base_checkpoint,
        )

        summary = summarize_page_active_learning(
            manuscript_root,
            "233_0001",
            current_text_payload={"1": "rama"},
            current_layout_fingerprint="layout-a",
            base_checkpoint_path=base_checkpoint,
        )

        self.assertEqual(summary["state"], "ready")
        self.assertTrue(summary["can_edit_text"])
        self.assertFalse(summary["needs_recognition"])
        self.assertEqual(summary["prediction"]["source_label"], "Pretrained local OCR")

    def test_page_summary_marks_page_stale_when_layout_changes_after_prediction(self):
        manuscript_root, base_checkpoint = self._make_manuscript_root("page_stale")
        configure_runtime(base_checkpoint, orchestrator=None)
        record_prediction(
            manuscript_root=manuscript_root,
            page_id="233_0001",
            predicted_lines={"1": "rama"},
            recognition_engine="local",
            checkpoint_id="base",
            checkpoint_path=base_checkpoint,
            confidences={},
            layout_fingerprint="layout-a",
            base_checkpoint_path=base_checkpoint,
        )

        summary = summarize_page_active_learning(
            manuscript_root,
            "233_0001",
            current_text_payload={"1": "rama"},
            current_layout_fingerprint="layout-b",
            base_checkpoint_path=base_checkpoint,
        )

        self.assertEqual(summary["state"], "stale_layout")
        self.assertFalse(summary["can_edit_text"])
        self.assertTrue(summary["needs_recognition"])
        self.assertFalse(summary["prediction"]["matches_current_layout"])

    def test_page_summary_marks_page_missing_when_no_text_exists(self):
        manuscript_root, base_checkpoint = self._make_manuscript_root("page_missing")
        configure_runtime(base_checkpoint, orchestrator=None)

        summary = summarize_page_active_learning(
            manuscript_root,
            "233_0001",
            current_text_payload={},
            current_layout_fingerprint="layout-a",
            base_checkpoint_path=base_checkpoint,
        )

        self.assertEqual(summary["state"], "missing_ocr")
        self.assertFalse(summary["can_edit_text"])
        self.assertTrue(summary["needs_recognition"])

    def test_manuscript_summary_clears_stale_pending_jobs_when_orchestrator_has_no_record(self):
        manuscript_root, base_checkpoint = self._make_manuscript_root("stale_pending_job")
        configure_runtime(base_checkpoint, orchestrator=None)
        registry = load_registry(manuscript_root, base_checkpoint)
        registry.enqueue_pending_job(
            {
                "job_id": "job-stale",
                "job_type": JobType.OCR_FINE_TUNE.value,
                "page_id": "233_0001",
                "revision_number": 1,
                "priority": 2,
                "state": "queued",
                "created_at": "2026-04-20T00:00:00+00:00",
            }
        )
        registry.set_status("queued", "AL: queued page 233_0001")

        summary = summarize_manuscript_active_learning(
            manuscript_root,
            base_checkpoint_path=base_checkpoint,
            orchestrator=_StubOrchestrator(),
        )

        self.assertEqual(summary["pending_jobs"], [])
        self.assertEqual(summary["code"], "idle")

    def test_resaving_a_consumed_page_marks_rebase_and_queues_rebuild(self):
        manuscript_root, base_checkpoint = self._make_manuscript_root("rebase")
        orchestrator = _StubOrchestrator()
        configure_runtime(base_checkpoint, orchestrator=None)

        first = handle_post_save(
            manuscript="rebase_manuscript",
            page="233_0001",
            save_intent="commit",
            active_learning_enabled=True,
            recognition_engine="local",
            text_payload={"1": "rama"},
            manuscript_root=manuscript_root,
            base_checkpoint_path=base_checkpoint,
            graph_payload={"nodes": [{"x": 1, "y": 2}], "edges": []},
            textbox_labels=[0],
            modifications=[],
            orchestrator=orchestrator,
        )
        registry = load_registry(manuscript_root, base_checkpoint)
        rebuilt_checkpoint = manuscript_root / "rebuilt_step.pth"
        rebuilt_checkpoint.write_text("ckpt", encoding="utf-8")
        registry.ensure_checkpoint_record("page1_ckpt", rebuilt_checkpoint, status="active")
        registry.mark_revision_consumed("233_0001", 1, "page1_ckpt")
        registry.data["active_checkpoint_id"] = "page1_ckpt"
        registry.save()
        orchestrator.jobs.clear()

        second = handle_post_save(
            manuscript="rebase_manuscript",
            page="233_0001",
            save_intent="commit",
            active_learning_enabled=True,
            recognition_engine="local",
            text_payload={"1": "rama changed"},
            manuscript_root=manuscript_root,
            base_checkpoint_path=base_checkpoint,
            graph_payload={"nodes": [{"x": 1, "y": 2}], "edges": [{"source": 0, "target": 0}]},
            textbox_labels=[0],
            modifications=[{"type": "delete"}],
            orchestrator=orchestrator,
        )

        self.assertEqual(first["revision"]["revision_number"], 1)
        self.assertEqual(second["revision"]["revision_number"], 2)
        self.assertTrue(second["active_learning"]["needs_rebase"])
        self.assertEqual(len(orchestrator.jobs), 1)
        self.assertEqual(orchestrator.jobs[0].job_type, JobType.OCR_REBASE.value)

    def test_run_ocr_finetune_job_promotes_candidate_created_by_training_step(self):
        manuscript_root, base_checkpoint = self._make_manuscript_root("promote_candidate_after_training")
        configure_runtime(base_checkpoint, orchestrator=None)
        registry = load_registry(manuscript_root, base_checkpoint)
        registry.record_page_revision(
            "233_0001",
            {
                "content_hash": "page-1",
                "save_intent": "commit",
                "supervision_present": True,
                "recognition_engine": "local",
                "text_line_count": 1,
                "text_non_empty_line_count": 1,
            },
        )
        candidate_checkpoint = manuscript_root / "candidate_1.pth"
        candidate_checkpoint.write_text("candidate", encoding="utf-8")

        def fake_train_candidate_step(job_payload):
            inner_registry = load_registry(manuscript_root, base_checkpoint)
            inner_registry.mark_candidate(
                {
                    "candidate_id": "ocr_233_0001_r0001",
                    "checkpoint_path": candidate_checkpoint,
                    "parent_checkpoint_id": "base",
                    "page_id": "233_0001",
                    "revision_number": 1,
                }
            )
            return {
                "candidate_id": "ocr_233_0001_r0001",
                "checkpoint_path": str(candidate_checkpoint),
                "verification": {"passed": True, "reason": "ok"},
                "training_revision_ref": {"page_id": "233_0001", "revision_number": 1},
            }

        with mock.patch("ocr_active_learning_runtime._train_candidate_step", side_effect=fake_train_candidate_step):
            result = run_ocr_finetune_job(
                {
                    "manuscript_root": str(manuscript_root),
                    "base_checkpoint_path": str(base_checkpoint),
                }
            )

        reloaded = load_registry(manuscript_root, base_checkpoint)
        self.assertTrue(result["promoted"])
        self.assertEqual(reloaded.active_checkpoint_id(), "ocr_233_0001_r0001")
        self.assertEqual(reloaded.find_revision("233_0001", 1)["consumed_into_checkpoint_id"], "ocr_233_0001_r0001")
        self.assertEqual(reloaded.data["checkpoints"]["ocr_233_0001_r0001"]["status"], "active")

    def test_rebuild_manuscript_lineage_promotes_final_candidate_created_by_training_steps(self):
        manuscript_root, base_checkpoint = self._make_manuscript_root("rebuild_promotes_latest")
        configure_runtime(base_checkpoint, orchestrator=None)
        registry = load_registry(manuscript_root, base_checkpoint)
        for page_id, content_hash in (("233_0001", "page-1"), ("233_0002", "page-2")):
            registry.record_page_revision(
                page_id,
                {
                    "content_hash": content_hash,
                    "save_intent": "commit",
                    "supervision_present": True,
                    "recognition_engine": "local",
                    "text_line_count": 1,
                    "text_non_empty_line_count": 1,
                },
            )

        def fake_train_candidate_step(job_payload):
            candidate_id = str(job_payload["candidate_id"])
            checkpoint_path = manuscript_root / f"{candidate_id}.pth"
            checkpoint_path.write_text(candidate_id, encoding="utf-8")
            inner_registry = load_registry(manuscript_root, base_checkpoint)
            inner_registry.mark_candidate(
                {
                    "candidate_id": candidate_id,
                    "checkpoint_path": checkpoint_path,
                    "parent_checkpoint_id": job_payload.get("parent_checkpoint_id"),
                    "page_id": job_payload["training_revision_ref"]["page_id"],
                    "revision_number": job_payload["training_revision_ref"]["revision_number"],
                }
            )
            return {
                "candidate_id": candidate_id,
                "checkpoint_path": str(checkpoint_path),
                "verification": {"passed": True, "reason": "ok"},
                "training_revision_ref": dict(job_payload["training_revision_ref"]),
            }

        approved_refs = [
            {"page_id": "233_0001", "revision_number": 1},
            {"page_id": "233_0002", "revision_number": 1},
        ]
        with mock.patch("ocr_active_learning_runtime._train_candidate_step", side_effect=fake_train_candidate_step):
            result = rebuild_manuscript_lineage(
                {
                    "manuscript_root": str(manuscript_root),
                    "base_checkpoint_path": str(base_checkpoint),
                    "approved_revision_refs": approved_refs,
                }
            )

        reloaded = load_registry(manuscript_root, base_checkpoint)
        self.assertTrue(result["rebuilt"])
        self.assertEqual(result["candidate_id"], "rebase_002_233_0002_r0001")
        self.assertEqual(reloaded.active_checkpoint_id(), "rebase_002_233_0002_r0001")
        self.assertEqual(reloaded.find_revision("233_0001", 1)["consumed_into_checkpoint_id"], "rebase_002_233_0002_r0001")
        self.assertEqual(reloaded.find_revision("233_0002", 1)["consumed_into_checkpoint_id"], "rebase_002_233_0002_r0001")


if __name__ == "__main__":
    unittest.main()
