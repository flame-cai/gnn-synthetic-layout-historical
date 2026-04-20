import shutil
import sys
import unittest
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
from ocr_active_learning_runtime import configure_runtime, handle_post_save


class _StubOrchestrator:
    def __init__(self):
        self.jobs = []

    def enqueue(self, job):
        self.jobs.append(job)
        return getattr(job, "job_id", f"job-{len(self.jobs)}")


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


if __name__ == "__main__":
    unittest.main()
