import json
import shutil
import sys
import unittest
from unittest import mock
from pathlib import Path
import xml.etree.ElementTree as ET

from PIL import Image


TESTS_ROOT = Path(__file__).resolve().parent
APP_ROOT = TESTS_ROOT.parent
REPO_ROOT = APP_ROOT.parent

if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.backend_app_import import backend_app_module
from job_orchestrator import JobType
from manuscript_ocr_registry import load_registry
from ocr_active_learning_runtime import (
    _compact_candidate_artifacts,
    _prune_obsolete_checkpoints,
    configure_runtime,
    handle_post_save,
    record_prediction,
    rebuild_manuscript_lineage,
    run_ocr_finetune_job,
    summarize_manuscript_active_learning,
    summarize_page_active_learning,
    _prepare_revision_pages,
)


class _StubOrchestrator:
    def __init__(self):
        self.jobs = []

    def enqueue(self, job):
        self.jobs.append(job)
        return getattr(job, "job_id", f"job-{len(self.jobs)}")

    def get_job_status(self, job_id):
        return {}


class _StatusRecordingOrchestrator(_StubOrchestrator):
    def __init__(self):
        super().__init__()
        self.statuses = {}

    def enqueue(self, job):
        job_id = super().enqueue(job)
        status = {
            "job_id": job_id,
            "job_type": str(job.job_type),
            "state": "queued",
            "priority": int(job.priority),
            "manuscript": job.manuscript,
            "payload": dict(job.payload),
            "created_at": job.created_at,
            "started_at": None,
        }
        self.statuses[job_id] = status
        registry = load_registry(job.manuscript_root, job.payload["base_checkpoint_path"])
        registry.enqueue_pending_job(
            {
                "job_id": job_id,
                "job_type": str(job.job_type),
                "page_id": job.payload.get("page_id"),
                "revision_number": job.payload.get("revision_number"),
                "priority": int(job.priority),
                "state": "queued",
                "created_at": job.created_at,
            }
        )
        registry.set_status("queued", f"Waiting to learn from page {job.payload.get('page_id')}", job_id=job_id)
        return job_id

    def get_job_status(self, job_id):
        return dict(self.statuses.get(str(job_id), {}))


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
        self.assertEqual(
            orchestrator.jobs[0].payload["recipe"]["sibling_checkpoint_strategy"],
            "best_norm_ed",
        )
        self.assertNotIn("verifier_revision_refs", orchestrator.jobs[0].payload)
        self.assertEqual(result["revision"]["revision_number"], 1)
        self.assertTrue(result["entered_active_learning"])
        summary = summarize_page_active_learning(
            manuscript_root,
            "233_0001",
            current_text_payload={"1": "rama"},
            current_graph_payload={"nodes": [{"x": 1, "y": 2}], "edges": []},
            current_textbox_labels=[0],
            base_checkpoint_path=base_checkpoint,
        )
        self.assertEqual(summary["review_status"], "ground_truth_saved")
        self.assertTrue(summary["has_ground_truth"])
        self.assertTrue(summary["current_revision_is_ground_truth"])

    def test_commit_save_returns_fresh_queued_active_learning_status(self):
        manuscript_root, base_checkpoint = self._make_manuscript_root("fresh_queued_status")
        orchestrator = _StatusRecordingOrchestrator()
        configure_runtime(base_checkpoint, orchestrator=None)

        result = handle_post_save(
            manuscript="fresh_queued_status_manuscript",
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

        self.assertEqual(result["active_learning"]["code"], "queued")
        self.assertEqual(len(result["active_learning"]["pending_jobs"]), 1)
        self.assertEqual(
            result["active_learning"]["pending_jobs"][0]["job_id"],
            result["queued_job_ids"][0],
        )

    def test_commit_save_after_gemini_prediction_enqueues_builtin_finetune_job(self):
        manuscript_root, base_checkpoint = self._make_manuscript_root("gemini_prediction_supervision")
        orchestrator = _StubOrchestrator()
        configure_runtime(base_checkpoint, orchestrator=None)
        record_prediction(
            manuscript_root=manuscript_root,
            page_id="233_0001",
            predicted_lines={"1": "builtin raw"},
            recognition_engine="local",
            checkpoint_id="base",
            checkpoint_path=base_checkpoint,
            confidences={},
            layout_fingerprint="layout-a",
            base_checkpoint_path=base_checkpoint,
        )
        record_prediction(
            manuscript_root=manuscript_root,
            page_id="233_0001",
            predicted_lines={"1": "gemini raw"},
            recognition_engine="gemini",
            checkpoint_id=None,
            checkpoint_path=None,
            confidences={},
            layout_fingerprint="layout-a",
            base_checkpoint_path=base_checkpoint,
        )

        result = handle_post_save(
            manuscript="gemini_prediction_supervision_manuscript",
            page="233_0001",
            save_intent="commit",
            active_learning_enabled=True,
            recognition_engine="gemini",
            text_payload={"1": "rama corrected"},
            manuscript_root=manuscript_root,
            base_checkpoint_path=base_checkpoint,
            graph_payload={"nodes": [{"x": 1, "y": 2}], "edges": []},
            textbox_labels=[0],
            modifications=[],
            orchestrator=orchestrator,
        )

        self.assertEqual(len(orchestrator.jobs), 1)
        job = orchestrator.jobs[0]
        self.assertEqual(job.job_type, JobType.OCR_FINE_TUNE.value)
        self.assertEqual(job.payload["parent_checkpoint_id"], "base")
        self.assertEqual(Path(job.payload["parent_checkpoint_path"]).resolve(), base_checkpoint.resolve())
        self.assertTrue(result["entered_active_learning"])

        registry_payload = load_registry(manuscript_root, base_checkpoint).snapshot()
        revision_payload = registry_payload["page_revisions"]["233_0001"][-1]
        self.assertEqual(revision_payload["recognition_engine"], "gemini")
        self.assertEqual(revision_payload["prediction_engine"], "gemini")
        self.assertIsNone(revision_payload["prediction_checkpoint_id"])
        self.assertEqual(
            registry_payload["last_prediction_by_page"]["233_0001"]["recognition_engine"],
            "gemini",
        )
        self.assertEqual(
            registry_payload["prediction_history_by_page"]["233_0001"][0]["recognition_engine"],
            "local",
        )
        self.assertEqual(
            registry_payload["last_prediction_by_page_and_engine"]["233_0001"]["local"]["predicted_lines"],
            {"1": "builtin raw"},
        )

        page_summary = json.loads(
            (manuscript_root / "active_learning" / "telemetry" / "page_edit_summary.json").read_text(encoding="utf-8")
        )
        text_metrics = page_summary["233_0001#r1"]["text_metrics"]
        self.assertEqual(text_metrics["prediction_source_engine"], "local")
        self.assertEqual(text_metrics["prediction_source_checkpoint_id"], "base")
        self.assertEqual(text_metrics["measurement_status"], "measured")
        self.assertEqual(text_metrics["per_line_diffs"][0]["predicted_text"], "builtin raw")
        self.assertNotEqual(text_metrics["per_line_diffs"][0]["predicted_text"], "gemini raw")

        human_summary = json.loads(
            (manuscript_root / "active_learning" / "telemetry" / "human_interventions.json").read_text(encoding="utf-8")
        )
        self.assertEqual(human_summary["read_mode_effort_curve"][0]["prediction_source_engine"], "local")

    def test_reader_capabilities_report_server_configured_gemini(self):
        client = backend_app_module.app.test_client()

        with (
            mock.patch.object(backend_app_module, "_server_gemini_api_key", return_value="server-key"),
            mock.patch.dict(backend_app_module.os.environ, {"GEMINI_OCR_TIMEOUT_SECONDS": "12"}, clear=False),
        ):
            response = client.get("/recognition/readers")

        response_json = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response_json["readers"]["local"]["available"])
        self.assertTrue(response_json["readers"]["gemini"]["available"])
        self.assertEqual(response_json["readers"]["gemini"]["requestTimeoutSeconds"], 12.0)
        self.assertEqual(response_json["defaultEngine"], "local")

    def test_gemini_failure_response_includes_recovery_metadata(self):
        client = backend_app_module.app.test_client()

        with (
            mock.patch.object(backend_app_module, "_server_gemini_api_key", return_value="server-key"),
            mock.patch.object(
                backend_app_module,
                "_run_gemini_recognition_internal",
                return_value={
                    "error": "Gemini did not finish within 12 seconds.",
                    "errorCode": "gemini_timeout",
                },
            ),
        ):
            response = client.post(
                "/recognize-text",
                json={
                    "manuscript": "any_manuscript",
                    "page": "233_0001",
                    "recognitionEngine": "gemini",
                },
            )

        response_json = response.get_json()
        self.assertEqual(response.status_code, 502)
        self.assertEqual(response_json["errorCode"], "gemini_timeout")
        self.assertEqual(response_json["failedEngine"], "gemini")
        self.assertTrue(response_json["retryable"])
        self.assertEqual(response_json["fallbackEngines"], ["local"])

    def test_parse_gemini_transcriptions_accepts_common_json_wrappers(self):
        wrapped_payload = """
        ```json
        {"transcriptions": [{"id": 0, "text": "zero"}, {"id": 7, "text": "rama"}, {"id": "8", "text": " sita "}]}
        ```
        """

        parsed = backend_app_module._parse_gemini_transcriptions(wrapped_payload)

        self.assertEqual(parsed, {"0": "zero", "7": "rama", "8": "sita"})

    def test_parse_gemini_transcriptions_rejects_empty_output(self):
        with self.assertRaises(ValueError):
            backend_app_module._parse_gemini_transcriptions("[]")

    def test_snapshot_page_revision_copies_line_segmentation_metadata_sidecar(self):
        manuscript_root, base_checkpoint = self._make_manuscript_root("snapshot_metadata")
        page_id = "233_0001"
        metadata_path = manuscript_root / "layout_analysis_output" / "page-xml-format" / f"{page_id}_line_segmentation_metadata.json"
        reading_metadata_path = manuscript_root / "layout_analysis_output" / "page-xml-format" / f"{page_id}_reading_direction_metadata.json"
        metadata_path.write_text('{"strategy_name":"legacy_axis_bound_v1","line_metadata":[]}', encoding="utf-8")
        reading_metadata_path.write_text(
            '{"schema_version":1,"line_annotations":[{"resolved_line_numeric_id":0}],"stale_annotations":[]}',
            encoding="utf-8",
        )
        configure_runtime(base_checkpoint, orchestrator=None)

        result = handle_post_save(
            manuscript="snapshot_metadata_manuscript",
            page=page_id,
            save_intent="commit",
            active_learning_enabled=False,
            recognition_engine="local",
            text_payload={"1": "rama"},
            manuscript_root=manuscript_root,
            base_checkpoint_path=base_checkpoint,
            graph_payload={"nodes": [{"x": 1, "y": 2}], "edges": []},
            textbox_labels=[0],
            modifications=[],
            orchestrator=None,
        )

        registry = load_registry(manuscript_root, base_checkpoint)
        snapshot_root = registry.revision_snapshot_root(page_id, result["revision"]["revision_number"])
        self.assertTrue((snapshot_root / "page-xml-format" / metadata_path.name).exists())
        self.assertTrue((snapshot_root / "page-xml-format" / reading_metadata_path.name).exists())

    def test_prepare_revision_pages_passes_snapshot_metadata_dir_without_heatmaps(self):
        manuscript_root, base_checkpoint = self._make_manuscript_root("prepare_revision_metadata")
        page_id = "233_0001"
        configure_runtime(base_checkpoint, orchestrator=None)
        result = handle_post_save(
            manuscript="prepare_revision_metadata_manuscript",
            page=page_id,
            save_intent="commit",
            active_learning_enabled=False,
            recognition_engine="local",
            text_payload={"1": "rama"},
            manuscript_root=manuscript_root,
            base_checkpoint_path=base_checkpoint,
            graph_payload={"nodes": [{"x": 1, "y": 2}], "edges": []},
            textbox_labels=[0],
            modifications=[],
            orchestrator=None,
        )
        registry = load_registry(manuscript_root, base_checkpoint)

        with mock.patch("ocr_active_learning_runtime.prepare_page_datasets", return_value={page_id: "prepared"}) as mock_prepare:
            prepared_pages = _prepare_revision_pages(
                registry,
                [{"page_id": page_id, "revision_number": result["revision"]["revision_number"]}],
                "unit_purpose",
            )

        self.assertEqual(prepared_pages, ["prepared"])
        _, kwargs = mock_prepare.call_args
        self.assertEqual(kwargs["heatmaps_dir"] if "heatmaps_dir" in kwargs else None, None)
        self.assertEqual(kwargs["line_segmentation_metadata_dir"], registry.revision_snapshot_root(page_id, 1) / "page-xml-format")

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
        layout_with_text_result = handle_post_save(
            manuscript="draft_manuscript",
            page="233_0001",
            save_intent="commit",
            active_learning_enabled=True,
            recognition_engine="local",
            text_payload={"1": "layout text should not train"},
            manuscript_root=manuscript_root,
            base_checkpoint_path=base_checkpoint,
            graph_payload={"nodes": [{"x": 1, "y": 2}], "edges": []},
            textbox_labels=[0],
            modifications=[],
            save_scope="layout",
            orchestrator=orchestrator,
        )

        self.assertEqual(orchestrator.jobs, [])
        self.assertFalse(draft_result["entered_active_learning"])
        self.assertFalse(layout_only_result["entered_active_learning"])
        self.assertFalse(layout_with_text_result["entered_active_learning"])
        self.assertFalse(layout_with_text_result["revision"]["supervision_present"])

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
        handle_post_save(
            manuscript="page_ready_manuscript",
            page="233_0001",
            save_intent="commit",
            active_learning_enabled=False,
            recognition_engine="local",
            text_payload={"1": "rama"},
            manuscript_root=manuscript_root,
            base_checkpoint_path=base_checkpoint,
            graph_payload={"nodes": [{"x": 1, "y": 2}], "edges": []},
            textbox_labels=[0],
            modifications=[],
            orchestrator=None,
        )
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
        self.assertEqual(summary["review_status"], "ocr_prediction_unreviewed")
        self.assertTrue(summary["has_ground_truth"])
        self.assertFalse(summary["current_revision_is_ground_truth"])
        self.assertTrue(summary["can_edit_text"])
        self.assertTrue(summary["can_resume_recognition"])
        self.assertFalse(summary["needs_recognition"])
        self.assertEqual(summary["prediction"]["source_label"], "Built-in reader")

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
        self.assertEqual(summary["review_status"], "ground_truth_stale_layout" if summary["has_ground_truth"] else "ocr_prediction_unreviewed")
        self.assertFalse(summary["can_edit_text"])
        self.assertTrue(summary["needs_recognition"])
        self.assertFalse(summary["prediction"]["matches_current_layout"])

    def test_page_summary_marks_ocr_prediction_without_commit_as_unreviewed(self):
        manuscript_root, base_checkpoint = self._make_manuscript_root("prediction_without_commit")
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

        self.assertEqual(summary["review_status"], "ocr_prediction_unreviewed")
        self.assertFalse(summary["has_ground_truth"])
        self.assertFalse(summary["current_revision_is_ground_truth"])

    def test_compute_page_layout_fingerprint_ignores_equivalent_point_order_changes(self):
        manuscript_root, _ = self._make_manuscript_root("layout_fingerprint_point_order")
        xml_dir = manuscript_root / "layout_analysis_output" / "page-xml-format"
        first_xml = xml_dir / "first.xml"
        second_xml = xml_dir / "second.xml"

        first_xml.write_text(
            """<?xml version="1.0" encoding="UTF-8"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">
  <Page imageFilename="233_0001.jpg" imageWidth="100" imageHeight="100">
    <TextRegion id="region_0">
      <TextLine id="line_0" custom="structure_line_id_7">
        <Coords points="10,10 40,10 40,20 10,20" />
        <Baseline points="10,15 25,15 40,15" />
      </TextLine>
    </TextRegion>
  </Page>
</PcGts>
""",
            encoding="utf-8",
        )
        second_xml.write_text(
            """<?xml version="1.0" encoding="UTF-8"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">
  <Page imageFilename="233_0001.jpg" imageWidth="100" imageHeight="100">
    <TextRegion id="region_0">
      <TextLine id="line_0" custom="structure_line_id_7">
        <Coords points="40,20 10,20 10,10 40,10" />
        <Baseline points="40,15 25,15 10,15" />
      </TextLine>
    </TextRegion>
  </Page>
</PcGts>
""",
            encoding="utf-8",
        )

        first_fingerprint = backend_app_module.compute_page_layout_fingerprint(str(first_xml))
        second_fingerprint = backend_app_module.compute_page_layout_fingerprint(str(second_xml))

        self.assertEqual(first_fingerprint, second_fingerprint)

    def test_compute_page_layout_fingerprint_changes_when_reading_direction_metadata_changes(self):
        manuscript_root, _ = self._make_manuscript_root("layout_fingerprint_reading_direction")
        xml_dir = manuscript_root / "layout_analysis_output" / "page-xml-format"
        xml_path = xml_dir / "233_0001.xml"
        xml_path.write_text(
            """<?xml version="1.0" encoding="UTF-8"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">
  <Page imageFilename="233_0001.jpg" imageWidth="100" imageHeight="100">
    <TextRegion id="region_0">
      <TextLine id="line_0" custom="structure_line_id_7">
        <Coords points="10,10 40,10 40,20 10,20" />
        <Baseline points="10,15 25,15 40,15" />
      </TextLine>
    </TextRegion>
  </Page>
</PcGts>
""",
            encoding="utf-8",
        )

        original_fingerprint = backend_app_module.compute_page_layout_fingerprint(str(xml_path))
        backend_app_module.default_reading_direction_metadata_path(xml_path).write_text(
            """{
  "schema_version": 1,
  "line_annotations": [
    {
      "resolved_line_numeric_id": 7,
      "component_node_indices": [0, 1],
      "cut_midpoint": [25, 15],
      "reading_direction": [1, 0]
    }
  ],
  "stale_annotations": []
}
""",
            encoding="utf-8",
        )
        updated_fingerprint = backend_app_module.compute_page_layout_fingerprint(str(xml_path))

        self.assertNotEqual(updated_fingerprint, original_fingerprint)

    def test_text_only_save_updates_text_without_regenerating_layout(self):
        manuscript_root, base_checkpoint = self._make_manuscript_root("text_only_save")
        page_id = "233_0001"
        xml_path = manuscript_root / "layout_analysis_output" / "page-xml-format" / f"{page_id}.xml"
        xml_path.write_text(
            """<?xml version="1.0" encoding="UTF-8"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">
  <Page imageFilename="233_0001.jpg" imageWidth="100" imageHeight="100">
    <TextRegion id="region_0">
      <TextLine id="line_0" custom="structure_line_id_7">
        <Coords points="10,10 40,10 40,20 10,20" />
        <Baseline points="10,15 25,15 40,15" />
        <TextEquiv>
          <Unicode>old text</Unicode>
        </TextEquiv>
      </TextLine>
    </TextRegion>
  </Page>
</PcGts>
""",
            encoding="utf-8",
        )
        reading_metadata_path = backend_app_module.default_reading_direction_metadata_path(xml_path)
        reading_metadata = """{
  "schema_version": 1,
  "line_annotations": [
    {
      "resolved_line_numeric_id": 7,
      "component_node_indices": [0, 1],
      "cut_midpoint": [25, 15],
      "reading_direction": [1, 0]
    }
  ],
  "stale_annotations": []
}
"""
        reading_metadata_path.write_text(reading_metadata, encoding="utf-8")

        original_fingerprint = backend_app_module.compute_page_layout_fingerprint(str(xml_path))
        client = backend_app_module.app.test_client()
        mock_active_learning_result = {
            "active_learning": {"label": "Not updating right now"},
            "revision": {
                "page_id": page_id,
                "revision_number": 1,
                "save_intent": "draft",
                "is_duplicate": False,
            },
            "queued_job_ids": [],
        }

        with (
            mock.patch.object(backend_app_module, "UPLOAD_FOLDER", str(manuscript_root.parent)),
            mock.patch.object(backend_app_module, "generate_xml_and_images_for_page") as mock_generate,
            mock.patch.object(
                backend_app_module,
                "handle_post_save",
                return_value=mock_active_learning_result,
            ),
            mock.patch.object(
                backend_app_module,
                "_build_page_workflow",
                return_value={"state": "ready", "can_edit_text": True, "needs_recognition": False},
            ),
        ):
            response = client.post(
                f"/semi-segment/{manuscript_root.name}/{page_id}",
                json={
                    "graph": {"nodes": [{"x": 1, "y": 2, "s": 3}], "edges": []},
                    "modifications": [],
                    "textlineLabels": [-1],
                    "textboxLabels": [0],
                    "textContent": {"7": "corrected text"},
                    "runRecognition": False,
                    "recognitionEngine": "local",
                    "saveIntent": "draft",
                    "saveScope": "text_only",
                    "activeLearningEnabled": False,
                },
            )

        response_json = response.get_json()
        self.assertEqual(response.status_code, 200, response_json)
        self.assertEqual(response_json["status"], "success")
        self.assertEqual(response_json["lines"], 1)
        mock_generate.assert_not_called()
        self.assertEqual(reading_metadata_path.read_text(encoding="utf-8"), reading_metadata)

        updated_fingerprint = backend_app_module.compute_page_layout_fingerprint(str(xml_path))
        self.assertEqual(updated_fingerprint, original_fingerprint)
        self.assertEqual(
            backend_app_module.get_existing_text_content(str(xml_path))["text"],
            {"7": "corrected text"},
        )

        ns = {"p": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}
        root = ET.parse(xml_path).getroot()
        textline = root.find(".//p:TextLine", ns)
        self.assertIsNotNone(textline)
        self.assertEqual(textline.find("./p:Coords", ns).get("points"), "10,10 40,10 40,20 10,20")
        self.assertEqual(textline.find("./p:Baseline", ns).get("points"), "10,15 25,15 40,15")

    def test_layout_save_normalizes_textbox_labels_before_persistence(self):
        manuscript_root, _ = self._make_manuscript_root("normalize_textbox_labels")
        page_id = "233_0001"
        client = backend_app_module.app.test_client()
        mock_active_learning_result = {
            "active_learning": {"label": "Not updating right now"},
            "revision": None,
            "queued_job_ids": [],
        }

        with (
            mock.patch.object(backend_app_module, "UPLOAD_FOLDER", str(manuscript_root.parent)),
            mock.patch.object(
                backend_app_module,
                "generate_xml_and_images_for_page",
                return_value={"status": "success", "lines": 0},
            ) as mock_generate,
            mock.patch.object(
                backend_app_module,
                "handle_post_save",
                return_value=mock_active_learning_result,
            ) as mock_handle_post_save,
            mock.patch.object(
                backend_app_module,
                "_build_page_workflow",
                return_value={"state": "ready", "can_edit_text": False, "needs_recognition": True},
            ) as mock_build_workflow,
        ):
            response = client.post(
                f"/semi-segment/{manuscript_root.name}/{page_id}",
                json={
                    "graph": {
                        "nodes": [
                            {"x": 1, "y": 2, "s": 3},
                            {"x": 2, "y": 3, "s": 3},
                            {"x": 3, "y": 4, "s": 3},
                            {"x": 4, "y": 5, "s": 3},
                        ],
                        "edges": [],
                    },
                    "modifications": [],
                    "textlineLabels": [-1, -1, -1, -1],
                    "textboxLabels": [4, -3, "bad", 2.7, 9],
                    "textContent": {},
                    "runRecognition": False,
                    "recognitionEngine": "local",
                    "saveIntent": "commit",
                    "saveScope": "layout",
                    "activeLearningEnabled": False,
                },
            )

        response_json = response.get_json()
        self.assertEqual(response.status_code, 200, response_json)
        expected_labels = [4, 0, 1, 2]
        self.assertEqual(mock_generate.call_args.kwargs["textbox_labels"], expected_labels)
        self.assertEqual(mock_handle_post_save.call_args.kwargs["textbox_labels"], expected_labels)
        self.assertEqual(mock_build_workflow.call_args.kwargs["textbox_labels"], expected_labels)

    def test_layout_save_assigns_unique_regions_to_unlabeled_text_lines(self):
        manuscript_root, _ = self._make_manuscript_root("unique_unlabeled_textbox_labels")
        page_id = "233_0001"
        client = backend_app_module.app.test_client()
        mock_active_learning_result = {
            "active_learning": {"label": "Not updating right now"},
            "revision": None,
            "queued_job_ids": [],
        }

        with (
            mock.patch.object(backend_app_module, "UPLOAD_FOLDER", str(manuscript_root.parent)),
            mock.patch.object(
                backend_app_module,
                "generate_xml_and_images_for_page",
                return_value={"status": "success", "lines": 2},
            ) as mock_generate,
            mock.patch.object(
                backend_app_module,
                "handle_post_save",
                return_value=mock_active_learning_result,
            ) as mock_handle_post_save,
            mock.patch.object(
                backend_app_module,
                "_build_page_workflow",
                return_value={"state": "ready", "can_edit_text": False, "needs_recognition": True},
            ) as mock_build_workflow,
        ):
            response = client.post(
                f"/semi-segment/{manuscript_root.name}/{page_id}",
                json={
                    "graph": {
                        "nodes": [
                            {"x": 1, "y": 2, "s": 3},
                            {"x": 2, "y": 2, "s": 3},
                            {"x": 1, "y": 8, "s": 3},
                            {"x": 2, "y": 8, "s": 3},
                        ],
                        "edges": [
                            {"source": 0, "target": 1},
                            {"source": 2, "target": 3},
                        ],
                    },
                    "modifications": [],
                    "textlineLabels": [-1, -1, -1, -1],
                    "textboxLabels": [-1, -1, -1, -1],
                    "textContent": {},
                    "runRecognition": False,
                    "recognitionEngine": "local",
                    "saveIntent": "commit",
                    "saveScope": "layout",
                    "activeLearningEnabled": False,
                },
            )

        response_json = response.get_json()
        self.assertEqual(response.status_code, 200, response_json)
        expected_labels = [0, 0, 1, 1]
        self.assertEqual(mock_generate.call_args.kwargs["textbox_labels"], expected_labels)
        self.assertEqual(mock_handle_post_save.call_args.kwargs["textbox_labels"], expected_labels)
        self.assertEqual(mock_build_workflow.call_args.kwargs["textbox_labels"], expected_labels)

    def test_layout_save_assigns_compact_unique_regions_after_manual_groups(self):
        manuscript_root, _ = self._make_manuscript_root("compact_unlabeled_textbox_labels")
        page_id = "233_0001"
        client = backend_app_module.app.test_client()
        mock_active_learning_result = {
            "active_learning": {"label": "Not updating right now"},
            "revision": None,
            "queued_job_ids": [],
        }

        nodes = [{"x": index, "y": index, "s": 3} for index in range(20)]
        edges = [
            {"source": 0, "target": 1},
            {"source": 2, "target": 3},
            {"source": 4, "target": 5},
            {"source": 6, "target": 7},
            {"source": 8, "target": 9},
            {"source": 10, "target": 11},
            {"source": 12, "target": 13},
            {"source": 14, "target": 15},
            {"source": 16, "target": 17},
            {"source": 18, "target": 19},
        ]
        textbox_labels = [0, 0] * 4 + [1, 1] * 4 + [-1, -1] * 2

        with (
            mock.patch.object(backend_app_module, "UPLOAD_FOLDER", str(manuscript_root.parent)),
            mock.patch.object(
                backend_app_module,
                "generate_xml_and_images_for_page",
                return_value={"status": "success", "lines": 10},
            ) as mock_generate,
            mock.patch.object(
                backend_app_module,
                "handle_post_save",
                return_value=mock_active_learning_result,
            ) as mock_handle_post_save,
            mock.patch.object(
                backend_app_module,
                "_build_page_workflow",
                return_value={"state": "ready", "can_edit_text": False, "needs_recognition": True},
            ) as mock_build_workflow,
        ):
            response = client.post(
                f"/semi-segment/{manuscript_root.name}/{page_id}",
                json={
                    "graph": {"nodes": nodes, "edges": edges},
                    "modifications": [],
                    "textlineLabels": [-1] * len(nodes),
                    "textboxLabels": textbox_labels,
                    "textContent": {},
                    "runRecognition": False,
                    "recognitionEngine": "local",
                    "saveIntent": "commit",
                    "saveScope": "layout",
                    "activeLearningEnabled": False,
                },
            )

        response_json = response.get_json()
        self.assertEqual(response.status_code, 200, response_json)
        expected_labels = [0, 0] * 4 + [1, 1] * 4 + [2, 2, 3, 3]
        self.assertEqual(mock_generate.call_args.kwargs["textbox_labels"], expected_labels)
        self.assertEqual(mock_handle_post_save.call_args.kwargs["textbox_labels"], expected_labels)
        self.assertEqual(mock_build_workflow.call_args.kwargs["textbox_labels"], expected_labels)

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

        self.assertEqual(summary["state"], "missing_page_xml")
        self.assertEqual(summary["review_status"], "layout_ready_no_text")
        self.assertFalse(summary["can_edit_text"])
        self.assertFalse(summary["can_resume_recognition"])
        self.assertTrue(summary["needs_recognition"])

    def test_page_summary_keeps_saved_text_editable_when_newer_checkpoint_becomes_active(self):
        manuscript_root, base_checkpoint = self._make_manuscript_root("page_checkpoint_advanced")
        configure_runtime(base_checkpoint, orchestrator=None)
        handle_post_save(
            manuscript="page_checkpoint_advanced_manuscript",
            page="233_0001",
            save_intent="commit",
            active_learning_enabled=False,
            recognition_engine="local",
            text_payload={"1": "rama corrected"},
            manuscript_root=manuscript_root,
            base_checkpoint_path=base_checkpoint,
            graph_payload={"nodes": [{"x": 1, "y": 2}], "edges": []},
            textbox_labels=[0],
            modifications=[],
            orchestrator=None,
        )
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
        registry = load_registry(manuscript_root, base_checkpoint)
        newer_checkpoint = manuscript_root / "newer_step.pth"
        newer_checkpoint.write_text("ckpt", encoding="utf-8")
        registry.ensure_checkpoint_record("ocr_233_0001_r0001", newer_checkpoint, status="active")
        registry.data["active_checkpoint_id"] = "ocr_233_0001_r0001"
        registry.save()

        summary = summarize_page_active_learning(
            manuscript_root,
            "233_0001",
            current_text_payload={"1": "rama corrected"},
            current_layout_fingerprint="layout-a",
            base_checkpoint_path=base_checkpoint,
        )

        self.assertEqual(summary["state"], "ready")
        self.assertEqual(summary["review_status"], "ground_truth_saved")
        self.assertTrue(summary["can_edit_text"])
        self.assertTrue(summary["can_resume_recognition"])
        self.assertFalse(summary["needs_recognition"])
        self.assertEqual(summary["prediction"]["checkpoint_id"], "base")

    def test_page_summary_can_resume_recognition_after_draft_when_committed_text_exists(self):
        manuscript_root, base_checkpoint = self._make_manuscript_root("resume_after_draft")
        configure_runtime(base_checkpoint, orchestrator=None)
        handle_post_save(
            manuscript="resume_after_draft_manuscript",
            page="233_0001",
            save_intent="commit",
            active_learning_enabled=False,
            recognition_engine="local",
            text_payload={"1": "rama"},
            manuscript_root=manuscript_root,
            base_checkpoint_path=base_checkpoint,
            graph_payload={"nodes": [{"x": 1, "y": 2}], "edges": []},
            textbox_labels=[0],
            modifications=[],
            orchestrator=None,
        )
        handle_post_save(
            manuscript="resume_after_draft_manuscript",
            page="233_0001",
            save_intent="draft",
            active_learning_enabled=False,
            recognition_engine="local",
            text_payload={"1": "rama draft edit"},
            manuscript_root=manuscript_root,
            base_checkpoint_path=base_checkpoint,
            graph_payload={"nodes": [{"x": 1, "y": 2}], "edges": []},
            textbox_labels=[0],
            modifications=[],
            orchestrator=None,
        )

        summary = summarize_page_active_learning(
            manuscript_root,
            "233_0001",
            current_text_payload={"1": "rama draft edit"},
            current_layout_fingerprint="layout-a",
            base_checkpoint_path=base_checkpoint,
        )

        self.assertEqual(summary["latest_revision_save_intent"], "draft")
        self.assertEqual(summary["latest_supervised_commit_revision_number"], 1)
        self.assertEqual(summary["review_status"], "ground_truth_with_draft_changes")
        self.assertTrue(summary["has_ground_truth"])
        self.assertFalse(summary["current_revision_is_ground_truth"])
        self.assertTrue(summary["can_edit_text"])
        self.assertTrue(summary["can_resume_recognition"])

    def test_page_summary_resumes_recognition_for_draft_only_text(self):
        manuscript_root, base_checkpoint = self._make_manuscript_root("draft_only_text")
        configure_runtime(base_checkpoint, orchestrator=None)
        handle_post_save(
            manuscript="draft_only_text_manuscript",
            page="233_0001",
            save_intent="draft",
            active_learning_enabled=False,
            recognition_engine="local",
            text_payload={"1": "draft only"},
            manuscript_root=manuscript_root,
            base_checkpoint_path=base_checkpoint,
            graph_payload={"nodes": [{"x": 1, "y": 2}], "edges": []},
            textbox_labels=[0],
            modifications=[],
            orchestrator=None,
        )

        summary = summarize_page_active_learning(
            manuscript_root,
            "233_0001",
            current_text_payload={"1": "draft only"},
            current_layout_fingerprint="layout-a",
            base_checkpoint_path=base_checkpoint,
        )

        self.assertEqual(summary["latest_revision_save_intent"], "draft")
        self.assertIsNone(summary["latest_supervised_commit_revision_number"])
        self.assertEqual(summary["review_status"], "draft_saved")
        self.assertFalse(summary["has_ground_truth"])
        self.assertTrue(summary["can_edit_text"])
        self.assertTrue(summary["can_resume_recognition"])

    def test_page_summary_marks_imported_text_as_legacy_review_needed(self):
        manuscript_root, base_checkpoint = self._make_manuscript_root("legacy_manual_text")
        configure_runtime(base_checkpoint, orchestrator=None)

        summary = summarize_page_active_learning(
            manuscript_root,
            "233_0001",
            current_text_payload={"1": "imported text"},
            current_layout_fingerprint="layout-a",
            base_checkpoint_path=base_checkpoint,
        )

        self.assertEqual(summary["review_status"], "legacy_text_needs_review")
        self.assertTrue(summary["can_edit_text"])
        self.assertFalse(summary["has_ground_truth"])

    def test_page_summary_marks_ground_truth_stale_after_layout_only_save(self):
        manuscript_root, base_checkpoint = self._make_manuscript_root("gt_stale_after_layout")
        configure_runtime(base_checkpoint, orchestrator=None)
        handle_post_save(
            manuscript="gt_stale_after_layout_manuscript",
            page="233_0001",
            save_intent="commit",
            active_learning_enabled=False,
            recognition_engine="local",
            text_payload={"1": "rama"},
            manuscript_root=manuscript_root,
            base_checkpoint_path=base_checkpoint,
            graph_payload={"nodes": [{"x": 1, "y": 2}], "edges": []},
            textbox_labels=[0],
            modifications=[],
            orchestrator=None,
        )
        handle_post_save(
            manuscript="gt_stale_after_layout_manuscript",
            page="233_0001",
            save_intent="commit",
            active_learning_enabled=False,
            recognition_engine="local",
            text_payload={},
            manuscript_root=manuscript_root,
            base_checkpoint_path=base_checkpoint,
            graph_payload={"nodes": [{"x": 1, "y": 2}], "edges": [{"source": 0, "target": 0}]},
            textbox_labels=[0],
            modifications=[{"type": "delete"}],
            save_scope="layout",
            orchestrator=None,
        )

        summary = summarize_page_active_learning(
            manuscript_root,
            "233_0001",
            current_text_payload={},
            current_layout_fingerprint="layout-b",
            base_checkpoint_path=base_checkpoint,
        )

        self.assertEqual(summary["review_status"], "ground_truth_stale_layout")
        self.assertTrue(summary["has_ground_truth"])
        self.assertFalse(summary["current_revision_is_ground_truth"])

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
                "promotion_summary": {"passed": True, "reason": "ok"},
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

    def test_compact_candidate_artifacts_preserves_model_and_removes_training_materialization(self):
        manuscript_root, base_checkpoint = self._make_manuscript_root("compact_candidate")
        registry = load_registry(manuscript_root, base_checkpoint)
        candidate_id = "ocr_233_0001_r0001"
        candidate_root = registry.checkpoints_root / candidate_id
        training_run = candidate_root / "training_run"
        training_run.mkdir(parents=True, exist_ok=True)
        selected_checkpoint = training_run / "best_norm_ED.pth"
        selected_checkpoint.write_bytes(b"selected checkpoint")
        (training_run / "best_accuracy.pth").write_bytes(b"unused checkpoint")
        (training_run / "iter_1.pth").write_bytes(b"iter checkpoint")
        (training_run / "log_train.txt").write_text("tiny log", encoding="utf-8")
        (candidate_root / "dataset").mkdir(parents=True, exist_ok=True)
        (candidate_root / "dataset" / "gt.txt").write_text("line", encoding="utf-8")
        (candidate_root / "lmdb").mkdir(parents=True, exist_ok=True)
        (candidate_root / "lmdb" / "data.mdb").write_bytes(b"lmdb")
        (candidate_root / "fine_tune_metadata.json").write_text(
            json.dumps({"output_checkpoint": str(selected_checkpoint)}),
            encoding="utf-8",
        )
        (registry.prepared_pages_root / f"train_{candidate_id}").mkdir(parents=True, exist_ok=True)
        (registry.prepared_pages_root / f"history_{candidate_id}").mkdir(parents=True, exist_ok=True)

        compacted_checkpoint, summary = _compact_candidate_artifacts(
            registry,
            candidate_id,
            selected_checkpoint,
        )

        compacted_checkpoint = Path(compacted_checkpoint)
        self.assertEqual(compacted_checkpoint, (candidate_root / "model.pth").resolve())
        self.assertEqual(compacted_checkpoint.read_bytes(), b"selected checkpoint")
        self.assertFalse(selected_checkpoint.exists())
        self.assertFalse((training_run / "best_accuracy.pth").exists())
        self.assertFalse((training_run / "iter_1.pth").exists())
        self.assertTrue((training_run / "log_train.txt").exists())
        self.assertFalse((candidate_root / "dataset").exists())
        self.assertFalse((candidate_root / "lmdb").exists())
        self.assertFalse((registry.prepared_pages_root / f"train_{candidate_id}").exists())
        self.assertFalse((registry.prepared_pages_root / f"history_{candidate_id}").exists())
        self.assertEqual(summary["errors"], [])

        metadata = json.loads((candidate_root / "fine_tune_metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(Path(metadata["output_checkpoint"]), compacted_checkpoint)
        self.assertEqual(Path(metadata["output_checkpoint_before_compaction"]), selected_checkpoint.resolve())
        self.assertGreaterEqual(len(metadata["artifact_compaction"]["removed_files"]), 3)

    def test_prune_obsolete_checkpoints_keeps_active_previous_and_pending_parent(self):
        manuscript_root, base_checkpoint = self._make_manuscript_root("prune_checkpoints")
        registry = load_registry(manuscript_root, base_checkpoint)

        for checkpoint_id in ("old", "previous", "active", "pending_parent"):
            checkpoint_dir = registry.checkpoints_root / checkpoint_id
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / "model.pth"
            checkpoint_path.write_text(checkpoint_id, encoding="utf-8")
            registry.ensure_checkpoint_record(checkpoint_id, checkpoint_path, status="active")

        registry.data["active_checkpoint_id"] = "active"
        registry.data["previous_active_checkpoint_id"] = "previous"
        registry.enqueue_pending_job(
            {
                "job_id": "pending",
                "job_type": JobType.OCR_FINE_TUNE.value,
                "parent_checkpoint_id": "pending_parent",
                "state": "queued",
                "priority": 2,
                "created_at": "2026-04-20T00:00:00+00:00",
            }
        )
        registry.save()

        summary = _prune_obsolete_checkpoints(registry)
        reloaded = load_registry(manuscript_root, base_checkpoint)

        self.assertIn("old", summary["pruned_checkpoint_ids"])
        self.assertFalse((registry.checkpoints_root / "old").exists())
        self.assertTrue((registry.checkpoints_root / "previous" / "model.pth").exists())
        self.assertTrue((registry.checkpoints_root / "active" / "model.pth").exists())
        self.assertTrue((registry.checkpoints_root / "pending_parent" / "model.pth").exists())
        self.assertEqual(reloaded.data["checkpoints"]["old"]["status"], "pruned")

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
                "promotion_summary": {"passed": True, "reason": "ok"},
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
