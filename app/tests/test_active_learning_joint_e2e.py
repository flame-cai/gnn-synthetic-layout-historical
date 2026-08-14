"""OCR and layout active learning running together on one manuscript.

The two lineages share a `JobOrchestrator`, a single state listener, and one
exclusive `gpu` lease, so "each works alone" is not evidence that both work at
once. This drives the real orchestrator -- real queue, real GPU lease, real
spawned child processes -- with an OCR fine-tune and a layout fine-tune queued
from the same page, and checks that both promote and that neither writes into
the other's registry.

Slow: it trains both models for real. Skips cleanly when the checkpoints or the
source manuscript are unavailable.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path
import xml.etree.ElementTree as ET


TESTS_ROOT = Path(__file__).resolve().parent
APP_ROOT = TESTS_ROOT.parent
REPO_ROOT = APP_ROOT.parent

for path in (str(APP_ROOT), str(REPO_ROOT), str(REPO_ROOT / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

from job_orchestrator import JobOrchestrator, JobType
from manuscript_layout_registry import GRAPH_FORMAT_SUFFIXES
import layout_active_learning_runtime as layout_runtime
import manuscript_layout_registry
import manuscript_ocr_registry
import ocr_active_learning_runtime as ocr_runtime


PAGE_NS = {"p": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}
BASE_GNN_CHECKPOINT = APP_ROOT / "pretrained_gnn" / "v2.pt"
BASE_OCR_CHECKPOINT = APP_ROOT / "recognition" / "pretrained_model" / "vadakautuhala.pth"
LAYOUT_RECIPE = APP_ROOT / "pretrained_gnn" / "gnn_active_learning.yaml"
SOURCE_MANUSCRIPT = APP_ROOT / "input_manuscripts" / "dense"
SOURCE_PAGE = "10"
JOB_TIMEOUT_SECONDS = 900.0


class ActiveLearningJointEndToEndTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for required in (BASE_GNN_CHECKPOINT, BASE_OCR_CHECKPOINT):
            if not required.is_file():
                raise unittest.SkipTest(f"Required checkpoint is missing: {required}")
        if not (SOURCE_MANUSCRIPT / "layout_analysis_output" / "gnn-format").is_dir():
            raise unittest.SkipTest(f"Source manuscript is missing: {SOURCE_MANUSCRIPT}")

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.manuscript_root = Path(self._tmp.name) / "joint"
        self._copy_page(SOURCE_PAGE)
        self.orchestrator = JobOrchestrator()
        ocr_runtime.configure_runtime(BASE_OCR_CHECKPOINT, self.orchestrator)
        layout_runtime.configure_runtime(
            BASE_GNN_CHECKPOINT, self.orchestrator, recipe_config_path=LAYOUT_RECIPE
        )

    def tearDown(self):
        self.orchestrator.shutdown_workers()
        self._tmp.cleanup()

    def _copy_page(self, page_id: str) -> None:
        source_output = SOURCE_MANUSCRIPT / "layout_analysis_output"
        xml_dir = self.manuscript_root / "layout_analysis_output" / "page-xml-format"
        image_dir = self.manuscript_root / "layout_analysis_output" / "images_resized"
        graph_dir = self.manuscript_root / "layout_analysis_output" / "gnn-format"
        for directory in (xml_dir, image_dir, graph_dir):
            directory.mkdir(parents=True, exist_ok=True)

        for name in (
            f"{page_id}.xml",
            f"{page_id}_line_segmentation_metadata.json",
            f"{page_id}_reading_direction_metadata.json",
        ):
            source = source_output / "page-xml-format" / name
            if source.is_file():
                shutil.copy2(source, xml_dir / name)
        shutil.copy2(source_output / "images_resized" / f"{page_id}.jpg", image_dir / f"{page_id}.jpg")
        for suffix in GRAPH_FORMAT_SUFFIXES:
            shutil.copy2(
                source_output / "gnn-format" / f"{page_id}{suffix}",
                graph_dir / f"{page_id}{suffix}",
            )

    def _page_text(self, page_id: str) -> dict:
        xml_path = self.manuscript_root / "layout_analysis_output" / "page-xml-format" / f"{page_id}.xml"
        root = ET.parse(xml_path).getroot()
        text = {}
        for index, line in enumerate(root.findall(".//p:TextLine", PAGE_NS)):
            unicode_text = line.findtext(".//p:Unicode", default="", namespaces=PAGE_NS)
            text[str(line.get("id") or index)] = unicode_text
        return text

    def _graph_payload(self, page_id: str) -> dict:
        import numpy as np

        graph_dir = self.manuscript_root / "layout_analysis_output" / "gnn-format"
        nodes = np.loadtxt(graph_dir / f"{page_id}_inputs_unnormalized.txt").reshape(-1, 3)
        edges = np.loadtxt(graph_dir / f"{page_id}_edges.txt", dtype=int, ndmin=2).reshape(-1, 2)
        return {
            "nodes": [{"x": float(x), "y": float(y), "s": 0.0} for x, y, _ in nodes],
            "edges": [{"source": int(u), "target": int(v), "label": 1} for u, v in edges],
        }

    def _wait_for_jobs(self, job_ids: list[str]) -> list[dict]:
        deadline = time.time() + JOB_TIMEOUT_SECONDS
        while time.time() < deadline:
            statuses = [self.orchestrator.get_job_status(job_id) for job_id in job_ids]
            if all(str(s.get("state")) in {"completed", "failed", "canceled"} for s in statuses):
                return statuses
            time.sleep(0.5)
        self.fail(f"Jobs did not finish within {JOB_TIMEOUT_SECONDS}s: {job_ids}")

    def test_ocr_and_layout_finetuning_both_promote_from_the_same_page(self):
        text_payload = self._page_text(SOURCE_PAGE)
        graph_payload = self._graph_payload(SOURCE_PAGE)
        self.assertGreater(sum(1 for v in text_payload.values() if v.strip()), 0)
        self.assertGreater(len(graph_payload["edges"]), 0)

        layout_result = layout_runtime.handle_post_layout_save(
            manuscript="joint",
            page=SOURCE_PAGE,
            save_intent="commit",
            save_scope="layout",
            layout_active_learning_enabled=True,
            graph_payload=graph_payload,
            manuscript_root=self.manuscript_root,
            base_checkpoint_path=BASE_GNN_CHECKPOINT,
            orchestrator=self.orchestrator,
        )
        ocr_result = ocr_runtime.handle_post_save(
            manuscript="joint",
            page=SOURCE_PAGE,
            save_intent="commit",
            active_learning_enabled=True,
            recognition_engine="local",
            text_payload=text_payload,
            manuscript_root=self.manuscript_root,
            base_checkpoint_path=BASE_OCR_CHECKPOINT,
            graph_payload=graph_payload,
            save_scope="text_only",
            orchestrator=self.orchestrator,
        )

        layout_job_ids = layout_result["queued_job_ids"]
        ocr_job_ids = ocr_result["queued_job_ids"]
        self.assertEqual(len(layout_job_ids), 1)
        self.assertEqual(len(ocr_job_ids), 1)

        statuses = self._wait_for_jobs(layout_job_ids + ocr_job_ids)
        for status in statuses:
            self.assertEqual(
                status.get("state"),
                "completed",
                f"{status.get('job_type')} failed: {status.get('error')}\n{status.get('traceback')}",
            )

        job_types = {str(s.get("job_type")) for s in statuses}
        self.assertEqual(job_types, {JobType.GNN_FINE_TUNE.value, JobType.OCR_FINE_TUNE.value})

        # Both lineages promoted a manuscript-local checkpoint.
        layout_registry = manuscript_layout_registry.load_registry(
            self.manuscript_root, base_checkpoint_path=BASE_GNN_CHECKPOINT
        )
        ocr_registry = manuscript_ocr_registry.load_registry(
            self.manuscript_root, base_checkpoint_path=BASE_OCR_CHECKPOINT
        )
        self.assertNotEqual(layout_registry.active_checkpoint_id(), "base")
        self.assertNotEqual(ocr_registry.active_checkpoint_id(), "base")
        self.assertTrue(layout_registry.active_checkpoint().is_file())
        self.assertTrue(ocr_registry.active_checkpoint().is_file())
        self.assertEqual(layout_registry.consumed_page_ids(), [SOURCE_PAGE])
        self.assertTrue(ocr_registry.has_consumed_revision(SOURCE_PAGE))

        # Neither lineage leaked into the other.
        self.assertEqual(layout_registry.pending_layout_work(), [])
        self.assertEqual(ocr_registry.pending_ocr_work(), [])
        layout_ids = set(layout_registry.data["checkpoints"])
        ocr_ids = set(ocr_registry.data["checkpoints"])
        self.assertEqual(layout_ids & ocr_ids, {"base"})
        self.assertNotEqual(
            layout_registry.data["base_checkpoint"]["path"],
            ocr_registry.data["base_checkpoint"]["path"],
        )
        self.assertNotEqual(
            layout_registry.active_checkpoint().resolve(),
            ocr_registry.active_checkpoint().resolve(),
        )

        # The layout checkpoint is a usable GNN, and page load would pick it up.
        import torch

        checkpoint = torch.load(
            layout_registry.active_checkpoint(), map_location="cpu", weights_only=False
        )
        self.assertIsInstance(checkpoint["model"], torch.nn.Module)
        resolved_path, resolved_id = layout_runtime.active_layout_checkpoint(
            self.manuscript_root, base_checkpoint_path=BASE_GNN_CHECKPOINT
        )
        self.assertEqual(Path(resolved_path), layout_registry.active_checkpoint().resolve())
        self.assertEqual(resolved_id, layout_registry.active_checkpoint_id())

        # Job telemetry records both families in one manuscript-level log.
        events_path = layout_registry.telemetry_root / "job_events.jsonl"
        recorded = [json.loads(line) for line in events_path.read_text().splitlines() if line.strip()]
        recorded_types = {record["job_type"] for record in recorded}
        self.assertIn(JobType.GNN_FINE_TUNE.value, recorded_types)
        self.assertIn(JobType.OCR_FINE_TUNE.value, recorded_types)


if __name__ == "__main__":
    unittest.main()
