"""Unit tests for layout (GNN) active learning.

These cover the parts that decide *whether* and *what* to train, without ever
running a training step: the supervision boundary, revision identity, snapshot
contents, job construction, rebase, backfill adoption, and the non-blocking
checkpoint resolution that page load depends on.

The training steps themselves are covered by
`test_layout_active_learning_e2e.py`, which is slow and needs a GPU.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock


TESTS_ROOT = Path(__file__).resolve().parent
APP_ROOT = TESTS_ROOT.parent
REPO_ROOT = APP_ROOT.parent

if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from job_orchestrator import JobOrchestrator, JobType, QueuedJob
from manuscript_layout_registry import GRAPH_FORMAT_SUFFIXES, load_registry
import layout_active_learning_runtime as runtime


class _StubOrchestrator:
    def __init__(self):
        self.jobs = []

    def enqueue(self, job):
        self.jobs.append(job)
        return getattr(job, "job_id", f"job-{len(self.jobs)}")

    def get_job_status(self, job_id):
        return {}


def _write_graph_files(graph_dir: Path, page_id: str, *, node_count=4, edges=((0, 1), (2, 3))):
    graph_dir.mkdir(parents=True, exist_ok=True)
    (graph_dir / f"{page_id}_dims.txt").write_text("100.0 200.0\n", encoding="utf-8")
    (graph_dir / f"{page_id}_inputs_normalized.txt").write_text(
        "\n".join(f"{0.1 * i:.6f} {0.2 * i:.6f} 0.000000" for i in range(node_count)) + "\n",
        encoding="utf-8",
    )
    (graph_dir / f"{page_id}_inputs_unnormalized.txt").write_text(
        "\n".join(f"{10.0 * i:.6f} {20.0 * i:.6f} 0.000000" for i in range(node_count)) + "\n",
        encoding="utf-8",
    )
    (graph_dir / f"{page_id}_edges.txt").write_text(
        "".join(f"{u} {v}\n" for u, v in edges), encoding="utf-8"
    )
    (graph_dir / f"{page_id}_labels_textline.txt").write_text(
        "".join(f"{i // 2}\n" for i in range(node_count)), encoding="utf-8"
    )
    (graph_dir / f"{page_id}_labels_textbox.txt").write_text(
        "".join("0\n" for _ in range(node_count)), encoding="utf-8"
    )


def _graph_payload(node_count=4, edges=((0, 1), (2, 3))):
    return {
        "nodes": [{"x": 10.0 * i, "y": 20.0 * i, "s": 0.0} for i in range(node_count)],
        "edges": [{"source": u, "target": v, "label": 1} for u, v in edges],
    }


class LayoutActiveLearningUnitTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.manuscript_root = self.root / "manuscript"
        self.manuscript_root.mkdir(parents=True)
        self.base_checkpoint = self.root / "base_gnn.pt"
        self.base_checkpoint.write_bytes(b"not-a-real-checkpoint")
        self.recipe = REPO_ROOT / "app" / "pretrained_gnn" / "gnn_active_learning.yaml"
        runtime._RUNTIME_STATE.update(
            {
                "base_checkpoint_path": str(self.base_checkpoint),
                "recipe_config_path": str(self.recipe),
                "orchestrator": None,
            }
        )
        self.orchestrator = _StubOrchestrator()

    def tearDown(self):
        self._tmp.cleanup()

    def _live_graph_dir(self) -> Path:
        return self.manuscript_root / "layout_analysis_output" / "gnn-format"

    def _save(self, page="p1", *, enabled=True, save_scope="layout", save_intent="commit", payload=None):
        return runtime.handle_post_layout_save(
            manuscript="m",
            page=page,
            save_intent=save_intent,
            save_scope=save_scope,
            layout_active_learning_enabled=enabled,
            graph_payload=payload if payload is not None else _graph_payload(),
            manuscript_root=self.manuscript_root,
            base_checkpoint_path=self.base_checkpoint,
            orchestrator=self.orchestrator,
        )

    # ---- supervision boundary --------------------------------------------

    def test_layout_commit_save_is_supervision_and_text_only_save_is_not(self):
        self.assertTrue(runtime.is_layout_supervision("commit", "layout", _graph_payload()))
        self.assertFalse(runtime.is_layout_supervision("draft", "layout", _graph_payload()))
        self.assertFalse(runtime.is_layout_supervision("commit", "text_only", _graph_payload()))
        self.assertFalse(runtime.is_layout_supervision("commit", "layout", {"nodes": []}))

    def test_unedited_layout_commit_still_counts_as_supervision(self):
        # An accepted prediction is a valid label: the human saved it.
        _write_graph_files(self._live_graph_dir(), "p1")
        result = self._save()
        self.assertTrue(result["supervision_present"])
        self.assertTrue(result["entered_active_learning"])

    def test_draft_and_text_only_saves_queue_nothing(self):
        _write_graph_files(self._live_graph_dir(), "p1")
        for scope, intent in (("layout", "draft"), ("text_only", "commit")):
            result = self._save(save_scope=scope, save_intent=intent)
            self.assertFalse(result["supervision_present"])
            self.assertEqual(result["queued_job_ids"], [])
        self.assertEqual(self.orchestrator.jobs, [])

    def test_disabled_toggle_records_the_revision_but_queues_no_training(self):
        _write_graph_files(self._live_graph_dir(), "p1")
        result = self._save(enabled=False)
        self.assertTrue(result["supervision_present"])
        self.assertFalse(result["entered_active_learning"])
        self.assertEqual(self.orchestrator.jobs, [])
        registry = load_registry(self.manuscript_root, base_checkpoint_path=self.base_checkpoint)
        self.assertIsNotNone(registry.latest_supervised_commit_revision("p1"))

    # ---- revision identity -----------------------------------------------

    def test_resaving_an_unchanged_graph_is_a_duplicate_and_queues_no_second_job(self):
        _write_graph_files(self._live_graph_dir(), "p1")
        first = self._save()
        second = self._save()
        self.assertFalse(first["revision"]["is_duplicate"])
        self.assertTrue(second["revision"]["is_duplicate"])
        self.assertEqual(len(self.orchestrator.jobs), 1)

    def test_sub_pixel_coordinate_noise_is_not_a_layout_edit(self):
        jittered = _graph_payload()
        jittered["nodes"][0]["x"] += 0.001
        self.assertEqual(
            runtime.build_layout_revision_hash(_graph_payload()),
            runtime.build_layout_revision_hash(jittered),
        )

    def test_edge_order_does_not_change_revision_identity_but_edge_content_does(self):
        reordered = _graph_payload(edges=((2, 3), (0, 1)))
        changed = _graph_payload(edges=((0, 1),))
        baseline = runtime.build_layout_revision_hash(_graph_payload())
        self.assertEqual(baseline, runtime.build_layout_revision_hash(reordered))
        self.assertNotEqual(baseline, runtime.build_layout_revision_hash(changed))

    def test_textbox_relabelling_alone_is_a_new_revision(self):
        baseline = runtime.build_layout_revision_hash(_graph_payload(), [0, 0, 0, 0])
        relabelled = runtime.build_layout_revision_hash(_graph_payload(), [0, 0, 1, 1])
        self.assertNotEqual(baseline, relabelled)

    # ---- snapshots --------------------------------------------------------

    def test_snapshot_freezes_the_corrected_graph_against_later_edits(self):
        _write_graph_files(self._live_graph_dir(), "p1", edges=((0, 1), (2, 3)))
        self._save()
        registry = load_registry(self.manuscript_root, base_checkpoint_path=self.base_checkpoint)
        snapshot = registry.revision_graph_dir("p1", 1)
        for suffix in GRAPH_FORMAT_SUFFIXES:
            self.assertTrue((snapshot / f"p1{suffix}").is_file(), suffix)
        frozen = (snapshot / "p1_edges.txt").read_text(encoding="utf-8")

        # The user keeps editing while the job is queued.
        _write_graph_files(self._live_graph_dir(), "p1", edges=((0, 3),))
        self.assertEqual((snapshot / "p1_edges.txt").read_text(encoding="utf-8"), frozen)

    def test_saving_without_corrected_graph_files_degrades_instead_of_failing_the_save(self):
        # Layout learning is a follow-up to the save, never a gate on it. With
        # no graph files to snapshot there is nothing to train on, so it must
        # skip and say so rather than raise into the save route.
        result = self._save()
        self.assertFalse(result["supervision_present"])
        self.assertFalse(result["entered_active_learning"])
        self.assertEqual(result["queued_job_ids"], [])
        self.assertIn("corrected graph files are missing", result["warning"])
        self.assertEqual(self.orchestrator.jobs, [])

        registry = load_registry(self.manuscript_root, base_checkpoint_path=self.base_checkpoint)
        self.assertIsNone(registry.latest_revision("p1"))

    # ---- job construction -------------------------------------------------

    def test_first_page_queues_an_incremental_fine_tune_from_the_active_checkpoint(self):
        _write_graph_files(self._live_graph_dir(), "p1")
        self._save()
        job = self.orchestrator.jobs[0]
        self.assertEqual(job.job_type, JobType.GNN_FINE_TUNE.value)
        self.assertTrue(job.isolated)
        self.assertEqual(job.resource_name, "gpu")
        self.assertEqual(job.payload["parent_checkpoint_id"], "base")
        self.assertEqual(job.payload["history_revision_refs"], [])

    def test_second_page_replays_the_first_page_only_after_it_was_consumed(self):
        _write_graph_files(self._live_graph_dir(), "p1")
        self._save(page="p1")
        _write_graph_files(self._live_graph_dir(), "p2")
        self._save(page="p2")
        # p1 has not been folded into a promoted checkpoint yet, so there is no
        # approved history to replay.
        self.assertEqual(self.orchestrator.jobs[1].payload["history_revision_refs"], [])

        registry = load_registry(self.manuscript_root, base_checkpoint_path=self.base_checkpoint)
        registry.mark_candidate({"candidate_id": "c1", "checkpoint_path": self.base_checkpoint})
        registry.promote_candidate("c1", {"passed": True})
        registry.mark_revision_consumed("p1", 1, "c1")

        _write_graph_files(self._live_graph_dir(), "p3")
        self._save(page="p3")
        history = self.orchestrator.jobs[2].payload["history_revision_refs"]
        self.assertEqual(history, [{"page_id": "p1", "revision_number": 1}])

    def test_recorrecting_a_consumed_page_queues_a_pooled_rebase_once(self):
        _write_graph_files(self._live_graph_dir(), "p1")
        self._save(page="p1")
        registry = load_registry(self.manuscript_root, base_checkpoint_path=self.base_checkpoint)
        registry.mark_candidate({"candidate_id": "c1", "checkpoint_path": self.base_checkpoint})
        registry.promote_candidate("c1", {"passed": True})
        registry.mark_revision_consumed("p1", 1, "c1")

        _write_graph_files(self._live_graph_dir(), "p1", edges=((0, 2),))
        result = self._save(page="p1", payload=_graph_payload(edges=((0, 2),)))
        self.assertTrue(result["entered_active_learning"])
        rebase_jobs = [j for j in self.orchestrator.jobs if j.job_type == JobType.GNN_REBASE.value]
        self.assertEqual(len(rebase_jobs), 1)
        self.assertEqual(rebase_jobs[0].payload["revision_refs"], [{"page_id": "p1", "revision_number": 2}])

        registry = load_registry(self.manuscript_root, base_checkpoint_path=self.base_checkpoint)
        registry.enqueue_pending_job(
            {"job_id": "queued-rebase", "job_type": JobType.GNN_REBASE.value, "state": "queued"}
        )
        _write_graph_files(self._live_graph_dir(), "p1", edges=((1, 3),))
        self._save(page="p1", payload=_graph_payload(edges=((1, 3),)))
        rebase_jobs = [j for j in self.orchestrator.jobs if j.job_type == JobType.GNN_REBASE.value]
        self.assertEqual(len(rebase_jobs), 1, "a second rebase must not stack on a pending one")

    # ---- backfill ---------------------------------------------------------

    def test_backfill_adopts_pages_corrected_before_the_feature_existed(self):
        graph_dir = self._live_graph_dir()
        for page_id in ("a1", "a2", "a3"):
            _write_graph_files(graph_dir, page_id)

        result = runtime.queue_layout_backfill(
            manuscript="m",
            manuscript_root=self.manuscript_root,
            base_checkpoint_path=self.base_checkpoint,
            orchestrator=self.orchestrator,
        )
        self.assertTrue(result["queued"])
        self.assertEqual(result["page_count"], 3)
        job = self.orchestrator.jobs[0]
        self.assertEqual(job.job_type, JobType.GNN_BACKFILL.value)
        self.assertEqual(len(job.payload["revision_refs"]), 3)

        registry = load_registry(self.manuscript_root, base_checkpoint_path=self.base_checkpoint)
        for page_id in ("a1", "a2", "a3"):
            self.assertTrue(registry.revision_graph_dir(page_id, 1).is_dir())

    def test_backfill_is_a_no_op_without_corrected_pages(self):
        result = runtime.queue_layout_backfill(
            manuscript="m",
            manuscript_root=self.manuscript_root,
            base_checkpoint_path=self.base_checkpoint,
            orchestrator=self.orchestrator,
        )
        self.assertFalse(result["queued"])
        self.assertEqual(result["reason"], "no_corrected_pages")
        self.assertEqual(self.orchestrator.jobs, [])

    def test_status_reports_how_many_corrected_pages_are_still_untrained(self):
        graph_dir = self._live_graph_dir()
        for page_id in ("a1", "a2"):
            _write_graph_files(graph_dir, page_id)
        status = runtime.summarize_manuscript_layout_active_learning(
            self.manuscript_root, base_checkpoint_path=self.base_checkpoint
        )
        self.assertEqual(status["corrected_page_count"], 2)
        self.assertEqual(status["trained_page_count"], 0)
        self.assertEqual(status["untrained_corrected_page_count"], 2)
        self.assertTrue(status["backfill_available"])

    # ---- checkpoint resolution -------------------------------------------

    def test_page_load_reads_the_promoted_checkpoint_and_ignores_in_flight_work(self):
        _write_graph_files(self._live_graph_dir(), "p1")
        self._save()
        path, checkpoint_id = runtime.active_layout_checkpoint(
            self.manuscript_root, base_checkpoint_path=self.base_checkpoint
        )
        # A job is queued but nothing is promoted, so page load still gets base.
        self.assertEqual(checkpoint_id, "base")
        self.assertEqual(Path(path), self.base_checkpoint.resolve())

        promoted = self.root / "promoted.pt"
        promoted.write_bytes(b"promoted")
        registry = load_registry(self.manuscript_root, base_checkpoint_path=self.base_checkpoint)
        registry.mark_candidate({"candidate_id": "c1", "checkpoint_path": promoted})
        registry.promote_candidate("c1", {"passed": True})

        path, checkpoint_id = runtime.active_layout_checkpoint(
            self.manuscript_root, base_checkpoint_path=self.base_checkpoint
        )
        self.assertEqual(checkpoint_id, "c1")
        self.assertEqual(Path(path), promoted.resolve())

    def test_a_missing_checkpoint_file_falls_back_instead_of_breaking_page_load(self):
        registry = load_registry(self.manuscript_root, base_checkpoint_path=self.base_checkpoint)
        vanished = self.root / "gone.pt"
        vanished.write_bytes(b"x")
        registry.mark_candidate({"candidate_id": "c1", "checkpoint_path": vanished})
        registry.promote_candidate("c1", {"passed": True})
        vanished.unlink()

        path, checkpoint_id = runtime.active_layout_checkpoint(
            self.manuscript_root, base_checkpoint_path=self.base_checkpoint
        )
        self.assertEqual(checkpoint_id, "base")
        self.assertEqual(Path(path), self.base_checkpoint.resolve())

    # ---- augmentation seeding --------------------------------------------

    def test_augmentation_page_index_is_stable_as_more_pages_are_added(self):
        registry = load_registry(self.manuscript_root, base_checkpoint_path=self.base_checkpoint)
        first = runtime._augmentation_page_index(registry, "p1")
        runtime._augmentation_page_index(registry, "p2")
        runtime._augmentation_page_index(registry, "p3")
        self.assertEqual(runtime._augmentation_page_index(registry, "p1"), first)

        reloaded = load_registry(self.manuscript_root, base_checkpoint_path=self.base_checkpoint)
        self.assertEqual(runtime._augmentation_page_index(reloaded, "p1"), first)

    # ---- inference outranks training --------------------------------------

    def test_automatic_training_is_queued_with_a_head_start(self):
        _write_graph_files(self._live_graph_dir(), "p1")
        self._save()
        job = self.orchestrator.jobs[0]
        self.assertIsNotNone(
            job.not_before,
            "an automatic layout fine-tune must leave room for an imminent read",
        )
        delay = (
            datetime.fromisoformat(job.not_before) - datetime.now(timezone.utc)
        ).total_seconds()
        self.assertGreater(delay, 0)
        self.assertLessEqual(delay, runtime.training_start_delay_seconds() + 1)

    def test_user_requested_backfill_starts_immediately(self):
        graph_dir = self._live_graph_dir()
        for page_id in ("a1", "a2"):
            _write_graph_files(graph_dir, page_id)
        runtime.queue_layout_backfill(
            manuscript="m",
            manuscript_root=self.manuscript_root,
            base_checkpoint_path=self.base_checkpoint,
            orchestrator=self.orchestrator,
        )
        # The user asked for this one and is watching it, so it does not wait.
        self.assertIsNone(self.orchestrator.jobs[0].not_before)

    def test_interactive_work_pushes_queued_layout_training_back(self):
        orchestrator = JobOrchestrator()
        try:
            _write_graph_files(self._live_graph_dir(), "p1")
            self._save(enabled=True)  # queues against the stub, not this one
            job = QueuedJob(
                job_type=JobType.GNN_FINE_TUNE.value,
                manuscript="m",
                manuscript_root=str(self.manuscript_root),
                payload={},
            )
            job_id = orchestrator.enqueue(job)
            before = orchestrator.get_job_status(job_id).get("not_before")
            self.assertIsNone(before)

            moved = runtime.defer_layout_training_for_interactive_work(
                orchestrator=orchestrator, delay_seconds=30
            )
            self.assertEqual(moved, [job_id])
            after = orchestrator.get_job_status(job_id).get("not_before")
            self.assertIsNotNone(after)
            self.assertGreater(
                (datetime.fromisoformat(after) - datetime.now(timezone.utc)).total_seconds(),
                0,
            )
        finally:
            orchestrator.shutdown_workers()

    def test_deferring_never_touches_ocr_training(self):
        orchestrator = JobOrchestrator()
        try:
            ocr_job_id = orchestrator.enqueue(
                QueuedJob(
                    job_type=JobType.OCR_FINE_TUNE.value,
                    manuscript="m",
                    manuscript_root=str(self.manuscript_root),
                    payload={},
                )
            )
            runtime.defer_layout_training_for_interactive_work(
                orchestrator=orchestrator, delay_seconds=30
            )
            self.assertIsNone(orchestrator.get_job_status(ocr_job_id).get("not_before"))
        finally:
            orchestrator.shutdown_workers()

    # ---- checkpoint pruning ------------------------------------------------

    def test_promotion_prunes_checkpoints_the_fallback_ladder_cannot_reach(self):
        registry = load_registry(self.manuscript_root, base_checkpoint_path=self.base_checkpoint)
        created = []
        for index in range(1, 5):
            checkpoint_id = f"c{index}"
            path = registry.checkpoints_root / checkpoint_id / "model.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            # Stand in for a real ~51 MB layout checkpoint.
            path.write_bytes(b"checkpoint")
            registry.mark_candidate({"candidate_id": checkpoint_id, "checkpoint_path": path})
            registry.promote_candidate(checkpoint_id, {"passed": True})
            created.append(checkpoint_id)
            runtime._prune_obsolete_checkpoints(registry)

        # Active (c4), its fallback (c3) and base survive; older ones do not.
        self.assertTrue((registry.checkpoints_root / "c4").is_dir())
        self.assertTrue((registry.checkpoints_root / "c3").is_dir())
        self.assertFalse((registry.checkpoints_root / "c1").is_dir())
        self.assertFalse((registry.checkpoints_root / "c2").is_dir())

        # Pruned records stay for lineage, and the base checkpoint is untouched.
        self.assertEqual(registry.data["checkpoints"]["c1"]["status"], "pruned")
        self.assertTrue(self.base_checkpoint.is_file())
        self.assertEqual(registry.active_checkpoint_id(), "c4")

    def test_pruning_can_be_disabled_for_debugging(self):
        registry = load_registry(self.manuscript_root, base_checkpoint_path=self.base_checkpoint)
        for index in (1, 2, 3):
            checkpoint_id = f"d{index}"
            path = registry.checkpoints_root / checkpoint_id / "model.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"checkpoint")
            registry.mark_candidate({"candidate_id": checkpoint_id, "checkpoint_path": path})
            registry.promote_candidate(checkpoint_id, {"passed": True})
        with mock.patch.dict(
            os.environ, {"LAYOUT_RUNTIME_PRUNE_OBSOLETE_CHECKPOINTS": "0"}, clear=False
        ):
            summary = runtime._prune_obsolete_checkpoints(registry)
        self.assertFalse(summary["enabled"])
        self.assertTrue((registry.checkpoints_root / "d1").is_dir())

    # ---- isolation from the OCR lineage -----------------------------------

    def test_layout_state_lives_beside_ocr_state_and_routes_its_own_events(self):
        _write_graph_files(self._live_graph_dir(), "p1")
        self._save()
        self.assertTrue((self.manuscript_root / "active_learning" / "layout" / "registry.json").is_file())
        self.assertFalse((self.manuscript_root / "active_learning" / "recognition").exists())

        import ocr_active_learning_runtime
        from active_learning_jobs import handle_orchestrator_event

        ocr_base = self.root / "base_ocr.pth"
        ocr_base.write_bytes(b"not-a-real-ocr-checkpoint")
        ocr_active_learning_runtime.configure_runtime(ocr_base)

        # An OCR job must land in the OCR registry, not the layout one.
        handle_orchestrator_event(
            "queued",
            {
                "job_id": "ocr-1",
                "job_type": JobType.OCR_FINE_TUNE.value,
                "priority": 2,
                "state": "queued",
                "created_at": "2026-01-01T00:00:00+00:00",
                "payload": {"manuscript_root": str(self.manuscript_root), "page_id": "p1"},
            },
        )
        registry = load_registry(self.manuscript_root, base_checkpoint_path=self.base_checkpoint)
        self.assertEqual(registry.pending_layout_work(), [])
        ocr_registry_path = self.manuscript_root / "active_learning" / "recognition" / "registry.json"
        self.assertTrue(ocr_registry_path.is_file())
        self.assertEqual(len(json.loads(ocr_registry_path.read_text())["pending_jobs"]), 1)

        handle_orchestrator_event(
            "queued",
            {
                "job_id": "gnn-1",
                "job_type": JobType.GNN_FINE_TUNE.value,
                "priority": 2,
                "state": "queued",
                "created_at": "2026-01-01T00:00:00+00:00",
                "payload": {"manuscript_root": str(self.manuscript_root), "page_id": "p1"},
            },
        )
        registry = load_registry(self.manuscript_root, base_checkpoint_path=self.base_checkpoint)
        self.assertEqual(len(registry.pending_layout_work()), 1)
        self.assertEqual(registry.get_status()["code"], "queued")


if __name__ == "__main__":
    unittest.main()
