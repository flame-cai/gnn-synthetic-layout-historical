"""End-to-end verification that layout (GNN) active learning actually learns.

Slow: each test runs real augmentation and real training. Needs the pretrained
GNN checkpoint and the released ground-truth graphs; skips cleanly without them.

Two complementary checks, because either one alone is weak evidence:

* `test_normal_finetuning_improves_held_out_layout` is the realistic case. It
  trains on the three pages of the released `circular_layout` fold_1 and
  measures text-line F1 on that fold's six held-out pages. Held-out pages are
  scored on their ground-truth node set, so predicted and ground-truth edges are
  directly comparable and the number measures the GNN's actual job.

* `test_extreme_all_edges_deleted_suppresses_edges_on_later_pages` is the
  falsifiable case. One page is corrected under a deliberately absurd
  convention -- every node isolated, no edges at all -- and a model that really
  learned from it must then decline to connect nodes on a *different*,
  never-seen page. A pipeline that silently trained on nothing, trained on the
  wrong page, or failed to reload the promoted checkpoint cannot pass this.

A third test checks the two lineages coexist: OCR and layout jobs sharing one
orchestrator must both promote without corrupting each other's registry.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


TESTS_ROOT = Path(__file__).resolve().parent
APP_ROOT = TESTS_ROOT.parent
REPO_ROOT = APP_ROOT.parent

for path in (str(APP_ROOT), str(REPO_ROOT), str(REPO_ROOT / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

from job_orchestrator import JobType
from manuscript_layout_registry import GRAPH_FORMAT_SUFFIXES, load_registry
import layout_active_learning_runtime as runtime

from gnn_training.gnn_finetuning import _process_graph_or_fail, load_gnn_finetuning_recipe
from gnn_training.training.engine import evaluate
from torch_geometric.loader import DataLoader


BASE_GNN_CHECKPOINT = APP_ROOT / "pretrained_gnn" / "v2.pt"
LAYOUT_RECIPE = APP_ROOT / "pretrained_gnn" / "gnn_active_learning.yaml"
RELEASE_ROOT = REPO_ROOT / "dataset_release"
RELEASE_MANUSCRIPT = "circular_layout"


def _release_graph_dir() -> Path:
    return RELEASE_ROOT / "manuscripts" / RELEASE_MANUSCRIPT / "labels" / "graph"


def _release_fold(fold_id: str = "fold_1") -> tuple[list[str], list[str]]:
    payload = json.loads((RELEASE_ROOT / "folds" / f"{RELEASE_MANUSCRIPT}.json").read_text())
    for fold in payload["folds"]:
        if fold["fold_id"] == fold_id:
            return list(fold["train_page_ids"]), list(fold["test_page_ids"])
    raise KeyError(fold_id)


def _copy_page_graph(source_dir: Path, target_dir: Path, page_id: str) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for suffix in GRAPH_FORMAT_SUFFIXES:
        source = source_dir / f"{page_id}{suffix}"
        if source.is_file():
            shutil.copy2(source, target_dir / source.name)


def _isolate_every_node(graph_dir: Path, page_id: str) -> int:
    """Rewrite one page as 'every node is its own text line, no edges'.

    This is the fake annotation convention the extreme test trains on. Only the
    text-line labels are load-bearing -- ground-truth edges are rebuilt from
    them -- but the edge file is emptied too so the snapshot is self-consistent.
    """
    nodes = np.loadtxt(graph_dir / f"{page_id}_inputs_normalized.txt").reshape(-1, 3)
    node_count = nodes.shape[0]
    (graph_dir / f"{page_id}_edges.txt").write_text("", encoding="utf-8")
    (graph_dir / f"{page_id}_labels_textline.txt").write_text(
        "".join(f"{index}\n" for index in range(node_count)), encoding="utf-8"
    )
    return node_count


class _DirectOrchestrator:
    """Runs each queued job inline, so a test can assert on its result."""

    def __init__(self):
        self.jobs = []
        self.results = []

    def enqueue(self, job):
        self.jobs.append(job)
        result = runtime.dispatch_isolated_job(str(job.job_type), dict(job.payload))
        self.results.append(result)
        return job.job_id

    def get_job_status(self, job_id):
        return {}


class LayoutActiveLearningEndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not BASE_GNN_CHECKPOINT.is_file():
            raise unittest.SkipTest(f"Pretrained GNN checkpoint is missing: {BASE_GNN_CHECKPOINT}")
        if not _release_graph_dir().is_dir():
            raise unittest.SkipTest(f"Released ground-truth graphs are missing: {_release_graph_dir()}")
        cls.recipe = load_gnn_finetuning_recipe(LAYOUT_RECIPE)
        cls.train_pages, cls.test_pages = _release_fold("fold_1")

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.manuscript_root = self.root / "manuscript"
        self.live_graph_dir = self.manuscript_root / "layout_analysis_output" / "gnn-format"
        self.live_graph_dir.mkdir(parents=True)
        runtime._RUNTIME_STATE.update(
            {
                "base_checkpoint_path": str(BASE_GNN_CHECKPOINT.resolve()),
                "recipe_config_path": str(LAYOUT_RECIPE.resolve()),
                "orchestrator": None,
            }
        )

    def tearDown(self):
        self._tmp.cleanup()

    # ---- helpers ----------------------------------------------------------

    def _correct_page(self, page_id: str, orchestrator, *, isolate_nodes: bool = False):
        """Simulate one Layout Mode commit save of an already-corrected page."""
        _copy_page_graph(_release_graph_dir(), self.live_graph_dir, page_id)
        if isolate_nodes:
            _isolate_every_node(self.live_graph_dir, page_id)
        nodes = np.loadtxt(self.live_graph_dir / f"{page_id}_inputs_unnormalized.txt").reshape(-1, 3)
        edges_raw = np.loadtxt(self.live_graph_dir / f"{page_id}_edges.txt", dtype=int, ndmin=2)
        edges = [] if edges_raw.size == 0 else [
            {"source": int(u), "target": int(v), "label": 1} for u, v in edges_raw.reshape(-1, 2)
        ]
        payload = {
            "nodes": [{"x": float(x), "y": float(y), "s": 0.0} for x, y, _ in nodes],
            "edges": edges,
        }
        return runtime.handle_post_layout_save(
            manuscript="release",
            page=page_id,
            save_intent="commit",
            save_scope="layout",
            layout_active_learning_enabled=True,
            graph_payload=payload,
            manuscript_root=self.manuscript_root,
            base_checkpoint_path=BASE_GNN_CHECKPOINT,
            orchestrator=orchestrator,
        )

    def _held_out_metrics(self, checkpoint_path: str | Path, page_ids: list[str]) -> dict:
        """Score a checkpoint on held-out pages, on their ground-truth node set.

        Predicting on the ground-truth nodes is what makes predicted and
        ground-truth edges comparable; on raw CRAFT nodes the two sets differ
        and the comparison would be meaningless.
        """
        graphs = [
            _process_graph_or_fail(page_id, _release_graph_dir(), self.recipe.preprocessing_config)
            for page_id in page_ids
        ]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = checkpoint["model"].to(device)
        loader = DataLoader(graphs, batch_size=1, shuffle=False)
        metrics, _, _ = evaluate(model, loader, torch.nn.CrossEntropyLoss(), device)

        positive_edges = 0
        total_edges = 0
        model.eval()
        with torch.no_grad():
            for graph in graphs:
                graph = graph.to(device)
                predictions = torch.argmax(
                    model(graph.x, graph.edge_index, graph.edge_attr), dim=1
                )
                positive_edges += int(predictions.sum().item())
                total_edges += int(predictions.numel())
                graph.cpu()
        metrics["predicted_positive_edge_rate"] = positive_edges / max(1, total_edges)
        return metrics

    # ---- the realistic case ----------------------------------------------

    def test_normal_finetuning_improves_held_out_layout(self):
        orchestrator = _DirectOrchestrator()
        before = self._held_out_metrics(BASE_GNN_CHECKPOINT, self.test_pages)

        for page_id in self.train_pages:
            result = self._correct_page(page_id, orchestrator)
            self.assertTrue(result["entered_active_learning"], page_id)

        self.assertEqual(len(orchestrator.jobs), len(self.train_pages))
        self.assertTrue(all(r["promoted"] for r in orchestrator.results))

        registry = load_registry(self.manuscript_root, base_checkpoint_path=BASE_GNN_CHECKPOINT)
        active_path = registry.active_checkpoint()
        self.assertNotEqual(active_path.resolve(), BASE_GNN_CHECKPOINT.resolve())
        self.assertEqual(sorted(registry.consumed_page_ids()), sorted(self.train_pages))

        after = self._held_out_metrics(active_path, self.test_pages)
        print(
            f"\n[normal] held-out textline F1 {before['textline_f1_score']:.4f} "
            f"-> {after['textline_f1_score']:.4f}; edge macro F1 "
            f"{before['f1_score_macro']:.4f} -> {after['f1_score_macro']:.4f}"
        )

        # Three corrected pages of a hard circular manuscript should move
        # held-out layout quality up, not merely leave it unchanged.
        self.assertGreater(after["textline_f1_score"], before["textline_f1_score"])
        self.assertGreaterEqual(after["f1_score_macro"], before["f1_score_macro"] - 0.01)

    # ---- the falsifiable case --------------------------------------------

    def test_extreme_all_edges_deleted_suppresses_edges_on_later_pages(self):
        orchestrator = _DirectOrchestrator()
        page_id = self.train_pages[0]
        before = self._held_out_metrics(BASE_GNN_CHECKPOINT, self.test_pages)
        self.assertGreater(
            before["predicted_positive_edge_rate"],
            0.02,
            "the pretrained model must connect nodes, or the test proves nothing",
        )

        result = self._correct_page(page_id, orchestrator, isolate_nodes=True)
        self.assertTrue(result["entered_active_learning"])
        self.assertTrue(orchestrator.results[0]["promoted"])

        registry = load_registry(self.manuscript_root, base_checkpoint_path=BASE_GNN_CHECKPOINT)
        after = self._held_out_metrics(registry.active_checkpoint(), self.test_pages)
        print(
            f"\n[extreme] predicted positive-edge rate on held-out pages "
            f"{before['predicted_positive_edge_rate']:.4f} -> "
            f"{after['predicted_positive_edge_rate']:.4f}"
        )

        # Taught that nothing connects, the model must generalise that to pages
        # it has never seen.
        self.assertLess(after["predicted_positive_edge_rate"], 0.01)
        self.assertLess(
            after["predicted_positive_edge_rate"],
            before["predicted_positive_edge_rate"] / 5.0,
        )

    # ---- the two lineages together ---------------------------------------

    def test_layout_training_reuses_snapshots_and_leaves_ocr_state_untouched(self):
        orchestrator = _DirectOrchestrator()
        page_id = self.train_pages[0]
        self._correct_page(page_id, orchestrator)

        registry = load_registry(self.manuscript_root, base_checkpoint_path=BASE_GNN_CHECKPOINT)
        snapshot = registry.revision_graph_dir(page_id, 1)
        self.assertTrue(snapshot.is_dir())
        augmentations = runtime._augmentation_dir(registry, page_id, 1)
        manifest = json.loads((augmentations / "augmentation_manifest.json").read_text())
        self.assertEqual(manifest["augmentations_per_page"], 50)
        self.assertEqual(len(manifest["augmented_page_ids"]), 50)

        # Layout training must not have created or touched OCR lineage.
        self.assertFalse((self.manuscript_root / "active_learning" / "recognition").exists())

        # A second, identical save is a duplicate: no new revision, no new job.
        result = self._correct_page(page_id, orchestrator)
        self.assertTrue(result["revision"]["is_duplicate"])
        self.assertEqual(len(orchestrator.jobs), 1)

    def test_replace_with_new_layout_repredicts_without_touching_saved_files(self):
        """The 'Replace With New Layout' contract, end to end.

        A saved page normally comes back exactly as the human left it. Only the
        explicit relayout flag re-runs the model, and even then it keeps the
        node set and region labels and writes nothing to disk.
        """
        from gnn_inference import run_gnn_prediction_for_page

        page_id = self.train_pages[0]
        _copy_page_graph(_release_graph_dir(), self.live_graph_dir, page_id)
        graph_dir = self.live_graph_dir
        saved_edges_before = (graph_dir / f"{page_id}_edges.txt").read_text(encoding="utf-8")
        config_path = APP_ROOT / "pretrained_gnn" / "gnn_preprocessing_v2.yaml"

        saved = run_gnn_prediction_for_page(
            str(self.manuscript_root), page_id, str(BASE_GNN_CHECKPOINT), str(config_path)
        )
        self.assertTrue(saved["from_saved_graph"])
        saved_pairs = {tuple(sorted((e["source"], e["target"]))) for e in saved["edges"]}
        self.assertEqual(
            saved_pairs,
            {
                tuple(sorted(map(int, line.split())))
                for line in saved_edges_before.splitlines()
                if line.strip()
            },
            "a saved page must come back exactly as the human left it",
        )

        regenerated = run_gnn_prediction_for_page(
            str(self.manuscript_root),
            page_id,
            str(BASE_GNN_CHECKPOINT),
            str(config_path),
            force_regenerate_edges=True,
        )
        self.assertFalse(regenerated["from_saved_graph"])
        self.assertEqual(len(regenerated["nodes"]), len(saved["nodes"]))
        self.assertGreater(len(regenerated["edges"]), 0)
        regenerated_pairs = {
            tuple(sorted((e["source"], e["target"]))) for e in regenerated["edges"]
        }
        self.assertNotEqual(
            regenerated_pairs, saved_pairs, "the model should predict its own edges"
        )
        # Region labels are per node and the node set is unchanged, so they survive.
        self.assertEqual(regenerated["textbox_labels"], saved["textbox_labels"])
        # Nothing is written; the user keeps the result by saving.
        self.assertEqual(
            (graph_dir / f"{page_id}_edges.txt").read_text(encoding="utf-8"),
            saved_edges_before,
        )

    def test_pooled_backfill_trains_one_checkpoint_from_every_corrected_page(self):
        orchestrator = _DirectOrchestrator()
        for page_id in self.train_pages:
            _copy_page_graph(_release_graph_dir(), self.live_graph_dir, page_id)

        result = runtime.queue_layout_backfill(
            manuscript="release",
            manuscript_root=self.manuscript_root,
            base_checkpoint_path=BASE_GNN_CHECKPOINT,
            orchestrator=orchestrator,
        )
        self.assertTrue(result["queued"])
        self.assertEqual(result["page_count"], len(self.train_pages))

        job = orchestrator.jobs[0]
        self.assertEqual(job.job_type, JobType.GNN_BACKFILL.value)
        job_result = orchestrator.results[0]
        self.assertTrue(job_result["promoted"])
        self.assertEqual(sorted(job_result["page_ids"]), sorted(self.train_pages))
        # Pooled: every page contributes all 50 augmentations to one run.
        self.assertEqual(job_result["training_sample_count"], 50 * len(self.train_pages))

        registry = load_registry(self.manuscript_root, base_checkpoint_path=BASE_GNN_CHECKPOINT)
        self.assertEqual(sorted(registry.consumed_page_ids()), sorted(self.train_pages))
        self.assertFalse(registry.data["needs_rebase"])

        after = self._held_out_metrics(registry.active_checkpoint(), self.test_pages)
        before = self._held_out_metrics(BASE_GNN_CHECKPOINT, self.test_pages)
        print(
            f"\n[backfill] held-out textline F1 {before['textline_f1_score']:.4f} "
            f"-> {after['textline_f1_score']:.4f}"
        )
        self.assertGreater(after["textline_f1_score"], before["textline_f1_score"])


if __name__ == "__main__":
    unittest.main()
