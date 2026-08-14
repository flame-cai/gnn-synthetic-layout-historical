"""Manuscript-local layout (GNN) active learning for the production app.

This is the layout twin of `ocr_active_learning_runtime`. A Layout Mode commit
save becomes a supervised revision; the revision's corrected graph is snapshot
to disk; a background job continues the manuscript's active GNN checkpoint on
it; and the promoted checkpoint is what `run_gnn_prediction_for_page` uses for
the next not-yet-corrected page.

Three things differ from the OCR lineage, and each one is deliberate.

**Supervision boundary.** OCR supervision is a Read Mode `text_only` commit with
non-empty text. Layout supervision is a Layout Mode `layout` commit with at
least one node: saving a page asserts the layout is correct, whether or not the
human had to edit it. Re-saving an unchanged graph is deduplicated by content
hash and queues nothing.

**Training schedule.** A per-save step is genuinely incremental, so it continues
the active checkpoint on the new page plus a 20% replay of earlier pages --
exactly the experiment recipe. A backfill or rebase already has every page in
hand, so it pools them into one run from the base checkpoint instead of walking
a ladder. Same hyperparameters either way; only the data schedule changes.

**Contention.** Inference outranks training, in both directions. GNN inference
runs on every page open, so it must never block: page load reads whatever
checkpoint is currently promoted and ignores in-flight work, and nothing here
calls `preempt_for_interactive`. Conversely, automatic training jobs are queued
with a short `not_before` head start, and every interactive read pushes that
deadline back, so the usual "save the layout, switch to Read Mode, read the
page" sequence finishes its inference while training is still merely queued.
Preemption remains the backstop for a job that did start.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from job_orchestrator import JobOrchestrator, JobPriority, JobType, QueuedJob
from manuscript_layout_registry import (
    GRAPH_FORMAT_SUFFIXES,
    ManuscriptLayoutRegistry,
    load_registry,
)
from profiling import (
    maybe_write_cuda_trace,
    should_capture_cuda_trace,
    summarize_gpu_job,
    write_profile_summary,
)
from telemetry import append_jsonl, utc_now_iso


REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

DEFAULT_LAYOUT_RECIPE_CONFIG = REPO_ROOT / "app" / "pretrained_gnn" / "gnn_active_learning.yaml"

_RUNTIME_STATE = {
    "base_checkpoint_path": None,
    "recipe_config_path": None,
    "orchestrator": None,
}

_MAX_AUGMENTATION_REVISIONS_PER_PAGE_ENV = "LAYOUT_RUNTIME_MAX_AUGMENTATION_REVISIONS"
_RUNTIME_COMPACT_ARTIFACTS_ENV = "LAYOUT_RUNTIME_COMPACT_CHECKPOINT_ARTIFACTS"
_RUNTIME_PRUNE_CHECKPOINTS_ENV = "LAYOUT_RUNTIME_PRUNE_OBSOLETE_CHECKPOINTS"
_TRAINING_START_DELAY_ENV = "LAYOUT_RUNTIME_TRAINING_START_DELAY_SECONDS"
DEFAULT_TRAINING_START_DELAY_SECONDS = 25.0


def training_start_delay_seconds() -> float:
    """How long an automatic layout training job waits before it may start.

    Saving a layout is very often followed within seconds by switching to Read
    Mode and reading that same page. Inference should win that race, and it
    already does -- interactive OCR preempts a running job -- but preemption
    throws the partial training away. In the first real GUI session, 5 of 6
    layout jobs were preempted and restarted from scratch.

    A short head start removes the race instead of resolving it: the reading
    happens while the job is still merely queued, and every interactive read
    pushes the deadline back again. Preemption stays as the backstop for a job
    that did manage to start.
    """
    raw = os.environ.get(_TRAINING_START_DELAY_ENV)
    if raw is None:
        return DEFAULT_TRAINING_START_DELAY_SECONDS
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return DEFAULT_TRAINING_START_DELAY_SECONDS


def _deferred_start_timestamp() -> str | None:
    delay = training_start_delay_seconds()
    if delay <= 0:
        return None
    return (datetime.now(timezone.utc) + timedelta(seconds=delay)).isoformat()


def defer_layout_training_for_interactive_work(
    orchestrator: JobOrchestrator | None = None,
    delay_seconds: float | None = None,
) -> list[str]:
    """Push queued layout training back because interactive work is starting."""
    orchestrator = _get_orchestrator(orchestrator)
    if orchestrator is None or not hasattr(orchestrator, "defer_queued_jobs"):
        return []
    delay = training_start_delay_seconds() if delay_seconds is None else float(delay_seconds)
    return orchestrator.defer_queued_jobs(
        delay,
        job_types={JobType.GNN_FINE_TUNE.value, JobType.GNN_REBASE.value},
    )


# ---------------------------------------------------------------------------
# runtime configuration
# ---------------------------------------------------------------------------


def configure_runtime(
    base_checkpoint_path: str | Path,
    orchestrator: JobOrchestrator | None = None,
    recipe_config_path: str | Path | None = None,
) -> None:
    _RUNTIME_STATE["base_checkpoint_path"] = str(Path(base_checkpoint_path).resolve())
    _RUNTIME_STATE["recipe_config_path"] = str(
        Path(recipe_config_path or DEFAULT_LAYOUT_RECIPE_CONFIG).resolve()
    )
    if orchestrator is not None:
        _RUNTIME_STATE["orchestrator"] = orchestrator
        from active_learning_jobs import handle_orchestrator_event

        orchestrator.set_state_listener(handle_orchestrator_event)
        orchestrator.start_workers()


def _base_checkpoint_path(explicit_base_checkpoint_path: str | Path | None = None) -> str:
    base_checkpoint_path = explicit_base_checkpoint_path or _RUNTIME_STATE.get("base_checkpoint_path")
    if not base_checkpoint_path:
        raise FileNotFoundError(
            "Layout active-learning runtime does not know the base GNN checkpoint path yet."
        )
    return str(Path(base_checkpoint_path).resolve())


def _recipe_config_path(explicit_recipe_config_path: str | Path | None = None) -> str:
    return str(
        Path(
            explicit_recipe_config_path
            or _RUNTIME_STATE.get("recipe_config_path")
            or DEFAULT_LAYOUT_RECIPE_CONFIG
        ).resolve()
    )


def _get_orchestrator(orchestrator: JobOrchestrator | None = None) -> JobOrchestrator | None:
    return orchestrator or _RUNTIME_STATE.get("orchestrator")


def _load_registry_for_manuscript(
    manuscript_root: str | Path, base_checkpoint_path: str | Path | None = None
) -> ManuscriptLayoutRegistry:
    return load_registry(
        manuscript_root, base_checkpoint_path=_base_checkpoint_path(base_checkpoint_path)
    )


def _safe_slug(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in str(value))


def _env_flag_enabled(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


# ---------------------------------------------------------------------------
# supervision boundary and revision identity
# ---------------------------------------------------------------------------


def is_layout_supervision(save_intent: str, save_scope: str, graph_payload: dict | None) -> bool:
    """A Layout Mode commit save with at least one node."""
    node_count = len((graph_payload or {}).get("nodes") or [])
    return (
        str(save_intent or "") == "commit"
        and str(save_scope or "") == "layout"
        and node_count > 0
    )


def build_layout_revision_hash(
    graph_payload: dict | None,
    textbox_labels=None,
    reading_direction_annotations=None,
) -> str:
    """Hash the graph as saved, so an unchanged re-save is a duplicate.

    Node coordinates are rounded to a tenth of a pixel: the frontend can emit
    float noise that is not a real layout edit, and a duplicate revision is the
    mechanism that stops a redundant training job.
    """
    graph = dict(graph_payload or {})
    nodes = [
        [round(float(node.get("x", 0.0)), 1), round(float(node.get("y", 0.0)), 1)]
        for node in (graph.get("nodes") or [])
    ]
    edges = sorted(
        tuple(sorted((int(edge["source"]), int(edge["target"]))))
        for edge in (graph.get("edges") or [])
        if edge.get("source") is not None and edge.get("target") is not None
    )
    payload = {
        "nodes": nodes,
        "edges": [list(edge) for edge in edges],
        "textbox_labels": list(textbox_labels or []),
        "reading_direction_annotations": _canonical_reading_annotations(
            reading_direction_annotations
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_reading_annotations(annotations) -> list:
    if not annotations:
        return []
    if isinstance(annotations, dict):
        annotations = annotations.get("lineAnnotations") or []
    canonical = []
    for annotation in annotations:
        if not isinstance(annotation, dict):
            continue
        canonical.append(
            {
                "line": annotation.get("resolved_line_numeric_id")
                or annotation.get("lineNumericId"),
                "direction": [
                    round(float(value), 4)
                    for value in (annotation.get("reading_direction") or [])
                ],
            }
        )
    return sorted(canonical, key=lambda item: json.dumps(item, sort_keys=True))


# ---------------------------------------------------------------------------
# snapshots and augmentation caching
# ---------------------------------------------------------------------------


def live_graph_dir(manuscript_root: str | Path) -> Path:
    return Path(manuscript_root) / "layout_analysis_output" / "gnn-format"


def page_has_corrected_graph(manuscript_root: str | Path, page_id: str) -> bool:
    graph_dir = live_graph_dir(manuscript_root)
    return all((graph_dir / f"{page_id}{suffix}").is_file() for suffix in GRAPH_FORMAT_SUFFIXES)


def discover_corrected_page_ids(manuscript_root: str | Path) -> list[str]:
    graph_dir = live_graph_dir(manuscript_root)
    if not graph_dir.is_dir():
        return []
    page_ids = {
        path.name[: -len("_edges.txt")]
        for path in graph_dir.glob("*_edges.txt")
        if path.is_file()
    }
    return sorted(
        page_id
        for page_id in page_ids
        if page_has_corrected_graph(manuscript_root, page_id)
    )


def _snapshot_page_revision(
    registry: ManuscriptLayoutRegistry, page_id: str, revision_number: int
) -> Path:
    """Freeze the six corrected graph files for this revision.

    Training reads the snapshot, never the live files, so a job that starts
    while the user keeps editing trains on the graph that was actually saved.
    """
    snapshot_root = registry.revision_snapshot_root(page_id, revision_number)
    if snapshot_root.exists():
        shutil.rmtree(snapshot_root)
    graph_dir = snapshot_root / "gnn-format"
    graph_dir.mkdir(parents=True, exist_ok=True)

    source_dir = live_graph_dir(registry.manuscript_root)
    missing = []
    for suffix in GRAPH_FORMAT_SUFFIXES:
        source = source_dir / f"{page_id}{suffix}"
        if not source.is_file():
            missing.append(source.name)
            continue
        shutil.copy2(source, graph_dir / source.name)
    if missing:
        raise FileNotFoundError(
            f"Cannot snapshot layout revision {page_id}#{revision_number}; "
            f"missing corrected graph files: {missing}"
        )
    return snapshot_root


def _augmentation_page_index(registry: ManuscriptLayoutRegistry, page_id: str) -> int:
    """Stable per-page seed offset for augmentation.

    `augment_corrected_graph_page` seeds from `base_seed + page_index * count`,
    so the index has to be stable for a page across the manuscript's whole life,
    not derived from its position in a list that grows.
    """
    indices = registry.data.setdefault("augmentation_page_index", {})
    page_id = str(page_id)
    if page_id not in indices:
        indices[page_id] = len(indices)
        registry.save()
    return int(indices[page_id])


def _augmentation_dir(
    registry: ManuscriptLayoutRegistry, page_id: str, revision_number: int
) -> Path:
    return (
        registry.runtime_root
        / "augmentations"
        / _safe_slug(str(page_id))
        / f"rev_{int(revision_number):04d}"
    )


def _prune_stale_augmentations(
    registry: ManuscriptLayoutRegistry, page_id: str, keep_revision_numbers: set[int]
) -> None:
    """Keep only the augmentations of revisions that can still be replayed."""
    limit = int(os.environ.get(_MAX_AUGMENTATION_REVISIONS_PER_PAGE_ENV, "2") or 2)
    page_root = registry.runtime_root / "augmentations" / _safe_slug(str(page_id))
    if not page_root.is_dir():
        return
    revision_dirs = sorted(
        (path for path in page_root.iterdir() if path.is_dir() and path.name.startswith("rev_")),
        key=lambda path: path.name,
    )
    keep_names = {f"rev_{int(number):04d}" for number in keep_revision_numbers}
    removable = [path for path in revision_dirs if path.name not in keep_names]
    surplus = len(revision_dirs) - max(1, limit)
    for path in removable[: max(0, surplus)]:
        shutil.rmtree(path, ignore_errors=True)


def _revision_ref(page_id: str, revision_number: int) -> dict:
    return {"page_id": str(page_id), "revision_number": int(revision_number)}


def _revision_refs_from_revisions(revisions: list[dict]) -> list[dict]:
    return [
        _revision_ref(revision["page_id"], revision["revision_number"])
        for revision in revisions or []
    ]


def register_existing_corrected_pages(
    registry: ManuscriptLayoutRegistry,
) -> list[dict]:
    """Adopt pages corrected before layout active learning was switched on.

    Those pages have `gnn-format` files but no revision and no snapshot. Record
    one supervised commit revision each and snapshot it, so a backfill can train
    on the work the annotator already did.
    """
    adopted = []
    for page_id in discover_corrected_page_ids(registry.manuscript_root):
        if registry.latest_supervised_commit_revision(page_id) is not None:
            continue
        graph_dir = live_graph_dir(registry.manuscript_root)
        node_count, edge_count = _graph_counts_from_dir(graph_dir, page_id)
        if node_count <= 0:
            continue
        content_hash = _graph_dir_content_hash(graph_dir, page_id)
        revision = registry.record_page_revision(
            page_id,
            {
                "content_hash": content_hash,
                "save_intent": "commit",
                "supervision_present": True,
                "node_count": node_count,
                "edge_count": edge_count,
                "adopted_from_existing_correction": True,
            },
        )
        if not revision.is_duplicate:
            _snapshot_page_revision(registry, page_id, revision.revision_number)
        adopted.append(_revision_ref(page_id, revision.revision_number))
    return adopted


def _graph_counts_from_dir(graph_dir: Path, page_id: str) -> tuple[int, int]:
    node_path = graph_dir / f"{page_id}_inputs_normalized.txt"
    edge_path = graph_dir / f"{page_id}_edges.txt"
    node_count = 0
    edge_count = 0
    if node_path.is_file():
        node_count = sum(1 for line in node_path.read_text(encoding="utf-8").splitlines() if line.strip())
    if edge_path.is_file():
        edge_count = sum(1 for line in edge_path.read_text(encoding="utf-8").splitlines() if line.strip())
    return node_count, edge_count


def _graph_dir_content_hash(graph_dir: Path, page_id: str) -> str:
    digest = hashlib.sha256()
    for suffix in GRAPH_FORMAT_SUFFIXES:
        path = graph_dir / f"{page_id}{suffix}"
        digest.update(suffix.encode("utf-8"))
        digest.update(path.read_bytes() if path.is_file() else b"")
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


def _build_status_payload(registry: ManuscriptLayoutRegistry) -> dict:
    status = registry.get_status()
    consumed = set(registry.consumed_page_ids())
    corrected_pages = discover_corrected_page_ids(registry.manuscript_root)
    untrained_pages = [page_id for page_id in corrected_pages if page_id not in consumed]
    return {
        "code": status.get("code", "idle"),
        "label": status.get("label", "Not improving layout right now"),
        "updated_at": status.get("updated_at"),
        "details": status.get("details", {}),
        "active_checkpoint_id": registry.active_checkpoint_id(),
        "active_checkpoint_path": str(registry.active_checkpoint()),
        "needs_rebase": bool(registry.data.get("needs_rebase", False)),
        "active_learning_enabled": registry.is_active_learning_enabled(),
        "pending_jobs": registry.pending_layout_work(),
        "corrected_page_count": len(corrected_pages),
        "trained_page_count": len(consumed),
        "untrained_corrected_page_count": len(untrained_pages),
        "untrained_corrected_page_ids": untrained_pages,
        "backfill_available": bool(untrained_pages),
    }


def _reconcile_pending_jobs_with_orchestrator(
    registry: ManuscriptLayoutRegistry, orchestrator: JobOrchestrator | None = None
) -> None:
    orchestrator = _get_orchestrator(orchestrator)
    if orchestrator is None:
        return
    removed_any = False
    for pending_job in registry.pending_layout_work():
        job_id = str(pending_job.get("job_id"))
        status = orchestrator.get_job_status(job_id)
        state = str(status.get("state") or "")
        if not status or state in {"completed", "failed", "canceled"}:
            registry.remove_pending_job(job_id)
            removed_any = True
            continue
        if state and state != str(pending_job.get("state") or ""):
            registry.update_pending_job(job_id, state=state, started_at=status.get("started_at"))
    if removed_any and not registry.pending_layout_work():
        current_status_code = str((registry.get_status() or {}).get("code") or "")
        if current_status_code in {"queued", "running"}:
            if registry.data.get("needs_rebase"):
                registry.set_status("needs_rebase", "Ready to relearn layout from saved pages")
            else:
                registry.set_status("idle", "Not improving layout right now")


def summarize_manuscript_layout_active_learning(
    manuscript_root: str | Path,
    base_checkpoint_path: str | Path | None = None,
    orchestrator: JobOrchestrator | None = None,
) -> dict:
    registry = _load_registry_for_manuscript(
        manuscript_root, base_checkpoint_path=base_checkpoint_path
    )
    _reconcile_pending_jobs_with_orchestrator(registry, orchestrator=orchestrator)
    return _build_status_payload(registry)


def active_layout_checkpoint(
    manuscript_root: str | Path, base_checkpoint_path: str | Path | None = None
) -> tuple[str, str]:
    """Resolve the checkpoint page load should predict with.

    Never waits on an in-flight job: the newest *promoted* checkpoint is the
    contract, so navigation stays instant and training is never restarted by a
    page open.
    """
    registry = _load_registry_for_manuscript(
        manuscript_root, base_checkpoint_path=base_checkpoint_path
    )
    return str(registry.active_checkpoint()), registry.active_checkpoint_id()


# ---------------------------------------------------------------------------
# job construction
# ---------------------------------------------------------------------------


def _build_fine_tune_job(
    *,
    manuscript: str,
    manuscript_root: Path,
    page: str,
    revision_number: int,
    candidate_id: str,
    parent_checkpoint_id: str,
    parent_checkpoint_path: str,
    base_checkpoint_path: str,
    history_revision_refs: list[dict],
    recipe_config_path: str,
) -> QueuedJob:
    return QueuedJob(
        job_type=JobType.GNN_FINE_TUNE.value,
        manuscript=manuscript,
        manuscript_root=str(manuscript_root),
        priority=int(JobPriority.BACKGROUND_TRAINING),
        resource_name="gpu",
        isolated=True,
        payload={
            "job_type": JobType.GNN_FINE_TUNE.value,
            "manuscript": manuscript,
            "manuscript_root": str(manuscript_root),
            "page_id": str(page),
            "revision_number": int(revision_number),
            "training_revision_ref": _revision_ref(page, revision_number),
            "history_revision_refs": list(history_revision_refs),
            "candidate_id": candidate_id,
            "parent_checkpoint_id": parent_checkpoint_id,
            "parent_checkpoint_path": parent_checkpoint_path,
            "base_checkpoint_path": base_checkpoint_path,
            "recipe_config_path": recipe_config_path,
        },
        not_before=_deferred_start_timestamp(),
    )


def _build_pooled_job(
    *,
    job_type: str,
    manuscript: str,
    manuscript_root: Path,
    candidate_id: str,
    revision_refs: list[dict],
    base_checkpoint_path: str,
    recipe_config_path: str,
    defer_start: bool = False,
) -> QueuedJob:
    return QueuedJob(
        job_type=str(job_type),
        manuscript=manuscript,
        manuscript_root=str(manuscript_root),
        priority=int(JobPriority.BULK_PREPROCESS),
        resource_name="gpu",
        isolated=True,
        payload={
            "job_type": str(job_type),
            "manuscript": manuscript,
            "manuscript_root": str(manuscript_root),
            "candidate_id": candidate_id,
            "revision_refs": list(revision_refs),
            "base_checkpoint_path": base_checkpoint_path,
            "recipe_config_path": recipe_config_path,
        },
        not_before=_deferred_start_timestamp() if defer_start else None,
    )


# ---------------------------------------------------------------------------
# save hook
# ---------------------------------------------------------------------------


def handle_post_layout_save(
    manuscript: str,
    page: str,
    save_intent: str,
    save_scope: str,
    layout_active_learning_enabled: bool,
    graph_payload: dict | None,
    manuscript_root: str | Path | None = None,
    base_checkpoint_path: str | Path | None = None,
    textbox_labels=None,
    reading_direction_annotations=None,
    layout_modification_count: int | None = None,
    orchestrator: JobOrchestrator | None = None,
) -> dict:
    manuscript_root = Path(manuscript_root or Path("input_manuscripts") / manuscript)
    registry = _load_registry_for_manuscript(
        manuscript_root, base_checkpoint_path=base_checkpoint_path
    )
    orchestrator_instance = _get_orchestrator(orchestrator)
    registry.set_active_learning_enabled(bool(layout_active_learning_enabled))

    supervision_present = is_layout_supervision(save_intent, save_scope, graph_payload)
    graph = dict(graph_payload or {})
    node_count = len(graph.get("nodes") or [])
    edge_count = len(graph.get("edges") or [])

    if not supervision_present:
        # Draft autosaves and text-only saves never touch layout lineage.
        return {
            "supervision_present": False,
            "entered_active_learning": False,
            "queued_job_ids": [],
            "layout_active_learning": _build_status_payload(registry),
        }

    if not page_has_corrected_graph(manuscript_root, page):
        # The page saved, but its graph-format files are not on disk, so there
        # is nothing to snapshot or train on. Degrade instead of raising: a
        # failure here must never cost the user their layout save.
        missing = [
            suffix
            for suffix in GRAPH_FORMAT_SUFFIXES
            if not (live_graph_dir(manuscript_root) / f"{page}{suffix}").is_file()
        ]
        warning = (
            f"Layout learning skipped page {page}: corrected graph files are missing "
            f"({missing})."
        )
        print(f"[layout-save] {warning}")
        return {
            "supervision_present": False,
            "entered_active_learning": False,
            "queued_job_ids": [],
            "warning": warning,
            "layout_active_learning": _build_status_payload(registry),
        }

    content_hash = build_layout_revision_hash(
        graph_payload, textbox_labels, reading_direction_annotations
    )
    revision = registry.record_page_revision(
        page,
        {
            "content_hash": content_hash,
            "save_intent": save_intent,
            "supervision_present": True,
            "node_count": node_count,
            "edge_count": edge_count,
            "textline_count": len(set(graph.get("textline_labels") or [])) or None,
            "textbox_count": len(set(textbox_labels or [])) or None,
            "layout_modification_count": layout_modification_count,
        },
    )

    if not revision.is_duplicate:
        _snapshot_page_revision(registry, page, revision.revision_number)

    entered_active_learning = False
    queued_job_ids: list[str] = []
    recipe_config = _recipe_config_path()

    if bool(layout_active_learning_enabled) and not revision.is_duplicate:
        if registry.has_consumed_revision(page):
            # This page already shaped the active checkpoint; its lineage is
            # stale, so relearn from every saved page at once rather than
            # stacking a second step for the same page on top.
            registry.mark_rebase_needed("consumed_page_revision_changed", page)
            entered_active_learning = True
            already_queued = any(
                str(job.get("job_type")) == JobType.GNN_REBASE.value
                for job in registry.pending_layout_work()
            )
            if not already_queued and orchestrator_instance is not None:
                refs = _revision_refs_from_revisions(
                    registry.latest_supervised_commit_revisions()
                )
                job = _build_pooled_job(
                    job_type=JobType.GNN_REBASE.value,
                    manuscript=manuscript,
                    manuscript_root=manuscript_root,
                    candidate_id=f"gnn_rebase_r{revision.revision_number:04d}",
                    revision_refs=refs,
                    base_checkpoint_path=_base_checkpoint_path(base_checkpoint_path),
                    recipe_config_path=recipe_config,
                    defer_start=True,
                )
                queued_job_ids.append(orchestrator_instance.enqueue(job))
        elif orchestrator_instance is not None:
            history_refs = _revision_refs_from_revisions(
                registry.approved_supervised_revisions()
            )
            candidate_id = f"gnn_{_safe_slug(page)}_r{revision.revision_number:04d}"
            job = _build_fine_tune_job(
                manuscript=manuscript,
                manuscript_root=manuscript_root,
                page=page,
                revision_number=revision.revision_number,
                candidate_id=candidate_id,
                parent_checkpoint_id=registry.active_checkpoint_id(),
                parent_checkpoint_path=str(registry.active_checkpoint()),
                base_checkpoint_path=_base_checkpoint_path(base_checkpoint_path),
                history_revision_refs=history_refs,
                recipe_config_path=recipe_config,
            )
            queued_job_ids.append(orchestrator_instance.enqueue(job))
            entered_active_learning = True

    append_jsonl(
        registry.telemetry_root / "layout_learning_events.jsonl",
        {
            "event": "layout_save",
            "recorded_at": utc_now_iso(),
            "manuscript": manuscript,
            "page_id": str(page),
            "revision_number": revision.revision_number,
            "revision_is_duplicate": bool(revision.is_duplicate),
            "active_learning_enabled": bool(layout_active_learning_enabled),
            "entered_active_learning": bool(entered_active_learning),
            "node_count": node_count,
            "edge_count": edge_count,
            "queued_job_ids": list(queued_job_ids),
        },
    )

    return {
        "supervision_present": True,
        "revision": revision.to_dict(),
        "entered_active_learning": entered_active_learning,
        "queued_job_ids": queued_job_ids,
        "layout_active_learning": _build_status_payload(registry),
    }


def queue_layout_backfill(
    manuscript: str,
    manuscript_root: str | Path | None = None,
    base_checkpoint_path: str | Path | None = None,
    orchestrator: JobOrchestrator | None = None,
) -> dict:
    """Train one checkpoint over every corrected page this manuscript has.

    Pooled rather than a per-page ladder: with all pages already in hand there
    is nothing to approximate, so each page contributes all of its
    augmentations to every epoch and the 20% history replay is not used.
    """
    manuscript_root = Path(manuscript_root or Path("input_manuscripts") / manuscript)
    registry = _load_registry_for_manuscript(
        manuscript_root, base_checkpoint_path=base_checkpoint_path
    )
    orchestrator_instance = _get_orchestrator(orchestrator)

    register_existing_corrected_pages(registry)
    refs = _revision_refs_from_revisions(registry.latest_supervised_commit_revisions())
    if not refs:
        return {
            "queued": False,
            "reason": "no_corrected_pages",
            "layout_active_learning": _build_status_payload(registry),
        }
    if any(
        str(job.get("job_type")) in {JobType.GNN_BACKFILL.value, JobType.GNN_REBASE.value}
        for job in registry.pending_layout_work()
    ):
        return {
            "queued": False,
            "reason": "already_running",
            "layout_active_learning": _build_status_payload(registry),
        }
    if orchestrator_instance is None:
        return {
            "queued": False,
            "reason": "no_orchestrator",
            "layout_active_learning": _build_status_payload(registry),
        }

    job = _build_pooled_job(
        job_type=JobType.GNN_BACKFILL.value,
        manuscript=manuscript,
        manuscript_root=manuscript_root,
        candidate_id=f"gnn_backfill_{len(refs):03d}p",
        revision_refs=refs,
        base_checkpoint_path=_base_checkpoint_path(base_checkpoint_path),
        recipe_config_path=_recipe_config_path(),
    )
    job_id = orchestrator_instance.enqueue(job)
    return {
        "queued": True,
        "job_id": job_id,
        "page_count": len(refs),
        "page_ids": [ref["page_id"] for ref in refs],
        "layout_active_learning": _build_status_payload(registry),
    }


# ---------------------------------------------------------------------------
# orchestrator event mirroring
# ---------------------------------------------------------------------------


LAYOUT_JOB_TYPES = frozenset(
    {JobType.GNN_FINE_TUNE.value, JobType.GNN_REBASE.value, JobType.GNN_BACKFILL.value}
)


def _job_summary_label(job_type: str, payload: dict, state: str) -> str:
    page_id = payload.get("page_id")
    if job_type == JobType.GNN_FINE_TUNE.value:
        if state == "running":
            return f"Learning page layout from page {page_id}"
        if state == "queued":
            return f"Waiting to learn page layout from page {page_id}"
    if job_type == JobType.GNN_BACKFILL.value:
        count = len(payload.get("revision_refs") or [])
        if state == "running":
            return f"Learning layout from {count} corrected pages"
        if state == "queued":
            return f"Waiting to learn layout from {count} corrected pages"
    if job_type == JobType.GNN_REBASE.value:
        if state == "running":
            return "Relearning layout from saved pages"
        if state == "queued":
            return "Waiting to relearn layout from saved pages"
    return f"Layout learning status: {state.replace('_', ' ')}"


def _completed_job_status(job_status: dict, registry: ManuscriptLayoutRegistry) -> tuple[str, str]:
    job_type = str(job_status.get("job_type") or "")
    payload = dict(job_status.get("payload") or {})
    result = dict(job_status.get("result") or {})
    if job_type == JobType.GNN_FINE_TUNE.value:
        if bool(result.get("promoted")):
            return "idle", f"Layout model improved from page {payload.get('page_id')}"
        return "idle", f"Kept the current layout model after checking page {payload.get('page_id')}"
    if job_type in {JobType.GNN_BACKFILL.value, JobType.GNN_REBASE.value}:
        if bool(result.get("promoted")):
            return "idle", f"Layout model improved from {len(result.get('page_ids') or [])} corrected pages"
        if registry.data.get("needs_rebase"):
            return "needs_rebase", "Saved pages still need to be merged into the layout model"
        return "idle", "Finished checking saved pages"
    if registry.data.get("needs_rebase"):
        return "needs_rebase", "Saved pages still need to be merged into the layout model"
    return "idle", "Not improving layout right now"


def _record_job_event(registry: ManuscriptLayoutRegistry, event_name: str, job_status: dict) -> None:
    append_jsonl(
        registry.telemetry_root / "job_events.jsonl",
        {
            "event": event_name,
            "recorded_at": utc_now_iso(),
            "job_id": job_status.get("job_id"),
            "job_type": job_status.get("job_type"),
            "state": job_status.get("state"),
            "priority": job_status.get("priority"),
            "manuscript": job_status.get("manuscript"),
            "page_id": (job_status.get("payload") or {}).get("page_id"),
            "revision_number": (job_status.get("payload") or {}).get("revision_number"),
            "queue_wait_seconds": job_status.get("queue_wait_seconds"),
            "error": job_status.get("error"),
            "cancel_reason": job_status.get("cancel_reason"),
        },
    )


def handle_orchestrator_event(event_name: str, job_status: dict) -> None:
    job_type = str(job_status.get("job_type") or "")
    if job_type not in LAYOUT_JOB_TYPES:
        return
    manuscript_root = (job_status.get("payload") or {}).get("manuscript_root") or job_status.get(
        "manuscript_root"
    )
    if not manuscript_root:
        return
    registry = _load_registry_for_manuscript(manuscript_root)
    job_id = str(job_status["job_id"])
    job_payload = dict(job_status.get("payload") or {})

    if event_name == "queued":
        registry.enqueue_pending_job(
            {
                "job_id": job_id,
                "job_type": job_type,
                "page_id": job_payload.get("page_id"),
                "revision_number": job_payload.get("revision_number"),
                "candidate_id": job_payload.get("candidate_id"),
                "parent_checkpoint_id": job_payload.get("parent_checkpoint_id"),
                "page_count": len(job_payload.get("revision_refs") or []) or None,
                "priority": job_status["priority"],
                "state": job_status["state"],
                "created_at": job_status["created_at"],
            }
        )
        registry.set_status("queued", _job_summary_label(job_type, job_payload, "queued"), job_id=job_id)
    elif event_name == "started":
        registry.update_pending_job(job_id, state=job_status["state"], started_at=job_status.get("started_at"))
        registry.set_status("running", _job_summary_label(job_type, job_payload, "running"), job_id=job_id)
    elif event_name == "requeued":
        registry.update_pending_job(job_id, state="queued", requeued_at=utc_now_iso())
        registry.set_status("queued", _job_summary_label(job_type, job_payload, "queued"), job_id=job_id)
    elif event_name in {"completed", "failed", "canceled"}:
        registry.remove_pending_job(job_id)
        if event_name == "completed":
            status_code, label = _completed_job_status(job_status, registry)
            registry.set_status(status_code, label, job_id=job_id)
        elif event_name == "failed":
            registry.set_status(
                "failed",
                "Could not improve the layout model from saved corrections",
                job_id=job_id,
                error=job_status.get("error"),
            )
        else:
            registry.set_status("idle", "Layout learning was interrupted", job_id=job_id)
    _record_job_event(registry, event_name, job_status)


# ---------------------------------------------------------------------------
# training jobs (run inside the isolated child process)
# ---------------------------------------------------------------------------


def _compact_candidate_checkpoint(
    registry: ManuscriptLayoutRegistry, candidate_id: str, trained_checkpoint: Path
) -> str:
    """Copy the selected checkpoint to a stable path and drop the run copy."""
    stable_dir = registry.checkpoints_root / candidate_id
    stable_dir.mkdir(parents=True, exist_ok=True)
    stable_path = stable_dir / "model.pt"
    shutil.copy2(trained_checkpoint, stable_path)
    if _env_flag_enabled(_RUNTIME_COMPACT_ARTIFACTS_ENV, True):
        try:
            Path(trained_checkpoint).unlink()
        except OSError:
            pass
    return str(stable_path.resolve())


def _protected_checkpoint_ids(registry: ManuscriptLayoutRegistry) -> set[str]:
    protected = {
        "base",
        str(registry.data.get("active_checkpoint_id") or ""),
        str(registry.data.get("previous_active_checkpoint_id") or ""),
        str(registry.data.get("in_flight_candidate_id") or ""),
    }
    for pending_job in registry.pending_layout_work():
        for key in ("candidate_id", "parent_checkpoint_id"):
            value = pending_job.get(key)
            if value:
                protected.add(str(value))
    return {value for value in protected if value and value != "None"}


def _prune_obsolete_checkpoints(registry: ManuscriptLayoutRegistry) -> dict:
    """Drop checkpoint directories no longer reachable by the fallback ladder.

    Each layout checkpoint is roughly 51 MB and one is produced per corrected
    page, so without this a long annotation session accumulates gigabytes. The
    protected set mirrors the OCR runtime: base, active, previous active (the
    fallback), the in-flight candidate, and anything a pending job still needs.
    Pruned records stay in the registry with `status="pruned"` for lineage.
    """
    summary = {
        "enabled": _env_flag_enabled(_RUNTIME_PRUNE_CHECKPOINTS_ENV, True),
        "protected_checkpoint_ids": sorted(_protected_checkpoint_ids(registry)),
        "pruned_checkpoint_ids": [],
        "errors": [],
    }
    if not summary["enabled"]:
        return summary

    protected = set(summary["protected_checkpoint_ids"])
    checkpoints_root = registry.checkpoints_root.resolve()
    for checkpoint_id, record in list((registry.data.get("checkpoints") or {}).items()):
        checkpoint_id = str(checkpoint_id)
        if checkpoint_id in protected or str(record.get("kind") or "") == "base":
            continue
        if str(record.get("status") or "") == "pruned":
            continue
        checkpoint_dir = checkpoints_root / checkpoint_id
        try:
            # Never delete outside the manuscript's own checkpoints root.
            checkpoint_dir.resolve().relative_to(checkpoints_root)
        except ValueError:
            continue
        try:
            if checkpoint_dir.is_dir():
                shutil.rmtree(checkpoint_dir)
        except OSError as exc:
            summary["errors"].append({"checkpoint_id": checkpoint_id, "error": str(exc)})
            continue
        record["status_before_prune"] = record.get("status")
        record["status"] = "pruned"
        record["pruned_at"] = utc_now_iso()
        summary["pruned_checkpoint_ids"].append(checkpoint_id)

    registry.data["last_checkpoint_prune_summary"] = summary
    registry.save()
    return summary


def _finalize_promoted_candidate(
    *,
    registry: ManuscriptLayoutRegistry,
    candidate_id: str,
    promotion_summary: dict,
    consumed_revision_refs: list[dict],
    clear_rebase: bool = False,
) -> dict:
    registry.promote_candidate(candidate_id, promotion_summary)
    for ref in consumed_revision_refs:
        try:
            registry.mark_revision_consumed(ref["page_id"], ref["revision_number"], candidate_id)
        except KeyError:
            continue
    if clear_rebase:
        registry.clear_rebase()
    return _prune_obsolete_checkpoints(registry)


def _graph_dir_for_ref(registry: ManuscriptLayoutRegistry, ref: dict) -> Path:
    graph_dir = registry.revision_graph_dir(ref["page_id"], ref["revision_number"])
    if not graph_dir.is_dir():
        raise FileNotFoundError(
            f"Layout revision snapshot is missing: {graph_dir}. "
            "Re-save the page in Layout Mode to recreate it."
        )
    return graph_dir


def run_gnn_finetune_job(job_payload: dict) -> dict:
    """One incremental step: continue the active checkpoint on the new page."""
    from gnn_training.gnn_finetuning import (
        _process_graph_or_fail,
        augment_corrected_graph_page,
        fine_tune_gnn_checkpoint,
        load_gnn_finetuning_recipe,
        select_history_replay_page_ids,
    )

    registry = _load_registry_for_manuscript(
        job_payload["manuscript_root"], base_checkpoint_path=job_payload.get("base_checkpoint_path")
    )
    recipe = load_gnn_finetuning_recipe(job_payload["recipe_config_path"])
    training_ref = dict(job_payload["training_revision_ref"])
    history_refs = list(job_payload.get("history_revision_refs") or [])
    candidate_id = str(job_payload["candidate_id"])
    candidate_root = registry.training_root / candidate_id

    def augment(ref: dict) -> tuple[Path, tuple[str, ...]]:
        page_id = str(ref["page_id"])
        revision_number = int(ref["revision_number"])
        output_dir = _augmentation_dir(registry, page_id, revision_number)
        augmented_ids = augment_corrected_graph_page(
            page_id=page_id,
            page_index=_augmentation_page_index(registry, page_id),
            source_dir=_graph_dir_for_ref(registry, ref),
            output_dir=output_dir,
            recipe=recipe,
        )
        return output_dir, augmented_ids

    current_dir, current_ids = augment(training_ref)
    training_sample_ids = list(current_ids)
    training_data = [
        _process_graph_or_fail(augmented_id, current_dir, recipe.preprocessing_config)
        for augmented_id in current_ids
    ]

    base_seed = int(recipe.augmentation_config.general.base_seed)
    step_index = len(history_refs) + 1
    for history_index, ref in enumerate(history_refs):
        history_dir, history_ids = augment(ref)
        replay_ids = select_history_replay_page_ids(
            history_ids,
            replay_ratio=recipe.history_replay_ratio,
            seed=base_seed + step_index * 10_000 + history_index,
        )
        training_sample_ids.extend(replay_ids)
        training_data.extend(
            _process_graph_or_fail(augmented_id, history_dir, recipe.preprocessing_config)
            for augmented_id in replay_ids
        )

    validation_refs = history_refs + [training_ref]
    validation_data = [
        _process_graph_or_fail(
            str(ref["page_id"]), _graph_dir_for_ref(registry, ref), recipe.preprocessing_config
        )
        for ref in validation_refs
    ]

    def run_step():
        return fine_tune_gnn_checkpoint(
            base_checkpoint=Path(job_payload["parent_checkpoint_path"]),
            training_data=training_data,
            validation_data=validation_data,
            output_dir=candidate_root / "step",
            recipe=recipe,
            step_index=step_index,
            current_page_id=str(training_ref["page_id"]),
            history_page_ids=tuple(str(ref["page_id"]) for ref in history_refs),
            training_sample_ids=tuple(training_sample_ids),
            validation_page_ids=tuple(str(ref["page_id"]) for ref in validation_refs),
        )

    trace_enabled = should_capture_cuda_trace("gnn_fine_tune", registry.profiling_root)
    step_result, summary = summarize_gpu_job(
        "gnn_fine_tune",
        {
            "job_type": JobType.GNN_FINE_TUNE.value,
            "candidate_id": candidate_id,
            "page_id": training_ref["page_id"],
            "revision_number": training_ref["revision_number"],
            "training_sample_count": len(training_sample_ids),
            "queue_wait_seconds": float(job_payload.get("queue_wait_seconds") or 0.0),
        },
        lambda: maybe_write_cuda_trace(
            "gnn_fine_tune", registry.profiling_root, trace_enabled, run_step
        ),
    )
    write_profile_summary(registry.profiling_root, "gnn_fine_tune", summary)

    checkpoint_path = _compact_candidate_checkpoint(
        registry, candidate_id, Path(step_result.output_checkpoint)
    )
    registry.mark_candidate(
        {
            "candidate_id": candidate_id,
            "checkpoint_path": checkpoint_path,
            "parent_checkpoint_id": job_payload.get("parent_checkpoint_id"),
            "page_id": training_ref["page_id"],
            "revision_number": training_ref["revision_number"],
            "metadata_path": str(step_result.metadata_path),
            "run_dir": str(candidate_root),
            "training_schedule": "incremental_page_plus_history_replay",
            "selected_epoch": step_result.selected_epoch,
            "selected_metric": step_result.selected_metric,
        }
    )
    registry.set_checkpoint_lineage(candidate_id, history_refs + [training_ref])
    promotion_summary = {
        "passed": True,
        "reason": "direct_promote_after_training",
        "selected_epoch": step_result.selected_epoch,
        "selected_metric": step_result.selected_metric,
    }
    _finalize_promoted_candidate(
        registry=registry,
        candidate_id=candidate_id,
        promotion_summary=promotion_summary,
        consumed_revision_refs=[training_ref],
    )
    _prune_stale_augmentations(
        registry,
        str(training_ref["page_id"]),
        {int(training_ref["revision_number"])},
    )
    return {
        "candidate_id": candidate_id,
        "checkpoint_path": checkpoint_path,
        "promoted": True,
        "page_ids": [str(training_ref["page_id"])],
        "promotion_summary": promotion_summary,
    }


def run_gnn_pooled_job(job_payload: dict) -> dict:
    """Backfill or rebase: one checkpoint from base over every saved page."""
    from gnn_training.gnn_finetuning import run_pooled_gnn_finetuning

    registry = _load_registry_for_manuscript(
        job_payload["manuscript_root"], base_checkpoint_path=job_payload.get("base_checkpoint_path")
    )
    job_type = str(job_payload.get("job_type") or JobType.GNN_BACKFILL.value)
    revision_refs = list(job_payload.get("revision_refs") or [])
    candidate_id = str(job_payload["candidate_id"])
    if not revision_refs:
        registry.clear_rebase()
        registry.set_status("idle", "Not improving layout right now")
        return {"promoted": False, "reason": "no_corrected_pages", "page_ids": []}

    page_graph_dirs = {
        str(ref["page_id"]): _graph_dir_for_ref(registry, ref) for ref in revision_refs
    }
    candidate_root = registry.training_root / candidate_id

    def run_step():
        return run_pooled_gnn_finetuning(
            page_graph_dirs=page_graph_dirs,
            base_checkpoint=Path(job_payload["base_checkpoint_path"]),
            output_root=candidate_root,
            config_path=job_payload["recipe_config_path"],
            kind=f"app_{job_type}",
        )

    trace_enabled = should_capture_cuda_trace("gnn_pooled", registry.profiling_root)
    pooled_result, summary = summarize_gpu_job(
        job_type,
        {
            "job_type": job_type,
            "candidate_id": candidate_id,
            "page_count": len(revision_refs),
            "queue_wait_seconds": float(job_payload.get("queue_wait_seconds") or 0.0),
        },
        lambda: maybe_write_cuda_trace(
            "gnn_pooled", registry.profiling_root, trace_enabled, run_step
        ),
    )
    write_profile_summary(registry.profiling_root, job_type, summary)

    checkpoint_path = _compact_candidate_checkpoint(
        registry, candidate_id, Path(pooled_result.output_checkpoint)
    )
    registry.mark_candidate(
        {
            "candidate_id": candidate_id,
            "checkpoint_path": checkpoint_path,
            "parent_checkpoint_id": "base",
            "page_ids": list(pooled_result.page_ids),
            "metadata_path": str(pooled_result.metadata_path),
            "summary_path": str(pooled_result.summary_path),
            "run_dir": str(candidate_root),
            "training_schedule": "pooled_all_pages",
            "selected_epoch": pooled_result.selected_epoch,
            "selected_metric": pooled_result.selected_metric,
        }
    )
    registry.set_checkpoint_lineage(candidate_id, revision_refs)
    promotion_summary = {
        "passed": True,
        "reason": f"direct_promote_after_{job_type}",
        "training_schedule": "pooled_all_pages",
        "page_count": len(revision_refs),
        "selected_epoch": pooled_result.selected_epoch,
        "selected_metric": pooled_result.selected_metric,
    }
    _finalize_promoted_candidate(
        registry=registry,
        candidate_id=candidate_id,
        promotion_summary=promotion_summary,
        consumed_revision_refs=revision_refs,
        clear_rebase=True,
    )
    return {
        "candidate_id": candidate_id,
        "checkpoint_path": checkpoint_path,
        "promoted": True,
        "page_ids": list(pooled_result.page_ids),
        "training_sample_count": pooled_result.training_sample_count,
        "promotion_summary": promotion_summary,
    }


def dispatch_isolated_job(job_type: str, payload: dict) -> dict:
    job_type = str(job_type)
    if job_type == JobType.GNN_FINE_TUNE.value:
        return run_gnn_finetune_job(payload)
    if job_type in {JobType.GNN_REBASE.value, JobType.GNN_BACKFILL.value}:
        return run_gnn_pooled_job(payload)
    raise KeyError(f"Unsupported isolated layout active-learning job type: {job_type}")
