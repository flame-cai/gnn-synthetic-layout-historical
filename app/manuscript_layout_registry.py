"""Manuscript-local registry for layout (GNN) active learning.

This mirrors `manuscript_ocr_registry.ManuscriptOcrRegistry` deliberately: the
two lineages have the same shape (page revisions -> candidate checkpoint ->
promotion -> rebase when a consumed page changes), so keeping the vocabulary
identical keeps the runtime, the routes, and the UI copy comprehensible.

The state lives beside, not inside, the OCR runtime state:

    active_learning/layout/registry.json
    active_learning/layout/checkpoints/<checkpoint_id>/model.pt
    active_learning/layout/revisions/<page>/rev_NNNN/gnn-format/...
    active_learning/layout/training/<candidate_id>/

The two differences from the OCR registry that matter:

* supervision is a Layout Mode `layout` commit save, not a Read Mode
  `text_only` one, so a revision counts nodes and edges rather than text lines;
* a revision snapshot copies the six corrected `gnn-format` files, so a job
  trains on the graph exactly as it was saved even if the user keeps editing.
"""
from __future__ import annotations

import copy
import json
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


_ATOMIC_WRITE_RETRIES = 10
_ATOMIC_WRITE_RETRY_DELAY_SECONDS = 0.05

GRAPH_FORMAT_SUFFIXES = (
    "_dims.txt",
    "_inputs_normalized.txt",
    "_inputs_unnormalized.txt",
    "_edges.txt",
    "_labels_textline.txt",
    "_labels_textbox.txt",
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _page_sort_key(page_id: str) -> tuple:
    parts = []
    chunk = ""
    is_digit = None
    for char in str(page_id):
        char_is_digit = char.isdigit()
        if is_digit is None or char_is_digit == is_digit:
            chunk += char
        else:
            # Tag each chunk so mixed numeric/alphabetic page ids remain comparable.
            parts.extend((0, int(chunk)) if is_digit else (1, chunk))
            chunk = char
        is_digit = char_is_digit
    if chunk:
        parts.extend((0, int(chunk)) if is_digit else (1, chunk))
    return tuple(parts)


@dataclass(frozen=True)
class LayoutPageRevision:
    page_id: str
    revision_number: int
    content_hash: str
    save_intent: str
    supervision_present: bool
    node_count: int
    edge_count: int
    created_at: str
    consumed_into_checkpoint_id: str | None = None
    is_duplicate: bool = False

    def to_dict(self) -> dict:
        return {
            "page_id": self.page_id,
            "revision_number": int(self.revision_number),
            "content_hash": self.content_hash,
            "save_intent": self.save_intent,
            "supervision_present": bool(self.supervision_present),
            "node_count": int(self.node_count),
            "edge_count": int(self.edge_count),
            "created_at": self.created_at,
            "consumed_into_checkpoint_id": self.consumed_into_checkpoint_id,
            "is_duplicate": bool(self.is_duplicate),
        }


class ManuscriptLayoutRegistry:
    def __init__(self, manuscript_root: str | Path, data: dict):
        self.manuscript_root = Path(manuscript_root)
        self.runtime_root = self.manuscript_root / "active_learning" / "layout"
        self.registry_path = self.runtime_root / "registry.json"
        self.checkpoints_root = self.runtime_root / "checkpoints"
        self.revisions_root = self.runtime_root / "revisions"
        self.training_root = self.runtime_root / "training"
        self.telemetry_root = self.manuscript_root / "active_learning" / "telemetry"
        self.profiling_root = self.runtime_root / "profiling"
        self.data = data
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        for directory in (
            self.runtime_root,
            self.checkpoints_root,
            self.revisions_root,
            self.training_root,
            self.telemetry_root,
            self.profiling_root,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def _replace_atomic_with_retry(self, tmp_path: Path) -> None:
        last_error = None
        for attempt in range(_ATOMIC_WRITE_RETRIES):
            try:
                tmp_path.replace(self.registry_path)
                return
            except PermissionError as exc:
                last_error = exc
                if attempt == (_ATOMIC_WRITE_RETRIES - 1):
                    break
                time.sleep(_ATOMIC_WRITE_RETRY_DELAY_SECONDS)
        if last_error is not None:
            raise last_error

    def _write_atomic(self) -> None:
        self._ensure_directories()
        tmp_path = self.runtime_root / (
            f"{self.registry_path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        try:
            tmp_path.write_text(
                json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            self._replace_atomic_with_retry(tmp_path)
        finally:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except FileNotFoundError:
                pass

    def save(self) -> None:
        self._write_atomic()

    def snapshot(self) -> dict:
        return copy.deepcopy(self.data)

    # ---- enablement and status -------------------------------------------

    def set_active_learning_enabled(self, enabled: bool) -> None:
        self.data["active_learning_enabled"] = bool(enabled)
        self.save()

    def is_active_learning_enabled(self) -> bool:
        return bool(self.data.get("active_learning_enabled", False))

    def set_status(self, code: str, label: str | None = None, **details) -> None:
        label = label or f"Layout learning status: {code.replace('_', ' ')}"
        self.data["status"] = {
            "code": str(code),
            "label": str(label),
            "updated_at": _utc_now_iso(),
            "details": details,
        }
        self.save()

    def get_status(self) -> dict:
        return dict(
            self.data.get("status")
            or {
                "code": "idle",
                "label": "Not improving layout right now",
                "updated_at": _utc_now_iso(),
            }
        )

    # ---- checkpoints ------------------------------------------------------

    def _checkpoint_record(self, checkpoint_id: str | None) -> dict | None:
        if not checkpoint_id:
            return None
        return self.data.setdefault("checkpoints", {}).get(str(checkpoint_id))

    def active_checkpoint_id(self) -> str:
        return str(
            self.data.get("active_checkpoint_id")
            or self.data["base_checkpoint"]["checkpoint_id"]
        )

    def active_checkpoint(self) -> Path:
        """Resolve the newest usable checkpoint, degrading rather than failing.

        Same fallback ladder as the OCR registry: active, then the previous
        active, then the pretrained base. Page load calls this on every
        navigation, so a missing or half-written checkpoint file must never
        become a dead end.
        """
        candidates = [
            self.data.get("active_checkpoint_id"),
            self.data.get("previous_active_checkpoint_id"),
            self.data["base_checkpoint"]["checkpoint_id"],
        ]
        for checkpoint_id in candidates:
            if not checkpoint_id:
                continue
            record = self._checkpoint_record(checkpoint_id)
            if not record:
                continue
            checkpoint_path = Path(record["path"])
            if checkpoint_path.exists():
                if checkpoint_id != self.data.get("active_checkpoint_id"):
                    self.data["last_checkpoint_fallback"] = {
                        "from_checkpoint_id": self.data.get("active_checkpoint_id"),
                        "to_checkpoint_id": checkpoint_id,
                        "recorded_at": _utc_now_iso(),
                    }
                    self.data["active_checkpoint_id"] = checkpoint_id
                    self.save()
                return checkpoint_path
        raise FileNotFoundError(
            "No usable layout GNN checkpoint is available for this manuscript."
        )

    def mark_candidate(self, candidate_payload: dict) -> None:
        payload = dict(candidate_payload or {})
        checkpoint_id = str(payload["candidate_id"])
        record = {
            "checkpoint_id": checkpoint_id,
            "path": str(Path(payload["checkpoint_path"]).resolve()),
            "status": str(payload.get("status", "candidate")),
            "parent_checkpoint_id": payload.get("parent_checkpoint_id"),
            "page_id": payload.get("page_id"),
            "page_ids": payload.get("page_ids"),
            "revision_number": payload.get("revision_number"),
            "created_at": payload.get("created_at", _utc_now_iso()),
            "metadata_path": payload.get("metadata_path"),
            "summary_path": payload.get("summary_path"),
            "run_dir": payload.get("run_dir"),
            "training_schedule": payload.get("training_schedule"),
            "selected_epoch": payload.get("selected_epoch"),
            "selected_metric": payload.get("selected_metric"),
        }
        self.data.setdefault("checkpoints", {})[checkpoint_id] = record
        self.data["in_flight_candidate_id"] = checkpoint_id
        self.save()

    def set_checkpoint_lineage(self, checkpoint_id: str, revision_refs: list[dict]) -> None:
        checkpoint = self._checkpoint_record(checkpoint_id)
        if checkpoint is None:
            raise KeyError(f"Unknown layout checkpoint id: {checkpoint_id}")
        checkpoint["lineage_revision_refs"] = copy.deepcopy(list(revision_refs or []))
        self.save()

    def promote_candidate(self, candidate_id: str, promotion_summary: dict) -> None:
        candidate_id = str(candidate_id)
        candidate = self._checkpoint_record(candidate_id)
        if candidate is None:
            raise KeyError(f"Unknown layout candidate checkpoint: {candidate_id}")
        previous_active = self.data.get("active_checkpoint_id")
        candidate["status"] = "active"
        candidate["promoted_at"] = _utc_now_iso()
        candidate["promotion_summary"] = promotion_summary
        self.data["previous_active_checkpoint_id"] = previous_active
        self.data["active_checkpoint_id"] = candidate_id
        self.data["in_flight_candidate_id"] = None
        self.data["last_successful_promotion_summary"] = promotion_summary
        self.save()

    def reject_candidate(self, candidate_id: str, promotion_summary: dict) -> None:
        candidate = self._checkpoint_record(candidate_id)
        if candidate is None:
            raise KeyError(f"Unknown layout candidate checkpoint: {candidate_id}")
        candidate["status"] = "rejected"
        candidate["rejected_at"] = _utc_now_iso()
        candidate["promotion_summary"] = promotion_summary
        if self.data.get("in_flight_candidate_id") == candidate_id:
            self.data["in_flight_candidate_id"] = None
        self.save()

    # ---- page revisions ---------------------------------------------------

    def _latest_commit_revision(self, page_id: str) -> dict | None:
        revisions = self.data.setdefault("page_revisions", {}).get(str(page_id), [])
        for revision in reversed(revisions):
            if str(revision.get("save_intent", "commit")) == "commit":
                return revision
        return None

    def _duplicate_source_revision(
        self, page_id: str, content_hash: str, save_intent: str
    ) -> dict | None:
        revisions = self.data.setdefault("page_revisions", {}).get(str(page_id), [])
        latest = revisions[-1] if revisions else None
        if latest is None or str(latest.get("content_hash")) != str(content_hash):
            return None
        latest_save_intent = str(latest.get("save_intent", "commit"))
        if str(save_intent) != "commit":
            return latest
        if latest_save_intent == "commit":
            return latest
        latest_commit = self._latest_commit_revision(page_id)
        if latest_commit is not None and str(latest_commit.get("content_hash")) == str(content_hash):
            return latest_commit
        return None

    def record_page_revision(self, page_id: str, revision_payload: dict) -> LayoutPageRevision:
        page_id = str(page_id)
        payload = dict(revision_payload or {})
        revisions = self.data.setdefault("page_revisions", {}).setdefault(page_id, [])
        latest = revisions[-1] if revisions else None
        content_hash = str(payload["content_hash"])
        save_intent = str(payload.get("save_intent", "commit"))

        duplicate_source = self._duplicate_source_revision(page_id, content_hash, save_intent)
        if duplicate_source is not None:
            # Re-saving an unchanged graph must not queue another training job.
            return LayoutPageRevision(
                page_id=page_id,
                revision_number=int(duplicate_source["revision_number"]),
                content_hash=content_hash,
                save_intent=str(duplicate_source["save_intent"]),
                supervision_present=bool(duplicate_source["supervision_present"]),
                node_count=int(duplicate_source.get("node_count", 0)),
                edge_count=int(duplicate_source.get("edge_count", 0)),
                created_at=str(duplicate_source["created_at"]),
                consumed_into_checkpoint_id=duplicate_source.get("consumed_into_checkpoint_id"),
                is_duplicate=True,
            )

        revision = LayoutPageRevision(
            page_id=page_id,
            revision_number=(int(latest["revision_number"]) + 1) if latest else 1,
            content_hash=content_hash,
            save_intent=save_intent,
            supervision_present=bool(payload.get("supervision_present", False)),
            node_count=int(payload.get("node_count", 0)),
            edge_count=int(payload.get("edge_count", 0)),
            created_at=str(payload.get("created_at") or _utc_now_iso()),
            consumed_into_checkpoint_id=None,
            is_duplicate=False,
        )
        revision_record = revision.to_dict()
        revision_record.update(
            {
                "page_sort_key": list(_page_sort_key(page_id)),
                "textline_count": payload.get("textline_count"),
                "textbox_count": payload.get("textbox_count"),
                "layout_modification_count": payload.get("layout_modification_count"),
            }
        )
        revisions.append(revision_record)
        self.save()
        return revision

    def find_revision(self, page_id: str, revision_number: int) -> dict | None:
        for revision in self.data.setdefault("page_revisions", {}).get(str(page_id), []):
            if int(revision["revision_number"]) == int(revision_number):
                return revision
        return None

    def latest_revision(self, page_id: str) -> dict | None:
        revisions = self.data.setdefault("page_revisions", {}).get(str(page_id), [])
        return copy.deepcopy(revisions[-1]) if revisions else None

    def latest_supervised_commit_revision(self, page_id: str) -> dict | None:
        revisions = self.data.setdefault("page_revisions", {}).get(str(page_id), [])
        for revision in reversed(revisions):
            if revision.get("save_intent") == "commit" and bool(revision.get("supervision_present")):
                return revision
        return None

    def latest_supervised_commit_revisions(self) -> list[dict]:
        revisions = []
        for page_id in sorted(self.data.setdefault("page_revisions", {}), key=_page_sort_key):
            revision = self.latest_supervised_commit_revision(page_id)
            if revision is not None:
                revisions.append(copy.deepcopy(revision))
        return revisions

    def approved_supervised_revisions(self) -> list[dict]:
        """Supervised revisions already folded into a promoted checkpoint."""
        approved = []
        for page_id in sorted(self.data.setdefault("page_revisions", {}), key=_page_sort_key):
            for revision in self.data["page_revisions"][page_id]:
                if revision.get("save_intent") != "commit":
                    continue
                if not revision.get("supervision_present"):
                    continue
                if not revision.get("consumed_into_checkpoint_id"):
                    continue
                approved.append(copy.deepcopy(revision))
        approved.sort(
            key=lambda revision: (_page_sort_key(revision["page_id"]), int(revision["revision_number"]))
        )
        return approved

    def has_consumed_revision(self, page_id: str) -> bool:
        for revision in self.data.setdefault("page_revisions", {}).get(str(page_id), []):
            if revision.get("consumed_into_checkpoint_id"):
                return True
        return False

    def consumed_page_ids(self) -> list[str]:
        pages = [
            page_id
            for page_id in self.data.setdefault("page_revisions", {})
            if self.has_consumed_revision(page_id)
        ]
        return sorted(pages, key=_page_sort_key)

    def mark_revision_consumed(self, page_id: str, revision_number: int, checkpoint_id: str) -> None:
        revision = self.find_revision(page_id, revision_number)
        if revision is None:
            raise KeyError(f"Unknown layout page revision: {page_id}#{revision_number}")
        revision["consumed_into_checkpoint_id"] = str(checkpoint_id)
        revision["consumed_at"] = _utc_now_iso()
        self.data["most_recent_promoted_page"] = {
            "page_id": str(page_id),
            "revision_number": int(revision_number),
            "checkpoint_id": str(checkpoint_id),
        }
        self.save()

    # ---- rebase -----------------------------------------------------------

    def mark_rebase_needed(self, reason: str, changed_page_id: str) -> None:
        self.data["needs_rebase"] = True
        self.data["rebase"] = {
            "needed": True,
            "reason": str(reason),
            "changed_page_id": str(changed_page_id),
            "marked_at": _utc_now_iso(),
        }
        self.save()

    def clear_rebase(self) -> None:
        self.data["needs_rebase"] = False
        self.data["rebase"] = {"needed": False, "cleared_at": _utc_now_iso()}
        self.save()

    # ---- pending jobs -----------------------------------------------------

    def pending_layout_work(self) -> list[dict]:
        return copy.deepcopy(self.data.setdefault("pending_jobs", []))

    def enqueue_pending_job(self, job_payload: dict) -> None:
        payload = copy.deepcopy(job_payload)
        job_id = str(payload["job_id"])
        pending_jobs = self.data.setdefault("pending_jobs", [])
        if any(str(existing.get("job_id")) == job_id for existing in pending_jobs):
            return
        pending_jobs.append(payload)
        self.save()

    def update_pending_job(self, job_id: str, **changes) -> None:
        for job in self.data.setdefault("pending_jobs", []):
            if str(job.get("job_id")) == str(job_id):
                job.update(changes)
                self.save()
                return

    def remove_pending_job(self, job_id: str) -> None:
        pending_jobs = self.data.setdefault("pending_jobs", [])
        self.data["pending_jobs"] = [
            job for job in pending_jobs if str(job.get("job_id")) != str(job_id)
        ]
        self.save()

    # ---- snapshots --------------------------------------------------------

    def revision_snapshot_root(self, page_id: str, revision_number: int) -> Path:
        return self.revisions_root / str(page_id) / f"rev_{int(revision_number):04d}"

    def revision_graph_dir(self, page_id: str, revision_number: int) -> Path:
        return self.revision_snapshot_root(page_id, revision_number) / "gnn-format"


def _initial_registry_payload(manuscript_root: Path, base_checkpoint_path: str | Path) -> dict:
    base_checkpoint_path = Path(base_checkpoint_path).resolve()
    base_record = {
        "checkpoint_id": "base",
        "path": str(base_checkpoint_path),
        "created_at": _utc_now_iso(),
        "status": "active",
        "kind": "base",
    }
    return {
        "schema_version": 1,
        "manuscript_root": str(manuscript_root.resolve()),
        "active_learning_enabled": False,
        "needs_rebase": False,
        "rebase": {"needed": False},
        "status": {
            "code": "idle",
            "label": "Not improving layout right now",
            "updated_at": _utc_now_iso(),
            "details": {},
        },
        "base_checkpoint": dict(base_record),
        "active_checkpoint_id": "base",
        "previous_active_checkpoint_id": None,
        "in_flight_candidate_id": None,
        "checkpoints": {"base": dict(base_record)},
        "page_revisions": {},
        "pending_jobs": [],
        "last_successful_promotion_summary": None,
        "last_checkpoint_fallback": None,
        "most_recent_promoted_page": None,
    }


def load_registry(
    manuscript_root: str | Path, base_checkpoint_path: str | Path | None = None
) -> ManuscriptLayoutRegistry:
    manuscript_root = Path(manuscript_root)
    registry_path = manuscript_root / "active_learning" / "layout" / "registry.json"
    if registry_path.exists():
        data = json.loads(registry_path.read_text(encoding="utf-8"))
        if base_checkpoint_path is not None:
            # Keep the recorded base pointing at the checkpoint this server run
            # is actually configured with, so moving the repo does not strand
            # the fallback ladder at a path that no longer exists.
            resolved_base = str(Path(base_checkpoint_path).resolve())
            base_record = data.setdefault("base_checkpoint", {})
            if str(base_record.get("path")) != resolved_base:
                base_record["path"] = resolved_base
                data.setdefault("checkpoints", {}).setdefault("base", {}).update(
                    {"checkpoint_id": "base", "path": resolved_base, "kind": "base"}
                )
        registry = ManuscriptLayoutRegistry(manuscript_root, data)
        registry.active_checkpoint()
        registry.save()
        return registry

    if base_checkpoint_path is None:
        raise FileNotFoundError(
            "Layout registry does not exist yet and no base checkpoint path was provided."
        )
    registry = ManuscriptLayoutRegistry(
        manuscript_root, _initial_registry_payload(manuscript_root, base_checkpoint_path)
    )
    registry.save()
    return registry
