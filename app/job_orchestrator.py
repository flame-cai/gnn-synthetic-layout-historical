from __future__ import annotations

import copy
import heapq
import multiprocessing
import queue
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum, IntEnum
from pathlib import Path
from typing import Callable

from device_leases import DeviceLeaseManager


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobType(str, Enum):
    OCR_INFER = "ocr_infer"
    OCR_FINE_TUNE = "ocr_fine_tune"
    OCR_REBASE = "ocr_rebase"
    CRAFT_BATCH_INFER = "craft_batch_infer"
    GNN_PAGE_INFER = "gnn_page_infer"


class JobPriority(IntEnum):
    INTERACTIVE = 0
    SAVE_FOLLOWUP = 1
    BACKGROUND_TRAINING = 2
    BULK_PREPROCESS = 3


class JobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"
    REQUEUED = "requeued"


@dataclass
class QueuedJob:
    job_type: JobType | str
    manuscript: str
    manuscript_root: str
    payload: dict
    priority: int = int(JobPriority.BACKGROUND_TRAINING)
    resource_name: str = "gpu"
    isolated: bool = False
    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=_utc_now_iso)


def _isolated_job_entry(job_type: str, payload: dict, result_queue) -> None:
    try:
        from ocr_active_learning_runtime import dispatch_isolated_job

        result_queue.put({"ok": True, "result": dispatch_isolated_job(job_type, payload)})
    except Exception as exc:  # pragma: no cover - process failures are integration concerns
        result_queue.put(
            {
                "ok": False,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )


class JobOrchestrator:
    def __init__(self, lease_manager: DeviceLeaseManager | None = None):
        self._lease_manager = lease_manager or DeviceLeaseManager()
        self._handlers: dict[str, Callable[[dict], dict | None]] = {}
        self._listener: Callable[[str, dict], None] | None = None
        self._jobs: dict[str, dict] = {}
        self._heap: list[tuple[int, int, str]] = []
        self._sequence = 0
        self._stop_event = threading.Event()
        self._condition = threading.Condition()
        self._worker_thread: threading.Thread | None = None
        self._running_job_id: str | None = None

    def register_handler(self, job_type: JobType | str, handler: Callable[[dict], dict | None]) -> None:
        self._handlers[str(job_type)] = handler

    def set_state_listener(self, listener: Callable[[str, dict], None]) -> None:
        self._listener = listener

    def _notify(self, event_name: str, job_id: str) -> None:
        if self._listener is None:
            return
        try:
            self._listener(event_name, self.get_job_status(job_id))
        except Exception:  # pragma: no cover - defensive isolation for background listeners
            traceback.print_exc()

    def _store_job(self, job: QueuedJob) -> dict:
        payload = asdict(job)
        payload["job_type"] = str(job.job_type)
        payload["state"] = JobState.QUEUED.value
        payload["updated_at"] = _utc_now_iso()
        payload["attempt_count"] = 0
        payload["cancel_requested"] = False
        payload["requeue_on_cancel"] = False
        payload["cancel_reason"] = None
        payload["started_at"] = None
        payload["finished_at"] = None
        payload["result"] = None
        payload["error"] = None
        self._jobs[job.job_id] = payload
        return payload

    def enqueue(self, job: QueuedJob | dict) -> str:
        if isinstance(job, dict):
            job = QueuedJob(**job)
        with self._condition:
            record = self._store_job(job)
            heapq.heappush(self._heap, (int(job.priority), self._sequence, job.job_id))
            self._sequence += 1
            self._condition.notify_all()
        self._notify("queued", job.job_id)
        return job.job_id

    def cancel(self, job_id: str, reason: str, requeue: bool = False) -> None:
        with self._condition:
            record = self._jobs.get(str(job_id))
            if record is None:
                return
            record["cancel_requested"] = True
            record["requeue_on_cancel"] = bool(requeue)
            record["cancel_reason"] = str(reason)
            if record["state"] == JobState.QUEUED.value and not requeue:
                record["state"] = JobState.CANCELED.value
                record["finished_at"] = _utc_now_iso()
                record["updated_at"] = _utc_now_iso()
        if not requeue:
            self._notify("canceled", str(job_id))

    def preempt_for_interactive(self, resource_name: str = "gpu", requester_priority: int = int(JobPriority.INTERACTIVE)) -> str | None:
        with self._condition:
            running_job_id = self._running_job_id
            if running_job_id is None:
                return None
            running = self._jobs.get(running_job_id)
            if running is None:
                return None
            if running.get("resource_name") != str(resource_name):
                return None
            if int(running.get("priority", JobPriority.BACKGROUND_TRAINING)) <= int(requester_priority):
                return None
            running["cancel_requested"] = True
            running["requeue_on_cancel"] = True
            running["cancel_reason"] = "interactive_preempt"
            running["updated_at"] = _utc_now_iso()
            return running_job_id

    def start_workers(self) -> None:
        with self._condition:
            if self._worker_thread and self._worker_thread.is_alive():
                return
            self._stop_event.clear()
            self._worker_thread = threading.Thread(target=self._worker_loop, name="job-orchestrator", daemon=True)
            self._worker_thread.start()

    def shutdown_workers(self) -> None:
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)

    def get_job_status(self, job_id: str) -> dict:
        return copy.deepcopy(self._jobs.get(str(job_id), {}))

    def _next_job_id(self) -> str | None:
        while self._heap:
            _, _, job_id = heapq.heappop(self._heap)
            record = self._jobs.get(job_id)
            if record is None:
                continue
            if record["state"] != JobState.QUEUED.value:
                continue
            return job_id
        return None

    def _run_direct_job(self, record: dict) -> dict | None:
        handler = self._handlers.get(str(record["job_type"]))
        if handler is None:
            raise KeyError(f"No handler registered for job type: {record['job_type']}")
        return handler(copy.deepcopy(record["payload"]))

    def _run_isolated_job(self, record: dict) -> dict | None:
        ctx = multiprocessing.get_context("spawn")
        result_queue = ctx.Queue()
        process = ctx.Process(
            target=_isolated_job_entry,
            args=(str(record["job_type"]), copy.deepcopy(record["payload"]), result_queue),
            daemon=True,
        )
        process.start()
        try:
            while process.is_alive():
                if bool(record.get("cancel_requested")):
                    process.terminate()
                    process.join(timeout=5.0)
                    raise InterruptedError(record.get("cancel_reason") or "canceled")
                time.sleep(0.1)
            process.join(timeout=5.0)
            try:
                result = result_queue.get_nowait()
            except queue.Empty:
                if process.exitcode == 0:
                    return {}
                raise RuntimeError(f"Isolated job exited without a result payload (exitcode={process.exitcode}).")
            if not result.get("ok"):
                raise RuntimeError(result.get("error") or "isolated job failed")
            return result.get("result") or {}
        finally:
            if process.is_alive():  # pragma: no cover - defensive cleanup
                process.terminate()
                process.join(timeout=5.0)

    def _requeue_record(self, record: dict, reason: str) -> None:
        record["state"] = JobState.QUEUED.value
        record["cancel_requested"] = False
        record["requeue_on_cancel"] = False
        record["cancel_reason"] = None
        record["updated_at"] = _utc_now_iso()
        record["last_requeue_reason"] = str(reason)
        self._sequence += 1
        heapq.heappush(self._heap, (int(record["priority"]), self._sequence, record["job_id"]))
        self._notify("requeued", record["job_id"])

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            with self._condition:
                job_id = self._next_job_id()
                if job_id is None:
                    self._condition.wait(timeout=0.2)
                    continue
                record = self._jobs[job_id]
                if record["state"] != JobState.QUEUED.value:
                    continue
                record["state"] = JobState.RUNNING.value
                record["started_at"] = _utc_now_iso()
                record["updated_at"] = _utc_now_iso()
                record["attempt_count"] = int(record.get("attempt_count", 0)) + 1
                self._running_job_id = job_id
            self._notify("started", job_id)

            lease = self._lease_manager.acquire(
                resource_name=str(record["resource_name"]),
                owner=str(job_id),
                priority=int(record["priority"]),
            )
            try:
                queue_wait_seconds = 0.0
                if record.get("created_at"):
                    queue_wait_seconds = max(
                        0.0,
                        (
                            datetime.fromisoformat(record["started_at"])
                            - datetime.fromisoformat(record["created_at"])
                        ).total_seconds(),
                    )
                if bool(record.get("isolated")):
                    result = self._run_isolated_job(record)
                else:
                    result = self._run_direct_job(record)
                record["state"] = JobState.COMPLETED.value
                record["result"] = result
                record["queue_wait_seconds"] = queue_wait_seconds
                record["finished_at"] = _utc_now_iso()
                record["updated_at"] = _utc_now_iso()
                self._notify("completed", job_id)
            except InterruptedError as exc:
                if bool(record.get("requeue_on_cancel")):
                    record["state"] = JobState.REQUEUED.value
                    record["updated_at"] = _utc_now_iso()
                    self._notify("canceled", job_id)
                    with self._condition:
                        self._requeue_record(record, str(exc))
                else:
                    record["state"] = JobState.CANCELED.value
                    record["error"] = str(exc)
                    record["finished_at"] = _utc_now_iso()
                    record["updated_at"] = _utc_now_iso()
                    self._notify("canceled", job_id)
            except Exception as exc:
                record["state"] = JobState.FAILED.value
                record["error"] = str(exc)
                record["traceback"] = traceback.format_exc()
                record["finished_at"] = _utc_now_iso()
                record["updated_at"] = _utc_now_iso()
                self._notify("failed", job_id)
            finally:
                self._lease_manager.release(lease)
                with self._condition:
                    if self._running_job_id == job_id:
                        self._running_job_id = None
