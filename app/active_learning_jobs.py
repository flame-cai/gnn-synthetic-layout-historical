"""Isolated-job router for the two active-learning lineages.

`JobOrchestrator` runs training in a spawned child process, so the child needs
one entry point that can reach either runtime. Keeping the router here rather
than in one of the runtimes avoids making the OCR module import the layout
module (or the reverse) just to dispatch.

Each runtime keeps its own `dispatch_isolated_job`; this only chooses between
them, and imports lazily so a child process spawned for a layout job never pays
to import the OCR stack.
"""
from __future__ import annotations

from job_orchestrator import JobType


_OCR_JOB_TYPES = {JobType.OCR_FINE_TUNE.value, JobType.OCR_REBASE.value}
_LAYOUT_JOB_TYPES = {
    JobType.GNN_FINE_TUNE.value,
    JobType.GNN_REBASE.value,
    JobType.GNN_BACKFILL.value,
}


def dispatch_isolated_job(job_type: str, payload: dict) -> dict:
    job_type = str(job_type)
    if job_type in _OCR_JOB_TYPES:
        from ocr_active_learning_runtime import dispatch_isolated_job as dispatch_ocr

        return dispatch_ocr(job_type, payload)
    if job_type in _LAYOUT_JOB_TYPES:
        from layout_active_learning_runtime import dispatch_isolated_job as dispatch_layout

        return dispatch_layout(job_type, payload)
    raise KeyError(f"Unsupported isolated active-learning job type: {job_type}")


def handle_orchestrator_event(event_name: str, job_status: dict) -> None:
    """Mirror one job-state change into the registry that owns that job type.

    The orchestrator holds a single listener, so both runtimes install this and
    it routes. Each runtime also re-checks the job type, so nothing writes OCR
    status for a layout job or the reverse.
    """
    job_type = str((job_status or {}).get("job_type") or "")
    if job_type in _OCR_JOB_TYPES:
        from ocr_active_learning_runtime import _handle_orchestrator_event as handle_ocr

        handle_ocr(event_name, job_status)
        return
    if job_type in _LAYOUT_JOB_TYPES:
        from layout_active_learning_runtime import handle_orchestrator_event as handle_layout

        handle_layout(event_name, job_status)
