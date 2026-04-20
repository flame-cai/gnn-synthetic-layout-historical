import shutil
import sys
import threading
import time
import unittest
from pathlib import Path


TESTS_ROOT = Path(__file__).resolve().parent
APP_ROOT = TESTS_ROOT.parent
REPO_ROOT = APP_ROOT.parent

if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from job_orchestrator import JobOrchestrator, JobPriority, JobType, QueuedJob


class JobOrchestratorUnitTest(unittest.TestCase):
    def test_higher_priority_job_runs_first_when_both_are_queued(self):
        order = []
        orchestrator = JobOrchestrator()
        orchestrator.register_handler(JobType.OCR_VERIFY.value, lambda payload: order.append(payload["name"]) or {})

        low_job = QueuedJob(
            job_type=JobType.OCR_VERIFY.value,
            manuscript="m",
            manuscript_root="m",
            payload={"name": "low"},
            priority=int(JobPriority.BACKGROUND_TRAINING),
            resource_name="cpu",
        )
        high_job = QueuedJob(
            job_type=JobType.OCR_VERIFY.value,
            manuscript="m",
            manuscript_root="m",
            payload={"name": "high"},
            priority=int(JobPriority.SAVE_FOLLOWUP),
            resource_name="cpu",
        )

        low_id = orchestrator.enqueue(low_job)
        high_id = orchestrator.enqueue(high_job)
        orchestrator.start_workers()

        deadline = time.time() + 5.0
        while time.time() < deadline:
            if orchestrator.get_job_status(low_id).get("state") == "completed" and orchestrator.get_job_status(high_id).get("state") == "completed":
                break
            time.sleep(0.05)
        orchestrator.shutdown_workers()

        self.assertEqual(order, ["high", "low"])

    def test_interactive_preempt_marks_running_low_priority_job_for_cancel_and_requeue(self):
        started = threading.Event()
        released = threading.Event()

        def slow_handler(_payload):
            started.set()
            released.wait(timeout=2.0)
            return {}

        orchestrator = JobOrchestrator()
        orchestrator.register_handler(JobType.OCR_VERIFY.value, slow_handler)
        job = QueuedJob(
            job_type=JobType.OCR_VERIFY.value,
            manuscript="m",
            manuscript_root="m",
            payload={"name": "slow"},
            priority=int(JobPriority.BACKGROUND_TRAINING),
            resource_name="gpu",
        )

        job_id = orchestrator.enqueue(job)
        orchestrator.start_workers()
        self.assertTrue(started.wait(timeout=2.0))

        preempted_id = orchestrator.preempt_for_interactive()
        released.set()
        deadline = time.time() + 5.0
        while time.time() < deadline:
            status = orchestrator.get_job_status(job_id)
            if status.get("cancel_requested"):
                break
            time.sleep(0.05)
        orchestrator.shutdown_workers()

        status = orchestrator.get_job_status(job_id)
        self.assertEqual(preempted_id, job_id)
        self.assertTrue(status["cancel_requested"])
        self.assertTrue(status["requeue_on_cancel"])

    def test_listener_exception_does_not_kill_worker_thread(self):
        completed = threading.Event()
        orchestrator = JobOrchestrator()

        def handler(_payload):
            completed.set()
            return {"ok": True}

        def flaky_listener(event_name, _status):
            if event_name == "started":
                raise RuntimeError("listener boom")

        orchestrator.register_handler(JobType.OCR_VERIFY.value, handler)
        orchestrator.set_state_listener(flaky_listener)
        job = QueuedJob(
            job_type=JobType.OCR_VERIFY.value,
            manuscript="m",
            manuscript_root="m",
            payload={"name": "listener_safe"},
            priority=int(JobPriority.SAVE_FOLLOWUP),
            resource_name="cpu",
        )

        job_id = orchestrator.enqueue(job)
        orchestrator.start_workers()
        self.assertTrue(completed.wait(timeout=2.0))

        deadline = time.time() + 5.0
        while time.time() < deadline:
            if orchestrator.get_job_status(job_id).get("state") == "completed":
                break
            time.sleep(0.05)
        orchestrator.shutdown_workers()

        self.assertEqual(orchestrator.get_job_status(job_id).get("state"), "completed")


if __name__ == "__main__":
    unittest.main()
