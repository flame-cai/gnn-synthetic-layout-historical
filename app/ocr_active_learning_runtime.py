from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path

from job_orchestrator import JobOrchestrator, JobPriority, JobType, QueuedJob
from manuscript_ocr_registry import ManuscriptOcrRegistry, load_registry
from profiling import (
    maybe_write_cuda_trace,
    should_capture_cuda_trace,
    summarize_gpu_job,
    write_profile_summary,
)
from recognition.active_learning import (
    fine_tune_checkpoint_on_pages,
    prepare_page_datasets,
    run_checkpoint_on_prepared_pages,
)
from recognition.active_learning_recipe import (
    DEFAULT_OCR_ACTIVE_LEARNING_RECIPE,
    OcrActiveLearningRecipe,
)
from telemetry import append_jsonl, compute_layout_edit_metrics, compute_text_edit_metrics, update_summary_json, utc_now_iso


_RUNTIME_STATE = {
    "base_checkpoint_path": None,
    "orchestrator": None,
}


def configure_runtime(base_checkpoint_path: str | Path, orchestrator: JobOrchestrator | None = None) -> None:
    _RUNTIME_STATE["base_checkpoint_path"] = str(Path(base_checkpoint_path).resolve())
    if orchestrator is not None:
        _RUNTIME_STATE["orchestrator"] = orchestrator
        orchestrator.set_state_listener(_handle_orchestrator_event)
        orchestrator.start_workers()


def _base_checkpoint_path(explicit_base_checkpoint_path: str | Path | None = None) -> str:
    base_checkpoint_path = explicit_base_checkpoint_path or _RUNTIME_STATE.get("base_checkpoint_path")
    if not base_checkpoint_path:
        raise FileNotFoundError("OCR active-learning runtime does not know the base checkpoint path yet.")
    return str(Path(base_checkpoint_path).resolve())


def _get_orchestrator(orchestrator: JobOrchestrator | None = None) -> JobOrchestrator | None:
    return orchestrator or _RUNTIME_STATE.get("orchestrator")


def _load_registry_for_manuscript(manuscript_root: str | Path, base_checkpoint_path: str | Path | None = None) -> ManuscriptOcrRegistry:
    return load_registry(manuscript_root, base_checkpoint_path=_base_checkpoint_path(base_checkpoint_path))


def _safe_slug(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in str(value))


def build_revision_hash(graph_payload: dict | None, text_payload: dict | None, textbox_labels=None) -> str:
    payload = {
        "graph": graph_payload or {},
        "text": text_payload or {},
        "textbox_labels": textbox_labels or [],
    }
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(serialized.encode("utf-8")).hexdigest()


def _recipe_from_payload(job_payload: dict | None = None) -> OcrActiveLearningRecipe:
    payload = dict(job_payload or {})
    recipe_payload = payload.get("recipe")
    if not recipe_payload:
        return DEFAULT_OCR_ACTIVE_LEARNING_RECIPE
    return OcrActiveLearningRecipe(**recipe_payload)


def _snapshot_page_revision(registry: ManuscriptOcrRegistry, page_id: str, revision_number: int) -> Path:
    snapshot_root = registry.revision_snapshot_root(page_id, revision_number)
    if snapshot_root.exists():
        shutil.rmtree(snapshot_root)
    (snapshot_root / "page-xml-format").mkdir(parents=True, exist_ok=True)
    (snapshot_root / "images_resized").mkdir(parents=True, exist_ok=True)

    live_xml = registry.manuscript_root / "layout_analysis_output" / "page-xml-format" / f"{page_id}.xml"
    live_image = registry.manuscript_root / "layout_analysis_output" / "images_resized" / f"{page_id}.jpg"
    if not live_image.exists():
        live_image = registry.manuscript_root / "images_resized" / f"{page_id}.jpg"

    shutil.copy2(live_xml, snapshot_root / "page-xml-format" / f"{page_id}.xml")
    shutil.copy2(live_image, snapshot_root / "images_resized" / f"{page_id}.jpg")
    return snapshot_root


def _revision_snapshot_dirs(registry: ManuscriptOcrRegistry, page_id: str, revision_number: int) -> tuple[Path, Path]:
    snapshot_root = registry.revision_snapshot_root(page_id, revision_number)
    return snapshot_root / "images_resized", snapshot_root / "page-xml-format"


def _prepare_revision_pages(
    registry: ManuscriptOcrRegistry,
    revision_refs: list[dict],
    purpose_slug: str,
) -> list:
    prepared_pages = []
    output_root = registry.prepared_pages_root / purpose_slug
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for revision_ref in revision_refs:
        page_id = str(revision_ref["page_id"])
        revision_number = int(revision_ref["revision_number"])
        images_dir, pagexml_dir = _revision_snapshot_dirs(registry, page_id, revision_number)
        prepared = prepare_page_datasets(
            images_dir=images_dir,
            pagexml_dir=pagexml_dir,
            page_ids=[page_id],
            output_root=output_root / f"{_safe_slug(page_id)}_r{revision_number:04d}",
        )
        prepared_pages.append(prepared[page_id])
    return prepared_pages


def _revision_ref(page_id: str, revision_number: int) -> dict:
    return {"page_id": str(page_id), "revision_number": int(revision_number)}


def _prediction_payload(predicted_lines: dict, recognition_engine: str, checkpoint_id: str | None, checkpoint_path: str | None, confidences: dict | None = None) -> dict:
    return {
        "predicted_lines": dict(predicted_lines or {}),
        "recognition_engine": str(recognition_engine),
        "checkpoint_id": checkpoint_id,
        "checkpoint_path": checkpoint_path,
        "confidences": dict(confidences or {}),
        "recorded_at": utc_now_iso(),
    }


def record_prediction(
    manuscript_root: str | Path,
    page_id: str,
    predicted_lines: dict,
    recognition_engine: str,
    checkpoint_id: str | None,
    checkpoint_path: str | Path | None,
    confidences: dict | None = None,
    base_checkpoint_path: str | Path | None = None,
) -> dict:
    registry = _load_registry_for_manuscript(manuscript_root, base_checkpoint_path=base_checkpoint_path)
    payload = _prediction_payload(
        predicted_lines=predicted_lines,
        recognition_engine=recognition_engine,
        checkpoint_id=checkpoint_id,
        checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
        confidences=confidences,
    )
    registry.remember_prediction(page_id, payload)
    return payload


def _build_status_payload(registry: ManuscriptOcrRegistry) -> dict:
    status = registry.get_status()
    active_checkpoint_path = str(registry.active_checkpoint())
    return {
        "code": status.get("code", "idle"),
        "label": status.get("label", "AL: idle"),
        "updated_at": status.get("updated_at"),
        "details": status.get("details", {}),
        "active_checkpoint_id": registry.active_checkpoint_id(),
        "active_checkpoint_path": active_checkpoint_path,
        "needs_rebase": bool(registry.data.get("needs_rebase", False)),
        "active_learning_enabled": bool(registry.data.get("active_learning_enabled", False)),
        "pending_jobs": registry.pending_ocr_work(),
    }


def summarize_manuscript_active_learning(
    manuscript_root: str | Path,
    base_checkpoint_path: str | Path | None = None,
) -> dict:
    registry = _load_registry_for_manuscript(manuscript_root, base_checkpoint_path=base_checkpoint_path)
    return _build_status_payload(registry)


def _job_summary_label(job_type: str, payload: dict, state: str) -> str:
    page_id = payload.get("page_id")
    if job_type == JobType.OCR_REBASE.value:
        if state == "running":
            return "AL: rebuilding manuscript"
        if state == "queued":
            return "AL: queued rebuild"
    if job_type == JobType.OCR_FINE_TUNE.value:
        if state == "running":
            return f"AL: training page {page_id}"
        if state == "queued":
            return f"AL: queued page {page_id}"
    return f"AL: {state.replace('_', ' ')}"


def _record_job_event(registry: ManuscriptOcrRegistry, event_name: str, job_status: dict) -> None:
    payload = {
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
    }
    append_jsonl(registry.telemetry_root / "job_events.jsonl", payload)


def _handle_orchestrator_event(event_name: str, job_status: dict) -> None:
    manuscript_root = (job_status.get("payload") or {}).get("manuscript_root") or job_status.get("manuscript_root")
    if not manuscript_root:
        return
    registry = _load_registry_for_manuscript(manuscript_root)
    job_id = str(job_status["job_id"])
    if event_name == "queued":
        registry.enqueue_pending_job(
            {
                "job_id": job_id,
                "job_type": job_status["job_type"],
                "page_id": (job_status.get("payload") or {}).get("page_id"),
                "revision_number": (job_status.get("payload") or {}).get("revision_number"),
                "priority": job_status["priority"],
                "state": job_status["state"],
                "created_at": job_status["created_at"],
            }
        )
        registry.set_status(
            "queued",
            _job_summary_label(str(job_status["job_type"]), job_status.get("payload") or {}, "queued"),
            job_id=job_id,
        )
    elif event_name == "started":
        registry.update_pending_job(job_id, state=job_status["state"], started_at=job_status.get("started_at"))
        registry.set_status(
            "running",
            _job_summary_label(str(job_status["job_type"]), job_status.get("payload") or {}, "running"),
            job_id=job_id,
        )
    elif event_name == "requeued":
        registry.update_pending_job(job_id, state="queued", requeued_at=utc_now_iso())
        registry.set_status("paused_for_ocr", "AL: paused for OCR", job_id=job_id)
    elif event_name in {"completed", "failed", "canceled"}:
        registry.remove_pending_job(job_id)
        if event_name == "completed":
            status_code = "needs_rebase" if registry.data.get("needs_rebase") else "idle"
            label = "AL: needs rebase" if registry.data.get("needs_rebase") else "AL: idle"
            registry.set_status(status_code, label, job_id=job_id)
        elif event_name == "failed":
            registry.set_status("failed", "AL: failed", job_id=job_id, error=job_status.get("error"))
        elif event_name == "canceled":
            registry.set_status("paused_for_ocr", "AL: paused for OCR", job_id=job_id)
    _record_job_event(registry, event_name, job_status)


def prepare_for_interactive_ocr(
    manuscript_root: str | Path,
    orchestrator: JobOrchestrator | None = None,
    timeout_seconds: float = 30.0,
) -> None:
    orchestrator = _get_orchestrator(orchestrator)
    if orchestrator is None:
        return
    running_job_id = orchestrator.preempt_for_interactive()
    if not running_job_id:
        return
    registry = _load_registry_for_manuscript(manuscript_root)
    registry.set_status("paused_for_ocr", "AL: paused for OCR", job_id=running_job_id)
    deadline = time.time() + float(timeout_seconds)
    while time.time() < deadline:
        status = orchestrator.get_job_status(running_job_id)
        if status.get("state") != "running":
            return
        time.sleep(0.1)


def handle_post_save(
    manuscript: str,
    page: str,
    save_intent: str,
    active_learning_enabled: bool,
    recognition_engine: str,
    text_payload: dict | None,
    manuscript_root: str | Path | None = None,
    base_checkpoint_path: str | Path | None = None,
    graph_payload: dict | None = None,
    textbox_labels=None,
    modifications: list[dict] | None = None,
    orchestrator: JobOrchestrator | None = None,
) -> dict:
    manuscript_root = Path(manuscript_root or Path("input_manuscripts") / manuscript)
    registry = _load_registry_for_manuscript(manuscript_root, base_checkpoint_path=base_checkpoint_path)
    registry.set_active_learning_enabled(bool(active_learning_enabled))

    text_payload = dict(text_payload or {})
    content_hash = build_revision_hash(graph_payload, text_payload, textbox_labels=textbox_labels)
    non_empty_text_lines = sum(1 for value in text_payload.values() if str(value or "").strip())
    supervision_present = non_empty_text_lines > 0
    last_prediction = registry.get_last_prediction(page) or {}

    revision = registry.record_page_revision(
        page,
        {
            "content_hash": content_hash,
            "save_intent": save_intent,
            "supervision_present": supervision_present,
            "recognition_engine": recognition_engine,
            "text_line_count": len(text_payload),
            "text_non_empty_line_count": non_empty_text_lines,
            "prediction_engine": last_prediction.get("recognition_engine"),
            "prediction_checkpoint_id": last_prediction.get("checkpoint_id"),
            "prediction_recorded_at": last_prediction.get("recorded_at"),
            "content_summary": {
                "node_count": len((graph_payload or {}).get("nodes", [])),
                "edge_count": len((graph_payload or {}).get("edges", [])),
            },
        },
    )

    if not revision.is_duplicate:
        _snapshot_page_revision(registry, page, revision.revision_number)

    entered_active_learning = False
    queued_job_ids = []
    save_intent = str(save_intent or "commit")

    if save_intent == "commit" and bool(active_learning_enabled) and not revision.is_duplicate:
        if registry.has_consumed_revision(page):
            registry.mark_rebase_needed("consumed_page_revision_changed", page)
            entered_active_learning = True
            existing_rebase_job = any(job.get("job_type") == JobType.OCR_REBASE.value for job in registry.pending_ocr_work())
            if not existing_rebase_job and _get_orchestrator(orchestrator) is not None:
                approved_refs = [
                    _revision_ref(revision_payload["page_id"], revision_payload["revision_number"])
                    for revision_payload in registry.latest_supervised_commit_revisions()
                ]
                rebase_job = QueuedJob(
                    job_type=JobType.OCR_REBASE.value,
                    manuscript=manuscript,
                    manuscript_root=str(manuscript_root),
                    priority=int(JobPriority.BULK_PREPROCESS),
                    resource_name="gpu",
                    isolated=True,
                    payload={
                        "job_type": JobType.OCR_REBASE.value,
                        "manuscript": manuscript,
                        "manuscript_root": str(manuscript_root),
                        "base_checkpoint_path": _base_checkpoint_path(base_checkpoint_path),
                        "approved_revision_refs": approved_refs,
                        "recipe": DEFAULT_OCR_ACTIVE_LEARNING_RECIPE.to_dict(),
                    },
                )
                queued_job_ids.append(_get_orchestrator(orchestrator).enqueue(rebase_job))
        elif supervision_present and _get_orchestrator(orchestrator) is not None:
            approved_history = registry.approved_supervised_revisions()
            candidate_id = f"ocr_{_safe_slug(page)}_r{revision.revision_number:04d}"
            fine_tune_job = QueuedJob(
                job_type=JobType.OCR_FINE_TUNE.value,
                manuscript=manuscript,
                manuscript_root=str(manuscript_root),
                priority=int(JobPriority.BACKGROUND_TRAINING),
                resource_name="gpu",
                isolated=True,
                payload={
                    "job_type": JobType.OCR_FINE_TUNE.value,
                    "manuscript": manuscript,
                    "manuscript_root": str(manuscript_root),
                    "page_id": str(page),
                    "revision_number": int(revision.revision_number),
                    "training_revision_ref": _revision_ref(page, revision.revision_number),
                    "history_revision_refs": [
                        _revision_ref(revision_payload["page_id"], revision_payload["revision_number"])
                        for revision_payload in approved_history
                    ],
                    "verifier_revision_refs": [
                        _revision_ref(revision_payload["page_id"], revision_payload["revision_number"])
                        for revision_payload in approved_history
                    ],
                    "candidate_id": candidate_id,
                    "parent_checkpoint_id": registry.active_checkpoint_id(),
                    "parent_checkpoint_path": str(registry.active_checkpoint()),
                    "base_checkpoint_path": _base_checkpoint_path(base_checkpoint_path),
                    "recipe": DEFAULT_OCR_ACTIVE_LEARNING_RECIPE.to_dict(),
                },
            )
            queued_job_ids.append(_get_orchestrator(orchestrator).enqueue(fine_tune_job))
            entered_active_learning = True

    text_edit_metrics = compute_text_edit_metrics(last_prediction.get("predicted_lines", {}), text_payload)
    layout_metrics = compute_layout_edit_metrics(modifications)
    page_event = {
        "recorded_at": utc_now_iso(),
        "manuscript": manuscript,
        "page_id": page,
        "revision_number": revision.revision_number,
        "save_intent": save_intent,
        "active_learning_enabled": bool(active_learning_enabled),
        "entered_active_learning": bool(entered_active_learning),
        "revision_is_duplicate": bool(revision.is_duplicate),
        "supervision_present": bool(supervision_present),
        "prediction_engine": last_prediction.get("recognition_engine"),
        "prediction_checkpoint_id": last_prediction.get("checkpoint_id"),
        "recognition_engine": recognition_engine,
        "layout_metrics": layout_metrics,
        "text_metrics": text_edit_metrics,
    }
    append_jsonl(registry.telemetry_root / "page_events.jsonl", page_event)
    update_summary_json(
        registry.telemetry_root / "page_edit_summary.json",
        f"{page}#r{revision.revision_number}",
        page_event,
    )

    if save_intent == "draft":
        registry.set_status("idle", "AL: idle")
    elif registry.data.get("needs_rebase"):
        registry.set_status("needs_rebase", "AL: needs rebase")

    return {
        "revision": revision.to_dict(),
        "active_learning": _build_status_payload(registry),
        "queued_job_ids": queued_job_ids,
        "entered_active_learning": entered_active_learning,
    }


def verify_candidate_against_bank(job_payload: dict) -> dict:
    registry = _load_registry_for_manuscript(job_payload["manuscript_root"], base_checkpoint_path=job_payload.get("base_checkpoint_path"))
    recipe = _recipe_from_payload(job_payload)
    verifier_refs = list(job_payload.get("verifier_revision_refs") or [])
    candidate_checkpoint_path = str(job_payload["candidate_checkpoint_path"])

    if not verifier_refs:
        return {
            "passed": True,
            "reason": "bootstrap_no_verifier_bank",
            "guard_abs": float(recipe.regression_guard_abs),
            "candidate_page_cer": None,
            "active_page_cer": None,
        }

    bank_pages = _prepare_revision_pages(
        registry,
        verifier_refs,
        purpose_slug=f"verify_{_safe_slug(job_payload['candidate_id'])}",
    )
    active_checkpoint_path = str(job_payload.get("active_checkpoint_path") or registry.active_checkpoint())
    active_result = run_checkpoint_on_prepared_pages(
        active_checkpoint_path,
        bank_pages,
        output_root=None,
        width_policy=recipe.width_policy,
    )
    candidate_result = run_checkpoint_on_prepared_pages(
        candidate_checkpoint_path,
        bank_pages,
        output_root=None,
        width_policy=recipe.width_policy,
    )
    active_page_cer = float(active_result.aggregate_metrics["page_cer"])
    candidate_page_cer = float(candidate_result.aggregate_metrics["page_cer"])
    passed = candidate_page_cer <= (active_page_cer + float(recipe.regression_guard_abs))
    return {
        "passed": passed,
        "reason": "candidate_non_regressing" if passed else "candidate_regressed",
        "guard_abs": float(recipe.regression_guard_abs),
        "candidate_page_cer": candidate_page_cer,
        "active_page_cer": active_page_cer,
        "candidate_metrics": candidate_result.aggregate_metrics,
        "active_metrics": active_result.aggregate_metrics,
        "candidate_per_page": candidate_result.per_page_metrics,
        "active_per_page": active_result.per_page_metrics,
    }


def _train_candidate_step(job_payload: dict) -> dict:
    registry = _load_registry_for_manuscript(job_payload["manuscript_root"], base_checkpoint_path=job_payload.get("base_checkpoint_path"))
    recipe = _recipe_from_payload(job_payload)
    training_revision_ref = dict(job_payload["training_revision_ref"])
    history_revision_refs = list(job_payload.get("history_revision_refs") or [])
    candidate_id = str(job_payload["candidate_id"])
    candidate_root = registry.checkpoints_root / candidate_id
    training_pages = _prepare_revision_pages(
        registry,
        [training_revision_ref],
        purpose_slug=f"train_{_safe_slug(candidate_id)}",
    )
    history_pages = _prepare_revision_pages(
        registry,
        history_revision_refs,
        purpose_slug=f"history_{_safe_slug(candidate_id)}",
    ) if history_revision_refs else None

    def run_step():
        return fine_tune_checkpoint_on_pages(
            prepared_pages=training_pages,
            base_checkpoint=job_payload["parent_checkpoint_path"],
            output_root=candidate_root,
            step_index=int(job_payload.get("step_index") or training_revision_ref["revision_number"]),
            validation_ratio=0.0,
            split_seed=42,
            oversampling_policy=recipe.oversampling_policy,
            augmentation_policy=recipe.augmentation_policy,
            history_source_pages=history_pages,
            history_sample_line_count=int(recipe.history_sample_line_count),
            width_policy=recipe.width_policy,
            lr_scheduler=recipe.lr_scheduler,
            optimizer_name=recipe.optimizer,
            background_plus_rotation_variant_count=int(recipe.background_plus_rotation_variant_count),
            shuffle_train_each_epoch=bool(recipe.shuffle_train_each_epoch),
            **recipe.to_training_overrides(),
        )

    trace_enabled = should_capture_cuda_trace("ocr_fine_tune", registry.profiling_root)
    fine_tune_result, summary = summarize_gpu_job(
        "ocr_fine_tune",
        {
            "job_type": "ocr_fine_tune",
            "candidate_id": candidate_id,
            "page_id": training_revision_ref["page_id"],
            "revision_number": training_revision_ref["revision_number"],
            "queue_wait_seconds": float(job_payload.get("queue_wait_seconds") or 0.0),
            "line_count": len(training_pages[0].records) if training_pages else 0,
        },
        lambda: maybe_write_cuda_trace(
            "ocr_fine_tune",
            registry.profiling_root,
            trace_enabled,
            run_step,
        ),
    )
    write_profile_summary(registry.profiling_root, "ocr_fine_tune", summary)

    registry.mark_candidate(
        {
            "candidate_id": candidate_id,
            "checkpoint_path": fine_tune_result.output_checkpoint,
            "parent_checkpoint_id": job_payload.get("parent_checkpoint_id"),
            "page_id": training_revision_ref["page_id"],
            "revision_number": training_revision_ref["revision_number"],
            "metadata_path": str(candidate_root / "fine_tune_metadata.json"),
            "run_dir": str(candidate_root),
            "status": "candidate",
        }
    )
    verification = verify_candidate_against_bank(
        {
            **job_payload,
            "candidate_checkpoint_path": fine_tune_result.output_checkpoint,
            "candidate_id": candidate_id,
        }
    )
    checkpoint_record = registry._checkpoint_record(candidate_id)
    if checkpoint_record is not None:
        checkpoint_record["lineage_revision_refs"] = history_revision_refs + [training_revision_ref]
        registry.save()
    return {
        "candidate_id": candidate_id,
        "checkpoint_path": fine_tune_result.output_checkpoint,
        "metadata": {
            "output_checkpoint": fine_tune_result.output_checkpoint,
            "metadata_path": str(candidate_root / "fine_tune_metadata.json"),
            "run_dir": str(candidate_root),
        },
        "verification": verification,
        "training_revision_ref": training_revision_ref,
    }


def run_ocr_finetune_job(job_payload: dict) -> dict:
    registry = _load_registry_for_manuscript(job_payload["manuscript_root"], base_checkpoint_path=job_payload.get("base_checkpoint_path"))
    step_result = _train_candidate_step(job_payload)
    verification = step_result["verification"]
    if verification["passed"]:
        registry.promote_candidate(step_result["candidate_id"], verification)
        registry.mark_revision_consumed(
            step_result["training_revision_ref"]["page_id"],
            step_result["training_revision_ref"]["revision_number"],
            step_result["candidate_id"],
        )
        registry.set_status("idle", "AL: idle", candidate_id=step_result["candidate_id"])
    else:
        registry.reject_candidate(step_result["candidate_id"], verification)
        registry.set_status("idle", "AL: idle", candidate_id=step_result["candidate_id"])
    return {
        "candidate_id": step_result["candidate_id"],
        "checkpoint_path": step_result["checkpoint_path"],
        "promoted": bool(verification["passed"]),
        "verification": verification,
    }


def rebuild_manuscript_lineage(job_payload: dict) -> dict:
    registry = _load_registry_for_manuscript(job_payload["manuscript_root"], base_checkpoint_path=job_payload.get("base_checkpoint_path"))
    approved_revision_refs = list(job_payload.get("approved_revision_refs") or [])
    if not approved_revision_refs:
        registry.clear_rebase()
        registry.set_status("idle", "AL: idle")
        return {"rebuilt": False, "reason": "no_supervised_pages"}

    parent_checkpoint_path = _base_checkpoint_path(job_payload.get("base_checkpoint_path"))
    parent_checkpoint_id = "base"
    final_candidate_id = None
    for step_index, revision_ref in enumerate(approved_revision_refs, start=1):
        candidate_id = f"rebase_{step_index:03d}_{_safe_slug(revision_ref['page_id'])}_r{int(revision_ref['revision_number']):04d}"
        step_result = _train_candidate_step(
            {
                **job_payload,
                "candidate_id": candidate_id,
                "parent_checkpoint_id": parent_checkpoint_id,
                "parent_checkpoint_path": parent_checkpoint_path,
                "training_revision_ref": revision_ref,
                "history_revision_refs": approved_revision_refs[: step_index - 1],
                "verifier_revision_refs": approved_revision_refs[: step_index - 1],
                "step_index": step_index,
            }
        )
        if not step_result["verification"]["passed"]:
            registry.reject_candidate(step_result["candidate_id"], step_result["verification"])
            registry.set_status("needs_rebase", "AL: needs rebase", failed_candidate=step_result["candidate_id"])
            return {
                "rebuilt": False,
                "reason": "verification_failed",
                "candidate_id": step_result["candidate_id"],
                "verification": step_result["verification"],
            }
        parent_checkpoint_path = step_result["checkpoint_path"]
        parent_checkpoint_id = step_result["candidate_id"]
        final_candidate_id = step_result["candidate_id"]

    final_verification = {
        "passed": True,
        "reason": "rebase_complete",
        "rebuilt_revision_refs": approved_revision_refs,
    }
    registry.promote_candidate(final_candidate_id, final_verification)
    for revision_ref in approved_revision_refs:
        registry.mark_revision_consumed(revision_ref["page_id"], revision_ref["revision_number"], final_candidate_id)
    registry.clear_rebase()
    registry.set_status("idle", "AL: idle", candidate_id=final_candidate_id)
    return {"rebuilt": True, "candidate_id": final_candidate_id}


def dispatch_isolated_job(job_type: str, payload: dict) -> dict:
    if str(job_type) == JobType.OCR_FINE_TUNE.value:
        return run_ocr_finetune_job(payload)
    if str(job_type) == JobType.OCR_REBASE.value:
        return rebuild_manuscript_lineage(payload)
    raise KeyError(f"Unsupported isolated OCR active-learning job type: {job_type}")
