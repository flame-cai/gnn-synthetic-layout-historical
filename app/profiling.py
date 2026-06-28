from __future__ import annotations

from contextlib import contextmanager, nullcontext
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator, Mapping, TypeVar

import torch


T = TypeVar("T")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_profile_summary(output_dir: str | Path, job_name: str, summary: dict) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_job_name = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in job_name)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    target = output_dir / f"{timestamp}_{safe_job_name}.json"
    target.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return target


def _coerce_bool(value) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on", "enabled"}:
        return True
    if normalized in {"0", "false", "no", "off", "disabled", ""}:
        return False
    return None


def layout_save_timing_enabled(config: Mapping[str, object] | None = None) -> bool:
    config = dict(config or {})
    for key in ("layout_save_timing_enabled", "LAYOUT_SAVE_TIMING_ENABLED"):
        configured = _coerce_bool(config.get(key))
        if configured is not None:
            return configured
    return _coerce_bool(os.getenv("LAYOUT_SAVE_TIMING_ENABLED")) is True


def _layout_save_timing_output_path(
    manuscript_root: str | Path,
    config: Mapping[str, object] | None = None,
) -> Path:
    config = dict(config or {})
    explicit_path = config.get("layout_save_timing_log_path") or os.getenv("LAYOUT_SAVE_TIMING_LOG_PATH")
    if explicit_path:
        return Path(str(explicit_path))
    explicit_dir = config.get("layout_save_timing_log_dir") or os.getenv("LAYOUT_SAVE_TIMING_LOG_DIR")
    if explicit_dir:
        return Path(str(explicit_dir)) / "layout_save_timings.jsonl"
    return Path(manuscript_root) / "layout_analysis_output" / "profiling" / "layout_save_timings.jsonl"


class LayoutSaveTimingRecorder:
    def __init__(
        self,
        *,
        enabled: bool,
        output_path: str | Path,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.output_path = Path(output_path)
        self.metadata = dict(metadata or {})
        self.started_at = _utc_now_iso()
        self._start = time.perf_counter()
        self._chunks: list[dict] = []
        self._written = False

    @contextmanager
    def chunk(self, name: str, metadata: Mapping[str, object] | None = None) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        chunk_start = time.perf_counter()
        chunk = {
            "name": str(name),
            "started_at": _utc_now_iso(),
            "metadata": dict(metadata or {}),
        }
        try:
            yield
        except Exception as exc:
            chunk["status"] = "failed"
            chunk["error"] = str(exc)
            raise
        else:
            chunk["status"] = "success"
        finally:
            chunk["finished_at"] = _utc_now_iso()
            chunk["duration_seconds"] = time.perf_counter() - chunk_start
            self._chunks.append(chunk)

    def finish(self, status: str = "success", metadata: Mapping[str, object] | None = None) -> Path | None:
        if not self.enabled or self._written:
            return None
        self._written = True
        payload = {
            "schema_version": 1,
            "event_type": "layout_save_timing",
            "status": str(status),
            "started_at": self.started_at,
            "finished_at": _utc_now_iso(),
            "duration_seconds": time.perf_counter() - self._start,
            "metadata": {**self.metadata, **dict(metadata or {})},
            "chunks": self._chunks,
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return self.output_path


class _DisabledLayoutSaveTimingRecorder:
    enabled = False
    output_path = None

    def chunk(self, name: str, metadata: Mapping[str, object] | None = None):
        return nullcontext()

    def finish(self, status: str = "success", metadata: Mapping[str, object] | None = None) -> None:
        return None


def create_layout_save_timing_recorder(
    manuscript_root: str | Path,
    *,
    page_id: str,
    config: Mapping[str, object] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> LayoutSaveTimingRecorder | _DisabledLayoutSaveTimingRecorder:
    if not layout_save_timing_enabled(config):
        return _DisabledLayoutSaveTimingRecorder()
    return LayoutSaveTimingRecorder(
        enabled=True,
        output_path=_layout_save_timing_output_path(manuscript_root, config),
        metadata={
            "manuscript": Path(manuscript_root).name,
            "page_id": str(page_id),
            **dict(metadata or {}),
        },
    )


def summarize_gpu_job(job_name: str, metadata: dict, fn: Callable[[], T]) -> tuple[T, dict]:
    metadata = dict(metadata or {})
    cuda_available = bool(torch.cuda.is_available())
    device_name = torch.cuda.get_device_name(torch.cuda.current_device()) if cuda_available else "cpu"

    if cuda_available:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    started_at = _utc_now_iso()
    start = time.perf_counter()
    result = fn()

    if cuda_available:
        torch.cuda.synchronize()

    finished_at = _utc_now_iso()
    summary = {
        "job_name": job_name,
        "started_at": started_at,
        "finished_at": finished_at,
        "wall_time_seconds": time.perf_counter() - start,
        "cuda_available": cuda_available,
        "device": device_name,
        "peak_cuda_memory_allocated": int(torch.cuda.max_memory_allocated()) if cuda_available else 0,
        "peak_cuda_memory_reserved": int(torch.cuda.max_memory_reserved()) if cuda_available else 0,
    }
    summary.update(metadata)
    return result, summary


def maybe_write_cuda_trace(job_name: str, output_dir: str | Path, enabled: bool, fn: Callable[[], T]) -> T:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not enabled or not torch.cuda.is_available():
        return fn()

    trace_name = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in job_name)
    trace_root = output_dir / f"{trace_name}_trace"
    trace_root.mkdir(parents=True, exist_ok=True)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(trace_root)),
    ) as profiler:
        result = fn()
        profiler.step()
    return result


def should_capture_cuda_trace(job_family: str, profiling_root: str | Path) -> bool:
    if os.getenv("ACTIVE_LEARNING_PROFILE_CUDA") != "1":
        return False
    profiling_root = Path(profiling_root)
    marker = profiling_root / f"{job_family}_trace_seen.marker"
    if marker.exists():
        return False
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(_utc_now_iso(), encoding="utf-8")
    return True
