from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, TypeVar

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


def summarize_gpu_job(job_name: str, metadata: dict, fn: Callable[[], T]) -> tuple[T, dict]:
    metadata = dict(metadata or {})
    cuda_available = bool(torch.cuda.is_available())
    device_name = torch.cuda.get_device_name(torch.cuda.current_device()) if cuda_available else "cpu"

    if cuda_available:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start = time.perf_counter()
    result = fn()

    if cuda_available:
        torch.cuda.synchronize()

    summary = {
        "job_name": job_name,
        "started_at": _utc_now_iso(),
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
    if os.getenv("ACTIVE_LEARNING_PROFILE_CUDA") == "1":
        return True
    profiling_root = Path(profiling_root)
    marker = profiling_root / f"{job_family}_trace_seen.marker"
    if marker.exists():
        return False
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(_utc_now_iso(), encoding="utf-8")
    return True
