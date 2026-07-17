from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import shutil
import threading
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

from dotenv import load_dotenv

from .adapter import AdapterError, VLM_END_TO_END_PROMPT, parse_json_payload, vlm_json_to_pagexml
from .pagexml import empty_page_like, load_pagexml, write_pagexml
from .splits import IMAGE_EXTENSIONS, Fold, ManuscriptPaths, default_manuscript_paths, discover_page_ids
from .vlm_providers import (
    VlmProviderResponse,
    VlmProviderSpec,
    invoke_provider,
    provider_by_id,
    provider_by_method_id,
    validate_provider_runtime,
)


CACHE_SCHEMA_VERSION = 1
OUTPUT_ADAPTER_VERSION = 2
DEFAULT_TIMEOUT_SECONDS = 45.0
DEFAULT_PAGE_WORKERS = 4
DEFAULT_REQUEST_SPACING_SECONDS = 0.25
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BASE_DELAY_SECONDS = 1.0


class VlmCacheError(RuntimeError):
    pass


class _RateLimiter:
    def __init__(self, spacing_seconds: float) -> None:
        self._spacing_seconds = max(0.0, float(spacing_seconds))
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait_seconds = max(0.0, self._next_allowed - now)
            self._next_allowed = max(now, self._next_allowed) + self._spacing_seconds
        if wait_seconds:
            time.sleep(wait_seconds)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _prompt_sha256() -> str:
    return _sha256_bytes(VLM_END_TO_END_PROMPT.encode("utf-8"))


def _canonical_sha256(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return _sha256_bytes(encoded)


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _find_page_image(paths: ManuscriptPaths, page_id: str) -> Path:
    for extension in IMAGE_EXTENSIONS:
        candidate = paths.images_dir / f"{page_id}{extension}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No page image found for {paths.manuscript_id}/{page_id}.")


def _cache_dir(cache_root: Path, manuscript_id: str, method_id: str) -> Path:
    return cache_root / manuscript_id / method_id


def _request_payload(
    *,
    paths: ManuscriptPaths,
    page_id: str,
    spec: VlmProviderSpec,
) -> dict:
    image_path = _find_page_image(paths, page_id)
    pagexml_path = paths.pagexml_dir / f"{page_id}.xml"
    identity = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "manuscript_id": paths.manuscript_id,
        "page_id": page_id,
        "provider_id": spec.provider_id,
        "method_id": spec.method_id,
        "model_id": spec.model_id,
        "prompt_sha256": _prompt_sha256(),
        "image_sha256": _sha256_file(image_path),
        "pagexml_sha256": _sha256_file(pagexml_path),
    }
    if spec.request_contract_version != 1:
        identity["request_contract_version"] = spec.request_contract_version
    return {
        **identity,
        "request_fingerprint": _canonical_sha256(identity),
        "image_path": str(image_path.resolve()),
        "pagexml_path": str(pagexml_path.resolve()),
        "prompt": VLM_END_TO_END_PROMPT,
        "input_order": ["page_image", "prompt"],
    }


def _validate_terminal_result(result: dict, expected_request: dict, result_path: Path) -> None:
    if result.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise VlmCacheError(f"Unsupported cache schema in {result_path}.")
    if result.get("request_fingerprint") != expected_request["request_fingerprint"]:
        raise VlmCacheError(
            f"Cached VLM request does not match the current provider/model/prompt/input: {result_path}. "
            "Use a new cache root for a new acquisition."
        )
    if result.get("status") is None:
        raise VlmCacheError(f"Cache result is not terminal: {result_path}.")
    prediction_path = result_path.parent / "prediction.xml"
    if not prediction_path.exists():
        raise VlmCacheError(f"Cached prediction is missing: {prediction_path}.")


def _classify_exception(exc: Exception) -> str:
    if isinstance(exc, AdapterError):
        return str(exc) or "adapter_error"
    text = f"{type(exc).__name__}: {exc}".lower()
    if "timeout" in text or "timed out" in text:
        return "api_timeout"
    return "api_error"


def _try_repair_cached_json_fence(
    *,
    paths: ManuscriptPaths,
    page_id: str,
    page_dir: Path,
    result: dict,
) -> dict:
    if result.get("status") != "json_parse_error":
        return result
    template_page = load_pagexml(
        paths.pagexml_dir / f"{page_id}.xml",
        repair_geometry=True,
    )
    for response_path in sorted(page_dir.glob("attempt_*_response.txt"), reverse=True):
        try:
            parsed_payload = parse_json_payload(response_path.read_text(encoding="utf-8"))
            vlm_json_to_pagexml(
                parsed_payload,
                template_page=template_page,
                output_path=page_dir / "prediction.xml",
            )
        except (AdapterError, ValueError, OSError):
            continue
        _write_json_atomic(page_dir / "normalized_response.json", parsed_payload)
        repaired = {
            **result,
            "status": "success",
            "error": None,
            "output_adapter_version": OUTPUT_ADAPTER_VERSION,
            "derived_output_updated_at_utc": _utc_now(),
            "local_output_repairs": [
                *(result.get("local_output_repairs") or []),
                {
                    "repair_id": "strip_single_json_code_fence_v1",
                    "original_status": result.get("status"),
                    "source_response_path": response_path.name,
                    "paid_api_call_made": False,
                },
            ],
        }
        _write_json_atomic(page_dir / "result.json", repaired)
        return repaired
    return result


def _is_retryable_exception(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int) and 400 <= status_code < 500 and status_code != 429:
        return False
    text = f"{type(exc).__name__}: {exc}".lower()
    return not any(
        marker in text
        for marker in (
            "invalid_request_error",
            "authentication_error",
            "permission_error",
        )
    )


def _acquire_page(
    *,
    paths: ManuscriptPaths,
    page_id: str,
    spec: VlmProviderSpec,
    cache_dir: Path,
    api_key: str,
    timeout_seconds: float,
    max_retries: int,
    retry_base_delay_seconds: float,
    rate_limiter: _RateLimiter,
    invoke: Callable[..., VlmProviderResponse],
) -> dict:
    page_dir = cache_dir / "pages" / page_id
    request = _request_payload(paths=paths, page_id=page_id, spec=spec)
    result_path = page_dir / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        _validate_terminal_result(result, request, result_path)
        return _try_repair_cached_json_fence(
            paths=paths,
            page_id=page_id,
            page_dir=page_dir,
            result=result,
        )
    if page_dir.exists():
        raise VlmCacheError(
            f"Ambiguous non-terminal paid acquisition at {page_dir}. "
            "Inspect it manually and use a new cache root; automatic replay is forbidden."
        )

    page_dir.mkdir(parents=True, exist_ok=False)
    _write_json_atomic(
        page_dir / "request.json",
        {
            **request,
            "state": "started",
            "started_at_utc": _utc_now(),
            "timeout_seconds": timeout_seconds,
            "max_retries_after_initial_attempt": max_retries,
        },
    )

    image_path = _find_page_image(paths, page_id)
    gt_page = load_pagexml(paths.pagexml_dir / f"{page_id}.xml", repair_geometry=True)
    attempts: list[dict] = []
    final_status = "api_error"
    final_error: str | None = None
    total_input_tokens = 0
    total_output_tokens = 0
    total_tokens = 0
    acquisition_started = time.monotonic()

    for attempt_index in range(max_retries + 1):
        if attempt_index:
            time.sleep(retry_base_delay_seconds * (2 ** (attempt_index - 1)))
        rate_limiter.wait()
        attempt_started = time.monotonic()
        raw_text = ""
        try:
            response = invoke(
                spec,
                api_key=api_key,
                image_path=image_path,
                prompt=VLM_END_TO_END_PROMPT,
                timeout_seconds=timeout_seconds,
            )
            raw_text = response.raw_text
            total_input_tokens += response.input_tokens
            total_output_tokens += response.output_tokens
            total_tokens += response.total_tokens
            attempt = {
                "attempt": attempt_index + 1,
                "status": "response_received",
                "elapsed_seconds": time.monotonic() - attempt_started,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "total_tokens": response.total_tokens,
                "response_id": response.response_id,
                "finish_reason": response.finish_reason,
            }
            (page_dir / f"attempt_{attempt_index + 1:02d}_response.txt").write_text(
                raw_text,
                encoding="utf-8",
            )
            parsed_payload = parse_json_payload(raw_text)
            vlm_json_to_pagexml(
                parsed_payload,
                template_page=gt_page,
                output_path=page_dir / "prediction.xml",
            )
            _write_json_atomic(page_dir / "normalized_response.json", parsed_payload)
            attempt["status"] = "success"
            attempts.append(attempt)
            final_status = "success"
            final_error = None
            break
        except Exception as exc:
            final_status = _classify_exception(exc)
            final_error = f"{type(exc).__name__}: {exc}"
            attempts.append(
                {
                    "attempt": attempt_index + 1,
                    "status": final_status,
                    "elapsed_seconds": time.monotonic() - attempt_started,
                    "error": final_error,
                }
            )
            if not _is_retryable_exception(exc):
                break
        finally:
            _write_json_atomic(page_dir / f"attempt_{attempt_index + 1:02d}.json", attempts[-1])

    if final_status != "success":
        write_pagexml(empty_page_like(gt_page), page_dir / "prediction.xml", lines=())

    result = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "request_fingerprint": request["request_fingerprint"],
        "manuscript_id": paths.manuscript_id,
        "page_id": page_id,
        "provider_id": spec.provider_id,
        "method_id": spec.method_id,
        "model_id": spec.model_id,
        "prompt_sha256": request["prompt_sha256"],
        "image_sha256": request["image_sha256"],
        "pagexml_sha256": request["pagexml_sha256"],
        "status": final_status,
        "error": final_error,
        "attempt_count": len(attempts),
        "attempts": attempts,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "elapsed_seconds": time.monotonic() - acquisition_started,
        "completed_at_utc": _utc_now(),
        "prediction_path": "prediction.xml",
        "output_adapter_version": OUTPUT_ADAPTER_VERSION,
    }
    _write_json_atomic(result_path, result)
    return result


def acquire_manuscript_provider(
    *,
    manuscript_root: str | Path,
    cache_root: str | Path,
    provider_id: str,
    env_path: str | Path,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    page_workers: int = DEFAULT_PAGE_WORKERS,
    request_spacing_seconds: float = DEFAULT_REQUEST_SPACING_SECONDS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_base_delay_seconds: float = DEFAULT_RETRY_BASE_DELAY_SECONDS,
    invoke: Callable[..., VlmProviderResponse] = invoke_provider,
) -> dict:
    paths = default_manuscript_paths(manuscript_root)
    spec = provider_by_id(provider_id)
    page_ids = discover_page_ids(paths)
    if not page_ids:
        raise VlmCacheError(f"No aligned image/PAGE-XML pages found in {paths.root}.")
    load_dotenv(Path(env_path))
    api_key = (os.getenv(spec.api_key_env) or "").strip()
    if not api_key:
        raise VlmCacheError(f"{spec.api_key_env} is missing; expected it in {env_path}.")
    if invoke is invoke_provider:
        validate_provider_runtime(spec)

    output = _cache_dir(Path(cache_root), paths.manuscript_id, spec.method_id)
    output.mkdir(parents=True, exist_ok=True)
    for page_id in page_ids:
        page_dir = output / "pages" / page_id
        result_path = page_dir / "result.json"
        expected_request = _request_payload(paths=paths, page_id=page_id, spec=spec)
        if result_path.exists():
            _validate_terminal_result(
                json.loads(result_path.read_text(encoding="utf-8")),
                expected_request,
                result_path,
            )
        elif page_dir.exists():
            raise VlmCacheError(
                f"Ambiguous non-terminal paid acquisition at {page_dir}. "
                "No new requests were started; inspect it manually and use a new cache root."
            )
    rate_limiter = _RateLimiter(request_spacing_seconds)
    results: dict[str, dict] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(page_workers))) as executor:
        futures = {
            executor.submit(
                _acquire_page,
                paths=paths,
                page_id=page_id,
                spec=spec,
                cache_dir=output,
                api_key=api_key,
                timeout_seconds=timeout_seconds,
                max_retries=max(0, int(max_retries)),
                retry_base_delay_seconds=max(0.0, float(retry_base_delay_seconds)),
                rate_limiter=rate_limiter,
                invoke=invoke,
            ): page_id
            for page_id in page_ids
        }
        for future in concurrent.futures.as_completed(futures):
            page_id = futures[future]
            results[page_id] = future.result()

    manifest = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "manuscript_id": paths.manuscript_id,
        "provider": asdict(spec),
        "prompt_sha256": _prompt_sha256(),
        "page_ids": list(page_ids),
        "page_count": len(page_ids),
        "success_count": sum(item["status"] == "success" for item in results.values()),
        "failure_count": sum(item["status"] != "success" for item in results.values()),
        "max_retries_after_initial_attempt": max(0, int(max_retries)),
        "pages": {
            page_id: {
                "status": results[page_id]["status"],
                "result_path": f"pages/{page_id}/result.json",
                "prediction_path": f"pages/{page_id}/prediction.xml",
            }
            for page_id in page_ids
        },
        "written_at_utc": _utc_now(),
    }
    _write_json_atomic(output / "manifest.json", manifest)
    return manifest


def validate_vlm_cache(
    *,
    paths: ManuscriptPaths,
    cache_root: str | Path,
    method_id: str,
) -> dict:
    spec = provider_by_method_id(method_id)
    output = _cache_dir(Path(cache_root), paths.manuscript_id, method_id)
    manifest_path = output / "manifest.json"
    if not manifest_path.exists():
        raise VlmCacheError(f"Missing VLM pre-prediction manifest: {manifest_path}.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    page_ids = discover_page_ids(paths)
    if tuple(manifest.get("page_ids", ())) != page_ids:
        raise VlmCacheError(
            f"VLM cache page set differs from the current manuscript page set: {manifest_path}."
        )
    provider = manifest.get("provider", {})
    if provider.get("provider_id") != spec.provider_id or provider.get("model_id") != spec.model_id:
        raise VlmCacheError(f"VLM cache provider/model mismatch: {manifest_path}.")
    if manifest.get("prompt_sha256") != _prompt_sha256():
        raise VlmCacheError(f"VLM cache prompt mismatch: {manifest_path}.")

    for page_id in page_ids:
        result_path = output / "pages" / page_id / "result.json"
        if not result_path.exists():
            raise VlmCacheError(f"Missing terminal VLM result: {result_path}.")
        expected_request = _request_payload(paths=paths, page_id=page_id, spec=spec)
        result = json.loads(result_path.read_text(encoding="utf-8"))
        _validate_terminal_result(result, expected_request, result_path)
    return manifest


def materialize_cached_fold(
    *,
    paths: ManuscriptPaths,
    fold: Fold,
    cache_root: str | Path,
    method_id: str,
    output_dir: Path,
) -> tuple[Path, dict[str, str], dict]:
    manifest = validate_vlm_cache(paths=paths, cache_root=cache_root, method_id=method_id)
    source = _cache_dir(Path(cache_root), paths.manuscript_id, method_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    statuses: dict[str, str] = {}
    for page_id in fold.test_page_ids:
        page_dir = source / "pages" / page_id
        result = json.loads((page_dir / "result.json").read_text(encoding="utf-8"))
        shutil.copy2(page_dir / "prediction.xml", output_dir / f"{page_id}.xml")
        statuses[page_id] = str(result["status"])
    cache_metadata = {
        "cache_manifest_path": str((source / "manifest.json").resolve()),
        "cache_root": str(Path(cache_root).resolve()),
        "provider": manifest["provider"],
        "prompt_sha256": manifest["prompt_sha256"],
        "page_count": manifest["page_count"],
    }
    return output_dir, statuses, cache_metadata


def acquire_vlm_predictions(
    *,
    manuscript_roots: Iterable[str | Path],
    cache_root: str | Path,
    provider_ids: Iterable[str],
    env_path: str | Path,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    page_workers: int = DEFAULT_PAGE_WORKERS,
    request_spacing_seconds: float = DEFAULT_REQUEST_SPACING_SECONDS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_base_delay_seconds: float = DEFAULT_RETRY_BASE_DELAY_SECONDS,
) -> list[dict]:
    manuscript_root_list = tuple(manuscript_roots)
    provider_id_list = tuple(provider_ids)
    manifests = []
    for provider_id in provider_id_list:
        for manuscript_root in manuscript_root_list:
            manifests.append(
                acquire_manuscript_provider(
                    manuscript_root=manuscript_root,
                    cache_root=cache_root,
                    provider_id=provider_id,
                    env_path=env_path,
                    timeout_seconds=timeout_seconds,
                    page_workers=page_workers,
                    request_spacing_seconds=request_spacing_seconds,
                    max_retries=max_retries,
                    retry_base_delay_seconds=retry_base_delay_seconds,
                )
            )
    return manifests
