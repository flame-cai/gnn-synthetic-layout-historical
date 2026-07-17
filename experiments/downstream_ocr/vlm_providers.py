from __future__ import annotations

import base64
import importlib.util
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .adapter import VLM_END_TO_END_PROMPT


@dataclass(frozen=True)
class VlmProviderSpec:
    provider_id: str
    method_id: str
    display_name: str
    model_id: str
    api_key_env: str
    request_contract_version: int = 1


@dataclass(frozen=True)
class VlmProviderResponse:
    raw_text: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    response_id: str | None = None
    finish_reason: str | None = None


VLM_PROVIDER_SPECS: tuple[VlmProviderSpec, ...] = (
    VlmProviderSpec(
        provider_id="gemini",
        method_id="gemini_e2e",
        display_name="Gemini (End-to-End)",
        model_id="gemini-3.5-flash",
        api_key_env="GEMINI_API_KEY",
    ),
    VlmProviderSpec(
        provider_id="openai",
        method_id="openai_e2e",
        display_name="OpenAI (End-to-End)",
        model_id="gpt-5.6-terra",
        api_key_env="OPENAI_API_KEY",
    ),
    VlmProviderSpec(
        provider_id="claude",
        method_id="claude_e2e",
        display_name="Claude (End-to-End)",
        model_id="claude-sonnet-5",
        api_key_env="CLAUDE_API_KEY",
        request_contract_version=2,
    ),
)

_BY_PROVIDER_ID = {spec.provider_id: spec for spec in VLM_PROVIDER_SPECS}
_BY_METHOD_ID = {spec.method_id: spec for spec in VLM_PROVIDER_SPECS}
_PROVIDER_IMPORT_NAMES = {
    "gemini": "google.genai",
    "openai": "openai",
    "claude": "anthropic",
}

if len(_BY_PROVIDER_ID) != len(VLM_PROVIDER_SPECS):
    raise RuntimeError("VLM provider_id values must be unique.")
if len(_BY_METHOD_ID) != len(VLM_PROVIDER_SPECS):
    raise RuntimeError("VLM method_id values must be unique.")


def provider_by_id(provider_id: str) -> VlmProviderSpec:
    try:
        return _BY_PROVIDER_ID[provider_id]
    except KeyError as exc:
        supported = ", ".join(sorted(_BY_PROVIDER_ID))
        raise ValueError(f"Unsupported VLM provider {provider_id!r}. Supported: {supported}.") from exc


def provider_by_method_id(method_id: str) -> VlmProviderSpec:
    try:
        return _BY_METHOD_ID[method_id]
    except KeyError as exc:
        raise ValueError(f"Method {method_id!r} is not a cached VLM provider method.") from exc


def is_vlm_method(method_id: str) -> bool:
    return method_id in _BY_METHOD_ID


def validate_provider_runtime(spec: VlmProviderSpec) -> None:
    import_name = _PROVIDER_IMPORT_NAMES[spec.provider_id]
    try:
        available = importlib.util.find_spec(import_name) is not None
    except ModuleNotFoundError:
        available = False
    if not available:
        raise RuntimeError(
            f"Provider {spec.provider_id!r} requires the {import_name!r} package. "
            "Install the repository requirements before starting paid acquisition."
        )


def _image_media_type(image_path: Path) -> str:
    media_type, _ = mimetypes.guess_type(image_path.name)
    return media_type or "image/jpeg"


def _base64_image(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("ascii")


def _value(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _invoke_gemini(
    spec: VlmProviderSpec,
    *,
    api_key: str,
    image_path: Path,
    prompt: str,
    timeout_seconds: float,
) -> VlmProviderResponse:
    from google import genai
    from google.genai import types

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=int(timeout_seconds * 1000),
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )
    response = client.models.generate_content(
        model=spec.model_id,
        contents=[
            types.Part.from_bytes(
                data=image_path.read_bytes(),
                mime_type=_image_media_type(image_path),
            ),
            prompt,
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.2,
        ),
    )
    usage = _value(response, "usage_metadata")
    input_tokens = int(_value(usage, "prompt_token_count", 0) or 0)
    output_tokens = int(_value(usage, "candidates_token_count", 0) or 0)
    total_tokens = int(_value(usage, "total_token_count", input_tokens + output_tokens) or 0)
    return VlmProviderResponse(
        raw_text=str(_value(response, "text", "") or ""),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        response_id=_value(response, "response_id"),
    )


def _invoke_openai(
    spec: VlmProviderSpec,
    *,
    api_key: str,
    image_path: Path,
    prompt: str,
    timeout_seconds: float,
) -> VlmProviderResponse:
    from openai import OpenAI

    media_type = _image_media_type(image_path)
    image_url = f"data:{media_type};base64,{_base64_image(image_path)}"
    client = OpenAI(api_key=api_key, timeout=timeout_seconds, max_retries=0)
    response = client.responses.create(
        model=spec.model_id,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": image_url},
                    {"type": "input_text", "text": prompt},
                ],
            }
        ],
        text={"format": {"type": "json_object"}},
        max_output_tokens=16384,
    )
    usage = _value(response, "usage")
    input_tokens = int(_value(usage, "input_tokens", 0) or 0)
    output_tokens = int(_value(usage, "output_tokens", 0) or 0)
    total_tokens = int(_value(usage, "total_tokens", input_tokens + output_tokens) or 0)
    return VlmProviderResponse(
        raw_text=str(_value(response, "output_text", "") or ""),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        response_id=_value(response, "id"),
        finish_reason=_value(response, "status"),
    )


def _invoke_claude(
    spec: VlmProviderSpec,
    *,
    api_key: str,
    image_path: Path,
    prompt: str,
    timeout_seconds: float,
) -> VlmProviderResponse:
    from anthropic import Anthropic

    client = Anthropic(api_key=api_key, timeout=timeout_seconds, max_retries=0)
    response = client.messages.create(
        model=spec.model_id,
        max_tokens=16384,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": _image_media_type(image_path),
                            "data": _base64_image(image_path),
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    raw_text = "".join(
        str(_value(block, "text", "") or "")
        for block in (_value(response, "content", ()) or ())
        if _value(block, "type") == "text"
    )
    usage = _value(response, "usage")
    input_tokens = int(_value(usage, "input_tokens", 0) or 0)
    output_tokens = int(_value(usage, "output_tokens", 0) or 0)
    return VlmProviderResponse(
        raw_text=raw_text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        response_id=_value(response, "id"),
        finish_reason=_value(response, "stop_reason"),
    )


_INVOKERS: dict[str, Callable[..., VlmProviderResponse]] = {
    "gemini": _invoke_gemini,
    "openai": _invoke_openai,
    "claude": _invoke_claude,
}

_REGISTERED_PROVIDER_IDS = set(_BY_PROVIDER_ID)
if set(_PROVIDER_IMPORT_NAMES) != _REGISTERED_PROVIDER_IDS:
    raise RuntimeError(
        "Every VLM provider must have exactly one runtime dependency import."
    )
if set(_INVOKERS) != _REGISTERED_PROVIDER_IDS:
    raise RuntimeError("Every VLM provider must have exactly one API invoker.")


def invoke_provider(
    spec: VlmProviderSpec,
    *,
    api_key: str,
    image_path: Path,
    prompt: str = VLM_END_TO_END_PROMPT,
    timeout_seconds: float = 45.0,
) -> VlmProviderResponse:
    if prompt != VLM_END_TO_END_PROMPT:
        raise ValueError("VLM acquisitions must use the exact shared end-to-end prompt.")
    return _INVOKERS[spec.provider_id](
        spec,
        api_key=api_key,
        image_path=image_path,
        prompt=prompt,
        timeout_seconds=timeout_seconds,
    )
