from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from experiments.downstream_ocr.adapter import (
    AdapterError,
    VLM_END_TO_END_PROMPT,
    parse_json_payload,
)
from experiments.downstream_ocr.pagexml import PageXmlPage, load_pagexml, write_pagexml
from experiments.downstream_ocr.runners import (
    _run_methods_with_finetuning_ladder,
    method_by_id,
    run_methods_experiment,
)
from experiments.downstream_ocr.splits import Fold, default_manuscript_paths
from experiments.downstream_ocr.vlm_cache import (
    VlmCacheError,
    acquire_manuscript_provider,
    acquire_vlm_predictions,
    materialize_cached_fold,
    validate_vlm_cache,
)
from experiments.downstream_ocr.vlm_providers import (
    VLM_PROVIDER_SPECS,
    VlmProviderResponse,
    _INVOKERS,
    _PROVIDER_IMPORT_NAMES,
    _invoke_claude,
)


def _make_manuscript(root: Path, page_ids: tuple[str, ...] = ("p1",)) -> Path:
    manuscript = root / "manuscript"
    images = manuscript / "images_resized"
    pagexml = manuscript / "layout_analysis_output" / "page-xml-format"
    images.mkdir(parents=True)
    pagexml.mkdir(parents=True)
    for page_id in page_ids:
        (images / f"{page_id}.jpg").write_bytes(f"image:{page_id}".encode("ascii"))
        write_pagexml(
            PageXmlPage(
                page_id=page_id,
                image_filename=f"{page_id}.jpg",
                width=100,
                height=80,
                lines=(),
            ),
            pagexml / f"{page_id}.xml",
            lines=(),
        )
    return manuscript


class VlmCacheTests(unittest.TestCase):
    def test_json_parser_accepts_only_a_whole_response_code_fence(self):
        fenced = """```json
{"status": "success", "regions": []}
```"""
        self.assertEqual(parse_json_payload(fenced)["status"], "success")
        with self.assertRaises(AdapterError):
            parse_json_payload(f"Here is the result:\n{fenced}")

    def test_cached_fenced_json_failure_is_repaired_without_an_api_call(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manuscript = _make_manuscript(root)
            cache_root = root / "cache"
            fenced = """```json
{"status": "success", "regions": []}
```"""

            def fenced_invoke(*args, **kwargs):
                return VlmProviderResponse(raw_text=fenced)

            with patch.dict("os.environ", {"CLAUDE_API_KEY": "test-key"}):
                with patch(
                    "experiments.downstream_ocr.vlm_cache.parse_json_payload",
                    side_effect=AdapterError("json_parse_error"),
                ):
                    failed = acquire_manuscript_provider(
                        manuscript_root=manuscript,
                        cache_root=cache_root,
                        provider_id="claude",
                        env_path=root / ".env",
                        page_workers=1,
                        request_spacing_seconds=0,
                        retry_base_delay_seconds=0,
                        max_retries=0,
                        invoke=fenced_invoke,
                    )
                repaired = acquire_manuscript_provider(
                    manuscript_root=manuscript,
                    cache_root=cache_root,
                    provider_id="claude",
                    env_path=root / ".env",
                    page_workers=1,
                    request_spacing_seconds=0,
                    invoke=lambda *args, **kwargs: self.fail(
                        "Local fence repair must not invoke the provider."
                    ),
                )

            self.assertEqual(failed["failure_count"], 1)
            self.assertEqual(repaired["success_count"], 1)
            result = json.loads(
                (
                    cache_root
                    / "manuscript"
                    / "claude_e2e"
                    / "pages"
                    / "p1"
                    / "result.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(result["status"], "success")
            self.assertFalse(result["local_output_repairs"][0]["paid_api_call_made"])

    def test_claude_sonnet_5_request_omits_deprecated_sampling_parameters(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "page.jpg"
            image_path.write_bytes(b"image")
            captured = {}

            def create_message(**kwargs):
                captured.update(kwargs)
                return SimpleNamespace(
                    id="msg_1",
                    content=[
                        SimpleNamespace(
                            type="text",
                            text=json.dumps({"status": "success", "regions": []}),
                        )
                    ],
                    usage=SimpleNamespace(input_tokens=10, output_tokens=4),
                    stop_reason="end_turn",
                )

            fake_client = SimpleNamespace(
                messages=SimpleNamespace(create=create_message)
            )
            claude_spec = next(
                spec for spec in VLM_PROVIDER_SPECS if spec.provider_id == "claude"
            )
            with patch("anthropic.Anthropic", return_value=fake_client) as client_cls:
                response = _invoke_claude(
                    claude_spec,
                    api_key="test-key",
                    image_path=image_path,
                    prompt=VLM_END_TO_END_PROMPT,
                    timeout_seconds=45.0,
                )

            self.assertNotIn("temperature", captured)
            self.assertNotIn("top_p", captured)
            self.assertNotIn("top_k", captured)
            self.assertEqual(captured["model"], "claude-sonnet-5")
            self.assertEqual(
                captured["messages"][0]["content"][1]["text"],
                VLM_END_TO_END_PROMPT,
            )
            self.assertEqual(response.total_tokens, 14)
            client_cls.assert_called_once_with(
                api_key="test-key",
                timeout=45.0,
                max_retries=0,
            )

    def test_nonretryable_provider_400_stops_after_one_attempt(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manuscript = _make_manuscript(root)
            call_count = 0

            class BadRequest(Exception):
                status_code = 400

            def failing_invoke(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                raise BadRequest("invalid_request_error")

            with patch.dict("os.environ", {"CLAUDE_API_KEY": "test-key"}):
                manifest = acquire_manuscript_provider(
                    manuscript_root=manuscript,
                    cache_root=root / "cache",
                    provider_id="claude",
                    env_path=root / ".env",
                    page_workers=1,
                    request_spacing_seconds=0,
                    retry_base_delay_seconds=0,
                    max_retries=3,
                    invoke=failing_invoke,
                )

            self.assertEqual(call_count, 1)
            self.assertEqual(manifest["failure_count"], 1)

    def test_provider_registry_pins_requested_models_and_one_prompt(self):
        by_provider = {spec.provider_id: spec for spec in VLM_PROVIDER_SPECS}
        self.assertEqual(by_provider["openai"].model_id, "gpt-5.6-terra")
        self.assertEqual(by_provider["claude"].model_id, "claude-sonnet-5")
        self.assertEqual(by_provider["gemini"].model_id, "gemini-3.5-flash")
        self.assertNotIn("deepseek", by_provider)
        self.assertNotIn("sarvam", by_provider)
        self.assertIn("Output ONLY raw valid JSON", VLM_END_TO_END_PROMPT)

    def test_provider_registry_has_complete_unique_dispatch(self):
        provider_ids = [spec.provider_id for spec in VLM_PROVIDER_SPECS]
        method_ids = [spec.method_id for spec in VLM_PROVIDER_SPECS]
        self.assertEqual(len(provider_ids), len(set(provider_ids)))
        self.assertEqual(len(method_ids), len(set(method_ids)))
        self.assertEqual(set(provider_ids), set(_PROVIDER_IMPORT_NAMES))
        self.assertEqual(set(provider_ids), set(_INVOKERS))

    def test_multi_provider_acquisition_reuses_generator_manuscript_roots(self):
        calls = []

        def fake_acquire(**kwargs):
            calls.append((kwargs["provider_id"], kwargs["manuscript_root"]))
            return {
                "provider": {"provider_id": kwargs["provider_id"]},
                "manuscript_id": str(kwargs["manuscript_root"]),
            }

        with patch(
            "experiments.downstream_ocr.vlm_cache.acquire_manuscript_provider",
            side_effect=fake_acquire,
        ):
            manifests = acquire_vlm_predictions(
                manuscript_roots=(root for root in ("m1", "m2")),
                cache_root=Path("cache"),
                provider_ids=(provider for provider in ("gemini", "openai")),
                env_path=Path(".env"),
            )

        self.assertEqual(
            calls,
            [
                ("gemini", "m1"),
                ("gemini", "m2"),
                ("openai", "m1"),
                ("openai", "m2"),
            ],
        )
        self.assertEqual(len(manifests), 4)

    def test_run_methods_preflights_vlm_cache_before_creating_fold_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manuscript = _make_manuscript(root)
            run_root = root / "run"
            with self.assertRaisesRegex(VlmCacheError, "Missing VLM pre-prediction manifest"):
                run_methods_experiment(
                    manuscript_root=manuscript,
                    output_root=run_root,
                    method_ids=("openai_e2e",),
                    vlm_predictions_root=root / "missing-cache",
                )
            self.assertFalse(run_root.exists())

    def test_method_ladder_forwards_vlm_cache_root_to_cached_method(self):
        cache_root = Path("cache-root")
        fold = Fold("fold_1", train_page_ids=(), test_page_ids=("p1",))
        paths = SimpleNamespace(manuscript_id="manuscript")
        method = method_by_id("gemini_e2e")

        with patch(
            "experiments.downstream_ocr.runners.run_method",
            return_value=(Path("predictions"), {"p1": "success"}),
        ) as run_method_mock, patch(
            "experiments.downstream_ocr.runners.evaluate_prediction_folder",
            return_value=[],
        ):
            _run_methods_with_finetuning_ladder(
                paths=paths,
                folds=(fold,),
                methods=(method,),
                output_root=Path("output"),
                vlm_predictions_root=cache_root,
            )

        run_method_mock.assert_called_once_with(
            paths=paths,
            fold=fold,
            method=method,
            run_dir=Path("output") / "runs" / "gemini_e2e" / "fold_1",
            vlm_predictions_root=cache_root,
        )

    def test_method_ladder_does_not_pass_vlm_cache_to_annotation_tool(self):
        cache_root = Path("cache-root")
        fold = Fold("fold_1", train_page_ids=(), test_page_ids=("p1",))
        paths = SimpleNamespace(manuscript_id="manuscript")
        method = method_by_id("annotation_tool_e2e")
        prepared_pages = {"p1": SimpleNamespace()}

        with patch(
            "experiments.downstream_ocr.runners._prepare_annotation_tool_predicted_layout_pages",
            return_value=prepared_pages,
        ), patch(
            "experiments.downstream_ocr.runners.run_annotation_tool_auto_layout",
            return_value=Path("predictions"),
        ) as annotation_run_mock, patch(
            "experiments.downstream_ocr.runners.evaluate_prediction_folder",
            return_value=[],
        ):
            _run_methods_with_finetuning_ladder(
                paths=paths,
                folds=(fold,),
                methods=(method,),
                output_root=Path("output"),
                vlm_predictions_root=cache_root,
            )

        annotation_run_mock.assert_called_once_with(
            paths=paths,
            fold=fold,
            method=method,
            run_dir=Path("output") / "runs" / "annotation_tool_e2e" / "fold_1",
            prepared_pages=prepared_pages,
        )

    def test_successful_page_is_acquired_once_and_reused_by_fold(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manuscript = _make_manuscript(root)
            cache_root = root / "cache"
            calls = []

            def fake_invoke(spec, **kwargs):
                calls.append((spec.provider_id, kwargs["prompt"], kwargs["image_path"].name))
                return VlmProviderResponse(
                    raw_text=json.dumps({"status": "success", "regions": []}),
                    input_tokens=10,
                    output_tokens=4,
                    total_tokens=14,
                )

            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
                first = acquire_manuscript_provider(
                    manuscript_root=manuscript,
                    cache_root=cache_root,
                    provider_id="openai",
                    env_path=root / ".env",
                    page_workers=1,
                    request_spacing_seconds=0,
                    max_retries=3,
                    invoke=fake_invoke,
                )
                second = acquire_manuscript_provider(
                    manuscript_root=manuscript,
                    cache_root=cache_root,
                    provider_id="openai",
                    env_path=root / ".env",
                    page_workers=1,
                    request_spacing_seconds=0,
                    max_retries=3,
                    invoke=fake_invoke,
                )

            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][1], VLM_END_TO_END_PROMPT)
            self.assertEqual(first["page_count"], 1)
            self.assertEqual(second["success_count"], 1)

            paths = default_manuscript_paths(manuscript)
            prediction_dir, statuses, metadata = materialize_cached_fold(
                paths=paths,
                fold=Fold("fold_1", train_page_ids=(), test_page_ids=("p1",)),
                cache_root=cache_root,
                method_id="openai_e2e",
                output_dir=root / "run" / "predictions",
            )
            self.assertEqual(statuses, {"p1": "success"})
            self.assertEqual(load_pagexml(prediction_dir / "p1.xml").page_id, "p1")
            self.assertEqual(metadata["provider"]["model_id"], "gpt-5.6-terra")

    def test_terminal_failure_is_not_paid_for_again(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manuscript = _make_manuscript(root)
            cache_root = root / "cache"
            call_count = 0

            def failing_invoke(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                raise TimeoutError("provider timeout")

            with patch.dict("os.environ", {"CLAUDE_API_KEY": "test-key"}):
                manifest = acquire_manuscript_provider(
                    manuscript_root=manuscript,
                    cache_root=cache_root,
                    provider_id="claude",
                    env_path=root / ".env",
                    page_workers=1,
                    request_spacing_seconds=0,
                    retry_base_delay_seconds=0,
                    max_retries=2,
                    invoke=failing_invoke,
                )
                acquire_manuscript_provider(
                    manuscript_root=manuscript,
                    cache_root=cache_root,
                    provider_id="claude",
                    env_path=root / ".env",
                    page_workers=1,
                    request_spacing_seconds=0,
                    retry_base_delay_seconds=0,
                    max_retries=2,
                    invoke=failing_invoke,
                )

            self.assertEqual(call_count, 3)
            self.assertEqual(manifest["failure_count"], 1)
            result = json.loads(
                (
                    cache_root
                    / "manuscript"
                    / "claude_e2e"
                    / "pages"
                    / "p1"
                    / "result.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(result["status"], "api_timeout")
            self.assertEqual(result["attempt_count"], 3)

    def test_changed_input_invalidates_cache_instead_of_reacquiring(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manuscript = _make_manuscript(root)
            cache_root = root / "cache"

            def fake_invoke(*args, **kwargs):
                return VlmProviderResponse(
                    raw_text=json.dumps({"status": "success", "regions": []})
                )

            with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
                acquire_manuscript_provider(
                    manuscript_root=manuscript,
                    cache_root=cache_root,
                    provider_id="gemini",
                    env_path=root / ".env",
                    page_workers=1,
                    request_spacing_seconds=0,
                    invoke=fake_invoke,
                )
            (manuscript / "images_resized" / "p1.jpg").write_bytes(b"changed-image")
            with self.assertRaisesRegex(VlmCacheError, "does not match"):
                validate_vlm_cache(
                    paths=default_manuscript_paths(manuscript),
                    cache_root=cache_root,
                    method_id="gemini_e2e",
                )

    def test_partial_page_directory_is_never_replayed_automatically(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manuscript = _make_manuscript(root)
            page_dir = root / "cache" / "manuscript" / "openai_e2e" / "pages" / "p1"
            page_dir.mkdir(parents=True)
            (page_dir / "request.json").write_text("{}", encoding="utf-8")

            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
                with self.assertRaisesRegex(VlmCacheError, "Ambiguous non-terminal"):
                    acquire_manuscript_provider(
                        manuscript_root=manuscript,
                        cache_root=root / "cache",
                        provider_id="openai",
                        env_path=root / ".env",
                        page_workers=1,
                        request_spacing_seconds=0,
                        invoke=lambda *args, **kwargs: self.fail(
                            "A partial cache must fail before provider invocation."
                        ),
                    )


if __name__ == "__main__":
    unittest.main()
