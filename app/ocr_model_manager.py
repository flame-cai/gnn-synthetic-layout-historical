from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import torch

from recognition.recognize_manuscript_text_v2_pretrained import (
    get_model_config,
    load_ocr_model,
    process_page_xml,
)


@dataclass
class OcrLoadedContext:
    checkpoint_path: str
    model: object
    converter: object
    config: object
    device: object


class ManuscriptAwareOcrModelManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._active_context: OcrLoadedContext | None = None

    def clear(self) -> None:
        with self._lock:
            self._active_context = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def get_context(self, checkpoint_path: str | Path) -> dict:
        checkpoint_path = str(Path(checkpoint_path).resolve())
        with self._lock:
            if self._active_context and self._active_context.checkpoint_path == checkpoint_path:
                return self._active_context.__dict__

            config = get_model_config(checkpoint_path)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model, converter = load_ocr_model(config, device)
            self._active_context = OcrLoadedContext(
                checkpoint_path=checkpoint_path,
                model=model,
                converter=converter,
                config=config,
                device=device,
            )
            return self._active_context.__dict__

    def recognize_page(self, manuscript_root: str | Path, page_id: str, checkpoint_path: str | Path) -> None:
        manuscript_root = Path(manuscript_root)
        xml_path = manuscript_root / "layout_analysis_output" / "page-xml-format" / f"{page_id}.xml"
        image_dirs = [str(manuscript_root / "images"), str(manuscript_root / "images_resized")]
        context = self.get_context(checkpoint_path)
        process_page_xml(
            str(xml_path),
            image_dirs,
            context["model"],
            context["converter"],
            context["config"],
            context["device"],
        )
