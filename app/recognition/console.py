from __future__ import annotations

import os
import sys


def configure_recognition_console_streams() -> None:
    """Keep recognition CLI output safe for Windows `conda run` capture."""
    if os.environ.get("OCR_ALLOW_UNICODE_CONSOLE") == "1":
        encoding = "utf-8"
    elif os.name == "nt":
        encoding = "ascii"
    else:
        encoding = "utf-8"

    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding=encoding, errors="backslashreplace")
