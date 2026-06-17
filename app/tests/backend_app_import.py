from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


TESTS_ROOT = Path(__file__).resolve().parent
APP_ROOT = TESTS_ROOT.parent
REPO_ROOT = APP_ROOT.parent
BACKEND_APP_MODULE_NAME = "_cai_flame_backend_app_module"


def _ensure_path_order() -> None:
    for path in (str(REPO_ROOT), str(APP_ROOT)):
        while path in sys.path:
            sys.path.remove(path)
    sys.path.insert(0, str(APP_ROOT))
    sys.path.insert(0, str(REPO_ROOT))


def load_backend_app_module():
    existing = sys.modules.get(BACKEND_APP_MODULE_NAME)
    if existing is not None:
        return existing

    _ensure_path_order()
    app_path = APP_ROOT / "app.py"
    spec = importlib.util.spec_from_file_location(BACKEND_APP_MODULE_NAME, app_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load backend app module from {app_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[BACKEND_APP_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


backend_app_module = load_backend_app_module()
