from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Iterable


TRACKED_DISTRIBUTIONS: tuple[str, ...] = (
    "numpy",
    "opencv-python",
    "opencv-contrib-python",
    "shapely",
    "networkx",
    "Pillow",
    "python-dotenv",
    "google-genai",
    "sarvamai",
    "torch",
    "torchvision",
    "rapidfuzz",
    "python-Levenshtein",
    "pylatexenc",
    "beautifulsoup4",
)

TRACKED_IMPORTS: tuple[str, ...] = (
    "cv2",
    "numpy",
    "shapely",
    "networkx",
    "PIL",
    "dotenv",
    "google.genai",
    "sarvamai",
    "torch",
    "rapidfuzz",
    "Levenshtein",
    "pylatexenc",
    "bs4",
)


def _run_git(repo_root: Path, args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _distribution_snapshot(name: str) -> dict:
    try:
        dist = metadata.distribution(name)
    except metadata.PackageNotFoundError:
        return {"name": name, "installed": False}

    snapshot: dict = {
        "name": name,
        "installed": True,
        "version": dist.version,
    }
    direct_url = dist.read_text("direct_url.json")
    if direct_url:
        try:
            snapshot["direct_url"] = json.loads(direct_url)
        except json.JSONDecodeError:
            snapshot["direct_url_raw"] = direct_url
    return snapshot


def _module_snapshot(module_name: str) -> dict:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return {
            "module": module_name,
            "importable": False,
            "error": exc.__class__.__name__,
        }
    version = getattr(module, "__version__", None)
    snapshot = {
        "module": module_name,
        "importable": True,
    }
    if version is not None:
        snapshot["version"] = str(version)
    module_file = getattr(module, "__file__", None)
    if module_file:
        snapshot["file"] = str(module_file)
    return snapshot


def _file_snapshot(path: Path, repo_root: Path) -> dict:
    resolved = path.resolve()
    try:
        relative = str(resolved.relative_to(repo_root))
    except ValueError:
        relative = str(resolved)

    if not resolved.exists():
        return {
            "path": relative,
            "exists": False,
        }

    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": relative,
        "exists": True,
        "size_bytes": resolved.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def collect_reproducibility_manifest(
    *,
    repo_root: str | Path,
    artifact_paths: Iterable[str | Path] = (),
) -> dict:
    root = Path(repo_root).resolve()
    status_short = _run_git(root, ["status", "--short"])
    git_snapshot = {
        "repo_root": str(root),
        "top_level": _run_git(root, ["rev-parse", "--show-toplevel"]),
        "branch": _run_git(root, ["rev-parse", "--abbrev-ref", "HEAD"]),
        "commit": _run_git(root, ["rev-parse", "HEAD"]),
        "commit_date": _run_git(root, ["log", "-1", "--format=%cI"]),
        "dirty": bool(status_short),
        "status_short": status_short.splitlines() if status_short else [],
    }
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "platform": platform.platform(),
            "implementation": platform.python_implementation(),
        },
        "environment": {
            "CONDA_DEFAULT_ENV": os.getenv("CONDA_DEFAULT_ENV"),
            "VIRTUAL_ENV": os.getenv("VIRTUAL_ENV"),
        },
        "git": git_snapshot,
        "distributions": [_distribution_snapshot(name) for name in TRACKED_DISTRIBUTIONS],
        "imports": [_module_snapshot(name) for name in TRACKED_IMPORTS],
        "artifacts": [_file_snapshot(Path(path), root) for path in artifact_paths],
    }


def write_reproducibility_manifest(
    output_root: str | Path,
    *,
    repo_root: str | Path,
    artifact_paths: Iterable[str | Path] = (),
) -> Path:
    output_path = Path(output_root) / "reproducibility.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = collect_reproducibility_manifest(repo_root=repo_root, artifact_paths=artifact_paths)
    output_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path
