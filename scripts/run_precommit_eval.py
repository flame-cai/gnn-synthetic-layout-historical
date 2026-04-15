from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = REPO_ROOT / "app"
TEST_COMMAND = ["-m", "unittest", "discover", "-s", "tests", "-p", "test_ci_e2e.py", "-v"]


def env_python_name() -> str:
    return "python.exe" if os.name == "nt" else "bin/python"


def candidate_python_paths() -> list[Path]:
    candidates: list[Path] = []

    configured = os.environ.get("GNN_LAYOUT_PYTHON")
    if configured:
        candidates.append(Path(configured).expanduser())

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        prefix_path = Path(conda_prefix)
        if prefix_path.name == "gnn_layout":
            candidates.append(prefix_path / env_python_name())

    current_python = Path(sys.executable)
    if current_python.exists():
        parent_name = current_python.parent.name
        grandparent_name = current_python.parent.parent.name if current_python.parent.parent else ""
        if "gnn_layout" in {parent_name, grandparent_name}:
            candidates.append(current_python)

    home = Path.home()
    if os.name == "nt":
        bases = [
            home / "miniconda3",
            home / "anaconda3",
            home / "mambaforge",
            home / "miniforge3",
            home / "micromamba",
        ]
    else:
        bases = [
            home / "miniconda3",
            home / "anaconda3",
            home / "mambaforge",
            home / "miniforge3",
            home / "micromamba",
            home / ".conda",
        ]

    for base in bases:
        candidates.append(base / "envs" / "gnn_layout" / env_python_name())

    seen: set[str] = set()
    unique_candidates: list[Path] = []
    for candidate in candidates:
        key = str(candidate).lower()
        if key not in seen:
            seen.add(key)
            unique_candidates.append(candidate)
    return unique_candidates


def resolve_python_command() -> list[str]:
    for candidate in candidate_python_paths():
        if candidate.exists():
            return [str(candidate)]

    conda = shutil.which("conda")
    if conda:
        return [conda, "run", "--no-capture-output", "-n", "gnn_layout", "python"]

    return []


def main() -> int:
    command_prefix = resolve_python_command()
    if not command_prefix:
        print("[pre-commit] Could not find the gnn_layout Python interpreter.", file=sys.stderr)
        print(
            "[pre-commit] Set GNN_LAYOUT_PYTHON to the environment's python executable or make 'conda run -n gnn_layout' available.",
            file=sys.stderr,
        )
        return 1

    full_command = command_prefix + TEST_COMMAND
    print(f"[pre-commit] Running {' '.join(full_command)}", flush=True)
    result = subprocess.run(full_command, cwd=APP_DIR)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
