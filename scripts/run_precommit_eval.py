from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = REPO_ROOT / "app"
LOGS_DIR = APP_DIR / "tests" / "logs"


@dataclass(frozen=True)
class PrecommitPhase:
    name: str
    command: list[str]
    skip_env_var: str
    artifact_paths: tuple[Path, ...]


PIPELINE_PHASE = PrecommitPhase(
    name="Full Pipeline Gate",
    command=["-m", "unittest", "discover", "-s", "tests", "-p", "test_ci_e2e.py", "-v"],
    skip_env_var="SKIP_PIPELINE_EVAL_HOOK",
    artifact_paths=(
        LOGS_DIR / "ci_eval_results_latest.txt",
        LOGS_DIR / "ci_eval_results_latest.json",
    ),
)


RECOGNITION_PHASE = PrecommitPhase(
    name="Recognition Fine-Tune Gate",
    command=["-m", "unittest", "tests.test_recognition_finetuning_precommit_e2e", "-v"],
    skip_env_var="SKIP_RECOGNITION_FT_HOOK",
    artifact_paths=(
        LOGS_DIR / "recognition_finetune_precommit_latest.md",
        LOGS_DIR / "recognition_finetune_precommit_latest.json",
        LOGS_DIR / "recognition_finetune_precommit_latest.txt",
    ),
)


PHASES = (PIPELINE_PHASE, RECOGNITION_PHASE)


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


def _phase_artifact_lines(phase: PrecommitPhase) -> list[str]:
    return [f"[pre-commit] Artifact: {artifact}" for artifact in phase.artifact_paths]


def _run_phase(command_prefix: list[str], phase: PrecommitPhase) -> int:
    full_command = command_prefix + phase.command
    env = dict(os.environ)
    env.setdefault("CONDA_NO_PLUGINS", "true")

    print(f"[pre-commit] === {phase.name} ===", flush=True)
    print(f"[pre-commit] Running {' '.join(full_command)}", flush=True)
    result = subprocess.run(full_command, cwd=APP_DIR, env=env)
    if result.returncode == 0:
        for artifact_line in _phase_artifact_lines(phase):
            print(artifact_line, flush=True)
        return 0

    print(f"[pre-commit] {phase.name} failed with exit code {result.returncode}.", file=sys.stderr, flush=True)
    for artifact_line in _phase_artifact_lines(phase):
        print(artifact_line, file=sys.stderr, flush=True)
    return result.returncode


def main() -> int:
    if os.environ.get("SKIP_EVAL_HOOK") == "1":
        print("[pre-commit] SKIP_EVAL_HOOK=1, skipping all evaluation phases.", flush=True)
        return 0

    command_prefix = resolve_python_command()
    if not command_prefix:
        print("[pre-commit] Could not find the gnn_layout Python interpreter.", file=sys.stderr)
        print(
            "[pre-commit] Set GNN_LAYOUT_PYTHON to the environment's python executable or make 'conda run -n gnn_layout' available.",
            file=sys.stderr,
        )
        return 1

    ran_any_phase = False
    for phase in PHASES:
        if os.environ.get(phase.skip_env_var) == "1":
            print(f"[pre-commit] {phase.skip_env_var}=1, skipping {phase.name}.", flush=True)
            continue
        ran_any_phase = True
        phase_returncode = _run_phase(command_prefix, phase)
        if phase_returncode != 0:
            return phase_returncode

    if not ran_any_phase:
        print("[pre-commit] All evaluation phases were skipped.", flush=True)
    else:
        print("[pre-commit] All requested evaluation phases passed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
