from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
HOOK_PATH = REPO_ROOT / ".githooks" / "pre-commit"


def main() -> int:
    subprocess.run(["git", "config", "core.hooksPath", ".githooks"], cwd=REPO_ROOT, check=True)
    if HOOK_PATH.exists():
        HOOK_PATH.chmod(HOOK_PATH.stat().st_mode | 0o111)
    print("Configured git hooks for this repository: core.hooksPath=.githooks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
