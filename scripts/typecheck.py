#!/usr/bin/env python3
"""Script to run type checking using mypy."""

import subprocess
import sys
from pathlib import Path


def run_command(command: list[str], description: str) -> bool:
    """Run a command and return True if successful."""
    print(f"Running {description}...")
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"[SUCCESS] {description} passed")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {description} failed")
        if e.stdout:
            print(e.stdout)
        if e.stderr:
            print(e.stderr)
        return False


def main():
    """Run type checking on the codebase."""
    print("[TYPECHECK] Running type checks...")

    # Get the project root directory
    project_root = Path(__file__).parent.parent
    backend_dir = project_root / "backend"

    # Run mypy for type checking on backend directory
    mypy_success = run_command(
        ["uv", "run", "mypy", str(backend_dir)], "type checking with mypy"
    )

    if mypy_success:
        print("[SUCCESS] All type checks passed!")
        return 0
    else:
        print("[ERROR] Type checking failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
