#!/usr/bin/env python3
"""Script to run linting checks using flake8."""

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
    """Run linting checks on the codebase."""
    print("[LINT] Running linting checks...")

    # Get the project root directory
    project_root = Path(__file__).parent.parent

    # Run flake8 for linting on backend directory only
    backend_dir = project_root / "backend"
    flake8_success = run_command(
        ["uv", "run", "flake8", str(backend_dir)], "linting with flake8"
    )

    if flake8_success:
        print("[SUCCESS] All linting checks passed!")
        return 0
    else:
        print("[ERROR] Linting checks failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
