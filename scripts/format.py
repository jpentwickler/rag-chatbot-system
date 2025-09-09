#!/usr/bin/env python3
"""Script to format Python code using black and isort."""

import subprocess
import sys
from pathlib import Path


def run_command(command: list[str], description: str) -> bool:
    """Run a command and return True if successful."""
    print(f"Running {description}...")
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"[SUCCESS] {description} completed successfully")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {description} failed")
        print(f"Error: {e.stderr}")
        return False


def main():
    """Format the codebase using black and isort."""
    print("[FORMAT] Formatting codebase...")

    # Get the project root directory
    project_root = Path(__file__).parent.parent

    # Run isort to organize imports
    isort_success = run_command(
        ["uv", "run", "isort", str(project_root)], "import sorting with isort"
    )

    # Run black to format code
    black_success = run_command(
        ["uv", "run", "black", str(project_root)], "code formatting with black"
    )

    if isort_success and black_success:
        print("[SUCCESS] All formatting completed successfully!")
        return 0
    else:
        print("[ERROR] Some formatting operations failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
