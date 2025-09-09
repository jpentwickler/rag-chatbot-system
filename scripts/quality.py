#!/usr/bin/env python3
"""Master script to run all code quality checks."""

import subprocess
import sys
from pathlib import Path


def run_script(script_name: str) -> bool:
    """Run a quality check script and return True if successful."""
    script_path = Path(__file__).parent / f"{script_name}.py"
    try:
        result = subprocess.run([sys.executable, str(script_path)], check=True)
        return True
    except subprocess.CalledProcessError:
        return False


def main():
    """Run all code quality checks in sequence."""
    print("[QUALITY] Running comprehensive code quality checks...\n")

    checks = [
        ("format", "Code formatting"),
        ("lint", "Linting"),
        ("typecheck", "Type checking"),
    ]

    failed_checks = []

    for script, description in checks:
        print(f"\n{'='*50}")
        print(f"{description.upper()}")
        print(f"{'='*50}")

        if not run_script(script):
            failed_checks.append(description)

    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")

    if not failed_checks:
        print("[SUCCESS] All quality checks passed!")
        return 0
    else:
        print(f"[ERROR] {len(failed_checks)} check(s) failed:")
        for check in failed_checks:
            print(f"  - {check}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
