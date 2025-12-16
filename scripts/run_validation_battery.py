#!/usr/bin/env python3
"""
Master validation script for task_space codebase.

Run before any release or after major changes:
    python scripts/run_validation_battery.py

Checks:
1. Unit tests pass
2. Integration tests pass (non-slow)
3. Canonical values protected
4. Required artifacts exist
"""
import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime


def run_command(cmd, description):
    """Run command and return success status."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    result = subprocess.run(
        cmd,
        shell=True,
        env={**dict(__import__('os').environ), 'PYTHONPATH': 'src'}
    )
    return result.returncode == 0


def check_canonical_artifacts():
    """Verify required canonical files exist."""
    required = [
        "outputs/canonical/mobility_cps.json",
        "outputs/canonical/mobility_asymmetric.json",
        "outputs/canonical/wage_comovement.json",
        "outputs/canonical/automation_v0653.json",
        "outputs/experiments/path_a_wasserstein_comparison_v0672.json",
        ".cache/artifacts/v1/wasserstein/d_wasserstein_onet.npz",
    ]

    print(f"\n{'='*60}")
    print("  Checking canonical artifacts")
    print(f"{'='*60}")

    missing = []
    for path in required:
        if Path(path).exists():
            print(f"  [OK] {path}")
        else:
            print(f"  [MISSING] {path}")
            missing.append(path)

    return len(missing) == 0


def check_version_consistency():
    """Verify version numbers match across files."""
    print(f"\n{'='*60}")
    print("  Checking version consistency")
    print(f"{'='*60}")

    # Get version from pyproject.toml
    with open("pyproject.toml") as f:
        for line in f:
            if line.startswith("version"):
                pyproject_version = line.split('"')[1]
                break

    print(f"  pyproject.toml: {pyproject_version}")

    # Check CLAUDE.md
    claude_ok = False
    with open("CLAUDE.md") as f:
        content = f.read()
        if pyproject_version in content:
            claude_ok = True
            print(f"  CLAUDE.md: [OK] contains {pyproject_version}")
        else:
            print(f"  CLAUDE.md: [MISMATCH] missing {pyproject_version}")

    # Check README.md
    readme_ok = False
    with open("README.md") as f:
        content = f.read()
        if pyproject_version in content:
            readme_ok = True
            print(f"  README.md: [OK] contains {pyproject_version}")
        else:
            print(f"  README.md: [MISMATCH] missing {pyproject_version}")

    return claude_ok and readme_ok


def main():
    start = datetime.now()
    results = {}

    print("\n" + "="*60)
    print("  TASK SPACE VALIDATION BATTERY")
    print("="*60)

    # 1. Unit tests
    results["unit_tests"] = run_command(
        ".venv/bin/pytest tests/unit --tb=short -q",
        "Running unit tests"
    )

    # 2. Integration tests (non-slow)
    results["integration_fast"] = run_command(
        ".venv/bin/pytest tests/integration --tb=short -q -m 'not slow'",
        "Running integration tests (fast)"
    )

    # 3. Canonical value protection
    results["canonical_values"] = run_command(
        ".venv/bin/pytest tests/integration/test_canonical_values.py --tb=short -q",
        "Verifying canonical values"
    )

    # 4. Artifact existence
    results["artifacts_exist"] = check_canonical_artifacts()

    # 5. Version consistency
    results["version_consistency"] = check_version_consistency()

    # Summary
    elapsed = (datetime.now() - start).total_seconds()

    print(f"\n{'='*60}")
    print("  VALIDATION SUMMARY")
    print(f"{'='*60}")

    all_passed = True
    for check, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {check}: {status}")
        if not passed:
            all_passed = False

    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"  Overall: {'[ALL CHECKS PASSED]' if all_passed else '[SOME CHECKS FAILED]'}")

    # Save report
    report = {
        "timestamp": datetime.now().isoformat(),
        "results": results,
        "all_passed": all_passed,
        "elapsed_seconds": elapsed,
    }

    report_path = Path("outputs/validation_report.json")
    report_path.parent.mkdir(exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved: {report_path}")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
