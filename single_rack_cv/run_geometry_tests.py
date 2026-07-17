#!/usr/bin/env python3
"""Run geometry validation layers in separate Python processes."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parent
SMOKE_MODULE = "tests.isaac_camera_smoke_test"
SMOKE_STATUS_ENV = "GEOMETRY_SMOKE_STATUS_FILE"


def _run_layer(
    command: list[str],
    *,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run one validation layer in a fresh process and stream its output."""
    return subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        check=False,
        text=True,
    )


def _smoke_status_passed(returncode: int, status_text: str | None) -> bool:
    """Require both a clean child exit and an explicit PASS sentinel."""
    if returncode != 0 or status_text is None:
        return False
    first_line = status_text.splitlines()[0].strip() if status_text else ""
    return first_line == "PASS"


def run_unit_tests() -> bool:
    """Run pure geometry tests in a process that may load YOLOE/OpenCV."""
    print(
        "\n============================================================\n"
        "LAYER 1: PURE CAMERA/STEREO GEOMETRY TESTS\n"
        "============================================================",
        flush=True,
    )

    completed = _run_layer(
        [
            sys.executable,
            "-m",
            "unittest",
            "-v",
            "tests.test_geometry_math",
        ]
    )

    if completed.returncode == 0:
        print("\n[PASS] PURE GEOMETRY TESTS COMPLETE", flush=True)
        return True

    print(
        "\n[FAIL] PURE GEOMETRY TESTS FAILED\n"
        "Isaac Sim smoke test will not run.",
        flush=True,
    )
    return False


def run_smoke_test_layer() -> bool:
    """Run Isaac Sim in a clean process and verify its explicit result."""
    print(
        "\n============================================================\n"
        "LAYER 2: LIVE ISAAC SIM CAMERA SMOKE TEST\n"
        "============================================================",
        flush=True,
    )

    fd, status_name = tempfile.mkstemp(
        prefix="geometry_smoke_",
        suffix=".status",
    )
    os.close(fd)
    status_path = Path(status_name)
    status_path.unlink(missing_ok=True)

    env = os.environ.copy()
    env[SMOKE_STATUS_ENV] = str(status_path)

    try:
        completed = _run_layer(
            [
                sys.executable,
                "-m",
                SMOKE_MODULE,
            ],
            env=env,
        )

        status_text = (
            status_path.read_text(encoding="utf-8")
            if status_path.exists()
            else None
        )

        if _smoke_status_passed(completed.returncode, status_text):
            return True

        print(
            "\n[FAIL] LIVE ISAAC SIM CAMERA SMOKE TEST",
            flush=True,
        )
        print(
            f"child return code: {completed.returncode}",
            flush=True,
        )

        if status_text is None:
            print(
                "smoke status: missing "
                "(the child crashed or exited before reporting a result)",
                flush=True,
            )
        else:
            print(
                "smoke status:\n" + status_text.rstrip(),
                flush=True,
            )
        return False

    finally:
        status_path.unlink(missing_ok=True)


def main() -> int:
    """Return zero only when both isolated validation layers pass."""
    print(
        "SINGLE-RACK STEREO GEOMETRY VALIDATION\n"
        f"project root: {ROOT}\n"
        "process isolation: enabled\n"
        "explicit smoke-result sentinel: enabled",
        flush=True,
    )

    if not run_unit_tests():
        return 1

    if not run_smoke_test_layer():
        return 1

    print(
        "\n============================================================\n"
        "ALL GEOMETRY VALIDATION CHECKS PASSED\n"
        "The existing detector/controller files were not modified.\n"
        "============================================================",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
