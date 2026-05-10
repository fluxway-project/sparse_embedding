"""
Custom build script for sparse_embedding.

Integrates Bazel into the setuptools build pipeline so that `pip install .`
and `pip wheel .` automatically compile the C++ backend and bundle the
resulting shared library into the wheel.

Environment variables
---------------------
SE_BUILD_WITH_LANCE
    Set to any non-empty value to build the Lance-enabled variant
    (//:core_so_with_lance) instead of the default no-Lance build
    (//:copy_core_so).
SE_SKIP_BAZEL
    Set to any non-empty value to skip the Bazel step entirely (useful when
    the .so is already present and only Python code needs repackaging).
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py

_PROJECT_DIR = Path(__file__).resolve().parent
_PKG_DIR = _PROJECT_DIR / "sparse_embedding"


def _invoke_bazel(with_lance: bool) -> Path:
    """Run Bazel and return the path to the compiled _core.so."""
    if with_lance:
        bazel_target = "//:core_so_with_lance"
        so_glob = "libcore_so_with_lance.so"
    else:
        bazel_target = "//:copy_core_so"
        so_glob = "_core.so"

    subprocess.run(
        ["bazel", "build", bazel_target],
        cwd=_PROJECT_DIR,
        check=True,
    )

    candidates = list((_PROJECT_DIR / "bazel-bin").rglob(so_glob))
    if not candidates:
        raise FileNotFoundError(
            f"Bazel target {bazel_target!r} produced no file matching {so_glob!r}"
            f" under bazel-bin/"
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


class BazelAwareBuildPy(build_py):
    """Extends build_py to run the Bazel compilation step first."""

    def run(self) -> None:
        if os.environ.get("SE_SKIP_BAZEL"):
            print(
                "[setup.py] SE_SKIP_BAZEL set — skipping Bazel build.", file=sys.stderr
            )
        else:
            with_lance = bool(os.environ.get("SE_BUILD_WITH_LANCE", ""))
            print(
                f"[setup.py] Building C++ backend via Bazel"
                f" (lance={'yes' if with_lance else 'no'})...",
                file=sys.stderr,
            )
            so_src = _invoke_bazel(with_lance)
            so_dst = _PKG_DIR / "_core.so"
            shutil.copy2(so_src, so_dst)
            print(f"[setup.py] Copied {so_src.name} -> {so_dst}", file=sys.stderr)

            if with_lance:
                lance_so = _PKG_DIR / "liblance_c.so"
                if not lance_so.exists():
                    bazel_lance = (
                        _PROJECT_DIR
                        / "bazel-bin"
                        / "sparse_embedding"
                        / "liblance_c.so"
                    )
                    if bazel_lance.exists():
                        shutil.copy2(bazel_lance, lance_so)
                        print(
                            f"[setup.py] Copied liblance_c.so -> {lance_so}",
                            file=sys.stderr,
                        )

        super().run()


setup(cmdclass={"build_py": BazelAwareBuildPy})
