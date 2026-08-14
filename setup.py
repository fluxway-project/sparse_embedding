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
from setuptools.dist import Distribution

_PROJECT_DIR = Path(__file__).resolve().parent
_PKG_DIR = _PROJECT_DIR / "sparse_embedding"


def _bazel_bin_dir() -> Path:
    """Ask Bazel for the canonical bazel-bin path, bypassing symlink issues."""
    result = subprocess.run(
        ["bazel", "info", "bazel-bin"],
        cwd=_PROJECT_DIR,
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(result.stdout.strip())


def _invoke_bazel(with_lance: bool) -> Path:
    """Run Bazel and return the path to the compiled _core.so."""
    if with_lance:
        bazel_target = "//:core_so_with_lance"
        so_glob = "libcore_so_with_lance.so"
    else:
        bazel_target = "//:copy_core_so"
        so_glob = "_core.so"

    cmd = ["bazel", "build", bazel_target]
    if with_lance:
        cmd.append("//:copy_liblance_c")
    env = os.environ.copy()
    if "BAZEL_COPTS" in env:
        import shlex

        cmd.extend(shlex.split(env.pop("BAZEL_COPTS")))

    subprocess.run(
        cmd,
        cwd=_PROJECT_DIR,
        env=env,
        check=True,
    )

    bazel_bin = _bazel_bin_dir()
    candidates = list(bazel_bin.rglob(so_glob))
    if not candidates:
        raise FileNotFoundError(
            f"Bazel target {bazel_target!r} produced no file matching {so_glob!r}"
            f" under {bazel_bin}"
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
            if so_dst.exists():
                so_dst.unlink()
            shutil.copy2(so_src, so_dst)
            print(f"[setup.py] Copied {so_src.name} -> {so_dst}", file=sys.stderr)

            if with_lance:
                lance_so = _PKG_DIR / "liblance_c.so"
                lance_src = _bazel_bin_dir() / "sparse_embedding" / "liblance_c.so"
                if not lance_src.is_file():
                    raise FileNotFoundError(
                        f"Lance target produced no shared library at {lance_src}"
                    )
                if lance_so.exists() or lance_so.is_symlink():
                    lance_so.unlink()
                shutil.copy2(lance_src, lance_so)
                print(
                    f"[setup.py] Copied {lance_src.name} -> {lance_so}", file=sys.stderr
                )

            # Generate the _core.pyi type stubs using nanobind.stubgen
            print("[setup.py] Generating type stubs (_core.pyi)...", file=sys.stderr)
            try:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "nanobind.stubgen",
                        "-P",
                        "-m",
                        "sparse_embedding._core",
                        "-O",
                        str(_PKG_DIR),
                    ],
                    cwd=str(_PROJECT_DIR),
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except subprocess.CalledProcessError as e:
                print(
                    f"[setup.py] nanobind.stubgen failed with exit code {e.returncode}",
                    file=sys.stderr,
                )
                print(f"[setup.py] stdout: {e.stdout}", file=sys.stderr)
                print(f"[setup.py] stderr: {e.stderr}", file=sys.stderr)
                raise
            print("[setup.py] Successfully generated _core.pyi", file=sys.stderr)

        super().run()


class BinaryDistribution(Distribution):
    """Distribution which forces a binary package with platform name"""

    def has_ext_modules(self):
        return True


setup(
    distclass=BinaryDistribution,
    cmdclass={"build_py": BazelAwareBuildPy},
    package_data={"sparse_embedding": ["*.so", "*.pyi", "py.typed"]},
)
