"""
Custom build script for sparse_embedding.

Integrates Bazel into the setuptools build pipeline so that `pip install .`,
`pip install -e .`, and `pip wheel .` automatically compile the C++ backend
and bundle the resulting shared libraries.

Environment variables
---------------------
SE_BUILD_WITH_LANCE
    Set to "0", "false", "off", or "no" to disable Lance-c integration and build
    the standalone core (//:package_core_extension). Defaults to "1" (Lance enabled).
SE_SKIP_BAZEL
    Set to any non-empty value to skip the Bazel step entirely (useful when
    the .so is already present and only Python code needs repackaging).
"""

import os
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.dist import Distribution

_PROJECT_DIR = Path(__file__).resolve().parent
_PKG_DIR = _PROJECT_DIR / "sparse_embedding"
_VERSIONS_FILE = _PROJECT_DIR / "versions.bzl"

if not _VERSIONS_FILE.is_file():
    raise FileNotFoundError(
        f"Missing Single Source of Truth version file: {_VERSIONS_FILE}"
    )

_VERSIONS = runpy.run_path(str(_VERSIONS_FILE))


def _is_lance_enabled() -> bool:
    """Return whether Lance integration is enabled (default: True)."""
    val = os.environ.get("SE_BUILD_WITH_LANCE", "1").strip().lower()
    return val not in ("0", "false", "off", "no")


def _sync_version_py() -> None:
    """Write sparse_embedding/_version.py directly from versions.bzl."""
    content = f'''# Generated from versions.bzl — Single Source of Truth.
__version__ = "{_VERSIONS["PACKAGE_VERSION"]}"
__python_requires__ = "{_VERSIONS["PYTHON_REQUIRES"]}"
__lance_c_version__ = "{_VERSIONS["LANCE_C_VERSION"]}"
__lance_c_commit__ = "{_VERSIONS["LANCE_C_COMMIT"]}"
'''
    version_file = _PKG_DIR / "_version.py"
    version_file.write_text(content)


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


def _remove_stale_extension_artifacts() -> None:
    """Remove ABI-tagged extensions left by older setuptools builds."""
    candidates = list(_PKG_DIR.glob("_core.*.so"))
    build_root = _PROJECT_DIR / "build"
    if build_root.is_dir():
        candidates.extend(build_root.glob("lib.*/sparse_embedding/_core.*.so"))
    for candidate in candidates:
        candidate.unlink()


def _invoke_bazel(with_lance: bool) -> Path:
    """Run Bazel and return the path to the compiled _core.so."""
    if with_lance:
        bazel_target = "//:core_extension_lance"
        so_glob = "libcore_extension_lance.so"
    else:
        bazel_target = "//:package_core_extension"
        so_glob = "_core.so"

    cmd = ["bazel", "build", bazel_target]
    if with_lance:
        cmd.append("//:package_liblance_c")
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


def _build_native_artifacts() -> None:
    """Compile native libraries via Bazel and copy them into package directory."""
    _sync_version_py()
    _remove_stale_extension_artifacts()
    if os.environ.get("SE_SKIP_BAZEL"):
        print("[setup.py] SE_SKIP_BAZEL set — skipping Bazel build.", file=sys.stderr)
        return

    with_lance = _is_lance_enabled()
    print(
        f"[setup.py] Building C++ backend via Bazel (lance={'yes' if with_lance else 'no'})...",
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
        print(f"[setup.py] Copied {lance_src.name} -> {lance_so}", file=sys.stderr)

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


class BazelAwareBuildPy(build_py):
    """Extends build_py to run the Bazel compilation step first."""

    def run(self) -> None:
        _build_native_artifacts()
        super().run()


class BinaryDistribution(Distribution):
    """Distribution which forces a binary package with platform name"""

    def has_ext_modules(self):
        return True


if __name__ == "__main__":
    setup(
        version=_VERSIONS["PACKAGE_VERSION"],
        python_requires=_VERSIONS["PYTHON_REQUIRES"],
        distclass=BinaryDistribution,
        cmdclass={"build_py": BazelAwareBuildPy},
        package_data={"sparse_embedding": ["*.so", "*.pyi", "py.typed"]},
    )
