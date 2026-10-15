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
"""

import os
import runpy
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import find_namespace_packages, setup
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
    if val in ("1", "true", "on", "yes"):
        return True
    if val in ("0", "false", "off", "no"):
        return False
    raise ValueError(f"Invalid SE_BUILD_WITH_LANCE value: {val!r}")


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


def _remove_stale_extension_artifacts(build_package_dir: Path) -> None:
    if build_package_dir.resolve().is_relative_to(_PKG_DIR):
        raise ValueError("The build destination must be outside the source package")
    if _PKG_DIR.is_relative_to(build_package_dir.resolve()):
        raise ValueError("The build destination must not contain the source package")
    if build_package_dir.exists():
        shutil.rmtree(build_package_dir)
    for candidate in _PKG_DIR.glob("*.so"):
        candidate.unlink()


def _invoke_bazel(with_lance: bool) -> list[Path]:
    target = "//:native_artifacts"
    env = os.environ.copy()
    options = shlex.split(env.pop("BAZEL_COPTS", ""))
    options.append(f"--//:with_lance={str(with_lance).lower()}")
    execution_root = subprocess.run(
        ["bazel", "info", "execution_root", *options],
        cwd=_PROJECT_DIR,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run(
        ["bazel", "build", target, *options],
        cwd=_PROJECT_DIR,
        env=env,
        check=True,
    )

    result = subprocess.run(
        ["bazel", "cquery", target, "--output=files", *options],
        cwd=_PROJECT_DIR,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    artifacts = [
        Path(execution_root) / path.strip()
        for path in result.stdout.splitlines()
        if path.strip()
    ]
    if not artifacts:
        raise RuntimeError(f"Bazel target {target} declared no native artifacts")
    for artifact in artifacts:
        if not artifact.is_file():
            raise FileNotFoundError(f"Bazel artifact does not exist: {artifact}")
    return artifacts


def _build_native_artifacts(build_package_dir: Path) -> None:
    """Compile native libraries via Bazel and copy them into package directory."""
    with_lance = _is_lance_enabled()
    _sync_version_py()
    print(
        f"[setup.py] Building C++ backend via Bazel (lance={'yes' if with_lance else 'no'})...",
        file=sys.stderr,
    )
    artifacts = _invoke_bazel(with_lance)
    _remove_stale_extension_artifacts(build_package_dir)
    for artifact in artifacts:
        destination = _PKG_DIR / artifact.name
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        shutil.copy2(artifact, destination)
        print(f"[setup.py] Copied {artifact} -> {destination}", file=sys.stderr)

    _generate_extension_stubs(
        extension_name="_core",
        extension_dir=_PKG_DIR,
        output_dir=_PKG_DIR,
    )


def _generate_extension_stubs(
    extension_name: str,
    extension_dir: Path,
    output_dir: Path,
) -> None:
    """Generate .pyi stubs from compiled extension binary without importing top-level package.

    Uses nanobind's import-path mechanism to inspect only the binary shared library,
    ensuring hermetic execution in PEP 517 isolated environments without requiring
    runtime package dependencies.
    """
    print(
        f"[setup.py] Generating type stubs for native extension {extension_name!r}...",
        file=sys.stderr,
    )
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "nanobind.stubgen",
                "-P",
                "-i",
                str(extension_dir),
                "-m",
                extension_name,
                "-O",
                str(output_dir),
            ],
            cwd=_PROJECT_DIR,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        print(
            f"[setup.py] nanobind.stubgen failed for {extension_name} with exit code {e.returncode}",
            file=sys.stderr,
        )
        print(f"[setup.py] stdout: {e.stdout}", file=sys.stderr)
        print(f"[setup.py] stderr: {e.stderr}", file=sys.stderr)
        raise
    print(
        f"[setup.py] Successfully generated {extension_name}.pyi",
        file=sys.stderr,
    )


class BazelAwareBuildPy(build_py):
    """Extends build_py to run the Bazel compilation step first."""

    def run(self) -> None:
        _build_native_artifacts(Path(self.build_lib) / "sparse_embedding")
        super().run()


class BinaryDistribution(Distribution):
    """Distribution which forces a binary package with platform name"""

    def has_ext_modules(self):
        return True


if __name__ == "__main__":
    setup(
        packages=find_namespace_packages(
            include=["sparse_embedding*"],
            exclude=["sparse_embedding.tests*"],
        ),
        version=_VERSIONS["PACKAGE_VERSION"],
        python_requires=_VERSIONS["PYTHON_REQUIRES"],
        distclass=BinaryDistribution,
        cmdclass={"build_py": BazelAwareBuildPy},
        include_package_data=False,
        exclude_package_data={"sparse_embedding": ["tests/*", "tests/**/*"]},
        package_data={"sparse_embedding": ["*.so", "*.pyi", "py.typed"]},
    )
