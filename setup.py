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


def _sync_version_py(build_package_dir: Path) -> None:
    """Write the built package's version metadata from versions.bzl."""
    content = f'''# Generated from versions.bzl — Single Source of Truth.
__version__ = "{_VERSIONS["PACKAGE_VERSION"]}"
__python_requires__ = "{_VERSIONS["PYTHON_REQUIRES"]}"
__lance_c_version__ = "{_VERSIONS["LANCE_C_VERSION"]}"
__lance_c_commit__ = "{_VERSIONS["LANCE_C_COMMIT"]}"
'''
    version_file = build_package_dir / "_version.py"
    version_file.write_text(content)


def _remove_stale_extension_artifacts(build_package_dir: Path) -> None:
    destination = build_package_dir.resolve()
    source = _PKG_DIR.resolve()
    if destination.is_relative_to(source):
        raise ValueError("The build destination must be outside the source package")
    if source.is_relative_to(destination):
        raise ValueError("The build destination must not contain the source package")
    if build_package_dir.exists():
        shutil.rmtree(build_package_dir)


def _invoke_bazel(with_lance: bool) -> list[Path]:
    target = "//:native_artifacts"
    env = os.environ.copy()
    options = shlex.split(env.pop("BAZEL_COPTS", ""))
    options.append(
        "--@rules_python//python/config_settings:python_version="
        f"{sys.version_info.major}.{sys.version_info.minor}"
    )
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
    """Compile native libraries and generate artifacts in the build package."""
    with_lance = _is_lance_enabled()
    print(
        f"[setup.py] Building C++ backend via Bazel (lance={'yes' if with_lance else 'no'})...",
        file=sys.stderr,
    )
    artifacts = _invoke_bazel(with_lance)
    build_package_dir.mkdir(parents=True, exist_ok=True)
    for candidate in build_package_dir.glob("*.so"):
        candidate.unlink()
    for artifact in artifacts:
        destination = build_package_dir / artifact.name
        shutil.copy2(artifact, destination)
        print(f"[setup.py] Copied {artifact} -> {destination}", file=sys.stderr)

    _sync_version_py(build_package_dir)
    (build_package_dir / "_core.pyi").unlink(missing_ok=True)
    native_stubs = build_package_dir / "_core"
    if native_stubs.exists():
        shutil.rmtree(native_stubs)
    _generate_extension_stubs(
        extension_name="_core",
        extension_dir=build_package_dir,
        output_dir=build_package_dir,
    )


def _generate_extension_stubs(
    extension_name: str,
    extension_dir: Path,
    output_dir: Path,
) -> None:
    """Generate recursive stubs from the staged extension under its package name."""
    print(
        f"[setup.py] Generating type stubs for native extension {extension_name!r}...",
        file=sys.stderr,
    )
    try:
        subprocess.run(
            [
                sys.executable,
                str(_PROJECT_DIR / "scripts/generate_native_stubs.py"),
                "--module",
                f"{_PKG_DIR.name}.{extension_name}",
                "--extension",
                str(extension_dir / f"{extension_name}.so"),
                "--output-dir",
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
        f"[setup.py] Successfully generated recursive {extension_name} stubs",
        file=sys.stderr,
    )


class BazelAwareBuildPy(build_py):
    """Assemble Python and native files in the selected build directory."""

    editable_mode = False

    def run(self) -> None:
        if self.editable_mode:
            _build_native_artifacts(_PKG_DIR)
            return
        build_package_dir = Path(self.build_lib) / "sparse_embedding"
        _remove_stale_extension_artifacts(build_package_dir)
        super().run()
        _build_native_artifacts(build_package_dir)


class BinaryDistribution(Distribution):
    """Distribution which forces a binary package with platform name"""

    def has_ext_modules(self):
        return True


if __name__ == "__main__":
    setup(
        packages=find_namespace_packages(
            include=["sparse_embedding*"],
            exclude=["sparse_embedding.tests*", "sparse_embedding._core*"],
        ),
        version=_VERSIONS["PACKAGE_VERSION"],
        python_requires=_VERSIONS["PYTHON_REQUIRES"],
        distclass=BinaryDistribution,
        cmdclass={"build_py": BazelAwareBuildPy},
        include_package_data=False,
        exclude_package_data={
            "sparse_embedding": [
                "tests/*",
                "tests/**/*",
                "*.so",
                "_core.pyi",
                "_core/*",
                "_core/**/*",
            ],
        },
        package_data={"sparse_embedding": ["py.typed"]},
    )
