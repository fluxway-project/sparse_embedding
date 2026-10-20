import ast
import runpy
import subprocess
import sys
import tomllib
from pathlib import Path
from unittest.mock import patch

import setuptools
from packaging.requirements import Requirement

PROJECT = Path(__file__).resolve().parents[1]


def test_stub_generation_uses_qualified_module_without_importing_parent(tmp_path):
    package = tmp_path / "packaging_probe"
    package.mkdir()
    (package / "__init__.py").write_text(
        "raise RuntimeError('Parent initialization ran')\n"
    )
    extension = package / "_core.py"
    extension.write_text(
        "import sys\n"
        "import types\n"
        "class EmbeddingStore: pass\n"
        "checkpoint = types.ModuleType(__name__ + '.checkpoint')\n"
        "checkpoint.__dict__['EmbeddingStore'] = EmbeddingStore\n"
        "exec('def validate_native_checkpoint_chain(store: EmbeddingStore) -> bool: return True', checkpoint.__dict__)\n"
        "sys.modules[checkpoint.__name__] = checkpoint\n"
    )
    output = tmp_path / "stubs"
    subprocess.run(
        [
            sys.executable,
            str(PROJECT / "scripts/generate_native_stubs.py"),
            "--module",
            "packaging_probe._core",
            "--extension",
            str(extension),
            "--output-dir",
            str(output),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    root = output / "_core/__init__.pyi"
    checkpoint = output / "_core/checkpoint.pyi"
    assert root.is_file()
    assert checkpoint.is_file()
    assert not (output / "_core.pyi").exists()
    assert "from . import checkpoint as checkpoint" in root.read_text()
    assert "packaging_probe._core" in checkpoint.read_text()
    assert "def validate_native_checkpoint_chain" in checkpoint.read_text()
    for stub in output.rglob("*.pyi"):
        ast.parse(stub.read_text(), filename=str(stub))


def test_stubgen_version_matches_native_dependency():
    module = ast.parse((PROJECT / "MODULE.bazel").read_text())
    native_versions = [
        ast.literal_eval(keyword.value)
        for node in ast.walk(module)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "bazel_dep"
        and any(
            keyword.arg == "name" and ast.literal_eval(keyword.value) == "nanobind"
            for keyword in node.keywords
        )
        for keyword in node.keywords
        if keyword.arg == "version"
    ]
    (native_version,) = native_versions
    project = tomllib.loads((PROJECT / "pyproject.toml").read_text())
    (requirement,) = (
        Requirement(value)
        for value in project["build-system"]["requires"]
        if Requirement(value).name == "nanobind"
    )
    assert str(requirement.specifier) == f"=={native_version}"


def test_setup_excludes_source_native_stub_packages():
    with (
        patch.object(setuptools, "find_namespace_packages", return_value=[]) as find,
        patch.object(setuptools, "setup"),
    ):
        runpy.run_path(str(PROJECT / "setup.py"), run_name="__main__")
    assert "sparse_embedding._core*" in find.call_args.kwargs["exclude"]
