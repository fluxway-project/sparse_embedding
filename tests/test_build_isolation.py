import hashlib
import runpy
import subprocess
from pathlib import Path

import pytest
import setuptools

SETUP_FILE = Path(__file__).resolve().parents[1] / "setup.py"


def _snapshot(directory):
    return {
        path.relative_to(directory): (
            hashlib.sha256(path.read_bytes()).hexdigest(),
            path.stat().st_ino,
            path.stat().st_mtime_ns,
        )
        for path in directory.rglob("*")
        if path.is_file()
    }


@pytest.fixture
def build_project(tmp_path, monkeypatch):
    metadata = {}
    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: metadata.update(kwargs))
    namespace = runpy.run_path(str(SETUP_FILE), run_name="__main__")
    state = namespace["_build_native_artifacts"].__globals__
    source = tmp_path / "source" / "sparse_embedding"
    source.mkdir(parents=True)
    for name, content in {
        "__init__.py": "SOURCE_MARKER = True\n",
        "_version.py": "SOURCE_VERSION_MARKER = True\n",
        "_core.so": "source CPython 3.11 native library",
        "liblance_c.so": "source Lance library",
        "obsolete.so": "obsolete native library",
        "_core.pyi": "source CPython 3.11 stubs",
        "py.typed": "",
    }.items():
        (source / name).write_text(content)
    (source / "_core").mkdir()
    (source / "_core/__init__.pyi").write_text("source native stubs")
    (source / "_core/checkpoint.pyi").write_text("source checkpoint stubs")
    monkeypatch.setitem(state, "_PROJECT_DIR", source.parent)
    monkeypatch.setitem(state, "_PKG_DIR", source)
    artifacts = tmp_path / "bazel-artifacts"
    artifacts.mkdir()
    (artifacts / "_core.so").write_text("built CPython 3.12 native library")
    (artifacts / "liblance_c.so").write_text("built Lance library")

    def invoke_bazel(with_lance):
        names = ["_core.so", "liblance_c.so"] if with_lance else ["_core.so"]
        return [artifacts / name for name in names]

    def generate_stubs(*, extension_name, extension_dir, output_dir):
        assert (extension_dir / f"{extension_name}.so").read_text() == (
            "built CPython 3.12 native library"
        )
        assert not (extension_dir / f"{extension_name}.pyi").exists()
        assert not (extension_dir / extension_name).exists()
        (output_dir / extension_name).mkdir()
        (output_dir / extension_name / "__init__.pyi").write_text(
            "built CPython 3.12 stubs"
        )
        (output_dir / extension_name / "checkpoint.pyi").write_text(
            "built checkpoint stubs"
        )

    monkeypatch.setitem(state, "_invoke_bazel", invoke_bazel)
    monkeypatch.setitem(state, "_generate_extension_stubs", generate_stubs)

    def command(destination):
        distribution = metadata["distclass"](
            {
                **{key: value for key, value in metadata.items() if key != "distclass"},
                "packages": ["sparse_embedding"],
                "package_dir": {"": str(source.parent)},
            }
        )
        distribution.script_name = str(SETUP_FILE)
        build = distribution.get_command_obj("build_py")
        build.build_lib = str(destination.parent)
        build.ensure_finalized()
        return build

    return source, state, command


@pytest.mark.parametrize("with_lance", [True, False])
def test_build_stages_artifacts_without_changing_source(
    build_project, tmp_path, monkeypatch, with_lance
):
    source, state, command = build_project
    before = _snapshot(source)
    destination = tmp_path / "build" / "sparse_embedding"
    destination.mkdir(parents=True)
    (destination / "removed-module.py").write_text("stale Python module")
    (destination / "liblance_c.so").write_text("stale build Lance library")
    monkeypatch.setenv("SE_BUILD_WITH_LANCE", str(int(with_lance)))

    command(destination).run()

    assert _snapshot(source) == before
    assert (destination / "__init__.py").read_bytes() == (
        source / "__init__.py"
    ).read_bytes()
    assert (destination / "_core.so").read_text() == "built CPython 3.12 native library"
    assert not (destination / "_core.pyi").exists()
    assert (
        destination / "_core/__init__.pyi"
    ).read_text() == "built CPython 3.12 stubs"
    assert (
        destination / "_core/checkpoint.pyi"
    ).read_text() == "built checkpoint stubs"
    assert (destination / "py.typed").is_file()
    assert not (destination / "obsolete.so").exists()
    assert not (destination / "removed-module.py").exists()
    expected_native = {"_core.so", "liblance_c.so"} if with_lance else {"_core.so"}
    assert {path.name for path in destination.glob("*.so")} == expected_native
    version = runpy.run_path(str(destination / "_version.py"))
    assert version["__version__"] == state["_VERSIONS"]["PACKAGE_VERSION"]
    assert version["__python_requires__"] == state["_VERSIONS"]["PYTHON_REQUIRES"]


@pytest.mark.parametrize("target", ["source", "ancestor", "symlink"])
def test_build_rejects_source_destinations_before_native_compilation(
    build_project, tmp_path, monkeypatch, target
):
    source, state, command = build_project
    before = _snapshot(source)

    def unexpected_bazel(*args):
        pytest.fail("Unsafe build destination reached native compilation")

    monkeypatch.setitem(state, "_invoke_bazel", unexpected_bazel)
    if target == "source":
        destination = source
    elif target == "ancestor":
        destination = source.parent
    else:
        destination = tmp_path / "alias" / "sparse_embedding"
        destination.parent.mkdir()
        destination.symlink_to(source, target_is_directory=True)

    with pytest.raises(ValueError, match="source package"):
        if target == "ancestor":
            state["_remove_stale_extension_artifacts"](destination)
        else:
            command(destination).run()
    assert _snapshot(source) == before


def test_stub_generation_failure_preserves_source(build_project, tmp_path, monkeypatch):
    source, state, command = build_project
    before = _snapshot(source)

    def failed_stubs(**kwargs):
        raise subprocess.CalledProcessError(1, ["stubgen"])

    monkeypatch.setitem(state, "_generate_extension_stubs", failed_stubs)
    with pytest.raises(subprocess.CalledProcessError):
        command(tmp_path / "failed-build" / "sparse_embedding").run()
    assert _snapshot(source) == before


@pytest.mark.parametrize("with_lance", [True, False])
def test_explicit_editable_build_updates_native_files_in_place(
    build_project, tmp_path, monkeypatch, with_lance
):
    source, state, command = build_project
    original_python = (source / "__init__.py").read_bytes()
    destination = tmp_path / "editable-build" / "sparse_embedding"
    build = command(destination)
    build.editable_mode = True
    monkeypatch.setenv("SE_BUILD_WITH_LANCE", str(int(with_lance)))

    build.run()

    assert (source / "__init__.py").read_bytes() == original_python
    assert (source / "_core.so").read_text() == "built CPython 3.12 native library"
    assert not (source / "_core.pyi").exists()
    assert (source / "_core/__init__.pyi").read_text() == "built CPython 3.12 stubs"
    assert (source / "_core/checkpoint.pyi").read_text() == "built checkpoint stubs"
    assert not (source / "obsolete.so").exists()
    assert (source / "liblance_c.so").exists() is with_lance
    assert not destination.exists()
