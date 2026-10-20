import base64
import csv
import hashlib
import io
import runpy
import struct
import sys
import zipfile
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
validate_wheel = runpy.run_path(str(PROJECT / "scripts/validate_wheel.py"))[
    "validate_wheel"
]


@pytest.fixture
def write_wheel(tmp_path):
    versions = runpy.run_path(str(PROJECT / "versions.bzl"))
    version = versions["PACKAGE_VERSION"]
    abi = f"cp{sys.version_info.major}{sys.version_info.minor}"
    dist = f"sparse_embedding-{version}.dist-info"
    elf = b"\x7fELF\x02\x01" + bytes(10) + struct.pack("<HH", 3, 62) + bytes(44)
    files = {
        "sparse_embedding/_core.so": elf,
        "sparse_embedding/liblance_c.so": elf,
        "sparse_embedding/_core/__init__.pyi": b"from . import checkpoint as checkpoint\n",
        "sparse_embedding/_core/checkpoint.pyi": b"def validate_native_checkpoint_chain() -> bool: ...\n",
        "sparse_embedding/py.typed": b"",
        f"{dist}/METADATA": f"Name: sparse_embedding\nVersion: {version}\nRequires-Python: {versions['PYTHON_REQUIRES']}\n".encode(),
        f"{dist}/WHEEL": f"Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: {abi}-{abi}-linux_x86_64\n".encode(),
    }

    def write(mutate=None, *, corrupt_record=False, duplicate=False):
        if mutate:
            mutate(files)
        rows = []
        for name, data in files.items():
            digest = (
                base64.urlsafe_b64encode(hashlib.sha256(data).digest())
                .rstrip(b"=")
                .decode()
            )
            rows.append((name, f"sha256={digest}", str(len(data))))
        if corrupt_record:
            rows[0] = (rows[0][0], "sha256=incorrect", rows[0][2])
        record = io.StringIO()
        csv.writer(record).writerows([*rows, (f"{dist}/RECORD", "", "")])
        wheel = tmp_path / f"sparse_embedding-{version}-{abi}-{abi}-linux_x86_64.whl"
        with zipfile.ZipFile(wheel, "w") as archive:
            for name, data in files.items():
                archive.writestr(name, data)
            archive.writestr(f"{dist}/RECORD", record.getvalue())
            if duplicate:
                with pytest.warns(UserWarning, match="Duplicate name"):
                    archive.writestr("sparse_embedding/py.typed", b"")
        return wheel

    return tmp_path, write


def test_validator_accepts_recursive_native_stubs(write_wheel):
    directory, write = write_wheel
    wheel = write()
    assert validate_wheel(directory) == wheel


@pytest.mark.parametrize(
    "missing", ["_core.so", "liblance_c.so", "_core/checkpoint.pyi"]
)
def test_validator_rejects_missing_native_artifacts(write_wheel, missing):
    directory, write = write_wheel
    write(lambda files: files.pop(f"sparse_embedding/{missing}"))
    with pytest.raises(ValueError, match="missing package artifacts"):
        validate_wheel(directory)


def test_validator_rejects_invalid_stub_syntax(write_wheel):
    directory, write = write_wheel
    write(
        lambda files: files.update(
            {
                "sparse_embedding/_core/checkpoint.pyi": b"def restore(options: dict = {}, generation: int) -> None: ...\n",
            }
        )
    )
    with pytest.raises(ValueError, match="Invalid native stub"):
        validate_wheel(directory)


def test_validator_rejects_top_level_native_import(write_wheel):
    directory, write = write_wheel
    write(
        lambda files: files.update(
            {
                "sparse_embedding/_core/checkpoint.pyi": b"import _core\n",
            }
        )
    )
    with pytest.raises(ValueError, match="top-level _core"):
        validate_wheel(directory)


def test_validator_rejects_obsolete_flat_stub(write_wheel):
    directory, write = write_wheel
    write(lambda files: files.update({"sparse_embedding/_core.pyi": b""}))
    with pytest.raises(ValueError, match="obsolete flat native stub"):
        validate_wheel(directory)


def test_validator_rejects_record_corruption(write_wheel):
    directory, write = write_wheel
    write(corrupt_record=True)
    with pytest.raises(ValueError, match="RECORD"):
        validate_wheel(directory)


def test_validator_rejects_duplicate_archive_entries(write_wheel):
    directory, write = write_wheel
    write(duplicate=True)
    with pytest.raises(ValueError, match="Duplicate wheel"):
        validate_wheel(directory)
