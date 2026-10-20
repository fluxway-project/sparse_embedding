import ast
import base64
import csv
import hashlib
import io
import runpy
import struct
import sys
import zipfile
from email.parser import BytesParser
from pathlib import Path

from packaging.tags import parse_tag, sys_tags
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version


def validate_wheel(directory: Path) -> Path:
    wheels = tuple(directory.glob("*.whl"))
    if len(wheels) != 1:
        raise ValueError(f"Expected one wheel in {directory}, found {len(wheels)}")
    wheel = wheels[0].resolve()
    name, version, _, tags = parse_wheel_filename(wheel.name)
    project_dir = Path(__file__).resolve().parents[1]
    versions = runpy.run_path(str(project_dir / "versions.bzl"))
    if name != canonicalize_name("sparse_embedding"):
        raise ValueError(f"Unexpected wheel package: {name}")
    if version != Version(versions["PACKAGE_VERSION"]):
        raise ValueError(f"Unexpected wheel version: {version}")
    interpreter = f"cp{sys.version_info.major}{sys.version_info.minor}"
    if any(tag.interpreter != interpreter or tag.abi != interpreter for tag in tags):
        raise ValueError(f"Expected {interpreter} Python and ABI tags, got {tags}")
    if not tags.intersection(sys_tags()):
        raise ValueError(f"Wheel is incompatible with this interpreter: {tags}")

    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        files = set(names)
        if len(names) != len(files):
            raise ValueError("Duplicate wheel archive entries")
        (metadata_path,) = (
            path for path in files if path.endswith(".dist-info/METADATA")
        )
        metadata = BytesParser().parsebytes(archive.read(metadata_path))
        if metadata["Requires-Python"] != versions["PYTHON_REQUIRES"]:
            raise ValueError(
                f"Unexpected Requires-Python: {metadata['Requires-Python']}"
            )
        (wheel_path,) = (path for path in files if path.endswith(".dist-info/WHEEL"))
        wheel_metadata = BytesParser().parsebytes(archive.read(wheel_path))
        metadata_tags = set().union(
            *(parse_tag(tag) for tag in wheel_metadata.get_all("Tag", []))
        )
        if metadata_tags != tags or wheel_metadata["Root-Is-Purelib"] != "false":
            raise ValueError("Wheel metadata disagrees with its native binary tags")
        package_files = (
            "_core.so",
            "liblance_c.so",
            "py.typed",
            "_core/__init__.pyi",
            "_core/checkpoint.pyi",
        )
        required = {f"sparse_embedding/{name}" for name in package_files}
        if missing := required - files:
            raise ValueError(f"Wheel is missing package artifacts: {sorted(missing)}")
        if any(path.startswith("sparse_embedding/tests/") for path in files):
            raise ValueError("Wheel contains the source test suite")
        if "sparse_embedding/_core.pyi" in files:
            raise ValueError("Wheel contains an obsolete flat native stub")
        for path in sorted(files):
            if path.startswith("sparse_embedding/_core/") and path.endswith(".pyi"):
                try:
                    tree = ast.parse(archive.read(path), filename=path)
                except SyntaxError as error:
                    raise ValueError(
                        f"Invalid native stub: {path}: {error.msg}"
                    ) from error
                for node in ast.walk(tree):
                    imports = []
                    if isinstance(node, ast.Import):
                        imports = [alias.name for alias in node.names]
                    elif isinstance(node, ast.ImportFrom) and node.level == 0:
                        imports = [node.module]
                    if any(
                        name == "_core" or name.startswith("_core.") for name in imports
                    ):
                        raise ValueError(f"Invalid top-level _core import in {path}")
            if path in {"sparse_embedding/_core.so", "sparse_embedding/liblance_c.so"}:
                with archive.open(path) as stream:
                    header = stream.read(64)
                if (
                    len(header) != 64
                    or header[:6] != b"\x7fELF\x02\x01"
                    or struct.unpack_from("<HH", header, 16) != (3, 62)
                ):
                    raise ValueError(f"Invalid Linux x86-64 native library: {path}")
        (record_path,) = (path for path in files if path.endswith(".dist-info/RECORD"))
        rows = list(csv.reader(io.StringIO(archive.read(record_path).decode())))
        if any(len(row) != 3 for row in rows):
            raise ValueError("Invalid wheel RECORD rows")
        record = {row[0]: row[1:] for row in rows}
        if len(record) != len(rows) or set(record) != files:
            raise ValueError("Wheel RECORD does not match archive entries")
        if record[record_path] != ["", ""]:
            raise ValueError("Wheel RECORD must not hash itself")
        for path in sorted(files - {record_path}):
            with archive.open(path) as stream:
                digest = hashlib.file_digest(stream, "sha256").digest()
            encoded = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
            expected = [f"sha256={encoded}", str(archive.getinfo(path).file_size)]
            if record[path] != expected:
                raise ValueError(f"Wheel RECORD hash or size mismatch: {path}")
    return wheel


if __name__ == "__main__":
    print(validate_wheel(Path(sys.argv[1])))
