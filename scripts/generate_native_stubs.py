import argparse
import ast
import importlib.util
import sys
from pathlib import Path

from nanobind.stubgen import main as stubgen


def generate_native_stubs(module_name: str, extension: Path, output_dir: Path) -> None:
    spec = importlib.util.spec_from_file_location(module_name, extension)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load native module {module_name} from {extension}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    stubgen(
        [
            "--module",
            module_name,
            "--recursive",
            "--include-private",
            "--output-dir",
            str(output_dir),
        ]
    )
    stub_package = output_dir / module_name.rsplit(".", 1)[-1]
    if not (stub_package / "__init__.pyi").is_file():
        raise RuntimeError(f"Missing recursive native stubs for {module_name}")
    for stub in stub_package.rglob("*.pyi"):
        ast.parse(stub.read_text(), filename=str(stub))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", required=True)
    parser.add_argument("--extension", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args()
    generate_native_stubs(arguments.module, arguments.extension, arguments.output_dir)
