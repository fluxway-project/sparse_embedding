# Building and Testing Guide

This document describes the build architecture, packaging contract, and verification workflows for `sparse_embedding`.

---

## 1. Architecture and Build Contract

`sparse_embedding` adopts a two-tier hybrid build architecture:

```
[pip / build / setuptools]
       │
       ▼ (PEP 517 build frontend)
    setup.py ──────────────────► Package assembly, stub generation, namespace discovery
       │
       ▼ (Subprocess invocation)
  Bazel System ────────────────► C++23 compilation, Nanobind bindings, native optimizations
```

- **Bazel as Native Engine**:
  - Compiles the high-performance C++23 backend (`//:core_extension`), SIMD optimizers, Hogwild storage engine, and DLPack zero-copy bridges.
  - Owns native test execution (`bazel test //sparse_embedding/...`).
  - Exports package shared libraries via the unified filegroup `//:native_artifacts`.
- **Setuptools as Distribution Owner**:
  - Serves as the standard PEP 517 build backend for `pip install`, editable installs, and binary wheel creation.
  - Discovers compiled artifacts dynamically by querying `//:native_artifacts` from Bazel, ensuring exact alignment with the active build configuration.
  - Bundles the PEP 561 marker (`py.typed`) and generates recursive Nanobind stubs (`_core/__init__.pyi` and `_core/checkpoint.pyi`).
  - Strictly isolates test suites from redistributable packages.

---

## 2. Feature Flags and Configuration

Lance checkpointing integration is controlled by a single feature switch across both build layers:

| Configuration | Bazel Flag | Environment Variable (pip) | Bundled Native Libraries |
| :--- | :--- | :--- | :--- |
| **Lance Enabled** *(Default)* | `--//:with_lance=true` | `SE_BUILD_WITH_LANCE=1` | `_core.so`, `liblance_c.so` |
| **Standalone Core** | `--//:with_lance=false` | `SE_BUILD_WITH_LANCE=0` | `_core.so` |

---

## 3. Standard Workflows and Commands

### Python Installation and Packaging

```bash
# Standard installation from source
pip install .

# Editable install for local development
pip install -e .

# Standalone build (disable Lance integration)
SE_BUILD_WITH_LANCE=0 pip install -e .

# Build redistributable binary wheel
pip wheel .
```

### Running Tests

```bash
# Install test dependencies
pip install -e ".[test]"

# Run full Python test suite
pytest

# Run non-slow unit tests
pytest -m "not slow"
```

### Native C++ Building and Testing via Bazel

```bash
# Build all C++ targets with optimizations
bazel build -c opt //sparse_embedding/...

# Run all C++ unit and integration tests
bazel test -c opt //sparse_embedding/...
```

---

## 4. Source Formatting

Local formatting and CI use the same entry point:

```bash
bash scripts/format.sh
bash scripts/format.sh --check
```

The script selects tracked and non-ignored untracked source files using Git. Build outputs, temporary wheel environments, and ignored dependencies are excluded. It runs clang-format 16 with the repository's LLVM style, buildifier 10.1.0, and Ruff 0.14.6. Ruff sorts imports and formats Python files. Check mode reports differences without rewriting source files. CI pins these tool versions in `.github/workflows/ci.yml`.

---

## 5. Packaging Invariants

All builds enforce the following packaging standards:

1. **Clean release wheels**: `setup.py` excludes `sparse_embedding.tests` so test code and fixtures are not shipped in distribution wheels.
2. **Isolated wheel builds**: Wheel builds stage and compile C++ libraries (`.so`), type stubs, and version metadata inside a dedicated build directory without modifying or polluting the source tree. Editable installs (`pip install -e .`) continue to support in-place development builds.
3. **First-class C++ type support**: Release packages bundle `py.typed` and `.pyi` stub files for `_core`. CI automatically verifies stub validity and wheel integrity, ensuring IDE autocomplete and static type checkers work out of the box for downstream users.
