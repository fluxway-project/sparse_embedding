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
  - Generates PEP 561 typing markers (`py.typed`) and Nanobind type stubs (`_core.pyi`).
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
pip wheel . --no-build-isolation
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

## 4. Packaging Invariants

All builds enforce the following packaging standards:

1. **Clean Distribution Tree**: Namespace package discovery in `setup.py` strictly excludes `sparse_embedding.tests`, preventing test fixtures and harness code from leaking into release wheels.
2. **Stale Artifact Invalidation**: `setup.py` cleans destination directories and purges obsolete `.so` files before copying new Bazel build outputs, preventing ABI mismatches across configuration switches.
3. **Type Safety & Metadata**: The package bundles `py.typed` alongside generated Nanobind `.pyi` stubs, providing full static typing support for downstream consumers.
