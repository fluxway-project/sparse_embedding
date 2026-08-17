"""
Single Source of Truth for package and external dependency versions.

This file is valid Starlark (.bzl) and valid Python (.py), allowing it to be
directly loaded by Bazel rules (load("//:versions.bzl", ...)) and Python
scripts (setup.py, tests, runtime introspection) without duplication.
"""

PACKAGE_VERSION = "0.1.0"
PYTHON_REQUIRES = ">=3.12"

# Lance-C Shared Library
LANCE_C_VERSION = "0.1.6"
LANCE_C_COMMIT = "98e4d54fcc78eaa933f140eef5798c31095acf4b"
