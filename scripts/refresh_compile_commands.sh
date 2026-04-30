#!/bin/bash
set -e

# When executed via `bazel run`, Bazel sets BUILD_WORKSPACE_DIRECTORY to the root of the workspace.
if [ -n "$BUILD_WORKSPACE_DIRECTORY" ]; then
    cd "$BUILD_WORKSPACE_DIRECTORY"
fi

# We delegate to the internal target, specifically injecting --build_python_zip
# to bypass the `rules_python` 1.9.0 build-runfiles space bug without polluting global .bazelrc.
exec bazel run --build_python_zip //:refresh_compile_commands_internal "$@"
