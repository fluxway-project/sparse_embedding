"""
Folly: Facebook's open-source C++ library.

Integration strategy:
  folly has no official Bazel support and a very deep CMake-based dependency
  tree.  new_git_repository fetches folly at a pinned tag so the source is
  reproducible, but because compiling folly from source requires dozens of
  generated config headers (produced by CMake), we still link against the
  system-installed shared library.

  The BUILD.bazel file exposes:
    @folly//:folly_concurrent_hash_map  -- only what the project uses
    @folly//:folly                      -- alias for full public surface

  Build machines must have libfolly-dev installed at the matching version:
    apt install libfolly-dev=<tag>   # or equivalent package manager
"""

load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

FOLLY_TAG = "v2026.04.20.00"

def folly_repo():
    new_git_repository(
        name = "folly",
        remote = "https://github.com/facebook/folly.git",
        tag = FOLLY_TAG,
        build_file = "//third_party/folly:BUILD.bazel",
        init_submodules = False,
        recursive_init_submodules = False,
    )
