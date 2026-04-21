"""Fetch tsl::robin_map"""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def robin_map_repo():
    http_archive(
        name = "robin_map",
        urls = ["https://github.com/Tessil/robin-map/archive/refs/tags/v1.3.0.tar.gz"],
        strip_prefix = "robin-map-1.3.0",
        build_file = "//third_party/robin_map:BUILD.bazel",
        sha256 = "a8424ad3b0affd4c57ed26f0f3d8a29604f0e1f2ef2089f497f614b1c94c7236",
    )
