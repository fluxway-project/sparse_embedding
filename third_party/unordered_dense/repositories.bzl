"""Fetch ankerl::unordered_dense"""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def unordered_dense_repo():
    http_archive(
        name = "unordered_dense",
        urls = ["https://github.com/martinus/unordered_dense/archive/refs/tags/v4.4.0.tar.gz"],
        strip_prefix = "unordered_dense-4.4.0",
        build_file = "//third_party/unordered_dense:BUILD.bazel",
        sha256 = "3976399793e8cb4db1409ce15610fbd9e5e406ced4745f262d393a9311efbd88",
    )
