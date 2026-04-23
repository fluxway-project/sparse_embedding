"""
Provides stdexec repository downloading rules.
Uses http_archive to fetch NVIDIA stdexec from GitHub.
"""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def stdexec_repo():
    http_archive(
        name = "stdexec",
        strip_prefix = "stdexec-931f256c3af0cdd2a3b49c04892c82d7945dc5a7",
        url = "https://github.com/NVIDIA/stdexec/archive/931f256c3af0cdd2a3b49c04892c82d7945dc5a7.zip",
        build_file = "//third_party/stdexec:BUILD.bazel",
    )
