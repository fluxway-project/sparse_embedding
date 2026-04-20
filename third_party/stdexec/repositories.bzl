"""
Provides stdexec repository downloading rules.
Uses http_archive to fetch NVIDIA stdexec from GitHub.
"""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def stdexec_repo():
    http_archive(
        name = "stdexec",
        strip_prefix = "stdexec-main",
        url = "https://github.com/NVIDIA/stdexec/archive/refs/heads/main.zip",
        build_file = "//third_party/stdexec:BUILD.bazel",
    )
