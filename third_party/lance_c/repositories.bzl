"""Fetch lance-c"""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def lance_c_repo():
    http_archive(
        name = "lance_c",
        urls = ["https://github.com/lance-format/lance-c/archive/refs/heads/main.tar.gz"],
        strip_prefix = "lance-c-main",
        build_file = "//third_party/lance_c:BUILD.lance_c.bazel",
        patch_cmds = [
            "find . -path ./BUILD.bazel -prune -o -name BUILD -exec rm {} + -o -name BUILD.bazel -exec rm {} +",
        ],
    )
