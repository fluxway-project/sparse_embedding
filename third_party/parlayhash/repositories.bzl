"""Fetch parlayhash"""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def parlayhash_repo():
    http_archive(
        name = "parlayhash",
        urls = ["https://github.com/cmuparlay/parlayhash/archive/081408ba6111ce78b5add9129f8d902fa2fea58f.tar.gz"],
        strip_prefix = "parlayhash-081408ba6111ce78b5add9129f8d902fa2fea58f",
        build_file = "//third_party/parlayhash:BUILD.bazel",
        sha256 = "26f9cd4f67ef82f5d50b2f3fe284eea0d826d759d9e15c9cbf7c76bab1c0a730",
        patch_cmds = ["find . -path ./BUILD.bazel -prune -o -name BUILD -exec rm {} + -o -name BUILD.bazel -exec rm {} +"],
        patches = ["//third_party/parlayhash:fix_get_index.patch"],
        patch_args = ["-p1"],
    )
