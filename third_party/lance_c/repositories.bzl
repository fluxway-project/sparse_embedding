"""Fetch lance-c"""

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def lance_c_repo():
    git_repository(
        name = "lance_c",
        remote = "https://github.com/lance-format/lance-c.git",
        commit = "c801dfe5cb4498ce9a1c04834b77d374afb680d9",
        build_file = "//third_party/lance_c:BUILD.lance_c.bazel",
        patch_cmds = [
            "find . -path ./BUILD.bazel -prune -o -name BUILD -exec rm {} + -o -name BUILD.bazel -exec rm {} +",
        ],
    )
