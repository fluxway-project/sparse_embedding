"""Fetch lance-c"""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def lance_c_repo():
    http_archive(
        name = "lance_c",
        urls = ["https://github.com/lance-format/lance-c/archive/a821a016b6c282f85e835536d1b48337396a13eb.tar.gz"],
        sha256 = "f71cab7d302c25351392f079da25f739e975226521c6bdde9b36db0871c45221",
        strip_prefix = "lance-c-a821a016b6c282f85e835536d1b48337396a13eb",
        build_file = "//third_party/lance_c:BUILD.lance_c.bazel",
        patch_cmds = [
            "find . -path ./BUILD.bazel -prune -o -name BUILD -exec rm {} + -o -name BUILD.bazel -exec rm {} +",
        ],
    )
