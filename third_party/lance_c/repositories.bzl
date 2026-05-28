"""Fetch lance-c"""

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def lance_c_repo(ctx):
    build_from_source = ctx.os.environ.get("SE_BUILD_LANCE_C_FROM_SOURCE", "0") == "1"

    if build_from_source:
        git_repository(
            name = "lance_c",
            remote = "https://github.com/lance-format/lance-c.git",
            commit = "c801dfe5cb4498ce9a1c04834b77d374afb680d9",
            build_file = "//third_party/lance_c:BUILD.lance_c.bazel",
            patch_cmds = [
                "find . -path ./BUILD.bazel -prune -o -name BUILD -exec rm {} + -o -name BUILD.bazel -exec rm {} +",
            ],
        )
    else:
        os_name = ctx.os.name
        arch = ctx.os.arch

        target_os = "unknown-linux-gnu"
        if "mac" in os_name:
            target_os = "apple-darwin"

        target_arch = "x86_64"
        if arch in ["aarch64", "arm64"]:
            target_arch = "aarch64"

        url = "https://github.com/lance-format/lance-c/releases/download/v0.1.2/lance-c-v0.1.2-{}-{}.tar.xz".format(target_arch, target_os)

        build_file_content = """
filegroup(
    name = "build_lance_c",
    srcs = glob(["lib/liblance_c.so", "lib/liblance_c.dylib"], allow_empty=True),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "lance_c",
    srcs = glob(["lib/liblance_c.so", "lib/liblance_c.dylib"], allow_empty=True),
    hdrs = glob([
        "include/lance/*.h",
        "include/lance/*.hpp",
    ]),
    includes = ["include"],
    visibility = ["//visibility:public"],
)
"""
        http_archive(
            name = "lance_c",
            urls = [url],
            build_file_content = build_file_content,
        )
