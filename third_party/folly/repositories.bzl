"""
Provides Folly local repository definition.
Maps the system's Folly installation to a Bazel repository.
"""

load("@bazel_tools//tools/build_defs/repo:local.bzl", "new_local_repository")

def folly_repo():
    new_local_repository(
        name = "folly",
        path = "/usr",
        build_file_content = """
cc_library(
    name = "folly",
    hdrs = glob(["include/folly/**/*.h", "include/folly/**/*.hpp"], allow_empty = True) + glob(["local/include/folly/**/*.h", "local/include/folly/**/*.hpp"], allow_empty = True),
    includes = ["include", "local/include"],
    linkopts = ["-lfolly", "-lglog", "-lgflags"],
    visibility = ["//visibility:public"],
)
""",
    )
