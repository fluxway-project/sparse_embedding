"""Proxy: Next Generation Polymorphism in C++."""

load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

def proxy_repo():
    new_git_repository(
        name = "proxy",
        remote = "https://github.com/ngcpp/proxy.git",
        commit = "1c7ca609aba6d73df06183c14db9d2d210399af3",
        build_file = "//third_party/proxy:BUILD.bazel",
    )
