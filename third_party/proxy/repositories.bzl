"""Proxy: Next Generation Polymorphism in C++."""

load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

def proxy_repo():
    new_git_repository(
        name = "proxy",
        remote = "https://github.com/ngcpp/proxy.git",
        commit = "f188f54f9ef3abfdeb8de5bc36d35eb046d2386f",
        build_file = "//third_party/proxy:BUILD.bazel",
    )
