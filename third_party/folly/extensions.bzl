"""
Bzlmod extensions for Folly dependency.
Defines module_extension to expose folly_repo mapping.
"""

load("//third_party/folly:repositories.bzl", "folly_repo")

def _folly_dep_impl(_ctx):
    folly_repo()

folly_dep = module_extension(
    implementation = _folly_dep_impl,
)
