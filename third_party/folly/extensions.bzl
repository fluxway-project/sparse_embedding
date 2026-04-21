"""
Bzlmod extension for Folly dependency.
Fetches folly at a pinned weekly tag via new_git_repository.
"""

load("//third_party/folly:repositories.bzl", "folly_repo")

def _folly_dep_impl(_ctx):
    folly_repo()

folly_dep = module_extension(
    implementation = _folly_dep_impl,
)
