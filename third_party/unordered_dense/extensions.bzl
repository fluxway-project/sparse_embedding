"""Unordered dense module extension."""

load("//third_party/unordered_dense:repositories.bzl", "unordered_dense_repo")

def _unordered_dense_impl(_ctx):
    unordered_dense_repo()

unordered_dense_dep = module_extension(
    implementation = _unordered_dense_impl,
)
