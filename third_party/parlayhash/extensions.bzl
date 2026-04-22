"""ParlayHash extension."""

load("//third_party/parlayhash:repositories.bzl", "parlayhash_repo")

def _parlayhash_impl(_ctx):
    parlayhash_repo()

parlayhash_dep = module_extension(
    implementation = _parlayhash_impl,
)
