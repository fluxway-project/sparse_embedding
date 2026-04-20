"""
Bzlmod extensions for stdexec dependencies.
Defines module_extension to expose stdexec_repo.
"""

load("//third_party/stdexec:repositories.bzl", "stdexec_repo")

def _stdexec_dep_impl(_ctx):
    stdexec_repo()

stdexec_dep = module_extension(
    implementation = _stdexec_dep_impl,
)
