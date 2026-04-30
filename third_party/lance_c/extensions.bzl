"""Lance-C extension."""

load("//third_party/lance_c:repositories.bzl", "lance_c_repo")

def _lance_c_impl(_ctx):
    lance_c_repo()

lance_c_dep = module_extension(
    implementation = _lance_c_impl,
)
