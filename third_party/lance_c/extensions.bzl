"""Lance-C extension."""

load("//third_party/lance_c:repositories.bzl", "lance_c_repo")

def _lance_c_impl(ctx):
    lance_c_repo(ctx)

lance_c_dep = module_extension(
    implementation = _lance_c_impl,
    environ = ["SE_BUILD_LANCE_C_FROM_SOURCE"],
)
