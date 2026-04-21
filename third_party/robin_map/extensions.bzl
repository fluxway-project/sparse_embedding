"""Robin Map extension."""

load("//third_party/robin_map:repositories.bzl", "robin_map_repo")

def _robin_map_impl(_ctx):
    robin_map_repo()

robin_map_dep = module_extension(
    implementation = _robin_map_impl,
)
