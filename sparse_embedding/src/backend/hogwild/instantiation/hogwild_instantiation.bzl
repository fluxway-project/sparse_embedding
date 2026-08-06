def hogwild_instantiation_library(name, src):
    native.cc_library(
        name = name,
        srcs = [
            src,
            "instantiation/hogwild_explicit_instantiation.hpp",
        ],
        deps = [":hogwild_headers"],
    )
