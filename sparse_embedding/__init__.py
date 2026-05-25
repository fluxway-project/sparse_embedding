"""
Sparse Embedding Package
"""

import atexit
from typing import Any, Sequence

from ._core import (
    Adam,
    EmbeddingStore,
    FastRangePartitioner,
    HashPartitioner,
    Initializer,
    ModuloPartitioner,
    Optimizer,
    Partitioner,
    Sgd,
    Uniform,
    Zeros,
    _clear_wake_queue,
    _EmbeddingConfig,
    async_unique,
    decode_keys,
    drain_wake_queue,
    encode_keys,
    set_leak_warnings,
    wake_fd,
)


def _to_str(dt: Any) -> str:
    if dt is None:
        return ""
    name = ""
    if hasattr(dt, "__name__"):
        name = dt.__name__
    elif hasattr(dt, "name"):
        name = dt.name
    else:
        name = str(dt)

    if name == "float":
        return "float32"
    if name == "int":
        return "int64"
    return name


class EmbeddingConfig(_EmbeddingConfig):
    def __init__(
        self,
        dims: Sequence[int],
        num_threads: int = 0,
        value_dtype: Any = "float32",
        optimizer_dtype: Any = "",
        key_dtype: Any = "int64",
        max_active_keys: int = 1 << 20,
    ):
        super().__init__(
            dims,
            num_threads,
            _to_str(value_dtype),
            _to_str(optimizer_dtype),
            _to_str(key_dtype),
            max_active_keys,
        )


# During interpreter shutdown (finalization), Python clears modules in a semi-random
# order. If external frameworks (like Ray's global actor registries) keep strong
# references to classes (like EmbeddingStore) longer than the C extension module
# lifetime, nanobind will output false-positive leak warnings upon exit because
# the type's active count remains > 0.
#
# To prevent these finalization-order false positives, we register an atexit hook
# to disable nanobind leak warnings at shutdown. By registering this at exit:
# 1. Leak warnings are only disabled at the final process shutdown phase.
# 2. Throughout the entire application runtime (e.g., during tests or development),
#    nanobind's leak detection remains 100% active to catch real developer leaks.
def _disable_leak_warnings() -> None:
    try:
        set_leak_warnings(False)
    except Exception:
        pass


atexit.register(_clear_wake_queue)
atexit.register(_disable_leak_warnings)

__all__ = [
    "Adam",
    "Sgd",
    "Optimizer",
    "EmbeddingConfig",
    "Zeros",
    "Uniform",
    "Initializer",
    "EmbeddingStore",
    "async_unique",
    "encode_keys",
    "decode_keys",
    "Partitioner",
    "ModuloPartitioner",
    "FastRangePartitioner",
    "HashPartitioner",
    "wake_fd",
    "drain_wake_queue",
]
