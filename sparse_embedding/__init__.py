"""
Sparse Embedding Package
"""

import atexit

from ._core import (
    Adam,
    EmbeddingConfig,
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
    async_unique,
    decode_keys,
    drain_wake_queue,
    encode_keys,
    set_leak_warnings,
    wake_fd,
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
