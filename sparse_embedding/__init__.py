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
    wake_fd,
)

atexit.register(_clear_wake_queue)

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
