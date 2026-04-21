"""
Sparse Embedding Package
"""

from ._core import (
    Adam,
    Sgd,
    Optimizer,
    EmbeddingConfig,
    Zeros,
    Uniform,
    Initializer,
    EmbeddingStore,
    async_unique,
    wake_fd,
    drain_wake_queue,
)

import functools

_registered_loops = set()


def _ensure_asyncio_hook():
    """
    Transparently hook the C++ WakeManager's eventfd into the current running loop.
    This guarantees compatibility with standard asyncio, uvloop, or framework loops
    (like FastAPI/Sanic) without forcing the user to manually call setup functions.
    """
    import asyncio

    try:
        loop = asyncio.get_running_loop()
        loop_id = id(loop)
        if loop_id not in _registered_loops:
            loop.add_reader(wake_fd(), drain_wake_queue)
            _registered_loops.add(loop_id)
    except RuntimeError:
        pass


def _async_wrap(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        _ensure_asyncio_hook()
        return func(*args, **kwargs)

    return wrapper


# Automatically wrap C++ async methods on import
EmbeddingStore.async_lookup = _async_wrap(EmbeddingStore.async_lookup)
EmbeddingStore.async_assign = _async_wrap(EmbeddingStore.async_assign)

_raw_async_unique = async_unique


@functools.wraps(_raw_async_unique)
def _wrapped_async_unique(*args, **kwargs):
    _ensure_asyncio_hook()
    return _raw_async_unique(*args, **kwargs)


async_unique = _wrapped_async_unique

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
]
