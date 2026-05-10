from .base import CheckpointBackend, DistributedCheckpointBackend

_BACKENDS = {}


def register_backend(name: str, backend_cls: type):
    _BACKENDS[name] = backend_cls


def get_backend(name: str, **kwargs) -> CheckpointBackend:
    if name not in _BACKENDS:
        raise ValueError(
            f"Unknown checkpoint backend: {name}. "
            f"Available: {list(_BACKENDS.keys())}. "
            f"Ensure the optional dependency is installed (e.g. `pip install sparse_embedding[pylance]`)."
        )
    return _BACKENDS[name](**kwargs)


try:
    from .lance_backend import LanceCheckpointBackend

    register_backend("lance", LanceCheckpointBackend)
except ImportError:
    pass

__all__ = [
    "CheckpointBackend",
    "DistributedCheckpointBackend",
    "get_backend",
    "register_backend",
]
