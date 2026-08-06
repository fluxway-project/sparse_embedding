from collections.abc import Callable

from .base import CheckpointBackend, DistributedCheckpointBackend

CheckpointBackendFactory = Callable[..., CheckpointBackend]

_BACKENDS: dict[str, CheckpointBackendFactory] = {}


def register_backend(name: str, backend_factory: CheckpointBackendFactory) -> None:
    if not name:
        raise ValueError("checkpoint backend name must not be empty")
    if name in _BACKENDS:
        raise ValueError(f"Checkpoint backend already registered: {name}")
    _BACKENDS[name] = backend_factory


def get_checkpointer(name: str, **kwargs) -> CheckpointBackend:
    if name not in _BACKENDS:
        raise ValueError(
            f"Unknown checkpoint backend: {name}. "
            f"Available: {list(_BACKENDS.keys())}. "
            f"Ensure the optional dependency is installed (e.g. `pip install sparse_embedding[pylance]`)."
        )
    return _BACKENDS[name](**kwargs)


def _create_lance_backend(**kwargs) -> CheckpointBackend:
    from .lance_backend import LanceCheckpointBackend

    return LanceCheckpointBackend(**kwargs)


register_backend("lance", _create_lance_backend)

__all__ = [
    "CheckpointBackend",
    "DistributedCheckpointBackend",
    "get_checkpointer",
    "register_backend",
]
