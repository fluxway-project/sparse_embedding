# coding=utf-8

import numpy as np
# try:
from . import _core  # _core.so/pyd from C++ build
# except ImportError as e:
#     raise ImportError(
#         "kv_store C++ extension '_core' not found. Build it from sparse_embedding with:\n"
#         "  mkdir -p build && cd build\n"
#         "  cmake .. -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=\"../python/kv_store\" "
#         "-Dpybind11_DIR=\"$(python3 -c 'import pybind11; print(pybind11.get_cmake_dir())')\"\n"
#         "  cmake --build . -j4\n"
#         "Then ensure python/kv_store/ contains _core.cpython-*.so"
#     ) from e

def _ensure_jax():
    try:
        import jax
        import jax.numpy as jnp
        return jax, jnp
    except ImportError as e:
        raise ImportError("JAX is required for lookup/lookup_or_insert with JAX arrays. Install with: pip install jax jaxlib") from e


def _fid_to_slot(fid: int) -> int:
    """Same as C++ fid_to_slot: high 16 bits of 64-bit fid."""
    return int((fid >> 48) & 0xFFFF)


class JaxKVStore:
    """
    A JAX-compatible Key-Value store backed by a high-performance C++ implementation.
    Wraps the C++ KVStore with per-slot feature metas and default values.
    """
    def __init__(self, feature_metas: dict, default_values: dict, num_shards: int = 16):
        """
        Args:
            feature_metas: dict mapping slot_id (int) -> embedding_dim (int).
            default_values: dict mapping slot_id (int) -> default vector (list or 1D ndarray of float).
            num_shards: number of shards for concurrent access.
        """
        self.feature_metas = feature_metas
        self.default_values = default_values
        self.num_shards = num_shards
        self._store = _core.KVStore(feature_metas, default_values, num_shards)

    def get_embedding_dim(self, fid: int) -> int:
        """Return embedding dimension for the given feature id (fid)."""
        return self._store.get_embedding_dim(int(fid))

    def insert_one(self, key: int, value: np.ndarray):
        """Insert a single key-value pair. Prefer batch insert for performance."""
        keys = np.array([key], dtype=np.uint64)
        values = np.expand_dims(value, axis=0).astype(np.float32)
        self._store.insert(keys, values)

    def insert(self, keys, values):
        """
        Batch insert or update key-value pairs.
        Keys and values are converted to NumPy on CPU if needed (e.g. from JAX/GPU).
        """
        keys_np = np.asarray(keys, dtype=np.uint64)
        values_np = np.asarray(values, dtype=np.float32)
        if values_np.ndim == 1:
            values_np = np.expand_dims(values_np, axis=0)
        self._store.insert(keys_np, values_np)

    def update(self, keys, values):
        """Batch update existing keys. Keys that do not exist are ignored."""
        keys_np = np.asarray(keys, dtype=np.uint64)
        values_np = np.asarray(values, dtype=np.float32)
        if values_np.ndim == 1:
            values_np = np.expand_dims(values_np, axis=0)
        self._store.update(keys_np, values_np)

    def _embedding_dim_for_keys(self, keys_np: np.ndarray) -> int:
        """Embedding dim for the batch: from first key's slot, or first meta value if empty."""
        if keys_np.size == 0:
            return next(iter(self.feature_metas.values()), 0)
        return self._store.get_embedding_dim(int(keys_np.flat[0]))

    def lookup(self, keys):
        """
        Batch lookup. Accepts NumPy or JAX arrays. Missing keys get zero vector (first key's slot dim).
        With JAX arrays, the call is JIT-compilable.
        """
        keys_np = np.asarray(keys, dtype=np.uint64)
        if not hasattr(keys, 'block_until_ready'):  # NumPy/list path
            dim = self._embedding_dim_for_keys(keys_np)
            results = self._store.lookup(keys_np)
            default_value = np.zeros(dim, dtype=np.float32)
            return np.array(
                [np.array(res, dtype=np.float32) if res is not None else default_value for res in results],
                dtype=np.float32,
            )
        jax, jnp = _ensure_jax()
        keys = jnp.asarray(keys, dtype=jnp.uint64)

        def lookup_callback(k):
            k_np = np.asarray(k)
            dim = self._embedding_dim_for_keys(k_np)
            results = self._store.lookup(k_np)
            default_value = np.zeros(dim, dtype=np.float32)
            return np.array(
                [np.array(res, dtype=np.float32) if res is not None else default_value for res in results],
                dtype=np.float32,
            )

        dim = self._embedding_dim_for_keys(keys_np)
        result_shape_dtype = jax.ShapeDtypeStruct(
            shape=(keys_np.shape[0], dim),
            dtype=jnp.float32,
        )
        return jax.pure_callback(lookup_callback, result_shape_dtype, keys, vectorized=False)

    def lookup_or_insert(self, keys):
        """
        Batch lookup or insert with default. Accepts NumPy or JAX arrays.
        With JAX arrays, the call is JIT-compilable.
        """
        keys_np = np.asarray(keys, dtype=np.uint64)
        if not hasattr(keys, 'block_until_ready'):  # NumPy/list path
            results = self._store.lookup_or_insert(keys_np)
            return np.array(results, dtype=np.float32)
        jax, jnp = _ensure_jax()
        keys = jnp.asarray(keys, dtype=jnp.uint64)

        def lookup_or_insert_callback(k):
            k_np = np.asarray(k)
            results = self._store.lookup_or_insert(k_np)
            return np.array(results, dtype=np.float32)

        dim = self._embedding_dim_for_keys(keys_np)
        result_shape_dtype = jax.ShapeDtypeStruct(
            shape=(keys_np.shape[0], dim),
            dtype=jnp.float32,
        )
        return jax.pure_callback(lookup_or_insert_callback, result_shape_dtype, keys, vectorized=False)
