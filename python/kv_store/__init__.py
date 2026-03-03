# coding=utf-8

import numpy as np
import jax
import jax.numpy as jnp

# try:
from . import _core  # _core.so/pyd from C++ build


def _fid_to_slot(fid: int) -> int:
    """Same as C++ fid_to_slot: high 16 bits of 64-bit fid."""
    return int((fid >> 48) & 0xFFFF)


class JaxKVStore:
    """
    A JAX-compatible Key-Value store backed by a high-performance C++ implementation.
    Wraps the C++ KVStore with per-slot feature metas and init_type.
    """
    def __init__(self, feature_metas: dict, init_type: int = 0, num_shards: int = 16):
        """
        Args:
            feature_metas: dict mapping slot_id (int) -> embedding_dim (int).
            init_type: 0 = 全 0 初始化；1 = 随机初始化（以 key 为 seed，可复现）。
            num_shards: number of shards for concurrent access.
        """
        self.feature_metas = feature_metas
        self.init_type = init_type
        self.num_shards = num_shards
        self._store = _core.KVStore(feature_metas, init_type, num_shards)

    def get_embedding_dim(self, fids):
        """Return embedding dimension for the given feature id (fid)."""
        return self._store.get_embedding_dim(fids)

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
        """Embedding dim for the batch: max of per-key dims, for ragged output shape."""
        if keys_np.size == 0:
            return next(iter(self.feature_metas.values()), 0)
        dims = self._store.get_embedding_dim(keys_np.flatten())
        return max(dims) if dims else 0

    def _ragged_to_array(self, values: np.ndarray, row_lengths: np.ndarray, default_dim: int):
        """将 RaggedTensors (values, row_lengths) 转为 2D 数组。缺失 key 用零向量填充。"""
        n = len(row_lengths)
        if n == 0:
            return np.zeros((0, default_dim), dtype=np.float32)
        max_len = int(row_lengths.max()) if row_lengths.max() > 0 else default_dim
        max_len = max(max_len, 1)
        out = np.zeros((n, max_len), dtype=np.float32)
        offset = 0
        for i in range(n):
            L = int(row_lengths[i])
            if L > 0:
                out[i, :L] = values[offset : offset + L]
                offset += L
        return out

    def lookup(self, keys):
        """
        Batch lookup. Accepts NumPy or JAX arrays. Missing keys get zero vector (first key's slot dim).
        With JAX arrays, the call is JIT-compilable.
        """
        keys_np = np.asarray(keys, dtype=np.uint64)
        dim = self._embedding_dim_for_keys(keys_np)

        def _do_lookup(k_np):
            values, row_lengths = self._store.lookup(k_np)
            return self._ragged_to_array(np.asarray(values), np.asarray(row_lengths), dim)

        if not hasattr(keys, 'block_until_ready'):
            return _do_lookup(keys_np).astype(np.float32)

        keys = jnp.asarray(keys, dtype=jnp.uint64)

        def lookup_callback(k):
            return _do_lookup(np.asarray(k))

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
        dim = self._embedding_dim_for_keys(keys_np)

        def _do_lookup_or_insert(k_np):
            values, row_lengths = self._store.lookup_or_insert(k_np)
            return self._ragged_to_array(np.asarray(values), np.asarray(row_lengths), dim)

        if not hasattr(keys, 'block_until_ready'):
            return _do_lookup_or_insert(keys_np).astype(np.float32)

        keys = jnp.asarray(keys, dtype=jnp.uint64)

        def lookup_or_insert_callback(k):
            return _do_lookup_or_insert(np.asarray(k))

        result_shape_dtype = jax.ShapeDtypeStruct(
            shape=(keys_np.shape[0], dim),
            dtype=jnp.float32,
        )
        return jax.pure_callback(lookup_or_insert_callback, result_shape_dtype, keys, vectorized=False)
