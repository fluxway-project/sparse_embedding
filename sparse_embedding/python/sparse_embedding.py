# coding=utf-8

"""
High-level sparse embedding wrapper on top of the JaxKVStore.

- 支持单机（非 SPMD）前向 lookup；
- 支持在 `jax.pmap` / SPMD 环境下，通过 `all_to_all` 做简单的分片前向。

当前仍假设同一 batch 内所有 key 都属于同一个 slot（相同 embedding 维度）。
反向 / 参数更新仍需根据具体训练流程设计。
"""

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np

from .kv_store import JaxKVStore


class SparseEmbedding:
    def __init__(self, feature_metas, default_values, num_shards: int = 256):
        """
        Args:
            feature_metas: slot_id -> embedding_dim
            default_values: slot_id -> default embedding (list / 1D ndarray)
            num_shards: passed through to underlying JaxKVStore
        """
        self.feature_metas = feature_metas
        self.default_values = default_values
        self.num_shards = num_shards

        # 主 embedding 存储在 C++ KVStore 中
        self.embedding_store = JaxKVStore(feature_metas, default_values, num_shards)

        # 为 Adam 优化器准备的 m / v KVStore，默认值为全 0
        zero_defaults = {
            slot: [0.0] * dim for slot, dim in feature_metas.items()
        }
        self.m_store = JaxKVStore(feature_metas, zero_defaults, num_shards)
        self.v_store = JaxKVStore(feature_metas, zero_defaults, num_shards)

        # Adam 超参数（逐 key、逐向量应用）
        self.learning_rate = 1e-3
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self._step = 0  # 全局步数，用于偏差校正

    def forward(self, keys, axis_name: str | None = None):
        """
        前向：给定一批 key（fid），返回对应 embedding。

        - 非 SPMD（axis_name=None）：在单机/单设备上做 unique + lookup；
        - SPMD（axis_name 不为 None，需在 jax.pmap 内调用）：
          - 对 keys 做一次 all_to_all，使 key 在参与设备间重新分布；
          - 在本地设备上 unique + lookup；
          - 将 embedding reshape 回分片形状，再做一次 all_to_all 还原到原始设备。

        注意：这是一个简化版的 SPMD 实现，假设 batch 维度在 axis 0 上分片，
        且所有设备共享同一个底层 KVStore 视图。
        """
        keys = jnp.asarray(keys, dtype=jnp.uint64)

        # 非 SPMD 情况：直接在当前设备上查表
        if axis_name is None:
            unique_keys, inverse = jnp.unique(keys, return_inverse=True)
            embeddings_unique = self.embedding_store.lookup_or_insert(unique_keys)
            embeddings = embeddings_unique[inverse]
            return embeddings

        # SPMD 情况：需在 pmap 中调用，并传入 axis_name，例如：
        #   @jax.pmap(axis_name="devices")
        #   def step(keys):
        #       return emb_layer.forward(keys, axis_name="devices")

        # 第一次 all_to_all：将各设备上的 keys 重新分布
        # 假设 keys 的 batch 维度在 axis=0 上
        keys_sharded = lax.all_to_all(keys, axis_name=axis_name, split_axis=0, concat_axis=0)

        # 在当前设备本地做 unique + lookup
        flat_keys = keys_sharded.reshape(-1)
        unique_keys, inverse = jnp.unique(flat_keys, return_inverse=True)

        embeddings_unique = self.embedding_store.lookup_or_insert(unique_keys)

        # 还原为本地分片形状
        flat_embeddings = embeddings_unique[inverse]
        emb_dim = flat_embeddings.shape[-1]
        embeddings_sharded = flat_embeddings.reshape(keys_sharded.shape + (emb_dim,))

        # 第二次 all_to_all：将 embedding 还原回原始设备分布
        embeddings = lax.all_to_all(
            embeddings_sharded,
            axis_name=axis_name,
            split_axis=0,
            concat_axis=0,
        )
        return embeddings

    def backward(self, keys, grad, axis_name: str | None = None):
        """
        反向：对当前 batch 的 embedding 计算梯度，并在 Python 端用 Adam 更新 KVStore。

        - 非 SPMD（axis_name=None）：在当前设备上聚合梯度并更新；
        - SPMD（axis_name 不为 None）：
          - 使用 all_to_all 收集所有设备上的 keys 和 grads；
          - 跨设备聚合相同 key 的梯度（sum）；
          - 在 device 0 上统一更新 KVStore（或每个设备更新自己负责的 shard）。

        注意：SPMD 模式下，keys 和 grad 需要在各设备间通过 all_to_all 通信。
        """
        keys = jnp.asarray(keys, dtype=jnp.uint64)
        grad = jnp.asarray(grad, dtype=jnp.float32)

        # SPMD 情况：需要跨设备聚合梯度
        if axis_name is not None:
            # 方案1：使用 all_gather 收集所有设备上的 keys 和 grads
            # 然后聚合相同 key 的梯度
            keys_gathered = lax.all_gather(keys, axis_name=axis_name, axis=0, tiled=True)
            grad_gathered = lax.all_gather(grad, axis_name=axis_name, axis=0, tiled=True)

            # 展平以便处理
            flat_keys = keys_gathered.reshape(-1)
            flat_grad = grad_gathered.reshape(-1, grad.shape[-1])

            # 转为 NumPy 进行聚合和更新
            keys_np = np.asarray(flat_keys, dtype=np.uint64)
            grad_np = np.asarray(flat_grad, dtype=np.float32)

            # 计算去重 key 及其索引映射
            unique_keys, inverse = np.unique(keys_np, return_inverse=True)

            # 将同一 key 的梯度累加（跨设备聚合）
            grad_accum = np.zeros((unique_keys.shape[0], grad_np.shape[1]), dtype=np.float32)
            np.add.at(grad_accum, inverse, grad_np)

            # 只在 device 0 上更新 KVStore（避免重复更新）
            # 注意：这要求所有设备共享同一个 KVStore 实例（通过进程间共享或参数服务器）
            device_idx = jax.lax.axis_index(axis_name)
            if device_idx == 0:
                self._update_kvstore_with_adam(unique_keys, grad_accum)
            
            # 可选：如果需要所有设备都更新，可以按 shard 分配：
            # 每个设备只更新自己负责的 shard 的 keys
            # 但这需要更复杂的逻辑来确定 key 属于哪个 shard
        else:
            # 非 SPMD 情况：直接在当前设备上处理
            keys_np = np.asarray(keys, dtype=np.uint64).reshape(-1)
            grad_np = np.asarray(grad, dtype=np.float32)
            grad_np = grad_np.reshape(keys_np.shape[0], -1)  # [B, D]

            # 计算去重 key 及其在原 batch 中的索引映射
            unique_keys, inverse = np.unique(keys_np, return_inverse=True)

            # 将同一 key 的梯度累加（sum 聚合，可根据需要改为 mean）
            grad_accum = np.zeros((unique_keys.shape[0], grad_np.shape[1]), dtype=np.float32)
            np.add.at(grad_accum, inverse, grad_np)

            # 更新 KVStore
            self._update_kvstore_with_adam(unique_keys, grad_accum)

        # 返回梯度，方便上游链路继续使用（可按需修改）
        return grad

    def _update_kvstore_with_adam(self, unique_keys: np.ndarray, grad_accum: np.ndarray):
        """
        内部方法：使用 Adam 更新 KVStore 中的 embedding。

        Args:
            unique_keys: [K] 去重后的 keys
            grad_accum: [K, D] 聚合后的梯度
        """

        # 从 KVStore 取出当前的 embedding / m / v（如果不存在，则用默认值插入）
        w_list = self.embedding_store.lookup_or_insert(unique_keys)
        m_list = self.m_store.lookup_or_insert(unique_keys)
        v_list = self.v_store.lookup_or_insert(unique_keys)

        # 将列表转为可写的 NumPy 数组
        w_arr = [np.asarray(w, dtype=np.float32) for w in w_list]
        m_arr = [np.asarray(m, dtype=np.float32) for m in m_list]
        v_arr = [np.asarray(v, dtype=np.float32) for v in v_list]

        # Adam 更新
        self._step += 1
        t = self._step
        lr = self.learning_rate
        b1, b2, eps = self.beta1, self.beta2, self.eps

        for i, g in enumerate(grad_accum):
            m = m_arr[i]
            v = v_arr[i]
            w = w_arr[i]

            # 一阶矩 / 二阶矩
            m = b1 * m + (1.0 - b1) * g
            v = b2 * v + (1.0 - b2) * (g * g)

            # 偏差修正
            m_hat = m / (1.0 - b1 ** t)
            v_hat = v / (1.0 - b2 ** t)

            # 参数更新
            w = w - lr * m_hat / (np.sqrt(v_hat) + eps)

            m_arr[i] = m
            v_arr[i] = v
            w_arr[i] = w

        # 写回 KVStore
        self.m_store.update(unique_keys, m_arr)
        self.v_store.update(unique_keys, v_arr)
        self.embedding_store.update(unique_keys, w_arr)
