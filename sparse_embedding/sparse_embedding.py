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

# from mpi4py import MPI

from kv_store import JaxKVStore


class SparseEmbedding:
    def __init__(self, feature_metas, init_type: int = 1, num_shards: int = 256):
        """
        Args:
            feature_metas: slot_id -> embedding_dim
            init_type: 0 = 全 0 初始化；1 = 随机初始化（以 key 为 seed）
            num_shards: passed through to underlying JaxKVStore
        """
        self.feature_metas = feature_metas
        self.init_type = init_type
        self.num_shards = num_shards

        # 主 embedding 存储在 C++ KVStore 中（init_type 可为 0 或 1）
        self.embedding_store = JaxKVStore(feature_metas, init_type, num_shards)

        # 为 Adam 优化器准备的 m / v KVStore，必须全 0 初始化
        self.m_store = JaxKVStore(feature_metas, init_type=0, num_shards=num_shards)
        self.v_store = JaxKVStore(feature_metas, init_type=0, num_shards=num_shards)

        # Adam 超参数（逐 key、逐向量应用）
        self.learning_rate = 1e-3
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self._step = 0  # 全局步数，用于偏差校正

        self.world_size = jax.process_count()

        self.cache = {}

    def forward(
        self,
        keys: dict[str, jnp.ndarray],
        pooling: str = "sum",
        axis_name: str | None = None,
    ):
        """
        前向：输入 keys（每个 key embedding dim 不同），返回 ragged embedding：
        (flat_embeddings, row_lengths)
        """

        slot_names = list(keys.keys())
        slot_lengths = [v.shape[0] for v in keys.values()]

        keys = jnp.concatenate(
            [jnp.asarray(v, dtype=jnp.uint64) for v in keys.values()], axis=0
        )

        # ---- Step 1:  flatten keys and compute target devices ----
        flat_keys = keys.reshape(-1)  # [N]
        target_devices = flat_keys % self.world_size  # routing rule

        # ---- Step 2: sort by target device ----
        idx = jnp.argsort(target_devices)
        sorted_keys = flat_keys[idx]
        sorted_targets = target_devices[idx]

        # ---- Step 3: build ragged send buffer ----
        send_counts = jnp.bincount(sorted_targets, length=self.world_size).astype(
            jnp.int32
        )
        input_offsets = jnp.cumsum(send_counts) - send_counts

        # ---- Step 4: exchange counts by all_to_all ----
        # lax.all_to_all requires (1, world_size) tensor
        send_counts_2d = send_counts[jnp.newaxis, :]  # (1, world_size)
        recv_counts_2d = lax.all_to_all(
            send_counts_2d, axis_name=axis_name, split_axis=1, concat_axis=1
        )
        recv_counts = recv_counts_2d[0]  # (world_size,)
        output_offsets = jnp.cumsum(recv_counts) - recv_counts
        total_recv = recv_counts.sum()

        recv_keys = jnp.zeros((total_recv,), dtype=jnp.uint64)

        # ---- Step 5: ragged all_to_all to get keys for this rank ----
        local_keys = lax.ragged_all_to_all(
            sorted_keys,
            recv_keys,
            input_offsets,
            send_counts,
            output_offsets,
            recv_counts,
            axis_name=axis_name,
        )
        # local_keys = [local_num_keys]

        # ---- Step 6: unique + lookup (ragged) ----
        unique_keys, inverse = jnp.unique(local_keys, return_inverse=True)

        # embedding_store.lookup_or_insert returns:
        #   flat_vals: concat of all embeddings
        #   row_lengths: per-key dim
        flat_vals, row_lengths = self.embedding_store.lookup_or_insert(unique_keys)
        # row_lengths: [num_unique_keys]

        # ---- Step 7: reconstruct ragged embedding for each received key ----
        # Expand ragged lookup to per-key embedding
        # first compute per-unique key offsets
        key_offsets = jnp.cumsum(row_lengths) - row_lengths  # [K]

        # Flatten local lookup result:
        # total_emb = flat_vals.shape[0]

        # For inverse mapping: get start/end offsets for each local key
        emb_start = key_offsets[inverse]
        emb_end = emb_start + row_lengths[inverse]

        # Build output flat array
        out_total = emb_end.sum() - emb_start.sum()  # same as total_len
        flat_output = jnp.zeros((flat_vals.dtype,))

        # More efficient: repeat ranges and index
        # build per-key ranges by "segmented gather"
        gather_index = jnp.concatenate(
            [jnp.arange(s, e) for (s, e) in zip(emb_start, emb_end)]
        )
        flat_output = flat_vals[gather_index]

        # per-key output row lengths
        out_row_lengths = row_lengths[inverse]  # ragged dim for each local key

        # ---- Step 8: Now we need to all_to_all ragged back to original owners ----

        # Return send_counts (embedding) = out_row_lengths
        send_counts2 = out_row_lengths
        input_offsets2 = jnp.cumsum(send_counts2) - send_counts2

        # Step 8.1: exchange recv_counts2
        send_counts2_2d = send_counts2[jnp.newaxis, :]
        recv_counts2_2d = lax.all_to_all(
            send_counts2_2d, axis_name=axis_name, split_axis=1, concat_axis=1
        )
        recv_counts2 = recv_counts2_2d[0]
        output_offsets2 = jnp.cumsum(recv_counts2) - recv_counts2
        total_recv2 = recv_counts2.sum()

        recv_embed = jnp.zeros((total_recv2,), dtype=flat_output.dtype)

        # ---- Step 8.2 ragged all_to_all embeddings ----
        final_output = lax.ragged_all_to_all(
            flat_output,
            recv_embed,
            input_offsets2,
            send_counts2,
            output_offsets2,
            recv_counts2,
            axis_name=axis_name,
        )

        # ---- Step 9: pooling and return python lis ----
        outputs = {}
        offset = 0

        # recv_counts2 = 每个 key 的 dim

        for i, slot in enumerate(slot_names):
            dim = self.feature_metas[slot]
            num_keys = slot_lengths[i]  # 这个 slot 的 key 数

            emb = final_output[offset : offset + num_keys * dim].reshape(-1, dim)
            offset += num_keys * dim

            # ===== pooling =====
            if pooling == "sum":
                pooled = jnp.sum(emb, axis=-1)
            elif pooling == "mean":
                pooled = jnp.mean(emb, axis=-1)
            elif pooling == "max":
                pooled = jnp.max(emb, axis=-1)
            else:
                raise ValueError(f"Unsupported pooling: {pooling}")

            outputs[slot] = pooled  # shape = [dim]

        return outputs

    def backward(
        self,
        keys: dict[str, jnp.ndarray],
        grad: dict[str, jnp.ndarray],
        pooling: str = "sum",
        axis_name: str | None = None,
    ):
        """
        backward 与 forward 完整对偶，grad 格式与 forward_outputs 对应：

        grad = {
            "slot1": [num_keys1],       (pooled grad)
            "slot2": [num_keys2],
            ...
        }
        """

        # ---- Step A: unpooling ----
        # 将每个 slot 的 pooling grad 还原到每个 key 的 full dim grad
        per_key_grads = []
        per_key_keys = []

        for slot, g in grad.items():
            dim = self.feature_metas[slot]
            g = jnp.asarray(g, dtype=jnp.float32)  # shape [num_keys]
            num_keys = g.shape[0]

            if pooling == "sum":
                # 每个 dim 同一个梯度值
                full = jnp.repeat(g[:, None], dim, axis=-1)

            elif pooling == "mean":
                full = jnp.repeat(g[:, None], dim, axis=-1) / dim

            elif pooling == "max":
                # ❗ 需要 forward 的 max mask，但你 forward 中没返回 mask
                # 暂时按 sum 处理，否则需要 forward 存 mask
                full = jnp.repeat(g[:, None], dim, axis=-1)

            else:
                raise ValueError(f"Unsupported pooling: {pooling}")

            per_key_grads.append(full.reshape(-1, dim))
            per_key_keys.append(keys[slot].reshape(-1))

        # ---- Step B: concat all slots (保持 forward 顺序) ----
        flat_keys = jnp.concatenate(per_key_keys, axis=0)  # [N]
        flat_grad = jnp.concatenate(per_key_grads, axis=0)  # [N, dim]

        # ---- Step C: 重新路由到 owner rank (ragged all_to_all) ----

        # routing rule
        target_devices = flat_keys % self.world_size

        # sort
        idx = jnp.argsort(target_devices)
        sorted_keys = flat_keys[idx]
        sorted_grad = flat_grad[idx]
        sorted_targets = target_devices[idx]

        # send counts
        send_counts = jnp.bincount(sorted_targets, length=self.world_size).astype(
            jnp.int32
        )
        input_offsets = jnp.cumsum(send_counts) - send_counts

        # exchange recv counts
        send_counts_2d = send_counts[jnp.newaxis, :]
        recv_counts_2d = lax.all_to_all(
            send_counts_2d,
            axis_name=axis_name,
            split_axis=1,
            concat_axis=1,
        )
        recv_counts = recv_counts_2d[0]
        output_offsets = jnp.cumsum(recv_counts) - recv_counts
        total_recv = recv_counts.sum()

        # ragged all_to_all for keys
        recv_keys = jnp.zeros((total_recv,), dtype=jnp.uint64)
        local_keys = lax.ragged_all_to_all(
            sorted_keys,
            recv_keys,
            input_offsets,
            send_counts,
            output_offsets,
            recv_counts,
            axis_name=axis_name,
        )

        # ragged all_to_all for grads (flat second axis)
        embed_dim = flat_grad.shape[-1]
        grad_flat_send = sorted_grad.reshape(-1)
        recv_grad_flat = jnp.zeros(
            (total_recv * embed_dim,), dtype=grad_flat_send.dtype
        )

        local_grad_flat = lax.ragged_all_to_all(
            grad_flat_send,
            recv_grad_flat,
            input_offsets * embed_dim,
            send_counts * embed_dim,
            output_offsets * embed_dim,
            recv_counts * embed_dim,
            axis_name=axis_name,
        )
        local_grad = local_grad_flat.reshape(total_recv, embed_dim)

        # ---- Step D: 聚合重复 key 的梯度 ----
        local_keys_np = np.asarray(local_keys, dtype=np.uint64)
        local_grad_np = np.asarray(local_grad, dtype=np.float32)

        unique_keys, inverse = np.unique(local_keys_np, return_inverse=True)
        grad_accum = np.zeros((unique_keys.shape[0], embed_dim), dtype=np.float32)
        np.add.at(grad_accum, inverse, local_grad_np)

        # ---- Step E: 更新 KVStore ----
        # 每个 rank 只更新自己 shard 的 key
        # （因为 routing rule = key % world_size）
        self._update_kvstore_with_adam(unique_keys, grad_accum)

        return

    def _update_kvstore_with_adam(
        self, unique_keys: np.ndarray, grad_accum: np.ndarray
    ):
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
            m_hat = m / (1.0 - b1**t)
            v_hat = v / (1.0 - b2**t)

            # 参数更新
            w = w - lr * m_hat / (np.sqrt(v_hat) + eps)

            m_arr[i] = m
            v_arr[i] = v
            w_arr[i] = w

        # 写回 KVStore
        self.m_store.update(unique_keys, m_arr)
        self.v_store.update(unique_keys, v_arr)
        self.embedding_store.update(unique_keys, w_arr)
