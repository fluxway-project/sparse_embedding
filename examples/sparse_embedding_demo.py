# coding=utf-8
"""
Minimal training loop demo for SparseEmbedding.

注意：
- 这是一个单机 / 非 SPMD 的简化示例；
- 仅用于演示 forward + backward + Adam 更新 KVStore 内嵌入的流程。
"""

import numpy as np
import jax
import jax.numpy as jnp

from kv_store import JaxKVStore
from sparse_embedding import SparseEmbedding


def build_layer():
    # 两个 slot：121 -> dim=5，2673 -> dim=6；init_type=1 随机初始化
    feature_metas = {121: 5, 2673: 6}
    layer = SparseEmbedding(feature_metas, init_type=1, num_shards=256)
    return layer


def make_keys(batch_size: int, slot: int = 121, key_base: int = 0) -> np.ndarray:
    """
    生成一批假的 fid：
    fid = (slot << 48) | local_id
    """
    slot_prefix = np.uint64(slot) << np.uint64(48)
    local_ids = np.arange(key_base, key_base + batch_size, dtype=np.uint64)
    return (slot_prefix | local_ids).astype(np.uint64)


def main():
    layer = build_layer()

    # 目标：让 embedding 接近全 1 向量
    def loss_from_emb(e: jnp.ndarray) -> jnp.ndarray:
        target = jnp.ones_like(e)
        return jnp.mean((e - target) ** 2)

    loss_and_grad = jax.value_and_grad(loss_from_emb)

    batch_size = 4
    num_steps = 20

    for step in range(num_steps):
        # 这里简单用固定 keys，你也可以换成随机的 local_id
        keys = make_keys(batch_size=batch_size, slot=121, key_base=0)

        # 1) 前向：根据 keys 查表得到 embedding（JAX 数组）
        emb = layer.forward(keys)  # [B, D]

        # 2) 用 JAX 对 embedding 计算 loss 和 d loss / d emb
        loss, grad_emb = loss_and_grad(emb)

        # 3) 在 Python / NumPy 端，用 Adam 更新 KVStore 中的嵌入
        layer.backward(keys, grad_emb)

        if step % 5 == 0 or step == num_steps - 1:
            print(f"step {step:02d} - loss: {float(loss):.6f}")

    # 训练结束后，可以再查一次 embedding 看看是否接近 1
    final_emb = layer.forward(make_keys(batch_size=batch_size, slot=121, key_base=0))
    print("final embeddings (first batch):")
    print(np.array(final_emb))


if __name__ == "__main__":
    main()
