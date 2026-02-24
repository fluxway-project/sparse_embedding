# coding=utf-8
"""
分布式训练 demo：仅使用 Flax NNX + shard_map 构建。
单机多卡 / 多机多卡均可，通过 Mesh 与 shard_map 指定分片。
单设备时自动退化为 jit，无需集体通信。
"""

import inspect
import os
import jax
import jax.numpy as jnp
from flax import nnx
import optax
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils

try:
    from jax import shard_map
except ImportError:
    from jax.experimental.shard_map import shard_map


# 多进程时初始化 JAX 分布式（run.sh 通过 mpirun 设置 JAX_NUM_PROCESSES 等）
process_id = int(os.environ.get("OMPI_COMM_WORLD_RANK", os.environ.get("PMI_RANK", 0)))
if "JAX_NUM_PROCESSES" in os.environ:
    nproc = int(os.environ["JAX_NUM_PROCESSES"])
    if "JAX_COORDINATOR_ADDRESS" not in os.environ and nproc > 1:
        os.environ.setdefault("JAX_COORDINATOR_ADDRESS", "localhost:34567")
    jax.distributed.initialize(
        num_processes=nproc,
        process_id=process_id,
    )

rank = jax.process_index()
total_devices = jax.device_count()
if rank == 0:
    print(f"Rank {rank}, total_devices={total_devices}, JAX_NUM_PROCESSES={os.environ.get('JAX_NUM_PROCESSES', '1')}")


# ---------- 1. NNX 模型 ----------
class MyModel(nnx.Module):
    """NNX 模块：前向中做一次 all_to_all（需在带 axis_name 的 shard_map 内调用）。"""

    def __init__(self, in_features: int = 64, out_features: int = 64, rngs: nnx.Rngs = nnx.Rngs(0)):
        self.dense = nnx.Linear(in_features, out_features, rngs=rngs)

    def __call__(self, x: jax.Array, *, axis_name: str | None = None) -> jax.Array:
        x = self.dense(x)
        # if axis_name is not None:
        #     x = jax.lax.all_to_all(x, axis_name=axis_name, split_axis=1, concat_axis=1)
        return jax.nn.relu(x)


# 1. 获取当前步骤的随机主键
def get_batch(step: int, batch_per_device: int = 16, features: int = 64):
    key = jax.random.PRNGKey(step)
    
    # 2. 分裂 Key：一个用于生成 features，一个用于生成 labels
    key_input, key_label = jax.random.split(key)

    # 3. 生成输入特征 (Input Features)
    inputs = jax.random.normal(key_input, (batch_per_device, features))
    
    # 4. 生成标签 (Labels)：整型类别 [0, num_classes)，shape (batch_per_device,)
    num_classes = 2
    labels = jax.random.randint(key_label, (batch_per_device,), minval=0, maxval=num_classes)
    return {"batch": inputs, "labels": labels}


def loss_fn(model, batch, num_classes: int = 2):
    logits = model(batch["batch"], axis_name=None)  # (batch, num_classes)
    # 分类：logits (B, C) 与 labels (B,) 用 cross-entropy
    return jnp.mean(
        optax.softmax_cross_entropy_with_integer_labels(logits, batch["labels"])
    )


@nnx.jit
def train_step(model, optimizer, batch, rngs):
    grad_fn = nnx.value_and_grad(loss_fn)
    loss, grads = grad_fn(model, batch)
    # print(loss, grads)
    # jax.debug.print(f"loss: {loss.shape}, grads: {grads}")
    optimizer.update(grads)
    return loss


def train():
    num_classes = 2
    model = MyModel(in_features=64, out_features=num_classes, rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adam(1e-3))
    
    model.train()
    rngs = jax.random.PRNGKey(0)
    for step in range(10):
        batch = get_batch(step)
        # print(step, batch)
        loss = train_step(model, optimizer, batch, rngs)
        if rank == 0:
            print(f"Step {step}, Loss: {float(loss):.6f}")


if __name__ == "__main__":
    train()