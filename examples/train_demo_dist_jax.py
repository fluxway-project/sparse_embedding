# coding=utf-8
"""
Flax NNX + shard_map 分布式训练示例。

- 模型：用 nnx.Module 定义（MyModel），初始化与单机逻辑一致。
- 多设备：shard_map 内只使用 JAX 类型（params pytree + opt_state），因此从 NNX 模型
  提取 params，用 optax 做优化；数据沿 mesh 轴 p 分片，梯度 pmean 聚合。
- 单设备：直接 jit(params, opt_state, batch)，无集体通信。
- 训练结束后可将 params 写回 NNX 模型，便于用 NNX 序列化/加载。

运行：python train_demo_dist.py（单机）
     或通过 run.sh / mpirun 设置 JAX_NUM_PROCESSES 做多机多卡。
"""

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


# os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'

"""
https://huggingface.co/blog/jiagaoxiang/jax-nnx-sharding
"""


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
    print(
        f"Rank {rank}, total_devices={total_devices}, JAX_NUM_PROCESSES={os.environ.get('JAX_NUM_PROCESSES', '1')}"
    )


# ---------- 1. 模型（用于初始化参数）----------
class MyModel(nnx.Module):
    def __init__(
        self,
        in_features: int = 64,
        out_features: int = 64,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        self.dense = nnx.Linear(in_features, out_features, rngs=rngs)

    def __call__(self, x: jax.Array, *, axis_name: str | None = None) -> jax.Array:
        x = self.dense(x)
        return jax.nn.relu(x)


# ---------- 2. Mesh 与分片 ----------
# cpu
devices = mesh_utils.create_device_mesh((total_devices,))
mesh = Mesh(devices, axis_names=("data",))
# 数据：batch 沿 p 分片，每设备 (batch_per_device, features) / (batch_per_device,)
# 定义数据分片：Batch 维分片，Feature 维不分片
data_sharding = NamedSharding(mesh, P("data", None))
# 定义模型分片：模型参数在所有设备上复制（DP 特点）
replicated_sharding = NamedSharding(mesh, P())


# ---------- 3. 数据 ----------
def get_batch(
    step: int, batch_per_device: int = 16, features: int = 64, num_classes: int = 2
):
    """返回 (inputs, labels)。多设备时 shape 为 (total_devices, batch_per_device, ...) 以便按 p 分片。"""
    key = jax.random.PRNGKey(step)
    k1, k2 = jax.random.split(key)
    inputs = jax.random.normal(k1, (total_devices, batch_per_device, features))
    labels = jax.random.randint(
        k2, (total_devices, batch_per_device), minval=0, maxval=num_classes
    )
    return inputs, labels


@nnx.jit(
    in_shardings=(
        replicated_sharding,
        replicated_sharding,
        data_sharding,
        data_sharding,
    ),
    out_shardings=replicated_sharding,
    donate_argnums=(0, 1),
)
def train_step(model, optimizer, batch_x, batch_y):
    # 此处的 batch_x 已经是被切分后的局部数据
    def loss_fn(x, model):
        pred = model(x)
        return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(pred, batch_y))

    # 计算局部梯度
    loss, (grads_x, grads_model) = nnx.value_and_grad(loss_fn, argnums=(0, 1))(
        batch_x, model
    )
    print(f"grads: {grads_x}")

    # 更新模型
    optimizer.update(grads_model)
    return loss, grads_x


# ---------- 6. 训练入口 ----------
def train():

    num_classes = 2
    model = MyModel(in_features=64, out_features=num_classes, rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adam(1e-3))

    for step in range(10):
        inputs, labels = get_batch(step)
        inputs = jax.device_put(inputs, data_sharding)
        labels = jax.device_put(labels, data_sharding)
        with mesh:
            loss, grads = train_step(model, optimizer, inputs, labels)
            if rank == 0:
                print(f"Step {step}, Loss: {float(loss):.6f}")


if __name__ == "__main__":
    train()
