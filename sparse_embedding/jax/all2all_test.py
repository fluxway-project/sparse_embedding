# coding=utf-8
"""
all_to_all 测试：Mesh + shard_map
支持：
  单进程多设备（CPU/GPU）
  mpirun 多进程

运行方式：

单进程：
  JAX_PLATFORMS=cpu python all2all_test.py

多进程：
  mpirun -np 4 \
    -x JAX_NUM_PROCESSES=4 \
    -x JAX_PLATFORMS=cpu \
    python all2all_test.py
"""

import os
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental import mesh_utils
from flax.nnx import nnx

try:
    from jax import shard_map
except ImportError:
    from jax.experimental.shard_map import shard_map


# ================================
# 分布式初始化
# ================================
process_id = int(os.environ.get("OMPI_COMM_WORLD_RANK", os.environ.get("PMI_RANK", 0)))

if "JAX_NUM_PROCESSES" in os.environ:
    nproc = int(os.environ["JAX_NUM_PROCESSES"])
    if "JAX_COORDINATOR_ADDRESS" not in os.environ and nproc > 1:
        os.environ["JAX_COORDINATOR_ADDRESS"] = "localhost:34567"

    jax.distributed.initialize(
        num_processes=nproc,
        process_id=process_id,
    )


proc_id = jax.process_index()
num_procs = jax.process_count()
devices = jax.devices()
num_devices = len(devices)


# ================================
# 构造 Mesh
# ================================
if num_procs > 1:
    # 跨进程 mesh
    devices = mesh_utils.create_device_mesh((num_procs,))
else:
    devices = devices

mesh = Mesh(devices, axis_names=("data",))


# ================================
# 每设备构造数据 + all_to_all
# ================================


@nnx.jit(in_shardings=(P(),), out_shardings=P("data", None))
def _create_and_all2all(n):
    """
    每设备构造 shape=(n, 4)
    然后在 data 轴做 all_to_all

    每个设备第 i 行来自不同设备
    """
    # 构造本地数据
    base = proc_id * 100
    data = jnp.arange(n * 4, dtype=jnp.int32).reshape(n, 4)
    data = data + base
    # print(data.addressable_data(0))

    # all_to_all
    return jax.lax.all_to_all(
        data,
        axis_name="data",
        split_axis=0,
        concat_axis=0,
    )


call_mapped = shard_map(
    _create_and_all2all,
    mesh=mesh,
    in_specs=(P(),),  # 标量 n 复制到每设备
    out_specs=P("data", None),
)


# ================================
# 主函数
# ================================
def main():
    n = mesh.shape["data"]

    if proc_id == 0:
        print("\n=== 每设备构造 (n,4) 然后 all_to_all ===")

    with mesh:
        output = call_mapped(n)

    print(output.addressable_data(0))

    # jax.debug.print(
    #     "rank {r} sees output:\n{x}\n",
    #     r=proc_id,
    #     x=output
    # )


if __name__ == "__main__":
    main()
