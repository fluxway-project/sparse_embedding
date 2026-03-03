#coding=utf-8

import os
import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental import mesh_utils

try:
    from jax import shard_map
except ImportError:
    from jax.experimental.shard_map import shard_map

# 多进程时初始化 JAX 分布式（test/jax/run.sh 通过 mpirun 设置 JAX_NUM_PROCESSES 等）
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
world_size = jax.process_count()
local_device_count = jax.local_device_count()


def demo_ragged_primitive():
    # --- 准备数据 (遵循文档中的 Example，在多进程 + pmap 场景下重写) ---
    #
    # 约定：world_size = 2，每个进程 local_device_count = 1，
    #       全局一共 2 个 replica，ragged_all_to_all 的 axis_size = 2。
    assert world_size == 2, f"预期 2 个进程(world_size=2)，当前为 {world_size}"
    assert local_device_count == 1, f"预期每进程 1 个本地设备，当前为 {local_device_count}"

    # 每个进程只构造自己这一行的数据，然后通过 pmap 参与全局 ragged_all_to_all。
    if rank == 0:
        # 对应文档中的 Device 0 行
        operand_row = jnp.array([1, 2, 2], dtype=jnp.int32)
        input_offsets_row = jnp.array([0, 1], dtype=jnp.int32)
        send_sizes_row = jnp.array([1, 2], dtype=jnp.int32)
        output_offsets_row = jnp.array([0, 0], dtype=jnp.int32)
        recv_sizes_row = jnp.array([1, 1], dtype=jnp.int32)
    else:
        # 对应文档中的 Device 1 行
        operand_row = jnp.array([3, 4, 0], dtype=jnp.int32)
        input_offsets_row = jnp.array([0, 1], dtype=jnp.int32)
        send_sizes_row = jnp.array([1, 1], dtype=jnp.int32)
        output_offsets_row = jnp.array([1, 2], dtype=jnp.int32)
        recv_sizes_row = jnp.array([2, 1], dtype=jnp.int32)


    operand = operand_row               # shape (3,)
    output = jnp.zeros((4,), dtype=jnp.int32)
    input_offsets = input_offsets_row   # shape (2,)
    send_sizes = send_sizes_row         # shape (2,)
    output_offsets = output_offsets_row # shape (2,)
    recv_sizes = recv_sizes_row         # shape (2,)

    # --- 执行算子 ---
    # 使用 Mesh + shard_map + axis_name='p'，让 ragged_all_to_all 在所有进程 + 设备间做真正的集合通信。
    def _call_ragged(op, out, i_off, s_sz, o_off, r_sz):
        return lax.ragged_all_to_all(
            op, out, i_off, s_sz, o_off, r_sz,
            axis_name='p'
        )

    # 创建 mesh，使用 shard_map 代替 pmap
    devices = mesh_utils.create_device_mesh((world_size,))
    mesh = Mesh(devices, axis_names=('p',))

    # 这里不在 mesh 轴上对输入做切分，而是按进程构造好各自的一行数据，
    # 在 mesh 上以“复制”的方式参与 ragged_all_to_all 的集合通信。
    call_ragged_mapped = shard_map(
        _call_ragged,
        mesh=mesh,
        in_specs=(P(), P(), P(), P(), P(), P()),
        out_specs=P(),
    )

    with mesh:
        results = call_ragged_mapped(
            operand, output, input_offsets, send_sizes, output_offsets, recv_sizes
        )

    # --- 结果打印 ---
    if rank == 0:
        print("Device 0 结果 (预期 [1, 3, 0, 0]):", results)
    else:
        print("Device 1 结果 (预期 [2, 2, 4, 0]):", results)

if __name__ == "__main__":
    demo_ragged_primitive()