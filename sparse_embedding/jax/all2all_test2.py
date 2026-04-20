# coding=utf-8

from functools import partial
import os
import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import numpy as np

# —— 1. 初始化分布式 —— #
process_id = int(os.environ.get("OMPI_COMM_WORLD_RANK", os.environ.get("PMI_RANK", 0)))

if "JAX_NUM_PROCESSES" in os.environ:
    nproc = int(os.environ["JAX_NUM_PROCESSES"])
    if "JAX_COORDINATOR_ADDRESS" not in os.environ and nproc > 1:
        os.environ["JAX_COORDINATOR_ADDRESS"] = "localhost:34567"

    jax.distributed.initialize(
        num_processes=nproc,
        process_id=process_id,
    )


def all2all(x, axis_name: str):
    return lax.all_to_all(x, axis_name=axis_name, split_axis=0, concat_axis=0)


@partial(jax.pmap, axis_name="hosts", devices=jax.devices())
def func(data):
    res = all2all(data, axis_name="hosts")
    return res


global_rank = jax.process_index()
world_size = jax.process_count()

local_data = jnp.ones((jax.local_device_count(), 4)) * jax.process_index()

res = func(local_data)
print(res)
