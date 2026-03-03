# coding=utf-8

from jax import ops
import jax.numpy as jnp

# 展平后的数据 [1, 2, 3, 4, 5]
values = jnp.array([1, 2, 3, 4, 5, 6])
# 对应的 Segment ID: 0号序列, 0号序列, 1号序列...
segment_ids = jnp.array([0, 0, 1, 1, 1, 2])

# 自动按 ID 分组求和
result = ops.segment_sum(values, segment_ids) # 输出 [3, 12]
print(result)