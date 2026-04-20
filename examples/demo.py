import jax
import jax.numpy as jnp
import numpy as np
from kv_store import JaxKVStore  # 导入我们的包装类

# 1. 初始化 C++ KV Store（init_type: 0=全0, 1=随机）
feature_metas = {121: 5, 2673: 6}
store = JaxKVStore(feature_metas, init_type=1)

# 2. 插入一些数据（key 的高 16 位为 slot，例如 (121 << 48) | id）
slot121 = 121 << 48
store.insert(
    np.array([slot121 | 10, slot121 | 25], dtype=np.uint64),
    np.array([[10.0] * 5, [25.0] * 5], dtype=np.float32),
)
print("Data inserted into C++ store.")


# 3. 定义一个 JAX 函数，它会调用我们的 lookup
def jax_function(keys_to_lookup):
    print("Tracing JAX function...")  # 这句话只会在第一次编译时打印

    # 在 JIT 函数内部调用 store.lookup！
    embeddings = store.lookup_or_insert(keys_to_lookup)

    # 做一些计算
    return jnp.sum(embeddings, axis=-1)


# 4. 运行 JIT 编译的函数（key = (slot << 48) | id）
print("\n--- Running JIT'd function for the first time (compilation) ---")
keys1 = jnp.array([slot121 | 10, slot121 | 99, slot121 | 25], dtype=jnp.uint64)
result1 = jax_function(keys1)
print(f"Keys: {keys1}")
print(f"Result (sum of embeddings): {result1}")

print("\n--- Running JIT'd function for the second time (cached) ---")
keys2 = jnp.array([slot121 | 5, slot121 | 10], dtype=jnp.uint64)
result2 = jax_function(keys2)
print(f"Keys: {keys2}")
print(f"Result (sum of embeddings): {result2}")


## init variables


def init_var():
    # init emb store（init_type: 0=全0, 1=随机）
    feature_metas = {121: 5, 2673: 6}
    store = JaxKVStore(feature_metas, init_type=1)
    # init monment


## forward


def sparse_forward_test():
    pass

    # all2all

    # unqiue

    # lookup_or_insert

    # all2all

    # gather


## backward


def sparse_backward_test(fid, grad):
    pass

    # all2all

    # unqiue

    # lookup

    # all2all

    # gather
