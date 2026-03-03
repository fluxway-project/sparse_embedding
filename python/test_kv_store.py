# Minimal test without JAX: C++ extension + JaxKVStore (numpy only)
# Run from sparse_embedding:  python python/test_kv_store.py
import sys
import os
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)
from kv_store import JaxKVStore
import numpy as np

# slot 121 (high 16 bits of fid), dim 5
feature_metas = {121: 5, 2673: 6}
# init_type: 0=全0, 1=随机
store = JaxKVStore(feature_metas, init_type=0)

# get_embedding_dim
# C++ KVStore::get_embedding_dim 返回 std::vector<int>，经 pybind11 映射为 Python list
print(store.get_embedding_dim(np.array([121 << 48], dtype=np.uint64)))
assert store.get_embedding_dim(np.array([121 << 48], dtype=np.uint64)) == [5]
assert store.get_embedding_dim(np.array([2673 << 48], dtype=np.uint64)) == [6]
print("get_embedding_dim: ok")

# insert batch
slot121 = 121 << 48
keys = np.array([slot121 + 10, slot121 + 25], dtype=np.uint64)
values = np.array([[10.0] * 5, [25.0] * 5], dtype=np.float32)
store.insert(keys, values)
print("insert: ok")

# lookup_or_insert (numpy path - no JAX)
keys_lo = np.array([slot121 + 10, slot121 + 99, slot121 + 25], dtype=np.uint64)
results = store.lookup_or_insert(keys_lo)
print(results)
assert results.shape == (3, 5)
assert list(results[0]) == [10.0] * 5   # 已插入
assert list(results[1]) == [0.0] * 5   # init_type=0 时新 key 全 0
assert list(results[2]) == [25.0] * 5
print("lookup_or_insert: ok")

# lookup (numpy path)
lookup_res = store.lookup(keys_lo)
print(lookup_res)
assert lookup_res.shape == (3, 5)
assert list(lookup_res[0]) == [10.0] * 5
assert list(lookup_res[1]) == [0.0] * 5  # 同上
assert list(lookup_res[2]) == [25.0] * 5
print("lookup: ok")

print("All tests passed.")
