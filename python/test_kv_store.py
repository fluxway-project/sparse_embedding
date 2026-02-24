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
default_values = {121: [1.0] * 5, 2673: [2.0] * 6}
store = JaxKVStore(feature_metas, default_values)

# get_embedding_dim
assert store.get_embedding_dim(121 << 48) == 5
assert store.get_embedding_dim(2673 << 48) == 6
print("get_embedding_dim: ok")

# insert batch
slot121 = 121 << 48
keys = np.array([slot121 | 10, slot121 | 25], dtype=np.uint64)
values = np.array([[10.0] * 5, [25.0] * 5], dtype=np.float32)
store.insert(keys, values)
print("insert: ok")

# lookup_or_insert (numpy path - no JAX)
keys_lo = np.array([slot121 | 10, slot121 | 99, slot121 | 25], dtype=np.uint64)
results = store.lookup_or_insert(keys_lo)
assert results.shape == (3, 5)
assert list(results[0]) == [10.0] * 5
assert list(results[1]) == [1.0] * 5  # default
assert list(results[2]) == [25.0] * 5
print("lookup_or_insert: ok")

# lookup (numpy path)
lookup_res = store.lookup(keys_lo)
assert lookup_res.shape == (3, 5)
assert list(lookup_res[0]) == [10.0] * 5
assert list(lookup_res[1]) == [1.0] * 5
assert list(lookup_res[2]) == [25.0] * 5
print("lookup: ok")

print("All tests passed.")
