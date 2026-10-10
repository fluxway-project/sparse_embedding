import numpy as np
import pytest

from sparse_embedding import (
    FastRangePartitioner,
    HashPartitioner,
    ModuloPartitioner,
    create_partitioner,
    is_partitioner_registered,
)
from sparse_embedding.checkpoint import (
    SPARSE_EMBEDDING_FAST_RANGE_ASSIGNMENT_NAME,
    SPARSE_EMBEDDING_HASH_ASSIGNMENT_NAME,
    SPARSE_EMBEDDING_MODULO_ASSIGNMENT_NAME,
)


def test_registry_registration_and_factories():
    assert is_partitioner_registered(SPARSE_EMBEDDING_MODULO_ASSIGNMENT_NAME)
    assert is_partitioner_registered(SPARSE_EMBEDDING_HASH_ASSIGNMENT_NAME)
    assert is_partitioner_registered(SPARSE_EMBEDDING_FAST_RANGE_ASSIGNMENT_NAME)
    assert not is_partitioner_registered("non_existent_algo")


def test_registry_fail_fast_on_unknown():
    with pytest.raises(Exception) as exc_info:
        create_partitioner("non_existent_algo", 4)
    msg = str(exc_info.value).lower()
    assert "unknown partitioner" in msg or "not registered" in msg


def test_builtin_partitioners_consistency():
    keys = np.arange(-16, 16, dtype=np.int64)

    # Modulo
    p_reg = create_partitioner(SPARSE_EMBEDDING_MODULO_ASSIGNMENT_NAME, 4)
    p_cls = ModuloPartitioner(num_shards=4)
    res_reg = np.from_dlpack(p_reg.partition(keys))
    res_cls = np.from_dlpack(p_cls.partition(keys))
    np.testing.assert_array_equal(res_reg, res_cls)

    # Hash
    h_reg = create_partitioner(SPARSE_EMBEDDING_HASH_ASSIGNMENT_NAME, 4)
    h_cls = HashPartitioner(num_shards=4)
    np.testing.assert_array_equal(
        np.from_dlpack(h_reg.partition(keys)),
        np.from_dlpack(h_cls.partition(keys)),
    )

    # FastRange
    f_reg = create_partitioner(SPARSE_EMBEDDING_FAST_RANGE_ASSIGNMENT_NAME, 4)
    f_cls = FastRangePartitioner(num_shards=4)
    np.testing.assert_array_equal(
        np.from_dlpack(f_reg.partition(keys)),
        np.from_dlpack(f_cls.partition(keys)),
    )


@pytest.mark.parametrize("dtype_str", ["int32", "uint32", "int64", "uint64"])
def test_partitioner_all_dtypes(dtype_str):
    keys = np.arange(0, 100, dtype=dtype_str)
    p = create_partitioner(SPARSE_EMBEDDING_MODULO_ASSIGNMENT_NAME, 8)
    shards = np.from_dlpack(p.partition(keys))
    assert shards.dtype == np.uint32
    assert len(shards) == len(keys)
    np.testing.assert_array_equal(shards, (keys % 8).astype(np.uint32))


def test_partitioner_filter_vectorized():
    keys = np.array([-4, -3, -2, -1, 0, 1, 2, 3], dtype=np.int64)
    p = ModuloPartitioner(num_shards=4)
    mask = np.from_dlpack(p.filter(keys, target_shard=1))
    expected = np.array([False, True, False, False, False, True, False, False])
    np.testing.assert_array_equal(mask, expected)
