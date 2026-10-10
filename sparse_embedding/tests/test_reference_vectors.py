import numpy as np
import pytest

from sparse_embedding.checkpoint import (
    SPARSE_EMBEDDING_FAST_RANGE_ASSIGNMENT_NAME,
    SPARSE_EMBEDDING_HASH_ASSIGNMENT_NAME,
    SPARSE_EMBEDDING_MODULO_ASSIGNMENT_NAME,
    generate_reference_vectors,
)


@pytest.mark.parametrize(
    "algo_name",
    [
        SPARSE_EMBEDDING_MODULO_ASSIGNMENT_NAME,
        SPARSE_EMBEDDING_HASH_ASSIGNMENT_NAME,
        SPARSE_EMBEDDING_FAST_RANGE_ASSIGNMENT_NAME,
    ],
)
@pytest.mark.parametrize("dtype_str", ["int32", "uint32", "int64", "uint64"])
def test_generate_reference_vectors_all_algorithms_and_dtypes(algo_name, dtype_str):
    keys, ranks = generate_reference_vectors(
        algo_name,
        key_dtype=dtype_str,
        partition_count=8,
    )
    assert len(keys) == len(ranks)
    assert keys.dtype == np.dtype(dtype_str)
    assert ranks.dtype == np.uint32
    assert ranks.max() < 8 and ranks.min() >= 0
    if algo_name == SPARSE_EMBEDDING_MODULO_ASSIGNMENT_NAME:
        expected = (keys % 8).astype(np.uint32)
        np.testing.assert_array_equal(ranks, expected)


def test_generate_reference_vectors_custom_keys():
    custom_keys = [10, 25, 42, 99, 100]
    keys, ranks = generate_reference_vectors(
        SPARSE_EMBEDDING_MODULO_ASSIGNMENT_NAME,
        key_dtype="int64",
        partition_count=5,
        sample_keys=custom_keys,
    )
    np.testing.assert_array_equal(keys, np.array(custom_keys, dtype=np.int64))
    np.testing.assert_array_equal(ranks, np.array([0, 0, 2, 4, 0], dtype=np.uint32))


def test_generate_reference_vectors_invalid_count():
    with pytest.raises(ValueError, match="must be > 0"):
        generate_reference_vectors(
            SPARSE_EMBEDDING_MODULO_ASSIGNMENT_NAME,
            key_dtype="int64",
            partition_count=0,
        )
