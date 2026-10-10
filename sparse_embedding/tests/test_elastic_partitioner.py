# Copyright 2026 The Fluxway Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import inspect

import numpy as np
import pytest

import sparse_embedding as se
from sparse_embedding.checkpoint import (
    SPARSE_EMBEDDING_FAST_RANGE_ASSIGNMENT_NAME,
    SPARSE_EMBEDDING_HASH_ASSIGNMENT_NAME,
    SPARSE_EMBEDDING_MODULO_ASSIGNMENT_NAME,
    SparsePartitionAssignmentSpec,
)


def test_strategy_contract_inspection():
    strategy = se.ModuloPartition()
    sig = inspect.signature(strategy)
    # Validate that it strictly accepts positional values and keyword arguments
    # matching Fluxway FunctionPartition's inspect.signature.bind check.
    bound = sig.bind(object(), partition_count=1, domain_size=1)
    assert bound.arguments["partition_count"] == 1
    assert bound.arguments["domain_size"] == 1


def test_strategy_spec_ssot():
    modulo = se.ModuloPartition()
    assert isinstance(modulo.spec, SparsePartitionAssignmentSpec)
    assert modulo.spec.sparse_assignment_name == SPARSE_EMBEDDING_MODULO_ASSIGNMENT_NAME
    assert modulo.spec.sparse_assignment_parameters == {}

    hashed = se.HashPartition(seed=42)
    assert hashed.spec.sparse_assignment_name == SPARSE_EMBEDDING_HASH_ASSIGNMENT_NAME
    assert hashed.spec.sparse_assignment_parameters == {"seed": "42"}

    fast_range = se.FastRangePartition()
    assert (
        fast_range.spec.sparse_assignment_name
        == SPARSE_EMBEDDING_FAST_RANGE_ASSIGNMENT_NAME
    )


def test_dynamic_topology_late_binding():
    strategy = se.ModuloPartition()
    keys = np.array([-10, -5, -1, 0, 1, 5, 10, 100], dtype=np.int64)

    # Late-binding partition on 4 shards
    res_4 = np.from_dlpack(strategy(keys, partition_count=4))
    expected_4 = (keys % 4).astype(np.uint32)
    np.testing.assert_array_equal(res_4, expected_4)

    # Same strategy object late-binding partition on 7 shards
    res_7 = np.from_dlpack(strategy(keys, partition_count=7))
    expected_7 = (keys % 7).astype(np.uint32)
    np.testing.assert_array_equal(res_7, expected_7)

    # Same strategy object late-binding partition on 8 shards
    res_8 = np.from_dlpack(strategy(keys, partition_count=8, domain_size=10000))
    expected_8 = (keys % 8).astype(np.uint32)
    np.testing.assert_array_equal(res_8, expected_8)


def test_specialization_cache_and_prebound():
    strategy = se.ModuloPartition()
    keys = np.array([1, 2, 3, 4], dtype=np.int64)

    kernel_1 = strategy.bind(8)
    kernel_2 = strategy.bind(8)
    assert kernel_1 is kernel_2
    assert kernel_1.num_shards == 8
    assert kernel_1.name == SPARSE_EMBEDDING_MODULO_ASSIGNMENT_NAME

    # Pre-bound fast path
    res_bound = np.from_dlpack(kernel_1(keys))
    np.testing.assert_array_equal(res_bound, (keys % 8).astype(np.uint32))

    # Pre-bound call with matching partition_count passes invariant check
    res_matching = np.from_dlpack(kernel_1(keys, partition_count=8))
    np.testing.assert_array_equal(res_matching, (keys % 8).astype(np.uint32))

    # Pre-bound call with mismatched partition_count raises ValueError
    with pytest.raises(ValueError) as exc_info:
        kernel_1(keys, partition_count=4)
    assert "does not match" in str(exc_info.value)


@pytest.mark.parametrize("dtype_str", ["int32", "uint32", "int64", "uint64"])
def test_elastic_partition_all_dtypes(dtype_str):
    keys = np.arange(0, 64, dtype=dtype_str)
    strategy = se.ModuloPartition()
    shards = np.from_dlpack(strategy(keys, partition_count=8))
    assert shards.dtype == np.uint32
    np.testing.assert_array_equal(shards, (keys % 8).astype(np.uint32))


def test_elastic_filter():
    strategy = se.ModuloPartition()
    keys = np.array([-4, -3, -2, -1, 0, 1, 2, 3], dtype=np.int64)

    mask = np.from_dlpack(strategy.filter(keys, target_shard=1, partition_count=4))
    expected = keys % 4 == 1
    np.testing.assert_array_equal(mask, expected)


def test_functional_apis():
    keys = np.array([10, 20, 30, 40], dtype=np.int64)

    res = np.from_dlpack(se.partition(keys, partition_count=4))
    np.testing.assert_array_equal(res, (keys % 4).astype(np.uint32))

    mask = np.from_dlpack(se.filter_partition(keys, target_shard=2, partition_count=4))
    np.testing.assert_array_equal(mask, (keys % 4 == 2))


@pytest.mark.parametrize(
    "algo_name,StrategyClass",
    [
        (SPARSE_EMBEDDING_MODULO_ASSIGNMENT_NAME, se.ModuloPartition),
        (SPARSE_EMBEDDING_HASH_ASSIGNMENT_NAME, se.HashPartition),
        (SPARSE_EMBEDDING_FAST_RANGE_ASSIGNMENT_NAME, se.FastRangePartition),
    ],
)
@pytest.mark.parametrize("dtype_str", ["int32", "uint32", "int64", "uint64"])
def test_trinity_api_equivalence(algo_name, StrategyClass, dtype_str):
    partition_count = 6
    ref_keys, ref_ranks = se.checkpoint.generate_reference_vectors(
        algo_name, key_dtype=dtype_str, partition_count=partition_count
    )

    # Form A: Callable strategy object
    strategy = StrategyClass()
    ranks_form_a = np.from_dlpack(strategy(ref_keys, partition_count=partition_count))
    np.testing.assert_array_equal(ranks_form_a, ref_ranks)

    # Form B: Functional partition(...)
    ranks_form_b = np.from_dlpack(
        se.partition(ref_keys, partition_count=partition_count, algorithm=algo_name)
    )
    np.testing.assert_array_equal(ranks_form_b, ref_ranks)

    # Form C: Pre-bound hot path strategy.bind(partition_count)
    bound_kernel = strategy.bind(partition_count)
    ranks_form_c = np.from_dlpack(bound_kernel(ref_keys))
    np.testing.assert_array_equal(ranks_form_c, ref_ranks)


def test_strict_validation_no_fallback():
    strategy = se.ModuloPartition()
    keys = np.array([1, 2, 3], dtype=np.int64)

    # Invalid partition_count
    with pytest.raises(ValueError):
        strategy(keys, partition_count=0)

    with pytest.raises(ValueError):
        strategy(keys, partition_count=-1)

    with pytest.raises(TypeError):
        strategy(keys, partition_count="8")  # type: ignore

    # Unknown algorithm
    with pytest.raises(ValueError):
        se.PartitionStrategy("unregistered_algo_name")

    # Invalid parameters
    with pytest.raises(TypeError):
        se.PartitionStrategy(
            SPARSE_EMBEDDING_MODULO_ASSIGNMENT_NAME, parameters="invalid"
        )  # type: ignore


def test_fluxway_function_partition_mock():
    """Simulates Fluxway FunctionPartition runtime contract."""

    class MockFunctionPartition:
        def __init__(self, target, arguments=None):
            self.target = target
            self.arguments = arguments or {}
            self.validate_contract()

        def validate_contract(self):
            inspect.signature(self.target).bind(
                object(),
                partition_count=1,
                domain_size=1,
                **self.arguments,
            )

        def __call__(self, indices, dim_size, num_workers):
            return self.target(
                indices,
                partition_count=num_workers,
                domain_size=dim_size,
                **self.arguments,
            )

    strategy = se.ModuloPartition()
    fp = MockFunctionPartition(target=strategy)

    keys = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
    # Simulate Fluxway dispatching across 3 workers
    ranks = np.from_dlpack(fp(keys, dim_size=100, num_workers=3))
    np.testing.assert_array_equal(ranks, (keys % 3).astype(np.uint32))

    # Also supports passing bound Partitioner
    bound_p = strategy.bind(3)
    fp_bound = MockFunctionPartition(target=bound_p)
    ranks_bound = np.from_dlpack(fp_bound(keys, dim_size=100, num_workers=3))
    np.testing.assert_array_equal(ranks_bound, (keys % 3).astype(np.uint32))
