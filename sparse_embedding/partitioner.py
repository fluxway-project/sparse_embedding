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

"""Elastic partitioning strategies and functional partition API."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from typing import Any

from ._core import (
    Partitioner,
    create_partitioner,
)
from .checkpoint.base import (
    SPARSE_EMBEDDING_FAST_RANGE_ASSIGNMENT_NAME,
    SPARSE_EMBEDDING_HASH_ASSIGNMENT_NAME,
    SPARSE_EMBEDDING_MODULO_ASSIGNMENT_NAME,
    PartitionSpec,
)


class PartitionStrategy:
    """Framework-agnostic, late-binding elastic partitioning strategy.

    Decouples algorithm semantics from cluster topology. Supports dynamic
    sharding via the Fluxway FunctionPartition SPI, direct execution,
    explicit binding, and transparent JIT kernel specialization caching.
    """

    __slots__ = ("_spec", "_kernel_cache", "_lock")

    def __init__(
        self,
        name: str = SPARSE_EMBEDDING_MODULO_ASSIGNMENT_NAME,
        parameters: Mapping[str, object] | None = None,
    ) -> None:
        self._spec = PartitionSpec(name, parameters)
        self._kernel_cache: dict[int, Partitioner] = {}
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return self._spec.name

    @property
    def parameters(self) -> Mapping[str, object]:
        return self._spec.parameters

    @property
    def spec(self) -> PartitionSpec:
        """Canonical partition specification serving as Single Source of Truth."""
        return self._spec

    def bind(self, partition_count: int) -> Partitioner:
        """Pre-bind and retrieve specialized C++ Partitioner for extreme hot loops."""
        if not isinstance(partition_count, int) or isinstance(partition_count, bool):
            raise TypeError("partition_count must be an integer")
        if partition_count <= 0:
            raise ValueError(f"partition_count must be positive; got {partition_count}")

        with self._lock:
            kernel = self._kernel_cache.get(partition_count)
            if kernel is None:
                kernel = create_partitioner(
                    self.name, partition_count, dict(self.parameters)
                )
                self._kernel_cache[partition_count] = kernel
            return kernel

    def __call__(
        self,
        values: Any,
        *,
        partition_count: int,
        domain_size: int | None = None,
        num_threads: int = 0,
        **kwargs: Any,
    ) -> Any:
        """Execute partitioning adhering to Fluxway FunctionPartition routing contract."""
        del domain_size, kwargs
        kernel = self.bind(partition_count)
        return kernel.partition(
            values,
            partition_count=partition_count,
            num_threads=num_threads,
        )

    def filter(
        self,
        values: Any,
        target_shard: int,
        *,
        partition_count: int,
        domain_size: int | None = None,
        num_threads: int = 0,
        **kwargs: Any,
    ) -> Any:
        """Filter items matching target shard on the specified partition topology."""
        del domain_size, kwargs
        kernel = self.bind(partition_count)
        return kernel.filter(
            values,
            target_shard=target_shard,
            partition_count=partition_count,
            num_threads=num_threads,
        )

    def __repr__(self) -> str:
        return f"PartitionStrategy(name={self.name!r}, parameters={dict(self.parameters)!r})"


class ModuloPartition(PartitionStrategy):
    """Canonical signed floor modulo partitioning strategy."""

    def __init__(self, parameters: Mapping[str, object] | None = None) -> None:
        super().__init__(
            name=SPARSE_EMBEDDING_MODULO_ASSIGNMENT_NAME,
            parameters=parameters,
        )


class HashPartition(PartitionStrategy):
    """High-throughput pseudo-random hash partitioning strategy."""

    def __init__(
        self,
        parameters: Mapping[str, object] | None = None,
        *,
        seed: int | None = None,
    ) -> None:
        params = dict(parameters or {})
        if seed is not None:
            params["seed"] = str(seed)
        super().__init__(
            name=SPARSE_EMBEDDING_HASH_ASSIGNMENT_NAME,
            parameters=params,
        )


class FastRangePartition(PartitionStrategy):
    """Lemire Fast Range division-free partitioning strategy."""

    def __init__(self, parameters: Mapping[str, object] | None = None) -> None:
        super().__init__(
            name=SPARSE_EMBEDDING_FAST_RANGE_ASSIGNMENT_NAME,
            parameters=parameters,
        )


_STRATEGY_CACHE: dict[tuple[str, tuple[tuple[str, str], ...]], PartitionStrategy] = {}
_STRATEGY_CACHE_LOCK = threading.Lock()


def _get_or_create_strategy(
    algorithm: str, parameters: Mapping[str, object] | None
) -> PartitionStrategy:
    params_key = tuple(sorted((str(k), str(v)) for k, v in (parameters or {}).items()))
    cache_key = (algorithm, params_key)
    with _STRATEGY_CACHE_LOCK:
        strategy = _STRATEGY_CACHE.get(cache_key)
        if strategy is None:
            strategy = PartitionStrategy(algorithm, parameters)
            _STRATEGY_CACHE[cache_key] = strategy
        return strategy


def partition(
    keys: Any,
    *,
    partition_count: int,
    domain_size: int | None = None,
    algorithm: str = SPARSE_EMBEDDING_MODULO_ASSIGNMENT_NAME,
    num_threads: int = 0,
    **parameters: Any,
) -> Any:
    """Convenience functional partitioning API with transparent specialization caching."""
    strategy = _get_or_create_strategy(algorithm, parameters)
    return strategy(
        keys,
        partition_count=partition_count,
        domain_size=domain_size,
        num_threads=num_threads,
    )


def filter_partition(
    keys: Any,
    target_shard: int,
    *,
    partition_count: int,
    domain_size: int | None = None,
    algorithm: str = SPARSE_EMBEDDING_MODULO_ASSIGNMENT_NAME,
    num_threads: int = 0,
    **parameters: Any,
) -> Any:
    """Convenience functional partition filter API with transparent specialization caching."""
    strategy = _get_or_create_strategy(algorithm, parameters)
    return strategy.filter(
        keys,
        target_shard=target_shard,
        partition_count=partition_count,
        domain_size=domain_size,
        num_threads=num_threads,
    )


def create_strategy(
    spec_or_name: PartitionSpec | str = SPARSE_EMBEDDING_MODULO_ASSIGNMENT_NAME,
    parameters: Mapping[str, object] | None = None,
) -> PartitionStrategy:
    """Factory creating an elastic PartitionStrategy (Form A) from PartitionSpec or algorithm name."""
    if isinstance(spec_or_name, PartitionSpec):
        return _get_or_create_strategy(spec_or_name.name, spec_or_name.parameters)
    if isinstance(spec_or_name, str):
        return _get_or_create_strategy(spec_or_name, parameters)
    raise TypeError(
        f"spec_or_name must be a PartitionSpec or string algorithm name; got {type(spec_or_name).__name__}"
    )
