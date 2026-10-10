import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

import sparse_embedding as se
from sparse_embedding.checkpoint import (
    SUPPORTED_KEY_DTYPES,
    ElasticRestore,
    SparseCheckpoint,
    SparsePartitionAssignmentSpec,
    encode_restore_options,
)
from sparse_embedding.checkpoint.lance_backend import LocalLanceStorage

_REFERENCE_VECTORS_PATH = (
    Path(__file__).parent / "data" / "modulo_assignment_reference_vectors.json"
)
_REFERENCE_VECTORS = json.loads(_REFERENCE_VECTORS_PATH.read_text())


def _require_reference_contract() -> None:
    assert _REFERENCE_VECTORS["schema_version"] == 1
    assert _REFERENCE_VECTORS["sparse_assignment_name"] == "sparse_embedding.modulo"
    assert _REFERENCE_VECTORS["sparse_assignment_parameters"] == {}
    vectors_by_group = _vectors_by_dtype_and_count()
    assert set(vectors_by_group) == {
        (key_dtype, partition_count)
        for key_dtype in SUPPORTED_KEY_DTYPES
        for partition_count in (3, 7)
    }
    for key_dtype in SUPPORTED_KEY_DTYPES:
        bits = int(key_dtype.removeprefix("uint").removeprefix("int"))
        if key_dtype.startswith("int"):
            required_keys = {-(1 << (bits - 1)), -1, 0, 1, (1 << (bits - 1)) - 1}
        else:
            required_keys = {0, 1, 1 << (bits - 1), (1 << bits) - 1}
        for partition_count in (3, 7):
            assert {
                vector["logical_key"]
                for vector in vectors_by_group[(key_dtype, partition_count)]
            } == required_keys


def _vectors_by_dtype_and_count() -> dict[tuple[str, int], list[dict[str, int | str]]]:
    vectors: dict[tuple[str, int], list[dict[str, int | str]]] = defaultdict(list)
    for vector in _REFERENCE_VECTORS["vectors"]:
        vectors[(vector["key_dtype"], vector["partition_count"])].append(vector)
    return vectors


def _store(key_dtype: str) -> se.EmbeddingStore:
    return se.EmbeddingStore(
        se.EmbeddingConfig(
            dims=[1, 2],
            key_dtype=key_dtype,
            thread_budget=1,
            value_dtype="float32",
            enable_optimizer=False,
        )
    )


def _keys_for_dtype(key_dtype: str) -> np.ndarray:
    logical_keys = sorted(
        {
            vector["logical_key"]
            for vector in _REFERENCE_VECTORS["vectors"]
            if vector["key_dtype"] == key_dtype
        }
    )
    return np.asarray(logical_keys, dtype=np.dtype(key_dtype))


@pytest.mark.parametrize("vector", _REFERENCE_VECTORS["vectors"])
def test_modulo_reference_vectors_match_mathematical_modulo(vector) -> None:
    _require_reference_contract()
    assert vector["logical_key"] % vector["partition_count"] == vector["partition_id"]


@pytest.mark.parametrize("key_dtype", sorted(SUPPORTED_KEY_DTYPES))
def test_modulo_reference_vectors_match_native_elastic_restore(
    tmp_path: Path, key_dtype: str
) -> None:
    _require_reference_contract()
    vectors_by_group = _vectors_by_dtype_and_count()
    keys = _keys_for_dtype(key_dtype)
    source = _store(key_dtype)
    values_by_dimension = {
        1: np.arange(keys.size, dtype=np.float32),
        2: np.column_stack(
            (
                np.arange(keys.size, dtype=np.float32),
                np.arange(keys.size, dtype=np.float32) + 10,
            )
        ),
    }
    for dimension_values in values_by_dimension.values():
        source.assign(keys, dimension_values)

    checkpoint = SparseCheckpoint(LocalLanceStorage(tmp_path / key_dtype))
    committed = checkpoint.save_training_checkpoint(source, "modulo-vectors")

    for (vector_dtype, partition_count), vectors in vectors_by_group.items():
        if vector_dtype != key_dtype:
            continue
        for partition_id in range(partition_count):
            target = _store(key_dtype)
            options = encode_restore_options(
                ElasticRestore(
                    SparsePartitionAssignmentSpec.modulo(),
                    partition_count,
                    partition_id,
                    key_dtype=key_dtype,
                )
            )
            checkpoint.load_training_checkpoint(
                target,
                committed.handle,
                options,
                load_optimizer_state=False,
            )

            expected_keys = np.asarray(
                sorted(
                    vector["logical_key"]
                    for vector in vectors
                    if vector["partition_id"] == partition_id
                ),
                dtype=np.dtype(key_dtype),
            )
            assert target.total_size() == 2 * expected_keys.size
            if expected_keys.size == 0:
                continue

            source_indices = np.searchsorted(keys, expected_keys)
            for dimension, source_values in values_by_dimension.items():
                restored_values, _ = target.lookup(expected_keys, dim=dimension)
                np.testing.assert_array_equal(
                    np.from_dlpack(restored_values).reshape((-1, dimension)),
                    source_values[source_indices].reshape((-1, dimension)),
                )
