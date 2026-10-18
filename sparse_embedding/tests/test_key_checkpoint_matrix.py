import json
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import pytest

import sparse_embedding as se
from sparse_embedding.checkpoint import (
    IngestionEngine,
    LocalLanceStorage,
    PartitionSpec,
    RestorePlan,
    SparseCheckpoint,
    encode_restore_options,
    lance_backend,
)
from sparse_embedding.checkpoint.base import (
    CheckpointParticipant,
    InvalidCheckpointFragment,
)
from sparse_embedding.checkpoint.schema import (
    ENTRIES_DATASET,
    SNAPSHOT_MANIFEST_DATASET,
)

_KEY_CASES = (
    ("int32", np.asarray([-(1 << 31), -1, 0, (1 << 31) - 1], dtype=np.int32)),
    ("uint32", np.asarray([0, 1, (1 << 32) - 1], dtype=np.uint32)),
    ("int64", np.asarray([-(1 << 63), -1, 0, (1 << 63) - 1], dtype=np.int64)),
    ("uint64", np.asarray([0, 1, 1 << 63, (1 << 64) - 1], dtype=np.uint64)),
)


def test_native_restore_has_no_untyped_selected_key_entry_point() -> None:
    assert not hasattr(se._core.checkpoint, "restore_native_entries")


@pytest.mark.parametrize(
    ("actual_dtype", "requested_dtype"),
    [
        ("int32", "uint64"),
        ("uint32", "uint64"),
        ("int64", "uint64"),
        ("uint64", "int64"),
    ],
)
@pytest.mark.parametrize("operation", ["validate", "restore"])
def test_direct_native_rejects_physical_key_dtype_mismatch(
    tmp_path: Path, actual_dtype: str, requested_dtype: str, operation: str
) -> None:
    storage = LocalLanceStorage(tmp_path / "checkpoint")
    source = _store(actual_dtype)
    _assign(source, np.asarray([0, 1], dtype=actual_dtype))
    saved = SparseCheckpoint(storage).save_training_checkpoint(source, "native-types")
    manifest = storage.resolve_checkpoint_manifest(saved.handle)
    root = storage._manifest_fragment_root_for_participant(manifest, 0)
    table = lance.dataset(
        storage._workspace.uri(f"{root}/{ENTRIES_DATASET}")
    ).to_table()
    metadata = dict(table.schema.metadata)
    metadata[b"sparse_embedding.key_dtype"] = requested_dtype.encode()
    uri = str(tmp_path / "forged-key-type")
    lance.write_dataset(table.replace_schema_metadata(metadata), uri)
    target = _store(requested_dtype)
    with pytest.raises(ValueError, match="schema key dtype"):
        if operation == "validate":
            se._core.checkpoint.validate_native_checkpoint_chain(
                [[uri]],
                [[0]],
                [2],
                requested_dtype,
                {},
                partition_count=1,
                assignment_name=PartitionSpec.fast_range().name,
            )
        else:
            se._core.checkpoint.restore_native_partition_entries(
                target, [uri], 0, 1, 0, requested_dtype
            )
    assert target.is_operational() is True
    assert target.total_size() == 0


def _store(key_dtype: str, *, optimizer: bool = True) -> se.EmbeddingStore:
    config = se.EmbeddingConfig(
        dims=[2],
        thread_budget=1,
        value_dtype="float32",
        key_dtype=key_dtype,
        enable_optimizer=optimizer,
    )
    if optimizer:
        return se.EmbeddingStore(config, optimizer_config=se.Adam(lr=0.01))
    return se.EmbeddingStore(config)


def _assign(store: se.EmbeddingStore, keys: np.ndarray, offset: float = 0) -> None:
    values = np.repeat(
        (np.arange(keys.size, dtype=np.float32) + offset)[:, None], 2, axis=1
    )
    store.assign(keys, values)


def _contains(store: se.EmbeddingStore, keys: np.ndarray) -> np.ndarray:
    return np.asarray(np.from_dlpack(store.contains(keys))).astype(bool)


def _values(store: se.EmbeddingStore, keys: np.ndarray) -> np.ndarray:
    values, _ = store.lookup(keys, dim=2)
    return np.from_dlpack(values).reshape((-1, 2))


def _elastic_options(key_dtype: str, partition_count: int = 1) -> RestorePlan:
    return RestorePlan.from_partition(
        PartitionSpec.modulo(),
        partition_count,
        0,
        key_dtype=key_dtype,
    )


def _target_restore(
    participant: CheckpointParticipant,
    *,
    key_dtype: str = "uint64",
    engine: IngestionEngine = IngestionEngine.RECONSTRUCT_ENTRIES,
) -> RestorePlan:
    return RestorePlan.from_partition(
        PartitionSpec.modulo(),
        participant.participant_count,
        participant.participant_index,
        key_dtype=key_dtype,
        engine=engine,
    )


@pytest.mark.parametrize(("key_dtype", "keys"), _KEY_CASES)
def test_four_key_encodings_save_and_restore_physical_and_elastic(
    tmp_path: Path, key_dtype: str, keys: np.ndarray
) -> None:
    root = tmp_path / "checkpoint"
    storage = LocalLanceStorage(root)
    checkpoint = SparseCheckpoint(storage)
    source = _store(key_dtype)
    _assign(source, keys)
    committed = checkpoint.save_training_checkpoint(source, "four-types")

    manifest = (
        lance.dataset(str(root / SNAPSHOT_MANIFEST_DATASET)).to_table().to_pylist()[0]
    )
    assert manifest["key_dtype"] == key_dtype
    fragment_root = root / storage._manifest_fragment_root_for_participant(manifest, 0)
    expected_arrow_type = getattr(pa, key_dtype)()
    assert (
        lance.dataset(str(fragment_root / ENTRIES_DATASET)).schema.field("key").type
        == expected_arrow_type
    )

    physical = _store(key_dtype)
    checkpoint.load_training_checkpoint(
        physical,
        committed.handle,
        _target_restore(
            CheckpointParticipant(0, 1),
            key_dtype=key_dtype,
        ),
    )
    assert np.array_equal(_contains(physical, keys), np.ones(keys.size, dtype=bool))
    np.testing.assert_allclose(_values(physical, keys), _values(source, keys))

    elastic = _store(key_dtype)
    checkpoint.load_training_checkpoint(
        elastic,
        committed.handle,
        _elastic_options(key_dtype),
    )
    assert np.array_equal(_contains(elastic, keys), np.ones(keys.size, dtype=bool))
    np.testing.assert_allclose(_values(elastic, keys), _values(source, keys))


@pytest.mark.parametrize(("key_dtype", "keys"), _KEY_CASES)
def test_four_key_encodings_page_image_and_entry_restore_survive_compaction(
    tmp_path: Path, key_dtype: str, keys: np.ndarray
) -> None:
    storage = LocalLanceStorage(tmp_path / "checkpoint")
    checkpoint = SparseCheckpoint(storage)
    source = _store(key_dtype, optimizer=True)
    _assign(source, keys)
    committed = checkpoint.save_training_checkpoint(
        source, "page-types", include_page_images=True
    )
    compacted = storage.compact_checkpoint(committed.handle, include_page_images=True)

    page_restore = _store(key_dtype, optimizer=True)
    checkpoint.load_training_checkpoint(
        page_restore,
        compacted,
        _target_restore(
            CheckpointParticipant(0, 1),
            engine=IngestionEngine.FAST_PAGE_CACHE,
            key_dtype=key_dtype,
        ),
    )
    entry_restore = _store(key_dtype, optimizer=True)
    checkpoint.load_training_checkpoint(
        entry_restore,
        compacted,
        _target_restore(
            CheckpointParticipant(0, 1),
            key_dtype=key_dtype,
        ),
    )
    assert np.array_equal(_contains(page_restore, keys), _contains(entry_restore, keys))
    np.testing.assert_allclose(
        _values(page_restore, keys), _values(entry_restore, keys)
    )


@pytest.mark.parametrize(("key_dtype", "keys"), _KEY_CASES)
def test_four_key_encodings_weight_reference_versions_survive_compaction(
    tmp_path: Path, key_dtype: str, keys: np.ndarray
) -> None:
    root = tmp_path / "checkpoint"
    first_asset = tmp_path / "first"
    second_asset = tmp_path / "second"
    first_asset.mkdir()
    second_asset.mkdir()
    storage = LocalLanceStorage(root)
    checkpoint = SparseCheckpoint(storage)

    first_source = _store(key_dtype)
    second_source = _store(key_dtype)
    _assign(first_source, keys, offset=1)
    _assign(second_source, keys, offset=9)
    first = checkpoint.save_training_checkpoint(
        first_source, "inference-types", publish_weight_path=first_asset
    )
    second = checkpoint.save_training_checkpoint(
        second_source, "inference-types", publish_weight_path=second_asset
    )
    storage.compact_checkpoint(first.handle)
    storage.compact_checkpoint(second.handle)

    for asset, expected_source in (
        (first_asset, first_source),
        (second_asset, second_source),
    ):
        target = _store(key_dtype, optimizer=False)
        LocalLanceStorage.from_published_weight(asset).load_inference_weights(
            target, encode_restore_options(_elastic_options(key_dtype))
        )
        assert target.optimizer_enabled() is False
        assert np.array_equal(_contains(target, keys), np.ones(keys.size, dtype=bool))
        np.testing.assert_allclose(
            _values(target, keys), _values(expected_source, keys)
        )


@pytest.mark.parametrize("corruption", ("manifest", "entries", "page_images"))
def test_key_encoding_corruption_fails_before_target_mutation(
    tmp_path: Path, corruption: str
) -> None:
    root = tmp_path / "checkpoint"
    storage = LocalLanceStorage(root)
    checkpoint = SparseCheckpoint(storage)
    key_dtype = "int32"
    keys = np.asarray([-(1 << 31), -1, 0, 1], dtype=np.int32)
    source = _store(key_dtype)
    _assign(source, keys)
    committed = checkpoint.save_training_checkpoint(
        source, "corrupt-types", include_page_images=True
    )
    manifest_dataset = lance.dataset(str(root / SNAPSHOT_MANIFEST_DATASET))
    manifest = manifest_dataset.to_table().to_pylist()[0]
    fragment_root = root / storage._manifest_fragment_root_for_participant(manifest, 0)
    descriptor = json.loads(next(iter(manifest["fragment_descriptors"])))

    if corruption == "manifest":
        manifest_dataset.update(
            {"key_dtype": "'uint64'"},
            where="checkpoint_name = 'corrupt-types'",
        )
        expected = RuntimeError
    else:
        if corruption == "entries":
            dataset = lance.dataset(str(fragment_root / ENTRIES_DATASET))
            metadata = dict(dataset.schema_metadata)
            metadata["sparse_embedding.key_dtype"] = "unknown"
            dataset.update_schema_metadata(metadata, replace=True)
        else:
            page_dataset = root / str(descriptor["page_dataset"])
            dataset = lance.dataset(str(page_dataset))
            metadata = dict(dataset.schema_metadata)
            metadata["sparse_embedding.key_dtype"] = "uint64"
            dataset.update_schema_metadata(metadata, replace=True)
            latest_page_version = dataset.version
            descriptor["page_version"] = latest_page_version
            manifest["fragment_descriptors"] = [json.dumps(descriptor, sort_keys=True)]
            manifest["participant_page_versions"] = [latest_page_version]
            manifest["fragment_digest"] = lance_backend._canonical_digest([descriptor])
            manifest_dataset.update(
                {
                    "fragment_descriptors": "['"
                    + manifest["fragment_descriptors"][0].replace("'", "''")
                    + "']",
                    "participant_page_versions": f"[{latest_page_version}]",
                    "fragment_digest": f"'{manifest['fragment_digest']}'",
                },
                where="checkpoint_name = 'corrupt-types'",
            )
        expected = (
            (ValueError, InvalidCheckpointFragment)
            if corruption == "entries"
            else InvalidCheckpointFragment
        )

    target = _store(key_dtype)
    with pytest.raises(expected):
        checkpoint.load_training_checkpoint(
            target,
            committed.handle,
            _target_restore(
                CheckpointParticipant(0, 1),
                key_dtype=key_dtype,
            ),
        )
    assert target.is_operational() is True
