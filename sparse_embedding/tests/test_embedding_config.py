import numpy as np

from sparse_embedding import AutoStorage, EmbeddingConfig, StorageTuningOptions


def test_embedding_config_default():
    cfg = EmbeddingConfig(dims=[128])
    assert cfg.value_dtype == "float32"
    assert cfg.optimizer_dtype == ""
    assert cfg.key_dtype == "int64"
    assert isinstance(cfg.storage, AutoStorage)


def test_embedding_config_native_types():
    cfg = EmbeddingConfig(
        dims=[128], value_dtype=float, optimizer_dtype=None, key_dtype=int
    )
    assert cfg.value_dtype == "float32"
    assert cfg.optimizer_dtype == ""
    assert cfg.key_dtype == "int64"


def test_embedding_config_numpy_types():
    cfg = EmbeddingConfig(
        dims=[128],
        value_dtype=np.float16,
        optimizer_dtype=np.float32,
        key_dtype=np.uint64,
    )
    assert cfg.value_dtype == "float16"
    assert cfg.optimizer_dtype == "float32"
    assert cfg.key_dtype == "uint64"


def test_embedding_config_string_types():
    cfg = EmbeddingConfig(
        dims=[128], value_dtype="bfloat16", optimizer_dtype="f32", key_dtype="uint32"
    )
    assert cfg.value_dtype == "bfloat16"
    assert cfg.optimizer_dtype == "f32"
    assert cfg.key_dtype == "uint32"


def test_embedding_config_auto_spill(tmp_path):
    storage = AutoStorage(
        memory_limit_bytes=0, spill_directory=tmp_path, max_spill_bytes=4 << 20
    )
    cfg = EmbeddingConfig(dims=[8], thread_budget=1, storage=storage)

    assert cfg.storage is storage


def test_embedding_config_auto_tuning_parameters():
    cfg = EmbeddingConfig(
        dims=[64],
        storage_tuning=StorageTuningOptions(
            target_capacity_keys=10_000_000,
            memory_budget_bytes=2 * 1024 * 1024 * 1024,
            entries_per_page=8192,
            max_pages=16384,
        ),
    )
    assert cfg.storage_tuning.target_capacity_keys == 10_000_000
    assert cfg.storage_tuning.memory_budget_bytes == 2 * 1024 * 1024 * 1024
    assert cfg.storage_tuning.entries_per_page == 8192
    assert cfg.storage_tuning.max_pages == 16384


def test_embedding_table_with_auto_tuned_topology():
    import numpy as np

    from sparse_embedding import EmbeddingStore, Zeros

    cfg = EmbeddingConfig(
        dims=[32],
        storage_tuning=StorageTuningOptions(
            target_capacity_keys=100_000, entries_per_page=4096
        ),
    )
    store = EmbeddingStore(
        embedding_config=cfg,
        initializer_config=Zeros(),
    )
    keys = np.arange(1000, dtype=np.int64)
    values = np.random.randn(1000, 32).astype(np.float32)

    store.assign(keys, values)
    out_vals, out_lens = store.lookup(keys, dim=32)
    np.testing.assert_allclose(
        np.from_dlpack(out_vals).reshape(1000, 32), values, rtol=1e-5
    )
    np.testing.assert_array_equal(
        np.from_dlpack(out_lens), np.full(1000, 32, dtype=np.int64)
    )
