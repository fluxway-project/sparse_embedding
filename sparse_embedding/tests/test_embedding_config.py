import numpy as np

from sparse_embedding import AutoStorage, EmbeddingConfig


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
