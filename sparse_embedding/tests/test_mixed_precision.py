"""
Mixed-precision tests covering all 9 (param_dtype, optimizer_dtype) combinations.

Validates:
 1. assign / lookup data integrity
 2. Adam optimizer step correctness (bias correction with per-ID step)
 3. Checkpoint save/load round-trip (C++ Lance path)
"""

import numpy as np
import pytest

import sparse_embedding as se
from sparse_embedding.checkpoint import (
    PartitionSpec,
    RestorePlan,
    SparseCheckpoint,
)
from sparse_embedding.checkpoint.lance_backend import LocalLanceStorage


def _modulo_restore(partition_count: int, partition_id: int) -> RestorePlan:
    return RestorePlan.from_partition(
        PartitionSpec.modulo(),
        partition_count,
        partition_id,
        key_dtype="uint64",
    )


# ---------------------------------------------------------------------------
# DLPack-compatible array helpers
# ---------------------------------------------------------------------------

# bfloat16 is not natively in numpy. We use JAX CPU arrays for DLPack.
try:
    import jax

    jax.config.update("jax_platforms", "cpu")
    import jax.numpy as jnp

    _HAS_JAX = True
except (ImportError, RuntimeError):
    _HAS_JAX = False


def _make_array(shape, dtype_str: str, rng: np.random.Generator):
    """Create a random DLPack-compatible array in the given dtype."""
    raw = rng.standard_normal(shape).astype(np.float32)
    if dtype_str in ("float32", "f32"):
        return raw
    elif dtype_str in ("float16", "f16"):
        return raw.astype(np.float16)
    elif dtype_str in ("bfloat16", "bf16"):
        if not _HAS_JAX:
            pytest.skip("JAX not available; required for bfloat16 DLPack arrays")
        return jnp.array(raw, dtype=jnp.bfloat16)
    else:
        raise ValueError(f"unknown dtype: {dtype_str}")


def _to_numpy_f32(dlpack_obj, shape, param_dtype="float32"):
    """Convert DLPack output back to float32 numpy for comparison."""
    if param_dtype in ("bfloat16", "bf16"):
        # bfloat16 DLPack can't go through numpy; use JAX directly
        if not _HAS_JAX:
            pytest.skip("JAX not available for bfloat16 DLPack conversion")
        arr = np.asarray(jnp.from_dlpack(dlpack_obj), dtype=np.float32)
    else:
        arr = np.from_dlpack(dlpack_obj)
    arr = arr.reshape(shape)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    return arr


# ---------------------------------------------------------------------------
# Parametrize all 9 (param, opt) combinations
# ---------------------------------------------------------------------------

DTYPES = ["float32", "float16", "bfloat16"]
ALL_COMBOS = [(p, o) for p in DTYPES for o in DTYPES]


def _combo_id(combo):
    short = {"float32": "f32", "float16": "f16", "bfloat16": "bf16"}
    return f"{short[combo[0]]}_{short[combo[1]]}"


ALL_COMBO_IDS = [_combo_id(c) for c in ALL_COMBOS]
GRADIENT_COMBOS = [
    (param_dtype, optimizer_dtype, gradient_dtype)
    for param_dtype in DTYPES
    for optimizer_dtype in DTYPES
    for gradient_dtype in DTYPES
]
GRADIENT_COMBO_IDS = [
    f"{_combo_id((p, o))}_grad_{g.replace('float', 'f').replace('b', 'b')}"
    for p, o, g in GRADIENT_COMBOS
]


# ---------------------------------------------------------------------------
# Test 1: Basic assign / lookup data integrity
# ---------------------------------------------------------------------------


# Lance materialized rows are currently defined as float32 by the protocol.
# Keep this integration test aligned with that schema; low-precision arithmetic
# remains covered by the non-checkpoint tests in this module.
@pytest.mark.parametrize("param_dtype,optimizer_dtype", ALL_COMBOS, ids=ALL_COMBO_IDS)
def test_assign_lookup_integrity(param_dtype, optimizer_dtype):
    """Values written via assign() should be exactly readable via lookup()."""
    if "bfloat16" in (param_dtype, optimizer_dtype) and not _HAS_JAX:
        pytest.skip("JAX required for bfloat16")

    dim = 8
    n_keys = 20
    rng = np.random.default_rng(42)

    cfg = se.EmbeddingConfig(
        dims=[dim],
        thread_budget=1,
        value_dtype=param_dtype,
        optimizer_dtype=optimizer_dtype,
        key_dtype="uint64",
    )
    store = se.EmbeddingStore(cfg)

    keys = np.arange(n_keys, dtype=np.uint64)
    vals = _make_array((n_keys, dim), param_dtype, rng)

    store.assign(keys, vals)
    assert store.total_size() == n_keys

    out_dlp, _ = store.lookup(keys, dim=dim)
    got = _to_numpy_f32(out_dlp, (n_keys, dim), param_dtype)
    expected = (
        np.asarray(vals, dtype=np.float32)
        if hasattr(vals, "__array__")
        else np.asarray(vals).astype(np.float32)
    )

    # float16/bfloat16 have limited precision (~3 decimal digits)
    atol = 1e-2 if param_dtype != "float32" else 1e-5
    np.testing.assert_allclose(got, expected, atol=atol)


# ---------------------------------------------------------------------------
# Test 2: Adam optimizer step with mixed precision
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("param_dtype,optimizer_dtype", ALL_COMBOS, ids=ALL_COMBO_IDS)
def test_adam_optimizer_step(param_dtype, optimizer_dtype):
    """Adam should produce non-trivial updates; per-ID step should work."""
    if "bfloat16" in (param_dtype, optimizer_dtype) and not _HAS_JAX:
        pytest.skip("JAX required for bfloat16")

    dim = 8
    rng = np.random.default_rng(123)

    cfg = se.EmbeddingConfig(
        dims=[dim],
        thread_budget=1,
        value_dtype=param_dtype,
        optimizer_dtype=optimizer_dtype,
        key_dtype="uint64",
    )
    store = se.EmbeddingStore(
        cfg,
        optimizer_config=se.Adam(lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8),
        initializer_config=se.Zeros(),
    )

    keys = np.array([0], dtype=np.uint64)
    store.lookup_or_insert(keys, dim=dim)

    grads = _make_array((1, dim), param_dtype, rng)
    store.accumulate_gradients(keys, grads)
    store.optimizer_step(1)

    out_dlp, _ = store.lookup(keys, dim=dim)
    got = _to_numpy_f32(out_dlp, (1, dim), param_dtype)
    assert np.all(got != 0.0), "optimizer step should produce non-zero values"

    keys2 = np.array([1], dtype=np.uint64)
    store.lookup_or_insert(keys2, dim=dim)
    for _ in range(50):
        store.accumulate_gradients(keys, grads)
        store.optimizer_step(1)
    grads2 = _make_array((1, dim), param_dtype, rng)
    store.accumulate_gradients(keys2, grads2)
    store.optimizer_step(1)
    out1_dlp, _ = store.lookup(keys2, dim=dim)
    val_key1 = _to_numpy_f32(out1_dlp, (1, dim), param_dtype)
    assert np.abs(val_key1).max() > 0.005


@pytest.mark.asyncio
@pytest.mark.parametrize("key_dtype", ["int32", "uint32", "int64", "uint64"])
async def test_async_accumulate_preserves_key_width(key_dtype):
    cfg = se.EmbeddingConfig(
        dims=[2],
        thread_budget=1,
        value_dtype="float32",
        optimizer_dtype="float32",
        gradient_dtype="float32",
        key_dtype=key_dtype,
    )
    store = se.EmbeddingStore(
        cfg, optimizer_config=se.Sgd(lr=1.0), initializer_config=se.Zeros()
    )
    keys = np.array([1, 2], dtype=np.dtype(key_dtype))
    store.lookup_or_insert(keys, dim=2)
    await store.async_accumulate_gradients(keys, np.ones((2, 2), dtype=np.float32))
    store.optimizer_step(1)
    out, _ = store.lookup(keys, dim=2)
    np.testing.assert_allclose(np.from_dlpack(out), -1.0)


# ---------------------------------------------------------------------------
# Test 3: Checkpoint round-trip with optimizer state
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("param_dtype,optimizer_dtype", ALL_COMBOS, ids=ALL_COMBO_IDS)
def test_checkpoint_roundtrip(param_dtype, optimizer_dtype, tmp_path):
    """Training checkpoint round-trip preserves embeddings and optimizer state."""
    if "bfloat16" in (param_dtype, optimizer_dtype) and not _HAS_JAX:
        pytest.skip("JAX required for bfloat16")

    dim = 8
    n_keys = 50
    rng = np.random.default_rng(7)

    cfg = se.EmbeddingConfig(
        dims=[dim],
        thread_budget=1,
        value_dtype=param_dtype,
        optimizer_dtype=optimizer_dtype,
        key_dtype="uint64",
    )
    adam = se.Adam(lr=0.01)

    store1 = se.EmbeddingStore(
        cfg, optimizer_config=adam, initializer_config=se.Zeros()
    )
    keys = np.arange(n_keys, dtype=np.uint64)
    store1.lookup_or_insert(keys, dim=dim)

    # Perform a few optimizer steps to populate opt state
    for _ in range(3):
        grads = _make_array((n_keys, dim), param_dtype, rng)
        store1.accumulate_gradients(keys, grads)
        store1.optimizer_step(1)

    uri = str(tmp_path / "ckpt")
    checkpoint = SparseCheckpoint(LocalLanceStorage(uri))
    locator = checkpoint.save_training_checkpoint(store1, "mixed")

    # Load into a fresh store
    store2 = se.EmbeddingStore(cfg, optimizer_config=adam)
    checkpoint.load_training_checkpoint(
        store2,
        locator.handle,
        _modulo_restore(1, 0),
    )

    assert store2.total_size() == n_keys

    # Verify embeddings match
    out1_dlp, _ = store1.lookup(keys, dim=dim)
    out2_dlp, _ = store2.lookup(keys, dim=dim)
    v1 = _to_numpy_f32(out1_dlp, (n_keys, dim), param_dtype)
    v2 = _to_numpy_f32(out2_dlp, (n_keys, dim), param_dtype)

    atol = 1e-2 if param_dtype != "float32" else 1e-5
    np.testing.assert_allclose(
        v2, v1, atol=atol, err_msg="embedding mismatch after load"
    )

    gradients = _make_array((n_keys, dim), param_dtype, np.random.default_rng(999))
    store1.accumulate_gradients(keys, gradients)
    store1.optimizer_step(1)
    store2.accumulate_gradients(keys, gradients)
    store2.optimizer_step(1)

    after1, _ = store1.lookup(keys, dim=dim)
    after2, _ = store2.lookup(keys, dim=dim)
    values1 = _to_numpy_f32(after1, (n_keys, dim), param_dtype)
    values2 = _to_numpy_f32(after2, (n_keys, dim), param_dtype)
    optimizer_tolerance = (
        0.05
        if "float16" in (param_dtype, optimizer_dtype)
        or "bfloat16" in (param_dtype, optimizer_dtype)
        else 1e-4
    )
    np.testing.assert_allclose(
        values2,
        values1,
        atol=optimizer_tolerance,
        err_msg="optimizer state diverged after checkpoint restore",
    )


@pytest.mark.parametrize(
    "param_dtype,optimizer_dtype,gradient_dtype",
    GRADIENT_COMBOS,
    ids=GRADIENT_COMBO_IDS,
)
def test_independent_gradient_accumulator_dtype(
    param_dtype, optimizer_dtype, gradient_dtype
):
    """Gradient storage dtype is independent from params and optimizer state."""
    if "bfloat16" in (param_dtype, optimizer_dtype, gradient_dtype) and not _HAS_JAX:
        pytest.skip("JAX required for bfloat16")

    cfg = se.EmbeddingConfig(
        dims=[4],
        thread_budget=1,
        value_dtype=param_dtype,
        optimizer_dtype=optimizer_dtype,
        gradient_dtype=gradient_dtype,
        key_dtype="uint64",
    )
    store = se.EmbeddingStore(
        cfg,
        optimizer_config=se.Sgd(lr=0.01),
        initializer_config=se.Zeros(),
    )
    keys = np.array([7], dtype=np.uint64)
    store.lookup_or_insert(keys, dim=4)
    gradients = _make_array((1, 4), gradient_dtype, np.random.default_rng(17))
    store.accumulate_gradients(keys, gradients)
    store.optimizer_step(1)

    out, _ = store.lookup(keys, dim=4)
    values = _to_numpy_f32(out, (1, 4), param_dtype)
    assert np.all(np.isfinite(values))
    assert np.any(values != 0.0)


@pytest.mark.parametrize(
    "param_dtype,gradient_dtype",
    [("float16", "float32"), ("float32", "float16"), ("bfloat16", "float32")],
)
@pytest.mark.asyncio
async def test_async_accumulate_accepts_configured_gradient_dtype(
    param_dtype, gradient_dtype
):
    if "bfloat16" in (param_dtype, gradient_dtype) and not _HAS_JAX:
        pytest.skip("JAX required for bfloat16")

    cfg = se.EmbeddingConfig(
        dims=[4],
        thread_budget=1,
        value_dtype=param_dtype,
        optimizer_dtype="float32",
        gradient_dtype=gradient_dtype,
        key_dtype="uint64",
    )
    store = se.EmbeddingStore(
        cfg,
        optimizer_config=se.Sgd(lr=0.01),
        initializer_config=se.Zeros(),
    )
    keys = np.array([8], dtype=np.uint64)
    store.lookup_or_insert(keys, dim=4)
    gradients = _make_array((1, 4), gradient_dtype, np.random.default_rng(23))
    await store.async_accumulate_gradients(keys, gradients)
    store.optimizer_step(1)


@pytest.mark.parametrize("gradient_scale", [1.0, 8.0])
def test_gradient_scale_round_trip(gradient_scale):
    """Scaling the stored FP16 gradient does not change the optimizer update."""
    key = np.array([11], dtype=np.uint64)
    grad = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float16)
    values = []
    for scale in (1.0, gradient_scale):
        cfg = se.EmbeddingConfig(
            dims=[4],
            thread_budget=1,
            value_dtype="float16",
            optimizer_dtype="float32",
            gradient_dtype="float16",
            gradient_scale=scale,
            key_dtype="uint64",
        )
        store = se.EmbeddingStore(
            cfg,
            optimizer_config=se.Sgd(lr=0.01),
            initializer_config=se.Zeros(),
        )
        store.lookup_or_insert(key, dim=4)
        store.accumulate_gradients(key, grad)
        store.optimizer_step(1)
        out, _ = store.lookup(key, dim=4)
        values.append(np.from_dlpack(out).astype(np.float32))
    np.testing.assert_allclose(values[0], values[1], atol=2e-4)


@pytest.mark.parametrize("normalization_steps", [1, 2, 3])
@pytest.mark.parametrize(
    "gradient_scale,gradient_dtype", [(1.0, "float32"), (8.0, "float16")]
)
def test_optimizer_step_normalizes_actual_accumulation_window(
    normalization_steps, gradient_scale, gradient_dtype
):
    keys = np.asarray([17], dtype=np.int64)
    config = se.EmbeddingConfig(
        dims=[1],
        thread_budget=1,
        value_dtype="float32",
        optimizer_dtype="float32",
        gradient_dtype=gradient_dtype,
        gradient_scale=gradient_scale,
        key_dtype="int64",
    )
    store = se.EmbeddingStore(
        config,
        optimizer_config=se.Sgd(lr=1.0),
        initializer_config=se.Zeros(),
    )
    store.lookup_or_insert(keys, dim=1)
    gradient = np.asarray([[3.0]], dtype=np.dtype(gradient_dtype))
    for _ in range(normalization_steps):
        store.accumulate_gradients(keys, gradient)

    with pytest.raises(ValueError, match="normalization_steps"):
        store.optimizer_step(normalization_steps=0)

    store.optimizer_step(normalization_steps=normalization_steps)
    values, _ = store.lookup(keys, dim=1)
    actual = np.from_dlpack(values).reshape(-1)
    np.testing.assert_allclose(actual, [-3.0], atol=2e-3)


@pytest.mark.parametrize("gradient_scale", [0.0, -1.0, float("nan"), float("inf")])
def test_gradient_scale_rejects_invalid_values(gradient_scale):
    with pytest.raises(ValueError, match="gradient_scale"):
        se.EmbeddingStore(
            se.EmbeddingConfig(
                dims=[2],
                thread_budget=1,
                value_dtype="float16",
                optimizer_dtype="float32",
                gradient_dtype="float16",
                gradient_scale=gradient_scale,
                key_dtype="uint64",
            ),
            optimizer_config=se.Sgd(lr=0.01),
        )


def test_wider_gradient_dtype_rejects_parameter_overflow():
    cfg = se.EmbeddingConfig(
        dims=[1],
        thread_budget=1,
        value_dtype="float16",
        optimizer_dtype="float32",
        gradient_dtype="float32",
        key_dtype="uint64",
    )
    store = se.EmbeddingStore(
        cfg, optimizer_config=se.Sgd(lr=1.0), initializer_config=se.Zeros()
    )
    key = np.array([1], dtype=np.uint64)
    store.lookup_or_insert(key, dim=1)
    with pytest.raises((OverflowError, ValueError), match="gradient"):
        store.accumulate_gradients(key, np.array([[100000.0]], dtype=np.float32))
