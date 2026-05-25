"""
Mixed-precision tests covering all 9 (param_dtype, opt_dtype) combinations.

Validates:
 1. assign / lookup data integrity
 2. Adam optimizer step correctness (bias correction with per-ID step)
 3. Checkpoint save/load round-trip (C++ Lance path)
"""

import numpy as np
import pytest

import sparse_embedding as se

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


# ---------------------------------------------------------------------------
# Test 1: Basic assign / lookup data integrity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("param_dtype,opt_dtype", ALL_COMBOS, ids=ALL_COMBO_IDS)
def test_assign_lookup_integrity(param_dtype, opt_dtype):
    """Values written via assign() should be exactly readable via lookup()."""
    if "bfloat16" in (param_dtype, opt_dtype) and not _HAS_JAX:
        pytest.skip("JAX required for bfloat16")

    dim = 8
    n_keys = 20
    rng = np.random.default_rng(42)

    cfg = se.EmbeddingConfig(
        dims=[dim], num_threads=1, value_dtype=param_dtype, optimizer_dtype=opt_dtype
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


@pytest.mark.parametrize("param_dtype,opt_dtype", ALL_COMBOS, ids=ALL_COMBO_IDS)
def test_adam_optimizer_step(param_dtype, opt_dtype):
    """Adam should produce non-trivial updates; per-ID step should work."""
    if "bfloat16" in (param_dtype, opt_dtype) and not _HAS_JAX:
        pytest.skip("JAX required for bfloat16")

    dim = 8
    rng = np.random.default_rng(123)

    cfg = se.EmbeddingConfig(
        dims=[dim], num_threads=1, value_dtype=param_dtype, optimizer_dtype=opt_dtype
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
    store.optimizer_step()

    out_dlp, _ = store.lookup(keys, dim=dim)
    got = _to_numpy_f32(out_dlp, (1, dim), param_dtype)

    # After one Adam step on zero-initialized params with unit-ish grads,
    # result should be approximately -lr = -0.01
    assert np.all(got != 0.0), "optimizer step should produce non-zero values"

    # Verify per-ID step independence: a new ID gets a fresh step counter
    keys2 = np.array([1], dtype=np.uint64)
    store.lookup_or_insert(keys2, dim=dim)

    # advance 50 steps on key 0 only
    for _ in range(50):
        store.accumulate_gradients(keys, grads)
        store.optimizer_step()

    # first step on key 1
    grads2 = _make_array((1, dim), param_dtype, rng)
    store.accumulate_gradients(keys2, grads2)
    store.optimizer_step()

    out1_dlp, _ = store.lookup(keys2, dim=dim)
    val_key1 = _to_numpy_f32(out1_dlp, (1, dim), param_dtype)
    # Result should be ~-lr (first step bias correction), not tiny
    assert np.abs(val_key1).max() > 0.005, (
        f"per-ID step seems broken: max abs = {np.abs(val_key1).max()}"
    )


# ---------------------------------------------------------------------------
# Test 3: Checkpoint round-trip with optimizer state
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("param_dtype,opt_dtype", ALL_COMBOS, ids=ALL_COMBO_IDS)
def test_checkpoint_roundtrip(param_dtype, opt_dtype, tmp_path):
    """save_checkpoint -> load_checkpoint should preserve embeddings and opt state."""
    if "bfloat16" in (param_dtype, opt_dtype) and not _HAS_JAX:
        pytest.skip("JAX required for bfloat16")

    dim = 8
    n_keys = 50
    rng = np.random.default_rng(7)

    cfg = se.EmbeddingConfig(
        dims=[dim], num_threads=1, value_dtype=param_dtype, optimizer_dtype=opt_dtype
    )
    adam = se.Adam(lr=0.01)

    store1 = se.EmbeddingStore(
        cfg, optimizer_config=adam, initializer_config=se.Zeros()
    )
    if not hasattr(store1, "set_checkpoint_factories"):
        pytest.skip("Lance checkpoint not compiled in")
    try:
        store1.set_checkpoint_factories("lance")
    except Exception:
        pytest.skip("Lance checkpoint backend not configured")

    keys = np.arange(n_keys, dtype=np.uint64)
    store1.lookup_or_insert(keys, dim=dim)

    # Perform a few optimizer steps to populate opt state
    for _ in range(3):
        grads = _make_array((n_keys, dim), param_dtype, rng)
        store1.accumulate_gradients(keys, grads)
        store1.optimizer_step()

    uri = str(tmp_path / "ckpt")
    store1.save_checkpoint(uri)

    # Load into a fresh store
    store2 = se.EmbeddingStore(cfg, optimizer_config=adam)
    if hasattr(store2, "set_checkpoint_factories"):
        store2.set_checkpoint_factories("lance")
    store2.load_checkpoint(uri)

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

    # Verify optimizer state is preserved: one more step should produce same result
    grads_check = _make_array((n_keys, dim), param_dtype, np.random.default_rng(999))
    store1.accumulate_gradients(keys, grads_check)
    store1.optimizer_step()
    store2.accumulate_gradients(keys, grads_check)
    store2.optimizer_step()

    out1_dlp, _ = store1.lookup(keys, dim=dim)
    out2_dlp, _ = store2.lookup(keys, dim=dim)
    v1_after = _to_numpy_f32(out1_dlp, (n_keys, dim), param_dtype)
    v2_after = _to_numpy_f32(out2_dlp, (n_keys, dim), param_dtype)

    # Allow slightly larger tolerance for low-precision combos
    atol_opt = (
        0.05
        if "float16" in (param_dtype, opt_dtype)
        or "bfloat16" in (param_dtype, opt_dtype)
        else 1e-4
    )
    np.testing.assert_allclose(
        v2_after, v1_after, atol=atol_opt, err_msg="post-load optimizer divergence"
    )
