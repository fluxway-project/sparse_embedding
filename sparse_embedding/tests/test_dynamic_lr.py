import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import sparse_embedding as se

DIM = 16


def make_store(policy, optimizer_type=se.Sgd, score_policy="manual", thread_budget=1):
    return se.EmbeddingStore(
        embedding_config=se.EmbeddingConfig(
            dims=[DIM], thread_budget=thread_budget, value_dtype="float32"
        ),
        initializer_config=se.Zeros(),
        optimizer_config=optimizer_type(lr=1.0, dynamic_lr_policy=policy),
        score_policy=score_policy,
    )


def update_for_score(policy, score, optimizer_type=se.Sgd):
    store = make_store(policy, optimizer_type)
    keys = np.array([7], dtype=np.int64)
    store.lookup_or_insert(keys, dim=DIM)
    store.assign_scores(keys, np.array([score], dtype=np.uint64))
    store.accumulate_gradients(keys, np.ones((1, DIM), dtype=np.float32))
    store.optimizer_step(1)
    return -np.from_dlpack(store.lookup(keys, dim=DIM)[0]).copy()


@pytest.mark.parametrize(
    ("policy", "score", "expected"),
    [
        (se.ConstantLrPolicy(), 37, 1.0),
        (se.SqrtDecayLrPolicy(), 0, 1.0),
        (se.SqrtDecayLrPolicy(), 4, 0.5),
        (se.LogDecayLrPolicy(), 2, 0.5),
        (se.InverseDecayLrPolicy(), 0, 1.0),
        (se.InverseDecayLrPolicy(), 4, 0.25),
        (se.CosineFrequencyLrPolicy(min_lr_ratio=0.2, max_score=4), 0, 1.0),
        (se.CosineFrequencyLrPolicy(min_lr_ratio=0.2, max_score=4), 2, 0.6),
        (se.CosineFrequencyLrPolicy(min_lr_ratio=0.2, max_score=4), 4, 0.2),
        (se.CosineFrequencyLrPolicy(min_lr_ratio=0.2, max_score=4), 5, 0.2),
        (se.CustomLutLrPolicy(lut=[0.25, 0.5, 0.75]), 0, 0.25),
        (se.CustomLutLrPolicy(lut=[0.25, 0.5, 0.75]), 2, 0.75),
        (se.CustomLutLrPolicy(lut=[0.25, 0.5, 0.75]), 3, 0.75),
    ],
)
def test_sgd_policy_exact_update(policy, score, expected):
    np.testing.assert_allclose(update_for_score(policy, score), expected, atol=1e-6)


@pytest.mark.parametrize(
    "policy",
    [
        se.ConstantLrPolicy(),
        se.SqrtDecayLrPolicy(),
        se.LogDecayLrPolicy(),
        se.InverseDecayLrPolicy(),
        se.CosineFrequencyLrPolicy(),
        se.CustomLutLrPolicy(lut=[1.0]),
    ],
)
def test_policy_is_supported_by_adam(policy):
    update = update_for_score(policy, 1, se.Adam)
    assert np.all(np.isfinite(update))
    assert np.all(update > 0)


def test_adam_applies_exact_dynamic_lr_across_steps():
    store = make_store(se.InverseDecayLrPolicy(), se.Adam)
    keys = np.array([11], dtype=np.int64)
    store.lookup_or_insert(keys, dim=DIM)
    store.assign_scores(keys, np.array([4], dtype=np.uint64))
    gradients = np.ones((1, DIM), dtype=np.float32)
    for _ in range(2):
        store.accumulate_gradients(keys, gradients)
        store.optimizer_step(1)
    np.testing.assert_allclose(
        np.from_dlpack(store.lookup(keys, dim=DIM)[0]), -0.5, atol=2e-5
    )


@pytest.mark.parametrize("score_policy", ["lfu", "lru"])
def test_runtime_score_policies_drive_learning_rate(score_policy):
    store = make_store(
        se.CosineFrequencyLrPolicy(min_lr_ratio=0.1, max_score=100),
        score_policy=score_policy,
    )
    hot = np.array([1], dtype=np.int64)
    cold = np.array([2], dtype=np.int64)
    store.lookup_or_insert(hot, dim=DIM)
    store.lookup_or_insert(cold, dim=DIM)
    for _ in range(20):
        store.lookup(hot, dim=DIM)
    gradients = np.ones((1, DIM), dtype=np.float32)
    store.accumulate_gradients(hot, gradients)
    store.accumulate_gradients(cold, gradients)
    store.optimizer_step(1)
    hot_update = -np.from_dlpack(store.lookup(hot, dim=DIM)[0]).copy()
    cold_update = -np.from_dlpack(store.lookup(cold, dim=DIM)[0]).copy()
    assert np.all(hot_update < cold_update)


@pytest.mark.asyncio
async def test_async_optimizer_step_uses_dynamic_lr():
    store = make_store(se.CustomLutLrPolicy(lut=[0.25]))
    keys = np.array([5], dtype=np.int64)
    store.lookup_or_insert(keys, dim=DIM)
    store.accumulate_gradients(keys, np.ones((1, DIM), dtype=np.float32))
    await store.async_optimizer_step(1)
    np.testing.assert_allclose(
        np.from_dlpack(store.lookup(keys, dim=DIM)[0]), -0.25, atol=1e-6
    )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: se.CustomLutLrPolicy(lut=[]),
        lambda: se.CustomLutLrPolicy(lut=[-0.1]),
        lambda: se.CustomLutLrPolicy(lut=[math.nan]),
        lambda: se.CosineFrequencyLrPolicy(min_lr_ratio=-0.1),
        lambda: se.CosineFrequencyLrPolicy(min_lr_ratio=1.1),
        lambda: se.CosineFrequencyLrPolicy(min_lr_ratio=math.nan),
        lambda: se.CosineFrequencyLrPolicy(max_score=0),
    ],
)
def test_invalid_policy_configuration_is_rejected(factory):
    with pytest.raises(ValueError):
        factory()


@pytest.mark.parametrize(
    "policy",
    [
        se.SqrtDecayLrPolicy(),
        se.LogDecayLrPolicy(),
        se.InverseDecayLrPolicy(),
        se.CosineFrequencyLrPolicy(),
        se.CustomLutLrPolicy(lut=[1.0]),
    ],
)
def test_score_dependent_policy_requires_score_storage(policy):
    with pytest.raises(ValueError, match="requires a score policy"):
        make_store(policy, score_policy="none")


def test_constant_policy_does_not_allocate_score_storage():
    store = make_store(se.ConstantLrPolicy(), score_policy="none")
    keys = np.array([1], dtype=np.int64)
    store.lookup_or_insert(keys, dim=DIM)
    store.accumulate_gradients(keys, np.ones((1, DIM), dtype=np.float32))
    store.optimizer_step(1)
    np.testing.assert_allclose(
        np.from_dlpack(store.lookup(keys, dim=DIM)[0]), -1.0, atol=1e-6
    )


def test_default_optimizer_uses_static_learning_rate_path():
    store = se.EmbeddingStore(
        embedding_config=se.EmbeddingConfig(
            dims=[DIM], thread_budget=1, value_dtype="float32"
        ),
        initializer_config=se.Zeros(),
        optimizer_config=se.Sgd(lr=0.25),
        score_policy="none",
    )
    keys = np.array([9], dtype=np.int64)
    store.lookup_or_insert(keys, dim=DIM)
    store.accumulate_gradients(keys, np.ones((1, DIM), dtype=np.float32))
    store.optimizer_step(1)
    np.testing.assert_allclose(
        np.from_dlpack(store.lookup(keys, dim=DIM)[0]), -0.25, atol=1e-6
    )


def test_reset_preserves_dynamic_policy():
    store = make_store(se.CustomLutLrPolicy(lut=[0.25]))
    store.clear()
    keys = np.array([3], dtype=np.int64)
    store.lookup_or_insert(keys, dim=DIM)
    store.accumulate_gradients(keys, np.ones((1, DIM), dtype=np.float32))
    store.optimizer_step(1)
    np.testing.assert_allclose(
        np.from_dlpack(store.lookup(keys, dim=DIM)[0]), -0.25, atol=1e-6
    )


def test_parallel_score_updates_and_optimizer_step():
    key_count = 2048
    store = make_store(se.CustomLutLrPolicy(lut=[0.25, 0.5, 1.0]), thread_budget=8)
    keys = np.arange(key_count, dtype=np.int64)
    store.lookup_or_insert(keys, dim=DIM)
    store.assign_scores(keys, (keys % 3).astype(np.uint64))

    def accumulate(chunk):
        store.accumulate_gradients(chunk, np.ones((len(chunk), DIM), dtype=np.float32))

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(accumulate, np.array_split(keys, 8)))
    store.optimizer_step(1)

    actual = (
        -np.from_dlpack(store.lookup(keys, dim=DIM)[0]).copy().reshape(-1, DIM)[:, 0]
    )
    expected = np.take(np.array([0.25, 0.5, 1.0]), keys % 3)
    np.testing.assert_allclose(actual, expected, atol=1e-6)
