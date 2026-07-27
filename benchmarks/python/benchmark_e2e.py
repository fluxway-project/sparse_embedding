import asyncio
import tempfile
import time
from math import ceil
from statistics import median

import numpy as np

import sparse_embedding as se

RUNS = 5
BATCH_SIZE = 4096
DIM = 16


def make_store(spill_directory: str) -> se.EmbeddingStore:
    return se.EmbeddingStore(
        embedding_config=se.EmbeddingConfig(
            dims=[DIM],
            num_threads=4,
            storage=se.AutoStorage(
                memory_limit_bytes=256 << 10,
                spill_directory=spill_directory,
                max_spill_bytes=1 << 30,
            ),
        ),
        initializer_config=se.Zeros(),
        optimizer_config=se.Sgd(lr=0.01),
        score_policy="lfu",
    )


def require_eviction(store: se.EmbeddingStore) -> None:
    for _ in range(100):
        stats = store.storage_stats()
        if stats["swap_out_count"] > 0:
            return
        time.sleep(0.01)
    raise RuntimeError("training benchmark did not trigger background eviction")


def summarize(name: str, durations: list[float], operations: int) -> None:
    ordered = sorted(durations)
    p95 = ordered[max(0, ceil(len(ordered) * 0.95) - 1)]
    values = [operations / duration / 1e6 for duration in durations]
    print(f"{name}_median_mops={median(values):.6f}")
    print(f"{name}_p95_seconds={p95:.6f}")


def run_sync(keys: np.ndarray, values: np.ndarray, gradients: np.ndarray) -> None:
    with tempfile.TemporaryDirectory() as spill_directory:
        store = make_store(spill_directory)
        store.lookup_or_insert(keys, dim=DIM)
        require_eviction(store)
        for _ in range(2):
            store.lookup(keys, dim=DIM)
            store.assign(keys, values)
            store.accumulate_gradients(keys, gradients)
            store.optimizer_step()

        durations = []
        for _ in range(RUNS):
            started = time.perf_counter()
            store.lookup(keys, dim=DIM)
            store.assign(keys, values)
            store.accumulate_gradients(keys, gradients)
            store.optimizer_step()
            durations.append(time.perf_counter() - started)
        require_eviction(store)
        summarize("sync_training_step", durations, len(keys))


async def run_async_measurements(
    keys: np.ndarray, values: np.ndarray, gradients: np.ndarray
) -> None:
    with tempfile.TemporaryDirectory() as spill_directory:
        store = make_store(spill_directory)
        await store.async_lookup_or_insert(keys, dim=DIM)
        require_eviction(store)
        for _ in range(2):
            await store.async_lookup(keys, dim=DIM)
            await store.async_assign(keys, values)
            await store.async_accumulate_gradients(keys, gradients)
            await store.async_optimizer_step()

        lookup_durations = []
        assign_durations = []
        optimizer_durations = []
        for _ in range(RUNS):
            started = time.perf_counter()
            await store.async_lookup(keys, dim=DIM)
            lookup_durations.append(time.perf_counter() - started)

            started = time.perf_counter()
            await store.async_assign(keys, values)
            assign_durations.append(time.perf_counter() - started)

            await store.async_accumulate_gradients(keys, gradients)
            started = time.perf_counter()
            await store.async_optimizer_step()
            optimizer_durations.append(time.perf_counter() - started)

        require_eviction(store)
        summarize("async_lookup", lookup_durations, len(keys))
        summarize("async_assign", assign_durations, len(keys))
        summarize("async_optimizer_step", optimizer_durations, len(keys))


def main() -> None:
    keys = np.arange(BATCH_SIZE, dtype=np.int64)
    values = np.full((BATCH_SIZE, DIM), 0.01, dtype=np.float32)
    gradients = np.full((BATCH_SIZE, DIM), 0.001, dtype=np.float32)
    run_sync(keys, values, gradients)
    asyncio.run(run_async_measurements(keys, values, gradients))


if __name__ == "__main__":
    main()
