import tempfile
import time
import warnings
from math import ceil
from statistics import median

import numpy as np

import sparse_embedding as se


def run_once() -> dict[str, int | float]:
    with tempfile.TemporaryDirectory() as spill_directory:
        store = se.EmbeddingStore(
            embedding_config=se.EmbeddingConfig(
                dims=[128],
                num_threads=2,
                max_active_keys=20_000,
                storage=se.AutoStorage(
                    memory_limit_bytes=5 << 20,
                    spill_directory=spill_directory,
                    max_spill_bytes=1 << 30,
                ),
            ),
            initializer_config=se.Zeros(),
            score_policy="lfu",
        )
        keys = np.arange(16_384, dtype=np.int64)
        start = time.perf_counter()
        store.lookup_or_insert(keys, dim=128)
        elapsed = time.perf_counter() - start
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            stats = store.storage_stats()
        for _ in range(100):
            if stats["swap_out_count"] > 0:
                break
            time.sleep(0.01)
            stats = store.storage_stats()
        if stats["swap_out_count"] == 0 or stats["eviction_selection_count"] == 0:
            raise RuntimeError("storage pressure benchmark did not trigger eviction")
        return {
            "mops": len(keys) / elapsed / 1e6,
            "swap_out_count": stats["swap_out_count"],
            "selection_count": stats["eviction_selection_count"],
            "evaluated_pages": stats["eviction_evaluated_pages"],
            "selection_ns": stats["eviction_selection_ns"],
            "invalidated_count": stats["eviction_candidate_invalidated_count"],
        }


def main() -> None:
    runs = []
    for run in range(1, 6):
        result = run_once()
        runs.append(result)
        print(f"run={run} {result}")
    for key in ("mops", "selection_ns"):
        values = [float(result[key]) for result in runs]
        ordered = sorted(values)
        p95 = ordered[max(0, ceil(len(ordered) * 0.95) - 1)]
        print(f"summary_{key}_median={median(values):.6f}")
        print(f"summary_{key}_p95={p95:.6f}")
    for key in (
        "swap_out_count",
        "selection_count",
        "evaluated_pages",
        "invalidated_count",
    ):
        values = [int(result[key]) for result in runs]
        print(f"summary_{key}_min={min(values)}")
        print(f"summary_{key}_max={max(values)}")


if __name__ == "__main__":
    main()
