"""Measure gradient accumulation and Adam commit on an initialized table."""

import argparse
import json
import time

import numpy as np

import sparse_embedding as se


def measure(dim, keys, threads, steps, journal):
    config = se.EmbeddingConfig(
        dims=[dim], thread_budget=threads, enable_mutation_journal=journal
    )
    store = se.EmbeddingStore(config, optimizer_config=se.Adam(lr=0.01))
    ids = np.arange(1, keys + 1, dtype=np.int64)
    values = np.random.default_rng(42).standard_normal((keys, dim)).astype(np.float32)
    gradients = (
        np.random.default_rng(99).standard_normal((keys, dim)).astype(np.float32)
    )
    store.assign(ids, values)

    def train_step():
        store.accumulate_gradients(ids, gradients)
        store.optimizer_step(1)

    for _ in range(5):
        train_step()

    samples = []
    for _ in range(steps):
        start = time.perf_counter_ns()
        train_step()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return {
        "journal": journal,
        "p50_ms": float(np.median(samples)),
        "p99_ms": float(np.percentile(samples, 99)),
        "million_keys_per_second": keys / (1000 * float(np.mean(samples))),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--keys", type=int, default=65536)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument(
        "--journal", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    if min(args.dim, args.keys, args.threads, args.steps) < 1:
        parser.error("dim, keys, threads and steps must be positive")
    print(json.dumps(measure(**vars(args))))
