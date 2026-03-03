# coding=utf-8
"""
Simple micro-benchmark for kv_store.JaxKVStore

用法（在已编译好 _core 的 py310 环境中）：

    cd fluxway/fluxway/sparse_embedding
    python python/benchmark_kv_store.py

会在不同 batch size / num_shards 下测量：
- insert
- lookup
- lookup_or_insert
的平均耗时与 QPS。
"""

import time
from dataclasses import dataclass
from typing import List, Dict

import numpy as np
import jax
import jax.numpy as jnp

from kv_store import JaxKVStore


@dataclass
class BenchmarkConfig:
    num_shards: int
    batch_size: int
    embedding_dim: int
    num_warmup: int = 3
    num_iters: int = 10


def make_kv_store(dim: int, num_shards: int) -> JaxKVStore:
    """
    构造一个简单配置的 JaxKVStore：
    - 仅一个 slot（121），维度 dim，init_type=0 全 0 初始化。
    """
    feature_metas: Dict[int, int] = {121: dim}
    return JaxKVStore(feature_metas, init_type=0, num_shards=num_shards)


def make_keys(batch_size: int, start_id: int = 0, slot: int = 121) -> np.ndarray:
    """
    生成一批伪 fid：
    fid = (slot << 48) | local_id
    """
    slot_prefix = np.uint64(slot) << np.uint64(48)
    local_ids = np.arange(start_id, start_id + batch_size, dtype=np.uint64)
    return (slot_prefix | local_ids).astype(np.uint64)


def make_values(batch_size: int, dim: int, base: float = 1.0) -> np.ndarray:
    """
    生成一批值，用于 insert / update。
    """
    return np.full((batch_size, dim), base, dtype=np.float32)


def run_one_benchmark(cfg: BenchmarkConfig) -> None:
    store = make_kv_store(cfg.embedding_dim, cfg.num_shards)

    keys = make_keys(cfg.batch_size)
    values = make_values(cfg.batch_size, cfg.embedding_dim, base=1.0)

    print(
        f"\n=== num_shards={cfg.num_shards}, "
        f"batch_size={cfg.batch_size}, dim={cfg.embedding_dim} ==="
    )

    # ---------- insert ----------
    for _ in range(cfg.num_warmup):
        store.insert(keys, values)

    t0 = time.perf_counter()
    for _ in range(cfg.num_iters):
        store.insert(keys, values)
    t1 = time.perf_counter()

    elapsed = t1 - t0
    avg_ms = elapsed * 1000.0 / cfg.num_iters
    qps = (cfg.batch_size * cfg.num_iters) / elapsed
    print(f"insert: {avg_ms:8.3f} ms/iter, {qps:10.0f} keys/s")

    # ---------- lookup ----------
    for _ in range(cfg.num_warmup):
        store.lookup(keys)

    t0 = time.perf_counter()
    for _ in range(cfg.num_iters):
        store.lookup(keys)
    t1 = time.perf_counter()

    elapsed = t1 - t0
    avg_ms = elapsed * 1000.0 / cfg.num_iters
    qps = (cfg.batch_size * cfg.num_iters) / elapsed
    print(f"lookup: {avg_ms:8.3f} ms/iter, {qps:10.0f} keys/s")

    # ---------- lookup_or_insert（对 miss 情况更敏感） ----------
    # 使用另一批 key，以确保大部分为 miss
    keys2 = make_keys(cfg.batch_size, start_id=1_000_000)

    for _ in range(cfg.num_warmup):
        store.lookup_or_insert(keys2)

    t0 = time.perf_counter()
    for _ in range(cfg.num_iters):
        store.lookup_or_insert(keys2)
    t1 = time.perf_counter()

    elapsed = t1 - t0
    avg_ms = elapsed * 1000.0 / cfg.num_iters
    qps = (cfg.batch_size * cfg.num_iters) / elapsed
    print(f"lookup_or_insert: {avg_ms:8.3f} ms/iter, {qps:10.0f} keys/s")


def main():
    # 可以根据需要调整配置
    dims = [16, 64, 128]
    batch_sizes = [1_000, 10_000, 100_000]
    shard_choices = [16, 64, 256]

    print("==== CPU-side KVStore micro-benchmark ====")
    for dim in dims:
        for bs in batch_sizes:
            for ns in shard_choices:
                cfg = BenchmarkConfig(
                    num_shards=ns,
                    batch_size=bs,
                    embedding_dim=dim,
                    num_warmup=3,
                    num_iters=10,
                )
                run_one_benchmark(cfg)

    # ---------------- GPU copy baseline ----------------
    print("\n==== Host->Device copy baseline (if GPU available) ====")
    devices = jax.devices()
    has_gpu = any(d.platform == "gpu" for d in devices)
    if not has_gpu:
        print("No GPU detected by JAX; skip GPU copy benchmark.")
        return

    def bench_device_put(batch_size: int, dim: int, iters: int = 20):
        x_host = np.random.randn(batch_size, dim).astype(np.float32)
        # warmup
        for _ in range(5):
            y = jax.device_put(x_host)
            _ = y.block_until_ready()

        t0 = time.perf_counter()
        for _ in range(iters):
            y = jax.device_put(x_host)
            _ = y.block_until_ready()
        t1 = time.perf_counter()
        elapsed = t1 - t0
        avg_ms = elapsed * 1000.0 / iters
        mb = batch_size * dim * 4 / (1024 * 1024)
        bandwidth = mb * iters / elapsed  # MB/s
        print(
            f"device_put: batch={batch_size:7d}, dim={dim:4d} -> "
            f"{avg_ms:7.3f} ms, ~{bandwidth:8.1f} MB/s"
        )

    for dim in dims:
        for bs in batch_sizes:
            bench_device_put(bs, dim)




if __name__ == "__main__":
    main()

