import time

import numpy as np

import sparse_embedding


class BenchmarkRunner:
    def __init__(self, name: str, iters: int = 10):
        self.name = name
        self.iters = iters

    def run(self, func, batch_size: int):
        start = time.perf_counter()
        for _ in range(self.iters):
            func()
        end = time.perf_counter()

        total_time = end - start
        total_operations = (
            len(batch_size) * self.iters
            if isinstance(batch_size, np.ndarray)
            else batch_size * self.iters
        )
        mops = (total_operations / total_time) / 1e6

        print(
            f"[{self.name}] Batch Size: {batch_size:<10} | "
            f"Time: {total_time * 1000:8.2f} ms | "
            f"Throughput: {mops:8.2f} MOPS"
        )


class EmbeddingTableBenchmark:
    def __init__(self, dim: int = 128, thread_budget: int = 4):
        self.dim = dim
        self.thread_budget = thread_budget
        self.store = self._create_store()

    def _create_store(self) -> sparse_embedding.EmbeddingStore:
        config = sparse_embedding.EmbeddingConfig(
            dims=[self.dim], thread_budget=self.thread_budget, value_dtype="float32"
        )
        initializer = sparse_embedding.Zeros()
        return sparse_embedding.EmbeddingStore(
            embedding_config=config, initializer_config=initializer
        )

    @staticmethod
    def _unique_keys(batch_size: int) -> np.ndarray:
        return np.arange(batch_size, dtype=np.int64)

    def benchmark_insertion(self, batch_sizes: list[int], iters: int = 10):
        print(
            f"\n=== Insertion Benchmark (lookup_or_insert) [Threads: {self.thread_budget}] ==="
        )
        for batch_size in batch_sizes:
            runner = BenchmarkRunner("Insert", iters=iters)
            keys = self._unique_keys(batch_size)
            # Warmup to ensure JIT/OS pages are ready
            for _ in range(3):
                self.store.lookup_or_insert(keys, dim=self.dim)
            runner.run(
                lambda: self.store.lookup_or_insert(keys, dim=self.dim), batch_size
            )

    def benchmark_query(self, batch_sizes: list[int], iters: int = 10):
        print(f"\n=== Query Benchmark (lookup) [Threads: {self.thread_budget}] ===")
        max_batch = max(batch_sizes) if batch_sizes else 1000000
        num_keys = max(2000000, max_batch * 2)

        pre_insert_keys = np.arange(num_keys, dtype=np.int64)
        pre_insert_values = np.random.rand(num_keys, self.dim).astype(np.float32)
        self.store.assign(pre_insert_keys, pre_insert_values)

        for batch_size in batch_sizes:
            runner = BenchmarkRunner("Lookup", iters=iters)
            keys = np.random.randint(0, num_keys, size=batch_size, dtype=np.int64)
            # Warmup
            for _ in range(3):
                self.store.lookup(keys, dim=self.dim)
            runner.run(lambda: self.store.lookup(keys, dim=self.dim), batch_size)

    def benchmark_assign(self, batch_sizes: list[int], iters: int = 10):
        print(
            f"\n=== Assign Benchmark (Raw Compute throughput) [Threads: {self.thread_budget}] ==="
        )
        for batch_size in batch_sizes:
            runner = BenchmarkRunner("Assign", iters=iters)
            keys = self._unique_keys(batch_size)
            values = np.random.rand(batch_size, self.dim).astype(np.float32)
            # Warmup
            for _ in range(3):
                self.store.assign(keys, values)
            runner.run(lambda: self.store.assign(keys, values), batch_size)

    def benchmark_end_to_end(self, batch_sizes: list[int], iters: int = 10):
        print(f"\n=== End-to-End Step Benchmark [Threads: {self.thread_budget}] ===")
        for batch_size in batch_sizes:
            runner = BenchmarkRunner("E2E Step", iters=iters)
            keys = self._unique_keys(batch_size)
            values = np.random.rand(batch_size, self.dim).astype(np.float32)
            # Warmup
            self.store.lookup_or_insert(keys, dim=self.dim)
            self.store.accumulate_gradients(keys, values)
            self.store.optimizer_step(1)

            def e2e_step():
                self.store.lookup(keys, dim=self.dim)
                self.store.accumulate_gradients(keys, values)
                self.store.optimizer_step(1)

            runner.run(e2e_step, batch_size)


def main():
    thread_counts = [8, 16, 32]
    batch_sizes = [65536, 131072]

    for thread_budget in thread_counts:
        print("\n\n===============================================")
        print(f"Testing with {thread_budget} Threads")
        print("===============================================")
        benchmark = EmbeddingTableBenchmark(dim=128, thread_budget=thread_budget)
        benchmark.benchmark_assign(batch_sizes, iters=10)
        benchmark.benchmark_insertion(batch_sizes, iters=10)
        benchmark.benchmark_query(batch_sizes, iters=10)
        benchmark.benchmark_end_to_end(batch_sizes, iters=10)


if __name__ == "__main__":
    main()
