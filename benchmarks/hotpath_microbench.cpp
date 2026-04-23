#include <atomic>
#include <chrono>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

#include "exec/static_thread_pool.hpp"
#include "parlay_hash/unordered_map.h"
#include "stdexec/execution.hpp"

namespace ex = stdexec;

// Reproduce the exact parlayhash types used in HogwildEmbeddingStore
struct SplitMix64Hash {
  size_t operator()(uint64_t key) const {
    uint64_t x = key;
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ull;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebull;
    x ^= x >> 31;
    return x;
  }
};

using MapData = parlay::MapData<uint64_t, uint32_t, SplitMix64Hash,
                                std::equal_to<uint64_t>>;
using Key = parlay::IndirectEntries<MapData>;
using KeyMap = parlay::parlay_hash<Key>;

static constexpr size_t kDim = 64;
static constexpr size_t kKeys = 100'000;

// Benchmark 1: pure parlayhash Find (to verify we match official perf)
void bench_pure_find(const std::vector<uint64_t> &keys, KeyMap &map,
                     Key &entries, size_t num_threads) {
  std::atomic<size_t> found_count{0};
  exec::static_thread_pool pool(num_threads);
  size_t n = keys.size();
  size_t chunk = (n + num_threads - 1) / num_threads;

  auto start = std::chrono::high_resolution_clock::now();
  ex::sync_wait(ex::schedule(pool.get_scheduler()) |
                ex::bulk(ex::par_unseq, num_threads, [&](size_t tid) {
                  epoch::with_epoch([&] {
                    size_t begin = std::min(tid * chunk, n);
                    size_t end = std::min(begin + chunk, n);
                    size_t local = 0;
                    for (size_t i = begin; i < end; ++i) {
                      auto r = map.Find(
                          Key::Entry::make_key(keys[i]),
                          [](const auto &e) { return e.get_entry().second; });
                      if (r.has_value())
                        ++local;
                    }
                    found_count.fetch_add(local, std::memory_order_relaxed);
                  });
                }));
  auto end = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration<double, std::milli>(end - start).count();
  std::cout << "  pure_find:    " << (n / ms * 1000 / 1e6)
            << " Mops/s  (found=" << found_count.load() << ")" << std::endl;
}

// Benchmark 2: Find + memcpy of embedding (hotpath overhead)
void bench_find_and_copy(const std::vector<uint64_t> &keys, KeyMap &map,
                         Key &entries, const std::vector<float> &arena_data,
                         std::vector<float> &out, size_t num_threads) {
  exec::static_thread_pool pool(num_threads);
  size_t n = keys.size();
  size_t chunk = (n + num_threads - 1) / num_threads;

  auto start = std::chrono::high_resolution_clock::now();
  ex::sync_wait(ex::schedule(pool.get_scheduler()) |
                ex::bulk(ex::par_unseq, num_threads, [&](size_t tid) {
                  epoch::with_epoch([&] {
                    size_t begin = std::min(tid * chunk, n);
                    size_t end = std::min(begin + chunk, n);
                    for (size_t i = begin; i < end; ++i) {
                      auto r = map.Find(
                          Key::Entry::make_key(keys[i]),
                          [](const auto &e) { return e.get_entry().second; });
                      if (r.has_value()) {
                        // Simulate embedding copy: 64 floats = 256 bytes
                        std::memcpy(out.data() + i * kDim,
                                    arena_data.data() + (*r) * kDim,
                                    kDim * sizeof(float));
                      }
                    }
                  });
                }));
  auto end = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration<double, std::milli>(end - start).count();
  std::cout << "  find+memcpy:  " << (n / ms * 1000 / 1e6) << " Mops/s"
            << std::endl;
}

// Benchmark 3: with_epoch overhead alone (empty body)
void bench_epoch_overhead(size_t num_threads, size_t n) {
  exec::static_thread_pool pool(num_threads);
  size_t chunk = (n + num_threads - 1) / num_threads;
  std::atomic<size_t> cnt{0};

  auto start = std::chrono::high_resolution_clock::now();
  ex::sync_wait(ex::schedule(pool.get_scheduler()) |
                ex::bulk(ex::par_unseq, num_threads, [&](size_t tid) {
                  size_t begin = std::min(tid * chunk, n);
                  size_t end = std::min(begin + chunk, n);
                  for (size_t i = begin; i < end; ++i) {
                    epoch::with_epoch(
                        [&] { cnt.fetch_add(1, std::memory_order_relaxed); });
                  }
                }));
  auto end = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration<double, std::milli>(end - start).count();
  std::cout << "  epoch_per_key:" << (n / ms * 1000 / 1e6)
            << " Mops/s (per-key epoch)" << std::endl;
}

// Benchmark 4: epoch hoisted outside loop (batch epoch)
void bench_epoch_hoisted(size_t num_threads, size_t n) {
  exec::static_thread_pool pool(num_threads);
  size_t chunk = (n + num_threads - 1) / num_threads;
  std::atomic<size_t> cnt{0};

  auto start = std::chrono::high_resolution_clock::now();
  ex::sync_wait(ex::schedule(pool.get_scheduler()) |
                ex::bulk(ex::par_unseq, num_threads, [&](size_t tid) {
                  epoch::with_epoch([&] {
                    size_t begin = std::min(tid * chunk, n);
                    size_t end = std::min(begin + chunk, n);
                    for (size_t i = begin; i < end; ++i) {
                      cnt.fetch_add(1, std::memory_order_relaxed);
                    }
                  });
                }));
  auto end = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration<double, std::milli>(end - start).count();
  std::cout << "  epoch_hoisted:" << (n / ms * 1000 / 1e6)
            << " Mops/s (batch epoch)" << std::endl;
}

int main() {
  // Build the map with all keys pre-inserted
  Key entries(false);
  KeyMap map(kKeys * 2, &entries, false);

  std::vector<uint64_t> keys(kKeys);
  std::mt19937_64 gen(42);
  for (size_t i = 0; i < kKeys; ++i) {
    keys[i] = gen() & 0x0000FFFFFFFFFFFFull;
  }

  // Pre-insert all keys
  {
    std::atomic<uint32_t> next_idx{0};
    for (size_t i = 0; i < kKeys; ++i) {
      uint32_t idx = next_idx.fetch_add(1, std::memory_order_relaxed);
      auto constr = [&](const auto &) {
        return entries.make_entry(Key::Entry::make_key(keys[i]),
                                  std::make_pair(keys[i], idx));
      };
      auto g = [&](std::optional<typename Key::Entry>)
          -> std::pair<uint64_t, uint32_t> {
        return {keys[i], idx};
      };
      map.Upsert(Key::Entry::make_key(keys[i]), constr, g);
    }
  }
  std::cout << "Map size: " << map.size() << std::endl;

  // Arena data: pre-allocated floats
  std::vector<float> arena_data(kKeys * kDim, 1.0f);
  std::vector<float> out(kKeys * kDim, 0.0f);

  for (size_t num_threads : {1UL, 16UL, 32UL, 64UL, 128UL}) {
    std::cout << "\n--- " << num_threads << " threads ---" << std::endl;
    bench_epoch_overhead(num_threads, kKeys);
    bench_epoch_hoisted(num_threads, kKeys);
    bench_pure_find(keys, map, entries, num_threads);
    bench_find_and_copy(keys, map, entries, arena_data, out, num_threads);
  }
  return 0;
}
