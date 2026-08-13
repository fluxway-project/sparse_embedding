#include <algorithm>
#include <barrier>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <span>
#include <stdexcept>
#include <thread>
#include <vector>

#include "sparse_embedding/src/backend/hogwild/hogwild_embedding_store.hpp"

namespace {

constexpr size_t kDimension = 32;
constexpr size_t kBatch = 1024;
constexpr size_t kRounds = 64;
constexpr size_t kWarmKeys = 1 << 20;

struct RoleResult {
  uint64_t elapsed_ns = 0;
  std::vector<uint64_t> latencies;
};

uint64_t percentile(std::vector<uint64_t> values) {
  if (values.empty())
    throw std::invalid_argument("mixed benchmark has no latency samples");
  std::ranges::sort(values);
  return values[(values.size() * 99) / 100];
}

struct MixedResult {
  size_t total_threads;
  size_t lookup_threads;
  size_t insert_threads;
  double lookup_mops;
  double insert_mops;
  uint64_t lookup_p99_ns;
  uint64_t insert_p99_ns;
  size_t requested_insert;
  size_t accepted_insert;
  size_t final_size;
};

MixedResult run(size_t total_threads) {
  using Store =
      se::HogwildEmbeddingStore<uint64_t, float, float, se::NoScorePolicy>;
  se::StorageOptions storage_options;
  storage_options.memory_limit_bytes = size_t{1} << 30;
  storage_options.spill_directory = std::filesystem::temp_directory_path();
  storage_options.max_spill_bytes = size_t{2} << 30;
  auto storage = se::create_page_storage(std::move(storage_options));
  Store store({kDimension}, {}, nullptr, se::NoScorePolicy{},
              se::HogwildStoreOptions{.thread_budget = total_threads,
                                      .storage = storage});
  std::vector<uint64_t> warm_keys(kWarmKeys);
  std::iota(warm_keys.begin(), warm_keys.end(), uint64_t{1});
  std::vector<int64_t> warm_lengths(kWarmKeys, kDimension);
  std::vector<float> warm_values(kWarmKeys * kDimension);
  store.lookup_or_insert<uint64_t>(warm_keys, warm_values, warm_lengths);

  const size_t lookup_threads = total_threads / 2;
  const size_t insert_threads = total_threads - lookup_threads;
  if (lookup_threads == 0 || insert_threads == 0)
    throw std::invalid_argument(
        "mixed benchmark requires at least two total threads");
  std::barrier start(static_cast<std::ptrdiff_t>(total_threads));
  std::vector<RoleResult> results(total_threads);
  std::vector<std::thread> workers;
  workers.reserve(total_threads);
  for (size_t worker = 0; worker < total_threads; ++worker) {
    workers.emplace_back([&, worker] {
      const bool lookup_role = worker < lookup_threads;
      auto &result = results[worker];
      result.latencies.reserve(kRounds);
      std::vector<uint64_t> keys(kBatch);
      std::vector<int64_t> lengths(kBatch, kDimension);
      std::vector<float> values(kBatch * kDimension, 1.0F);
      std::vector<float> output(kBatch * kDimension);
      start.arrive_and_wait();
      const auto began_all = std::chrono::steady_clock::now();
      for (size_t round = 0; round < kRounds; ++round) {
        const size_t ordinal = worker * kRounds + round;
        for (size_t index = 0; index < kBatch; ++index) {
          if (lookup_role) {
            keys[index] = 1 + ((ordinal * kBatch + index) % kWarmKeys);
          } else {
            keys[index] = 2 * kWarmKeys + ordinal * kBatch + index;
          }
        }
        const auto began = std::chrono::steady_clock::now();
        if (lookup_role)
          store.forward_lookup<uint64_t>(keys, output, lengths);
        else
          store.lookup_or_insert<uint64_t>(keys, values, lengths);
        result.latencies.push_back(static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - began)
                .count()));
      }
      result.elapsed_ns = static_cast<uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - began_all)
              .count());
    });
  }
  for (auto &worker : workers)
    worker.join();

  uint64_t lookup_elapsed = 0;
  uint64_t insert_elapsed = 0;
  std::vector<uint64_t> lookup_latencies;
  std::vector<uint64_t> insert_latencies;
  for (size_t worker = 0; worker < total_threads; ++worker) {
    auto &result = results[worker];
    if (worker < lookup_threads) {
      lookup_elapsed = std::max(lookup_elapsed, result.elapsed_ns);
      lookup_latencies.insert(lookup_latencies.end(), result.latencies.begin(),
                              result.latencies.end());
    } else {
      insert_elapsed = std::max(insert_elapsed, result.elapsed_ns);
      insert_latencies.insert(insert_latencies.end(), result.latencies.begin(),
                              result.latencies.end());
    }
  }
  const double lookup_operations =
      static_cast<double>(lookup_threads * kRounds * kBatch);
  const double insert_operations =
      static_cast<double>(insert_threads * kRounds * kBatch);
  const size_t requested_insert = insert_threads * kRounds * kBatch;
  const size_t final_size = store.size();
  if (final_size < kWarmKeys)
    throw std::runtime_error("mixed benchmark lost warm keys");
  return {total_threads,
          lookup_threads,
          insert_threads,
          lookup_operations / (static_cast<double>(lookup_elapsed) / 1e9) / 1e6,
          insert_operations / (static_cast<double>(insert_elapsed) / 1e9) / 1e6,
          percentile(std::move(lookup_latencies)),
          percentile(std::move(insert_latencies)),
          requested_insert,
          final_size - kWarmKeys,
          final_size};
}

} // namespace

int main() {
  for (size_t threads : {size_t{16}, size_t{64}}) {
    const MixedResult result = run(threads);
    std::cout << "mode=mixed total_threads=" << result.total_threads
              << " lookup_threads=" << result.lookup_threads
              << " insert_threads=" << result.insert_threads
              << " lookup_mops=" << result.lookup_mops
              << " insert_mops=" << result.insert_mops
              << " lookup_latency_p99_ns=" << result.lookup_p99_ns
              << " insert_latency_p99_ns=" << result.insert_p99_ns
              << " requested_insert=" << result.requested_insert
              << " accepted_insert=" << result.accepted_insert
              << " final_size=" << result.final_size << '\n';
  }
}
