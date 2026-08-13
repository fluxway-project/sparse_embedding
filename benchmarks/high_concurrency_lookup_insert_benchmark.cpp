#include <algorithm>
#include <barrier>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <vector>

#include "sparse_embedding/src/backend/hogwild/hogwild_embedding_store.hpp"

namespace {
constexpr size_t kDimension = 32;
constexpr size_t kBatch = 1024;
constexpr size_t kLookupRounds = 64;
constexpr size_t kInsertRounds = 8;
constexpr size_t kWarmKeys = 1 << 20;
constexpr size_t kRepetitions = 5;
constexpr size_t kBenchmarkMemoryLimitBytes = size_t{1} << 30;

struct Result {
  size_t threads;
  double lookup_mops;
  double insert_mops;
  uint64_t lookup_p99_ns;
  uint64_t insert_p99_ns;
};

uint64_t percentile(std::vector<uint64_t> values) {
  std::ranges::sort(values);
  return values[(values.size() * 99) / 100];
}

Result run(size_t threads, size_t repetition) {
  static_cast<void>(repetition);
  using Store =
      se::HogwildEmbeddingStore<uint64_t, float, float, se::NoScorePolicy>;
  se::StorageOptions storage_options;
  storage_options.mode = se::StorageMode::kMemoryOnly;
  storage_options.memory_limit_bytes = kBenchmarkMemoryLimitBytes;
  auto storage = se::create_page_storage(std::move(storage_options));
  Store store(
      {kDimension}, {}, nullptr, se::NoScorePolicy{},
      se::HogwildStoreOptions{.thread_budget = threads, .storage = storage});
  std::vector<uint64_t> warm_keys(kWarmKeys);
  std::iota(warm_keys.begin(), warm_keys.end(), uint64_t{1});
  std::vector<int64_t> warm_lengths(kWarmKeys, kDimension);
  std::vector<float> warm_values(kWarmKeys * kDimension);
  store.lookup_or_insert<uint64_t>(warm_keys, warm_values, warm_lengths);
  std::barrier start(static_cast<std::ptrdiff_t>(threads));
  std::vector<uint64_t> lookup_ns(threads * kLookupRounds);
  std::vector<uint64_t> insert_ns(threads * kInsertRounds);
  std::vector<uint64_t> lookup_elapsed_ns(threads);
  std::vector<uint64_t> insert_elapsed_ns(threads);
  std::vector<std::exception_ptr> errors(threads);
  std::vector<size_t> failure_retired_bytes(threads);
  std::vector<std::thread> workers;
  workers.reserve(threads);
  for (size_t worker = 0; worker < threads; ++worker) {
    workers.emplace_back([&, worker] {
      try {
        std::vector<uint64_t> keys(kBatch);
        std::vector<uint64_t> insert_keys(kBatch);
        std::vector<int64_t> lengths(kBatch, kDimension);
        std::vector<float> output(kBatch * kDimension);
        std::vector<float> values(kBatch * kDimension, 1.0F);
        start.arrive_and_wait();
        const auto lookup_started = std::chrono::steady_clock::now();
        for (size_t round = 0; round < kLookupRounds; ++round) {
          for (size_t index = 0; index < kBatch; ++index) {
            keys[index] =
                1 +
                ((worker * kLookupRounds + round) * kBatch + index) % kWarmKeys;
            insert_keys[index] = 2 * kWarmKeys +
                                 (worker * kInsertRounds + round) * kBatch +
                                 index;
          }
          auto began = std::chrono::steady_clock::now();
          store.forward_lookup<uint64_t>(keys, output, lengths);
          lookup_ns[worker * kLookupRounds + round] = static_cast<uint64_t>(
              std::chrono::duration_cast<std::chrono::nanoseconds>(
                  std::chrono::steady_clock::now() - began)
                  .count());
        }
        lookup_elapsed_ns[worker] = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - lookup_started)
                .count());
        start.arrive_and_wait();
        const auto insert_started = std::chrono::steady_clock::now();
        for (size_t round = 0; round < kInsertRounds; ++round) {
          for (size_t index = 0; index < kBatch; ++index)
            insert_keys[index] = 2 * kWarmKeys +
                                 (worker * kInsertRounds + round) * kBatch +
                                 index;
          auto began = std::chrono::steady_clock::now();
          store.lookup_or_insert<uint64_t>(insert_keys, values, lengths);
          insert_ns[worker * kInsertRounds + round] = static_cast<uint64_t>(
              std::chrono::duration_cast<std::chrono::nanoseconds>(
                  std::chrono::steady_clock::now() - began)
                  .count());
        }
        insert_elapsed_ns[worker] = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - insert_started)
                .count());
      } catch (...) {
        failure_retired_bytes[worker] = storage->stats().retired_bytes;
        errors[worker] = std::current_exception();
      }
    });
  }
  for (auto &worker : workers)
    worker.join();
  for (const auto &error : errors) {
    if (error == nullptr)
      continue;
    const se::StorageStats stats = storage->stats();
    std::cerr << "threads=" << threads
              << " retired_bytes=" << stats.retired_bytes
              << " failure_retired_bytes="
              << *std::ranges::max_element(failure_retired_bytes)
              << " retirement_pressure_state="
              << static_cast<unsigned>(stats.retirement_pressure_state)
              << " retirement_backpressure_count="
              << stats.retirement_backpressure_count << '\n';
    std::rethrow_exception(error);
  }
  const se::StorageStats stats = storage->stats();
  if (stats.retired_bytes != 0)
    throw std::runtime_error("benchmark directory retirement did not drain");
  const double lookup_operations =
      static_cast<double>(threads * kLookupRounds * kBatch);
  const double insert_operations =
      static_cast<double>(threads * kInsertRounds * kBatch);
  const double lookup_seconds =
      static_cast<double>(*std::ranges::max_element(lookup_elapsed_ns)) / 1e9;
  const double insert_seconds =
      static_cast<double>(*std::ranges::max_element(insert_elapsed_ns)) / 1e9;
  return {threads, lookup_operations / lookup_seconds / 1e6,
          insert_operations / insert_seconds / 1e6, percentile(lookup_ns),
          percentile(insert_ns)};
}

struct Summary {
  double median;
  double p95;
  double p99;
};

Summary summarize(std::vector<double> values) {
  std::ranges::sort(values);
  const auto at = [&](double fraction) {
    return values[std::min(values.size() - 1,
                           static_cast<size_t>(fraction * values.size()))];
  };
  return {values[values.size() / 2], at(0.95), at(0.99)};
}
} // namespace

int main(int argc, char **argv) {
  std::vector<size_t> thread_counts{size_t{1}, size_t{16}, size_t{64}};
  if (argc == 3 && std::string_view(argv[1]) == "--threads") {
    const size_t threads = std::strtoull(argv[2], nullptr, 10);
    if (threads == 0)
      throw std::invalid_argument("benchmark thread count must be positive");
    thread_counts.assign(1, threads);
  } else if (argc != 1) {
    throw std::invalid_argument("usage: benchmark [--threads COUNT]");
  }
  for (size_t threads : thread_counts) {
    std::vector<double> lookup_mops;
    std::vector<double> insert_mops;
    std::vector<double> lookup_p99;
    std::vector<double> insert_p99;
    for (size_t repetition = 0; repetition < kRepetitions; ++repetition) {
      const Result result = run(threads, repetition);
      lookup_mops.push_back(result.lookup_mops);
      insert_mops.push_back(result.insert_mops);
      lookup_p99.push_back(result.lookup_p99_ns);
      insert_p99.push_back(result.insert_p99_ns);
    }
    const Summary lookup = summarize(std::move(lookup_mops));
    const Summary insert = summarize(std::move(insert_mops));
    const Summary lookup_tail = summarize(std::move(lookup_p99));
    const Summary insert_tail = summarize(std::move(insert_p99));
    std::cout << "threads=" << threads
              << " lookup_mops_median=" << lookup.median
              << " lookup_mops_p95=" << lookup.p95
              << " lookup_mops_p99=" << lookup.p99
              << " insert_mops_median=" << insert.median
              << " insert_mops_p95=" << insert.p95
              << " insert_mops_p99=" << insert.p99
              << " lookup_latency_p99_median_ns=" << lookup_tail.median
              << " insert_latency_p99_median_ns=" << insert_tail.median << '\n';
  }
}
