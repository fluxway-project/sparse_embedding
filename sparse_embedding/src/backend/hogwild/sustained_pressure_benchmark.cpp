#include "sparse_embedding/src/backend/hogwild/hogwild_embedding_store.hpp"

#include <algorithm>
#include <array>
#include <barrier>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

constexpr size_t kDimension = 8;
constexpr size_t kProducerCount = 8;
constexpr size_t kBatchSize = 512;
constexpr size_t kBatchesPerProducer = 256;
constexpr size_t kTotalKeys = kProducerCount * kBatchSize * kBatchesPerProducer;
constexpr size_t kMemoryLimitBytes = size_t{8} << 20;

uint64_t percentile(std::vector<uint64_t> values, size_t numerator) {
  if (values.empty() || numerator == 0 || numerator > 100)
    throw std::invalid_argument("pressure percentile is invalid");
  std::ranges::sort(values);
  const size_t rank = (values.size() * numerator + 99) / 100;
  return values[rank - 1];
}

struct DirectoryGuard {
  std::filesystem::path path;
  ~DirectoryGuard() { std::filesystem::remove_all(path); }
};

float expected_value(uint64_t key) { return static_cast<float>(key % 997 + 1); }

} // namespace

int main() {
  DirectoryGuard spill_directory{
      std::filesystem::temp_directory_path() /
      ("sparse-embedding-hp601-" + std::to_string(getpid()))};
  std::filesystem::remove_all(spill_directory.path);
  std::filesystem::create_directories(spill_directory.path);

  se::StorageOptions options;
  options.mode = se::StorageMode::kAuto;
  options.memory_limit_bytes = kMemoryLimitBytes;
  options.spill_directory = spill_directory.path.string();
  options.max_spill_bytes = size_t{512} << 20;
  options.watermark = se::WatermarkConfig{0.75, 0.50};
  auto storage = se::create_page_storage(std::move(options));
  using Store =
      se::HogwildEmbeddingStore<uint64_t, float, float, se::LfuScorePolicy>;
  Store store({kDimension}, {}, nullptr, se::LfuScorePolicy{},
              se::HogwildStoreOptions{.thread_budget = 16, .storage = storage});

  std::barrier start(static_cast<std::ptrdiff_t>(kProducerCount + 1));
  std::array<std::vector<uint64_t>, kProducerCount> latencies;
  std::array<std::exception_ptr, kProducerCount> errors{};
  std::vector<std::thread> producers;
  producers.reserve(kProducerCount);
  for (size_t producer = 0; producer < kProducerCount; ++producer) {
    producers.emplace_back([&, producer] {
      try {
        std::vector<uint64_t> keys(kBatchSize);
        std::vector<int64_t> lengths(kBatchSize, kDimension);
        std::vector<float> values(kBatchSize * kDimension);
        latencies[producer].reserve(kBatchesPerProducer);
        start.arrive_and_wait();
        for (size_t batch = 0; batch < kBatchesPerProducer; ++batch) {
          const uint64_t first_key =
              1 + producer * kBatchesPerProducer * kBatchSize +
              batch * kBatchSize;
          for (size_t index = 0; index < kBatchSize; ++index) {
            const uint64_t key = first_key + index;
            keys[index] = key;
            std::fill_n(values.data() + index * kDimension, kDimension,
                        expected_value(key));
          }
          const auto began = std::chrono::steady_clock::now();
          store.assign<uint64_t>(keys, values, lengths);
          latencies[producer].push_back(static_cast<uint64_t>(
              std::chrono::duration_cast<std::chrono::nanoseconds>(
                  std::chrono::steady_clock::now() - began)
                  .count()));
        }
      } catch (...) {
        errors[producer] = std::current_exception();
      }
    });
  }
  const auto pressure_started = std::chrono::steady_clock::now();
  start.arrive_and_wait();
  for (auto &producer : producers)
    producer.join();
  const auto producers_finished = std::chrono::steady_clock::now();
  for (const auto &error : errors) {
    if (error != nullptr)
      std::rethrow_exception(error);
  }

  const size_t low_watermark_bytes = kMemoryLimitBytes / 2;
  se::StorageStats stats = storage->stats();
  try {
    storage->drain_page_eviction_backlog(std::chrono::seconds(30));
    stats = storage->stats();
  } catch (...) {
    stats = storage->stats();
    std::cerr << "backlog_failure_memory_bytes=" << stats.memory_bytes << '\n'
              << "backlog_failure_evictable_memory_bytes="
              << stats.evictable_memory_bytes << '\n'
              << "backlog_failure_low_watermark_bytes=" << low_watermark_bytes
              << '\n'
              << "backlog_failure_spill_count=" << stats.spill_count << '\n'
              << "backlog_failure_swap_out_count=" << stats.swap_out_count
              << '\n'
              << "backlog_failure_eviction_selection_count="
              << stats.eviction_selection_count << '\n'
              << "backlog_failure_eviction_evaluated_pages="
              << stats.eviction_evaluated_pages << '\n'
              << "backlog_failure_candidate_invalidated_count="
              << stats.eviction_candidate_invalidated_count << '\n';
    throw;
  }
  if (stats.memory_bytes > low_watermark_bytes) {
    std::cerr << "backlog_failure_memory_bytes=" << stats.memory_bytes << '\n'
              << "backlog_failure_evictable_memory_bytes="
              << stats.evictable_memory_bytes << '\n'
              << "backlog_failure_low_watermark_bytes=" << low_watermark_bytes
              << '\n'
              << "backlog_failure_spill_count=" << stats.spill_count << '\n'
              << "backlog_failure_swap_out_count=" << stats.swap_out_count
              << '\n'
              << "backlog_failure_eviction_selection_count="
              << stats.eviction_selection_count << '\n'
              << "backlog_failure_eviction_evaluated_pages="
              << stats.eviction_evaluated_pages << '\n'
              << "backlog_failure_candidate_invalidated_count="
              << stats.eviction_candidate_invalidated_count << '\n';
    throw std::runtime_error("sustained pressure backlog did not drain");
  }
  const auto backlog_drained = std::chrono::steady_clock::now();

  constexpr size_t sample_count = kBatchSize;
  std::vector<uint64_t> sample_keys(sample_count);
  std::vector<int64_t> sample_lengths(sample_count, kDimension);
  std::vector<float> sample_values(sample_count * kDimension);
  for (size_t sample = 0; sample < sample_count; ++sample)
    sample_keys[sample] = 1 + sample;
  store.lookup<uint64_t>(sample_keys, sample_values, sample_lengths);
  for (size_t sample = 0; sample < sample_count; ++sample) {
    const float expected = expected_value(sample_keys[sample]);
    for (size_t offset = 0; offset < kDimension; ++offset) {
      if (sample_values[sample * kDimension + offset] != expected)
        throw std::runtime_error("stress lookup changed a payload");
    }
  }

  std::vector<uint64_t> verification_keys(kTotalKeys);
  std::iota(verification_keys.begin(), verification_keys.end(), uint64_t{1});
  std::vector<int64_t> verification_lengths(kTotalKeys, kDimension);
  std::vector<float> verification_values(kTotalKeys * kDimension);
  std::vector<uint8_t> verification_present(kTotalKeys);
  store.contains<uint64_t>(verification_keys, verification_present);
  constexpr size_t kVerificationBatch = 4096;
  for (size_t begin = 0; begin < kTotalKeys; begin += kVerificationBatch) {
    const size_t count = std::min(kVerificationBatch, kTotalKeys - begin);
    store.lookup<uint64_t>(
        std::span<const uint64_t>(verification_keys).subspan(begin, count),
        std::span<float>(verification_values)
            .subspan(begin * kDimension, count * kDimension),
        std::span<const int64_t>(verification_lengths).subspan(begin, count));
  }
  size_t present_count = 0;
  for (size_t index = 0; index < kTotalKeys; ++index) {
    if (verification_present[index] == 0)
      continue;
    ++present_count;
    const float expected = expected_value(verification_keys[index]);
    for (size_t offset = 0; offset < kDimension; ++offset) {
      if (verification_present[index] != 0 &&
          verification_values[index * kDimension + offset] != expected)
        throw std::runtime_error(
            "full pressure verification changed a payload at key=" +
            std::to_string(verification_keys[index]) + " value=" +
            std::to_string(verification_values[index * kDimension + offset]) +
            " expected=" + std::to_string(expected) +
            " present=" + std::to_string(verification_present[index]) + "\n");
    }
  }

  std::vector<uint64_t> all_latencies;
  all_latencies.reserve(kProducerCount * kBatchesPerProducer);
  for (const auto &producer_latencies : latencies) {
    all_latencies.insert(all_latencies.end(), producer_latencies.begin(),
                         producer_latencies.end());
  }
  const uint64_t p95_ns = percentile(all_latencies, 95);
  const uint64_t p99_ns = percentile(all_latencies, 99);
  const uint64_t pressure_pages = stats.spill_count + stats.swap_out_count;
  if (pressure_pages == 0)
    throw std::runtime_error(
        "sustained pressure did not reach a pressure path");
  const uint64_t backlog_drain_ns = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(backlog_drained -
                                                           producers_finished)
          .count());
  const uint64_t pressure_elapsed_ns = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(producers_finished -
                                                           pressure_started)
          .count());

  std::cout << "producer_count=" << kProducerCount << '\n'
            << "key_count=" << kTotalKeys << '\n'
            << "pressure_pages=" << pressure_pages << '\n'
            << "spill_count=" << stats.spill_count << '\n'
            << "swap_out_count=" << stats.swap_out_count << '\n'
            << "memory_bytes=" << stats.memory_bytes << '\n'
            << "spill_bytes=" << stats.spill_bytes << '\n'
            << "retired_bytes=" << stats.retired_bytes << '\n'
            << "allocation_p95_ns=" << p95_ns << '\n'
            << "allocation_p99_ns=" << p99_ns << '\n'
            << "full_lookup_verified=1\n"
            << "present_key_count=" << present_count << '\n'
            << "missing_key_count=" << (kTotalKeys - present_count) << '\n'
            << "backlog_drain_ns=" << backlog_drain_ns << '\n'
            << "pressure_elapsed_ns=" << pressure_elapsed_ns << '\n';
}
