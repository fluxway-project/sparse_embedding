#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "sparse_embedding/src/backend/hogwild/hogwild_embedding_store.hpp"
#include "sparse_embedding/src/eviction/score_policy.hpp"

namespace {

constexpr size_t kDimension = 128;
constexpr size_t kBatchSize = 512;
constexpr size_t kResidentKeyBudget = 8192;
constexpr size_t kEpochs = 6;
constexpr size_t kHotKeysPerBatch = 64;
constexpr size_t kRepetitions = 5;

struct Result {
  bool use_threshold_eviction;
  double operations_per_second;
  uint64_t swap_out_count;
  uint64_t spill_count;
};

Result run_once(bool use_threshold_eviction, size_t repetition) {
  const std::string spill_directory =
      "/tmp/sparse_embedding_eviction_policy_" +
      std::string(use_threshold_eviction ? "threshold" : "baseline") + "_" +
      std::to_string(repetition);
  std::filesystem::remove_all(spill_directory);
  std::filesystem::create_directories(spill_directory);

  se::StorageOptions options;
  options.mode = se::StorageMode::kAuto;
  options.memory_limit_bytes = 6 << 20;
  options.spill_directory = spill_directory;
  options.max_spill_bytes = 1 << 30;
  options.watermark = se::WatermarkConfig{0.8, 0.6};

  auto storage = se::create_page_storage(options);
  se::HogwildEmbeddingStore<uint64_t, float, float, se::LfuScorePolicy> store(
      {kDimension}, {}, nullptr, se::LfuScorePolicy{},
      se::HogwildStoreOptions{.thread_budget = 2, .storage = storage});
  std::vector<uint64_t> keys(kBatchSize);
  std::vector<int64_t> lengths(kBatchSize, kDimension);
  std::vector<float> values(kBatchSize * kDimension, 1.0f);
  std::vector<uint64_t> evicted_keys(kBatchSize);
  std::vector<float> evicted_values(kBatchSize * kDimension);

  const auto started = std::chrono::steady_clock::now();
  for (size_t epoch = 0; epoch < kEpochs; ++epoch) {
    for (size_t offset = 0; offset < kResidentKeyBudget; offset += kBatchSize) {
      for (size_t index = 0; index < kBatchSize; ++index)
        keys[index] = epoch * kResidentKeyBudget + offset + index;
      store.lookup_or_insert<uint64_t>(keys, values, lengths);
      store.forward_lookup<uint64_t>(
          std::span<const uint64_t>(keys.data(), kHotKeysPerBatch),
          std::span<float>(values.data(), kHotKeysPerBatch * kDimension),
          std::span<const int64_t>(lengths.data(), kHotKeysPerBatch));
      if (use_threshold_eviction && store.size() > kResidentKeyBudget) {
        size_t evicted_count = 0;
        const size_t eviction_count = store.size() - kResidentKeyBudget;
        store.evict_by_score<uint64_t>(0, eviction_count, evicted_keys,
                                       evicted_values, evicted_count);
        if (evicted_count != eviction_count)
          throw std::runtime_error(
              "threshold eviction did not meet its budget");
      }
    }
  }
  const auto finished = std::chrono::steady_clock::now();
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  const se::StorageStats stats = storage->stats();
  const double elapsed_seconds =
      std::chrono::duration<double>(finished - started).count();
  const double operations = static_cast<double>(kEpochs * kResidentKeyBudget);
  Result result{use_threshold_eviction, operations / elapsed_seconds,
                stats.swap_out_count, stats.spill_count};
  std::filesystem::remove_all(spill_directory);
  return result;
}

double median(std::vector<double> values) {
  std::ranges::sort(values);
  return values[values.size() / 2];
}

double median_swaps(const std::vector<Result> &results, bool threshold) {
  std::vector<double> values;
  for (const Result &result : results) {
    if (result.use_threshold_eviction == threshold)
      values.push_back(static_cast<double>(result.swap_out_count));
  }
  return median(std::move(values));
}

double median_qps(const std::vector<Result> &results, bool threshold) {
  std::vector<double> values;
  for (const Result &result : results) {
    if (result.use_threshold_eviction == threshold)
      values.push_back(result.operations_per_second);
  }
  return median(std::move(values));
}

double median_spills(const std::vector<Result> &results, bool threshold) {
  std::vector<double> values;
  for (const Result &result : results) {
    if (result.use_threshold_eviction == threshold)
      values.push_back(static_cast<double>(result.spill_count));
  }
  return median(std::move(values));
}

} // namespace

int main() {
  std::vector<Result> results;
  results.reserve(kRepetitions * 2);
  for (size_t repetition = 0; repetition < kRepetitions; ++repetition) {
    results.push_back(run_once(false, repetition));
    results.push_back(run_once(true, repetition));
  }

  const double baseline_swaps = median_swaps(results, false);
  const double threshold_swaps = median_swaps(results, true);
  const double baseline_spills = median_spills(results, false);
  const double threshold_spills = median_spills(results, true);
  const double baseline_qps = median_qps(results, false);
  const double threshold_qps = median_qps(results, true);
  if (baseline_qps <= 0.0 || threshold_qps <= 0.0)
    throw std::runtime_error(
        "eviction benchmark produced non-positive throughput");
  if (baseline_spills <= 0.0)
    throw std::runtime_error(
        "eviction benchmark did not exercise direct-spill pressure");
  if (threshold_spills > baseline_spills)
    throw std::runtime_error(
        "threshold eviction increased direct-spill allocations");
  std::cout << "baseline_median_swap_out_count=" << baseline_swaps << '\n';
  std::cout << "threshold_median_swap_out_count=" << threshold_swaps << '\n';
  std::cout << "baseline_median_spill_count=" << baseline_spills << '\n';
  std::cout << "threshold_median_spill_count=" << threshold_spills << '\n';
  std::cout << "baseline_median_operations_per_second=" << baseline_qps << '\n';
  std::cout << "threshold_median_operations_per_second=" << threshold_qps
            << '\n';
  if (baseline_swaps > 0.0) {
    std::cout << "swap_out_reduction_ratio="
              << (baseline_swaps - threshold_swaps) / baseline_swaps << '\n';
  }
  std::cout << "spill_reduction_ratio="
            << (baseline_spills - threshold_spills) / baseline_spills << '\n';
  return 0;
}
