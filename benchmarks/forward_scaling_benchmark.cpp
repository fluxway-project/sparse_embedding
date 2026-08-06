#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <optional>
#include <random>
#include <span>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <unordered_set>
#include <utility>
#include <vector>

#include "sparse_embedding/src/backend/hogwild/hogwild_embedding_store.hpp"
#include "sparse_embedding/src/execution/topology/thread_affinity.hpp"

namespace {

constexpr size_t kDimension = 32;
constexpr size_t kBatchSize = 4096;
constexpr size_t kVocabulary = kBatchSize;
constexpr size_t kRepetitions = 400;
constexpr std::array<size_t, 8> kThreadCounts{1, 2, 4, 8, 16, 24, 32, 64};

std::vector<uint64_t> make_unique_batch() {
  std::vector<uint64_t> keys(kBatchSize);
  for (size_t index = 0; index < keys.size(); ++index)
    keys[index] = index + 1;
  std::mt19937_64 generator(0x5EED5CA1EULL);
  std::shuffle(keys.begin(), keys.end(), generator);
  return keys;
}

double run_forward(size_t threads, const std::vector<uint64_t> &batch,
                   std::span<const int64_t> row_lengths, bool numa_aware) {
  const size_t physical_workers = threads;
  using Store =
      se::HogwildEmbeddingStore<uint64_t, float, float, se::NoScorePolicy>;
  Store store(
      {kDimension}, {}, nullptr, se::NoScorePolicy{},
      se::HogwildStoreOptions{
          .thread_budget = physical_workers,
          .locality = se::LocalityPolicy{
              numa_aware ? se::LocalityMode::kNuma : se::LocalityMode::kAuto,
              numa_aware ? se::LocalityMode::kNuma : se::LocalityMode::kAuto}});
  store.reserve(kVocabulary, kDimension);

  std::vector<uint64_t> warmup_keys(kVocabulary);
  for (size_t index = 0; index < warmup_keys.size(); ++index)
    warmup_keys[index] = index + 1;
  std::vector<int64_t> warmup_lengths(kVocabulary, kDimension);
  std::vector<float> warmup_values(kVocabulary * kDimension);
  std::vector<uint8_t> inserted(kVocabulary);
  store.lookup_or_insert<uint64_t>(warmup_keys, warmup_values, warmup_lengths,
                                   inserted);

  std::vector<float> output(kBatchSize * kDimension);
  store.forward_lookup<uint64_t>(batch, output, row_lengths);
  const auto started = std::chrono::steady_clock::now();
  for (size_t repetition = 0; repetition < kRepetitions; ++repetition)
    store.forward_lookup<uint64_t>(batch, output, row_lengths);
  const auto elapsed = std::chrono::steady_clock::now() - started;
  const double seconds = std::chrono::duration<double>(elapsed).count();
  if (seconds <= 0.0)
    throw std::runtime_error("forward benchmark duration is non-positive");
  return static_cast<double>(kBatchSize * kRepetitions) / seconds / 1e6;
}

} // namespace

size_t parse_thread_count(std::string_view value) {
  if (value.empty())
    throw std::invalid_argument("thread count is empty");
  size_t result = 0;
  const auto parsed =
      std::from_chars(value.data(), value.data() + value.size(), result);
  if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() ||
      result == 0)
    throw std::invalid_argument("thread count is invalid");
  return result;
}

int main(int argc, char **argv) {
  bool numa_aware = false;
  std::optional<size_t> only_threads;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--numa") {
      numa_aware = true;
      continue;
    }
    constexpr std::string_view prefix = "--threads=";
    if (argument.starts_with(prefix)) {
      if (only_threads.has_value())
        throw std::invalid_argument("thread count specified twice");
      only_threads = parse_thread_count(argument.substr(prefix.size()));
      continue;
    }
    throw std::invalid_argument("expected --numa or --threads=N");
  }
  const std::vector<int> cpus = se::ThreadAffinity::allowed_cpus();
  const std::vector<uint64_t> batch = make_unique_batch();
  std::unordered_set<uint64_t> unique_keys(batch.begin(), batch.end());
  const size_t duplicate_rows = batch.size() - unique_keys.size();
  std::cout
      << "seed=0x5EED5CA1E vocabulary=" << kVocabulary
      << " batch=" << kBatchSize << " dimension=" << kDimension
      << " allowed_cpus=" << cpus.size() << " minimum_bytes_per_worker="
      << (numa_aware
              ? se::OperationDispatchConfig::kNumaMinimumBytesPerThread
              : se::OperationDispatchConfig::kDefaultMinimumBytesPerThread)
      << " numa_physical_cores="
      << (numa_aware ? se::NumaTopology::physical_core_count() : 0)
      << " numa_aware=" << (numa_aware ? "true" : "false")
      << " unique_keys=" << unique_keys.size()
      << " duplicate_rows=" << duplicate_rows << '\n';
  if (duplicate_rows != 0)
    throw std::runtime_error(
        "forward benchmark batch must not contain duplicates");
  const std::vector<int64_t> lengths(kBatchSize, kDimension);
  std::array<double, kThreadCounts.size()> throughput{};
  for (size_t index = 0; index < kThreadCounts.size(); ++index) {
    if (only_threads.has_value() && kThreadCounts[index] != *only_threads)
      continue;
    if (numa_aware && kThreadCounts[index] < se::NumaTopology::node_count())
      continue;
    throughput[index] =
        run_forward(kThreadCounts[index], batch, lengths, numa_aware);
    std::cout << "forward_mops_" << kThreadCounts[index] << '=' << std::fixed
              << std::setprecision(3) << throughput[index] << '\n';
  }
  if (only_threads.has_value())
    return throughput[std::ranges::find(kThreadCounts, *only_threads) -
                      kThreadCounts.begin()] > 0.0
               ? 0
               : 1;
  const size_t baseline_index = numa_aware ? 1 : 0;
  if (throughput[baseline_index] <= 0.0)
    throw std::runtime_error(
        "single-thread forward throughput is non-positive");
  for (size_t index = baseline_index; index < 3; ++index) {
    const double ratio = throughput[index + 1] / throughput[baseline_index];
    std::cout << "forward_scaling_" << kThreadCounts[index + 1]
              << "_ratio=" << std::fixed << std::setprecision(3) << ratio
              << '\n';
  }
  std::cout << "forward_scaling_16_vs_8_ratio=" << std::fixed
            << std::setprecision(3) << throughput[4] / throughput[3] << '\n';
  std::cout << "forward_scaling_24_vs_16_ratio=" << std::fixed
            << std::setprecision(3) << throughput[5] / throughput[4] << '\n';
  if (numa_aware)
    std::cout << "forward_scaling_32_vs_24_ratio=" << std::fixed
              << std::setprecision(3) << throughput[6] / throughput[5] << '\n';
}
