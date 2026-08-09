#include "sparse_embedding/src/backend/hogwild/hogwild_embedding_store.hpp"
#include "sparse_embedding/src/common/function.hpp"
#include "sparse_embedding/src/interface/optimizer_interface.hpp"
#include "sparse_embedding/src/optimizer/sgd_optimizer.hpp"
#include "sparse_embedding/src/storage/page_storage.hpp"

#include <algorithm>
#include <array>
#include <barrier>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

constexpr size_t kDimension = 128;
constexpr size_t kWorkers = 6;
constexpr size_t kInitialKeys = 16'000;
constexpr size_t kBatch = 256;
constexpr size_t kRounds = 8;
constexpr size_t kMemoryLimit = 64 << 20;
constexpr size_t kPressurePageBytes = 1 << 20;
constexpr size_t kPressurePageCount = 64;

void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

struct DirectoryGuard {
  std::filesystem::path path;
  ~DirectoryGuard() { std::filesystem::remove_all(path); }
};

} // namespace

int main() {
  DirectoryGuard spill_directory{
      std::filesystem::temp_directory_path() /
      ("sparse-embedding-all-paths-" + std::to_string(getpid()))};
  std::filesystem::remove_all(spill_directory.path);
  std::filesystem::create_directories(spill_directory.path);

  se::StorageOptions options;
  options.mode = se::StorageMode::kAuto;
  options.memory_limit_bytes = kMemoryLimit;
  options.spill_directory = spill_directory.path.string();
  options.max_spill_bytes = size_t{128} << 20;
  options.watermark = se::WatermarkConfig{0.75, 0.50};
  auto storage = se::create_page_storage(std::move(options));

  se::SgdConfig sgd_config;
  sgd_config.lr = 0.01F;
  auto optimizer_factory = pro::make_proxy<se::FunctionFacade<
      pro::proxy<se::OptimizerFacade<float, float>>(size_t)>>(
      [storage, sgd_config](size_t dim) mutable {
        return pro::make_proxy<
            se::OptimizerFacade<float, float>,
            se::SgdOptimizer<float, float, se::StaticLearningRate>>(
            dim, sgd_config, storage, se::StaticLearningRate{sgd_config.lr});
      });
  using Store =
      se::HogwildEmbeddingStore<uint64_t, float, float, se::LfuScorePolicy>;
  Store store(
      {kDimension}, {}, std::move(optimizer_factory), se::LfuScorePolicy{},
      se::HogwildStoreOptions{.thread_budget = kWorkers, .storage = storage});

  std::vector<uint64_t> initial_keys(kInitialKeys);
  std::vector<int64_t> initial_lengths(kInitialKeys, kDimension);
  std::vector<float> initial_values(kInitialKeys * kDimension, 1.0F);
  std::iota(initial_keys.begin(), initial_keys.end(), uint64_t{1});
  constexpr size_t kInitialBatch = 256;
  for (size_t begin = 0; begin < kInitialKeys; begin += kInitialBatch) {
    const size_t count = std::min(kInitialBatch, kInitialKeys - begin);
    store.lookup_or_insert<uint64_t>(
        std::span<const uint64_t>(initial_keys).subspan(begin, count),
        std::span<float>(initial_values)
            .subspan(begin * kDimension, count * kDimension),
        std::span<const int64_t>(initial_lengths).subspan(begin, count));
  }
  if (store.size() != kInitialKeys)
    throw std::runtime_error("initial pressure population lost keys");

  std::vector<se::StoragePage> pressure_pages;
  pressure_pages.reserve(kPressurePageCount);
  for (size_t page = 0; page < kPressurePageCount; ++page) {
    pressure_pages.push_back(storage->allocate(
        kPressurePageBytes, 64, [] { return uint64_t{0}; },
        se::PageEvictionMode::kEnabled));
    pressure_pages.back().data()[0] = static_cast<std::byte>(page);
  }

  std::barrier start(static_cast<std::ptrdiff_t>(kWorkers + 1));
  std::array<std::exception_ptr, kWorkers> errors{};
  std::vector<std::thread> workers;
  workers.reserve(kWorkers);
  for (size_t worker = 0; worker < kWorkers; ++worker) {
    workers.emplace_back([&, worker] {
      try {
        start.arrive_and_wait();
        for (size_t round = 0; round < kRounds; ++round) {
          std::vector<uint64_t> keys(kBatch);
          std::vector<int64_t> lengths(kBatch, kDimension);
          std::vector<float> values(kBatch * kDimension);
          std::vector<uint8_t> inserted(kBatch);
          if (worker == 0) {
            for (size_t i = 0; i < kBatch; ++i)
              keys[i] = 1 + ((round * kBatch + i) % 4096);
            store.forward_lookup<uint64_t>(keys, values, lengths);
          } else if (worker == 1) {
            for (size_t i = 0; i < kBatch; ++i)
              keys[i] = 10'000 + ((round * kBatch + i) % 4096);
            std::fill(values.begin(), values.end(), 2.0F);
            store.assign<uint64_t>(keys, values, lengths);
          } else if (worker == 2) {
            for (size_t i = 0; i < kBatch; ++i)
              keys[i] = 20'000 + ((round * kBatch + i) % 4096);
            store.contains<uint64_t>(keys, inserted);
          } else if (worker == 3) {
            for (size_t i = 0; i < kBatch; ++i)
              keys[i] = 60'000 + i;
            store.erase<uint64_t>(keys);
          } else if (worker == 4) {
            for (size_t i = 0; i < kBatch; ++i)
              keys[i] = 40'000 + i;
            std::fill(values.begin(), values.end(), 3.0F);
            store.lookup_or_insert<uint64_t>(keys, values, lengths);
          } else {
            for (size_t i = 0; i < kBatch; ++i)
              keys[i] = 30'000 + (i % 64);
            std::fill(values.begin(), values.end(), 1.0F);
            store.accumulate_gradients<uint64_t>(keys, values, lengths);
            store.optimizer_step();
          }
        }
      } catch (...) {
        errors[worker] = std::current_exception();
      }
    });
  }
  start.arrive_and_wait();
  for (auto &worker : workers)
    worker.join();
  for (const auto &error : errors) {
    if (error != nullptr)
      std::rethrow_exception(error);
  }

  storage->drain_page_eviction_backlog(std::chrono::seconds(30));
  const se::StorageStats stats = storage->stats();
  require(stats.swap_out_count > 0,
          "small-memory pressure did not perform a real swap-out");
  require(stats.eviction_selection_count > 0,
          "small-memory pressure did not select eviction candidates");
  require(stats.memory_bytes <= static_cast<size_t>(kMemoryLimit * 0.50),
          "small-memory pressure did not drain to the low watermark");
  for (size_t page = 0; page < pressure_pages.size(); ++page) {
    require(pressure_pages[page].data()[0] == static_cast<std::byte>(page),
            "swapped pressure page changed its payload");
  }

  std::vector<float> output(kBatch * kDimension);
  std::vector<uint64_t> verify_keys(kBatch);
  std::vector<int64_t> verify_lengths(kBatch, kDimension);
  for (size_t i = 0; i < kBatch; ++i)
    verify_keys[i] = 1 + i;
  store.forward_lookup<uint64_t>(verify_keys, output, verify_lengths);
  require(std::all_of(output.begin(), output.end(),
                      [](float value) { return std::isfinite(value); }),
          "post-swap forward returned a non-finite payload");

  std::cout << "initial_keys=" << kInitialKeys << '\n'
            << "workers=" << kWorkers << '\n'
            << "swap_out_count=" << stats.swap_out_count << '\n'
            << "spill_count=" << stats.spill_count << '\n'
            << "eviction_selection_count=" << stats.eviction_selection_count
            << '\n'
            << "memory_bytes=" << stats.memory_bytes << '\n';
  return 0;
}
