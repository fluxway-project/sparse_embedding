#include "sparse_embedding/src/backend/hogwild/hogwild_embedding_store.hpp"
#include "sparse_embedding/src/eviction/score_policy.hpp"
#include "sparse_embedding/src/optimizer/adam_optimizer.hpp"

#include <algorithm>
#include <array>
#include <barrier>
#include <chrono>
#include <cmath>
#include <exception>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <unordered_set>
#include <vector>

using namespace se;

namespace {

void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

void run_optimizer_step_eviction_regression() {
  StorageOptions options;
  options.memory_limit_bytes = 64 << 20;
  options.spill_directory = std::filesystem::temp_directory_path();
  options.max_spill_bytes = 256 << 20;
  auto storage = create_page_storage(std::move(options));
  AdamConfig config;
  auto factory =
      pro::make_proxy<FunctionFacade<OptimizerExtension<float, float>(size_t)>>(
          [storage, config](size_t dim) mutable {
            using Adam = AdamOptimizer<float, float, StaticLearningRate>;
            return pro::make_proxy<OptimizerExtensionFacade<float, float>,
                                   Adam>(dim, config, storage,
                                         StaticLearningRate{1.0e-3F});
          });
  using Store =
      HogwildEmbeddingStore<uint64_t, float, float, ManualScorePolicy>;
  Store store({1}, {}, std::move(factory), ManualScorePolicy{},
              HogwildStoreOptions{.thread_budget = 4, .storage = storage});
  constexpr size_t key_count = 1024;
  constexpr size_t eviction_count = key_count / 2;
  std::vector<uint64_t> keys(key_count);
  std::iota(keys.begin(), keys.end(), uint64_t{1});
  std::vector<int64_t> lengths(key_count, 1);
  std::vector<float> values(key_count);
  std::iota(values.begin(), values.end(), 1.0F);
  store.lookup_or_insert<uint64_t>(keys, values, lengths);
  std::vector<uint64_t> scores(key_count, 1);
  store.assign_scores<uint64_t>(keys, scores);

  std::barrier start(2);
  std::array<std::exception_ptr, 2> errors;
  std::vector<uint64_t> evicted_keys(eviction_count);
  std::vector<float> evicted_values(eviction_count);
  size_t evicted_count = 0;
  std::thread optimizer([&] {
    try {
      start.arrive_and_wait();
      std::vector<float> gradients(key_count, 1.0F);
      for (size_t round = 0; round < 128; ++round) {
        store.accumulate_gradients<uint64_t>(keys, gradients, lengths);
        store.optimizer_step();
      }
    } catch (...) {
      errors[0] = std::current_exception();
    }
  });
  std::thread eviction([&] {
    try {
      start.arrive_and_wait();
      store.evict_by_score<uint64_t>(1, eviction_count, evicted_keys,
                                     evicted_values, evicted_count);
    } catch (...) {
      errors[1] = std::current_exception();
    }
  });
  optimizer.join();
  eviction.join();
  for (const auto &error : errors)
    if (error != nullptr)
      std::rethrow_exception(error);

  require(evicted_count == eviction_count,
          "optimizer/eviction regression evicted the wrong key count");
  require(std::unordered_set<uint64_t>(evicted_keys.begin(), evicted_keys.end())
                  .size() == eviction_count,
          "optimizer/eviction regression produced duplicate keys");
  require(std::ranges::all_of(evicted_values,
                              [](float value) { return std::isfinite(value); }),
          "optimizer/eviction regression produced a non-finite value");
  std::vector<uint8_t> present(key_count);
  store.contains<uint64_t>(keys, present);
  require(std::ranges::count(present, uint8_t{1}) == key_count - eviction_count,
          "optimizer/eviction regression produced an invalid survivor count");
}

} // namespace

int main(int argc, char **argv) {
  if (argc > 1) {
    if (argc != 2 || std::string_view(argv[1]) != "optimizer_step_eviction")
      throw std::invalid_argument("unknown eviction regression");
    run_optimizer_step_eviction_regression();
    return 0;
  }
  run_optimizer_step_eviction_regression();
  std::cout << "Starting test..." << std::endl;
  StorageOptions options;
  options.mode = StorageMode::kAuto;
  options.memory_limit_bytes = 10 * 1024 * 1024; // 10MB
  options.spill_directory = "/tmp";
  options.max_spill_bytes = 100 * 1024 * 1024;   // 100MB
  options.watermark = WatermarkConfig{0.8, 0.6}; // Swap at 8MB, stop at 6MB

  auto storage = create_page_storage(options);
  std::vector<size_t> dims = {128};
  HogwildEmbeddingStore<uint64_t, float, float, ManualScorePolicy> store(
      dims, {}, nullptr, ManualScorePolicy{},
      HogwildStoreOptions{.thread_budget = 4, .storage = storage});

  size_t num_keys = 50000;
  std::vector<uint64_t> keys(num_keys);
  std::vector<int64_t> lengths(num_keys, 128);
  std::vector<float> values(num_keys * 128, 1.0f);
  for (size_t i = 0; i < num_keys; ++i)
    keys[i] = i;

  store.lookup_or_insert<uint64_t>(keys, values, lengths);
  require(store.size() == num_keys, "inserted key count mismatch");
  std::vector<uint64_t> scores(num_keys, 1);
  scores[0] = 1000;
  store.assign_scores<uint64_t>(keys, scores);
  std::this_thread::sleep_for(std::chrono::seconds(5));
  auto stats_before = storage->stats();
  std::cout << "Stats after insert: swap_out_count="
            << stats_before.swap_out_count << std::endl;

  size_t evicted_count = 0;
  std::vector<uint64_t> evicted_keys(10000);
  std::vector<float> evicted_values(10000 * 128);
  require(!store.evict_by_score<uint64_t>(100, 10000, evicted_keys,
                                          evicted_values, evicted_count),
          "score eviction returned an error");
  require(evicted_count == 10000, "score eviction count mismatch");
  require(store.size() == num_keys - evicted_count,
          "post-eviction key count mismatch");
  std::vector<uint8_t> hot_exists(1);
  store.contains<uint64_t>(std::span<const uint64_t>(keys.data(), 1),
                           hot_exists);
  require(hot_exists[0] == 1, "hot key was evicted");
  std::cout << "Evicted " << evicted_count << " keys." << std::endl;
  const auto stats_after_eviction = storage->stats();
  std::cout << "Retired bytes after eviction: "
            << stats_after_eviction.retired_bytes << std::endl;
  std::cout << "Retirement pressure state after eviction: "
            << static_cast<int>(stats_after_eviction.retirement_pressure_state)
            << std::endl;
  require(stats_after_eviction.retired_bytes == 0,
          "eviction retirement backlog did not drain at quiescence");

  std::vector<uint64_t> new_keys(10000);
  for (size_t i = 0; i < 10000; ++i)
    new_keys[i] = num_keys + i;
  std::vector<int64_t> new_lengths(10000, 128);
  std::vector<float> new_values(10000 * 128, 1.0f);

  auto start = std::chrono::steady_clock::now();
  store.lookup_or_insert<uint64_t>(new_keys, new_values, new_lengths);
  require(store.size() == num_keys,
          "fragmented insertion did not replace all evicted keys");
  auto end = std::chrono::steady_clock::now();

  std::this_thread::sleep_for(std::chrono::seconds(5));
  auto stats_after = storage->stats();
  require(stats_after.retired_bytes == 0,
          "fragmented insertion left a retirement backlog");

  std::cout << "Stats after fragmented insert: swap_out_count="
            << stats_after.swap_out_count << std::endl;
  std::cout << "Swap outs caused by fragmentation: "
            << (stats_after.swap_out_count - stats_before.swap_out_count)
            << std::endl;
  std::cout << "Time taken: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                     start)
                   .count()
            << " ms" << std::endl;
  return 0;
}
