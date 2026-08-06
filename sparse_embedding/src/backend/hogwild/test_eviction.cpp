#include "sparse_embedding/src/backend/hogwild/hogwild_embedding_store.hpp"
#include "sparse_embedding/src/eviction/score_policy.hpp"
#include <chrono>
#include <iostream>
#include <span>
#include <stdexcept>
#include <vector>

using namespace se;

namespace {

void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

} // namespace

int main() {
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

  std::vector<uint8_t> inserted(num_keys, 0);
  store.lookup_or_insert<uint64_t>(keys, values, lengths, inserted);
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

  std::vector<uint64_t> new_keys(10000);
  for (size_t i = 0; i < 10000; ++i)
    new_keys[i] = num_keys + i;
  std::vector<int64_t> new_lengths(10000, 128);
  std::vector<float> new_values(10000 * 128, 1.0f);

  auto start = std::chrono::steady_clock::now();
  std::vector<uint8_t> new_inserted(10000, 0);
  store.lookup_or_insert<uint64_t>(new_keys, new_values, new_lengths,
                                   new_inserted);
  auto end = std::chrono::steady_clock::now();

  std::this_thread::sleep_for(std::chrono::seconds(5));
  auto stats_after = storage->stats();

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
