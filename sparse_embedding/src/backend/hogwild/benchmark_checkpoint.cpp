#include <chrono>
#include <iostream>
#include <vector>

#include "sparse_embedding/src/backend/hogwild/hogwild_embedding_store.hpp"
#include "sparse_embedding/src/eviction/score_policy.hpp"

using namespace se;

int main() {
  std::cout << "==========================================\n";
  std::cout << "Starting Zero-Copy Lance Checkpoint Benchmark\n";
  std::cout << "==========================================\n";

  StorageOptions options;
  options.mode = StorageMode::kMemoryOnly;
  auto storage = create_page_storage(options);
  std::vector<size_t> dims = {128};

  HogwildEmbeddingStore<uint64_t, float, float, LfuScorePolicy> store(
      dims, {}, nullptr, LfuScorePolicy{},
      HogwildStoreOptions{.thread_budget = 4, .storage = storage});

  size_t num_keys = 500000;
  size_t batch_size = 4096;

  std::vector<uint64_t> keys(batch_size);
  std::vector<int64_t> lengths(batch_size, 128);
  std::vector<float> values(batch_size * 128, 1.0f);
  std::vector<uint8_t> inserted_mask(batch_size, 0);

  std::cout << "Inserting " << num_keys << " keys...\n";
  for (size_t i = 0; i < num_keys; i += batch_size) {
    for (size_t j = 0; j < batch_size; ++j) {
      keys[j] = i + j;
    }
    store.lookup_or_insert<uint64_t>(keys, values, lengths, inserted_mask);
  }

  std::string uri = "/tmp/lance_checkpoint_bench.lance";
  std::cout << "Saving checkpoint to " << uri << "...\n";

  // Clean up if exists
  if (std::system(("rm -rf " + uri).c_str()) != 0) {
    std::cerr << "Failed to remove previous checkpoint" << std::endl;
  }

  CheckpointExportOptions export_options;

  auto start = std::chrono::high_resolution_clock::now();
  store.save_lance_checkpoint<uint64_t>(uri, export_options);
  auto end = std::chrono::high_resolution_clock::now();

  auto duration_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();
  std::cout << "Export Time: " << duration_ms << " ms\n";

  // Now clear and load
  std::cout << "Clearing store...\n";
  store.clear();

  std::cout << "Loading checkpoint...\n";
  start = std::chrono::high_resolution_clock::now();
  store.load_lance_checkpoint<uint64_t>(uri);
  end = std::chrono::high_resolution_clock::now();

  duration_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();
  std::cout << "Load Time: " << duration_ms << " ms\n";

  return 0;
}
