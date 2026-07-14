#include "sparse_embedding/src/backend/hogwild/hogwild_embedding_store.hpp"
#include "sparse_embedding/src/storage/page_storage.hpp"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string_view>
#include <thread>

namespace {

void require(bool condition, const char *message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

} // namespace

int main() {
  se::StorageOptions options;
  options.memory_limit_bytes = 0;
  options.spill_directory = std::filesystem::temp_directory_path();
  options.max_spill_bytes = 4 << 20;
  auto storage = se::PageStorage::create(std::move(options));

  using Store =
      se::HogwildEmbeddingStore<uint64_t, float, float, se::NoScorePolicy>;
  Store store({8}, 1, {}, 1024, nullptr, se::NoScorePolicy{}, 16, storage);

  constexpr std::array<uint64_t, 3> keys{11, 22, 33};
  constexpr std::array<int64_t, 3> lengths{8, 8, 8};
  std::array<float, 24> values{};
  std::array<uint8_t, 3> inserted{};
  store.lookup_or_insert<uint64_t>(keys, values, lengths, inserted);

  require(inserted == std::array<uint8_t, 3>{1, 1, 1},
          "keys were not inserted");
  require(store.size() == keys.size(), "store size mismatch");
  require(storage->stats().spill_bytes > 0,
          "store did not use shared spill storage");

  bool rejected_before_allocation = false;
  try {
    constexpr std::array<size_t, 1> excessive_reserve{
        std::numeric_limits<size_t>::max()};
    store.reserve_per_shard(excessive_reserve, 8);
  } catch (const std::length_error &error) {
    rejected_before_allocation = std::string_view(error.what()) ==
                                 "reserve exceeds max_active_keys capacity";
  }
  require(rejected_before_allocation,
          "excessive reserve was not rejected before allocation");

  se::StorageOptions duplicate_options;
  duplicate_options.mode = se::StorageMode::kMemoryOnly;
  duplicate_options.memory_limit_bytes = 64 << 20;
  auto duplicate_storage =
      se::PageStorage::create(std::move(duplicate_options));
  Store duplicate_store({8}, 32, {}, 1, nullptr, se::NoScorePolicy{}, 16,
                        duplicate_storage);
  require(duplicate_storage->stats().memory_bytes == 0,
          "empty store eagerly allocated transient pages");
  std::array<uint64_t, 4096> duplicate_keys{};
  duplicate_keys.fill(44);
  std::array<int64_t, 4096> duplicate_lengths{};
  duplicate_lengths.fill(8);
  std::array<float, 4096 * 8> duplicate_values{};
  std::array<uint8_t, 4096> duplicate_inserted{};
  duplicate_store.lookup_or_insert<uint64_t>(
      duplicate_keys, duplicate_values, duplicate_lengths, duplicate_inserted);
  require(std::accumulate(duplicate_inserted.begin(), duplicate_inserted.end(),
                          size_t{0}) == 1,
          "concurrent duplicate insertion published more than one key");
  require(duplicate_store.size() == 1,
          "concurrent duplicate insertion consumed active-key capacity");

  se::StorageOptions reuse_options;
  reuse_options.memory_limit_bytes = 0;
  reuse_options.spill_directory = std::filesystem::temp_directory_path();
  reuse_options.max_spill_bytes = 1 << 20;
  auto reuse_storage = se::PageStorage::create(std::move(reuse_options));
  Store reuse_store({8}, 1, {}, 1, nullptr, se::NoScorePolicy{}, 16,
                    reuse_storage);
  std::array<uint64_t, 1> reuse_key{};
  constexpr std::array<int64_t, 1> reuse_length{8};
  std::array<float, 8> reuse_value{};
  std::array<uint8_t, 1> reuse_inserted{};
  for (uint64_t key = 1; key <= 10000; ++key) {
    reuse_key[0] = key;
    reuse_store.lookup_or_insert<uint64_t>(reuse_key, reuse_value, reuse_length,
                                           reuse_inserted);
    reuse_store.erase<uint64_t>(reuse_key);
  }
  require(reuse_storage->stats().spill_count == 1,
          "erased embedding indices were not reused");

  bool arena_limit_rejected = false;
  try {
    Store oversized_store({8}, 1, {}, size_t{1} << 40, nullptr,
                          se::NoScorePolicy{}, 16, duplicate_storage);
  } catch (const std::length_error &error) {
    arena_limit_rejected = std::string_view(error.what()) ==
                           "max_active_keys exceeds arena capacity";
  }
  require(arena_limit_rejected,
          "oversized key index was not rejected before allocation");

  se::TransientArena<float> bounded_transient(8, duplicate_storage, 1);
  bounded_transient.get_or_allocate(1);
  bool transient_limit_rejected = false;
  try {
    bounded_transient.get_or_allocate(2);
  } catch (const std::length_error &error) {
    transient_limit_rejected =
        std::string_view(error.what()) == "transient key capacity exceeded";
  }
  require(transient_limit_rejected, "transient key capacity was not enforced");

  Store lifecycle_store({8}, 4, {}, 64, nullptr, se::NoScorePolicy{}, 16,
                        duplicate_storage);
  std::array<uint64_t, 32> lifecycle_keys{};
  std::iota(lifecycle_keys.begin(), lifecycle_keys.end(), uint64_t{1});
  std::array<int64_t, 32> lifecycle_lengths{};
  lifecycle_lengths.fill(8);
  std::array<float, 32 * 8> lifecycle_values{};
  std::array<uint8_t, 32> lifecycle_inserted{};
  lifecycle_store.lookup_or_insert<uint64_t>(
      lifecycle_keys, lifecycle_values, lifecycle_lengths, lifecycle_inserted);
  std::atomic<bool> reader_ok{true};
  std::thread reader([&] {
    try {
      for (size_t i = 0; i < 200; ++i) {
        lifecycle_store.lookup<uint64_t>(lifecycle_keys, lifecycle_values,
                                         lifecycle_lengths);
      }
    } catch (...) {
      reader_ok.store(false, std::memory_order_relaxed);
    }
  });
  for (size_t i = 0; i < 50; ++i) {
    lifecycle_store.clear();
    lifecycle_store.lookup_or_insert<uint64_t>(lifecycle_keys, lifecycle_values,
                                               lifecycle_lengths,
                                               lifecycle_inserted);
  }
  reader.join();
  require(reader_ok.load(std::memory_order_relaxed),
          "concurrent clear invalidated an active reader");
}
