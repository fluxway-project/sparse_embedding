#include "sparse_embedding/src/backend/hogwild/hogwild_embedding_store.hpp"
#include "sparse_embedding/src/storage/page_storage.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
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
  auto storage = se::create_page_storage(std::move(options));

  using Store =
      se::HogwildEmbeddingStore<uint64_t, float, float, se::NoScorePolicy>;
  using ScoredStore =
      se::HogwildEmbeddingStore<uint64_t, float, float, se::ManualScorePolicy>;
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
      se::create_page_storage(std::move(duplicate_options));
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

  ScoredStore memory_scored_store(
      {8}, 1, {}, 16, nullptr, se::ManualScorePolicy{}, 16, duplicate_storage);
  constexpr std::array<uint64_t, 1> memory_scored_key{55};
  constexpr std::array<int64_t, 1> memory_scored_length{8};
  std::array<float, 8> memory_scored_values{};
  std::array<uint8_t, 1> memory_scored_inserted{};
  memory_scored_store.lookup_or_insert<uint64_t>(
      memory_scored_key, memory_scored_values, memory_scored_length,
      memory_scored_inserted);
  std::unique_lock memory_scored_lifecycle(
      *duplicate_storage->lifecycle_mutex());
  std::atomic<bool> memory_scored_operation_completed{false};
  std::thread memory_scored_operation([&] {
    memory_scored_store.contains<uint64_t>(memory_scored_key,
                                           memory_scored_inserted);
    memory_scored_operation_completed.store(true, std::memory_order_release);
  });
  for (size_t attempt = 0;
       attempt < 100 &&
       !memory_scored_operation_completed.load(std::memory_order_acquire);
       ++attempt) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  const bool memory_scored_completed_before_unlock =
      memory_scored_operation_completed.load(std::memory_order_acquire);
  memory_scored_lifecycle.unlock();
  memory_scored_operation.join();
  require(memory_scored_completed_before_unlock,
          "MemoryOnly scored store used the shared swap lifecycle");

  se::StorageOptions reuse_options;
  reuse_options.memory_limit_bytes = 0;
  reuse_options.spill_directory = std::filesystem::temp_directory_path();
  reuse_options.max_spill_bytes = 1 << 20;
  auto reuse_storage = se::create_page_storage(std::move(reuse_options));
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

  se::StorageOptions watermark_options;
  watermark_options.memory_limit_bytes = 300 << 10;
  watermark_options.spill_directory = std::filesystem::temp_directory_path();
  watermark_options.max_spill_bytes = 1 << 20;
  auto watermark_storage =
      se::create_page_storage(std::move(watermark_options));
  ScoredStore watermark_store({8}, 1, {}, 16, nullptr, se::ManualScorePolicy{},
                              16, watermark_storage);
  constexpr std::array<uint64_t, 1> watermark_key{91};
  constexpr std::array<int64_t, 1> watermark_length{8};
  std::array<float, 8> watermark_values{};
  std::array<uint8_t, 1> watermark_inserted{};
  watermark_store.lookup_or_insert<uint64_t>(
      watermark_key, watermark_values, watermark_length, watermark_inserted);
  for (size_t attempt = 0;
       attempt < 100 && watermark_storage->stats().swap_out_count == 0;
       ++attempt) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  require(watermark_storage->stats().swap_out_count == 1,
          "scored arena page was not registered for background swap-out");
  watermark_values.fill(7.0F);
  watermark_store.assign<uint64_t>(watermark_key, watermark_values,
                                   watermark_length);
  watermark_values.fill(0.0F);
  watermark_store.lookup<uint64_t>(watermark_key, watermark_values,
                                   watermark_length);
  require(std::all_of(watermark_values.begin(), watermark_values.end(),
                      [](float value) { return value == 7.0F; }),
          "swapped page lost a post-swap assignment");

  se::StorageOptions shared_options;
  shared_options.memory_limit_bytes = 300 << 10;
  shared_options.spill_directory = std::filesystem::temp_directory_path();
  shared_options.max_spill_bytes = 4 << 20;
  auto shared_storage = se::create_page_storage(std::move(shared_options));
  constexpr std::array<uint64_t, 1> shared_key{123};
  constexpr std::array<int64_t, 1> shared_length{8};
  std::array<float, 8> shared_values{};
  std::array<uint8_t, 1> shared_inserted{};
  {
    auto first = std::make_unique<ScoredStore>(
        std::vector<size_t>{8}, 1, ScoredStore::Initializer{}, 16, nullptr,
        se::ManualScorePolicy{}, 16, shared_storage);
    auto second = std::make_unique<ScoredStore>(
        std::vector<size_t>{8}, 1, ScoredStore::Initializer{}, 16, nullptr,
        se::ManualScorePolicy{}, 16, shared_storage);
    first->lookup_or_insert<uint64_t>(shared_key, shared_values, shared_length,
                                      shared_inserted);
    second->lookup_or_insert<uint64_t>(shared_key, shared_values, shared_length,
                                       shared_inserted);
    first.reset();
    shared_values.fill(3.0F);
    second->assign<uint64_t>(shared_key, shared_values, shared_length);
    shared_values.fill(0.0F);
    second->lookup<uint64_t>(shared_key, shared_values, shared_length);
    require(std::all_of(shared_values.begin(), shared_values.end(),
                        [](float value) { return value == 3.0F; }),
            "shared storage lost the surviving store after peer destruction");

    auto memory_store = std::make_unique<Store>(
        std::vector<size_t>{8}, 1, Store::Initializer{}, 16, nullptr,
        se::NoScorePolicy{}, 16, shared_storage);
    std::array<uint64_t, 1> memory_key{456};
    std::array<float, 8> memory_values{};
    std::array<uint8_t, 1> memory_inserted{};
    memory_store->lookup_or_insert<uint64_t>(memory_key, memory_values,
                                             shared_length, memory_inserted);
    require(memory_inserted[0] == 1,
            "pure memory store could not share a scored storage");
    std::unique_lock scored_lifecycle(*shared_storage->lifecycle_mutex());
    std::atomic<bool> memory_operation_completed{false};
    std::thread memory_operation([&] {
      memory_store->contains<uint64_t>(memory_key, memory_inserted);
      memory_operation_completed.store(true, std::memory_order_release);
    });
    for (size_t attempt = 0; attempt < 100 && !memory_operation_completed.load(
                                                  std::memory_order_acquire);
         ++attempt) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    const bool memory_completed_before_unlock =
        memory_operation_completed.load(std::memory_order_acquire);
    scored_lifecycle.unlock();
    memory_operation.join();
    require(memory_completed_before_unlock,
            "pure memory store was blocked by scored swap lifecycle");
  }

  se::StorageOptions reverse_options;
  reverse_options.memory_limit_bytes = 300 << 10;
  reverse_options.spill_directory = std::filesystem::temp_directory_path();
  reverse_options.max_spill_bytes = 4 << 20;
  auto reverse_storage = se::create_page_storage(std::move(reverse_options));
  auto surviving = std::make_unique<ScoredStore>(
      std::vector<size_t>{8}, 1, ScoredStore::Initializer{}, 16, nullptr,
      se::ManualScorePolicy{}, 16, reverse_storage);
  auto departing = std::make_unique<ScoredStore>(
      std::vector<size_t>{8}, 1, ScoredStore::Initializer{}, 16, nullptr,
      se::ManualScorePolicy{}, 16, reverse_storage);
  departing.reset();
  surviving->lookup_or_insert<uint64_t>(shared_key, shared_values,
                                        shared_length, shared_inserted);
  for (size_t attempt = 0;
       attempt < 100 && reverse_storage->stats().swap_out_count == 0;
       ++attempt) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  require(reverse_storage->stats().swap_out_count > 0,
          "surviving shared store lost background page eviction proxy");

  se::StorageOptions stress_options;
  stress_options.mode = se::StorageMode::kMemoryOnly;
  stress_options.memory_limit_bytes = 128 << 20;
  auto stress_storage = se::create_page_storage(std::move(stress_options));
  Store stress_store({8}, 8, {}, 1000000, nullptr, se::NoScorePolicy{}, 16,
                     stress_storage);
  constexpr size_t stress_batch_size = 8192;
  constexpr size_t stress_rounds = 8;
  std::vector<uint64_t> stress_keys(stress_batch_size);
  std::vector<int64_t> stress_lengths(stress_batch_size, 8);
  std::vector<float> stress_values(stress_batch_size * 8);
  std::vector<uint8_t> stress_inserted(stress_batch_size);
  size_t stress_inserted_total = 0;
  for (size_t round = 0; round < stress_rounds; ++round) {
    for (size_t i = 0; i < stress_batch_size; ++i)
      stress_keys[i] = round * stress_batch_size + i;
    stress_store.lookup_or_insert<uint64_t>(stress_keys, stress_values,
                                            stress_lengths, stress_inserted);
    stress_inserted_total += std::accumulate(stress_inserted.begin(),
                                             stress_inserted.end(), size_t{0});
  }
  const size_t expected_stress_size = stress_batch_size * stress_rounds;
  if (stress_inserted_total != expected_stress_size)
    throw std::runtime_error("8-thread unique insertion mask mismatch: got " +
                             std::to_string(stress_inserted_total) +
                             " expected " +
                             std::to_string(expected_stress_size));
  const size_t observed_stress_size = stress_store.size();
  if (observed_stress_size != expected_stress_size)
    throw std::runtime_error("8-thread unique insertion lost keys: got " +
                             std::to_string(observed_stress_size) +
                             " expected " +
                             std::to_string(expected_stress_size));

  ScoredStore scored_stress_store({8}, 8, {}, expected_stress_size, nullptr,
                                  se::ManualScorePolicy{}, 16, stress_storage);
  size_t scored_inserted_total = 0;
  for (size_t round = 0; round < stress_rounds; ++round) {
    for (size_t i = 0; i < stress_batch_size; ++i)
      stress_keys[i] = round * stress_batch_size + i;
    scored_stress_store.lookup_or_insert<uint64_t>(
        stress_keys, stress_values, stress_lengths, stress_inserted);
    scored_inserted_total += std::accumulate(stress_inserted.begin(),
                                             stress_inserted.end(), size_t{0});
  }
  require(scored_inserted_total == expected_stress_size,
          "8-thread scored insertion mask mismatch");
  require(scored_stress_store.size() == expected_stress_size,
          "8-thread scored insertion lost keys");
}
