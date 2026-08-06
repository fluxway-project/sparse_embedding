#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <thread>
#include <unordered_set>
#include <vector>

#include "sparse_embedding/src/backend/hogwild/core/cohort_page_allocator.hpp"

namespace {

void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

uint64_t splitmix_hash(uint64_t key) {
  key ^= key >> 30;
  key *= 0xbf58476d1ce4e5b9ULL;
  key ^= key >> 27;
  key *= 0x94d049bb133111ebULL;
  return key ^ (key >> 31);
}

void prove_layout() {
  constexpr size_t kShards = 64;
  constexpr size_t kCohortSize = 16;
  constexpr size_t kEntries = 4096;
  constexpr size_t kKeys = 262144;
  se::CohortPageAllocator allocator(kShards, kCohortSize, kEntries, 64);
  std::array<uint32_t, 256> page_cohorts{};
  page_cohorts.fill(std::numeric_limits<uint32_t>::max());
  std::unordered_set<uint32_t> indices;
  indices.reserve(kKeys);
  for (uint64_t key = 700000; key < 700000 + kKeys; ++key) {
    const uint32_t shard = splitmix_hash(key) & (kShards - 1);
    const auto allocation = allocator.allocate(shard);
    require(allocation.index == allocation.page_id * kEntries + allocation.slot,
            "cohort allocation index is not page contiguous");
    require(allocation.cohort_id == shard / kCohortSize,
            "cohort allocation chose the wrong cohort");
    require(indices.insert(allocation.index).second,
            "cohort allocation produced a duplicate index");
    uint32_t &page_cohort = page_cohorts[allocation.page_id];
    if (page_cohort == std::numeric_limits<uint32_t>::max())
      page_cohort = allocation.cohort_id;
    require(page_cohort == allocation.cohort_id,
            "cohort allocation mixed cohorts in one page");
    require(allocator.page_cohort(allocation.page_id) == allocation.cohort_id,
            "cohort allocation page ownership changed");
  }
  require(allocator.allocated_page_count() == 66,
          "cohort allocation page count diverges from HP-616 model");
}

void prove_concurrent_unique_indices() {
  constexpr size_t kThreads = 16;
  constexpr size_t kAllocationsPerThread = 1024;
  se::CohortPageAllocator allocator(64, 16, 64, 512);
  std::array<std::vector<uint32_t>, kThreads> indices;
  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (size_t thread = 0; thread < kThreads; ++thread) {
    indices[thread].resize(kAllocationsPerThread);
    threads.emplace_back([&, thread] {
      for (size_t offset = 0; offset < kAllocationsPerThread; ++offset) {
        const uint32_t shard =
            static_cast<uint32_t>((thread * 17 + offset) & 63);
        indices[thread][offset] = allocator.allocate(shard).index;
      }
    });
  }
  for (std::thread &thread : threads)
    thread.join();
  std::unordered_set<uint32_t> unique;
  unique.reserve(kThreads * kAllocationsPerThread);
  for (const auto &thread_indices : indices) {
    for (uint32_t index : thread_indices)
      require(unique.insert(index).second,
              "concurrent cohort allocation produced a duplicate index");
  }
}

void prove_exact_binding() {
  se::CohortPageAllocator allocator(64, 16, 64, 32);
  allocator.bind_exact(2, 3, 7);
  require(allocator.page_cohort(7) == 2,
          "exact cohort page binding lost ownership");
  bool collision_threw = false;
  try {
    allocator.bind_exact(1, 0, 7);
  } catch (const std::logic_error &) {
    collision_threw = true;
  }
  require(collision_threw,
          "exact cohort binding accepted a physical collision");
}

} // namespace

int main() {
  prove_layout();
  prove_concurrent_unique_indices();
  prove_exact_binding();
}
