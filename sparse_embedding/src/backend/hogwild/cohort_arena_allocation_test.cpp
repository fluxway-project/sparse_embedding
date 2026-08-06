#include <array>
#include <cstdint>
#include <stdexcept>
#include <thread>
#include <unordered_set>
#include <vector>

#include "sparse_embedding/src/backend/hogwild/core/cohort_arena_allocation.hpp"

namespace {

void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

void prove_segment_arithmetic() {
  se::CohortArenaAllocation allocation(64, 16, 4096, 8192);
  for (uint32_t shard = 0; shard < 64; ++shard) {
    const auto entry = allocation.acquire_fresh(shard);
    require(entry.index == entry.page_id * 4096 + entry.slot,
            "cohort allocation index arithmetic changed");
    require(entry.page_id == shard / 16,
            "cohort allocation chose an invalid initial segment page");
    require(entry.slot == shard % 16,
            "cohort allocation initial slot is not cohort contiguous");
  }
}

void prove_exact_binding() {
  se::CohortArenaAllocation allocation(64, 16, 64, 128);
  constexpr uint32_t kPage = 6;
  constexpr uint32_t kIndex = kPage * 64 + 17;
  allocation.bind_exact(kIndex, 33);
  require(allocation.page_cohort(kPage) == 2,
          "exact cohort binding lost page ownership");
  const auto entry = allocation.acquire_fresh(33);
  require(entry.page_id == 10 && entry.slot == 0,
          "exact cohort binding did not reserve its segment round");
  bool conflict_threw = false;
  try {
    allocation.bind_exact(kIndex, 1);
  } catch (const std::logic_error &) {
    conflict_threw = true;
  }
  require(conflict_threw, "exact cohort binding accepted a conflict");
}

void prove_contiguous_range_exclusion() {
  se::CohortArenaAllocation allocation(64, 16, 64, 128);
  allocation.reserve_contiguous_range(4 * 64, 5 * 64);
  for (size_t allocation_count = 0; allocation_count < 64; ++allocation_count)
    allocation.acquire_fresh(0);
  const auto next = allocation.acquire_fresh(0);
  require(next.page_id == 8 && next.slot == 0,
          "cohort allocation reused a contiguous-range page");
}

void prove_concurrent_uniqueness() {
  constexpr size_t kThreads = 16;
  constexpr size_t kAllocationsPerThread = 1024;
  se::CohortArenaAllocation allocation(64, 16, 64, 8192);
  std::array<std::vector<uint32_t>, kThreads> indices;
  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (size_t thread = 0; thread < kThreads; ++thread) {
    indices[thread].resize(kAllocationsPerThread);
    threads.emplace_back([&, thread] {
      for (size_t offset = 0; offset < kAllocationsPerThread; ++offset) {
        indices[thread][offset] =
            allocation
                .acquire_fresh(static_cast<uint32_t>((thread + offset) & 63))
                .index;
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
              "cohort allocation produced a duplicate index");
  }
}

void prove_recycled_page_remains_contiguous() {
  se::CohortArenaAllocation allocation(16, 4, 64, 128);
  const auto original = allocation.acquire_fresh(0);
  for (size_t slot = 1; slot < 64; ++slot)
    allocation.acquire_fresh(0);
  allocation.release_page(original.page_id);
  for (uint32_t slot = 0; slot < 64; ++slot) {
    const auto recycled = allocation.acquire_fresh(0);
    require(recycled.page_id == original.page_id && recycled.slot == slot,
            "recycled page did not preserve contiguous slot order");
  }
}

} // namespace

int main() {
  prove_segment_arithmetic();
  prove_exact_binding();
  prove_contiguous_range_exclusion();
  prove_concurrent_uniqueness();
  prove_recycled_page_remains_contiguous();
}
