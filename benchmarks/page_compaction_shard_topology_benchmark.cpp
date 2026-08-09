#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "sparse_embedding/src/backend/hogwild/hogwild_embedding_store.hpp"
#include "sparse_embedding/src/storage/page_storage.hpp"

namespace {

constexpr size_t kDimension = 128;
constexpr size_t kShardCount = 64;
constexpr size_t kPageCount = 64;
constexpr size_t kPagesPerWorker = 4;
constexpr size_t kWorkerCount = kPageCount / kPagesPerWorker;
constexpr size_t kKeysPerPage = se::EmbeddingArena<float>::kEntriesPerPage;
constexpr size_t kKeyCount = kPageCount * kKeysPerPage;

using Store =
    se::HogwildEmbeddingStore<uint64_t, float, float, se::NoScorePolicy>;

uint64_t splitmix_hash(uint64_t key) {
  key ^= key >> 30;
  key *= 0xbf58476d1ce4e5b9ULL;
  key ^= key >> 27;
  key *= 0x94d049bb133111ebULL;
  return key ^ (key >> 31);
}

size_t shard_for_key(uint64_t key) {
  return splitmix_hash(key) & (kShardCount - 1);
}

int run() {
  se::StorageOptions options;
  options.mode = se::StorageMode::kMemoryOnly;
  options.memory_limit_bytes = size_t{1} << 30;
  auto storage = se::create_page_storage(std::move(options));
  Store store({kDimension}, {}, nullptr, se::NoScorePolicy{},
              se::HogwildStoreOptions{.thread_budget = kShardCount,
                                      .storage = storage});

  std::vector<uint64_t> keys(kKeyCount);
  std::iota(keys.begin(), keys.end(), uint64_t{700000});
  std::vector<int64_t> lengths(kKeyCount, kDimension);
  std::vector<float> values(kKeyCount * kDimension, 1.0F);
  store.lookup_or_insert<uint64_t>(keys, values, lengths);

  std::array<uint64_t, kPageCount> page_masks{};
  std::array<uint8_t, kPageCount> seen_pages{};
  size_t exported_keys = 0;
  store.export_page_images([&](size_t dim,
                               se::PageImageExport<float, float> &&image) {
    if (dim != kDimension || image.physical_page_id >= kPageCount)
      throw std::runtime_error("topology benchmark exported an invalid page");
    const size_t page_id = static_cast<size_t>(image.physical_page_id);
    if (seen_pages[page_id] != 0)
      throw std::runtime_error("topology benchmark exported a duplicate page");
    uint64_t mask = 0;
    for (uint64_t key : image.keys)
      mask |= uint64_t{1} << shard_for_key(key);
    if (mask == 0 || image.keys.size() != kKeysPerPage)
      throw std::runtime_error(
          "topology benchmark exported an incomplete page");
    page_masks[page_id] = mask;
    seen_pages[page_id] = 1;
    exported_keys += image.keys.size();
    return true;
  });
  if (exported_keys != kKeyCount ||
      !std::ranges::all_of(seen_pages, [](uint8_t seen) { return seen == 1; }))
    throw std::runtime_error(
        "topology benchmark did not export the expected layout");

  std::array<uint64_t, kWorkerCount> worker_masks{};
  size_t page_coverage_total = 0;
  size_t page_coverage_min = kShardCount;
  size_t page_coverage_max = 0;
  for (size_t page = 0; page < kPageCount; ++page) {
    const size_t coverage = std::popcount(page_masks[page]);
    page_coverage_total += coverage;
    page_coverage_min = std::min(page_coverage_min, coverage);
    page_coverage_max = std::max(page_coverage_max, coverage);
    worker_masks[page / kPagesPerWorker] |= page_masks[page];
  }

  size_t worker_coverage_total = 0;
  size_t worker_coverage_min = kShardCount;
  size_t worker_coverage_max = 0;
  for (uint64_t mask : worker_masks) {
    const size_t coverage = std::popcount(mask);
    worker_coverage_total += coverage;
    worker_coverage_min = std::min(worker_coverage_min, coverage);
    worker_coverage_max = std::max(worker_coverage_max, coverage);
  }

  size_t shared_shards_total = 0;
  size_t full_overlap_pairs = 0;
  size_t pair_count = 0;
  for (size_t left = 0; left < kWorkerCount; ++left) {
    for (size_t right = left + 1; right < kWorkerCount; ++right) {
      const size_t shared =
          std::popcount(worker_masks[left] & worker_masks[right]);
      shared_shards_total += shared;
      full_overlap_pairs += shared == kShardCount;
      ++pair_count;
    }
  }

  std::cout << "page_shard_topology pages=" << kPageCount
            << " keys=" << kKeyCount
            << " page_coverage_min=" << page_coverage_min
            << " page_coverage_mean="
            << static_cast<double>(page_coverage_total) / kPageCount
            << " page_coverage_max=" << page_coverage_max
            << " worker_coverage_min=" << worker_coverage_min
            << " worker_coverage_mean="
            << static_cast<double>(worker_coverage_total) / kWorkerCount
            << " worker_coverage_max=" << worker_coverage_max
            << " pair_shared_shards_mean="
            << static_cast<double>(shared_shards_total) / pair_count
            << " full_overlap_pairs=" << full_overlap_pairs
            << " pair_count=" << pair_count << '\n';
  return 0;
}

} // namespace

int main() { return run(); }
