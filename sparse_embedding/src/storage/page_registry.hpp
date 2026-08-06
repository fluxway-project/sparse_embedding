#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <random>
#include <unordered_map>
#include <vector>

#include "sparse_embedding/src/storage/page_eviction_selection.hpp"
#include "sparse_embedding/src/storage/page_storage.hpp"

namespace se::storage_detail {

enum class PageBacking {
  kHeap,
  kAnonymousMapping,
  kSwapClaimed,
  kSpillMapping,
};

struct PageRecord {
  std::byte *addr;
  size_t allocated_size;
  size_t offset;
  size_t page_idx;
  PageBacking backing;
  bool evaluator_registered;
};

class PageRegistry {
public:
  using EvictionPageSnapshot = storage_detail::EvictionPageSnapshot;

  size_t allocate_page_index();
  void insert(PageRecord record);
  PageRecord erase(std::byte *address);
  PageRecord *find(std::byte *address) noexcept;
  const PageRecord *find(std::byte *address) const noexcept;

  void register_evictable(EvictionPageSnapshot page);
  void unregister_evictable(size_t page_index) noexcept;
  std::vector<EvictionPageSnapshot> snapshot_evictable_pages();

  void register_evaluator(size_t page_index, PageEvaluator evaluator);
  void unregister_evaluator(size_t page_index) noexcept;
  const std::unordered_map<size_t, PageEvaluator> &evaluators() const noexcept;

private:
  static constexpr size_t kExactEvictionPageThreshold = 1024;
  static constexpr size_t kSampledEvictionPageCount = 256;
  static constexpr uint64_t kEvictionRandomSeed = 0x9e3779b97f4a7c15ULL;

  std::vector<PageRecord> records_;
  std::unordered_map<std::byte *, size_t> positions_;
  std::vector<EvictionPageSnapshot> evictable_pages_;
  std::unordered_map<size_t, size_t> evictable_positions_;
  std::unordered_map<size_t, PageEvaluator> evaluators_;
  std::atomic<size_t> next_page_index_{0};
  std::mt19937_64 sampling_rng_{kEvictionRandomSeed};
};

} // namespace se::storage_detail
