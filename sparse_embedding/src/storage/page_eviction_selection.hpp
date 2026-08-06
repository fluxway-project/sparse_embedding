#pragma once

#include "sparse_embedding/src/storage/page_storage.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace se::storage_detail {

struct EvictionPageSnapshot {
  std::byte *addr;
  size_t page_idx;
  size_t allocated_size;
};

struct EvictionCandidate {
  std::byte *addr;
  size_t page_idx;
};

struct EvictionSelectionResult {
  std::vector<EvictionCandidate> candidates;
  size_t evaluated_pages;
};

inline EvictionSelectionResult
select_coldest_pages(std::span<const EvictionPageSnapshot> pages,
                     size_t bytes_to_evict, size_t page_size,
                     PageEvictionProxy &proxy) {
  if (!proxy || !proxy->score_enabled())
    throw std::runtime_error("page eviction requires a score evaluator");

  struct ScoredCandidate {
    uint64_t score;
    EvictionCandidate candidate;
    size_t allocated_size;
  };

  std::vector<ScoredCandidate> scored_pages;
  scored_pages.reserve(pages.size());
  for (const auto &page : pages) {
    uint64_t score = 0;
    if (!proxy->evaluate_page(page.page_idx, score))
      continue;
    scored_pages.push_back({score, EvictionCandidate{page.addr, page.page_idx},
                            page.allocated_size});
  }

  auto colder = [](const ScoredCandidate &left, const ScoredCandidate &right) {
    if (left.score != right.score)
      return left.score > right.score;
    return left.candidate.page_idx > right.candidate.page_idx;
  };
  std::make_heap(scored_pages.begin(), scored_pages.end(), colder);

  std::vector<EvictionCandidate> result;
  const size_t required_pages =
      bytes_to_evict / page_size + (bytes_to_evict % page_size != 0);
  result.reserve(std::min(scored_pages.size(), required_pages));
  size_t selected_bytes = 0;
  while (!scored_pages.empty() && selected_bytes < bytes_to_evict) {
    std::pop_heap(scored_pages.begin(), scored_pages.end(), colder);
    ScoredCandidate candidate = scored_pages.back();
    scored_pages.pop_back();
    result.push_back(candidate.candidate);
    if (candidate.allocated_size >= bytes_to_evict - selected_bytes)
      selected_bytes = bytes_to_evict;
    else
      selected_bytes += candidate.allocated_size;
  }

  return {std::move(result), pages.size()};
}

} // namespace se::storage_detail
