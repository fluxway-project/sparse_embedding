#include "sparse_embedding/src/storage/page_registry.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace se::storage_detail {

size_t PageRegistry::allocate_page_index() {
  size_t current = next_page_index_.load(std::memory_order_relaxed);
  while (true) {
    if (current == std::numeric_limits<size_t>::max())
      throw std::overflow_error("storage page index exhausted");
    if (next_page_index_.compare_exchange_weak(current, current + 1,
                                               std::memory_order_relaxed,
                                               std::memory_order_relaxed)) {
      return current;
    }
  }
}

void PageRegistry::insert(PageRecord record) {
  records_.push_back(record);
  try {
    const auto [_, inserted] =
        positions_.emplace(record.addr, records_.size() - 1);
    if (!inserted)
      throw std::logic_error("duplicate storage page address");
  } catch (...) {
    records_.pop_back();
    throw;
  }
}

PageRecord PageRegistry::erase(std::byte *address) {
  const auto position = positions_.find(address);
  if (position == positions_.end())
    throw std::logic_error("storage page address was not registered");
  const size_t record_position = position->second;
  const PageRecord removed = records_[record_position];
  positions_.erase(position);
  const size_t last_position = records_.size() - 1;
  if (record_position != last_position) {
    records_[record_position] = records_[last_position];
    const auto moved = positions_.find(records_[record_position].addr);
    if (moved == positions_.end())
      throw std::logic_error("moved storage page address was not indexed");
    moved->second = record_position;
  }
  records_.pop_back();
  return removed;
}

PageRecord *PageRegistry::find(std::byte *address) noexcept {
  const auto position = positions_.find(address);
  return position == positions_.end() ? nullptr : &records_[position->second];
}

const PageRecord *PageRegistry::find(std::byte *address) const noexcept {
  const auto position = positions_.find(address);
  return position == positions_.end() ? nullptr : &records_[position->second];
}

void PageRegistry::register_evictable(EvictionPageSnapshot page) {
  const size_t position = evictable_pages_.size();
  evictable_pages_.push_back(page);
  try {
    const auto [_, inserted] =
        evictable_positions_.emplace(page.page_idx, position);
    if (!inserted)
      throw std::logic_error("duplicate evictable page index");
  } catch (...) {
    evictable_pages_.pop_back();
    throw;
  }
}

void PageRegistry::unregister_evictable(size_t page_index) noexcept {
  const auto position = evictable_positions_.find(page_index);
  if (position == evictable_positions_.end())
    std::terminate();
  const size_t removed_position = position->second;
  const size_t last_position = evictable_pages_.size() - 1;
  if (removed_position != last_position) {
    evictable_pages_[removed_position] = evictable_pages_[last_position];
    const auto moved =
        evictable_positions_.find(evictable_pages_[removed_position].page_idx);
    if (moved == evictable_positions_.end())
      std::terminate();
    moved->second = removed_position;
  }
  evictable_pages_.pop_back();
  evictable_positions_.erase(position);
}

std::vector<PageRegistry::EvictionPageSnapshot>
PageRegistry::snapshot_evictable_pages() {
  if (evictable_pages_.size() <= kExactEvictionPageThreshold)
    return evictable_pages_;
  const size_t sampled_count =
      std::min(kSampledEvictionPageCount, evictable_pages_.size());
  std::unordered_set<size_t> sampled_positions;
  sampled_positions.reserve(sampled_count * 2);
  for (size_t position = evictable_pages_.size() - sampled_count;
       position < evictable_pages_.size(); ++position) {
    std::uniform_int_distribution<size_t> distribution(0, position);
    const size_t sampled_position = distribution(sampling_rng_);
    if (!sampled_positions.insert(sampled_position).second)
      sampled_positions.insert(position);
  }
  std::vector<EvictionPageSnapshot> snapshot;
  snapshot.reserve(sampled_count);
  for (size_t position : sampled_positions)
    snapshot.push_back(evictable_pages_[position]);
  return snapshot;
}

void PageRegistry::register_evaluator(size_t page_index,
                                      PageEvaluator evaluator) {
  if (!evaluator)
    throw std::invalid_argument("page evaluator must not be empty");
  const auto [_, inserted] =
      evaluators_.emplace(page_index, std::move(evaluator));
  if (!inserted)
    throw std::logic_error("duplicate page evaluator index");
}

void PageRegistry::unregister_evaluator(size_t page_index) noexcept {
  evaluators_.erase(page_index);
}

const std::unordered_map<size_t, PageEvaluator> &
PageRegistry::evaluators() const noexcept {
  return evaluators_;
}

} // namespace se::storage_detail
