#pragma once

#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "sparse_embedding/src/storage/topology/storage_topology.hpp"

// =============================================================================
// CohortArenaAllocation: Group-partitioned lock-free physical slot allocator.
//
// A "cohort" represents an autonomous group of directory shards sharing a local
// lock-free ticket dispenser and physical memory locality (typically 1:1 with a
// NUMA node / CPU socket cohort).
//
// Why "Cohort" instead of "Numa":
// 1. Concurrency isolation: Cohort partitioning alleviates global atomic ticket
//    contention across threads even on single-node (UMA) architectures.
// 2. Hardware decoupling: The allocator operates on pure geometric mathematics
//    without hard dependencies on libnuma. Physical NUMA memory binding is
//    applied orthogonally at the page level.
// 3. Mathematical mapping:
//      cohort_id = shard_id / cohort_size
//      page_id   = (ticket >> entries_shift) * cohort_count + cohort_id
//      index     = page_id * entries_per_page + (ticket & slot_mask)
// =============================================================================

namespace se {

class CohortArenaAllocation final {
public:
  struct Allocation {
    uint32_t index;
    uint32_t page_id;
    uint32_t slot;
  };

  CohortArenaAllocation(size_t shard_count, size_t cohort_size,
                        size_t entries_per_page, size_t maximum_page_count)
      : shard_count_(checked_power_of_two(shard_count, "shard count")),
        cohort_size_(checked_power_of_two(cohort_size, "cohort size")),
        entries_per_page_(
            checked_power_of_two(entries_per_page, "entries per page")),
        entries_shift_(std::countr_zero(entries_per_page_)),
        slot_mask_(entries_per_page_ - 1),
        cohort_count_(shard_count_ / cohort_size_),
        maximum_page_count_(
            checked_positive(maximum_page_count, "maximum page count")),
        pages_per_cohort_(maximum_page_count_ / cohort_count_),
        tickets_(std::make_unique<CohortTicket[]>(cohort_count_)),
        recycled_pages_(std::make_unique<RecycledPages[]>(cohort_count_)),
        page_cohorts_(
            std::make_unique<std::atomic<uint32_t>[]>(maximum_page_count_)) {
    if (cohort_size_ > shard_count_ || shard_count_ % cohort_size_ != 0)
      throw std::invalid_argument("cohort size does not divide shard count");
    if (maximum_page_count_ % cohort_count_ != 0)
      throw std::invalid_argument(
          "maximum page count does not divide cohort count");
    if (entries_per_page_ > std::numeric_limits<uint32_t>::max())
      throw std::invalid_argument("entries per page exceeds uint32 range");
    if (maximum_page_count_ >
        (static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) + 1) /
            entries_per_page_) {
      throw std::invalid_argument(
          "cohort allocation index range overflows uint32");
    }
    for (size_t page = 0; page < maximum_page_count_; ++page)
      page_cohorts_[page].store(kUnassigned, std::memory_order_relaxed);
  }

  explicit CohortArenaAllocation(const storage::StorageTopology &topology)
      : CohortArenaAllocation(topology.shard_count(), topology.cohort_size(),
                              topology.entries_per_page(),
                              topology.max_pages()) {}

  CohortArenaAllocation(const CohortArenaAllocation &) = delete;
  CohortArenaAllocation &operator=(const CohortArenaAllocation &) = delete;

  Allocation acquire_fresh(uint32_t shard_id) {
    if (shard_id >= shard_count_)
      throw std::out_of_range("cohort allocation shard id is out of range");
    const uint32_t cohort_id = static_cast<uint32_t>(shard_id / cohort_size_);
    if (const auto recycled = take_recycled_slot(cohort_id);
        recycled.has_value()) {
      const uint64_t index =
          static_cast<uint64_t>(recycled->page_id) * entries_per_page_ +
          recycled->slot;
      return {static_cast<uint32_t>(index), recycled->page_id, recycled->slot};
    }
    while (true) {
      const uint64_t ticket =
          tickets_[cohort_id].value.fetch_add(1, std::memory_order_relaxed);
      if ((ticket >> entries_shift_) >= pages_per_cohort_)
        throw std::length_error("cohort allocation page capacity exceeded");
      const uint64_t page_id =
          (ticket >> entries_shift_) * cohort_count_ + cohort_id;
      const uint64_t index =
          page_id * entries_per_page_ + (ticket & slot_mask_);
      if (index > std::numeric_limits<uint32_t>::max())
        throw std::overflow_error("cohort allocation index overflows uint32");
      if (!claim_page(static_cast<uint32_t>(page_id), cohort_id))
        continue;
      return {static_cast<uint32_t>(index), static_cast<uint32_t>(page_id),
              static_cast<uint32_t>(ticket & slot_mask_)};
    }
  }

  template <typename Accept, typename Visit>
  void acquire_fresh_batch(uint32_t shard_id, size_t count, Accept &&accept,
                           Visit &&visit) {
    if (shard_id >= shard_count_)
      throw std::out_of_range("cohort allocation shard id is out of range");
    if (count == 0)
      throw std::invalid_argument("cohort allocation batch is empty");
    const uint32_t cohort_id = static_cast<uint32_t>(shard_id / cohort_size_);
    size_t produced = 0;
    while (produced < count) {
      uint64_t ticket = 0;
      size_t reserved = 0;
      reserve_ticket_range(cohort_id, count - produced, ticket, reserved);
      for (size_t offset = 0; offset < reserved; ++offset) {
        const uint64_t candidate_ticket = ticket + offset;
        const uint32_t page_id = static_cast<uint32_t>(
            (candidate_ticket >> entries_shift_) * cohort_count_ + cohort_id);
        if (!claim_page(page_id, cohort_id))
          continue;
        const uint32_t slot =
            static_cast<uint32_t>(candidate_ticket & slot_mask_);
        const uint64_t index =
            static_cast<uint64_t>(page_id) * entries_per_page_ + slot;
        if (index > std::numeric_limits<uint32_t>::max())
          throw std::overflow_error("cohort allocation index overflows uint32");
        Allocation allocation{static_cast<uint32_t>(index), page_id, slot};
        if (!accept(allocation))
          continue;
        visit(allocation);
        ++produced;
        if (produced == count)
          break;
      }
    }
  }

  void acquire_fresh_batch(uint32_t shard_id, std::span<Allocation> output) {
    size_t position = 0;
    acquire_fresh_batch(
        shard_id, output.size(), [](const Allocation &) { return true; },
        [&](const Allocation &allocation) { output[position++] = allocation; });
  }

  void reserve_contiguous_range(uint32_t first, uint32_t end) {
    if (first >= end)
      throw std::invalid_argument("cohort reserved range is empty");
    if (end > maximum_page_count_ * entries_per_page_)
      throw std::out_of_range("cohort reserved range exceeds arena capacity");
    const uint32_t first_page = static_cast<uint32_t>(first >> entries_shift_);
    const uint32_t last_page =
        static_cast<uint32_t>((end - 1) >> entries_shift_);
    for (uint32_t page_id = first_page; page_id <= last_page; ++page_id)
      reserve_page(page_id);
  }

  void bind_exact(uint32_t index, uint32_t shard_id) {
    if (shard_id >= shard_count_)
      throw std::out_of_range("cohort exact binding shard id is out of range");
    const uint32_t page_id = static_cast<uint32_t>(index >> entries_shift_);
    if (page_id >= maximum_page_count_)
      throw std::out_of_range("cohort exact binding page id is out of range");
    const uint32_t cohort_id = static_cast<uint32_t>(shard_id / cohort_size_);
    if (page_id % cohort_count_ != cohort_id) {
      throw std::logic_error(
          "cohort exact binding page does not match shard cohort");
    }
    if (!claim_page(page_id, cohort_id))
      throw std::logic_error("cohort exact binding page is range reserved");
    const uint64_t required_ticket =
        ((static_cast<uint64_t>(page_id / cohort_count_) + 1)
         << entries_shift_);
    advance_ticket(tickets_[cohort_id].value, required_ticket);
  }

  uint32_t page_cohort(uint32_t page_id) const {
    if (page_id >= maximum_page_count_)
      throw std::out_of_range("cohort allocation page id is out of range");
    const uint32_t cohort =
        page_cohorts_[page_id].load(std::memory_order_acquire);
    if (cohort == kUnassigned)
      throw std::logic_error("cohort allocation page is unbound");
    return cohort;
  }

  void release_page(uint32_t page_id) noexcept {
    if (page_id >= maximum_page_count_)
      std::terminate();
    const uint32_t cohort_id =
        page_cohorts_[page_id].load(std::memory_order_acquire);
    if (cohort_id >= cohort_count_)
      std::terminate();
    {
      std::lock_guard lock(recycled_pages_[cohort_id].mutex);
      recycled_pages_[cohort_id].pages.push_back({page_id, 0});
    }
  }

  void reset() noexcept {
    for (size_t cohort = 0; cohort < cohort_count_; ++cohort)
      tickets_[cohort].value.store(0, std::memory_order_relaxed);
    for (size_t page = 0; page < maximum_page_count_; ++page)
      page_cohorts_[page].store(kUnassigned, std::memory_order_relaxed);
    for (size_t cohort = 0; cohort < cohort_count_; ++cohort) {
      std::lock_guard lock(recycled_pages_[cohort].mutex);
      recycled_pages_[cohort].pages.clear();
    }
  }

private:
  static constexpr uint32_t kUnassigned = std::numeric_limits<uint32_t>::max();
  static constexpr uint32_t kReserved = kUnassigned - 1;

  struct alignas(64) CohortTicket {
    std::atomic<uint64_t> value{0};
  };

  struct RecycledSlot {
    uint32_t page_id;
    uint32_t slot;
  };

  struct RecycledPage {
    uint32_t page_id;
    uint32_t next_slot;
  };

  struct RecycledPages {
    std::mutex mutex;
    std::vector<RecycledPage> pages;
  };

  static size_t checked_positive(size_t value, const char *name) {
    if (value == 0)
      throw std::invalid_argument(std::string(name) + " is zero");
    return value;
  }

  static size_t checked_power_of_two(size_t value, const char *name) {
    checked_positive(value, name);
    if (!std::has_single_bit(value))
      throw std::invalid_argument(std::string(name) + " is not a power of two");
    return value;
  }

  bool claim_page(uint32_t page_id, uint32_t cohort_id) {
    uint32_t expected = page_cohorts_[page_id].load(std::memory_order_acquire);
    if (expected == cohort_id)
      return true;
    if (expected == kReserved)
      return false;
    if (expected != kUnassigned)
      throw std::logic_error(
          "cohort allocation physical page has another cohort");
    if (page_cohorts_[page_id].compare_exchange_strong(
            expected, cohort_id, std::memory_order_release,
            std::memory_order_acquire)) {
      return true;
    }
    if (expected == kReserved)
      return false;
    if (expected != cohort_id)
      throw std::logic_error(
          "cohort allocation physical page has another cohort");
    return true;
  }

  std::optional<RecycledSlot> take_recycled_slot(uint32_t cohort_id) {
    RecycledPages &recycled = recycled_pages_[cohort_id];
    std::lock_guard lock(recycled.mutex);
    if (recycled.pages.empty())
      return std::nullopt;
    RecycledPage &page = recycled.pages.back();
    if (page.next_slot >= entries_per_page_)
      std::terminate();
    const RecycledSlot result{page.page_id, page.next_slot++};
    if (page.next_slot == entries_per_page_)
      recycled.pages.pop_back();
    return result;
  }

  void reserve_page(uint32_t page_id) {
    uint32_t expected = kUnassigned;
    if (page_cohorts_[page_id].compare_exchange_strong(
            expected, kReserved, std::memory_order_release,
            std::memory_order_acquire)) {
      return;
    }
    if (expected != kReserved)
      throw std::logic_error("cohort reserved range overlaps a cohort page");
  }

  static void advance_ticket(std::atomic<uint64_t> &ticket,
                             uint64_t required_ticket) {
    uint64_t observed = ticket.load(std::memory_order_acquire);
    while (observed < required_ticket &&
           !ticket.compare_exchange_weak(observed, required_ticket,
                                         std::memory_order_acq_rel,
                                         std::memory_order_acquire)) {
    }
  }

  void reserve_ticket_range(uint32_t cohort_id, size_t requested,
                            uint64_t &first, size_t &count) {
    const uint64_t limit = static_cast<uint64_t>(pages_per_cohort_)
                           << entries_shift_;
    uint64_t observed =
        tickets_[cohort_id].value.load(std::memory_order_acquire);
    while (true) {
      if (observed >= limit)
        throw std::length_error("cohort allocation page capacity exceeded");
      const uint64_t available = limit - observed;
      const size_t batch = static_cast<size_t>(
          std::min<uint64_t>(available, static_cast<uint64_t>(requested)));
      if (tickets_[cohort_id].value.compare_exchange_weak(
              observed, observed + batch, std::memory_order_acq_rel,
              std::memory_order_acquire)) {
        first = observed;
        count = batch;
        return;
      }
    }
  }

  size_t shard_count_;
  size_t cohort_size_;
  size_t entries_per_page_;
  unsigned entries_shift_;
  size_t slot_mask_;
  size_t cohort_count_;
  size_t maximum_page_count_;
  size_t pages_per_cohort_;
  std::unique_ptr<CohortTicket[]> tickets_;
  std::unique_ptr<RecycledPages[]> recycled_pages_;
  std::unique_ptr<std::atomic<uint32_t>[]> page_cohorts_;
};

} // namespace se
