#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

namespace se {

class CohortPageAllocator final {
public:
  struct Allocation {
    uint32_t index;
    uint32_t page_id;
    uint32_t slot;
    uint32_t cohort_id;
  };

  CohortPageAllocator(size_t shard_count, size_t cohort_size,
                      size_t entries_per_page,
                      size_t maximum_logical_pages_per_cohort)
      : shard_count_(checked_power_of_two(shard_count, "shard count")),
        cohort_size_(checked_power_of_two(cohort_size, "cohort size")),
        entries_per_page_(
            checked_power_of_two(entries_per_page, "entries per page")),
        cohort_count_(shard_count_ / cohort_size_),
        maximum_logical_pages_per_cohort_(
            checked_positive(maximum_logical_pages_per_cohort,
                             "maximum logical pages per cohort")),
        maximum_physical_pages_(
            checked_product(cohort_count_, maximum_logical_pages_per_cohort_)),
        cohorts_(std::make_unique<CohortState[]>(cohort_count_)),
        page_ids_(
            std::make_unique<std::atomic<uint32_t>[]>(maximum_physical_pages_)),
        page_cohorts_(std::make_unique<std::atomic<uint32_t>[]>(
            maximum_physical_pages_)) {
    if (cohort_size_ > shard_count_ || shard_count_ % cohort_size_ != 0)
      throw std::invalid_argument("cohort size does not divide shard count");
    if (entries_per_page_ > std::numeric_limits<uint32_t>::max())
      throw std::invalid_argument("entries per page exceeds uint32 range");
    for (size_t index = 0; index < maximum_physical_pages_; ++index) {
      page_ids_[index].store(kUnassigned, std::memory_order_relaxed);
      page_cohorts_[index].store(kUnassigned, std::memory_order_relaxed);
    }
  }

  CohortPageAllocator(const CohortPageAllocator &) = delete;
  CohortPageAllocator &operator=(const CohortPageAllocator &) = delete;

  Allocation allocate(uint32_t shard_id) {
    if (shard_id >= shard_count_)
      throw std::out_of_range("cohort allocator shard id is out of range");
    const uint32_t cohort_id = static_cast<uint32_t>(shard_id / cohort_size_);
    CohortState &cohort = cohorts_[cohort_id];
    const uint64_t ticket =
        cohort.next_slot.fetch_add(1, std::memory_order_relaxed);
    const uint64_t logical_page = ticket / entries_per_page_;
    if (logical_page >= maximum_logical_pages_per_cohort_)
      throw std::length_error(
          "cohort allocator logical page capacity exceeded");
    const uint32_t slot = static_cast<uint32_t>(ticket % entries_per_page_);
    std::atomic<uint32_t> &page_id =
        page_ids_[cohort_id * maximum_logical_pages_per_cohort_ + logical_page];
    const uint32_t physical_page = materialize_page(page_id, cohort_id);
    const uint64_t index =
        static_cast<uint64_t>(physical_page) * entries_per_page_ + slot;
    if (index > std::numeric_limits<uint32_t>::max())
      throw std::overflow_error("cohort allocator index overflows uint32");
    return {static_cast<uint32_t>(index), physical_page, slot, cohort_id};
  }

  void bind_exact(uint32_t cohort_id, uint32_t logical_page,
                  uint32_t physical_page) {
    if (cohort_id >= cohort_count_ ||
        logical_page >= maximum_logical_pages_per_cohort_ ||
        physical_page >= maximum_physical_pages_) {
      throw std::out_of_range("cohort allocator exact page is out of range");
    }
    std::atomic<uint32_t> &page_id =
        page_ids_[cohort_id * maximum_logical_pages_per_cohort_ + logical_page];
    uint32_t expected = kUnassigned;
    if (!page_id.compare_exchange_strong(expected, kAllocating,
                                         std::memory_order_acq_rel,
                                         std::memory_order_acquire)) {
      throw std::logic_error("cohort allocator logical page is already bound");
    }
    std::atomic<uint32_t> &owner = page_cohorts_[physical_page];
    expected = kUnassigned;
    if (!owner.compare_exchange_strong(expected, cohort_id,
                                       std::memory_order_acq_rel,
                                       std::memory_order_acquire)) {
      page_id.store(kUnassigned, std::memory_order_release);
      page_id.notify_all();
      throw std::logic_error("cohort allocator physical page is already bound");
    }
    page_id.store(physical_page, std::memory_order_release);
    page_id.notify_all();
    advance_page_count(physical_page + 1);
  }

  uint32_t page_cohort(uint32_t physical_page) const {
    if (physical_page >= maximum_physical_pages_)
      throw std::out_of_range("cohort allocator page id is out of range");
    const uint32_t cohort =
        page_cohorts_[physical_page].load(std::memory_order_acquire);
    if (cohort == kUnassigned)
      throw std::logic_error("cohort allocator page is unbound");
    return cohort;
  }

  uint32_t allocated_page_count() const {
    return next_physical_page_.load(std::memory_order_acquire);
  }

private:
  static constexpr uint32_t kUnassigned = std::numeric_limits<uint32_t>::max();
  static constexpr uint32_t kAllocating = kUnassigned - 1;

  struct alignas(64) CohortState {
    std::atomic<uint64_t> next_slot{0};
  };

  static size_t checked_positive(size_t value, const char *name) {
    if (value == 0)
      throw std::invalid_argument(std::string(name) + " is zero");
    return value;
  }

  static size_t checked_power_of_two(size_t value, const char *name) {
    checked_positive(value, name);
    if ((value & (value - 1)) != 0)
      throw std::invalid_argument(std::string(name) + " is not a power of two");
    return value;
  }

  static size_t checked_product(size_t left, size_t right) {
    if (left > std::numeric_limits<size_t>::max() / right)
      throw std::overflow_error("cohort allocator page capacity overflow");
    return left * right;
  }

  uint32_t materialize_page(std::atomic<uint32_t> &page_id,
                            uint32_t cohort_id) {
    uint32_t physical_page = page_id.load(std::memory_order_acquire);
    if (physical_page == kUnassigned) {
      uint32_t expected = kUnassigned;
      if (page_id.compare_exchange_strong(expected, kAllocating,
                                          std::memory_order_acq_rel,
                                          std::memory_order_acquire)) {
        physical_page =
            next_physical_page_.fetch_add(1, std::memory_order_acq_rel);
        if (physical_page >= maximum_physical_pages_)
          throw std::length_error(
              "cohort allocator physical page capacity exceeded");
        uint32_t unassigned = kUnassigned;
        if (!page_cohorts_[physical_page].compare_exchange_strong(
                unassigned, cohort_id, std::memory_order_release,
                std::memory_order_acquire)) {
          throw std::logic_error("cohort allocator physical page was prebound");
        }
        page_id.store(physical_page, std::memory_order_release);
        page_id.notify_all();
        return physical_page;
      }
      physical_page = expected;
    }
    while (physical_page == kAllocating) {
      page_id.wait(kAllocating, std::memory_order_acquire);
      physical_page = page_id.load(std::memory_order_acquire);
    }
    if (physical_page == kUnassigned)
      throw std::logic_error("cohort allocator page publication was cancelled");
    return physical_page;
  }

  void advance_page_count(uint32_t required) {
    uint32_t current = next_physical_page_.load(std::memory_order_acquire);
    while (current < required &&
           !next_physical_page_.compare_exchange_weak(
               current, required, std::memory_order_acq_rel,
               std::memory_order_acquire)) {
    }
  }

  size_t shard_count_;
  size_t cohort_size_;
  size_t entries_per_page_;
  size_t cohort_count_;
  size_t maximum_logical_pages_per_cohort_;
  size_t maximum_physical_pages_;
  std::unique_ptr<CohortState[]> cohorts_;
  std::unique_ptr<std::atomic<uint32_t>[]> page_ids_;
  std::unique_ptr<std::atomic<uint32_t>[]> page_cohorts_;
  std::atomic<uint32_t> next_physical_page_{0};
};

} // namespace se
