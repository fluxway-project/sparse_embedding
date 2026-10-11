#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace se::storage {

enum class StorageWorkload : uint8_t {
  kForwardLookup,
  kLookupInsert,
  kCompaction,
  kTraining,
};

struct StorageTuningOptions {
  size_t target_capacity_keys = 0;
  size_t memory_budget_bytes = 0;
  size_t entries_per_page = 0;
  size_t max_pages = 0;
  StorageWorkload workload = StorageWorkload::kForwardLookup;
  bool calibrate_page_geometry = true;
  size_t calibration_samples = 3;
};

struct PageGeometry {
  static constexpr size_t kSlotAlignment = 64;

  static size_t aligned_slot_stride(size_t row_bytes) {
    if (row_bytes == 0 ||
        row_bytes > std::numeric_limits<size_t>::max() - (kSlotAlignment - 1))
      throw std::overflow_error("page slot stride is out of range");
    return (row_bytes + kSlotAlignment - 1) & ~(kSlotAlignment - 1);
  }

  explicit PageGeometry(size_t entries_per_page)
      : entries_per_page_(validate(entries_per_page)),
        entries_shift_(
            static_cast<unsigned>(std::countr_zero(entries_per_page_))),
        slot_mask_(static_cast<uint32_t>(entries_per_page_ - 1)) {}

  [[nodiscard]] size_t entries_per_page() const noexcept {
    return entries_per_page_;
  }
  [[nodiscard]] unsigned shift() const noexcept { return entries_shift_; }
  [[nodiscard]] uint32_t slot_mask() const noexcept { return slot_mask_; }
  [[nodiscard]] uint32_t page_id(uint32_t index) const noexcept {
    if (entries_shift_ == 12)
      return index >> 12;
    return index >> entries_shift_;
  }
  [[nodiscard]] uint32_t slot(uint32_t index) const noexcept {
    if (entries_shift_ == 12)
      return index & ((uint32_t{1} << 12) - 1);
    return index & slot_mask_;
  }

private:
  static size_t validate(size_t value) {
    if (value == 0 || !std::has_single_bit(value))
      throw std::invalid_argument("entries per page must be a power of two");
    if (value > std::numeric_limits<uint32_t>::max())
      throw std::overflow_error("entries per page exceeds uint32");
    return value;
  }

  const size_t entries_per_page_;
  const unsigned entries_shift_;
  const uint32_t slot_mask_;
};

class StorageTopology {
public:
  // cohort_size defines the number of shards assigned to each autonomous
  // ticket/allocation group (cohort), aligning with NUMA nodes or concurrency
  // partitioning domains to eliminate cross-thread contention.
  StorageTopology(size_t execution_workers, size_t shard_count,
                  size_t entries_per_page, size_t max_pages, size_t cohort_size,
                  bool numa_execution, bool numa_memory,
                  size_t target_capacity_keys = 0)
      : execution_workers_(execution_workers), shard_count_(shard_count),
        shard_mask_(checked_mask(shard_count)),
        page_geometry_(entries_per_page), max_pages_(max_pages),
        cohort_size_(cohort_size), numa_execution_(numa_execution),
        numa_memory_(numa_memory), target_capacity_keys_(target_capacity_keys) {
    if (execution_workers_ == 0)
      throw std::invalid_argument("execution workers must be positive");
    if (shard_count_ == 0 || !std::has_single_bit(shard_count_))
      throw std::invalid_argument("shard count must be a power of two");
    if (max_pages_ == 0)
      throw std::invalid_argument("max pages must be positive");
    if (cohort_size_ == 0 || !std::has_single_bit(cohort_size_))
      throw std::invalid_argument("cohort size must be a power of two");
    if (cohort_size_ > shard_count_ || shard_count_ % cohort_size_ != 0)
      throw std::invalid_argument("cohort size must divide shard count");
    if (max_pages_ > std::numeric_limits<uint32_t>::max() /
                         page_geometry_.entries_per_page())
      throw std::overflow_error("total topology capacity overflows uint32");
  }

  [[nodiscard]] size_t execution_workers() const noexcept {
    return execution_workers_;
  }
  [[nodiscard]] size_t shard_count() const noexcept { return shard_count_; }
  [[nodiscard]] size_t shard_mask() const noexcept { return shard_mask_; }
  [[nodiscard]] const PageGeometry &page_geometry() const noexcept {
    return page_geometry_;
  }
  [[nodiscard]] size_t entries_per_page() const noexcept {
    return page_geometry_.entries_per_page();
  }
  [[nodiscard]] unsigned entries_shift() const noexcept {
    return page_geometry_.shift();
  }
  [[nodiscard]] size_t slot_mask() const noexcept {
    return page_geometry_.slot_mask();
  }
  [[nodiscard]] size_t max_pages() const noexcept { return max_pages_; }
  [[nodiscard]] size_t cohort_size() const noexcept { return cohort_size_; }
  [[nodiscard]] size_t cohort_count() const noexcept {
    return shard_count_ / cohort_size_;
  }
  [[nodiscard]] size_t max_capacity_keys() const noexcept {
    return max_pages_ * page_geometry_.entries_per_page();
  }
  [[nodiscard]] size_t target_capacity_keys() const noexcept {
    return target_capacity_keys_;
  }
  [[nodiscard]] bool numa_execution() const noexcept { return numa_execution_; }
  [[nodiscard]] bool numa_memory() const noexcept { return numa_memory_; }

  [[nodiscard]] inline uint32_t page_id(uint32_t index) const noexcept {
    return page_geometry_.page_id(index);
  }

  [[nodiscard]] inline uint32_t page_slot(uint32_t index) const noexcept {
    return page_geometry_.slot(index);
  }

  [[nodiscard]] inline size_t shard_id(uint64_t hash) const noexcept {
    return hash & shard_mask_;
  }

private:
  static size_t checked_power_of_two(size_t value, const char *name) {
    if (value == 0 || !std::has_single_bit(value))
      throw std::invalid_argument(std::string(name) +
                                  " must be a power of two");
    return value;
  }

  static size_t checked_mask(size_t value) {
    return checked_power_of_two(value, "shard count") - 1;
  }

  size_t execution_workers_;
  size_t shard_count_;
  size_t shard_mask_;
  PageGeometry page_geometry_;
  size_t max_pages_;
  size_t cohort_size_;
  bool numa_execution_;
  bool numa_memory_;
  size_t target_capacity_keys_ = 0;
};

} // namespace se::storage
