#pragma once

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>

namespace se::hogwild_internal {

class FlatKeyDirectory {
public:
  static constexpr uint32_t kMaximumSlotId = 0x3FFFFFFFU;

  explicit FlatKeyDirectory(size_t bucket_count, unsigned hash_shift)
      : bucket_count_(validated_bucket_count(bucket_count)),
        power_of_two_mask_(
            std::has_single_bit(bucket_count_) ? bucket_count_ - 1 : 0),
        hash_shift_(hash_shift),
        slots_(std::make_unique<uint32_t[]>(bucket_count_)) {}

  FlatKeyDirectory(const FlatKeyDirectory &other)
      : bucket_count_(other.bucket_count_),
        power_of_two_mask_(other.power_of_two_mask_),
        hash_shift_(other.hash_shift_), entry_count_(other.entry_count_),
        slots_(std::make_unique<uint32_t[]>(other.bucket_count_)) {
    std::copy_n(other.slots_.get(), bucket_count_, slots_.get());
  }

  FlatKeyDirectory &operator=(const FlatKeyDirectory &) = delete;
  FlatKeyDirectory(FlatKeyDirectory &&) noexcept = default;
  FlatKeyDirectory &operator=(FlatKeyDirectory &&) noexcept = default;

  template <typename KeyForSlot>
  std::optional<uint32_t>
  find_hashed(uint64_t key, uint64_t hash,
              KeyForSlot &&key_for_slot) const noexcept {
    size_t position = bucket_position(hash);
    for (size_t probe = 0; probe < bucket_count_; ++probe) {
      const uint32_t slot_id = slots_[position];
      if (slot_id == kEmptySlot)
        return std::nullopt;
      if (slot_id != kTombstoneSlot && key_for_slot(slot_id) == key)
        return slot_id;
      advance(position);
    }
    return std::nullopt;
  }

  template <typename KeyForSlot>
  void insert_hashed(uint64_t key, uint64_t hash, uint32_t slot_id,
                     KeyForSlot &&key_for_slot) {
    if (slot_id == 0 || slot_id > kMaximumSlotId)
      throw std::invalid_argument("key directory slot id is invalid");
    size_t position = bucket_position(hash);
    size_t first_tombstone = bucket_count_;
    for (size_t probe = 0; probe < bucket_count_; ++probe) {
      const uint32_t existing_slot = slots_[position];
      if (existing_slot != kEmptySlot && existing_slot != kTombstoneSlot) {
        if (key_for_slot(existing_slot) == key)
          throw std::logic_error("key directory has a duplicate key");
      } else if (existing_slot == kTombstoneSlot) {
        if (first_tombstone == bucket_count_)
          first_tombstone = position;
      } else {
        publish(first_tombstone == bucket_count_ ? position : first_tombstone,
                slot_id);
        return;
      }
      advance(position);
    }
    if (first_tombstone != bucket_count_) {
      publish(first_tombstone, slot_id);
      return;
    }
    throw std::overflow_error("key directory is full");
  }

  template <typename KeyForSlot>
  std::optional<uint32_t> remove_hashed(uint64_t key, uint64_t hash,
                                        KeyForSlot &&key_for_slot) {
    size_t position = bucket_position(hash);
    for (size_t probe = 0; probe < bucket_count_; ++probe) {
      const uint32_t slot_id = slots_[position];
      if (slot_id == kEmptySlot)
        return std::nullopt;
      if (slot_id != kTombstoneSlot && key_for_slot(slot_id) == key) {
        slots_[position] = kTombstoneSlot;
        --entry_count_;
        return slot_id;
      }
      advance(position);
    }
    return std::nullopt;
  }

  template <typename Fn> void for_each(Fn &&fn) const {
    for (size_t index = 0; index < bucket_count_; ++index) {
      const uint32_t slot_id = slots_[index];
      if (slot_id != kEmptySlot && slot_id != kTombstoneSlot)
        fn(slot_id);
    }
  }

  size_t bucket_count() const noexcept { return bucket_count_; }

  size_t allocation_bytes() const noexcept {
    return bucket_count_ * kBytesPerBucket;
  }

  bool uses_power_of_two_mask() const noexcept {
    return power_of_two_mask_ != 0;
  }

  static size_t validated_bucket_count(size_t bucket_count) {
    if (bucket_count < 2)
      return 2;
    if (bucket_count > maximum_bucket_count())
      throw std::overflow_error("key directory capacity exceeds byte range");
    return bucket_count;
  }

  static size_t bucket_count_for_key_capacity(size_t key_capacity) {
    constexpr size_t kLegacyBucketMultiplier = 4;
    constexpr size_t kProbeBudgetNumerator = 25;
    constexpr size_t kProbeBudgetDenominator = 4;
    constexpr size_t kBucketAlignment = 16;
    if (key_capacity == 0)
      return 2;
    if (key_capacity > maximum_key_capacity())
      throw std::overflow_error("key directory key capacity exceeds range");

    const size_t probe_budget_capacity =
        (key_capacity * kProbeBudgetNumerator + (kProbeBudgetDenominator - 1)) /
        kProbeBudgetDenominator;
    const size_t alignment_remainder = probe_budget_capacity % kBucketAlignment;
    const size_t aligned_probe_budget_capacity =
        alignment_remainder == 0
            ? probe_budget_capacity
            : checked_add(probe_budget_capacity,
                          kBucketAlignment - alignment_remainder);
    if (aligned_probe_budget_capacity > maximum_bucket_count())
      throw std::overflow_error("key directory aligned capacity exceeds range");

    const size_t legacy_minimum = key_capacity * kLegacyBucketMultiplier;
    const size_t maximum_power_of_two_capacity =
        std::bit_floor(maximum_bucket_count());
    const size_t legacy_capacity =
        legacy_minimum <= maximum_power_of_two_capacity
            ? checked_bit_ceil(legacy_minimum)
            : aligned_probe_budget_capacity;
    return std::min(aligned_probe_budget_capacity, legacy_capacity);
  }

  static constexpr size_t maximum_bucket_count() noexcept {
    return std::numeric_limits<size_t>::max() / kBytesPerBucket;
  }

  static constexpr size_t maximum_key_capacity() noexcept {
    return ((maximum_bucket_count() - 15) * 4 - 3) / 25;
  }

private:
  static constexpr size_t kBytesPerBucket = sizeof(uint32_t);
  static constexpr uint32_t kEmptySlot = 0;
  static constexpr uint32_t kTombstoneSlot =
      std::numeric_limits<uint32_t>::max();

  static size_t checked_add(size_t left, size_t right) {
    if (right > std::numeric_limits<size_t>::max() - left)
      throw std::overflow_error("key directory capacity addition overflow");
    return left + right;
  }

  static size_t checked_bit_ceil(size_t value) {
    if (value <= 1)
      return 1;
    const unsigned width = std::bit_width(value - 1);
    if (width >= std::numeric_limits<size_t>::digits)
      throw std::overflow_error("key directory power-of-two capacity overflow");
    const size_t capacity = size_t{1} << width;
    if (capacity > maximum_bucket_count())
      throw std::overflow_error("key directory capacity exceeds byte range");
    return capacity;
  }

  size_t bucket_position(uint64_t hash) const noexcept {
    if (power_of_two_mask_ != 0)
      return (hash >> hash_shift_) & power_of_two_mask_;
    return static_cast<size_t>(
        (static_cast<unsigned __int128>(hash) * bucket_count_) >> 64);
  }

  void advance(size_t &position) const noexcept {
    ++position;
    if (position == bucket_count_)
      position = 0;
  }

  void publish(size_t position, uint32_t slot_id) noexcept {
    slots_[position] = slot_id;
    ++entry_count_;
  }

  size_t bucket_count_;
  size_t power_of_two_mask_;
  unsigned hash_shift_;
  size_t entry_count_ = 0;
  std::unique_ptr<uint32_t[]> slots_;
};

} // namespace se::hogwild_internal
