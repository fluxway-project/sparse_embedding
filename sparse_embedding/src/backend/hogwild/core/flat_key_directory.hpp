#pragma once

#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>

namespace se::hogwild_internal {

class FlatKeyDirectory {
  using Bucket = std::atomic<uint32_t>;
  static_assert(Bucket::is_always_lock_free &&
                sizeof(Bucket) == sizeof(uint32_t));

public:
  static constexpr uint32_t kMaximumSlotId = 0x3FFFFFFFU;

  explicit FlatKeyDirectory(size_t bucket_count, unsigned hash_shift)
      : bucket_count_(validated_bucket_count(bucket_count)),
        power_of_two_mask_(
            std::has_single_bit(bucket_count_) ? bucket_count_ - 1 : 0),
        hash_shift_(hash_shift),
        slots_(std::make_unique<Bucket[]>(bucket_count_)) {}

  FlatKeyDirectory(const FlatKeyDirectory &other)
      : bucket_count_(other.bucket_count_),
        power_of_two_mask_(other.power_of_two_mask_),
        hash_shift_(other.hash_shift_),
        slots_(std::make_unique<Bucket[]>(other.bucket_count_)) {
    for (size_t i = 0; i < bucket_count_; ++i)
      slots_[i].store(other.slots_[i].load(std::memory_order_relaxed),
                      std::memory_order_relaxed);
  }

  FlatKeyDirectory &operator=(const FlatKeyDirectory &) = delete;
  FlatKeyDirectory(FlatKeyDirectory &&) noexcept = default;
  FlatKeyDirectory &operator=(FlatKeyDirectory &&) noexcept = default;

  void prefetch_bucket_cacheline(uint64_t hash) const noexcept {
    const size_t position = bucket_position(hash);
    __builtin_prefetch(&slots_[position], 0, 3);
  }

  template <typename KeyForSlot>
  std::optional<uint32_t>
  find_hashed(uint64_t key, uint64_t hash,
              KeyForSlot &&key_for_slot) const noexcept {
    return locate(key, hash, key_for_slot).slot;
  }

  template <typename KeyForSlot>
  void insert_hashed(uint64_t key, uint64_t hash, uint32_t slot_id,
                     KeyForSlot &&key_for_slot) {
    if (slot_id == 0 || slot_id > kMaximumSlotId)
      throw std::invalid_argument("key directory slot id is invalid");
    const Match target = locate(key, hash, key_for_slot);
    if (target.slot)
      throw std::logic_error("key directory has a duplicate key");
    if (target.position == bucket_count_)
      throw std::overflow_error("key directory is full");
    slots_[target.position].store(slot_id, std::memory_order_release);
  }

  void publish_reserved_absent(uint64_t hash, uint32_t slot_id) noexcept {
    if (!try_publish_absent(hash, slot_id))
      std::terminate();
  }

  template <typename KeyForSlot>
  std::optional<uint32_t> remove_hashed(uint64_t key, uint64_t hash,
                                        KeyForSlot &&key_for_slot) {
    const auto found = locate(key, hash, key_for_slot);
    if (found.slot)
      slots_[found.position].store(kTombstoneSlot, std::memory_order_release);
    return found.slot;
  }

  template <typename Fn> void for_each(Fn &&fn) const {
    for (size_t index = 0; index < bucket_count_; ++index) {
      const uint32_t slot_id = slots_[index].load(std::memory_order_acquire);
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
    if (key_capacity == 0)
      return 2;
    if (key_capacity > maximum_key_capacity())
      throw std::overflow_error("key directory key capacity exceeds range");

    return std::bit_floor(key_capacity) * kBucketsPerCapacityStep;
  }

  static constexpr size_t maximum_bucket_count() noexcept {
    return std::numeric_limits<size_t>::max() / kBytesPerBucket;
  }

  static constexpr size_t maximum_key_capacity() noexcept {
    return (std::bit_floor(maximum_bucket_count()) / kBucketsPerCapacityStep) *
               2 -
           1;
  }

private:
  static constexpr size_t kBucketsPerCapacityStep = 4;
  static constexpr size_t kBytesPerBucket = sizeof(uint32_t);
  static constexpr uint32_t kEmptySlot = 0;
  static constexpr uint32_t kTombstoneSlot =
      std::numeric_limits<uint32_t>::max();

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

  struct Match {
    size_t position;
    std::optional<uint32_t> slot;
  };

  template <typename KeyForSlot>
  Match locate(uint64_t key, uint64_t hash,
               KeyForSlot &key_for_slot) const noexcept {
    size_t position = bucket_position(hash);
    size_t first_tombstone = bucket_count_;
    for (size_t probe = 0; probe < bucket_count_; ++probe) {
      const auto slot = slots_[position].load(std::memory_order_acquire);
      if (slot == kEmptySlot)
        return {first_tombstone == bucket_count_ ? position : first_tombstone,
                std::nullopt};
      if (slot == kTombstoneSlot) {
        if (first_tombstone == bucket_count_)
          first_tombstone = position;
        advance(position);
        continue;
      }
      if (key_for_slot(slot) == key)
        return {position, slot};
      advance(position);
    }
    return {first_tombstone, std::nullopt};
  }

  bool try_publish_absent(uint64_t hash, uint32_t slot_id) noexcept {
    size_t position = bucket_position(hash);
    for (size_t probe = 0; probe < bucket_count_; ++probe) {
      const auto slot = slots_[position].load(std::memory_order_relaxed);
      if (slot == kEmptySlot || slot == kTombstoneSlot) {
        slots_[position].store(slot_id, std::memory_order_release);
        return true;
      }
      advance(position);
    }
    return false;
  }

  size_t bucket_count_;
  size_t power_of_two_mask_;
  unsigned hash_shift_;
  std::unique_ptr<Bucket[]> slots_;
};

} // namespace se::hogwild_internal
