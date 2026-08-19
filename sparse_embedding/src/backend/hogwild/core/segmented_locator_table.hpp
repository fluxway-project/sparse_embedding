#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <vector>

#include "sparse_embedding/src/memory/seqlock.hpp"

namespace se::hogwild_internal {

using se::memory::PackedWriteGuard;

struct alignas(8) LocatorSlot {
  std::atomic<uint64_t> packed{0};
  uint64_t key{0};
  std::atomic<uint32_t> readers{0};
};

static_assert(sizeof(LocatorSlot) == 24);
static_assert(alignof(LocatorSlot) == 8);
static_assert(std::atomic<uint64_t>::is_always_lock_free);

class LocatorSequenceExhausted final : public std::runtime_error {
public:
  LocatorSequenceExhausted()
      : std::runtime_error("locator slot sequence requires rotation") {}
};

class SegmentedLocatorTable {
public:
  static constexpr uint32_t kEntriesPerSegment = 4096;
  static constexpr uint32_t kMaxSegments = 8192;
  static constexpr uint32_t kMaximumSlotId = kEntriesPerSegment * kMaxSegments;
  static constexpr uint32_t kRotationSequence =
      std::numeric_limits<uint32_t>::max() - 3U;

  SegmentedLocatorTable() = default;
  ~SegmentedLocatorTable() {
    for (auto &segment : segments_)
      delete segment.load(std::memory_order_relaxed);
  }

  SegmentedLocatorTable(const SegmentedLocatorTable &) = delete;
  SegmentedLocatorTable &operator=(const SegmentedLocatorTable &) = delete;

  uint32_t allocate(uint64_t key, uint32_t arena_index) {
    const std::optional<uint32_t> recycled = take_recycled_slot();
    const uint32_t slot_id =
        recycled.has_value()
            ? *recycled
            : next_slot_.fetch_add(1, std::memory_order_relaxed);
    if (slot_id == 0 || slot_id > kMaximumSlotId)
      throw std::length_error("locator slot capacity exhausted");
    LocatorSlot &slot = locate_or_create(slot_id);
    if (recycled.has_value()) {
      PackedWriteGuard guard(slot.packed, slot.readers);
      const uint64_t current = slot.packed.load(std::memory_order_acquire);
      if (decode_index(current) != 0)
        throw std::logic_error("recycled locator slot is still live");
      slot.key = key;
      slot.packed.store(pack(version_of(current), encode(arena_index)),
                        std::memory_order_release);
    } else {
      slot.key = key;
      slot.packed.store(pack(0, encode(arena_index)),
                        std::memory_order_release);
    }
    return slot_id;
  }

  bool release_after_epoch(uint32_t slot_id) {
    LocatorSlot &slot = locate_required(slot_id);
    const uint32_t version =
        version_of(slot.packed.load(std::memory_order_acquire));
    if (version >= kRotationSequence)
      return false;
    PackedWriteGuard guard(slot.packed, slot.readers);
    const uint64_t current = slot.packed.load(std::memory_order_acquire);
    if (decode_index(current) == 0)
      throw std::logic_error("locator slot was already released");
    slot.packed.store(pack(version_of(current), 0), std::memory_order_release);
    {
      std::lock_guard lock(recycle_mutex_);
      recycled_slots_.push_back(slot_id);
    }
    return true;
  }

  std::optional<uint32_t> load(uint32_t slot_id) const noexcept {
    const LocatorSlot *slot = locate(slot_id);
    if (slot == nullptr)
      return std::nullopt;
    const uint32_t index =
        decode_index(slot->packed.load(std::memory_order_acquire));
    if (index == 0)
      return std::nullopt;
    return decode(index);
  }

  uint64_t key(uint32_t slot_id) const noexcept {
    const LocatorSlot *slot = locate(slot_id);
    if (slot == nullptr)
      std::terminate();
    return slot->key;
  }

  bool requires_rotation(uint32_t slot_id) const {
    const LocatorSlot *slot = locate(slot_id);
    if (slot == nullptr)
      throw std::out_of_range("locator slot is not allocated");
    return version_of(slot->packed.load(std::memory_order_acquire)) >=
           kRotationSequence;
  }

  void set_sequence_for_test(uint32_t slot_id, uint32_t sequence) {
    if ((sequence & 1U) != 0U ||
        sequence > std::numeric_limits<uint32_t>::max() - 1U)
      throw std::invalid_argument(
          "test locator sequence is not an allowed even value");
    LocatorSlot &slot = locate_required(slot_id);
    const uint64_t current = slot.packed.load(std::memory_order_acquire);
    slot.packed.store(pack(sequence, decode_index(current)),
                      std::memory_order_release);
  }

  uint64_t load_packed(uint32_t slot_id) const {
    const LocatorSlot *slot = locate(slot_id);
    if (slot == nullptr)
      throw std::out_of_range("locator slot is not allocated");
    return load_packed_at(slot);
  }

  const LocatorSlot *slot_address(uint32_t slot_id) const {
    const LocatorSlot *slot = locate(slot_id);
    if (slot == nullptr)
      throw std::out_of_range("locator slot is not allocated");
    return slot;
  }

  void prefetch_slot(uint32_t slot_id) const noexcept {
    const LocatorSlot *slot = locate(slot_id);
    if (slot != nullptr) {
      __builtin_prefetch(slot, 0, 3);
    }
  }

  static uint64_t load_packed_at(const LocatorSlot *slot) noexcept {
    return slot->packed.load(std::memory_order_acquire);
  }

  static std::optional<uint32_t> decode_packed(uint64_t packed) noexcept {
    const uint32_t encoded = decode_index(packed);
    if (encoded == 0)
      return std::nullopt;
    return decode(encoded);
  }

  template <typename Read> decltype(auto) read(uint32_t slot_id, Read &&read) {
    LocatorSlot &slot = locate_required(slot_id);
    while (true) {
      const uint64_t before = slot.packed.load(std::memory_order_acquire);
      if ((static_cast<uint32_t>(before >> 32) & 1U) != 0U) {
        se::memory::cpu_relax();
        continue;
      }
      const uint32_t encoded = decode_index(before);
      if (encoded == 0)
        throw std::logic_error("locator slot is not live");
      se::memory::compiler_barrier();
      if constexpr (std::is_void_v<std::invoke_result_t<Read, uint32_t>>) {
        std::forward<Read>(read)(decode(encoded));
        se::memory::compiler_barrier();
        if (before == slot.packed.load(std::memory_order_acquire))
          return;
      } else {
        auto result = std::forward<Read>(read)(decode(encoded));
        se::memory::compiler_barrier();
        if (before == slot.packed.load(std::memory_order_acquire))
          return result;
      }
    }
  }

  template <typename Read> bool try_read(uint32_t slot_id, Read &&read) {
    const LocatorSlot *slot = locate(slot_id);
    if (slot == nullptr)
      return false;
    while (true) {
      const uint64_t before = slot->packed.load(std::memory_order_acquire);
      if ((static_cast<uint32_t>(before >> 32) & 1U) != 0U) {
        se::memory::cpu_relax();
        continue;
      }
      const uint32_t encoded = decode_index(before);
      if (encoded == 0)
        return false;
      se::memory::compiler_barrier();
      std::forward<Read>(read)(decode(encoded));
      se::memory::compiler_barrier();
      if (before == slot->packed.load(std::memory_order_acquire))
        return true;
    }
  }

  template <typename Read>
  bool try_read_key(uint32_t slot_id, uint64_t expected_key, Read &&read) {
    const LocatorSlot *slot = locate(slot_id);
    if (slot == nullptr)
      return false;
    while (true) {
      const uint64_t before = slot->packed.load(std::memory_order_acquire);
      if ((static_cast<uint32_t>(before >> 32) & 1U) != 0U) {
        se::memory::cpu_relax();
        continue;
      }
      const uint32_t encoded = decode_index(before);
      if (encoded == 0 || slot->key != expected_key)
        return false;
      se::memory::compiler_barrier();
      std::forward<Read>(read)(decode(encoded));
      se::memory::compiler_barrier();
      if (before == slot->packed.load(std::memory_order_acquire))
        return true;
    }
  }

  template <typename Write>
  decltype(auto) write(uint32_t slot_id, Write &&write) {
    LocatorSlot &slot = locate_required(slot_id);
    if (version_of(slot.packed.load(std::memory_order_acquire)) >
        kRotationSequence)
      throw LocatorSequenceExhausted();
    PackedWriteGuard guard(slot.packed, slot.readers);
    const uint64_t current = slot.packed.load(std::memory_order_acquire);
    const uint32_t encoded = decode_index(current);
    if (encoded == 0)
      throw std::logic_error("locator slot is not live");
    return std::forward<Write>(write)(decode(encoded), slot.packed);
  }

  template <typename Write> bool try_write(uint32_t slot_id, Write &&write) {
    LocatorSlot &slot = locate_required(slot_id);
    if (version_of(slot.packed.load(std::memory_order_acquire)) >
        kRotationSequence)
      throw LocatorSequenceExhausted();
    PackedWriteGuard guard(slot.packed, slot.readers);
    const uint64_t current = slot.packed.load(std::memory_order_acquire);
    const uint32_t encoded = decode_index(current);
    if (encoded == 0)
      return false;
    std::forward<Write>(write)(decode(encoded), slot.packed);
    return true;
  }

  template <typename Write>
  bool try_write_key(uint32_t slot_id, uint64_t expected_key, Write &&write) {
    LocatorSlot &slot = locate_required(slot_id);
    if (version_of(slot.packed.load(std::memory_order_acquire)) >
        kRotationSequence)
      throw LocatorSequenceExhausted();
    PackedWriteGuard guard(slot.packed, slot.readers);
    const uint64_t current = slot.packed.load(std::memory_order_acquire);
    const uint32_t encoded = decode_index(current);
    if (encoded == 0 || slot.key != expected_key)
      return false;
    std::forward<Write>(write)(decode(encoded), slot.packed);
    return true;
  }

  template <typename Retire>
  bool retire(uint32_t slot_id, uint32_t expected_index, Retire &&retire) {
    LocatorSlot &slot = locate_required(slot_id);
    if (version_of(slot.packed.load(std::memory_order_acquire)) >
        kRotationSequence)
      throw LocatorSequenceExhausted();
    PackedWriteGuard guard(slot.packed, slot.readers);
    const uint64_t current = slot.packed.load(std::memory_order_acquire);
    if (decode_index(current) != encode(expected_index))
      return false;
    std::forward<Retire>(retire)(expected_index);
    slot.packed.store(pack(version_of(current), 0), std::memory_order_release);
    return true;
  }

  template <typename Retire>
  bool retire_current(uint32_t slot_id, Retire &&retire) noexcept {
    static_assert(std::is_nothrow_invocable_v<Retire, uint32_t>);
    LocatorSlot *located = const_cast<LocatorSlot *>(locate(slot_id));
    if (located == nullptr)
      std::terminate();
    LocatorSlot &slot = *located;
    PackedWriteGuard guard(slot.packed, slot.readers);
    const uint64_t current = slot.packed.load(std::memory_order_acquire);
    const uint32_t encoded = decode_index(current);
    if (encoded == 0)
      return false;
    std::forward<Retire>(retire)(decode(encoded));
    slot.packed.store(pack(version_of(current), 0), std::memory_order_release);
    return true;
  }

  bool recycle_retired_after_epoch(uint32_t slot_id) noexcept {
    LocatorSlot &slot = locate_required(slot_id);
    const uint32_t version =
        version_of(slot.packed.load(std::memory_order_acquire));
    if (version >= kRotationSequence)
      return false;
    PackedWriteGuard guard(slot.packed, slot.readers);
    if (decode_index(slot.packed.load(std::memory_order_acquire)) != 0)
      std::terminate();
    try {
      std::lock_guard lock(recycle_mutex_);
      recycled_slots_.push_back(slot_id);
    } catch (...) {
      std::terminate();
    }
    return true;
  }

  bool publish(uint32_t slot_id, uint32_t expected_index,
               uint32_t destination_index) {
    LocatorSlot &slot = locate_required(slot_id);
    if (version_of(slot.packed.load(std::memory_order_acquire)) >
        kRotationSequence)
      throw LocatorSequenceExhausted();
    PackedWriteGuard guard(slot.packed, slot.readers);
    const uint64_t current = slot.packed.load(std::memory_order_acquire);
    if (decode_index(current) != encode(expected_index))
      return false;
    slot.packed.store(pack(version_of(current), encode(destination_index)),
                      std::memory_order_release);
    return true;
  }

  template <typename Copy>
  bool relocate(uint32_t slot_id, uint32_t expected_index,
                uint32_t destination_index, Copy &&copy) {
    LocatorSlot &slot = locate_required(slot_id);
    if (version_of(slot.packed.load(std::memory_order_acquire)) >
        kRotationSequence)
      throw LocatorSequenceExhausted();
    PackedWriteGuard guard(slot.packed, slot.readers);
    const uint64_t current = slot.packed.load(std::memory_order_acquire);
    if (decode_index(current) != encode(expected_index))
      return false;
    std::forward<Copy>(copy)(expected_index);
    slot.packed.store(pack(version_of(current), encode(destination_index)),
                      std::memory_order_release);
    return true;
  }

private:
  using Segment = std::array<LocatorSlot, kEntriesPerSegment>;

  static uint32_t segment_id(uint32_t slot_id) noexcept {
    return (slot_id - 1U) / kEntriesPerSegment;
  }

  static uint32_t offset(uint32_t slot_id) noexcept {
    return (slot_id - 1U) % kEntriesPerSegment;
  }

  static uint32_t encode(uint32_t index) {
    if (index == std::numeric_limits<uint32_t>::max())
      throw std::overflow_error("locator arena index encoding overflows");
    return index + 1U;
  }

  static uint32_t decode(uint32_t encoded) noexcept { return encoded - 1U; }

  static uint64_t pack(uint32_t version, uint32_t encoded) noexcept {
    return (static_cast<uint64_t>(version) << 32) | encoded;
  }

  static uint32_t version_of(uint64_t packed) noexcept {
    return static_cast<uint32_t>(packed >> 32);
  }

  static uint32_t decode_index(uint64_t packed) noexcept {
    return static_cast<uint32_t>(packed);
  }

  static uint32_t next_version(uint64_t packed) {
    const uint32_t version = version_of(packed);
    if (version >= kRotationSequence)
      throw LocatorSequenceExhausted();
    return version + 2U;
  }

  const LocatorSlot *locate(uint32_t slot_id) const noexcept {
    if (slot_id == 0 || slot_id > kMaximumSlotId)
      return nullptr;
    Segment *segment =
        segments_[segment_id(slot_id)].load(std::memory_order_acquire);
    return segment == nullptr ? nullptr : &(*segment)[offset(slot_id)];
  }

  LocatorSlot &locate_required(uint32_t slot_id) {
    LocatorSlot *slot = const_cast<LocatorSlot *>(locate(slot_id));
    if (slot == nullptr)
      throw std::out_of_range("locator slot is not allocated");
    return *slot;
  }

  LocatorSlot &locate_or_create(uint32_t slot_id) {
    std::atomic<Segment *> &published = segments_[segment_id(slot_id)];
    Segment *segment = published.load(std::memory_order_acquire);
    if (segment == nullptr) {
      auto candidate = std::make_unique<Segment>();
      Segment *expected = nullptr;
      if (published.compare_exchange_strong(expected, candidate.get(),
                                            std::memory_order_release,
                                            std::memory_order_acquire)) {
        segment = candidate.release();
      } else {
        segment = expected;
      }
    }
    return (*segment)[offset(slot_id)];
  }

  std::optional<uint32_t> take_recycled_slot() {
    std::lock_guard lock(recycle_mutex_);
    if (recycled_slots_.empty())
      return std::nullopt;
    const uint32_t slot_id = recycled_slots_.back();
    recycled_slots_.pop_back();
    return slot_id;
  }

  std::array<std::atomic<Segment *>, kMaxSegments> segments_{};
  std::atomic<uint32_t> next_slot_{1};
  std::mutex recycle_mutex_;
  std::vector<uint32_t> recycled_slots_;
};

static_assert(sizeof(LocatorSlot) * SegmentedLocatorTable::kEntriesPerSegment ==
              98304);

} // namespace se::hogwild_internal
