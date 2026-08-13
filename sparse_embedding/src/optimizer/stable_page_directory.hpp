#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace se {

template <typename T, size_t EntriesPerSegment = 4096,
          size_t RootEntries = 1024>
class StablePageDirectory {
  static_assert(EntriesPerSegment != 0);
  static_assert((EntriesPerSegment & (EntriesPerSegment - 1)) == 0);

  struct Segment {
    std::unique_ptr<std::atomic<T *>[]> slots;

    Segment() : slots(std::make_unique<std::atomic<T *>[]>(EntriesPerSegment)) {
      for (size_t i = 0; i < EntriesPerSegment; ++i)
        slots[i].store(nullptr, std::memory_order_relaxed);
    }
  };

public:
  static constexpr size_t kEntriesPerSegment = EntriesPerSegment;
  static constexpr size_t kMaximumEntries = EntriesPerSegment * RootEntries;

  StablePageDirectory() {
    for (auto &root : root_)
      root.store(nullptr, std::memory_order_relaxed);
  }

  StablePageDirectory(const StablePageDirectory &) = delete;
  StablePageDirectory &operator=(const StablePageDirectory &) = delete;

  ~StablePageDirectory() = default;

  T *load(size_t index) const {
    const size_t id = segment_id(index);
    Segment *segment = id == 0 ? first_segment_.load(std::memory_order_acquire)
                               : segment_at(index);
    return segment == nullptr
               ? nullptr
               : segment->slots[offset(index)].load(std::memory_order_acquire);
  }

  void publish(size_t index, T *value) {
    Segment *segment = ensure_segment(index);
    segment->slots[offset(index)].store(value, std::memory_order_release);
  }

  size_t segment_count() const noexcept {
    return segments_.load(std::memory_order_acquire);
  }

  size_t page_count() const noexcept {
    return entries_.load(std::memory_order_acquire);
  }

  void advance_count() {
    size_t observed = entries_.load(std::memory_order_relaxed);
    while (true) {
      if (observed >= kMaximumEntries)
        throw std::length_error(
            "stable page directory entry capacity exceeded");
      if (entries_.compare_exchange_weak(observed, observed + 1,
                                         std::memory_order_release,
                                         std::memory_order_relaxed))
        return;
    }
  }

  void clear() {
    std::lock_guard lock(mutex_);
    for (auto &root : root_)
      root.store(nullptr, std::memory_order_release);
    first_segment_.store(nullptr, std::memory_order_release);
    owned_segments_.clear();
    segments_.store(0, std::memory_order_release);
    entries_.store(0, std::memory_order_release);
  }

private:
  static size_t segment_id(size_t index) noexcept {
    return index / EntriesPerSegment;
  }

  static size_t offset(size_t index) noexcept {
    return index & (EntriesPerSegment - 1);
  }

  Segment *segment_at(size_t index) const {
    const size_t id = segment_id(index);
    if (id >= RootEntries)
      throw std::length_error("stable page directory capacity exceeded");
    return root_[id].load(std::memory_order_acquire);
  }

  Segment *ensure_segment(size_t index) {
    const size_t id = segment_id(index);
    if (id >= RootEntries)
      throw std::length_error("stable page directory capacity exceeded");
    Segment *segment = root_[id].load(std::memory_order_acquire);
    if (segment != nullptr)
      return segment;
    std::lock_guard lock(mutex_);
    segment = root_[id].load(std::memory_order_relaxed);
    if (segment == nullptr) {
      auto owned = std::make_unique<Segment>();
      segment = owned.get();
      owned_segments_.push_back(std::move(owned));
      root_[id].store(segment, std::memory_order_release);
      if (id == 0)
        first_segment_.store(segment, std::memory_order_release);
      segments_.fetch_add(1, std::memory_order_release);
    }
    return segment;
  }

  std::array<std::atomic<Segment *>, RootEntries> root_{};
  std::atomic<Segment *> first_segment_{nullptr};
  std::vector<std::unique_ptr<Segment>> owned_segments_;
  std::atomic<size_t> segments_{0};
  std::atomic<size_t> entries_{0};
  mutable std::mutex mutex_;
};

} // namespace se
