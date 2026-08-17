#pragma once

#ifndef SPARSE_EMBEDDING_SEALED_HOT_PAGE_HPP_
#define SPARSE_EMBEDDING_SEALED_HOT_PAGE_HPP_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace se {

enum class PageThermalTier : uint8_t { kHot, kCold };

struct SealedPageLayout {
  PageThermalTier tier;
  uint32_t cluster_id;
  uint64_t layout_generation;
  uint32_t entry_count;
  uint32_t entry_capacity;
  uint32_t embedding_dim;
  uint32_t page_slot_stride_bytes;
};

struct SealedPageByteDeleter {
  void operator()(std::byte *pointer) const noexcept {
    if (pointer != nullptr)
      ::operator delete(pointer, std::align_val_t{64});
  }
};

template <typename Value> class SealedHotPage {
public:
  const SealedPageLayout &layout() const noexcept { return layout_; }
  std::span<const uint64_t> keys() const noexcept { return keys_; }
  std::span<const uint64_t> seq_counts() const noexcept { return seq_counts_; }

  std::span<const Value> value(uint32_t slot) const {
    if (slot >= layout_.entry_count)
      throw std::out_of_range("sealed page slot is out of range");
    const auto *data = reinterpret_cast<const Value *>(
        payload_.get() +
        static_cast<size_t>(slot) * layout_.page_slot_stride_bytes);
    return {data, layout_.embedding_dim};
  }

  const std::byte *payload_data() const noexcept { return payload_.get(); }
  size_t payload_size() const noexcept {
    return static_cast<size_t>(layout_.entry_count) *
           layout_.page_slot_stride_bytes;
  }

private:
  template <typename> friend class SealedHotPageBuilder;

  SealedPageLayout layout_{};
  std::unique_ptr<std::byte[], SealedPageByteDeleter> payload_;
  std::vector<uint64_t> keys_;
  std::vector<uint64_t> seq_counts_;
};

template <typename Value> class SealedHotPageBuilder {
public:
  static constexpr uint32_t kCacheLineSize = 64;

  SealedHotPageBuilder(PageThermalTier tier, uint32_t cluster_id,
                       uint64_t page_layout_generation, uint32_t entry_capacity,
                       uint32_t embedding_dim)
      : layout_{tier,           cluster_id,    page_layout_generation,       0,
                entry_capacity, embedding_dim, aligned_stride(embedding_dim)},
        payload_(static_cast<std::byte *>(
            ::operator new(static_cast<size_t>(entry_capacity) *
                               layout_.page_slot_stride_bytes,
                           std::align_val_t{kCacheLineSize}))) {
    if (page_layout_generation == 0 || entry_capacity == 0 ||
        embedding_dim == 0)
      throw std::invalid_argument("sealed page layout is incomplete");
    keys_.reserve(entry_capacity);
    seq_counts_.reserve(entry_capacity);
  }

  uint32_t append(uint64_t key, uint64_t seq_count,
                  std::span<const Value> value) {
    if (sealed_)
      throw std::logic_error("sealed page cannot accept another entry");
    if (seq_count == 0 || value.size() != layout_.embedding_dim)
      throw std::invalid_argument("sealed page entry is invalid");
    if (layout_.entry_count == layout_.entry_capacity)
      throw std::length_error("sealed page capacity is exhausted");
    const uint32_t slot = layout_.entry_count;
    std::byte *destination =
        payload_.get() +
        static_cast<size_t>(slot) * layout_.page_slot_stride_bytes;
    std::memcpy(destination, value.data(),
                static_cast<size_t>(layout_.embedding_dim) * sizeof(Value));
    keys_.push_back(key);
    seq_counts_.push_back(seq_count);
    ++layout_.entry_count;
    return slot;
  }

  SealedHotPage<Value> seal() {
    if (sealed_)
      throw std::logic_error("sealed page was already published");
    if (layout_.entry_count == 0)
      throw std::logic_error("empty sealed page cannot be published");
    sealed_ = true;
    SealedHotPage<Value> page;
    page.layout_ = layout_;
    page.payload_ = std::move(payload_);
    page.keys_ = std::move(keys_);
    page.seq_counts_ = std::move(seq_counts_);
    return page;
  }

private:
  static uint32_t aligned_stride(uint32_t embedding_dim) {
    const size_t bytes = static_cast<size_t>(embedding_dim) * sizeof(Value);
    if (bytes > UINT32_MAX - (kCacheLineSize - 1))
      throw std::overflow_error("sealed page slot stride overflows uint32");
    return static_cast<uint32_t>((bytes + kCacheLineSize - 1) &
                                 ~(size_t{kCacheLineSize} - 1));
  }

  SealedPageLayout layout_{};
  std::unique_ptr<std::byte[], SealedPageByteDeleter> payload_;
  std::vector<uint64_t> keys_;
  std::vector<uint64_t> seq_counts_;
  bool sealed_ = false;
};

// SealedHotPagePool owns immutable destination pages produced by compaction.
// It is deliberately outside the live key directory: page ids are physical
// metadata only, while lookups continue to use the authoritative shard map.
template <typename Value> class SealedHotPagePool {
public:
  struct PoolState {
    size_t maximum_pages;
    mutable std::mutex mutex;
    size_t next_page_id = 0;
    std::vector<uint32_t> free_page_ids;
    std::vector<bool> reserved;
    std::vector<std::shared_ptr<const SealedHotPage<Value>>> pages;

    explicit PoolState(size_t page_count)
        : maximum_pages(page_count), reserved(page_count, false),
          pages(page_count) {
      free_page_ids.reserve(page_count);
    }
  };

  struct Reservation {
    std::shared_ptr<PoolState> state;
    uint32_t page_id;
    SealedHotPageBuilder<Value> builder;

    Reservation(std::shared_ptr<PoolState> state_in, uint32_t page_id_in,
                SealedHotPageBuilder<Value> &&builder_in)
        : state(std::move(state_in)), page_id(page_id_in),
          builder(std::move(builder_in)) {}

    Reservation(const Reservation &) = delete;
    Reservation &operator=(const Reservation &) = delete;

    Reservation(Reservation &&other) noexcept
        : state(std::move(other.state)), page_id(other.page_id),
          builder(std::move(other.builder)) {
      other.page_id = 0;
    }

    Reservation &operator=(Reservation &&other) noexcept {
      if (this == &other)
        return *this;
      release();
      state = std::move(other.state);
      page_id = other.page_id;
      builder = std::move(other.builder);
      other.page_id = 0;
      return *this;
    }

    ~Reservation() { release(); }

  private:
    void release() noexcept {
      if (state != nullptr)
        SealedHotPagePool::release_reservation(state, page_id);
      state.reset();
    }

    friend class SealedHotPagePool;
  };

  SealedHotPagePool(size_t maximum_pages, uint32_t entry_capacity,
                    uint32_t embedding_dim)
      : state_(make_state(maximum_pages)), entry_capacity_(entry_capacity),
        embedding_dim_(embedding_dim) {
    if (maximum_pages == 0 || entry_capacity == 0 || embedding_dim == 0)
      throw std::invalid_argument("sealed page pool dimensions are invalid");
    if (maximum_pages > std::numeric_limits<uint32_t>::max())
      throw std::overflow_error("sealed page pool id space is exhausted");
  }

  Reservation reserve(PageThermalTier tier, uint32_t cluster_id,
                      uint64_t page_layout_generation) {
    SealedHotPageBuilder<Value> builder(tier, cluster_id,
                                        page_layout_generation, entry_capacity_,
                                        embedding_dim_);
    std::lock_guard lock(state_->mutex);
    uint32_t page_id = 0;
    if (!state_->free_page_ids.empty()) {
      page_id = state_->free_page_ids.back();
      state_->free_page_ids.pop_back();
    } else if (state_->next_page_id == state_->maximum_pages) {
      throw std::length_error("sealed page pool capacity is exhausted");
    } else {
      page_id = static_cast<uint32_t>(state_->next_page_id++);
    }
    state_->reserved[page_id] = true;
    return Reservation{state_, page_id, std::move(builder)};
  }

  std::shared_ptr<const SealedHotPage<Value>>
  publish(Reservation &&reservation) {
    {
      std::lock_guard lock(state_->mutex);
      if (reservation.state != state_ ||
          reservation.page_id >= state_->next_page_id ||
          !state_->reserved[reservation.page_id] ||
          state_->pages[reservation.page_id] != nullptr)
        throw std::logic_error("sealed page reservation is not publishable");
    }
    auto page =
        std::make_shared<SealedHotPage<Value>>(reservation.builder.seal());
    if (page->layout().entry_count == 0 ||
        page->layout().entry_count > entry_capacity_ ||
        page->layout().embedding_dim != embedding_dim_)
      throw std::logic_error("sealed page publication has invalid layout");
    std::lock_guard lock(state_->mutex);
    if (reservation.state != state_ ||
        reservation.page_id >= state_->next_page_id ||
        !state_->reserved[reservation.page_id] ||
        state_->pages[reservation.page_id] != nullptr)
      throw std::logic_error("sealed page reservation is not publishable");
    state_->pages[reservation.page_id] = page;
    state_->reserved[reservation.page_id] = false;
    reservation.state.reset();
    return page;
  }

  std::shared_ptr<const SealedHotPage<Value>> snapshot(uint32_t page_id) const {
    std::lock_guard lock(state_->mutex);
    if (page_id >= state_->next_page_id || state_->pages[page_id] == nullptr)
      throw std::out_of_range("sealed page is not published");
    return state_->pages[page_id];
  }

  size_t published_count() const {
    std::lock_guard lock(state_->mutex);
    size_t count = 0;
    for (const auto &page : state_->pages)
      count += page != nullptr;
    return count;
  }

private:
  static std::shared_ptr<PoolState> make_state(size_t maximum_pages) {
    if (maximum_pages == 0)
      throw std::invalid_argument("sealed page pool dimensions are invalid");
    if (maximum_pages > std::numeric_limits<uint32_t>::max())
      throw std::overflow_error("sealed page pool id space is exhausted");
    return std::make_shared<PoolState>(maximum_pages);
  }

  static void release_reservation(const std::shared_ptr<PoolState> &state,
                                  uint32_t page_id) noexcept {
    std::lock_guard lock(state->mutex);
    if (page_id >= state->next_page_id || !state->reserved[page_id] ||
        state->pages[page_id] != nullptr)
      return;
    state->reserved[page_id] = false;
    state->free_page_ids.push_back(page_id);
  }

  std::shared_ptr<PoolState> state_;
  uint32_t entry_capacity_;
  uint32_t embedding_dim_;
};

} // namespace se

#endif
