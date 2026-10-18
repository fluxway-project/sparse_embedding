// Copyright 2026 The Fluxway Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#pragma once

#ifndef SPARSE_EMBEDDING_HOGWILD_EMBEDDING_STORE_HPP_
#define SPARSE_EMBEDDING_HOGWILD_EMBEDDING_STORE_HPP_

#include <ankerl/unordered_dense.h>
#include <parlay/parallel.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <exec/static_thread_pool.hpp>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <random>
#include <ranges>
#include <span>
#include <stdexcept>
#include <stdexec/execution.hpp>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "sparse_embedding/src/arena/gradient_accumulator.hpp"
#include "sparse_embedding/src/arena/transient_arena.hpp"
#include "sparse_embedding/src/backend/hogwild/compaction/page_compaction_profile.hpp"
#include "sparse_embedding/src/backend/hogwild/core/cohort_arena_allocation.hpp"
#include "sparse_embedding/src/backend/hogwild/core/flat_key_directory.hpp"
#include "sparse_embedding/src/backend/hogwild/core/segmented_locator_table.hpp"
#include "sparse_embedding/src/backend/hogwild/io/hogwild_page_image_codec.hpp"
#include "sparse_embedding/src/backend/hogwild/operation_dispatch_policy.hpp"
#include "sparse_embedding/src/backend/hogwild/partition/hogwild_partition_topology.hpp"
#include "sparse_embedding/src/backend/hogwild/public/hogwild_store_options.hpp"
#include "sparse_embedding/src/backend/hogwild/store_page_eviction_proxy.hpp"
#include "sparse_embedding/src/checkpoint/ordered_checkpoint.hpp"
#include "sparse_embedding/src/common/function.hpp"
#include "sparse_embedding/src/concurrency/epoch_retirement.hpp"
#include "sparse_embedding/src/eviction/adaptive_sealed_page_planner.hpp"
#include "sparse_embedding/src/eviction/page_telemetry.hpp"
#include "sparse_embedding/src/eviction/score_policy.hpp"
#include "sparse_embedding/src/execution/reusable_workspace_pool.hpp"
#include "sparse_embedding/src/execution/topology/thread_affinity.hpp"
#include "sparse_embedding/src/interface/embedding_table_interface.hpp"
#include "sparse_embedding/src/interface/initializer_interface.hpp"
#include "sparse_embedding/src/interface/optimizer_interface.hpp"
#include "sparse_embedding/src/observability/diagnostic_sink.hpp"
#include "sparse_embedding/src/observability/telemetry_ring.hpp"
#include "sparse_embedding/src/optimizer/stable_page_directory.hpp"
#include "sparse_embedding/src/simd/memory_ops.hpp"
#include "sparse_embedding/src/simd/prefetch.hpp"
#include "sparse_embedding/src/storage/page_storage.hpp"
#include "sparse_embedding/src/storage/topology/storage_resource_tuner.hpp"
#include "sparse_embedding/src/storage/topology/storage_topology.hpp"

namespace se {

#ifdef SPARSE_EMBEDDING_TEST_HOOKS
template <typename ParamT, typename OptT> class HogwildEmbeddingStoreCore;
namespace testing {
template <typename ParamT, typename OptT>
void set_lru_clock_for_testing(HogwildEmbeddingStoreCore<ParamT, OptT> &store,
                               uint64_t value);
} // namespace testing
#endif

// =============================================================================
// EmbeddingArena: Lock-free, 64-byte cache-line aligned region allocator.
// Index allocation uses atomic ranges. Page readiness is published per physical
// page. Score stored separately to keep data() SIMD-aligned.
// =============================================================================

template <typename ValT> class EmbeddingArena {
public:
  static constexpr size_t kCacheLineSize = 64;
  static constexpr size_t kEntriesPerPage = 4096;
  static constexpr size_t kMaxPages = 8192;
  static constexpr size_t kCohortSize = 16;
  using PageAllocationObserver = void (*)(void *, uint64_t);

  struct OwnerSnapshot {
    uint64_t key;
    uint64_t seq_count;
    uint32_t shard_id;
  };

  struct OwnerRecord {
    std::atomic<uint64_t> seq_count{0};
    std::atomic<uint64_t> key{0};
    std::atomic<uint32_t> shard_id{0};
    std::atomic<bool> live{false};
  };

private:
  enum class PageAllocationPhase : uint8_t {
    kUnallocated,
    kAllocating,
    kReady,
    kRetiring,
    kRetired,
  };

  struct alignas(kCacheLineSize) PageAllocationState {
    std::atomic<uint8_t> phase;

    PageAllocationState()
        : phase(static_cast<uint8_t>(PageAllocationPhase::kUnallocated)) {}
  };

  struct PageTable {
    StablePageDirectory<std::byte> data;
    StablePageDirectory<uint64_t> score;
  };

  struct ArenaPageState {
    std::unique_ptr<PageAllocationState[]> states;
    CohortArenaAllocation allocation;

    explicit ArenaPageState(const storage::StorageTopology &topology)
        : states(std::make_unique<PageAllocationState[]>(topology.max_pages())),
          allocation(topology) {}

    ArenaPageState(size_t partition_count, size_t cohort_size,
                   size_t entries_per_page = 4096, size_t max_pages = 8192)
        : states(std::make_unique<PageAllocationState[]>(max_pages)),
          allocation(partition_count, cohort_size, entries_per_page,
                     max_pages) {}
  };

public:
  explicit EmbeddingArena(
      size_t dim, bool enable_score, PageStorage storage,
      std::shared_ptr<RetirementPressure> retirement_pressure,
      PageAllocationObserver page_allocation_observer,
      void *page_allocation_context, const storage::StorageTopology &topology)
      : dim_(dim), enable_score_(enable_score), topology_(topology),
        page_geometry_(topology.page_geometry()),
        pages_(std::make_shared<PageTable>()), storage_(std::move(storage)),
        retirement_pressure_(std::move(retirement_pressure)),
        partition_count_(topology.shard_count()),
        numa_aware_pages_(topology.numa_memory()),
        page_cohort_size_(topology.cohort_size()),
        data_pages_(std::make_unique<StoragePage[]>(topology.max_pages())),
        score_pages_(std::make_unique<StoragePage[]>(topology.max_pages())),
        page_state_(std::make_shared<ArenaPageState>(topology)),
        page_allocation_observer_(page_allocation_observer),
        page_allocation_context_(page_allocation_context),
        owner_pages_(std::make_unique<std::unique_ptr<OwnerRecord[]>[]>(
            topology.max_pages())),
        live_words_(
            std::make_unique<std::atomic<uint64_t>[]>(live_word_count())) {
    if ((page_allocation_observer_ == nullptr) !=
        (page_allocation_context_ == nullptr)) {
      throw std::invalid_argument(
          "embedding arena page allocation observer is incomplete");
    }
    if (!retirement_pressure_)
      throw std::invalid_argument(
          "embedding arena retirement pressure is null");
    if (numa_aware_pages_ && !NumaTopology::available())
      throw std::runtime_error("NUMA-aware pages require libnuma support");
    data_stride_ =
        storage::PageGeometry::aligned_slot_stride(dim * sizeof(ValT));
    for (size_t i = 0; i < live_word_count(); ++i)
      live_words_[i].store(0, std::memory_order_relaxed);
  }

  explicit EmbeddingArena(
      size_t dim, bool enable_score, PageStorage storage,
      std::shared_ptr<RetirementPressure> retirement_pressure,
      PageAllocationObserver page_allocation_observer = nullptr,
      void *page_allocation_context = nullptr, size_t partition_count = 1,
      bool numa_aware_pages = false)
      : EmbeddingArena(
            dim, enable_score, std::move(storage),
            std::move(retirement_pressure), page_allocation_observer,
            page_allocation_context,
            storage::StorageTopology(
                partition_count, partition_count, 4096, 8192,
                numa_aware_pages
                    ? numa_page_cohort_size(partition_count, numa_aware_pages)
                    : std::min(size_t{16}, partition_count),
                numa_aware_pages, numa_aware_pages)) {}

  ~EmbeddingArena() = default;

  EmbeddingArena(const EmbeddingArena &) = delete;
  EmbeddingArena &operator=(const EmbeddingArena &) = delete;
  EmbeddingArena(EmbeddingArena &&) = delete;
  EmbeddingArena &operator=(EmbeddingArena &&) = delete;

  size_t dim() const noexcept { return dim_; }
  bool score_enabled() const noexcept { return enable_score_; }
  bool numa_aware_pages() const noexcept { return numa_aware_pages_; }
  const storage::StorageTopology &topology() const noexcept {
    return topology_;
  }
  size_t entries_per_page() const noexcept {
    return page_geometry_.entries_per_page();
  }
  unsigned entries_shift() const noexcept { return page_geometry_.shift(); }
  size_t slot_mask() const noexcept { return page_geometry_.slot_mask(); }
  size_t max_pages() const noexcept { return topology_.max_pages(); }
  size_t max_entries() const noexcept { return topology_.max_capacity_keys(); }
  size_t live_word_count() const noexcept { return (max_entries() + 63) / 64; }

  template <typename Prepare> uint32_t allocate(Prepare &&prepare) {
    uint32_t idx = next_index_.load(std::memory_order_relaxed);
    while (true) {
      ensure_capacity(idx);
      prepare(idx);
      if (next_index_.compare_exchange_weak(idx, idx + 1,
                                            std::memory_order_relaxed,
                                            std::memory_order_relaxed)) {
        prepare_unpublished(idx);
        return idx;
      }
    }
  }

  uint32_t allocate() {
    return allocate([](uint32_t) {});
  }

  template <typename Prepare>
  uint32_t allocate_fresh(uint32_t shard_id, Prepare &&prepare) {
    return allocate_fresh(shard_id, std::forward<Prepare>(prepare),
                          [](uint32_t) { return true; });
  }

  template <typename Prepare, typename Accept>
  uint32_t allocate_fresh(uint32_t shard_id, Prepare &&prepare,
                          Accept &&accept) {
    while (true) {
      const auto allocation = page_state_->allocation.acquire_fresh(shard_id);
      ensure_page_ready(allocation.page_id, numa_node_for_shard(shard_id));
      if (!accept(allocation.index))
        continue;
      if (allocation.slot == 0)
        advance_allocation_watermark_past_page(allocation.page_id);
      prepare(allocation.index);
      prepare_unpublished(allocation.index);
      return allocation.index;
    }
  }

  template <typename Prepare, typename Accept>
  void allocate_fresh_batch(uint32_t shard_id, size_t count,
                            std::span<uint32_t> output, Prepare &&prepare,
                            Accept &&accept) {
    if (count == 0 || output.size() != count)
      throw std::invalid_argument("embedding arena batch size is invalid");
    size_t output_position = 0;
    page_state_->allocation.acquire_fresh_batch(
        shard_id, count,
        [&](const CohortArenaAllocation::Allocation &allocation) {
          return accept(allocation.index);
        },
        [&](const CohortArenaAllocation::Allocation &allocation) {
          ensure_page_ready(allocation.page_id, numa_node_for_shard(shard_id));
          if (allocation.slot == 0)
            advance_allocation_watermark_past_page(allocation.page_id);
          output[output_position++] = allocation.index;
        });
    if (output_position != count)
      throw std::logic_error("embedding arena batch allocation count mismatch");
    prepare(output.front(), output.back() + 1);
    for (uint32_t index : output)
      prepare_unpublished(index);
  }

  void bind_exact(uint32_t index, uint32_t shard_id) {
    const size_t pid = page_geometry_.page_id(index);
    ensure_page_ready(pid, numa_node_for_shard(shard_id));
    page_state_->allocation.bind_exact(index, shard_id);
    advance_allocation_watermark_past_page(pid);
  }

  template <typename Prepare>
  uint32_t allocate_range(size_t count, Prepare &&prepare) {
    if (count == 0)
      throw std::invalid_argument("embedding arena range must not be empty");
    if (count > max_entries())
      throw std::length_error("embedding arena range exceeds capacity");
    if (numa_aware_pages_)
      throw std::logic_error(
          "NUMA-aware arena does not support unsharded range allocation");
    uint32_t first = next_index_.load(std::memory_order_relaxed);
    while (true) {
      if (first > max_entries() - count)
        throw std::length_error("embedding arena range exceeds capacity");
      const uint32_t end = static_cast<uint32_t>(first + count);
      if (!next_index_.compare_exchange_weak(first, end,
                                             std::memory_order_acq_rel,
                                             std::memory_order_acquire)) {
        continue;
      }
      page_state_->allocation.reserve_contiguous_range(first, end);
      ensure_range_ready(first, end - 1);
      prepare(first, end);
      return first;
    }
  }

  void reserve_entries(size_t capacity) {
    if (capacity == 0)
      return;
    if (capacity > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
      throw std::overflow_error("embedding arena capacity exceeds uint32_t");
    }
    if (numa_aware_pages_)
      throw std::logic_error(
          "NUMA-aware arena does not support unsharded capacity reservation");
    ensure_range_ready(0, static_cast<uint32_t>(capacity - 1));
  }

  void reserve_exact_page(uint64_t page_id) {
    if (page_id >= max_pages())
      throw std::out_of_range("embedding arena page id is out of range");
    if (numa_aware_pages_)
      throw std::logic_error(
          "NUMA-aware arena page-image import requires shard placement");
    ensure_page_ready(static_cast<size_t>(page_id), 0);
  }

  void reserve_exact_page(uint64_t page_id, uint32_t shard_id) {
    if (page_id >= max_pages())
      throw std::out_of_range("embedding arena page id is out of range");
    ensure_page_ready(static_cast<size_t>(page_id),
                      numa_node_for_shard(shard_id));
  }

  void set_allocation_count(uint32_t count) {
    if (count > max_entries())
      throw std::length_error(
          "embedding arena allocation count exceeds capacity");
    if (count != 0)
      ensure_capacity(count - 1);
    next_index_.store(count, std::memory_order_release);
  }

  uint64_t &score_raw(uint32_t index) const {
    auto *page = score_page_for(index);
    if (page == nullptr)
      throw std::logic_error("embedding arena score page is unavailable");
    return page[slot_for(index)];
  }

  ValT *data(uint32_t index) const {
    if (data_page_for(index) == nullptr)
      throw std::logic_error("embedding arena index belongs to a retired page");
    return data_trusted(index);
  }

  ValT *data_trusted(uint32_t index) const noexcept {
    return reinterpret_cast<ValT *>(data_raw(index));
  }

  size_t allocated_count() const {
    return next_index_.load(std::memory_order_relaxed);
  }

  size_t capacity() const {
    return num_pages_.load(std::memory_order_acquire) * entries_per_page();
  }

  size_t page_count() const {
    return num_pages_.load(std::memory_order_acquire);
  }

  bool page_allocated(uint64_t page_id) const {
    return page_id < page_count() &&
           page_state_->states[page_id].phase.load(std::memory_order_acquire) ==
               page_phase(PageAllocationPhase::kReady);
  }

  void advance_allocation_past_page(uint64_t page_id) {
    if (page_id >= max_pages())
      throw std::out_of_range("arena allocation page id is out of range");
    const uint64_t end = (page_id + 1) * entries_per_page();
    if (end > std::numeric_limits<uint32_t>::max())
      throw std::overflow_error("arena allocation watermark overflows");
    ensure_capacity(static_cast<uint32_t>(end));
    advance_allocation_watermark_past_page(page_id);
  }

  bool page_has_live_entries(uint64_t page_id) const {
    if (page_id >= page_count())
      throw std::out_of_range("arena live-page id is out of range");
    const size_t live_words_per_page = entries_per_page() / 64;
    const size_t first_word = page_id * live_words_per_page;
    for (size_t word = 0; word < live_words_per_page; ++word) {
      if (live_words_[first_word + word].load(std::memory_order_acquire) != 0)
        return true;
    }
    return false;
  }

  template <typename OnReclaimed>
  void retire_empty_page(uint64_t page_id, OnReclaimed &&on_reclaimed) {
    if (page_id >= page_count())
      throw std::out_of_range("arena retire page id is out of range");
    if (page_has_live_entries(page_id))
      throw std::logic_error("arena page still has live entries");
    using Reclaim = std::decay_t<OnReclaimed>;
    static_assert(noexcept(std::declval<Reclaim &>()()));
    struct DeferredPageRelease {
      std::shared_ptr<PageTable> page_table;
      std::shared_ptr<ArenaPageState> page_state;
      uint64_t page_id;
      StoragePage data;
      StoragePage score;
      std::unique_ptr<OwnerRecord[]> owner;
      Reclaim on_reclaimed;

      ~DeferredPageRelease() noexcept {
        page_table->data.publish(page_id, nullptr);
        page_table->score.publish(page_id, nullptr);
        data = StoragePage{};
        score = StoragePage{};
        owner.reset();
        page_state->states[page_id].phase.store(
            page_phase(PageAllocationPhase::kUnallocated),
            std::memory_order_release);
        page_state->allocation.release_page(static_cast<uint32_t>(page_id));
        on_reclaimed();
      }
    };
    const size_t retired_bytes = data_pages_[page_id].allocated_size() +
                                 score_pages_[page_id].allocated_size();
    se::concurrency::EpochRetirement<DeferredPageRelease> release(
        retirement_pressure_, retired_bytes);
    PageAllocationState &state = page_state_->states[page_id];
    uint8_t expected = page_phase(PageAllocationPhase::kReady);
    if (!state.phase.compare_exchange_strong(
            expected, page_phase(PageAllocationPhase::kRetiring),
            std::memory_order_acq_rel, std::memory_order_acquire)) {
      throw std::logic_error("arena page is not ready for retirement");
    }
    release.emplace(
        pages_, page_state_, page_id, std::move(data_pages_[page_id]),
        std::move(score_pages_[page_id]), std::move(owner_pages_[page_id]),
        std::forward<OnReclaimed>(on_reclaimed));
    release.publish();
    state.phase.store(page_phase(PageAllocationPhase::kRetired),
                      std::memory_order_release);
  }

  void reset() {
    pages_ = std::make_shared<PageTable>();
    page_state_ = std::make_shared<ArenaPageState>(topology_);
    for (size_t i = 0; i < max_pages(); ++i) {
      data_pages_[i] = StoragePage{};
      score_pages_[i] = StoragePage{};
      owner_pages_[i].reset();
    }
    for (size_t i = 0; i < live_word_count(); ++i)
      live_words_[i].store(0, std::memory_order_relaxed);
    num_pages_.store(0, std::memory_order_relaxed);
    next_index_.store(0, std::memory_order_relaxed);
  }

  size_t data_stride() const noexcept { return data_stride_; }

  void mark_live(uint32_t index, bool live) {
    const uint64_t mask = uint64_t{1} << (index % 64);
    OwnerRecord &owner =
        owner_pages_[index >> entries_shift()][index & slot_mask()];
    if (live) {
      live_words_[index / 64].fetch_or(mask, std::memory_order_release);
      owner.live.store(true, std::memory_order_release);
    } else {
      owner.live.store(false, std::memory_order_release);
      live_words_[index / 64].fetch_and(~mask, std::memory_order_release);
    }
  }

  void mark_owner_not_live(uint32_t index) {
    owner_pages_[index >> entries_shift()][index & slot_mask()].live.store(
        false, std::memory_order_release);
  }

  void clear_live_mask(size_t word_index, uint64_t mask) {
    if (word_index >= live_word_count())
      throw std::out_of_range("embedding arena live word is out of range");
    if (mask == 0)
      throw std::invalid_argument("embedding arena live mask is empty");
    live_words_[word_index].fetch_and(~mask, std::memory_order_release);
  }

  bool is_live(uint32_t index) const {
    const uint64_t mask = uint64_t{1} << (index % 64);
    return (live_words_[index / 64].load(std::memory_order_acquire) & mask) !=
           0;
  }

  void prepare_reused(uint32_t index) { prepare_unpublished(index); }

  void prepare_unpublished(uint32_t index) {
    if (enable_score_)
      score_raw(index) = 0;
    mark_live(index, false);
  }

  void publish_owner(uint32_t index, uint64_t key, uint32_t shard_id) {
    OwnerRecord &owner =
        owner_pages_[index >> entries_shift()][index & slot_mask()];
    uint64_t seq_count = owner.seq_count.load(std::memory_order_relaxed);
    if ((seq_count & 1U) != 0)
      throw std::logic_error("embedding arena owner seq_count is unstable");
    owner.seq_count.store(seq_count + 1, std::memory_order_release);
    owner.key.store(key, std::memory_order_relaxed);
    owner.shard_id.store(shard_id, std::memory_order_relaxed);
    owner.seq_count.store(seq_count + 2, std::memory_order_release);
  }

  void publish_owner_exact(uint32_t index, uint64_t key, uint32_t shard_id,
                           uint64_t seq_count) {
    if (seq_count == 0 || (seq_count & 1U) != 0)
      throw std::invalid_argument("page-image owner seq_count is invalid");
    OwnerRecord &owner =
        owner_pages_[index >> entries_shift()][index & slot_mask()];
    if (owner.live.load(std::memory_order_acquire))
      throw std::logic_error("page-image owner slot is already live");
    owner.seq_count.store(seq_count - 1, std::memory_order_release);
    owner.key.store(key, std::memory_order_relaxed);
    owner.shard_id.store(shard_id, std::memory_order_relaxed);
    owner.seq_count.store(seq_count, std::memory_order_release);
  }

  std::optional<OwnerSnapshot> owner_snapshot(uint32_t index) const {
    OwnerRecord &owner =
        owner_pages_[index >> entries_shift()][index & slot_mask()];
    if (!is_live(index) || !owner.live.load(std::memory_order_acquire))
      return std::nullopt;
    for (size_t attempt = 0; attempt < 8; ++attempt) {
      const uint64_t before = owner.seq_count.load(std::memory_order_acquire);
      if ((before & 1U) != 0)
        continue;
      const uint64_t key = owner.key.load(std::memory_order_relaxed);
      const uint32_t shard_id = owner.shard_id.load(std::memory_order_relaxed);
      const uint64_t after = owner.seq_count.load(std::memory_order_acquire);
      if (before == after && (after & 1U) == 0 &&
          owner.live.load(std::memory_order_acquire) && is_live(index))
        return OwnerSnapshot{key, after, shard_id};
    }
    throw std::runtime_error(
        "embedding arena owner snapshot did not stabilize");
  }

private:
  uint64_t page_score(size_t page) const {
    size_t first = page * entries_per_page();
    size_t allocated = allocated_count();
    if (first >= allocated)
      return 0;
    size_t count = std::min(entries_per_page(), allocated - first);
    uint64_t total = 0;
    for (auto index : std::views::iota(static_cast<uint32_t>(first),
                                       static_cast<uint32_t>(first + count))) {
      if (!is_live(index))
        continue;
      uint64_t score = std::atomic_ref<uint64_t>(score_raw(index))
                           .load(std::memory_order_relaxed);
      if (total > std::numeric_limits<uint64_t>::max() - score)
        return std::numeric_limits<uint64_t>::max();
      total += score;
    }
    return total;
  }

  std::byte *data_raw(uint32_t index) const {
    auto *page = data_page_for(index);
    if (page == nullptr)
      throw std::logic_error("embedding arena data page is unavailable");
    return page + slot_for(index) * data_stride_;
  }

  std::byte *data_page_for(uint32_t index) const noexcept {
    return pages_->data.load(index >> entries_shift());
  }

  uint64_t *score_page_for(uint32_t index) const noexcept {
    return pages_->score.load(index >> entries_shift());
  }

  uint32_t slot_for(uint32_t index) const noexcept {
    return index & page_geometry_.slot_mask();
  }

  static constexpr uint8_t page_phase(PageAllocationPhase phase) {
    return static_cast<uint8_t>(phase);
  }

  static size_t numa_page_cohort_size(size_t shard_count,
                                      bool numa_aware_pages) {
    if (shard_count == 0 || !std::has_single_bit(shard_count))
      throw std::invalid_argument("arena shard count is not a power of two");
    if (!numa_aware_pages)
      return std::min(size_t{16}, shard_count);
    if (!NumaTopology::available())
      throw std::runtime_error("NUMA-aware pages require libnuma support");
    const size_t nodes = NumaTopology::node_count();
    if (nodes == 0 || !std::has_single_bit(nodes) || shard_count < nodes ||
        shard_count % nodes != 0) {
      throw std::invalid_argument(
          "NUMA-aware pages require a power-of-two node count that divides "
          "the shard count");
    }
    return shard_count / nodes;
  }

  int numa_node_for_shard(uint32_t shard_id) const {
    if (!numa_aware_pages_)
      return 0;
    return NumaTopology::node_for_shard(shard_id, partition_count_);
  }

  int numa_node_for_page(size_t page_id) const {
    if (!numa_aware_pages_)
      return 0;
    const uint32_t cohort =
        page_state_->allocation.page_cohort(static_cast<uint32_t>(page_id));
    const size_t shard = static_cast<size_t>(cohort) * page_cohort_size_;
    return numa_node_for_shard(static_cast<uint32_t>(shard));
  }

  static void bind_page_to_numa_node(void *address, size_t bytes, int node) {
    if (address == nullptr || bytes == 0)
      throw std::invalid_argument(
          "NUMA page binding requires allocated memory");
    const std::vector<int> nodes = NumaTopology::node_ids();
    if (node < 0 || std::ranges::find(nodes, node) == nodes.end())
      throw std::out_of_range("NUMA page binding node is out of range");
    numa_tonode_memory(address, bytes, node);
    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0)
      throw std::runtime_error("NUMA page size query failed");
    const size_t stride = static_cast<size_t>(page_size);
    const size_t page_count = (bytes + stride - 1) / stride;
    std::array<void *, 2> pages{};
    std::array<int, 2> placement{-1, -1};
    auto *base = static_cast<std::byte *>(address);
    pages[0] = base;
    pages[1] = base + (page_count - 1) * stride;
    static_cast<std::byte *>(pages[0])[0] = std::byte{0};
    static_cast<std::byte *>(pages[1])[0] = std::byte{0};
    if (numa_move_pages(0, pages.size(), pages.data(), nullptr,
                        placement.data(), 0) != 0)
      throw std::runtime_error("NUMA page placement query failed");
    if (std::ranges::find_if(placement, [node](int observed) {
          return observed != node;
        }) != placement.end()) {
      throw std::runtime_error("NUMA page placement does not match shard node");
    }
  }

  void advance_allocation_watermark_past_page(uint64_t page_id) {
    if (page_id >= max_pages())
      throw std::out_of_range("arena allocation page id is out of range");
    const uint64_t end = (page_id + 1) * entries_per_page();
    if (end > std::numeric_limits<uint32_t>::max())
      throw std::overflow_error("arena allocation watermark overflows");
    uint32_t observed = next_index_.load(std::memory_order_acquire);
    while (observed < end &&
           !next_index_.compare_exchange_weak(
               observed, static_cast<uint32_t>(end), std::memory_order_acq_rel,
               std::memory_order_acquire)) {
    }
  }

  void ensure_range_ready(uint32_t first, uint32_t last) {
    if (last >= max_entries())
      throw std::length_error("embedding arena page capacity exceeded");
    const size_t first_page = first >> entries_shift();
    const size_t last_page = last >> entries_shift();
    for (size_t page_id = first_page; page_id <= last_page; ++page_id)
      ensure_page_ready(page_id, numa_node_for_page(page_id));
  }

  void ensure_capacity(uint32_t idx) {
    ensure_page_ready(idx >> entries_shift(),
                      numa_node_for_page(idx >> entries_shift()));
  }

  void ensure_page_ready(size_t page_id, int numa_node) {
    const size_t page_needed = page_id;
    if (page_needed >= max_pages()) {
      throw std::length_error("embedding arena page capacity exceeded");
    }

    PageAllocationState &state = page_state_->states[page_needed];
    uint8_t observed = state.phase.load(std::memory_order_acquire);
    while (true) {
      if (observed == page_phase(PageAllocationPhase::kReady))
        return;
      if (observed == page_phase(PageAllocationPhase::kAllocating)) {
        state.phase.wait(observed, std::memory_order_acquire);
        observed = state.phase.load(std::memory_order_acquire);
        continue;
      }
      if (observed == page_phase(PageAllocationPhase::kUnallocated)) {
        if (!state.phase.compare_exchange_weak(
                observed, page_phase(PageAllocationPhase::kAllocating),
                std::memory_order_acq_rel, std::memory_order_acquire)) {
          continue;
        }
        try {
          allocate_page(page_needed, numa_node);
          publish_page_count(page_needed + 1);
          state.phase.store(page_phase(PageAllocationPhase::kReady),
                            std::memory_order_release);
          state.phase.notify_all();
          return;
        } catch (...) {
          state.phase.store(page_phase(PageAllocationPhase::kUnallocated),
                            std::memory_order_release);
          state.phase.notify_all();
          throw;
        }
      }
      if (observed == page_phase(PageAllocationPhase::kRetiring)) {
        throw std::logic_error("arena page ready transition was lost");
      }
      if (observed == page_phase(PageAllocationPhase::kRetired)) {
        throw std::logic_error("arena page is retired");
      }
      throw std::logic_error("arena page has an invalid readiness state");
    }
  }

  void publish_page_count(size_t minimum_page_count) {
    size_t current = num_pages_.load(std::memory_order_acquire);
    while (current < minimum_page_count &&
           !num_pages_.compare_exchange_weak(current, minimum_page_count,
                                             std::memory_order_release,
                                             std::memory_order_acquire)) {
    }
  }

  void allocate_page(size_t page_id, int numa_node) {
    PageEvaluator evaluator;
    if (enable_score_)
      evaluator = [this, page_id] { return page_score(page_id); };
    const PageEvictionMode eviction_mode = enable_score_
                                               ? PageEvictionMode::kEnabled
                                               : PageEvictionMode::kDisabled;
    const PagePlacementMode placement_mode =
        numa_aware_pages_ ? PagePlacementMode::kPageAlignedMapping
                          : PagePlacementMode::kDefault;
    auto data_page =
        storage_->allocate(entries_per_page() * data_stride_, kCacheLineSize,
                           evaluator, eviction_mode, placement_mode);
    StoragePage score_page;
    if (enable_score_) {
      score_page = storage_->allocate(
          entries_per_page() * sizeof(uint64_t), kCacheLineSize,
          [] { return uint64_t{0}; }, PageEvictionMode::kEnabled,
          placement_mode);
    }
    auto owner_page = std::make_unique<OwnerRecord[]>(entries_per_page());
    if (numa_aware_pages_) {
      bind_page_to_numa_node(data_page.data(), data_page.allocated_size(),
                             numa_node);
      if (enable_score_)
        bind_page_to_numa_node(score_page.data(), score_page.allocated_size(),
                               numa_node);
    }
    if (page_allocation_observer_ != nullptr)
      page_allocation_observer_(page_allocation_context_, page_id);
    data_pages_[page_id] = std::move(data_page);
    score_pages_[page_id] = std::move(score_page);
    owner_pages_[page_id] = std::move(owner_page);
    pages_->data.publish(page_id, data_pages_[page_id].data());
    pages_->data.advance_count();
    if (enable_score_) {
      pages_->score.publish(
          page_id, reinterpret_cast<uint64_t *>(score_pages_[page_id].data()));
      pages_->score.advance_count();
    }
  }

  size_t dim_;
  size_t data_stride_;
  bool enable_score_;
  storage::StorageTopology topology_;
  storage::PageGeometry page_geometry_;
  std::shared_ptr<PageTable> pages_;
  PageStorage storage_;
  std::shared_ptr<RetirementPressure> retirement_pressure_;
  size_t partition_count_;
  bool numa_aware_pages_;
  size_t page_cohort_size_;
  std::unique_ptr<StoragePage[]> data_pages_;
  std::unique_ptr<StoragePage[]> score_pages_;
  std::shared_ptr<ArenaPageState> page_state_;
  PageAllocationObserver page_allocation_observer_;
  void *page_allocation_context_;
  std::unique_ptr<std::unique_ptr<OwnerRecord[]>[]> owner_pages_;
  std::unique_ptr<std::atomic<uint64_t>[]> live_words_;
  std::atomic<uint32_t> next_index_{0};
  std::atomic<size_t> num_pages_{0};
};

// =============================================================================
// HogwildEmbeddingStore — Lock-free architecture via ParlayHash
//
// ParlayHash (parlay::parlay_unordered_map) provides O(1) lock-free
// concurrent Key→uint32_t index mapping. EmbeddingArena provides
// 64-byte aligned contiguous memory for SIMD-friendly dense vectors.
// =============================================================================

template <typename ParamT, typename OptT> class HogwildEmbeddingStoreCore {
public:
#ifdef SPARSE_EMBEDDING_TEST_HOOKS
  template <typename TestParamT, typename TestOptT>
  friend void testing::set_lru_clock_for_testing(
      HogwildEmbeddingStoreCore<TestParamT, TestOptT> &, uint64_t);
#endif

  using KeyT = uint64_t;
  using ValueType = ParamT;
  using Initializer = pro::proxy<InitializerFacade<ParamT>>;
  using OptimizerFactory =
      se::Function<OptimizerExtension<ParamT, OptT>(size_t)>;

private:
  struct CheckpointMutationRecord {
    static constexpr uint64_t kDeleted = uint64_t{1} << 63;
    static constexpr uint64_t kMaximumSequence = kDeleted - 1;
    static constexpr uint64_t kNoPhysicalPage =
        std::numeric_limits<uint64_t>::max();

    uint32_t dimension;
    uint64_t key;
    const size_t owner_slot;
    std::atomic<uint64_t> state{0};
    std::atomic<bool> queued{false};
    uint64_t deleted_page_id = kNoPhysicalPage;
    CheckpointMutationRecord *next = nullptr;

    CheckpointMutationRecord(uint32_t dimension_in, uint64_t key_in,
                             size_t owner_slot_in)
        : dimension(dimension_in), key(key_in), owner_slot(owner_slot_in) {}
  };

  struct alignas(64) ShardDirtyHead {
    std::atomic<CheckpointMutationRecord *> head{nullptr};
  };

  CheckpointMutationRecord *allocate_checkpoint_record(uint32_t dimension,
                                                       uint64_t key) {
    std::lock_guard structure_lock(checkpoint_record_structure_mutex_);
    if (checkpoint_free_record_slots_.capacity() <
        checkpoint_record_storage_.size() + 1)
      checkpoint_free_record_slots_.reserve(
          std::bit_ceil(checkpoint_record_storage_.size() + 1));
    const size_t owner_slot = checkpoint_free_record_slots_.empty()
                                  ? checkpoint_record_storage_.size()
                                  : checkpoint_free_record_slots_.back();
    auto record =
        std::make_unique<CheckpointMutationRecord>(dimension, key, owner_slot);
    CheckpointMutationRecord *pointer = record.get();
    if (checkpoint_free_record_slots_.empty()) {
      checkpoint_record_storage_.push_back(std::move(record));
    } else {
      checkpoint_free_record_slots_.pop_back();
      checkpoint_record_storage_[owner_slot] = std::move(record);
    }
    return pointer;
  }

  void discard_unpublished_checkpoint_record(
      CheckpointMutationRecord *record) noexcept {
    std::lock_guard structure_lock(checkpoint_record_structure_mutex_);
    if (record->state.load(std::memory_order_relaxed) != 0 ||
        record->queued.load(std::memory_order_relaxed))
      std::terminate();
    const size_t owner_slot = record->owner_slot;
    checkpoint_record_storage_[owner_slot].reset();
    checkpoint_free_record_slots_.push_back(owner_slot);
  }

  static CheckpointMutationRecord *
  record_from_pointer(hogwild_internal::CheckpointRecordRef pointer) {
    if (!pointer.has_value())
      throw std::logic_error("live locator has no checkpoint mutation record");
    return static_cast<CheckpointMutationRecord *>(pointer.opaque_address());
  }

  size_t checkpoint_shard(uint64_t key) const noexcept {
    return SplitMixHash<uint64_t>{}(key)&storage_topology_.shard_mask();
  }

  void admit_checkpoint_record(CheckpointMutationRecord &record) noexcept {
    if (record.queued.exchange(true, std::memory_order_acq_rel))
      return;
    auto &head = checkpoint_dirty_heads_[checkpoint_shard(record.key)].head;
    CheckpointMutationRecord *current = head.load(std::memory_order_relaxed);
    do {
      record.next = current;
    } while (!head.compare_exchange_weak(current, &record,
                                         std::memory_order_release,
                                         std::memory_order_relaxed));
  }

  uint64_t reserve_checkpoint_sequences(size_t count) {
    if (count == 0)
      throw std::invalid_argument(
          "checkpoint sequence reservation cannot be empty");
    if (!checkpoint_journal_valid_.load(std::memory_order_acquire))
      throw std::overflow_error("checkpoint mutation journal is invalid");
    constexpr uint64_t maximum = CheckpointMutationRecord::kMaximumSequence;
    if (count > maximum) {
      checkpoint_journal_valid_.store(false, std::memory_order_release);
      throw std::overflow_error("checkpoint mutation sequence exhausted");
    }
    const uint64_t width = static_cast<uint64_t>(count);
    const uint64_t previous = checkpoint_mutation_sequence_.fetch_add(
        width, std::memory_order_relaxed);
    if (previous > maximum - width) {
      checkpoint_journal_valid_.store(false, std::memory_order_release);
      throw std::overflow_error("checkpoint mutation sequence exhausted");
    }
    return previous + 1;
  }

  uint64_t reserve_checkpoint_sequence() {
    return reserve_checkpoint_sequences(1);
  }

  void ensure_page_layout_generation_available() const {
    if (checkpoint_page_layout_generation_.load(std::memory_order_acquire) ==
        std::numeric_limits<uint64_t>::max())
      throw std::overflow_error("checkpoint page layout generation exhausted");
  }

  void record_page_layout_change() {
    ensure_page_layout_generation_available();
    checkpoint_page_layout_generation_.fetch_add(1, std::memory_order_release);
  }

  void
  record_checkpoint_mutation(hogwild_internal::CheckpointRecordRef pointer) {
    CheckpointMutationRecord &record = *record_from_pointer(pointer);
    const uint64_t sequence = reserve_checkpoint_sequence();
    if (sequence == 0)
      throw std::overflow_error("checkpoint mutation sequence exhausted");
    uint64_t previous = record.state.load(std::memory_order_acquire);
    while (previous < sequence) {
      if ((previous & CheckpointMutationRecord::kDeleted) != 0)
        return;
      if (record.state.compare_exchange_weak(previous, sequence,
                                             std::memory_order_acq_rel,
                                             std::memory_order_acquire))
        break;
    }
    if ((previous & CheckpointMutationRecord::kDeleted) == 0)
      admit_checkpoint_record(record);
  }

  void
  publish_checkpoint_deletion(hogwild_internal::CheckpointRecordRef pointer,
                              uint64_t reserved_sequence,
                              uint64_t physical_page_id) noexcept {
    CheckpointMutationRecord &record = *record_from_pointer(pointer);
    if (reserved_sequence == 0 ||
        reserved_sequence > CheckpointMutationRecord::kMaximumSequence)
      std::terminate();
    if (physical_page_id == CheckpointMutationRecord::kNoPhysicalPage)
      std::terminate();
    record.deleted_page_id = physical_page_id;
    uint64_t previous = record.state.load(std::memory_order_acquire);
    for (;;) {
      if ((previous & CheckpointMutationRecord::kDeleted) != 0)
        return;
      if (record.state.compare_exchange_weak(
              previous,
              std::max(reserved_sequence, previous) |
                  CheckpointMutationRecord::kDeleted,
              std::memory_order_acq_rel, std::memory_order_acquire)) {
        admit_checkpoint_record(record);
        return;
      }
    }
  }

  class InvocationGate {
    static_assert(std::atomic<size_t>::is_always_lock_free);
    static constexpr size_t kClosed =
        size_t{1} << (std::numeric_limits<size_t>::digits - 1);
    static constexpr size_t kReconfiguring = kClosed >> 1;
    static constexpr size_t kStateMask = kClosed | kReconfiguring;
    static constexpr size_t kActiveMask = ~kStateMask;
    std::atomic<size_t> admission_{0};

  public:
    enum class State : uint8_t { kAccepting, kReconfiguring, kClosed };

    State state() const noexcept {
      const size_t value = admission_.load(std::memory_order_acquire);
      if ((value & kClosed) != 0)
        return State::kClosed;
      return (value & kReconfiguring) != 0 ? State::kReconfiguring
                                           : State::kAccepting;
    }

    size_t enter() {
      size_t observed = admission_.load(std::memory_order_acquire);
      for (;;) {
        if ((observed & kClosed) != 0)
          throw std::runtime_error("store is not accepting invocations");
        if ((observed & kReconfiguring) != 0) {
          admission_.wait(observed, std::memory_order_acquire);
          observed = admission_.load(std::memory_order_acquire);
          continue;
        }
        if (observed == kActiveMask)
          throw std::overflow_error("active invocation count overflows");
        if (admission_.compare_exchange_weak(observed, observed + 1,
                                             std::memory_order_acq_rel,
                                             std::memory_order_acquire))
          return observed + 1;
      }
    }

    void leave() noexcept {
      const size_t previous =
          admission_.fetch_sub(1, std::memory_order_release);
      if ((previous & kActiveMask) == 0)
        std::terminate();
      if ((previous & kActiveMask) == 1)
        admission_.notify_all();
    }

    void wait_until_inactive() noexcept {
      size_t observed = admission_.load(std::memory_order_acquire);
      while ((observed & kActiveMask) != 0) {
        admission_.wait(observed, std::memory_order_acquire);
        observed = admission_.load(std::memory_order_acquire);
      }
    }

    void begin_reconfiguration_and_wait() {
      size_t observed = admission_.load(std::memory_order_acquire);
      for (;;) {
        if ((observed & kClosed) != 0)
          throw std::runtime_error("store is in a terminal failure state");
        if ((observed & kReconfiguring) != 0)
          throw std::logic_error("store reconfiguration is already active");
        if (admission_.compare_exchange_weak(
                observed, observed | kReconfiguring, std::memory_order_acq_rel,
                std::memory_order_acquire))
          break;
      }
      wait_until_inactive();
    }

    void require_quiescence() {
      if (state() != State::kReconfiguring)
        throw std::logic_error(
            "index import requires reconfiguration admission");
      wait_until_inactive();
    }

    void resume_accepting() noexcept {
      size_t expected = kReconfiguring;
      admission_.compare_exchange_strong(expected, 0, std::memory_order_acq_rel,
                                         std::memory_order_acquire);
      admission_.notify_all();
    }

    void close_without_wait() noexcept {
      admission_.fetch_or(kClosed, std::memory_order_acq_rel);
      admission_.notify_all();
    }

    void close_and_wait() noexcept {
      close_without_wait();
      wait_until_inactive();
    }
  };

  class ActiveInvocationGuard {
  public:
    explicit ActiveInvocationGuard(InvocationGate &gate,
                                   const PageStorage *storage = nullptr)
        : gate_(&gate),
          storage_(storage != nullptr &&
                           (*storage)->supports_background_page_eviction()
                       ? storage
                       : nullptr),
          count_(gate_->enter()) {
      try {
        if (storage_ != nullptr)
          (*storage_)->begin_read();
      } catch (...) {
        gate_->leave();
        throw;
      }
    }

    ~ActiveInvocationGuard() {
      if (storage_ != nullptr)
        (*storage_)->end_read();
      gate_->leave();
    }

    ActiveInvocationGuard(const ActiveInvocationGuard &) = delete;
    ActiveInvocationGuard &operator=(const ActiveInvocationGuard &) = delete;

    size_t count() const noexcept { return count_; }

  private:
    InvocationGate *gate_;
    const PageStorage *storage_;
    size_t count_;
  };

  void ensure_operational() const {
    if (invocation_gate_.state() == InvocationGate::State::kClosed)
      throw std::runtime_error("store is in a terminal failure state");
  }

public:
  // Marks the store into a non-recoverable terminal failure state (fail-stop
  // circuit breaker). Transitions the InvocationGate state to State::kClosed
  // without waiting for active readers/writers. Notifies all waiting threads
  // and guarantees that any subsequent read, write, or control-plane operation
  // immediately aborts with std::runtime_error. This prevents tainted or
  // partially recovered stores from being silently operated upon.
  void mark_terminal_failure() noexcept {
    invocation_gate_.close_without_wait();
  }

  // Non-blocking and noexcept probe querying whether the store is healthy and
  // accepting operations. Returns false if mark_terminal_failure() was
  // previously invoked or the invocation gate is closed.
  bool is_operational() const noexcept {
    return invocation_gate_.state() != InvocationGate::State::kClosed;
  }

  class PageReuseState {
  public:
    enum class State : uint8_t {
      kAvailable,
      kDraining,
      kRetired,
    };

    struct alignas(EmbeddingArena<ParamT>::kCacheLineSize) PageState {
      std::atomic<uint8_t> state;

      PageState() : state(static_cast<uint8_t>(State::kAvailable)) {}
    };

    explicit PageReuseState(size_t max_pages = 8192,
                            size_t entries_per_page = 4096)
        : max_pages_(max_pages), entries_shift_(static_cast<unsigned>(
                                     std::countr_zero(entries_per_page))),
          page_states_(std::make_unique<PageState[]>(max_pages)) {}

    bool unavailable(uint32_t index) const noexcept {
      return page_states_[index >> entries_shift_].state.load(
                 std::memory_order_acquire) !=
             static_cast<uint8_t>(State::kAvailable);
    }

    bool destination_available(uint32_t index) const noexcept {
      const size_t page_id = index >> entries_shift_;
      return page_states_[page_id].state.load(std::memory_order_acquire) ==
             static_cast<uint8_t>(State::kAvailable);
    }

    void begin_draining(uint64_t page_id) {
      if (page_id >= max_pages_)
        throw std::out_of_range("arena page drain id is out of range");
      uint8_t expected = static_cast<uint8_t>(State::kAvailable);
      if (page_states_[page_id].state.compare_exchange_strong(
              expected, static_cast<uint8_t>(State::kDraining),
              std::memory_order_acq_rel, std::memory_order_acquire))
        return;
      if (expected == static_cast<uint8_t>(State::kRetired))
        throw std::logic_error("arena page was already retired");
      throw std::logic_error("arena page is already draining");
    }

    void cancel_draining(uint64_t page_id) noexcept {
      uint8_t expected = static_cast<uint8_t>(State::kDraining);
      if (page_states_[page_id].state.compare_exchange_strong(
              expected, static_cast<uint8_t>(State::kAvailable),
              std::memory_order_acq_rel, std::memory_order_acquire))
        return;
      if (expected != static_cast<uint8_t>(State::kRetired))
        std::terminate();
    }

    void retire(uint64_t page_id) {
      uint8_t expected = static_cast<uint8_t>(State::kDraining);
      if (!page_states_[page_id].state.compare_exchange_strong(
              expected, static_cast<uint8_t>(State::kRetired),
              std::memory_order_acq_rel, std::memory_order_acquire))
        throw std::logic_error("arena page was not draining before retirement");
    }

    void make_available(uint64_t page_id) noexcept {
      uint8_t expected = static_cast<uint8_t>(State::kRetired);
      if (!page_states_[page_id].state.compare_exchange_strong(
              expected, static_cast<uint8_t>(State::kAvailable),
              std::memory_order_release, std::memory_order_acquire) &&
          expected != static_cast<uint8_t>(State::kAvailable))
        std::terminate();
    }

    void reset() noexcept {
      for (size_t page = 0; page < max_pages_; ++page) {
        page_states_[page].state.store(static_cast<uint8_t>(State::kAvailable),
                                       std::memory_order_relaxed);
      }
    }

  private:
    size_t max_pages_;
    unsigned entries_shift_;
    std::unique_ptr<PageState[]> page_states_;
  };

  template <typename K> struct SplitMixHash {
    size_t operator()(K x) const {
      uint64_t z = static_cast<uint64_t>(x);
      z ^= z >> 30;
      z *= 0xbf58476d1ce4e5b9ull;
      z ^= z >> 27;
      z *= 0x94d049bb133111ebull;
      z ^= z >> 31;
      return z;
    }

#if defined(__AVX512F__) && defined(__AVX512DQ__)
    __m512i simd_hash(__m512i x) const {
      x = _mm512_xor_si512(x, _mm512_srli_epi64(x, 30));
      __m512i c1 = _mm512_set1_epi64(0xbf58476d1ce4e5b9ull);
      x = _mm512_mullo_epi64(x, c1);
      x = _mm512_xor_si512(x, _mm512_srli_epi64(x, 27));
      __m512i c2 = _mm512_set1_epi64(0x94d049bb133111ebull);
      x = _mm512_mullo_epi64(x, c2);
      x = _mm512_xor_si512(x, _mm512_srli_epi64(x, 31));
      return x;
    }
#endif
  };

  struct KeyShard {
    static constexpr size_t kInitialRecycleCapacity = 64;

    struct MapBundle {
      explicit MapBundle(
          size_t key_capacity, unsigned hash_shift,
          const hogwild_internal::SegmentedLocatorTable *locator_table)
          : directory(hogwild_internal::FlatKeyDirectory::
                          bucket_count_for_key_capacity(key_capacity),
                      hash_shift),
            locators(locator_table) {
        if (locators == nullptr)
          throw std::invalid_argument("key directory locator table is null");
      }

      MapBundle(const MapBundle &other)
          : directory(other.directory), locators(other.locators) {}

      std::optional<uint32_t> find(uint64_t key) const noexcept {
        return find_hashed(key, SplitMixHash<uint64_t>{}(key));
      }

      void prefetch_bucket_cacheline(size_t hash) const noexcept {
        directory.prefetch_bucket_cacheline(hash);
      }

      std::optional<uint32_t> find_hashed(uint64_t key,
                                          size_t hash) const noexcept {
        return directory.find_hashed(key, hash, [&](uint32_t slot_id) noexcept {
          return locators->key(slot_id);
        });
      }

      void insert(uint64_t key, uint32_t slot_id) {
        directory.insert_hashed(key, SplitMixHash<uint64_t>{}(key), slot_id,
                                [&](uint32_t existing_slot_id) noexcept {
                                  return locators->key(existing_slot_id);
                                });
      }

      std::optional<uint32_t> remove(uint64_t key) {
        return directory.remove_hashed(
            key, SplitMixHash<uint64_t>{}(key),
            [&](uint32_t slot_id) noexcept { return locators->key(slot_id); });
      }

      template <typename Fn> void for_each(Fn &&fn) const {
        directory.for_each([&](uint32_t slot_id) {
          std::forward<Fn>(fn)(locators->key(slot_id), slot_id);
        });
      }

      size_t retirement_bytes() const {
        return checked_bytes_add(sizeof(MapBundle),
                                 directory.allocation_bytes());
      }

      size_t directory_capacity() const noexcept {
        return directory.bucket_count();
      }

    private:
      static_assert(hogwild_internal::SegmentedLocatorTable::kMaximumSlotId <=
                    hogwild_internal::FlatKeyDirectory::kMaximumSlotId);

      static size_t checked_bytes_add(size_t total, size_t increment) {
        if (increment > std::numeric_limits<size_t>::max() - total)
          throw std::overflow_error("key directory byte count overflow");
        return total + increment;
      }

      hogwild_internal::FlatKeyDirectory directory;
      const hogwild_internal::SegmentedLocatorTable *locators;
    };

    struct RecycleState {
      std::mutex mutex;
      std::vector<uint32_t> ready_indices;
      std::atomic<size_t> ready_count{0};
      std::atomic<size_t> reserved_capacity{0};
      std::shared_ptr<PageReuseState> page_reuse;

      explicit RecycleState(std::shared_ptr<PageReuseState> reuse)
          : page_reuse(std::move(reuse)) {}

      void release(uint32_t index) noexcept {
        if (page_reuse->unavailable(index))
          return;
        std::lock_guard lock(mutex);
        if (page_reuse->unavailable(index))
          return;
        if (ready_indices.size() == ready_indices.capacity())
          std::terminate();
        ready_indices.push_back(index);
        ready_count.store(ready_indices.size(), std::memory_order_release);
      }
    };

    struct DeferredRecycle {
      std::shared_ptr<RecycleState> state;
      uint32_t index;

      DeferredRecycle(std::shared_ptr<RecycleState> recycle_state,
                      uint32_t recycle_index)
          : state(std::move(recycle_state)), index(recycle_index) {}

      ~DeferredRecycle() noexcept { state->release(index); }
    };

    struct DeferredSlotRecycle {
      std::shared_ptr<hogwild_internal::SegmentedLocatorTable> table;
      uint32_t slot_id;

      DeferredSlotRecycle(
          std::shared_ptr<hogwild_internal::SegmentedLocatorTable>
              locator_table,
          uint32_t retired_slot_id)
          : table(std::move(locator_table)), slot_id(retired_slot_id) {}

      ~DeferredSlotRecycle() noexcept { table->release_after_epoch(slot_id); }
    };

    struct DeferredErasedBatchRecycle {
      struct Entry {
        uint32_t slot_id;
        uint32_t index = std::numeric_limits<uint32_t>::max();
      };

      std::shared_ptr<RecycleState> state;
      std::shared_ptr<hogwild_internal::SegmentedLocatorTable> table;
      std::vector<Entry> entries;

      DeferredErasedBatchRecycle(
          std::shared_ptr<RecycleState> recycle_state,
          std::shared_ptr<hogwild_internal::SegmentedLocatorTable>
              locator_table,
          std::span<const uint32_t> retired_slot_ids)
          : state(std::move(recycle_state)), table(std::move(locator_table)) {
        entries.reserve(retired_slot_ids.size());
        for (uint32_t slot_id : retired_slot_ids)
          entries.push_back({slot_id});
      }

      ~DeferredErasedBatchRecycle() noexcept {
        for (const Entry entry : entries) {
          if (entry.index == std::numeric_limits<uint32_t>::max())
            std::terminate();
          state->release(entry.index);
          table->recycle_retired_after_epoch(entry.slot_id);
        }
      }
    };

    struct DeferredBundle {
      MapBundle *bundle;
      explicit DeferredBundle(MapBundle *retired) : bundle(retired) {}
      ~DeferredBundle() { delete bundle; }
    };

    struct alignas(EmbeddingArena<ParamT>::kCacheLineSize) ShardReadView {
      std::atomic<MapBundle *> bundle;
      hogwild_internal::SegmentedLocatorTable *locators;
      uint64_t table_identity;
      std::atomic<uint64_t> generation{1};

      ShardReadView(MapBundle *initial,
                    hogwild_internal::SegmentedLocatorTable *locator_table,
                    uint64_t identity)
          : bundle(initial), locators(locator_table), table_identity(identity) {
      }

      void prefetch_bucket_cacheline(size_t hash) const noexcept {
        const MapBundle *b = bundle.load(std::memory_order_relaxed);
        if (b != nullptr) {
          b->prefetch_bucket_cacheline(hash);
        }
      }
    };

    size_t reserved_keys;
    unsigned hash_shift_;
    std::unique_ptr<ShardReadView> read_view;
    std::atomic<size_t> count{0};
    std::mutex writer_mutex;
    std::shared_ptr<RecycleState> recycle_state;
    std::shared_ptr<RetirementPressure> retirement_pressure;
    size_t index_retirement_bytes;
    std::atomic<size_t> owned_index_count{0};
    std::shared_ptr<hogwild_internal::SegmentedLocatorTable> locators;
    HogwildEmbeddingStoreCore &checkpoint_owner;
    uint32_t dimension;

    explicit KeyShard(size_t requested_keys,
                      std::shared_ptr<PageReuseState> page_reuse,
                      std::shared_ptr<RetirementPressure> pressure,
                      size_t retired_index_bytes, unsigned hash_shift,
                      uint32_t dimension_in,
                      HogwildEmbeddingStoreCore &checkpoint_owner_in)
        : reserved_keys(requested_keys), hash_shift_(hash_shift),
          recycle_state(std::make_shared<RecycleState>(std::move(page_reuse))),
          retirement_pressure(std::move(pressure)),
          index_retirement_bytes(retired_index_bytes),
          locators(std::make_shared<hogwild_internal::SegmentedLocatorTable>()),
          checkpoint_owner(checkpoint_owner_in), dimension(dimension_in) {
      if (!retirement_pressure)
        throw std::invalid_argument("key shard retirement pressure is null");
      auto initial = std::make_unique<MapBundle>(requested_keys, hash_shift_,
                                                 locators.get());
      read_view = std::make_unique<ShardReadView>(
          initial.release(), locators.get(), reinterpret_cast<uintptr_t>(this));
    }

    ~KeyShard() { delete read_view->bundle.load(std::memory_order_relaxed); }

    MapBundle *writer_bundle_unlocked() const noexcept {
      return read_view->bundle.load(std::memory_order_acquire);
    }

    static size_t checked_bundle_bytes_add(size_t total, size_t increment) {
      if (increment > std::numeric_limits<size_t>::max() - total)
        throw std::overflow_error("map bundle retirement bytes overflow");
      return total + increment;
    }

    static size_t checked_bundle_bytes_product(size_t count, size_t element) {
      if (count != 0 && element > std::numeric_limits<size_t>::max() / count)
        throw std::overflow_error("map bundle retirement bytes overflow");
      return count * element;
    }

    static size_t bundle_retirement_bytes(const MapBundle &map_bundle) {
      return map_bundle.retirement_bytes();
    }

    void publish_bundle_unlocked(std::unique_ptr<MapBundle> next) {
      MapBundle *current = writer_bundle_unlocked();
      se::concurrency::EpochRetirement<DeferredBundle> retirement(
          retirement_pressure, bundle_retirement_bytes(*current));
      MapBundle *retired =
          read_view->bundle.exchange(next.release(), std::memory_order_acq_rel);
      retirement.emplace(retired);
      retirement.publish();
      read_view->generation.fetch_add(1, std::memory_order_acq_rel);
    }

    std::unique_ptr<MapBundle> clone_bundle_unlocked(size_t capacity) const {
      MapBundle *current = writer_bundle_unlocked();
      if (capacity == reserved_keys) {
        auto next = std::make_unique<MapBundle>(*current);
        return next;
      }
      auto next =
          std::make_unique<MapBundle>(capacity, hash_shift_, locators.get());
      current->for_each(
          [&](uint64_t key, uint32_t slot_id) { next->insert(key, slot_id); });
      return next;
    }

    static size_t compute_next_capacity(size_t current, size_t required) {
      size_t next = current;
      while (next < required) {
        if (next < 65536) { // < 64K, use 2x
          next = std::max(static_cast<size_t>(1024), next * 2);
        } else if (next < 1048576) { // < 1M, use 1.5x
          next += next / 2;
        } else { // >= 1M, use 1.25x
          next += next / 4;
        }
      }
      return next;
    }

    struct LocatorRef {
      uint32_t slot_id;
      uint32_t index;
    };

    std::optional<uint32_t> find_unlocked(uint64_t key) {
      auto slot_id = find_slot_unlocked(key);
      return slot_id ? locators->load(*slot_id) : std::nullopt;
    }

    std::optional<uint32_t> find_slot_unlocked(uint64_t key) const {
      return writer_bundle_unlocked()->find(key);
    }

    std::optional<uint32_t> find_slot_hashed_unlocked(uint64_t key,
                                                      size_t hash) const {
      return writer_bundle_unlocked()->find_hashed(key, hash);
    }

    std::optional<LocatorRef> find_locator_unlocked(uint64_t key) {
      auto slot_id = find_slot_unlocked(key);
      if (!slot_id)
        return std::nullopt;
      auto index = locators->load(*slot_id);
      if (!index)
        return std::nullopt;
      return LocatorRef{*slot_id, *index};
    }

    std::optional<uint32_t> find(uint64_t key) { return find_unlocked(key); }
    std::optional<uint32_t> find(uint64_t key) const {
      const auto slot_id =
          read_view->bundle.load(std::memory_order_acquire)->find(key);
      return slot_id ? locators->load(*slot_id) : std::nullopt;
    }

    template <typename Fn> void for_each_entry(Fn &&fn) const {
      writer_bundle_unlocked()->for_each(std::forward<Fn>(fn));
    }

    void expand(size_t target_keys) {
      if (target_keys <= reserved_keys)
        return;
      auto next = clone_bundle_unlocked(target_keys);
      publish_bundle_unlocked(std::move(next));
      reserved_keys = target_keys;
    }

    void reserve_new_indices(size_t count) {
      if (count == 0)
        throw std::invalid_argument("key shard reserve count is zero");
      size_t owned = owned_index_count.load(std::memory_order_acquire);
      while (true) {
        if (count > std::numeric_limits<size_t>::max() - owned)
          throw std::overflow_error("key shard index count overflow");
        const size_t required_capacity = owned + count;
        if (required_capacity <=
            recycle_state->reserved_capacity.load(std::memory_order_acquire)) {
          if (owned_index_count.compare_exchange_weak(
                  owned, required_capacity, std::memory_order_acq_rel,
                  std::memory_order_acquire)) {
            return;
          }
          continue;
        }
        std::lock_guard lock(recycle_state->mutex);
        while (true) {
          owned = owned_index_count.load(std::memory_order_acquire);
          if (count > std::numeric_limits<size_t>::max() - owned)
            throw std::overflow_error("key shard index count overflow");
          const size_t locked_required_capacity = owned + count;
          if (locked_required_capacity > recycle_state->reserved_capacity.load(
                                             std::memory_order_acquire)) {
            const size_t next_capacity =
                compute_next_capacity(recycle_state->ready_indices.capacity(),
                                      locked_required_capacity);
            recycle_state->ready_indices.reserve(next_capacity);
            recycle_state->reserved_capacity.store(
                recycle_state->ready_indices.capacity(),
                std::memory_order_release);
          }
          if (owned_index_count.compare_exchange_weak(
                  owned, locked_required_capacity, std::memory_order_acq_rel,
                  std::memory_order_acquire)) {
            return;
          }
        }
      }
    }

    void reserve_new_index() { reserve_new_indices(1); }

    std::optional<uint32_t> take_recycled_index() {
      if (recycle_state->ready_count.load(std::memory_order_acquire) == 0)
        return std::nullopt;
      std::lock_guard lock(recycle_state->mutex);
      while (!recycle_state->ready_indices.empty()) {
        uint32_t index = recycle_state->ready_indices.back();
        recycle_state->ready_indices.pop_back();
        if (recycle_state->page_reuse->unavailable(index))
          continue;
        recycle_state->ready_count.store(recycle_state->ready_indices.size(),
                                         std::memory_order_release);
        return index;
      }
      recycle_state->ready_count.store(0, std::memory_order_release);
      return std::nullopt;
    }

    void recycle_unpublished_index(uint32_t index) noexcept {
      recycle_state->release(index);
    }

    void retire_published_index(uint32_t index) {
      se::concurrency::EpochRetirement<DeferredRecycle> retirement(
          retirement_pressure, index_retirement_bytes);
      retirement.emplace(recycle_state, index);
      retirement.publish();
    }

    se::concurrency::EpochRetirement<DeferredSlotRecycle>
    prepare_slot_retirement(uint32_t slot_id) {
      return {retirement_pressure, sizeof(hogwild_internal::LocatorSlot)};
    }

    LocatorRef rotate_locator_unlocked(uint64_t key,
                                       uint32_t expected_slot_id) {
      auto current = find_locator_unlocked(key);
      if (!current.has_value())
        throw std::logic_error("key disappeared before locator rotation");
      if (current->slot_id != expected_slot_id)
        return *current;
      if (!locators->requires_rotation(expected_slot_id))
        return *current;

      auto next = clone_bundle_unlocked(reserved_keys);
      const auto removed = next->remove(key);
      if (!removed || *removed != expected_slot_id)
        throw std::logic_error(
            "locator rotation rebuilt the wrong key mapping");
      const uint32_t replacement_slot_id = locators->allocate(
          key, current->index, locators->checkpoint_record(expected_slot_id));
      next->insert(key, replacement_slot_id);

      auto retirement = prepare_slot_retirement(expected_slot_id);
      publish_bundle_unlocked(std::move(next));
      retirement.emplace(locators, expected_slot_id);
      retirement.publish();
      return LocatorRef{replacement_slot_id, current->index};
    }

    void insert_index_unlocked(uint64_t key, uint32_t idx) {
      const std::array<uint64_t, 1> keys{key};
      const std::array<uint32_t, 1> indices{idx};
      publish_indices_batch_unlocked(keys, indices);
    }

    enum class IndexPublication { kConcurrent, kQuiesced };

    void publish_indices_batch_unlocked(std::span<const uint64_t> keys,
                                        std::span<const uint32_t> indices) {
      update_indices_batch_unlocked<IndexPublication::kConcurrent>(keys,
                                                                   indices);
    }

    void import_indices_quiesced(std::span<const uint64_t> keys,
                                 std::span<const uint32_t> indices) {
      update_indices_batch_unlocked<IndexPublication::kQuiesced>(keys, indices);
    }

    template <IndexPublication Publication>
    void update_indices_batch_unlocked(std::span<const uint64_t> keys,
                                       std::span<const uint32_t> indices) {
      if (keys.empty() || keys.size() != indices.size())
        throw std::invalid_argument("key shard publication batch is invalid");
      const size_t current_count = count.load(std::memory_order_acquire);
      if (keys.size() > std::numeric_limits<size_t>::max() - current_count)
        throw std::overflow_error("key shard publication count overflows");
      const size_t next_count = current_count + keys.size();
      const size_t next_reserved =
          next_count > reserved_keys
              ? compute_next_capacity(reserved_keys, next_count)
              : reserved_keys;
      if constexpr (Publication == IndexPublication::kQuiesced)
        checkpoint_owner.invocation_gate_.require_quiescence();
      std::unique_ptr<MapBundle> replacement;
      MapBundle *next;
      if constexpr (Publication == IndexPublication::kQuiesced) {
        if (next_reserved != reserved_keys)
          expand(next_reserved);
        next = writer_bundle_unlocked();
      } else {
        replacement = clone_bundle_unlocked(next_reserved);
        next = replacement.get();
      }
      std::vector<CheckpointMutationRecord *> records(keys.size(), nullptr);
      std::vector<uint32_t> allocated_slots(keys.size(), 0);
      size_t linked_count = 0;
      try {
        for (size_t index = 0; index < keys.size(); ++index) {
          const uint64_t key = keys[index];
          if (next->find(key).has_value())
            throw std::invalid_argument(
                "key shard batch contains duplicate key");
          records[index] =
              checkpoint_owner.allocate_checkpoint_record(dimension, key);
          const uint32_t slot_id = locators->allocate(
              key, indices[index],
              hogwild_internal::CheckpointRecordRef::from(records[index]));
          allocated_slots[index] = slot_id;
          next->insert(key, slot_id);
          ++linked_count;
        }
        if constexpr (Publication == IndexPublication::kConcurrent)
          publish_bundle_unlocked(std::move(replacement));
        else
          read_view->generation.fetch_add(1, std::memory_order_acq_rel);
      } catch (...) {
        size_t first_unpublished = 0;
        if constexpr (Publication == IndexPublication::kQuiesced) {
          checkpoint_owner.mark_terminal_failure();
          first_unpublished = linked_count;
        }
        for (size_t index = first_unpublished; index < records.size();
             ++index) {
          if (records[index] != nullptr)
            checkpoint_owner.discard_unpublished_checkpoint_record(
                records[index]);
          if (allocated_slots[index] != 0)
            locators->release_after_epoch(allocated_slots[index]);
        }
        throw;
      }
      reserved_keys = next_reserved;
      count.store(next_count, std::memory_order_release);
    }

    template <typename OnRetire>
    size_t erase_indices_unlocked(std::span<const uint64_t> keys,
                                  OnRetire &&on_retire) {
      static_assert(
          std::is_nothrow_invocable_v<OnRetire, uint64_t, uint32_t,
                                      hogwild_internal::CheckpointRecordRef,
                                      uint64_t>);
      if (keys.empty())
        return 0;
      for (uint64_t key : keys) {
        auto slot_id = find_slot_unlocked(key);
        if (slot_id && locators->requires_rotation(*slot_id))
          rotate_locator_unlocked(key, *slot_id);
      }
      MapBundle *current = writer_bundle_unlocked();
      auto next = clone_bundle_unlocked(reserved_keys);
      struct Removed {
        uint64_t key;
        uint32_t slot_id;
      };
      std::vector<Removed> removed;
      removed.reserve(keys.size());
      for (uint64_t key : keys) {
        auto slot_id = current->find(key);
        if (!slot_id)
          continue;
        auto erased = next->remove(key);
        if (!erased)
          continue;
        if (*erased != *slot_id)
          throw std::logic_error(
              "rebuilt directory returned the wrong locator slot");
        removed.push_back({key, *slot_id});
      }
      if (removed.empty())
        return 0;
      const uint64_t deletion_sequence_begin =
          checkpoint_owner.reserve_checkpoint_sequences(removed.size());
      const size_t bytes_per_entry = checked_bundle_bytes_add(
          index_retirement_bytes, sizeof(hogwild_internal::LocatorSlot));
      const size_t retirement_bytes =
          checked_bundle_bytes_product(removed.size(), bytes_per_entry);
      std::vector<uint32_t> retired_slot_ids;
      retired_slot_ids.reserve(removed.size());
      for (const Removed entry : removed)
        retired_slot_ids.push_back(entry.slot_id);
      se::concurrency::EpochRetirement<DeferredErasedBatchRecycle> retirement(
          retirement_pressure, retirement_bytes);
      DeferredErasedBatchRecycle &retired_batch = retirement.emplace(
          recycle_state, locators, std::span<const uint32_t>(retired_slot_ids));
      publish_bundle_unlocked(std::move(next));
      for (size_t removed_index = 0; removed_index < removed.size();
           ++removed_index) {
        const Removed entry = removed[removed_index];
        const bool retired = locators->retire_current(
            entry.slot_id,
            [&](uint32_t index,
                hogwild_internal::CheckpointRecordRef record) noexcept {
              const uint64_t deletion_sequence =
                  deletion_sequence_begin + removed_index;
              on_retire(entry.key, index, record, deletion_sequence);
              retired_batch.entries[removed_index].index = index;
            });
        if (!retired)
          std::terminate();
      }
      retirement.publish();
      count.fetch_sub(removed.size(), std::memory_order_release);
      return removed.size();
    }
  };

  struct SlotStore {
    static constexpr size_t kInitialKeysPerShard = 1;

    struct PendingKey {
      uint64_t key;
      uint64_t seq_count;
      uint32_t source_index;
      uint32_t shard_id;
      uint32_t source_page_position;
      uint8_t cluster;
    };

    struct CompactionWorkspace {
      std::vector<uint64_t> sorted_page_ids;
      std::vector<PendingKey> pending;
      std::vector<uint64_t> source_clear_masks;
      std::vector<uint32_t> source_cohort_shards;
      std::vector<size_t> shard_counts;
      std::vector<uint32_t> destination_indices;
      std::vector<uint8_t> destination_pending;
      std::vector<OptT> optimizer_state;
      std::vector<size_t> migration_order;
      std::vector<size_t> shard_offsets;
      std::vector<size_t> shard_positions;
      std::vector<uint8_t> pending_shards;
      std::vector<size_t> next_shard_order;
    };

    size_t dim;
    ScorePolicyRuntime &score_policy;
    std::atomic<uint64_t> lookup_generation{1};
    std::shared_ptr<PageReuseState> page_reuse;
    std::shared_ptr<RetirementPressure> retirement_pressure;
    std::vector<std::unique_ptr<KeyShard>> key_shards;
    size_t shard_mask;
    TelemetryRing &telemetry_ring;
    std::shared_ptr<PageResidencyTelemetry> page_telemetry;
    StripedPageAccessTelemetry access_telemetry;
    EmbeddingArena<ParamT> arena;
    SealedHotPagePool<ParamT> sealed_pages;
    OptimizerExtension<ParamT, OptT> optimizer;
    std::atomic<uint64_t> optimizer_locator_revalidation_failures{0};
    std::vector<std::unique_ptr<GradientAccumulator<ParamT>>> transient_shards;
    OptimizerFactory opt_factory;
    std::atomic<uint64_t> global_step{0};
    std::atomic<uint64_t> sealed_page_layout_generation{0};
    std::shared_ptr<const SealedHotPage<ParamT>> last_promoted_hot_page;
    mutable std::mutex promoted_hot_page_mutex;
    execution::ReusableWorkspacePool<CompactionWorkspace> compaction_workspaces;
    HogwildEmbeddingStoreCore &checkpoint_owner;

    struct OptimizerBatchScratch {
      std::vector<hogwild_internal::SegmentedLocatorTable::KeyedWriteSlot>
          locator_entries;
      std::vector<OptimizerBatchRow<ParamT>> rows;
      std::vector<size_t> order;
      std::vector<size_t> positions;
    };

    static OptimizerBatchScratch &optimizer_batch_scratch() {
      static thread_local OptimizerBatchScratch scratch;
      return scratch;
    }

    void handle_optimizer_locator_revalidation_failure(uint64_t key) {
#ifndef NDEBUG
      throw std::runtime_error("optimizer locator batch became invalid key=" +
                               std::to_string(key));
#else
      optimizer_locator_revalidation_failures.fetch_add(
          1, std::memory_order_relaxed);
      telemetry_ring.try_publish_diagnostic(
          {DiagnosticKind::kOptimizerLocatorRevalidation, key});
#endif
    }

    static void register_arena_page(void *context, uint64_t page_id) {
      if (page_id > std::numeric_limits<uint32_t>::max())
        throw std::overflow_error("arena page telemetry id overflows uint32");
      static_cast<PageResidencyTelemetry *>(context)->register_resident(
          static_cast<uint32_t>(page_id));
    }

    size_t index_retirement_bytes() const {
      size_t bytes = arena.data_stride();
      if (score_policy.enabled())
        bytes += sizeof(uint64_t);
      if (optimizer.has_value())
        bytes += optimizer->allocation_bytes_per_entry();
      return bytes;
    }

    SlotStore(size_t d, ScorePolicyRuntime &score_policy_in,
              OptimizerFactory opt_factory_in,
              const storage::StorageTopology &topology,
              size_t transient_shard_count, bool enable_score,
              PageStorage storage,
              std::shared_ptr<RetirementPressure> retirement_pressure,
              TelemetryRing &telemetry_ring_in, GradientDType gradient_dtype,
              float gradient_scale,
              HogwildEmbeddingStoreCore &checkpoint_owner_in)
        : dim(d), score_policy(score_policy_in),
          page_reuse(std::make_shared<PageReuseState>(
              topology.max_pages(), topology.entries_per_page())),
          retirement_pressure(std::move(retirement_pressure)),
          shard_mask(topology.shard_mask()), telemetry_ring(telemetry_ring_in),
          page_telemetry(
              std::make_shared<PageResidencyTelemetry>(topology.max_pages())),
          access_telemetry(topology.max_pages(), topology.entries_per_page(),
                           transient_shard_count),
          arena(d, enable_score, storage, this->retirement_pressure,
                &SlotStore::register_arena_page, page_telemetry.get(),
                topology),
          sealed_pages(topology.max_pages(), topology.entries_per_page(),
                       static_cast<uint32_t>(d)),
          opt_factory(std::move(opt_factory_in)),
          checkpoint_owner(checkpoint_owner_in) {
      if (opt_factory.has_value()) {
        optimizer = (*opt_factory)(d);
        if (optimizer->requires_score() && !enable_score) {
          throw std::invalid_argument(
              "dynamic learning-rate policy requires score storage");
        }
      }
      if (!this->retirement_pressure)
        throw std::invalid_argument("slot retirement pressure is null");
      if (optimizer.has_value())
        optimizer->set_page_eviction_enabled(enable_score);
      const size_t initial_keys_per_shard =
          topology.target_capacity_keys() > 0
              ? std::max(size_t{1}, (topology.target_capacity_keys() +
                                     topology.shard_count() - 1) /
                                        topology.shard_count())
              : kInitialKeysPerShard;
      const size_t initial_with_headroom =
          topology.target_capacity_keys() > 0
              ? std::max(initial_keys_per_shard,
                         (initial_keys_per_shard * 5 + 3) / 4)
              : initial_keys_per_shard;
      for (size_t i = 0; i < topology.shard_count(); ++i) {
        key_shards.push_back(std::make_unique<KeyShard>(
            initial_with_headroom, page_reuse, this->retirement_pressure,
            index_retirement_bytes(),
            static_cast<unsigned>(std::countr_zero(topology.shard_count())),
            static_cast<uint32_t>(d), checkpoint_owner_in));
      }
      for (size_t i = 0; i < transient_shard_count; ++i) {
        transient_shards.push_back(
            std::make_unique<GradientAccumulator<ParamT>>(
                d, storage, enable_score, gradient_dtype, gradient_scale));
      }
#if defined(PAGE_COMPACTION_SHARD_PROFILE)
      configure_page_compaction_shard_profile(key_shards.size());
#endif
    }

    SlotStore(size_t d, ScorePolicyRuntime &score_policy_in,
              OptimizerFactory opt_factory_in, size_t partition_count,
              size_t transient_shard_count, bool enable_score,
              PageStorage storage,
              std::shared_ptr<RetirementPressure> retirement_pressure,
              bool numa_aware_pages, TelemetryRing &telemetry_ring_in,
              GradientDType gradient_dtype, float gradient_scale,
              HogwildEmbeddingStoreCore &checkpoint_owner_in)
        : SlotStore(
              d, score_policy_in, std::move(opt_factory_in),
              storage::StorageTopology(
                  partition_count, partition_count, 4096, 8192,
                  numa_aware_pages
                      ? numa_page_cohort_size(partition_count, numa_aware_pages)
                      : std::min(size_t{16}, partition_count),
                  numa_aware_pages, numa_aware_pages),
              transient_shard_count, enable_score, std::move(storage),
              std::move(retirement_pressure), telemetry_ring_in, gradient_dtype,
              gradient_scale, checkpoint_owner_in) {}

  private:
    static size_t numa_page_cohort_size(size_t shard_count,
                                        bool numa_aware_pages) {
      if (shard_count == 0 || !std::has_single_bit(shard_count))
        throw std::invalid_argument("arena shard count is not a power of two");
      if (!numa_aware_pages)
        return std::min(size_t{16}, shard_count);
      if (!NumaTopology::available())
        throw std::runtime_error("NUMA-aware pages require libnuma support");
      const size_t nodes = NumaTopology::node_count();
      if (nodes == 0 || !std::has_single_bit(nodes) || shard_count < nodes ||
          shard_count % nodes != 0) {
        throw std::invalid_argument(
            "NUMA-aware pages require a power-of-two node count that divides "
            "the shard count");
      }
      return shard_count / nodes;
    }

  public:
    size_t shard_id_for_key(uint64_t key) const {
      return SplitMixHash<uint64_t>{}(key)&shard_mask;
    }

    size_t shard_id_for_hash(size_t hash) const { return hash & shard_mask; }

    KeyShard &shard_for_key(uint64_t key) {
      return *key_shards[shard_id_for_key(key)];
    }

    const KeyShard &shard_for_key(uint64_t key) const {
      return *key_shards[shard_id_for_key(key)];
    }

    std::optional<uint32_t> find(uint64_t key) {
      auto &shard = shard_for_key(key);
      return shard.find(key);
    }

    std::optional<uint64_t> page_id_for_key(uint64_t key) const {
      const auto &shard = shard_for_key(key);
      const auto slot_id = shard.find_slot_unlocked(key);
      if (!slot_id.has_value())
        return std::nullopt;
      std::optional<uint64_t> page_id;
      if (!shard.locators->try_read(*slot_id, [&](uint32_t index) {
            const auto owner = arena.owner_snapshot(index);
            if (owner.has_value() && owner->key == key &&
                owner->shard_id == shard_id_for_key(key))
              page_id = index >> arena.entries_shift();
          }))
        return std::nullopt;
      return page_id;
    }

    void observe_page_access(size_t worker, uint32_t index,
                             uint32_t weight) noexcept {
      access_telemetry.observe(worker, index >> arena.entries_shift(), weight);
    }

    void observe_coaccess(size_t worker, std::span<const uint64_t> keys,
                          uint32_t weight) noexcept {
      access_telemetry.observe_coaccess_batch(worker, keys, weight);
    }

    static void consume_telemetry(void *context,
                                  const TelemetryRing::Event &event) noexcept {
      auto *slot = static_cast<SlotStore *>(context);
      for (uint32_t index = 0; index < event.page_count; ++index)
        slot->access_telemetry.observe(event.stripe, event.page_ids[index],
                                       event.weight);
      if (event.key_count != 0)
        slot->access_telemetry.observe_coaccess_batch(
            event.stripe,
            std::span<const uint64_t>(event.keys.data(), event.key_count),
            event.weight);
    }

    struct StablePageSnapshot {
      uint64_t physical_page_id;
      uint64_t bytes;
      PageThermalSignal signal;
      std::vector<uint64_t> keys;
      std::vector<uint64_t> seq_counts;
      std::vector<ParamT> values;
    };

    std::optional<StablePageSnapshot> snapshot_page(uint64_t page_id) {
      if (page_id >= arena.page_count())
        throw std::out_of_range("arena page snapshot id is out of range");
      if (!arena.page_allocated(page_id))
        return std::nullopt;
      StablePageSnapshot snapshot{
          page_id,
          static_cast<uint64_t>(arena.entries_per_page() * arena.data_stride()),
          {static_cast<float>(
               access_telemetry.hits(static_cast<uint32_t>(page_id))),
           0.0F, page_telemetry->residency_age(static_cast<uint32_t>(page_id))},
          {},
          {},
          {}};
      snapshot.keys.reserve(arena.entries_per_page());
      snapshot.seq_counts.reserve(arena.entries_per_page());
      snapshot.values.reserve(arena.entries_per_page() * dim);
      const uint32_t first =
          static_cast<uint32_t>(page_id * arena.entries_per_page());
      const uint32_t limit = static_cast<uint32_t>(
          std::min(arena.allocated_count(),
                   static_cast<size_t>(first) + arena.entries_per_page()));
      for (uint32_t index = first; index < limit; ++index) {
        auto owner = arena.owner_snapshot(index);
        if (!owner.has_value())
          continue;
        bool copied = false;
        with_shared_entry(owner->key, [&](uint32_t current) {
          if (current != index)
            return;
          auto validated = arena.owner_snapshot(current);
          if (!validated.has_value() ||
              validated->seq_count != owner->seq_count ||
              validated->key != owner->key ||
              validated->shard_id != owner->shard_id)
            return;
          snapshot.keys.push_back(owner->key);
          snapshot.seq_counts.push_back(owner->seq_count);
          const ParamT *source = arena.data(current);
          snapshot.values.insert(snapshot.values.end(), source, source + dim);
          copied = true;
        });
        if (!copied)
          continue;
      }
      if (snapshot.keys.empty())
        return std::nullopt;
      std::array<uint64_t, CoaccessSketch::kMaximumBatchSample> sample{};
      const size_t sample_count = std::min(snapshot.keys.size(), sample.size());
      std::copy_n(snapshot.keys.begin(), sample_count, sample.begin());
      snapshot.signal.coaccess_density = access_telemetry.coaccess_density(
          std::span<const uint64_t>(sample.data(), sample_count));
      return snapshot;
    }

    struct PageSlotEntry {
      uint64_t key;
      uint64_t seq_count;
      uint32_t slot_id;
      uint32_t slot_ordinal;
    };
    static_assert(sizeof(PageSlotEntry) == 24);

    template <typename Visitor>
    bool export_page_images(Visitor &&visitor,
                            std::span<const uint64_t> requested_keys = {},
                            bool directed = false,
                            std::span<const uint64_t> requested_page_ids = {}) {
      const size_t optimizer_state_dim =
          optimizer.has_value() ? optimizer->state_dim() : 0;
      std::map<uint64_t, std::vector<PageSlotEntry>> page_entries;
      std::vector<uint64_t> selected_page_ids(requested_page_ids.begin(),
                                              requested_page_ids.end());
      const bool page_directed = !selected_page_ids.empty();
      if (page_directed) {
        std::ranges::sort(selected_page_ids);
        if (std::ranges::adjacent_find(selected_page_ids) !=
            selected_page_ids.end())
          throw std::invalid_argument(
              "checkpoint dirty page selection contains duplicates");
        for (const uint64_t page_id : selected_page_ids) {
          if (page_id >= arena.max_pages())
            throw std::out_of_range(
                "checkpoint dirty page selection is out of range");
          page_entries.try_emplace(page_id);
        }
      }

      auto collect_entry = [&](uint64_t key, uint32_t slot_id,
                               const KeyShard &key_shard) {
        std::optional<PageSlotEntry> candidate;
        uint64_t candidate_page_id = 0;
        const bool stable =
            key_shard.locators->try_read(slot_id, [&](uint32_t index) {
              candidate.reset();
              const auto owner = arena.owner_snapshot(index);
              if (!owner.has_value() || owner->key != key ||
                  owner->shard_id != shard_id_for_key(key))
                return;

              candidate = PageSlotEntry{
                  key,
                  owner->seq_count,
                  slot_id,
                  static_cast<uint32_t>(index & arena.slot_mask()),
              };
              candidate_page_id = index >> arena.entries_shift();
            });
        if (!stable || !candidate.has_value()) {
          if (directed)
            throw std::runtime_error(
                "checkpoint dirty key has no stable owner");
          return;
        }
        if (candidate_page_id >= arena.max_pages())
          throw std::out_of_range(
              "page-image export observed an invalid source page");
        page_entries[candidate_page_id].push_back(*candidate);
      };

      if (page_directed) {
        for (const uint64_t page_id : selected_page_ids) {
          const uint64_t first = page_id * arena.entries_per_page();
          const uint64_t limit = std::min<uint64_t>(
              arena.allocated_count(), first + arena.entries_per_page());
          for (uint64_t index = first; index < limit; ++index) {
            if (!arena.is_live(static_cast<uint32_t>(index)))
              continue;
            const auto owner =
                arena.owner_snapshot(static_cast<uint32_t>(index));
            if (!owner.has_value())
              continue;
            const auto &key_shard = shard_for_key(owner->key);
            const auto slot_id = key_shard.find_slot_unlocked(owner->key);
            if (!slot_id.has_value())
              throw std::logic_error(
                  "checkpoint dirty page owner is missing from the key shard");
            collect_entry(owner->key, *slot_id, key_shard);
          }
        }
      } else if (directed) {
        for (uint64_t key : requested_keys) {
          const auto &key_shard = shard_for_key(key);
          const auto slot_id = key_shard.find_slot_unlocked(key);
          if (!slot_id.has_value())
            throw std::runtime_error("checkpoint dirty key is missing");
          collect_entry(key, *slot_id, key_shard);
        }
      } else {
        for (const auto &key_shard : key_shards)
          key_shard->for_each_entry([&](uint64_t key, uint32_t slot_id) {
            collect_entry(key, slot_id, *key_shard);
          });
      }

      for (auto &[page_id, entries] : page_entries) {
        if (entries.empty()) {
          if (!page_directed)
            continue;
          PageImageExport<ParamT, OptT> deleted_page{
              page_id,
              static_cast<uint32_t>(dim),
              static_cast<uint32_t>(arena.data_stride()),
              {},
              {},
              {},
              {},
              {},
              {},
              {},
              true};
          if (!visitor(std::move(deleted_page)))
            return false;
          continue;
        }

        std::vector<ParamT> value_scratch(dim);
        std::vector<OptT> optimizer_scratch(optimizer_state_dim);
        uint64_t score_scratch = 0;
        uint32_t step_scratch = 0;
        PageImageExport<ParamT, OptT> image;
        while (true) {
          std::ranges::sort(entries, [](const auto &left, const auto &right) {
            return left.slot_ordinal < right.slot_ordinal;
          });
          if (std::ranges::adjacent_find(
                  entries, [](const auto &left, const auto &right) {
                    return left.slot_ordinal == right.slot_ordinal;
                  }) != entries.end())
            throw std::logic_error(
                "page-image export observed duplicate slot ordinals");

          image = PageImageExport<ParamT, OptT>{
              page_id,
              static_cast<uint32_t>(dim),
              static_cast<uint32_t>(arena.data_stride()),
              {},
              {},
              {},
              {},
              {},
              {},
              {}};
          image.slot_ordinals.reserve(entries.size());
          image.keys.reserve(entries.size());
          image.seq_counts.reserve(entries.size());
          image.values.reserve(entries.size() * dim);
          if (score_policy.enabled())
            image.scores.reserve(entries.size());
          if (optimizer_state_dim != 0) {
            image.optimizer_states.reserve(entries.size() *
                                           optimizer_state_dim);
            image.steps.reserve(entries.size());
          }

          bool restart = false;
          size_t survivor_count = 0;
          for (const auto &entry : entries) {
            const uint32_t expected_slot_ordinal = entry.slot_ordinal;
            bool captured = false;
            uint32_t resolved_index = static_cast<uint32_t>(
                page_id * arena.entries_per_page() + entry.slot_ordinal);
            uint32_t resolved_slot_ordinal = entry.slot_ordinal;
            uint64_t resolved_seq_count = entry.seq_count;
            const auto &key_shard = shard_for_key(entry.key);
            if (!key_shard.locators->try_read(
                    entry.slot_id, [&](uint32_t current_index) {
                      captured = false;
                      const auto owner = arena.owner_snapshot(current_index);
                      if (!owner.has_value() || owner->key != entry.key ||
                          owner->shard_id != shard_id_for_key(entry.key))
                        return;
                      resolved_index = current_index;
                      resolved_slot_ordinal = static_cast<uint32_t>(
                          current_index & arena.slot_mask());
                      resolved_seq_count = owner->seq_count;
                      simd::simd_copy(arena.data(current_index),
                                      value_scratch.data(), dim);
                      if (score_policy.enabled())
                        score_scratch = std::atomic_ref<uint64_t>(
                                            arena.score_raw(current_index))
                                            .load(std::memory_order_relaxed);
                      if (optimizer_state_dim != 0) {
                        optimizer->state_data(
                            current_index,
                            std::span<OptT>(optimizer_scratch.data(),
                                            optimizer_state_dim));
                        step_scratch = optimizer->step_data(current_index);
                      }
                      captured = true;
                    }))
              continue;
            if (!captured)
              continue;

            const uint64_t resolved_page_id =
                resolved_index >> arena.entries_shift();
            if (resolved_page_id != page_id) {
              if (resolved_page_id < page_id)
                throw std::logic_error("page-image export observed a "
                                       "relocation to an emitted page");
              if (resolved_page_id >= arena.max_pages())
                throw std::out_of_range(
                    "page-image export observed an invalid relocation page");
              page_entries[resolved_page_id].push_back(
                  {entry.key, resolved_seq_count, entry.slot_id,
                   resolved_slot_ordinal});
              continue;
            }

            const PageSlotEntry resolved{entry.key, resolved_seq_count,
                                         entry.slot_id, resolved_slot_ordinal};
            entries[survivor_count++] = resolved;
            if (resolved_slot_ordinal != expected_slot_ordinal) {
              restart = true;
              continue;
            }
            image.slot_ordinals.push_back(resolved_slot_ordinal);
            image.keys.push_back(entry.key);
            image.seq_counts.push_back(resolved_seq_count);
            image.values.insert(image.values.end(), value_scratch.begin(),
                                value_scratch.end());
            if (score_policy.enabled())
              image.scores.push_back(score_scratch);
            if (optimizer_state_dim != 0) {
              image.optimizer_states.insert(image.optimizer_states.end(),
                                            optimizer_scratch.begin(),
                                            optimizer_scratch.end());
              image.steps.push_back(step_scratch);
            }
          }
          entries.resize(survivor_count);
          if (!restart)
            break;
        }

        entries.clear();

        if (image.keys.empty())
          continue;
        if (!visitor(std::move(image)))
          return false;
      }
      return true;
    }

    std::vector<PageImageExport<ParamT, OptT>> materialize_page_images() {
      std::vector<PageImageExport<ParamT, OptT>> images;
      export_page_images([&](PageImageExport<ParamT, OptT> &&image) {
        images.push_back(std::move(image));
        return true;
      });
      return images;
    }

    struct PageImportScratch {
      std::vector<size_t> offsets, positions;
      std::vector<uint32_t> rows, indices;
      std::vector<uint64_t> keys, ordinal_mask;
    };

    void import_page_image(const PageImageImport<ParamT, OptT> &import,
                           bool load_optimizer_state,
                           PageImportScratch &scratch) {
      const auto &image = import.image;
      const size_t optimizer_state_dim =
          load_optimizer_state && optimizer.has_value() ? optimizer->state_dim()
                                                        : 0;
      const size_t count = hogwild::io::validate_page_image(
          import, dim, arena.data_stride(), score_policy.enabled(),
          optimizer_state_dim, arena.entries_per_page());
      if (image.physical_page_id >= arena.max_pages())
        throw std::out_of_range("page-image physical page id is out of range");
      const uint64_t page_end = checkpoint::checked_byte_count(
          checkpoint::checked_sum(image.physical_page_id, 1),
          arena.entries_per_page());
      const uint64_t allocation_count =
          std::max<uint64_t>(arena.allocated_count(), page_end);
      if (!std::in_range<uint32_t>(allocation_count))
        throw std::overflow_error("page-image allocation watermark overflows");
      const uint64_t expected_size = checkpoint::checked_sum(size(), count);
      constexpr size_t word_bits = std::numeric_limits<uint64_t>::digits;
      scratch.ordinal_mask.assign(
          (arena.entries_per_page() + word_bits - 1) / word_bits, 0);
      scratch.offsets.assign(key_shards.size() + 1, 0);
      for (size_t row = 0; row < count; ++row) {
        const uint32_t ordinal = image.slot_ordinals[row];
        auto &word = scratch.ordinal_mask[ordinal / word_bits];
        const uint64_t mask = uint64_t{1} << (ordinal % word_bits);
        if ((word & mask) != 0)
          throw std::invalid_argument("page-image import has duplicate slot");
        word |= mask;
        ++scratch.offsets[shard_id_for_key(image.keys[row]) + 1];
      }
      std::partial_sum(scratch.offsets.begin(), scratch.offsets.end(),
                       scratch.offsets.begin());
      scratch.positions = scratch.offsets;
      scratch.rows.resize(count);
      scratch.keys.resize(count);
      scratch.indices.resize(count);
      for (size_t row = 0; row < count; ++row) {
        const size_t position =
            scratch.positions[shard_id_for_key(image.keys[row])]++;
        scratch.rows[position] = row;
        scratch.keys[position] = image.keys[row];
        scratch.indices[position] = static_cast<uint32_t>(
            image.physical_page_id * arena.entries_per_page() +
            image.slot_ordinals[row]);
      }
      arena.reserve_exact_page(
          image.physical_page_id,
          static_cast<uint32_t>(shard_id_for_key(image.keys.front())));
      if (optimizer.has_value())
        optimizer->reserve(allocation_count);
      for (size_t shard_id = 0; shard_id < key_shards.size(); ++shard_id) {
        const size_t begin = scratch.offsets[shard_id];
        const size_t end = scratch.offsets[shard_id + 1];
        if (begin == end)
          continue;
        auto &key_shard = *key_shards[shard_id];
        std::unique_lock map_lock(key_shard.writer_mutex);
        key_shard.reserve_new_indices(end - begin);
        for (size_t position = begin; position < end; ++position) {
          const size_t row = scratch.rows[position];
          const uint64_t key = scratch.keys[position];
          const uint32_t index = scratch.indices[position];
          if (key_shard.find_unlocked(key).has_value())
            throw std::invalid_argument("page-image import has duplicate key");
          if (arena.is_live(index))
            throw std::invalid_argument("page-image import has duplicate slot");
          arena.bind_exact(index, static_cast<uint32_t>(shard_id));
          if (optimizer.has_value())
            optimizer->allocate(index);
          arena.publish_owner_exact(index, key, static_cast<uint32_t>(shard_id),
                                    image.seq_counts[row]);
          simd::simd_copy(image.values.data() + row * dim, arena.data(index),
                          dim);
          if (score_policy.enabled()) {
            const uint64_t score = image.scores.empty() ? 0 : image.scores[row];
            std::atomic_ref<uint64_t>(arena.score_raw(index))
                .store(score, std::memory_order_relaxed);
          }
          if (optimizer_state_dim != 0) {
            optimizer->set_state(
                index, std::span<const OptT>(image.optimizer_states.data() +
                                                 row * optimizer_state_dim,
                                             optimizer_state_dim));
            optimizer->set_step(index, image.steps[row]);
          }
        }
        key_shard.import_indices_quiesced(
            std::span<const uint64_t>(scratch.keys).subspan(begin, end - begin),
            std::span<const uint32_t>(scratch.indices)
                .subspan(begin, end - begin));
        for (size_t position = begin; position < end; ++position)
          arena.mark_live(scratch.indices[position], true);
      }
      arena.set_allocation_count(static_cast<uint32_t>(allocation_count));
      if (size() != expected_size)
        throw std::logic_error("page-image import entry count mismatch");
    }

    std::vector<std::shared_ptr<const SealedHotPage<ParamT>>>
    seal_adaptive_pages(float memory_pressure,
                        const AdaptivePageSelectorConfig &config,
                        PageThermalTier tier, uint32_t cluster_id,
                        uint64_t page_layout_generation) {
      page_telemetry->advance_residency_epoch();
      std::vector<StablePageSnapshot> snapshots;
      snapshots.reserve(arena.page_count());
      for (uint64_t page_id = 0; page_id < arena.page_count(); ++page_id) {
        auto snapshot = snapshot_page(page_id);
        if (snapshot.has_value())
          snapshots.push_back(std::move(*snapshot));
      }
      if (snapshots.empty())
        throw std::runtime_error("adaptive sealing has no live arena pages");
      std::vector<std::vector<PageLayoutEntryView<ParamT>>> entries;
      entries.reserve(snapshots.size());
      std::vector<PageLayoutInput<ParamT>> inputs;
      inputs.reserve(snapshots.size());
      for (const StablePageSnapshot &snapshot : snapshots) {
        auto &page_entries = entries.emplace_back();
        page_entries.reserve(snapshot.keys.size());
        for (size_t entry = 0; entry < snapshot.keys.size(); ++entry) {
          page_entries.push_back(
              {snapshot.keys[entry], snapshot.seq_counts[entry],
               std::span<const ParamT>(snapshot.values.data() + entry * dim,
                                       dim)});
        }
        inputs.push_back({snapshot.physical_page_id, snapshot.bytes,
                          snapshot.signal, page_entries});
      }
      AdaptiveSealedPagePlanner<ParamT> planner(AdaptivePageSelector{},
                                                sealed_pages);
      return planner
          .plan(inputs, memory_pressure, config, tier, cluster_id,
                page_layout_generation)
          .pages;
    }

    std::vector<uint64_t>
    select_adaptive_pages(float memory_pressure,
                          const AdaptivePageSelectorConfig &config) {
      std::vector<uint64_t> ids;
      std::vector<uint64_t> bytes;
      std::vector<PageThermalSignal> signals;
      for (uint64_t page_id = 0; page_id < arena.page_count(); ++page_id) {
        if (page_reuse->unavailable(
                static_cast<uint32_t>(page_id * arena.entries_per_page())))
          continue;
        auto snapshot = snapshot_page(page_id);
        if (!snapshot.has_value())
          continue;
        ids.push_back(snapshot->physical_page_id);
        bytes.push_back(snapshot->bytes);
        signals.push_back(snapshot->signal);
      }
      if (ids.empty())
        throw std::runtime_error("adaptive compaction has no eligible pages");
      return AdaptivePageSelector{}
          .select(ids, signals, bytes, memory_pressure, config)
          .page_ids;
    }

    size_t compact_adaptive_pages(float memory_pressure,
                                  const AdaptivePageSelectorConfig &config) {
      page_telemetry->advance_residency_epoch();
      const std::vector<uint64_t> page_ids =
          select_adaptive_pages(memory_pressure, config);
      access_telemetry.decay(1);
      return compact_pages(page_ids, true);
    }

    template <typename Visit>
    bool with_shared_entry(uint64_t key, Visit &&visit) {
      const size_t hash = SplitMixHash<uint64_t>{}(key);
      return with_shared_entry_hashed(key, hash, std::forward<Visit>(visit));
    }

    template <typename Visit>
    bool with_shared_entry_hashed(uint64_t key, size_t hash, Visit &&visit) {
      auto &shard = *key_shards[hash & shard_mask];
      struct TlsLookupEntry {
        const SlotStore *owner = nullptr;
        uint64_t key = 0;
        uint32_t slot_id = 0;
        const hogwild_internal::LocatorSlot *slot = nullptr;
        uint64_t packed = 0;
        uint64_t generation = 0;
        uint64_t shard_generation = 0;
        uint64_t table_identity = 0;
      };
      constexpr size_t kTlsLookupEntries = 1024;
      thread_local std::array<TlsLookupEntry, kTlsLookupEntries> tls_cache{};
      const unsigned kShardBits =
          std::countr_zero(static_cast<size_t>(shard_mask + 1));
      TlsLookupEntry &cached =
          tls_cache[(hash >> kShardBits) & (kTlsLookupEntries - 1)];
      const uint64_t generation =
          lookup_generation.load(std::memory_order_acquire);
      const uint64_t shard_generation =
          shard.read_view->generation.load(std::memory_order_acquire);
      const uint64_t table_identity = shard.read_view->table_identity;
      if (cached.owner == this && cached.key == key && cached.slot_id != 0 &&
          cached.generation == generation &&
          cached.shard_generation == shard_generation &&
          cached.table_identity == table_identity) {
        const uint64_t before =
            hogwild_internal::SegmentedLocatorTable::load_packed_at(
                cached.slot);
        if (before == cached.packed && (before >> 32 & 1U) == 0U) {
          if (!shard.locators->try_read(cached.slot_id, [&](uint32_t index) {
                __builtin_prefetch(arena.data_trusted(index), 0, 1);
                visit(index);
              }))
            return false;
          return true;
        }
      }
      auto slot_id = shard.find_slot_hashed_unlocked(key, hash);
      if (!slot_id.has_value())
        return false;
      const auto *locator = shard.read_view->locators->slot_address(*slot_id);
      if (!shard.locators->try_read(*slot_id, [&](uint32_t index) {
            __builtin_prefetch(arena.data_trusted(index), 0, 1);
            visit(index);
          }))
        return false;
      const uint64_t packed =
          hogwild_internal::SegmentedLocatorTable::load_packed_at(locator);
      cached = {this,   key,        *slot_id,         locator,
                packed, generation, shard_generation, table_identity};
      return true;
    }

    template <typename Visit>
    bool with_read_view_entry_hashed(uint64_t key, size_t hash,
                                     Visit &&visit) const {
      const auto &shard = *key_shards[hash & shard_mask];
      const auto slot_id = shard.find_slot_hashed_unlocked(key, hash);
      if (!slot_id.has_value())
        return false;
      return shard.read_view->locators->try_read(*slot_id,
                                                 std::forward<Visit>(visit));
    }

    void prefetch_entry_cacheline([[maybe_unused]] uint64_t key,
                                  size_t hash) const noexcept {
      const auto &shard = *key_shards[hash & shard_mask];
      const auto *rv = shard.read_view.get();
      if (rv != nullptr) {
        rv->prefetch_bucket_cacheline(hash);
      }
    }

    template <typename Visit>
    bool with_shared_write_entry(uint64_t key, Visit &&visit) {
      const size_t hash = SplitMixHash<uint64_t>{}(key);
      return with_shared_write_entry_hashed(key, hash,
                                            std::forward<Visit>(visit));
    }

    template <typename Visit>
    bool with_shared_write_entry_hashed(uint64_t key, size_t hash,
                                        Visit &&visit) {
      auto &shard = *key_shards[hash & shard_mask];
      const auto slot_id = shard.find_slot_hashed_unlocked(key, hash);
      if (!slot_id.has_value())
        return false;
      return shard.locators->try_read_with_record(*slot_id,
                                                  std::forward<Visit>(visit));
    }

    template <typename Batch, typename Visit>
    bool with_shared_optimizer_batch(const Batch &batch,
                                     std::span<const size_t> positions,
                                     Visit &&visit) {
      if (positions.empty())
        throw std::invalid_argument("optimizer locator batch is empty");
      OptimizerBatchScratch &scratch = optimizer_batch_scratch();
      auto &locator_entries = scratch.locator_entries;
      locator_entries.clear();
      locator_entries.reserve(positions.size());
      for (const size_t position : positions) {
        const auto slot_id = key_shards[batch.hash(position) & shard_mask]
                                 ->find_slot_hashed_unlocked(
                                     batch.key(position), batch.hash(position));
        if (!slot_id.has_value())
          return false;
        locator_entries.push_back({*slot_id, batch.key(position)});
      }
      const size_t key_shard_id = batch.hash(positions.front()) & shard_mask;
      for (const size_t position : positions) {
        if ((batch.hash(position) & shard_mask) != key_shard_id)
          throw std::logic_error("optimizer batch crosses key shard boundary");
      }
      return key_shards[key_shard_id]->locators->try_write_key_batch(
          locator_entries,
          [&](std::span<const uint32_t> indices,
              std::span<const hogwild_internal::CheckpointRecordRef>
                  checkpoint_records) {
            std::forward<Visit>(visit)(batch, positions, indices,
                                       checkpoint_records);
          });
    }

    template <typename Batch, typename Visit>
    bool with_shared_optimizer_batch_contiguous(const Batch &batch,
                                                size_t begin, size_t end,
                                                Visit &&visit) {
      if (begin >= end || end > batch.size())
        throw std::invalid_argument("optimizer contiguous batch is invalid");
      OptimizerBatchScratch &scratch = optimizer_batch_scratch();
      auto &locator_entries = scratch.locator_entries;
      locator_entries.clear();
      locator_entries.reserve(end - begin);
      const size_t key_shard_id = batch.hash(begin) & shard_mask;
      for (size_t position = begin; position < end; ++position) {
        if ((batch.hash(position) & shard_mask) != key_shard_id)
          throw std::logic_error("optimizer contiguous batch crosses shard");
        const auto slot_id =
            key_shards[key_shard_id]->find_slot_hashed_unlocked(
                batch.key(position), batch.hash(position));
        if (!slot_id.has_value())
          return false;
        locator_entries.push_back({*slot_id, batch.key(position)});
      }
      return key_shards[key_shard_id]->locators->try_write_key_batch(
          locator_entries,
          [&](std::span<const uint32_t> indices,
              std::span<const hogwild_internal::CheckpointRecordRef>
                  checkpoint_records) {
            std::forward<Visit>(visit)(batch, begin, indices,
                                       checkpoint_records);
          });
    }

    template <typename Batch, typename Updated>
    void update_batch(const Batch &batch, uint64_t global_step,
                      size_t normalization_steps, Updated &&on_updated) {
      if (batch.size() == 0)
        return;
      if (normalization_steps == 0)
        throw std::invalid_argument("normalization_steps must be positive");
      if (!optimizer.has_value())
        return;
      const size_t first_key_shard = batch.hash(0) & shard_mask;
      const bool single_key_shard = std::ranges::all_of(
          std::views::iota(size_t{0}, batch.size()),
          [this, first_key_shard, &batch](size_t position) {
            return (batch.hash(position) & shard_mask) == first_key_shard;
          });
      OptimizerBatchScratch &scratch = optimizer_batch_scratch();
      auto &rows = scratch.rows;
      rows.clear();
      rows.reserve(batch.size());
      const auto execute_group = [&](std::span<const size_t> positions) {
        const bool updated = with_shared_optimizer_batch(
            batch, positions,
            [&](const auto &batch_view, std::span<const size_t> batch_positions,
                std::span<const uint32_t> indices,
                std::span<const hogwild_internal::CheckpointRecordRef>
                    checkpoint_records) {
              rows.clear();
              for (size_t row = 0; row < indices.size(); ++row) {
                ParamT *parameter = arena.data(indices[row]);
                const uint64_t score =
                    score_policy.enabled()
                        ? std::atomic_ref<uint64_t>(
                              arena.score_raw(indices[row]))
                              .load(std::memory_order_relaxed)
                        : 0;
                rows.push_back(
                    {indices[row], std::span<ParamT>(parameter, dim),
                     std::span<const ParamT>(
                         batch_view.gradient(batch_positions[row]), dim),
                     score});
              }
              optimizer->update(
                  {std::span<const OptimizerBatchRow<ParamT>>(rows), dim},
                  global_step);
              for (size_t row = 0; row < batch_positions.size(); ++row)
                on_updated(batch_view.key(batch_positions[row]),
                           checkpoint_records[row]);
            });
        if (!updated)
          handle_optimizer_locator_revalidation_failure(
              batch.key(positions.front()));
      };
      if (single_key_shard) {
        const bool updated = with_shared_optimizer_batch_contiguous(
            batch, 0, batch.size(),
            [&](const auto &batch_view, size_t begin,
                std::span<const uint32_t> indices,
                std::span<const hogwild_internal::CheckpointRecordRef>
                    checkpoint_records) {
              rows.clear();
              for (size_t row = 0; row < indices.size(); ++row) {
                ParamT *parameter = arena.data(indices[row]);
                const uint64_t score =
                    score_policy.enabled()
                        ? std::atomic_ref<uint64_t>(
                              arena.score_raw(indices[row]))
                              .load(std::memory_order_relaxed)
                        : 0;
                rows.push_back({indices[row], std::span<ParamT>(parameter, dim),
                                std::span<const ParamT>(
                                    batch_view.gradient(begin + row), dim),
                                score});
              }
              optimizer->update(
                  {std::span<const OptimizerBatchRow<ParamT>>(rows), dim},
                  global_step);
              for (size_t row = 0; row < indices.size(); ++row)
                on_updated(batch_view.key(begin + row),
                           checkpoint_records[row]);
            });
        if (!updated)
          handle_optimizer_locator_revalidation_failure(batch.key(0));
        return;
      }
      auto &order = scratch.order;
      order.resize(batch.size());
      std::ranges::iota(order, size_t{0});
      std::ranges::sort(order, {}, [&](size_t position) {
        return batch.hash(position) & shard_mask;
      });
      size_t begin = 0;
      while (begin < order.size()) {
        const size_t key_shard_id = batch.hash(order[begin]) & shard_mask;
        size_t end = begin + 1;
        while (end < order.size() &&
               (batch.hash(order[end]) & shard_mask) == key_shard_id)
          ++end;
        auto &positions = scratch.positions;
        positions.clear();
        positions.reserve(end - begin);
        for (size_t position = begin; position < end; ++position)
          positions.push_back(order[position]);
        execute_group(positions);
        begin = end;
      }
    }

    template <typename Visit>
    bool assign_existing(uint64_t key, Visit &&visit) {
      return with_shared_write_entry(key, std::forward<Visit>(visit));
    }

    template <typename Visit>
    bool assign_existing_hashed(uint64_t key, size_t hash, Visit &&visit) {
      return with_shared_write_entry_hashed(key, hash,
                                            std::forward<Visit>(visit));
    }

    template <typename Fn> void for_each(Fn &&fn) {
      for (auto &shard : key_shards) {
        shard->for_each_entry([&](uint64_t key, uint32_t slot_id) {
          auto index = shard->locators->load(slot_id);
          if (index)
            fn(key, *index);
        });
      }
    }

    size_t size() {
      size_t total = 0;
      for (auto &shard : key_shards) {
        total += shard->count.load(std::memory_order_acquire);
      }
      return total;
    }

    template <typename OnRemove>
    size_t clear_keys(OnRemove &&on_remove, uint64_t sequence_begin) {
      static_assert(
          std::is_nothrow_invocable_v<OnRemove, uint64_t, uint32_t,
                                      hogwild_internal::CheckpointRecordRef,
                                      uint64_t>);
      lookup_generation.fetch_add(1, std::memory_order_acq_rel);
      size_t total = 0;
      for (const auto &shard : key_shards)
        total += shard->count.load(std::memory_order_acquire);
      struct RemovedKey {
        uint64_t key;
        uint32_t index;
        hogwild_internal::CheckpointRecordRef record;
      };
      std::vector<std::unique_ptr<KeyShard>> replacements;
      replacements.reserve(key_shards.size());
      for (const auto &shard : key_shards)
        replacements.push_back(std::make_unique<KeyShard>(
            shard->reserved_keys, page_reuse, retirement_pressure,
            index_retirement_bytes(),
            static_cast<unsigned>(std::countr_zero(shard_mask + 1)),
            static_cast<uint32_t>(dim), checkpoint_owner));
      std::vector<RemovedKey> removed_keys;
      removed_keys.reserve(total);
      for (const auto &shard : key_shards) {
        shard->for_each_entry([&](uint64_t key, uint32_t slot_id) {
          if (removed_keys.size() == removed_keys.capacity())
            throw std::logic_error("clear key count changed during capture");
          const auto index = shard->locators->load(slot_id);
          if (!index.has_value())
            throw std::logic_error("clear observed a missing locator index");
          removed_keys.push_back(
              {key, *index, shard->locators->checkpoint_record(slot_id)});
        });
      }
      if (!removed_keys.empty() && sequence_begin == 0)
        throw std::logic_error("clear key sequence range is missing");
      for (size_t shard_index = 0; shard_index < key_shards.size();
           ++shard_index)
        key_shards[shard_index].swap(replacements[shard_index]);
      for (size_t index = 0; index < removed_keys.size(); ++index)
        on_remove(removed_keys[index].key, removed_keys[index].index,
                  removed_keys[index].record, sequence_begin + index);
      return removed_keys.size();
    }

    uint32_t allocate_entry(uint32_t shard_id) {
      uint32_t idx = arena.allocate_fresh(shard_id, [&](uint32_t candidate) {
        if (optimizer.has_value() && optimizer->capacity() <= candidate) {
          optimizer->reserve(static_cast<size_t>(candidate) + 1);
        }
      });
      if (optimizer.has_value())
        optimizer->allocate(idx);
      return idx;
    }

    uint32_t allocate_compaction_destination(uint32_t shard_id) {
      uint32_t idx = arena.allocate_fresh(
          shard_id,
          [&](uint32_t candidate) {
            if (optimizer.has_value() && optimizer->capacity() <= candidate)
              optimizer->reserve(static_cast<size_t>(candidate) + 1);
          },
          [&](uint32_t candidate) {
            return page_reuse->destination_available(candidate);
          });
      if (optimizer.has_value())
        optimizer->allocate(idx);
      return idx;
    }

    void allocate_compaction_destinations(uint32_t shard_id, size_t count,
                                          std::span<uint32_t> destinations) {
      if (count == 0 || destinations.size() != count)
        throw std::invalid_argument(
            "compaction destination batch size is invalid");
      arena.allocate_fresh_batch(
          shard_id, count, destinations,
          [&](uint32_t, uint32_t end) {
            if (optimizer.has_value() && optimizer->capacity() < end)
              optimizer->reserve(end);
          },
          [&](uint32_t candidate) {
            return page_reuse->destination_available(candidate);
          });
      if (optimizer.has_value()) {
        for (uint32_t destination : destinations)
          optimizer->allocate(destination);
      }
    }

    uint32_t allocate_entry_range(size_t count) {
#if defined(PAGE_COMPACTION_DEEP_PROFILE)
      const auto destination_ready_started = std::chrono::steady_clock::now();
#endif
      const uint32_t first =
          arena.allocate_range(count, [&](uint32_t, uint32_t end) {
            if (optimizer.has_value() && optimizer->capacity() < end)
              optimizer->reserve(end);
          });
#if defined(PAGE_COMPACTION_DEEP_PROFILE)
      const uint32_t last = static_cast<uint32_t>(first + count - 1);
      page_compaction_profile.destination_ready_ns.fetch_add(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - destination_ready_started)
              .count(),
          std::memory_order_relaxed);
      page_compaction_profile.destination_ready_calls.fetch_add(
          1, std::memory_order_relaxed);
      page_compaction_profile.destination_ready_pages.fetch_add(
          (last >> arena.entries_shift()) - (first >> arena.entries_shift()) +
              1,
          std::memory_order_relaxed);
#endif
      return first;
    }

    struct BatchEntry {
      uint64_t key;
      size_t hash;
      size_t position;
    };

    template <typename Initialize, typename Visit, typename Published>
    void find_or_insert_batch(std::span<const BatchEntry> entries,
                              Initialize &&initialize, Visit &&visit,
                              Published &&on_published,
                              bool write_existing_payload,
                              bool coalesce_duplicates = false,
                              size_t retry_attempt = 0) {
      if (entries.empty())
        return;
      constexpr size_t kMaximumConcurrentMutationRetries = 8;
      if (retry_attempt >= kMaximumConcurrentMutationRetries)
        throw std::runtime_error(
            "concurrent key mutation retry limit exceeded");
      struct ExistingEntry {
        BatchEntry entry;
        uint32_t slot_id;
      };
      struct ShardBatch {
        std::vector<ExistingEntry> existing;
        std::vector<BatchEntry> candidates;
        struct AliasEntry {
          uint32_t candidate_index;
          uint32_t winner_order;
        };
        std::vector<AliasEntry> aliases;
        std::vector<BatchEntry> inserted;
        std::vector<BatchEntry> published;
        std::vector<uint32_t> winner_candidates;
        std::vector<uint64_t> inserted_keys;
        std::vector<uint32_t> inserted_indices;
        struct PendingAllocation {
          BatchEntry entry;
          uint32_t index;
          uint32_t winner_order;
          bool recycled = false;
          bool published = false;
        };
        std::vector<PendingAllocation> allocations;
        std::vector<uint32_t> winner_slots;
      };
      std::vector<ShardBatch> by_shard(key_shards.size());
      std::vector<uint32_t> resolved_slots;
      thread_local ankerl::unordered_dense::map<uint64_t, uint32_t>
          winner_by_key;
      if (coalesce_duplicates) {
        const size_t maximum_position =
            std::ranges::max(entries, {}, &BatchEntry::position).position;
        if (maximum_position == std::numeric_limits<size_t>::max())
          throw std::overflow_error("batch position range overflows");
        resolved_slots.assign(maximum_position + 1, 0);
      }
      for (ShardBatch &batch : by_shard) {
        batch.existing.reserve(entries.size() / by_shard.size() + 1);
        batch.candidates.reserve(entries.size() / by_shard.size() + 1);
        batch.aliases.reserve(entries.size() / by_shard.size() + 1);
        batch.inserted.reserve(entries.size() / by_shard.size() + 1);
        batch.published.reserve(entries.size() / by_shard.size() + 1);
        batch.winner_candidates.reserve(entries.size() / by_shard.size() + 1);
        batch.inserted_keys.reserve(entries.size() / by_shard.size() + 1);
        batch.inserted_indices.reserve(entries.size() / by_shard.size() + 1);
        batch.allocations.reserve(entries.size() / by_shard.size() + 1);
      }
      auto pressure_call = retirement_pressure->writer_call();
      pressure_call.apply();
      auto retirement_admission = retirement_pressure->try_control_plane_call();
      for (const BatchEntry &entry : entries)
        by_shard[shard_id_for_hash(entry.hash)].candidates.push_back(entry);
      try {
        for (size_t shard_id = 0; shard_id < by_shard.size(); ++shard_id) {
          ShardBatch &batch = by_shard[shard_id];
          if (batch.candidates.empty())
            continue;
          {
            auto &key_shard = *key_shards[shard_id];
            std::unique_lock writer_lock(key_shard.writer_mutex);
            for (const BatchEntry &entry : batch.candidates) {
              auto located =
                  key_shard.find_slot_hashed_unlocked(entry.key, entry.hash);
              std::optional<typename KeyShard::LocatorRef> resolved;
              if (located.has_value()) {
                auto index = key_shard.locators->load(*located);
                if (!index)
                  throw std::logic_error(
                      "key directory references an empty locator slot");
                resolved = typename KeyShard::LocatorRef{*located, *index};
              }
              if (resolved.has_value()) {
                if (key_shard.locators->requires_rotation(resolved->slot_id))
                  resolved = key_shard.rotate_locator_unlocked(
                      entry.key, resolved->slot_id);
                batch.existing.push_back({entry, resolved->slot_id});
                if (coalesce_duplicates)
                  resolved_slots[entry.position] = resolved->slot_id;
                continue;
              }
              if (!retirement_admission.admitted())
                retirement_admission.require_admitted();
              batch.inserted.push_back(entry);
            }
            batch.winner_candidates.clear();
            batch.aliases.clear();
            if (coalesce_duplicates) {
              if (batch.inserted.size() > std::numeric_limits<uint32_t>::max())
                throw std::overflow_error("miss batch exceeds index range");
              winner_by_key.clear();
              winner_by_key.reserve(batch.inserted.size());
              for (uint32_t candidate_index = 0;
                   candidate_index < batch.inserted.size(); ++candidate_index) {
                const BatchEntry &entry = batch.inserted[candidate_index];
                auto [winner, inserted] = winner_by_key.try_emplace(
                    entry.key,
                    static_cast<uint32_t>(batch.winner_candidates.size()));
                if (inserted)
                  batch.winner_candidates.push_back(candidate_index);
                else
                  batch.aliases.push_back({candidate_index, winner->second});
              }
            } else {
              batch.winner_candidates.resize(batch.inserted.size());
              std::ranges::iota(batch.winner_candidates, uint32_t{0});
            }
          }
        }

        struct AllocationOrder {
          size_t shard_id;
          const BatchEntry *entry;
          uint32_t winner_order;
        };
        std::vector<AllocationOrder> allocation_order;
        allocation_order.reserve(entries.size());
        for (size_t shard_id = 0; shard_id < by_shard.size(); ++shard_id) {
          const ShardBatch &batch = by_shard[shard_id];
          for (uint32_t winner_order = 0;
               winner_order < batch.winner_candidates.size(); ++winner_order)
            allocation_order.push_back(
                {shard_id,
                 &batch.inserted[batch.winner_candidates[winner_order]],
                 winner_order});
        }
        std::ranges::sort(
            allocation_order, {},
            [](const AllocationOrder &item) { return item.entry->position; });
        for (const AllocationOrder &item : allocation_order) {
          auto &batch = by_shard[item.shard_id];
          auto &key_shard = *key_shards[item.shard_id];
          uint32_t arena_index;
          if (auto recycled = key_shard.take_recycled_index()) {
            arena_index = *recycled;
            arena.prepare_reused(arena_index);
            if (optimizer.has_value())
              optimizer->allocate(arena_index);
          } else {
            key_shard.reserve_new_index();
            arena_index = allocate_entry(static_cast<uint32_t>(item.shard_id));
          }
          batch.allocations.push_back(
              {*item.entry, arena_index, item.winner_order});
        }

        for (size_t shard_id = 0; shard_id < by_shard.size(); ++shard_id) {
          ShardBatch &batch = by_shard[shard_id];
          if (batch.allocations.empty())
            continue;
          {
            auto &key_shard = *key_shards[shard_id];
            std::unique_lock writer_lock(key_shard.writer_mutex);
            batch.published.clear();
            batch.inserted_keys.clear();
            batch.inserted_indices.clear();
            batch.winner_slots.assign(batch.winner_candidates.size(), 0);
            for (auto &allocation : batch.allocations) {
              auto located = key_shard.find_slot_hashed_unlocked(
                  allocation.entry.key, allocation.entry.hash);
              if (located.has_value()) {
                auto index = key_shard.locators->load(*located);
                if (!index)
                  throw std::logic_error(
                      "key directory references an empty locator slot");
                batch.existing.push_back({allocation.entry, *located});
                batch.winner_slots[allocation.winner_order] = *located;
                if (coalesce_duplicates)
                  resolved_slots[allocation.entry.position] = *located;
                key_shard.recycle_unpublished_index(allocation.index);
                allocation.recycled = true;
                continue;
              }
              arena.publish_owner(allocation.index, allocation.entry.key,
                                  static_cast<uint32_t>(shard_id));
              initialize(allocation.entry, allocation.index);
              batch.published.push_back(allocation.entry);
              batch.inserted_keys.push_back(allocation.entry.key);
              batch.inserted_indices.push_back(allocation.index);
            }
            if (!batch.inserted_keys.empty()) {
              key_shards[shard_id]->publish_indices_batch_unlocked(
                  batch.inserted_keys, batch.inserted_indices);
              for (auto &allocation : batch.allocations)
                if (!allocation.recycled)
                  allocation.published = true;
              for (const uint32_t index : batch.inserted_indices)
                arena.mark_live(index, true);
              for (const auto &entry : batch.published) {
                const auto slot_id = key_shard.find_slot_hashed_unlocked(
                    entry.key, SplitMixHash<uint64_t>{}(entry.key));
                if (!slot_id.has_value())
                  throw std::logic_error(
                      "published key disappeared before checkpoint record");
                on_published(entry,
                             key_shard.locators->checkpoint_record(*slot_id));
              }
              for (auto &allocation : batch.allocations) {
                if (allocation.recycled)
                  continue;
                auto slot_id = key_shard.find_slot_hashed_unlocked(
                    allocation.entry.key, allocation.entry.hash);
                if (!slot_id)
                  throw std::logic_error(
                      "published key disappeared before alias resolution");
                batch.winner_slots[allocation.winner_order] = *slot_id;
                batch.existing.push_back({allocation.entry, *slot_id});
                if (coalesce_duplicates)
                  resolved_slots[allocation.entry.position] = *slot_id;
              }
            }
            for (const auto &alias : batch.aliases)
              if (coalesce_duplicates)
                resolved_slots[batch.inserted[alias.candidate_index].position] =
                    batch.winner_slots[alias.winner_order];
              else
                batch.existing.push_back(
                    {batch.inserted[alias.candidate_index],
                     batch.winner_slots[alias.winner_order]});
          }
        }
      } catch (...) {
        for (size_t shard_id = 0; shard_id < by_shard.size(); ++shard_id) {
          ShardBatch &batch = by_shard[shard_id];
          for (auto &allocation : batch.allocations) {
            if (allocation.recycled || allocation.published)
              continue;
            arena.mark_live(allocation.index, false);
            key_shards[shard_id]->recycle_unpublished_index(allocation.index);
          }
        }
        throw;
      }
      if (coalesce_duplicates) {
        std::vector<BatchEntry> retry_entries;
        retry_entries.reserve(entries.size());
        for (size_t shard_id = 0; shard_id < by_shard.size(); ++shard_id) {
          {
            auto &entry_shard = *key_shards[shard_id];
            for (const BatchEntry &entry : by_shard[shard_id].candidates) {
              const uint32_t slot_id = resolved_slots[entry.position];
              if (slot_id == 0)
                throw std::logic_error("resolved lookup slot is zero");
              if (!entry_shard.locators->try_read_key_with_record(
                      slot_id, entry.key,
                      [&](uint32_t index,
                          hogwild_internal::CheckpointRecordRef record) {
                        visit(entry, index, record);
                      }))
                retry_entries.push_back(entry);
            }
          }
        }
        if (!retry_entries.empty())
          find_or_insert_batch(std::span<const BatchEntry>(retry_entries),
                               initialize, visit, on_published,
                               write_existing_payload, coalesce_duplicates,
                               retry_attempt + 1);
        return;
      }
      std::vector<BatchEntry> retry_entries;
      retry_entries.reserve(entries.size());
      for (size_t shard_id = 0; shard_id < by_shard.size(); ++shard_id) {
        {
          ShardBatch &batch = by_shard[shard_id];
          auto &key_shard = *key_shards[shard_id];
          for (const ExistingEntry &entry : batch.existing) {
            if (!key_shard.locators->try_read_key_with_record(
                    entry.slot_id, entry.entry.key,
                    [&](uint32_t index,
                        hogwild_internal::CheckpointRecordRef record) {
                      visit(entry.entry, index, record);
                    }))
              retry_entries.push_back(entry.entry);
          }
        }
      }
      if (!retry_entries.empty())
        find_or_insert_batch(std::span<const BatchEntry>(retry_entries),
                             initialize, visit, on_published,
                             write_existing_payload, coalesce_duplicates,
                             retry_attempt + 1);
    }

    void reserve(size_t keys_count) {
      size_t shard_count = key_shards.size();
      if (shard_count == 0)
        return;
      std::vector<size_t> counts(shard_count, keys_count / shard_count);
      for (size_t i = 0; i < keys_count % shard_count; ++i)
        ++counts[i];
      reserve_per_shard(counts);
    }

    void reserve_per_shard(std::span<const size_t> counts) {
      if (counts.size() != key_shards.size()) {
        throw std::invalid_argument("reserve_per_shard counts size mismatch");
      }
      size_t total_keys = 0;
      for (size_t i = 0; i < counts.size(); ++i) {
        size_t current_size =
            key_shards[i]->count.load(std::memory_order_acquire);
        size_t target_keys = std::max(counts[i], current_size);
        if (target_keys >
            hogwild_internal::FlatKeyDirectory::maximum_key_capacity())
          throw std::overflow_error("key shard capacity overflow");
        if (target_keys > std::numeric_limits<size_t>::max() - total_keys) {
          throw std::overflow_error(
              "reserve_per_shard total capacity overflow");
        }
        total_keys += target_keys;
      }
      for (size_t i = 0; i < counts.size(); ++i) {
        std::unique_lock map_lock(key_shards[i]->writer_mutex);
        size_t current_size =
            key_shards[i]->count.load(std::memory_order_acquire);
        size_t target_keys = std::max(counts[i], current_size);
        if (target_keys <= key_shards[i]->reserved_keys) {
          continue;
        }
        key_shards[i]->expand(target_keys);
      }
      size_t arena_capacity =
          std::max(total_keys, static_cast<size_t>(arena.allocated_count()));
      if (!arena.numa_aware_pages())
        arena.reserve_entries(arena_capacity);
      if (optimizer.has_value()) {
        optimizer->reserve(arena_capacity);
      }
    }

    bool relocate_key_to_destination(
        uint64_t key, uint32_t destination, bool initialize_destination,
        std::optional<std::pair<uint32_t, uint64_t>> expected_source =
            std::nullopt,
        bool retire_source_slot = true) {
      auto &key_shard = shard_for_key(key);
      bool locator_published = false;
      try {
        if (initialize_destination) {
          arena.prepare_unpublished(destination);
          if (optimizer.has_value())
            optimizer->allocate(destination);
        }
        std::optional<std::pair<uint32_t, uint64_t>> source = expected_source;
        if (!source.has_value()) {
          with_shared_entry(key, [&](uint32_t index) {
            auto owner = arena.owner_snapshot(index);
            if (!owner.has_value() || owner->key != key ||
                owner->shard_id != shard_id_for_key(key)) {
              throw std::logic_error(
                  "live entry owner does not match key shard");
            }
            source = std::make_pair(index, owner->seq_count);
          });
        }
        if (!source.has_value()) {
          key_shard.recycle_unpublished_index(destination);
          return false;
        }
        const size_t optimizer_state_dim =
            optimizer.has_value() ? optimizer->state_dim() : 0;
        std::vector<OptT> optimizer_state(optimizer_state_dim);
        auto current = key_shard.find_locator_unlocked(key);
        if (!current.has_value() || current->index != source->first) {
          key_shard.recycle_unpublished_index(destination);
          return false;
        }
        auto current_owner = arena.owner_snapshot(current->index);
        if (!current_owner.has_value() || current_owner->key != key ||
            current_owner->shard_id != shard_id_for_key(key) ||
            current_owner->seq_count != source->second) {
          key_shard.recycle_unpublished_index(destination);
          return false;
        }
        if (key_shard.locators->requires_rotation(current->slot_id)) {
          key_shard.recycle_unpublished_index(destination);
          return false;
        }
        const bool relocated = key_shard.locators->relocate(
            current->slot_id, source->first, destination,
            [&](uint32_t locked_index) {
              arena.publish_owner(destination, key,
                                  static_cast<uint32_t>(shard_id_for_key(key)));
              simd::simd_copy(arena.data(locked_index), arena.data(destination),
                              dim);
              if (score_policy.enabled()) {
                const uint64_t score =
                    std::atomic_ref<uint64_t>(arena.score_raw(locked_index))
                        .load(std::memory_order_relaxed);
                std::atomic_ref<uint64_t>(arena.score_raw(destination))
                    .store(score, std::memory_order_relaxed);
              }
              if (optimizer.has_value()) {
                if (optimizer_state_dim != 0) {
                  optimizer->state_data(locked_index,
                                        std::span<OptT>(optimizer_state));
                  optimizer->set_state(destination,
                                       std::span<const OptT>(optimizer_state));
                }
                optimizer->set_step(destination,
                                    optimizer->step_data(locked_index));
              }
              arena.mark_live(destination, true);
            });
        if (!relocated) {
          key_shard.recycle_unpublished_index(destination);
          return false;
        }

        locator_published = true;
        arena.mark_live(current->index, false);
        if (retire_source_slot)
          key_shard.retire_published_index(current->index);
        return true;
      } catch (...) {
        if (!locator_published) {
          arena.mark_live(destination, false);
          key_shard.recycle_unpublished_index(destination);
        }
        throw;
      }
    }

    void copy_source_key_to_destination(uint64_t key, uint32_t source_index,
                                        uint32_t destination,
                                        uint32_t expected_shard_id,
                                        std::span<OptT> optimizer_state) {
#if defined(PAGE_COMPACTION_DEEP_PROFILE)
      const auto owner_publish_started = std::chrono::steady_clock::now();
#endif
      arena.publish_owner(destination, key, expected_shard_id);
#if defined(PAGE_COMPACTION_DEEP_PROFILE)
      page_compaction_profile.owner_publish_ns.fetch_add(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - owner_publish_started)
              .count(),
          std::memory_order_relaxed);
      page_compaction_profile.owner_publish_calls.fetch_add(
          1, std::memory_order_relaxed);
#endif
      simd::simd_copy(arena.data(source_index), arena.data(destination), dim);
      if (score_policy.enabled()) {
        const uint64_t score =
            std::atomic_ref<uint64_t>(arena.score_raw(source_index))
                .load(std::memory_order_relaxed);
        std::atomic_ref<uint64_t>(arena.score_raw(destination))
            .store(score, std::memory_order_relaxed);
      }
      if (optimizer.has_value()) {
        if (!optimizer_state.empty()) {
          optimizer->state_data(source_index, optimizer_state);
          optimizer->set_state(destination, optimizer_state);
        }
        optimizer->set_step(destination, optimizer->step_data(source_index));
      }
    }

    bool relocate_expected_key_with_locked_shard(
        uint64_t key, uint32_t source_index, uint64_t source_seq_count,
        uint32_t destination, uint32_t expected_shard_id, KeyShard &key_shard,
        std::span<OptT> optimizer_state, bool &locator_published) {
#if defined(PAGE_COMPACTION_DEEP_PROFILE)
      const auto locator_lookup_started = std::chrono::steady_clock::now();
#endif
      auto current = key_shard.find_locator_unlocked(key);
#if defined(PAGE_COMPACTION_DEEP_PROFILE)
      page_compaction_profile.locator_lookup_ns.fetch_add(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - locator_lookup_started)
              .count(),
          std::memory_order_relaxed);
      page_compaction_profile.locator_lookup_calls.fetch_add(
          1, std::memory_order_relaxed);
#endif
      if (!current.has_value() || current->index != source_index)
        return false;
#if defined(PAGE_COMPACTION_DEEP_PROFILE)
      const auto owner_validate_started = std::chrono::steady_clock::now();
#endif
      auto current_owner = arena.owner_snapshot(current->index);
#if defined(PAGE_COMPACTION_DEEP_PROFILE)
      page_compaction_profile.owner_validate_ns.fetch_add(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - owner_validate_started)
              .count(),
          std::memory_order_relaxed);
      page_compaction_profile.owner_validate_calls.fetch_add(
          1, std::memory_order_relaxed);
#endif
      if (!current_owner.has_value() || current_owner->key != key ||
          current_owner->shard_id != expected_shard_id ||
          current_owner->seq_count != source_seq_count) {
        return false;
      }
      if (key_shard.locators->requires_rotation(current->slot_id))
        return false;
#if defined(PAGE_COMPACTION_DEEP_PROFILE)
      const auto locator_publish_started = std::chrono::steady_clock::now();
#endif
      if (!key_shard.locators->relocate(
              current->slot_id, source_index, destination,
              [&](uint32_t locked_index) {
                copy_source_key_to_destination(key, locked_index, destination,
                                               expected_shard_id,
                                               optimizer_state);
                arena.mark_live(destination, true);
              }))
        return false;
#if defined(PAGE_COMPACTION_DEEP_PROFILE)
      page_compaction_profile.locator_publish_ns.fetch_add(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - locator_publish_started)
              .count(),
          std::memory_order_relaxed);
      page_compaction_profile.locator_publish_calls.fetch_add(
          1, std::memory_order_relaxed);
#endif
      locator_published = true;
      return true;
    }

    bool compact_key(uint64_t key) {
      auto &key_shard = shard_for_key(key);
      if (!with_shared_entry(key, [](uint32_t) {}))
        return false;
      uint32_t destination;
      if (auto recycled = key_shard.take_recycled_index()) {
        destination = *recycled;
        try {
          arena.prepare_reused(destination);
          if (optimizer.has_value())
            optimizer->allocate(destination);
        } catch (...) {
          key_shard.recycle_unpublished_index(destination);
          throw;
        }
      } else {
        key_shard.reserve_new_index();
        destination =
            allocate_entry(static_cast<uint32_t>(shard_id_for_key(key)));
      }
      return relocate_key_to_destination(key, destination, false);
    }

    size_t compact_page(uint64_t page_id) {
      return compact_pages(std::span<const uint64_t>(&page_id, 1), false);
    }

    // Promote only the hottest keys from one physical page.  The source page
    // is put into draining state so no new allocation can reuse its slots,
    // but it is deliberately not retired: cold keys remain addressable in
    // that page.  The authoritative shard directory is updated by the same
    // relocation transaction used by full compaction.
    size_t promote_hot_keys(uint64_t page_id, size_t maximum_keys,
                            uint64_t minimum_score) {
      if (!score_policy.enabled()) {
        throw std::runtime_error(
            "hot-key promotion requires a score-enabled policy");
      } else {
        if (maximum_keys == 0)
          throw std::invalid_argument("hot-key promotion maximum is zero");
        if (page_id >= arena.page_count())
          throw std::out_of_range("hot-key promotion page id is out of range");
        if (!arena.page_allocated(page_id))
          throw std::logic_error("hot-key promotion source page is retired");

        struct Candidate {
          uint64_t key;
          uint64_t score;
        };
        std::vector<Candidate> candidates;
        candidates.reserve(arena.entries_per_page());
        const uint32_t first =
            static_cast<uint32_t>(page_id * arena.entries_per_page());
        const uint32_t limit = static_cast<uint32_t>(
            std::min(arena.allocated_count(),
                     static_cast<size_t>(first) + arena.entries_per_page()));
        for (uint32_t index = first; index < limit; ++index) {
          auto owner = arena.owner_snapshot(index);
          if (!owner.has_value())
            continue;
          const uint64_t score =
              std::atomic_ref<uint64_t>(arena.score_raw(index))
                  .load(std::memory_order_relaxed);
          if (score >= minimum_score)
            candidates.push_back({owner->key, score});
        }
        std::ranges::sort(candidates,
                          [](const Candidate &left, const Candidate &right) {
                            if (left.score != right.score)
                              return left.score > right.score;
                            return left.key < right.key;
                          });
        if (candidates.size() > maximum_keys)
          candidates.resize(maximum_keys);
        if (candidates.empty())
          return 0;

        page_reuse->begin_draining(page_id);
        arena.advance_allocation_past_page(page_id);
        auto reservation = sealed_pages.reserve(
            PageThermalTier::kHot, static_cast<uint32_t>(page_id),
            sealed_page_layout_generation.fetch_add(1,
                                                    std::memory_order_relaxed) +
                1);
        size_t promoted = 0;
        for (const Candidate &candidate : candidates) {
          if (!compact_key(candidate.key))
            continue;
          with_shared_entry(candidate.key, [&](uint32_t index) {
            auto owner = arena.owner_snapshot(index);
            if (!owner.has_value() || owner->key != candidate.key)
              throw std::logic_error("promoted hot-key owner is unstable");
            reservation.builder.append(
                candidate.key, owner->seq_count,
                std::span<const ParamT>(arena.data(index), dim));
          });
          ++promoted;
        }
        if (promoted != 0) {
          auto published = sealed_pages.publish(std::move(reservation));
          std::lock_guard lock(promoted_hot_page_mutex);
          last_promoted_hot_page = std::move(published);
        }
        return promoted;
      }
    }

    size_t promoted_hot_page_size() const {
      std::lock_guard lock(promoted_hot_page_mutex);
      if (last_promoted_hot_page == nullptr)
        return 0;
      return last_promoted_hot_page->layout().entry_count;
    }

    size_t compact_pages(std::span<const uint64_t> page_ids,
                         bool cluster_by_coaccess) {
#if defined(PAGE_COMPACTION_PROFILE)
      const auto enumeration_started = std::chrono::steady_clock::now();
#endif
      if (page_ids.empty())
        throw std::invalid_argument("compaction requires at least one page");
      auto pressure_call = retirement_pressure->writer_call();
      pressure_call.apply();
      [[maybe_unused]] auto retirement_admission =
          retirement_pressure->control_plane_call();
      auto workspace_lease = compaction_workspaces.acquire();
      CompactionWorkspace &workspace = *workspace_lease;
      workspace.sorted_page_ids.assign(page_ids.begin(), page_ids.end());
      auto &sorted_page_ids = workspace.sorted_page_ids;
      std::ranges::sort(sorted_page_ids);
      if (std::ranges::adjacent_find(sorted_page_ids) != sorted_page_ids.end())
        throw std::invalid_argument("compaction pages contain duplicate ids");
      const uint64_t last_page_id = sorted_page_ids.back();
      if (last_page_id >= arena.page_count())
        throw std::out_of_range("compaction page id is out of range");
      for (uint64_t page_id : sorted_page_ids) {
        if (!arena.page_allocated(page_id))
          throw std::logic_error("compaction source page is not allocated");
      }
      size_t drained_count = 0;
      try {
        for (uint64_t page_id : sorted_page_ids) {
          page_reuse->begin_draining(page_id);
          ++drained_count;
        }
      } catch (...) {
        while (drained_count != 0)
          page_reuse->cancel_draining(sorted_page_ids[--drained_count]);
        throw;
      }
      struct DrainResolution {
        PageReuseState &reuse;
        std::span<const uint64_t> page_ids;
        bool completed = false;

        ~DrainResolution() {
          if (!completed) {
            for (uint64_t page_id : page_ids)
              reuse.cancel_draining(page_id);
          }
        }
      } drain_resolution{*page_reuse, sorted_page_ids};
      arena.advance_allocation_past_page(last_page_id);

      auto &pending = workspace.pending;
      pending.clear();
      pending.reserve(sorted_page_ids.size() * arena.entries_per_page());
      workspace.source_cohort_shards.assign(
          sorted_page_ids.size(), std::numeric_limits<uint32_t>::max());
      auto &source_cohort_shards = workspace.source_cohort_shards;
      std::array<uint64_t, CoaccessSketch::kMaximumBatchSample> anchors{};
      size_t anchor_count = 0;
      for (size_t source_page_position = 0;
           source_page_position < sorted_page_ids.size();
           ++source_page_position) {
        const uint64_t page_id = sorted_page_ids[source_page_position];
        const uint32_t first =
            static_cast<uint32_t>(page_id * arena.entries_per_page());
        const uint32_t limit = static_cast<uint32_t>(
            std::min(arena.allocated_count(),
                     static_cast<size_t>(first) + arena.entries_per_page()));
        for (uint32_t index = first; index < limit; ++index) {
          auto owner = arena.owner_snapshot(index);
          if (!owner.has_value())
            continue;
          if (owner->shard_id >= key_shards.size() ||
              owner->shard_id != shard_id_for_key(owner->key)) {
            throw std::logic_error(
                "arena source owner does not match key shard");
          }
          uint32_t &source_shard = source_cohort_shards[source_page_position];
          if (source_shard == std::numeric_limits<uint32_t>::max()) {
            source_shard = owner->shard_id;
          } else if (source_shard / arena.topology().cohort_size() !=
                     owner->shard_id / arena.topology().cohort_size()) {
            throw std::logic_error(
                "compaction source page mixes shard cohorts");
          }
          pending.push_back({owner->key, owner->seq_count, index,
                             owner->shard_id,
                             static_cast<uint32_t>(source_page_position), 0});
          if (anchor_count < anchors.size())
            anchors[anchor_count++] = owner->key;
        }
      }
      for (size_t source_page_position = 0;
           source_page_position < sorted_page_ids.size();
           ++source_page_position) {
        const uint32_t source_shard =
            source_cohort_shards[source_page_position];
        if (source_shard == std::numeric_limits<uint32_t>::max())
          continue;
        arena.bind_exact(
            static_cast<uint32_t>(sorted_page_ids[source_page_position] *
                                  arena.entries_per_page()),
            source_shard);
      }
      if (cluster_by_coaccess && anchor_count > 1) {
        for (PendingKey &entry : pending) {
          uint32_t strongest = 0;
          uint8_t cluster = static_cast<uint8_t>(anchor_count);
          for (size_t candidate = 0; candidate < anchor_count; ++candidate) {
            if (entry.key == anchors[candidate]) {
              cluster = static_cast<uint8_t>(candidate);
              break;
            }
            const uint64_t affinity = access_telemetry.coaccess_estimate(
                entry.key, anchors[candidate]);
            if (affinity > strongest) {
              strongest = affinity;
              cluster = static_cast<uint8_t>(candidate);
            }
          }
          entry.cluster = cluster;
        }
        std::ranges::stable_sort(
            pending, [](const PendingKey &left, const PendingKey &right) {
              if (left.cluster != right.cluster)
                return left.cluster < right.cluster;
              return left.key < right.key;
            });
      }
#if defined(PAGE_COMPACTION_PROFILE)
      page_compaction_profile.enumeration_ns.fetch_add(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - enumeration_started)
              .count(),
          std::memory_order_relaxed);
      const auto preparation_started = std::chrono::steady_clock::now();
#endif
      size_t migrated = 0;
      const size_t live_words_per_page = arena.entries_per_page() / 64;
      workspace.source_clear_masks.assign(
          sorted_page_ids.size() * live_words_per_page, 0);
      auto &source_clear_masks = workspace.source_clear_masks;
      if (!pending.empty()) {
        workspace.shard_counts.assign(key_shards.size(), 0);
        auto &shard_counts = workspace.shard_counts;
        for (const PendingKey &entry : pending)
          ++shard_counts[entry.shard_id];
        for (size_t shard_id = 0; shard_id < key_shards.size(); ++shard_id) {
          if (shard_counts[shard_id] != 0)
            key_shards[shard_id]->reserve_new_indices(shard_counts[shard_id]);
        }
        workspace.migration_order.resize(pending.size());
        workspace.shard_offsets.assign(key_shards.size() + 1, 0);
        auto &migration_order = workspace.migration_order;
        auto &shard_offsets = workspace.shard_offsets;
        for (size_t shard_id = 0; shard_id < key_shards.size(); ++shard_id)
          shard_offsets[shard_id + 1] =
              shard_offsets[shard_id] + shard_counts[shard_id];
        workspace.shard_positions.assign(shard_offsets.begin(),
                                         shard_offsets.end() - 1);
        auto &shard_positions = workspace.shard_positions;
        for (size_t pending_index = 0; pending_index < pending.size();
             ++pending_index) {
          const uint32_t shard_id = pending[pending_index].shard_id;
          migration_order[shard_positions[shard_id]++] = pending_index;
        }
        workspace.destination_indices.resize(pending.size());
        auto &destination_indices = workspace.destination_indices;
        workspace.destination_pending.assign(pending.size(), 0);
        auto &destination_pending = workspace.destination_pending;
        try {
          if (cluster_by_coaccess) {
            for (size_t pending_index = 0; pending_index < pending.size();
                 ++pending_index) {
              const PendingKey &entry = pending[pending_index];
              KeyShard &key_shard = *key_shards[entry.shard_id];
              uint32_t destination;
              if (auto recycled = key_shard.take_recycled_index()) {
                destination = *recycled;
                try {
                  if (!page_reuse->destination_available(destination))
                    throw std::logic_error(
                        "recycled compaction destination is unavailable");
                  arena.prepare_reused(destination);
                  if (optimizer.has_value())
                    optimizer->allocate(destination);
                } catch (...) {
                  key_shard.recycle_unpublished_index(destination);
                  throw;
                }
              } else {
                destination = allocate_compaction_destination(entry.shard_id);
              }
              destination_indices[pending_index] = destination;
              destination_pending[pending_index] = 1;
            }
          } else {
            std::vector<uint32_t> fresh_pending_indices;
            for (size_t shard_id = 0; shard_id < key_shards.size();
                 ++shard_id) {
              const size_t order_begin = shard_offsets[shard_id];
              const size_t order_end = shard_offsets[shard_id + 1];
              if (order_begin == order_end)
                continue;
              fresh_pending_indices.clear();
              fresh_pending_indices.reserve(order_end - order_begin);
              KeyShard &key_shard = *key_shards[shard_id];
              for (size_t order_index = order_begin; order_index < order_end;
                   ++order_index) {
                const size_t pending_index = migration_order[order_index];
                if (auto recycled = key_shard.take_recycled_index()) {
                  const uint32_t destination = *recycled;
                  try {
                    if (!page_reuse->destination_available(destination))
                      throw std::logic_error(
                          "recycled compaction destination is unavailable");
                    arena.prepare_reused(destination);
                    if (optimizer.has_value())
                      optimizer->allocate(destination);
                    destination_indices[pending_index] = destination;
                    destination_pending[pending_index] = 1;
                  } catch (...) {
                    key_shard.recycle_unpublished_index(destination);
                    throw;
                  }
                } else {
                  fresh_pending_indices.push_back(
                      static_cast<uint32_t>(pending_index));
                }
              }
              if (fresh_pending_indices.empty())
                continue;
              std::vector<uint32_t> allocated(fresh_pending_indices.size());
              allocate_compaction_destinations(static_cast<uint32_t>(shard_id),
                                               allocated.size(), allocated);
              for (size_t position = 0; position < allocated.size();
                   ++position) {
                const size_t pending_index = fresh_pending_indices[position];
                destination_indices[pending_index] = allocated[position];
                destination_pending[pending_index] = 1;
              }
            }
          }
#if defined(PAGE_COMPACTION_PROFILE)
          page_compaction_profile.reservation_ns.fetch_add(
              std::chrono::duration_cast<std::chrono::nanoseconds>(
                  std::chrono::steady_clock::now() - preparation_started)
                  .count(),
              std::memory_order_relaxed);
#endif
          const size_t optimizer_state_dim =
              optimizer.has_value() ? optimizer->state_dim() : 0;
          workspace.optimizer_state.resize(optimizer_state_dim);
          auto &optimizer_state = workspace.optimizer_state;
#if defined(PAGE_COMPACTION_PROFILE)
          page_compaction_profile.preparation_ns.fetch_add(
              std::chrono::duration_cast<std::chrono::nanoseconds>(
                  std::chrono::steady_clock::now() - preparation_started)
                  .count(),
              std::memory_order_relaxed);
          const auto publication_started = std::chrono::steady_clock::now();
#endif
          workspace.pending_shards.assign(key_shards.size(), 0);
          workspace.next_shard_order.assign(shard_offsets.begin(),
                                            shard_offsets.end() - 1);
          auto &pending_shards = workspace.pending_shards;
          auto &next_shard_order = workspace.next_shard_order;
          size_t remaining_shards = 0;
          for (size_t shard_id = 0; shard_id < key_shards.size(); ++shard_id) {
            if (shard_offsets[shard_id] == shard_offsets[shard_id + 1])
              continue;
            pending_shards[shard_id] = 1;
            ++remaining_shards;
          }
          constexpr size_t kPublicationPagesPerChunk = 1;
          const size_t publication_chunk_size = std::max(
              size_t{1}, kPublicationPagesPerChunk * arena.entries_per_page() /
                             key_shards.size());
          const size_t rotation = sorted_page_ids.front() & shard_mask;
          while (remaining_shards != 0) {
            bool progressed = false;
            for (size_t shard_offset = 0; shard_offset < key_shards.size();
                 ++shard_offset) {
              const size_t shard_id = (rotation + shard_offset) & shard_mask;
              if (pending_shards[shard_id] == 0)
                continue;
              auto &key_shard = *key_shards[shard_id];
#if defined(PAGE_COMPACTION_SHARD_PROFILE)
              const auto shard_hold_started = std::chrono::steady_clock::now();
#endif
              const size_t order_begin = next_shard_order[shard_id];
              const size_t order_end =
                  std::min(order_begin + publication_chunk_size,
                           shard_offsets[shard_id + 1]);
              for (size_t order_index = order_begin; order_index < order_end;
                   ++order_index) {
                const size_t pending_index = migration_order[order_index];
                const PendingKey &entry = pending[pending_index];
                const uint32_t destination = destination_indices[pending_index];
                bool locator_published = false;
                try {
                  if (relocate_expected_key_with_locked_shard(
                          entry.key, entry.source_index, entry.seq_count,
                          destination, entry.shard_id, key_shard,
                          std::span<OptT>(optimizer_state),
                          locator_published)) {
                    ++migrated;
#if defined(PAGE_COMPACTION_DEEP_PROFILE)
                    const auto owner_clear_started =
                        std::chrono::steady_clock::now();
#endif
                    arena.mark_owner_not_live(entry.source_index);
#if defined(PAGE_COMPACTION_DEEP_PROFILE)
                    page_compaction_profile.owner_clear_ns.fetch_add(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() -
                            owner_clear_started)
                            .count(),
                        std::memory_order_relaxed);
                    page_compaction_profile.owner_clear_calls.fetch_add(
                        1, std::memory_order_relaxed);
#endif
                    const size_t source_slot =
                        entry.source_index & arena.slot_mask();
                    source_clear_masks[entry.source_page_position *
                                           live_words_per_page +
                                       (source_slot / 64)] |=
                        uint64_t{1} << (source_slot % 64);
                  } else {
                    key_shard.recycle_unpublished_index(destination);
                  }
                  destination_pending[pending_index] = 0;
                } catch (...) {
                  if (!locator_published)
                    key_shard.recycle_unpublished_index(destination);
                  destination_pending[pending_index] = 0;
                  throw;
                }
              }
              next_shard_order[shard_id] = order_end;
              if (order_end == shard_offsets[shard_id + 1]) {
                pending_shards[shard_id] = 0;
                --remaining_shards;
              }
#if defined(PAGE_COMPACTION_SHARD_PROFILE)
              const auto shard_hold_ended = std::chrono::steady_clock::now();
              record_page_compaction_shard_acquisition(
                  shard_id, order_end - order_begin,
                  static_cast<uint64_t>(
                      std::chrono::duration_cast<std::chrono::nanoseconds>(
                          shard_hold_ended - shard_hold_started)
                          .count()));
#endif
              progressed = true;
            }
            if (!progressed)
              throw std::runtime_error(
                  "compaction publication made no progress");
          }
#if defined(PAGE_COMPACTION_PROFILE)
          page_compaction_profile.publication_ns.fetch_add(
              std::chrono::duration_cast<std::chrono::nanoseconds>(
                  std::chrono::steady_clock::now() - publication_started)
                  .count(),
              std::memory_order_relaxed);
#endif
        } catch (...) {
          for (size_t remaining = 0; remaining < pending.size(); ++remaining) {
            if (destination_pending[remaining] == 0)
              continue;
            const uint32_t destination = destination_indices[remaining];
            key_shards[pending[remaining].shard_id]->recycle_unpublished_index(
                destination);
          }
          throw;
        }
      }
#if defined(PAGE_COMPACTION_PROFILE)
      const auto retirement_started = std::chrono::steady_clock::now();
#endif
      for (size_t source_page_position = 0;
           source_page_position < sorted_page_ids.size();
           ++source_page_position) {
        const uint32_t first = static_cast<uint32_t>(
            sorted_page_ids[source_page_position] * arena.entries_per_page());
        for (size_t word = 0; word < live_words_per_page; ++word) {
          const uint64_t mask =
              source_clear_masks[source_page_position * live_words_per_page +
                                 word];
          if (mask != 0)
            arena.clear_live_mask(first / 64 + word, mask);
        }
      }
      for (uint64_t page_id : sorted_page_ids) {
        if (arena.page_has_live_entries(page_id))
          throw std::runtime_error("draining arena page retained a live entry");
        page_reuse->retire(page_id);
        arena.retire_empty_page(page_id, [reuse = page_reuse,
                                          telemetry = page_telemetry,
                                          page_id]() noexcept {
          telemetry->mark_nonresident(static_cast<uint32_t>(page_id));
          reuse->make_available(page_id);
        });
      }
#if defined(PAGE_COMPACTION_PROFILE)
      page_compaction_profile.retirement_ns.fetch_add(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - retirement_started)
              .count(),
          std::memory_order_relaxed);
#endif
      drain_resolution.completed = true;
      return migrated;
    }
  };

  struct StoreSnapshot {
    StoreSnapshot(std::vector<SlotStore *> dims,
                  std::vector<SlotStore *> slot_list)
        : dim_to_slot(std::move(dims)), slots(std::move(slot_list)) {}

    size_t retirement_bytes() const {
      const size_t pointer_capacity = dim_to_slot.capacity() + slots.capacity();
      if (pointer_capacity < dim_to_slot.capacity() ||
          pointer_capacity >
              (std::numeric_limits<size_t>::max() - sizeof(StoreSnapshot)) /
                  sizeof(SlotStore *)) {
        throw std::overflow_error("store snapshot retirement bytes overflow");
      }
      return sizeof(StoreSnapshot) + pointer_capacity * sizeof(SlotStore *);
    }

    std::vector<SlotStore *> dim_to_slot;
    std::vector<SlotStore *> slots;
  };

  struct DeferredSnapshot {
    StoreSnapshot *snapshot;

    explicit DeferredSnapshot(StoreSnapshot *retired) : snapshot(retired) {}

    ~DeferredSnapshot() { delete snapshot; }
  };

  struct NumaForwardWorkspace {
    std::vector<std::vector<uint32_t>> indices_by_node;
    std::vector<size_t> hashes;
  };

  StoreSnapshot *make_store_snapshot() const {
    std::vector<SlotStore *> slots;
    slots.reserve(slot_storage_.size());
    for (const auto &slot : slot_storage_)
      slots.push_back(slot.get());
    return new StoreSnapshot(dim_to_slot_, std::move(slots));
  }

  void publish_store_snapshot() {
    std::unique_ptr<StoreSnapshot> next(make_store_snapshot());
    StoreSnapshot *current = snapshot_.load(std::memory_order_acquire);
    if (current == nullptr)
      throw std::logic_error("store snapshot is unavailable");
    se::concurrency::EpochRetirement<DeferredSnapshot> retirement(
        retirement_pressure_, current->retirement_bytes());
    StoreSnapshot *retired =
        snapshot_.exchange(next.release(), std::memory_order_acq_rel);
    retirement.emplace(retired);
    retirement.publish();
  }

  static PageStorage require_storage(PageStorage storage) {
    if (!storage)
      throw std::invalid_argument("hogwild storage is empty");
    return storage;
  }

  static size_t checked_slot_table_bytes(size_t max_dim) {
    if (max_dim >
        (std::numeric_limits<size_t>::max() - 1) / sizeof(SlotStore *)) {
      throw std::overflow_error("slot index table size overflows");
    }
    return (max_dim + 1) * sizeof(SlotStore *);
  }

  static const std::vector<size_t> &
  validate_table_dims(const std::vector<size_t> &table_dims,
                      const HogwildStoreOptions &options) {
    if (table_dims.empty()) {
      throw std::invalid_argument("table_dims must not be empty");
    }
    size_t memory_budget = options.storage_tuning.memory_budget_bytes;
    if (memory_budget == 0) {
      const size_t available_ram =
          storage::SystemResourceProbe::available_ram_bytes();
      memory_budget = static_cast<size_t>(available_ram * 0.75);
    }
    const size_t max_d = std::ranges::max(table_dims);
    const size_t slot_table_bytes = checked_slot_table_bytes(max_d);
    if (slot_table_bytes > memory_budget) {
      throw std::invalid_argument(
          "dim is too large: slot index table exceeds memory budget");
    }
    for (size_t dim : table_dims) {
      if (dim == 0) {
        throw std::invalid_argument("dim must be > 0");
      }
      storage::StorageResourceTuner::derive(storage::TopologyTuneRequest{
          .thread_budget = options.thread_budget,
          .locality = options.locality,
          .embedding_dim = dim,
          .param_bytes_per_element = sizeof(ParamT),
          .tuning = options.storage_tuning,
      });
    }
    return table_dims;
  }

  explicit HogwildEmbeddingStoreCore(const std::vector<size_t> &table_dims,
                                     Initializer initializer,
                                     OptimizerFactory opt_factory,
                                     ScorePolicyKind score_policy_kind,
                                     HogwildStoreOptions options = {})
      : storage_topology_(
            storage::StorageResourceTuner::derive(storage::TopologyTuneRequest{
                .thread_budget = options.thread_budget,
                .locality = options.locality,
                .embedding_dim =
                    validate_table_dims(table_dims, options).front(),
                .param_bytes_per_element = sizeof(ParamT),
                .tuning = options.storage_tuning,
            })),
        num_threads_(storage_topology_.execution_workers()),
        operation_dispatch_topology_(OperationDispatchPolicy::make_topology(
            num_threads_, options.dispatch_config())),
        pool_(static_cast<uint32_t>(
            operation_dispatch_topology_.physical_threads)),
        initializer_(std::move(initializer)), score_policy_(score_policy_kind),
        storage_(require_storage(std::move(options.storage))),
        retirement_pressure_(storage_->stats().retirement_pressure_domain),
        background_page_eviction_enabled_(
            score_policy_.enabled() &&
            storage_->supports_background_page_eviction()),
        bind_worker_affinity_(storage_topology_.numa_execution()),
        numa_memory_locality_(storage_topology_.numa_memory()),
        telemetry_ring_(1024,
                        bind_worker_affinity_
                            ? std::optional<int>(
                                  ThreadAffinity::cpu_for_worker(num_threads_))
                            : std::nullopt,
                        &write_diagnostic_to_clog) {
    checkpoint_dirty_heads_ =
        std::make_unique<ShardDirtyHead[]>(storage_topology_.shard_count());
    for (size_t shard = 0; shard < storage_topology_.shard_count(); ++shard)
      checkpoint_dirty_heads_[shard].head.store(nullptr,
                                                std::memory_order_relaxed);
    bool enable_score = score_policy_.enabled();

    if (numa_memory_locality_) {
      if (!NumaTopology::available())
        throw std::runtime_error(
            "NUMA-aware sharding requires libnuma support");
      const size_t nodes = NumaTopology::node_count();
      if (nodes == 0 || !std::has_single_bit(nodes) ||
          storage_topology_.shard_count() < nodes ||
          storage_topology_.shard_count() % nodes != 0) {
        throw std::invalid_argument(
            "NUMA-aware sharding requires a power-of-two node count that "
            "divides the directory shard count");
      }
    }

    max_dim_ = std::ranges::max(table_dims);
    dim_to_slot_.resize(max_dim_ + 1, nullptr);

    for (size_t dim : table_dims) {
      if (dim_to_slot_[dim] == nullptr) {
        slot_storage_.push_back(std::make_unique<SlotStore>(
            dim, score_policy_, opt_factory, storage_topology_, num_threads_,
            enable_score, storage_, retirement_pressure_, telemetry_ring_,
            options.gradient_dtype, options.gradient_scale, *this));
        dim_to_slot_[dim] = slot_storage_.back().get();
      }
    }
    snapshot_.store(make_store_snapshot(), std::memory_order_release);

    telemetry_ring_.start();

    if (background_page_eviction_enabled_) {
      auto proxy = pro::make_proxy_shared<PageEvictionProxyFacade,
                                          StorePageEvictionProxy>();
      page_eviction_registration_ =
          storage_->set_page_eviction_proxy(std::move(proxy));
    }
  }

  HogwildEmbeddingStoreCore(const std::vector<size_t> &table_dims,
                            Initializer initializer,
                            OptimizerFactory opt_factory, NoScorePolicy,
                            HogwildStoreOptions options = {})
      : HogwildEmbeddingStoreCore(table_dims, std::move(initializer),
                                  std::move(opt_factory),
                                  ScorePolicyKind::kNone, std::move(options)) {}

  HogwildEmbeddingStoreCore(const std::vector<size_t> &table_dims,
                            Initializer initializer,
                            OptimizerFactory opt_factory, LfuScorePolicy,
                            HogwildStoreOptions options = {})
      : HogwildEmbeddingStoreCore(table_dims, std::move(initializer),
                                  std::move(opt_factory), ScorePolicyKind::kLfu,
                                  std::move(options)) {}

  HogwildEmbeddingStoreCore(const std::vector<size_t> &table_dims,
                            Initializer initializer,
                            OptimizerFactory opt_factory,
                            const LruScorePolicy &,
                            HogwildStoreOptions options = {})
      : HogwildEmbeddingStoreCore(table_dims, std::move(initializer),
                                  std::move(opt_factory), ScorePolicyKind::kLru,
                                  std::move(options)) {}

  HogwildEmbeddingStoreCore(const std::vector<size_t> &table_dims,
                            Initializer initializer,
                            OptimizerFactory opt_factory, ManualScorePolicy,
                            HogwildStoreOptions options = {})
      : HogwildEmbeddingStoreCore(
            table_dims, std::move(initializer), std::move(opt_factory),
            ScorePolicyKind::kManual, std::move(options)) {}

  ~HogwildEmbeddingStoreCore() {
    invocation_gate_.close_and_wait();
    telemetry_ring_.stop();
    if (page_eviction_registration_.value != 0)
      storage_->clear_page_eviction_proxy(page_eviction_registration_);
    delete snapshot_.exchange(nullptr, std::memory_order_acq_rel);
    slot_storage_.clear();
    dim_to_slot_.clear();
  }

  size_t max_dim() const { return max_dim_; }

  size_t entries_per_page() const noexcept {
    return storage_topology_.entries_per_page();
  }

  std::string_view name() const { return "hogwild"; }

  exec::static_thread_pool &pool() { return pool_; }

  // =========================================================================
  // Forward Pass: Lock-free lookup via ParlayHash
  // =========================================================================

  template <typename K>
  std::error_code forward_lookup(std::span<const K> keys,
                                 std::span<ParamT> out_flat_values,
                                 std::span<const int64_t> row_lengths) {
    ActiveInvocationGuard invocation(invocation_gate_, &storage_);
    StoreSnapshot *snapshot = snapshot_.load(std::memory_order_acquire);
    if (snapshot == nullptr)
      throw std::runtime_error("store snapshot is unavailable");
    size_t n = keys.size();
    thread_local std::vector<size_t> forward_offsets;
    const std::optional<size_t> fixed_dim = fixed_row_length(row_lengths);
    if (!fixed_dim.has_value())
      compute_offsets_into(row_lengths, forward_offsets);
    const std::span<const size_t> offsets(forward_offsets);
    const size_t bytes_per_entry =
        n == 0                  ? sizeof(ParamT)
        : fixed_dim.has_value() ? checked_row_payload_bytes(*fixed_dim)
                                : row_payload_bytes(row_lengths, offsets);
    const size_t logical_worker_count =
        n == 0 ? 1
               : operation_worker_count(HogwildOperation::kForwardLookup, n,
                                        bytes_per_entry);
    const size_t requested_task_count =
        n == 0 ? 1
               : OperationDispatchPolicy::task_count(
                     HogwildOperation::kForwardLookup, n, logical_worker_count,
                     operation_dispatch_topology_);
    const size_t payload_capped_task_count =
        cap_forward_tasks_by_row_granularity(requested_task_count, n,
                                             bytes_per_entry);
    const size_t worker_count =
        OperationDispatchPolicy::concurrency_limited_thread_count(
            payload_capped_task_count,
            operation_dispatch_topology_.memory_parallelism,
            invocation.count());
    constexpr size_t kTelemetrySamplingPeriod = 64;
    const uint64_t telemetry_sequence =
        telemetry_sample_sequence_.fetch_add(1, std::memory_order_relaxed);
    const bool collect_telemetry =
        telemetry_sequence % kTelemetrySamplingPeriod == 0;

    const size_t telemetry_partition_base = n / worker_count;
    const size_t telemetry_partition_remainder = n % worker_count;
    const auto telemetry_partition_offset = [&](size_t partition) {
      return partition * telemetry_partition_base +
             std::min(partition, telemetry_partition_remainder);
    };
    std::span<const size_t> routed_hashes;
    auto do_indices = [&](size_t worker, auto &&index_at, size_t count) {
      se::concurrency::with_epoch([&] {
        std::array<uint64_t, CoaccessSketch::kMaximumBatchSample> sampled_keys;
        std::array<SlotStore *, CoaccessSketch::kMaximumBatchSample>
            sampled_slots;
        std::array<uint32_t, CoaccessSketch::kMaximumBatchSample>
            sampled_indices;
        size_t sampled_count = 0;
        for (size_t position = 0; position < count; ++position) {
          const size_t i = index_at(position);
          const bool collects_telemetry =
              collect_telemetry && i >= telemetry_partition_offset(worker) &&
              i < telemetry_partition_offset(worker + 1);
          const size_t prefetch_distance =
              operation_dispatch_topology_.forward_prefetch_distance;
          if (prefetch_distance != 0 && position + prefetch_distance < count) {
            const size_t prefetch_index =
                index_at(position + prefetch_distance);
            __builtin_prefetch(&keys[prefetch_index], 0, 1);
            __builtin_prefetch(&row_lengths[prefetch_index], 0, 1);
            if (fixed_dim.has_value()) {
              __builtin_prefetch(
                  out_flat_values.data() + prefetch_index * *fixed_dim, 1, 1);
            } else {
              __builtin_prefetch(&offsets[prefetch_index], 0, 1);
              __builtin_prefetch(
                  out_flat_values.data() + offsets[prefetch_index], 1, 1);
            }
            const uint64_t prefetch_key =
                static_cast<uint64_t>(keys[prefetch_index]);
            const size_t prefetch_dim = fixed_dim.value_or(
                static_cast<size_t>(row_lengths[prefetch_index]));
            if (prefetch_dim < snapshot->dim_to_slot.size() &&
                snapshot->dim_to_slot[prefetch_dim] != nullptr) [[likely]] {
              auto &prefetch_slot = *snapshot->dim_to_slot[prefetch_dim];
              const size_t prefetch_hash =
                  !routed_hashes.empty()
                      ? routed_hashes[prefetch_index]
                      : SplitMixHash<uint64_t>{}(prefetch_key);
              prefetch_slot.prefetch_entry_cacheline(prefetch_key,
                                                     prefetch_hash);
            }
          }
          uint64_t key = static_cast<uint64_t>(keys[i]);
          size_t dim = fixed_dim.value_or(static_cast<size_t>(row_lengths[i]));
          if (dim >= snapshot->dim_to_slot.size() ||
              snapshot->dim_to_slot[dim] == nullptr) [[unlikely]] {
            throw std::runtime_error(
                "invalid dim access: no physical table for dim " +
                std::to_string(dim));
          }
          auto &slot = *snapshot->dim_to_slot[dim];
          ParamT *out = out_flat_values.data() +
                        (fixed_dim.has_value() ? i * *fixed_dim : offsets[i]);

          auto copy_and_collect_telemetry = [&](uint32_t index) {
            const ParamT *src = slot.arena.data_trusted(index);
            if constexpr (simd::is_simd_supported_v<ParamT>) {
              simd::simd_copy(src, out, dim);
            } else {
              std::copy_n(src, dim, out);
            }
            if (collects_telemetry && sampled_count < sampled_keys.size()) {
              sampled_keys[sampled_count] = key;
              sampled_slots[sampled_count] = &slot;
              sampled_indices[sampled_count] = index;
              ++sampled_count;
            }
          };
          auto read_visit = [&](uint32_t index) {
            copy_and_collect_telemetry(index);
          };
          auto scored_visit =
              [&](uint32_t index,
                  hogwild_internal::CheckpointRecordRef checkpoint_record) {
                copy_and_collect_telemetry(index);
                const auto score_result =
                    score_policy_.on_access(&slot.arena.score_raw(index));
                if (score_result == ScoreAccessResult::kUpdated) {
                  record_checkpoint_mutation(checkpoint_record);
                } else if (score_result == ScoreAccessResult::kClockExhausted) {
                  mark_terminal_failure();
                  throw std::overflow_error(
                      "LRU logical clock exhausted; store must be recreated");
                }
              };
          const bool found = [&] {
            const size_t hash = !routed_hashes.empty()
                                    ? routed_hashes[i]
                                    : SplitMixHash<uint64_t>{}(key);
            if (!score_policy_.enabled()) {
              return slot.with_read_view_entry_hashed(key, hash, read_visit);
            }
            return slot.with_shared_write_entry_hashed(key, hash, scored_visit);
          }();
          if (!found) {
            std::fill_n(out, dim, ParamT{0});
          }
        }
        if (sampled_count == 0)
          return;
        for (size_t sample = 0; sample < sampled_count; ++sample) {
          bool first_for_slot = true;
          for (size_t earlier = 0; earlier < sample; ++earlier) {
            if (sampled_slots[earlier] == sampled_slots[sample]) {
              first_for_slot = false;
              break;
            }
          }
          if (!first_for_slot)
            continue;
          TelemetryRing::Event event;
          event.target = sampled_slots[sample];
          event.consume = &SlotStore::consume_telemetry;
          event.stripe = static_cast<uint32_t>(worker);
          event.weight =
              static_cast<uint32_t>(worker_count * kTelemetrySamplingPeriod);
          for (size_t other = 0; other < sampled_count; ++other) {
            if (sampled_slots[other] != sampled_slots[sample])
              continue;
            event.page_ids[event.page_count++] =
                sampled_indices[other] >>
                sampled_slots[other]->arena.entries_shift();
            event.keys[event.key_count++] = sampled_keys[other];
          }
          telemetry_ring_.try_publish(event);
        }
      });
    };
    auto do_range = [&](size_t worker, size_t begin, size_t end) {
      do_indices(
          worker, [begin](size_t position) { return begin + position; },
          end - begin);
    };

    const size_t node_count =
        numa_memory_locality_ ? NumaTopology::node_count() : 1;
    constexpr size_t kMinimumPayloadForNumaRouting = 512 * 1024;
    const bool payload_is_large_enough_for_numa_routing =
        n >
        kMinimumPayloadForNumaRouting / std::max<size_t>(1, bytes_per_entry);
    if (!numa_memory_locality_ || !payload_is_large_enough_for_numa_routing ||
        worker_count < node_count || worker_count % node_count != 0) {
      dispatch_parallel_with_worker_limited(n, do_range, worker_count);
      return {};
    }
    auto workspace_lease = numa_forward_workspaces_.acquire();
    NumaForwardWorkspace &workspace = *workspace_lease;
    workspace.hashes.resize(n);
    workspace.indices_by_node.resize(node_count);
    for (auto &indices : workspace.indices_by_node)
      indices.clear();
    for (size_t i = 0; i < n; ++i) {
      const size_t hash = SplitMixHash<K>{}(keys[i]);
      workspace.hashes[i] = hash;
      const size_t shard = hash & storage_topology_.shard_mask();
      workspace
          .indices_by_node[NumaTopology::node_ordinal_for_shard(
              shard, storage_topology_.shard_count())]
          .push_back(static_cast<uint32_t>(i));
    }
    routed_hashes = std::span<const size_t>(workspace.hashes);
    dispatch_parallel_with_worker_limited(
        worker_count,
        [&](size_t worker, size_t worker_begin, size_t worker_end) {
          for (size_t logical_worker = worker_begin;
               logical_worker < worker_end; ++logical_worker) {
            const size_t node = logical_worker % node_count;
            const auto &indices = workspace.indices_by_node[node];
            const size_t workers_per_node = worker_count / node_count;
            const size_t local_worker = logical_worker / node_count;
            const size_t base = indices.size() / workers_per_node;
            const size_t remainder = indices.size() % workers_per_node;
            const size_t begin =
                local_worker * base + std::min(local_worker, remainder);
            const size_t end = begin + base + (local_worker < remainder);
            if (begin < end)
              do_indices(
                  worker,
                  [&indices, begin](size_t routed_position) {
                    return static_cast<size_t>(
                        indices[begin + routed_position]);
                  },
                  end - begin);
          }
        },
        worker_count, 1);
    return {};
  }

  // =========================================================================
  // Backward Gradient Accumulation into TransientArena
  // =========================================================================

  template <typename K>
  std::error_code accumulate_gradients(std::span<const K> keys,
                                       std::span<const ParamT> flat_grads,
                                       std::span<const int64_t> row_lengths) {
    return accumulate_gradients<K, ParamT>(keys, flat_grads, row_lengths);
  }

  template <typename K, typename GradT>
  std::error_code accumulate_gradients(std::span<const K> keys,
                                       std::span<const GradT> flat_grads,
                                       std::span<const int64_t> row_lengths) {
    ActiveInvocationGuard invocation(invocation_gate_, &storage_);
    StoreSnapshot *snapshot = snapshot_.load(std::memory_order_acquire);
    if (snapshot == nullptr)
      throw std::runtime_error("store snapshot is unavailable");
    size_t n = keys.size();
    if (row_lengths.size() != n)
      throw std::invalid_argument(
          "gradient key and row-length counts must match");
    size_t expected_gradient_scalars = 0;
    for (const int64_t row_length : row_lengths) {
      if (row_length <= 0)
        throw std::invalid_argument("gradient row length must be positive");
      const size_t length = static_cast<size_t>(row_length);
      if (expected_gradient_scalars >
          std::numeric_limits<size_t>::max() - length)
        throw std::overflow_error("gradient scalar count overflows size_t");
      expected_gradient_scalars += length;
    }
    if (flat_grads.size() != expected_gradient_scalars)
      throw std::invalid_argument(
          "flat gradient scalar count does not match row lengths");
    if (!snapshot->slots.empty() &&
        !snapshot->slots.front()->transient_shards.empty())
      snapshot->slots.front()->transient_shards.front()->validate_input(
          flat_grads);
    auto offsets = compute_offsets(row_lengths);
    const size_t worker_count =
        n == 0
            ? 1
            : operation_worker_count(HogwildOperation::kAccumulateGradients, n,
                                     row_payload_bytes(row_lengths, offsets));

    size_t num_shards = num_threads_;
    for (SlotStore *slot : snapshot->slots)
      for (auto &transient : slot->transient_shards)
        transient->reserve(n / std::max<size_t>(1, num_shards));

    if (num_shards > std::numeric_limits<size_t>::max() / num_shards ||
        num_shards * num_shards >
            std::numeric_limits<size_t>::max() / sizeof(uint32_t)) {
      throw std::overflow_error("shard workspace size overflow");
    }
    size_t workspace_size = num_shards * num_shards;
    std::vector<uint32_t> gradient_thread_counts(workspace_size, 0);
    std::vector<uint32_t> gradient_thread_offsets(workspace_size, 0);
    std::vector<uint32_t> gradient_shard_offsets(num_shards + 1, 0);

    SplitMixHash<K> hasher;

    thread_local std::unique_ptr<uint64_t[]> gradient_hashes_buf;
    thread_local size_t gradient_hashes_cap = 0;
    if (n > gradient_hashes_cap) {
      const size_t next_cap = std::max(n, gradient_hashes_cap * 2);
      gradient_hashes_buf = std::unique_ptr<uint64_t[]>(new uint64_t[next_cap]);
      gradient_hashes_cap = next_cap;
    }
    std::span<uint64_t> gradient_hashes(gradient_hashes_buf.get(), n);

    auto hash_keys_simd = [&](size_t start, size_t end) {
      size_t i = start;
      if constexpr (sizeof(K) == 8) {
#if defined(__AVX512F__) && defined(__AVX512DQ__)
        for (; i + 8 <= end; i += 8) {
          __m512i v_keys =
              _mm512_loadu_si512(reinterpret_cast<const void *>(&keys[i]));
          __m512i v_hash = hasher.simd_hash(v_keys);
          _mm512_storeu_si512(gradient_hashes.data() + i, v_hash);
        }
#endif
      }
      for (; i < end; ++i) {
        uint64_t key = static_cast<uint64_t>(keys[i]);
        gradient_hashes[i] = hasher(key);
      }
    };

    auto do_hash_and_count = [&](size_t t_begin, size_t t_end) {
      for (size_t t = t_begin; t < t_end; ++t)
        hash_keys_simd(t * n / num_shards, (t + 1) * n / num_shards);
      for (size_t t = t_begin; t < t_end; ++t) {
        size_t start = t * n / num_shards;
        size_t end = (t + 1) * n / num_shards;
        for (size_t idx = start; idx < end; ++idx) {
          const uint32_t shard =
              static_cast<uint32_t>(gradient_hashes[idx] & (num_shards - 1));
          gradient_thread_counts[t * num_shards + shard]++;
        }
      }
    };
    dispatch_parallel_limited(num_shards, do_hash_and_count, worker_count, 1);

    // 2. Prefix Sum (Sequential)
    for (size_t s = 0; s < num_shards; ++s) {
      uint32_t current_offset = gradient_shard_offsets[s];
      for (size_t t = 0; t < num_shards; ++t) {
        gradient_thread_offsets[t * num_shards + s] = current_offset;
        current_offset += gradient_thread_counts[t * num_shards + s];
      }
      gradient_shard_offsets[s + 1] = current_offset;
    }

    // 3. Parallel Scatter
    thread_local std::unique_ptr<uint32_t[]> shuffled_indices_buf;
    thread_local size_t shuffled_indices_cap = 0;

    if (n > shuffled_indices_cap) {
      size_t new_cap = std::max(n, shuffled_indices_cap * 2);
      shuffled_indices_buf = std::unique_ptr<uint32_t[]>(new uint32_t[new_cap]);
      shuffled_indices_cap = new_cap;
    }

    std::span<uint32_t> shuffled_indices(shuffled_indices_buf.get(), n);
    auto do_scatter = [&](size_t t_begin, size_t t_end) {
      for (size_t t = t_begin; t < t_end; ++t) {
        size_t start = t * n / num_shards;
        size_t end = (t + 1) * n / num_shards;
        for (size_t idx = start; idx < end; ++idx) {
          const uint32_t shard =
              static_cast<uint32_t>(gradient_hashes[idx] & (num_shards - 1));
          shuffled_indices[gradient_thread_offsets[t * num_shards + shard]++] =
              static_cast<uint32_t>(idx);
        }
      }
    };
    dispatch_parallel_limited(num_shards, do_scatter, worker_count, 1);

    auto do_shard = [&](size_t shard_begin, size_t shard_end) {
      for (size_t shard_id = shard_begin; shard_id < shard_end; ++shard_id) {
        size_t start = gradient_shard_offsets[shard_id];
        size_t end = gradient_shard_offsets[shard_id + 1];
        for (size_t idx = start; idx < end; ++idx) {
          if (idx + 16 < end) {
            __builtin_prefetch(&shuffled_indices[idx + 16], 0, 1);
            uint32_t prefetch_i = shuffled_indices[idx + 16];
            __builtin_prefetch(&keys[prefetch_i], 0, 1);
            __builtin_prefetch(&offsets[prefetch_i], 0, 1);
            __builtin_prefetch(flat_grads.data() + offsets[prefetch_i], 0, 1);
          }
          uint32_t i = shuffled_indices[idx];
          uint64_t key = static_cast<uint64_t>(keys[i]);
          size_t dim = static_cast<size_t>(row_lengths[i]);
          if (dim >= snapshot->dim_to_slot.size() ||
              snapshot->dim_to_slot[dim] == nullptr) [[unlikely]] {
            throw std::runtime_error(
                "invalid dim access: no physical table for dim " +
                std::to_string(dim));
          }
          auto &slot = *snapshot->dim_to_slot[dim];
          const GradT *grad_src = flat_grads.data() + offsets[i];
          slot.transient_shards[shard_id]->accumulate_hashed(
              key, gradient_hashes[i], grad_src);
        }
      }
    };

    dispatch_parallel_limited(num_shards, do_shard, worker_count, 1);

    // Manual step requirement: optimizer_step() must be called explicitly

    return {};
  }

  // =========================================================================
  // Optimizer Step — shard-batched proxy update under locator write guards
  // =========================================================================

  std::error_code optimizer_step(size_t normalization_steps) {
    if (normalization_steps == 0)
      throw std::invalid_argument("normalization_steps must be positive");
    ActiveInvocationGuard invocation(invocation_gate_, &storage_);
    StoreSnapshot *snapshot = snapshot_.load(std::memory_order_acquire);
    if (snapshot == nullptr)
      throw std::runtime_error("store snapshot is unavailable");
    for (SlotStore *slot : snapshot->slots) {
      if (slot->optimizer.has_value()) {
        slot->global_step.fetch_add(1, std::memory_order_relaxed);
      }
    }

    size_t num_shards = num_threads_;
    const size_t worker_count = operation_worker_count(
        HogwildOperation::kOptimizerStep,
        std::max(size_t{1}, accumulated_gradient_count()),
        std::max(size_t{1}, max_dim_) * sizeof(ParamT));

    auto do_shard = [&](size_t shard_begin, size_t shard_end) {
      se::concurrency::with_epoch([&] {
        for (size_t shard_id = shard_begin; shard_id < shard_end; ++shard_id) {
          for (SlotStore *slot : snapshot->slots) {
            auto &shard = slot->transient_shards[shard_id];
            shard->with_accumulated_batch(
                [&](const auto &batch) {
                  if (batch.size() != 0)
                    slot->update_batch(
                        batch,
                        slot->global_step.load(std::memory_order_relaxed),
                        normalization_steps,
                        [&](uint64_t, hogwild_internal::CheckpointRecordRef
                                          checkpoint_record) {
                          record_checkpoint_mutation(checkpoint_record);
                        });
                },
                normalization_steps);
            shard->clear();
          }
        }
      });
    };

    dispatch_parallel_limited(num_shards, do_shard, worker_count, 1);

    return {};
  }

  // =========================================================================
  // Legacy API (backward compatible)
  // =========================================================================

  template <typename K>
  std::error_code lookup(std::span<const K> keys,
                         std::span<ParamT> out_flat_values,
                         std::span<const int64_t> row_lengths) {
    return forward_lookup(keys, out_flat_values, row_lengths);
  }

  template <typename K>
  std::error_code assign(std::span<const K> keys,
                         std::span<const ParamT> flat_values,
                         std::span<const int64_t> row_lengths) {
    ActiveInvocationGuard invocation(invocation_gate_, &storage_);
    StoreSnapshot *snapshot = snapshot_.load(std::memory_order_acquire);
    if (snapshot == nullptr)
      throw std::runtime_error("store snapshot is unavailable");
    size_t n = keys.size();
    validate_unique_mutation_batch(keys, row_lengths);
    auto offsets = compute_offsets(row_lengths);
    const size_t worker_count =
        n == 0
            ? 1
            : operation_worker_count(HogwildOperation::kAssign, n,
                                     row_payload_bytes(row_lengths, offsets));

    auto do_range = [&](size_t begin, size_t end) {
      se::concurrency::with_epoch_reclamation([&] {
        struct MissingBatch {
          SlotStore *slot;
          std::vector<typename SlotStore::BatchEntry> entries;
        };
        std::vector<MissingBatch> missing_batches;
        for (size_t i = begin; i < end; ++i) {
          uint64_t key = static_cast<uint64_t>(keys[i]);
          const size_t hash = SplitMixHash<uint64_t>{}(key);
          size_t dim = static_cast<size_t>(row_lengths[i]);
          if (dim >= snapshot->dim_to_slot.size() ||
              snapshot->dim_to_slot[dim] == nullptr) [[unlikely]] {
            throw std::runtime_error("invalid dim access");
          }
          auto &slot = *snapshot->dim_to_slot[dim];
          const ParamT *src = flat_values.data() + offsets[i];
          if (slot.assign_existing_hashed(
                  key, hash,
                  [&](uint32_t index,
                      hogwild_internal::CheckpointRecordRef checkpoint_record) {
                    if constexpr (simd::is_simd_supported_v<ParamT>)
                      simd::simd_copy(src, slot.arena.data(index), slot.dim);
                    else
                      std::copy_n(src, slot.dim, slot.arena.data(index));
                    record_checkpoint_mutation(checkpoint_record);
                  }))
            continue;
          auto batch =
              std::ranges::find(missing_batches, &slot, &MissingBatch::slot);
          if (batch == missing_batches.end()) {
            missing_batches.push_back({&slot, {}});
            batch = std::prev(missing_batches.end());
          }
          batch->entries.push_back({key, hash, i});
        }
        for (MissingBatch &batch : missing_batches) {
          SlotStore &slot = *batch.slot;
          slot.find_or_insert_batch(
              batch.entries,
              [&](const typename SlotStore::BatchEntry &entry,
                  uint32_t new_index) {
                const ParamT *src =
                    flat_values.data() + offsets[entry.position];
                if constexpr (simd::is_simd_supported_v<ParamT>)
                  simd::simd_copy(src, slot.arena.data(new_index), slot.dim);
                else
                  std::copy_n(src, slot.dim, slot.arena.data(new_index));
              },
              [&](const typename SlotStore::BatchEntry &entry, uint32_t index,
                  hogwild_internal::CheckpointRecordRef checkpoint_record) {
                const ParamT *src =
                    flat_values.data() + offsets[entry.position];
                if constexpr (simd::is_simd_supported_v<ParamT>)
                  simd::simd_copy(src, slot.arena.data(index), slot.dim);
                else
                  std::copy_n(src, slot.dim, slot.arena.data(index));
                record_checkpoint_mutation(checkpoint_record);
              },
              [&](const typename SlotStore::BatchEntry &entry,
                  hogwild_internal::CheckpointRecordRef checkpoint_record) {
                record_checkpoint_mutation(checkpoint_record);
              },
              true);
        }
      });
    };
    dispatch_parallel_limited(n, do_range, worker_count);
    return {};
  }

  template <typename K>
  std::error_code lookup_or_insert(std::span<const K> keys,
                                   std::span<ParamT> out_flat_values,
                                   std::span<const int64_t> row_lengths) {
    ActiveInvocationGuard invocation(invocation_gate_, &storage_);
    StoreSnapshot *snapshot = snapshot_.load(std::memory_order_acquire);
    if (snapshot == nullptr)
      throw std::runtime_error("store snapshot is unavailable");
    size_t n = keys.size();
    auto offsets = compute_offsets(row_lengths);
    struct DimensionBatch {
      SlotStore *slot;
      std::vector<typename SlotStore::BatchEntry> entries;
    };
    std::vector<DimensionBatch> batches;
    for (size_t i = 0; i < n; ++i) {
      const size_t dim = static_cast<size_t>(row_lengths[i]);
      if (dim >= snapshot->dim_to_slot.size() ||
          snapshot->dim_to_slot[dim] == nullptr)
        throw std::runtime_error("invalid dim access");
      SlotStore *slot = snapshot->dim_to_slot[dim];
      auto batch = std::ranges::find(batches, slot, &DimensionBatch::slot);
      if (batch == batches.end()) {
        batches.push_back({slot, {}});
        batch = std::prev(batches.end());
      }
      const uint64_t key = static_cast<uint64_t>(keys[i]);
      batch->entries.push_back({key, SplitMixHash<uint64_t>{}(key), i});
    }
    for (DimensionBatch &batch : batches) {
      SlotStore &slot = *batch.slot;
      slot.find_or_insert_batch(
          batch.entries,
          [&](const typename SlotStore::BatchEntry &entry, uint32_t index) {
            const size_t dim = static_cast<size_t>(row_lengths[entry.position]);
            if (initializer_.has_value()) {
              initializer_->initialize(
                  std::span<ParamT>(slot.arena.data(index), dim), entry.key);
            } else {
              std::fill_n(slot.arena.data(index), dim, ParamT{0});
            }
          },
          [&](const typename SlotStore::BatchEntry &entry, uint32_t index,
              hogwild_internal::CheckpointRecordRef) {
            const size_t dim = static_cast<size_t>(row_lengths[entry.position]);
            std::copy_n(slot.arena.data(index), dim,
                        out_flat_values.data() + offsets[entry.position]);
          },
          [&](const typename SlotStore::BatchEntry &entry,
              hogwild_internal::CheckpointRecordRef checkpoint_record) {
            record_checkpoint_mutation(checkpoint_record);
          },
          false, true);
    }
    return {};
  }

  template <typename K>
  std::error_code contains(std::span<const K> keys,
                           std::span<uint8_t> out_exists) {
    ActiveInvocationGuard invocation(invocation_gate_, &storage_);
    StoreSnapshot *snapshot = snapshot_.load(std::memory_order_acquire);
    if (snapshot == nullptr)
      throw std::runtime_error("store snapshot is unavailable");
    size_t n = keys.size();
    const size_t worker_count =
        operation_worker_count(HogwildOperation::kContains, n, sizeof(K));
    auto do_range = [&](size_t begin, size_t end) {
      se::concurrency::with_epoch([&] {
        for (size_t i = begin; i < end; ++i) {
          uint64_t key = static_cast<uint64_t>(keys[i]);
          bool found = false;
          for (SlotStore *slot : snapshot->slots) {
            if (slot->find(key).has_value()) {
              found = true;
              break;
            }
          }
          out_exists[i] = found;
        }
      });
    };
    dispatch_parallel_limited(n, do_range, worker_count);
    return {};
  }

  template <typename K> std::error_code erase(std::span<const K> keys) {
    ActiveInvocationGuard invocation(invocation_gate_, &storage_);
    StoreSnapshot *snapshot = snapshot_.load(std::memory_order_acquire);
    if (snapshot == nullptr)
      throw std::runtime_error("store snapshot is unavailable");
    auto pressure_call = retirement_pressure_->writer_call();
    pressure_call.apply();
    [[maybe_unused]] auto retirement_admission =
        retirement_pressure_->control_plane_call();
    size_t n = keys.size();
    const size_t worker_count =
        operation_worker_count(HogwildOperation::kErase, n, sizeof(K));
    auto do_range = [&](size_t begin, size_t end) {
      se::concurrency::with_epoch_reclamation([&] {
        for (SlotStore *slot : snapshot->slots) {
          std::vector<std::vector<uint64_t>> keys_by_shard(
              slot->key_shards.size());
          for (size_t i = begin; i < end; ++i) {
            const uint64_t key = static_cast<uint64_t>(keys[i]);
            keys_by_shard[slot->shard_id_for_key(key)].push_back(key);
          }
          for (size_t shard_id = 0; shard_id < keys_by_shard.size();
               ++shard_id) {
            auto &shard_keys = keys_by_shard[shard_id];
            if (shard_keys.empty())
              continue;
            auto &shard = *slot->key_shards[shard_id];
            std::unique_lock map_lock(shard.writer_mutex);
            std::vector<uint64_t> existing_keys;
            existing_keys.reserve(shard_keys.size());
            for (uint64_t key : shard_keys) {
              if (shard.find_unlocked(key).has_value())
                existing_keys.push_back(key);
            }
            shard.erase_indices_unlocked(
                existing_keys,
                [&](uint64_t, uint32_t index,
                    hogwild_internal::CheckpointRecordRef checkpoint_record,
                    uint64_t deletion_sequence) noexcept {
                  publish_checkpoint_deletion(
                      checkpoint_record, deletion_sequence,
                      index >> slot->arena.entries_shift());
                  slot->arena.mark_live(index, false);
                });
          }
        }
      });
    };
    dispatch_parallel_limited(n, do_range, worker_count);
    return {};
  }

  template <typename K>
  std::error_code erase_dimension(size_t dim, std::span<const K> keys) {
    ActiveInvocationGuard invocation(invocation_gate_, &storage_);
    StoreSnapshot *snapshot = snapshot_.load(std::memory_order_acquire);
    if (snapshot == nullptr)
      throw std::runtime_error("store snapshot is unavailable");
    if (dim >= snapshot->dim_to_slot.size() ||
        snapshot->dim_to_slot[dim] == nullptr)
      throw std::invalid_argument("erase dimension is not configured");
    SlotStore &slot = *snapshot->dim_to_slot[dim];
    std::vector<uint64_t> typed_keys;
    typed_keys.reserve(keys.size());
    for (const K key : keys)
      typed_keys.push_back(static_cast<uint64_t>(key));
    std::vector<std::vector<uint64_t>> keys_by_shard(slot.key_shards.size());
    for (const uint64_t key : typed_keys)
      keys_by_shard[slot.shard_id_for_key(key)].push_back(key);
    se::concurrency::with_epoch_reclamation([&] {
      for (size_t shard_id = 0; shard_id < keys_by_shard.size(); ++shard_id) {
        auto &shard_keys = keys_by_shard[shard_id];
        if (shard_keys.empty())
          continue;
        auto &shard = *slot.key_shards[shard_id];
        std::unique_lock map_lock(shard.writer_mutex);
        std::vector<uint64_t> existing_keys;
        existing_keys.reserve(shard_keys.size());
        for (const uint64_t key : shard_keys) {
          if (shard.find_unlocked(key).has_value())
            existing_keys.push_back(key);
        }
        shard.erase_indices_unlocked(
            existing_keys,
            [&](uint64_t, uint32_t index,
                hogwild_internal::CheckpointRecordRef checkpoint_record,
                uint64_t deletion_sequence) noexcept {
              publish_checkpoint_deletion(checkpoint_record, deletion_sequence,
                                          index >> slot.arena.entries_shift());
              slot.arena.mark_live(index, false);
            });
      }
    });
    return {};
  }

  std::error_code clear() {
    std::lock_guard control_plane_lock(control_plane_mutex_);
    [[maybe_unused]] auto retirement_admission =
        retirement_pressure_->control_plane_call();
    invocation_gate_.begin_reconfiguration_and_wait();
    struct InvocationResume {
      InvocationGate &gate;

      ~InvocationResume() { gate.resume_accepting(); }
    } invocation_resume{invocation_gate_};
    size_t clear_count = 0;
    for (const auto &slot_ptr : slot_storage_)
      clear_count += slot_ptr->size();
    const uint64_t clear_sequence_begin =
        clear_count == 0 ? 0 : reserve_checkpoint_sequences(clear_count);
    size_t clear_sequence_offset = 0;
    for (const auto &slot_ptr : slot_storage_) {
      auto *slot = slot_ptr.get();
      const size_t removed = slot->clear_keys(
          [&](uint64_t, uint32_t index,
              hogwild_internal::CheckpointRecordRef checkpoint_record,
              uint64_t deletion_sequence) noexcept {
            publish_checkpoint_deletion(checkpoint_record, deletion_sequence,
                                        index >> slot->arena.entries_shift());
          },
          clear_sequence_begin + clear_sequence_offset);
      clear_sequence_offset += removed;
      slot->page_reuse->reset();
      slot->page_telemetry->reset();
      slot->access_telemetry.reset();
      if (slot->optimizer.has_value()) {
        slot->optimizer->clear();
      }
      slot->arena.reset();
      slot->optimizer = slot->opt_factory.has_value()
                            ? (*slot->opt_factory)(slot->dim)
                            : nullptr;
      if (slot->optimizer.has_value() && slot->optimizer->requires_score() &&
          !score_policy_.enabled()) {
        throw std::invalid_argument(
            "dynamic learning-rate policy requires score storage");
      }
      if (slot->optimizer.has_value())
        slot->optimizer->set_page_eviction_enabled(score_policy_.enabled());
    }
    publish_store_snapshot();
    return {};
  }

  std::error_code reserve(size_t keys_count, size_t dim) {
    std::lock_guard control_plane_lock(control_plane_mutex_);
    auto pressure_call = retirement_pressure_->writer_call();
    pressure_call.apply();
    [[maybe_unused]] auto retirement_admission =
        retirement_pressure_->control_plane_call();
    invocation_gate_.begin_reconfiguration_and_wait();
    struct InvocationResume {
      InvocationGate &gate;

      ~InvocationResume() { gate.resume_accepting(); }
    } invocation_resume{invocation_gate_};
    StoreSnapshot *snapshot = snapshot_.load(std::memory_order_acquire);
    if (snapshot == nullptr)
      throw std::runtime_error("store snapshot is unavailable");
    snapshot_slot_for_dim(*snapshot, dim).reserve(keys_count);
    publish_store_snapshot();
    return {};
  }

  std::error_code reserve_per_shard(std::span<const size_t> counts,
                                    size_t dim) {
    std::lock_guard control_plane_lock(control_plane_mutex_);
    auto pressure_call = retirement_pressure_->writer_call();
    pressure_call.apply();
    [[maybe_unused]] auto retirement_admission =
        retirement_pressure_->control_plane_call();
    invocation_gate_.begin_reconfiguration_and_wait();
    struct InvocationResume {
      InvocationGate &gate;

      ~InvocationResume() { gate.resume_accepting(); }
    } invocation_resume{invocation_gate_};
    StoreSnapshot *snapshot = snapshot_.load(std::memory_order_acquire);
    if (snapshot == nullptr)
      throw std::runtime_error("store snapshot is unavailable");
    snapshot_slot_for_dim(*snapshot, dim).reserve_per_shard(counts);
    publish_store_snapshot();
    return {};
  }

  size_t internal_shard_count() const {
    return storage_topology_.shard_count();
  }

  template <typename K>
  std::error_code assign_scores(std::span<const K> keys,
                                std::span<const uint64_t> scores,
                                std::span<const int64_t> row_lengths = {}) {
    ActiveInvocationGuard invocation(invocation_gate_, &storage_);
    se::concurrency::EpochAnnouncement epoch_announcement(true);
    StoreSnapshot *snapshot = snapshot_.load(std::memory_order_acquire);
    if (snapshot == nullptr)
      throw std::runtime_error("store snapshot is unavailable");
    if (!score_policy_.enabled()) {
      throw std::runtime_error("assign_scores requires a score-enabled policy");
    }
    size_t n = keys.size();
    if (scores.size() != n || (!row_lengths.empty() && row_lengths.size() != n))
      throw std::invalid_argument("assign_scores row counts differ");
    for (int64_t dim : row_lengths) {
      if (dim <= 0)
        throw std::invalid_argument("assign_scores dimension must be positive");
      (void)snapshot_slot_for_dim(*snapshot, static_cast<size_t>(dim));
    }
    auto assign_slot_score = [&](SlotStore &slot, uint64_t key,
                                 uint64_t score) {
      return slot.with_shared_write_entry(
          key, [&](uint32_t idx,
                   hogwild_internal::CheckpointRecordRef checkpoint_record) {
            std::atomic_ref<uint64_t>(slot.arena.score_raw(idx))
                .store(score, std::memory_order_relaxed);
            record_checkpoint_mutation(checkpoint_record);
          });
    };
    auto do_range = [&](size_t begin, size_t end) {
      se::concurrency::with_epoch([&] {
        if (!row_lengths.empty()) {
          for (size_t i = begin; i < end; ++i) {
            auto &slot = snapshot_slot_for_dim(*snapshot, row_lengths[i]);
            if (!assign_slot_score(slot, static_cast<uint64_t>(keys[i]),
                                   scores[i]))
              throw std::invalid_argument(
                  "assign_scores key is absent from dimension table");
          }
          return;
        }
        for (size_t i = begin; i < end; ++i) {
          uint64_t key = static_cast<uint64_t>(keys[i]);
          for (SlotStore *slot : snapshot->slots) {
            if (assign_slot_score(*slot, key, scores[i])) {
              break; // Key found and updated, proceed to next key
            }
          }
        }
      });
    };
    dispatch_parallel(n, do_range);
    return {};
  }

  size_t size() const {
    ActiveInvocationGuard invocation(invocation_gate_, &storage_);
    StoreSnapshot *snapshot = snapshot_.load(std::memory_order_acquire);
    if (snapshot == nullptr)
      throw std::runtime_error("store snapshot is unavailable");
    size_t s = 0;
    for (SlotStore *slot : snapshot->slots)
      s += slot->size();
    return s;
  }

  size_t num_slots() const {
    ActiveInvocationGuard invocation(invocation_gate_, &storage_);
    StoreSnapshot *snapshot = snapshot_.load(std::memory_order_acquire);
    if (snapshot == nullptr)
      throw std::runtime_error("store snapshot is unavailable");
    return snapshot->slots.size();
  }

private:
  template <typename K>
  static void validate_unique_mutation_batch(std::span<const K> keys,
                                             std::span<const int64_t> dims) {
#ifndef NDEBUG
    if (keys.size() != dims.size())
      throw std::invalid_argument(
          "mutation batch key and dimension counts differ");
    struct BatchKey {
      uint64_t key;
      int64_t dim;
      bool operator==(const BatchKey &) const = default;
    };
    struct BatchKeyHash {
      size_t operator()(const BatchKey &value) const noexcept {
        return SplitMixHash<uint64_t>{}(value.key) ^
               (SplitMixHash<uint64_t>{}(static_cast<uint64_t>(value.dim))
                << 1U);
      }
    };
    std::unordered_set<BatchKey, BatchKeyHash> seen;
    seen.reserve(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
      if (!seen.emplace(BatchKey{static_cast<uint64_t>(keys[i]), dims[i]})
               .second)
        throw std::invalid_argument(
            "mutation batch contains duplicate dimension and key");
    }
#else
    (void)keys;
    (void)dims;
#endif
  }

  SlotStore &slot_for_dim(size_t dim) {
    if (dim >= dim_to_slot_.size() || dim_to_slot_[dim] == nullptr) {
      throw std::runtime_error(
          "invalid dim access: no physical table for dim " +
          std::to_string(dim));
    }
    return *dim_to_slot_[dim];
  }

  static SlotStore &snapshot_slot_for_dim(const StoreSnapshot &snapshot,
                                          size_t dim) {
    if (dim >= snapshot.dim_to_slot.size() ||
        snapshot.dim_to_slot[dim] == nullptr) {
      throw std::runtime_error(
          "invalid dim access: no physical table for dim " +
          std::to_string(dim));
    }
    return *snapshot.dim_to_slot[dim];
  }

  static size_t row_payload_bytes(std::span<const int64_t> row_lengths,
                                  std::span<const size_t> offsets) {
    if (row_lengths.empty())
      throw std::invalid_argument("row payload bytes require one row");
    const int64_t last_length = row_lengths.back();
    if (last_length <= 0)
      throw std::invalid_argument("row length must be positive");
    const size_t last = static_cast<size_t>(last_length);
    if (offsets.back() > std::numeric_limits<size_t>::max() - last)
      throw std::overflow_error("row payload scalar count overflows size_t");
    const size_t total_scalars = offsets.back() + last;
    const size_t average_scalars = total_scalars / row_lengths.size() +
                                   (total_scalars % row_lengths.size() != 0);
    if (average_scalars > std::numeric_limits<size_t>::max() / sizeof(ParamT))
      throw std::overflow_error("row payload byte count overflows size_t");
    return average_scalars * sizeof(ParamT);
  }

  static size_t checked_row_payload_bytes(size_t scalars) {
    if (scalars == 0)
      throw std::invalid_argument("row length must be positive");
    if (scalars > std::numeric_limits<size_t>::max() / sizeof(ParamT))
      throw std::overflow_error("row payload byte count overflows size_t");
    return scalars * sizeof(ParamT);
  }

  static std::optional<size_t>
  fixed_row_length(std::span<const int64_t> row_lengths) {
    if (row_lengths.empty())
      return std::nullopt;
    const int64_t first = row_lengths.front();
    if (first <= 0)
      throw std::invalid_argument("row length must be positive");
    for (const int64_t length : row_lengths) {
      if (length <= 0)
        throw std::invalid_argument("row length must be positive");
      if (length != first)
        return std::nullopt;
    }
    return static_cast<size_t>(first);
  }

  static size_t cap_forward_tasks_by_row_granularity(size_t requested_tasks,
                                                     size_t batch_size,
                                                     size_t bytes_per_row) {
    if (requested_tasks == 0)
      throw std::invalid_argument("forward task count is zero");
    if (batch_size == 0 || bytes_per_row == 0)
      return 1;
    constexpr size_t kMinimumBytesPerForwardTask = 64 * 1024;
    const size_t minimum_rows =
        std::max(size_t{1}, (kMinimumBytesPerForwardTask + bytes_per_row - 1) /
                                bytes_per_row);
    const size_t row_limited_tasks =
        std::max(size_t{1}, batch_size / minimum_rows);
    return std::min(requested_tasks, row_limited_tasks);
  }

  size_t accumulated_gradient_count() const {
    size_t count = 0;
    for (const auto &slot_ptr : slot_storage_) {
      for (const auto &shard : slot_ptr->transient_shards) {
        if (count > std::numeric_limits<size_t>::max() - shard->active_slots())
          throw std::overflow_error("accumulated gradient count overflows");
        count += shard->active_slots();
      }
    }
    return count;
  }

  size_t operation_worker_count(HogwildOperation operation, size_t batch_size,
                                size_t bytes_per_entry) const {
    if (batch_size == 0)
      return 1;
    return OperationDispatchPolicy::thread_count(
        operation, batch_size, bytes_per_entry, operation_dispatch_topology_);
  }

  void validate_dispatch_worker_count(size_t worker_count) const {
    if (worker_count == 0 || worker_count > num_threads_) {
      throw std::invalid_argument("dispatch worker count is outside the pool");
    }
  }

  void bind_worker_if_requested(size_t worker) const {
    if (!bind_worker_affinity_)
      return;
    thread_local int bound_cpu = -1;
    const int target_cpu = numa_memory_locality_
                               ? NumaTopology::cpu_for_worker(worker)
                               : ThreadAffinity::cpu_for_worker(worker);
    if (bound_cpu == target_cpu)
      return;
    ThreadAffinity::bind_current_thread_to_cpu(target_cpu);
    bound_cpu = target_cpu;
  }

  template <typename Fn>
  void dispatch_parallel_with_worker(size_t n, Fn &&fn,
                                     size_t threshold = kParallelThreshold) {
    dispatch_parallel_with_worker_limited(n, std::forward<Fn>(fn), num_threads_,
                                          threshold);
  }

  template <typename Fn>
  void
  dispatch_parallel_with_worker_limited(size_t n, Fn &&fn, size_t worker_count,
                                        size_t threshold = kParallelThreshold) {
    if (n == 0)
      return;
    validate_dispatch_worker_count(worker_count);
    if (n >= threshold && worker_count > 1) {
      const size_t chunk_size = (n + worker_count - 1) / worker_count;
      stdexec::sync_wait(
          stdexec::schedule(pool_.get_scheduler()) |
          stdexec::bulk(stdexec::par, worker_count,
                        [this, &fn, n, chunk_size](size_t worker) {
                          bind_worker_if_requested(worker);
                          const size_t begin = std::min(worker * chunk_size, n);
                          const size_t end = std::min(begin + chunk_size, n);
                          if (begin < end)
                            fn(worker, begin, end);
                        }));
      return;
    }
    fn(0, 0, n);
  }

  template <typename Fn>
  void dispatch_parallel(size_t n, Fn &&fn,
                         size_t threshold = kParallelThreshold) {
    dispatch_parallel_limited(n, std::forward<Fn>(fn), num_threads_, threshold);
  }

  template <typename Fn>
  void dispatch_parallel_limited(size_t n, Fn &&fn, size_t worker_count,
                                 size_t threshold = kParallelThreshold) {
    if (n == 0)
      return;
    validate_dispatch_worker_count(worker_count);
    if (n >= threshold && worker_count > 1) {
      size_t chunk_size = (n + worker_count - 1) / worker_count;
      stdexec::sync_wait(stdexec::schedule(pool_.get_scheduler()) |
                         stdexec::bulk(stdexec::par, worker_count,
                                       [this, &fn, n, chunk_size](size_t i) {
                                         bind_worker_if_requested(i);
                                         size_t begin =
                                             std::min(i * chunk_size, n);
                                         size_t end =
                                             std::min(begin + chunk_size, n);
                                         if (begin < end) {
                                           fn(begin, end);
                                         }
                                       }));
      return;
    }
    fn(0, n);
  }

  static void compute_offsets_into(std::span<const int64_t> row_lengths,
                                   std::vector<size_t> &offsets) {
    offsets.resize(row_lengths.size());
    auto casted = row_lengths | std::views::transform([](int64_t v) {
                    return static_cast<size_t>(v);
                  });
    std::exclusive_scan(casted.begin(), casted.end(), offsets.begin(),
                        size_t{0});
  }

  std::vector<size_t>
  compute_offsets(std::span<const int64_t> row_lengths) const {
    std::vector<size_t> offsets;
    compute_offsets_into(row_lengths, offsets);
    return offsets;
  }

  enum class ExportOwnership { kSpan, kMove };

  template <typename K, ExportOwnership Policy, typename VisitorFn>
  void export_with_opt_impl(VisitorFn &&visitor, size_t batch_size) {
    ActiveInvocationGuard invocation(invocation_gate_, &storage_);
    se::concurrency::EpochAnnouncement epoch_announcement(true);
    StoreSnapshot *snapshot = snapshot_.load(std::memory_order_acquire);
    if (snapshot == nullptr)
      throw std::runtime_error("store snapshot is unavailable");
    for (SlotStore *slot : snapshot->slots) {
      size_t emb_dim = slot->dim;
      size_t opt_dim =
          slot->optimizer.has_value() ? slot->optimizer->state_dim() : 0;
      int num_workers = static_cast<int>(parlay::num_workers());

      const bool has_score = score_policy_.enabled();

      struct Buffer {
        std::vector<K> keys;
        std::vector<ParamT> vals;
        std::vector<uint64_t> scores;
        std::vector<OptT> opts;
        std::vector<uint32_t> steps;
      };

      std::vector<Buffer> local_bufs(num_workers);
      for (int i = 0; i < num_workers; ++i) {
        local_bufs[i].keys.reserve(256);
        local_bufs[i].vals.reserve(256 * emb_dim);
        if (has_score)
          local_bufs[i].scores.reserve(256);
        if (opt_dim > 0) {
          local_bufs[i].opts.reserve(256 * opt_dim);
          local_bufs[i].steps.reserve(256);
        }
      }

      Buffer global_buf;
      global_buf.keys.reserve(batch_size);
      global_buf.vals.reserve(batch_size * emb_dim);
      if (has_score)
        global_buf.scores.reserve(batch_size);
      if (opt_dim > 0) {
        global_buf.opts.reserve(batch_size * opt_dim);
        global_buf.steps.reserve(batch_size);
      }

      std::atomic<bool> stopped{false};
      std::mutex merge_m;
      std::mutex visitor_m;

      auto flush_global = [&](Buffer &buf) -> bool {
        if (buf.keys.empty())
          return true;
        if constexpr (Policy == ExportOwnership::kSpan) {
          return visitor(emb_dim, std::span<const K>(buf.keys),
                         std::span<const ParamT>(buf.vals),
                         std::span<const uint64_t>(buf.scores),
                         std::span<const OptT>(buf.opts),
                         std::span<const uint32_t>(buf.steps));
        } else {
          bool cont = visitor(emb_dim, std::move(buf.keys), std::move(buf.vals),
                              std::move(buf.scores), std::move(buf.opts),
                              std::move(buf.steps));
          buf.keys.clear();
          buf.vals.clear();
          buf.scores.clear();
          buf.opts.clear();
          buf.steps.clear();
          buf.keys.reserve(batch_size);
          buf.vals.reserve(batch_size * emb_dim);
          if (has_score)
            buf.scores.reserve(batch_size);
          if (opt_dim > 0) {
            buf.opts.reserve(batch_size * opt_dim);
            buf.steps.reserve(batch_size);
          }
          return cont;
        }
      };

      slot->for_each([&](uint64_t key, uint32_t idx) {
        if (stopped.load(std::memory_order_relaxed))
          return;

        int w = parlay::worker_id();
        auto &lb = local_bufs[w];

        lb.keys.push_back(static_cast<K>(key));
        const ParamT *src = slot->arena.data(idx);
        lb.vals.insert(lb.vals.end(), src, src + emb_dim);
        if (has_score) {
          lb.scores.push_back(
              std::atomic_ref<uint64_t>(slot->arena.score_raw(idx))
                  .load(std::memory_order_relaxed));
        }
        if (opt_dim > 0 && slot->optimizer.has_value()) {
          // Optimizer state_data fills a caller-provided span (so the
          // optimizer is free to materialise its state from non-contiguous
          // internal buffers). Reserve room at the back of `lb.opts` and let
          // the optimizer write directly there to avoid a temporary copy.
          size_t old_size = lb.opts.size();
          lb.opts.resize(old_size + opt_dim);
          slot->optimizer->state_data(
              idx, std::span<OptT>(lb.opts.data() + old_size, opt_dim));
          lb.steps.push_back(slot->optimizer->step_data(idx));
        }

        if (lb.keys.size() >= 256) {
          bool need_flush = false;
          Buffer local_flush_buf;
          {
            std::lock_guard lock(merge_m);
            if (stopped.load(std::memory_order_relaxed))
              return;
            global_buf.keys.insert(global_buf.keys.end(), lb.keys.begin(),
                                   lb.keys.end());
            global_buf.vals.insert(global_buf.vals.end(), lb.vals.begin(),
                                   lb.vals.end());
            if (has_score) {
              global_buf.scores.insert(global_buf.scores.end(),
                                       lb.scores.begin(), lb.scores.end());
            }
            if (opt_dim > 0) {
              global_buf.opts.insert(global_buf.opts.end(), lb.opts.begin(),
                                     lb.opts.end());
              global_buf.steps.insert(global_buf.steps.end(), lb.steps.begin(),
                                      lb.steps.end());
            }
            lb.keys.clear();
            lb.vals.clear();
            lb.scores.clear();
            lb.opts.clear();
            lb.steps.clear();

            if (global_buf.keys.size() >= batch_size) {
              local_flush_buf = std::move(global_buf);
              global_buf.keys.reserve(batch_size);
              global_buf.vals.reserve(batch_size * emb_dim);
              if (has_score)
                global_buf.scores.reserve(batch_size);
              if (opt_dim > 0) {
                global_buf.opts.reserve(batch_size * opt_dim);
                global_buf.steps.reserve(batch_size);
              }
              need_flush = true;
            }
          }
          if (need_flush) {
            std::lock_guard vlock(visitor_m);
            if (stopped.load(std::memory_order_relaxed))
              return;
            if (!flush_global(local_flush_buf))
              stopped.store(true, std::memory_order_relaxed);
          }
        }
      });

      if (stopped.load(std::memory_order_relaxed))
        break;

      for (int w = 0; w < num_workers; ++w) {
        auto &lb = local_bufs[w];
        if (!lb.keys.empty()) {
          global_buf.keys.insert(global_buf.keys.end(), lb.keys.begin(),
                                 lb.keys.end());
          global_buf.vals.insert(global_buf.vals.end(), lb.vals.begin(),
                                 lb.vals.end());
          if (has_score) {
            global_buf.scores.insert(global_buf.scores.end(), lb.scores.begin(),
                                     lb.scores.end());
          }
          if (opt_dim > 0) {
            global_buf.opts.insert(global_buf.opts.end(), lb.opts.begin(),
                                   lb.opts.end());
            global_buf.steps.insert(global_buf.steps.end(), lb.steps.begin(),
                                    lb.steps.end());
          }
        }
      }

      if (!stopped.load(std::memory_order_relaxed) &&
          !global_buf.keys.empty()) {
        flush_global(global_buf);
      }
    }
  }

  storage::StorageTopology storage_topology_;
  std::vector<SlotStore *> dim_to_slot_;
  std::vector<std::unique_ptr<SlotStore>> slot_storage_;
  std::atomic<StoreSnapshot *> snapshot_{nullptr};
  mutable InvocationGate invocation_gate_;
  size_t num_threads_ = 0;
  DispatchTopology operation_dispatch_topology_;
  exec::static_thread_pool pool_;
  Initializer initializer_;
  ScorePolicyRuntime score_policy_;
  PageStorage storage_;
  std::shared_ptr<RetirementPressure> retirement_pressure_;
  bool background_page_eviction_enabled_;
  mutable std::mutex control_plane_mutex_;
  PageEvictionRegistration page_eviction_registration_;
  std::atomic<uint64_t> telemetry_sample_sequence_{0};
  std::atomic<uint64_t> checkpoint_mutation_sequence_{0};
  std::atomic<uint64_t> checkpoint_page_layout_generation_{1};
  mutable std::mutex checkpoint_mutations_mutex_;
  std::vector<std::unique_ptr<CheckpointMutationRecord>>
      checkpoint_record_storage_;
  std::vector<size_t> checkpoint_free_record_slots_;
  std::unique_ptr<ShardDirtyHead[]> checkpoint_dirty_heads_;
  mutable std::mutex checkpoint_record_structure_mutex_;
  std::string checkpoint_baseline_;
  uint64_t checkpoint_capture_sequence_ = 0;
  uint64_t checkpoint_acknowledged_capture_ = 0;
  uint64_t checkpoint_acknowledged_watermark_ = 0;
  struct PendingCheckpointCut {
    uint64_t watermark;
    std::string parent_baseline;
    bool incremental;
  };
  std::map<uint64_t, PendingCheckpointCut> checkpoint_cuts_;
  std::string checkpoint_journal_identity_ = [] {
    std::random_device entropy;
    std::string identity;
    for (size_t index = 0; index < 4; ++index) {
      if (index != 0)
        identity += ":";
      identity += std::to_string(entropy());
    }
    return identity;
  }();
  std::atomic<bool> checkpoint_journal_valid_{true};
  bool bind_worker_affinity_ = false;
  bool numa_memory_locality_ = false;
  TelemetryRing telemetry_ring_;
  execution::ReusableWorkspacePool<NumaForwardWorkspace>
      numa_forward_workspaces_;
  size_t max_dim_ = 0;
  static constexpr size_t kParallelThreshold = 64;

  struct CheckpointMutationCapture {
    std::vector<CheckpointMutation> mutations;
    ankerl::unordered_dense::map<uint32_t, std::vector<uint64_t>>
        deleted_page_ids;
  };

  CheckpointMutationCapture checkpoint_mutations_quiesced() const {
    ankerl::unordered_dense::map<std::pair<uint32_t, uint64_t>, size_t>
        winner_indices;
    CheckpointMutationCapture capture;
    for (size_t shard = 0; shard < storage_topology_.shard_count(); ++shard) {
      for (CheckpointMutationRecord *record =
               checkpoint_dirty_heads_[shard].head.load(
                   std::memory_order_acquire);
           record != nullptr; record = record->next) {
        const uint64_t state = record->state.load(std::memory_order_acquire);
        const uint64_t sequence = state & ~CheckpointMutationRecord::kDeleted;
        if (sequence != 0) {
          const bool deleted =
              (state & CheckpointMutationRecord::kDeleted) != 0;
          if (deleted) {
            if (record->deleted_page_id ==
                CheckpointMutationRecord::kNoPhysicalPage)
              throw std::logic_error(
                  "deleted checkpoint mutation has no physical page");
            capture.deleted_page_ids[record->dimension].push_back(
                record->deleted_page_id);
          }
          const CheckpointMutation mutation{record->dimension, record->key,
                                            sequence, deleted};
          const auto identity = std::pair{record->dimension, record->key};
          auto [winner, inserted] =
              winner_indices.try_emplace(identity, capture.mutations.size());
          if (inserted) {
            capture.mutations.push_back(mutation);
            continue;
          }
          CheckpointMutation &current = capture.mutations[winner->second];
          if (sequence == current.mutation_sequence)
            throw std::logic_error(
                "checkpoint mutation sequence is duplicated for an identity");
          if (sequence < current.mutation_sequence)
            continue;
          current = mutation;
        }
      }
    }
    std::ranges::sort(capture.mutations, {},
                      &CheckpointMutation::mutation_sequence);
    for (auto &[dimension, page_ids] : capture.deleted_page_ids) {
      std::ranges::sort(page_ids);
      page_ids.erase(std::ranges::unique(page_ids).begin(), page_ids.end());
    }
    return capture;
  }

public:
  bool has_score_policy() const { return score_policy_.enabled(); }

  void
  validate_optimizer_state_layout(std::span<const uint32_t> dimensions,
                                  std::span<const uint32_t> strides) const {
    if (dimensions.size() != strides.size())
      throw std::invalid_argument(
          "checkpoint optimizer layout row counts differ");
    ActiveInvocationGuard invocation(invocation_gate_, &storage_);
    StoreSnapshot *snapshot = snapshot_.load(std::memory_order_acquire);
    if (snapshot == nullptr)
      throw std::runtime_error("store snapshot is unavailable");
    for (size_t row = 0; row < dimensions.size(); ++row) {
      const auto &slot = snapshot_slot_for_dim(*snapshot, dimensions[row]);
      const size_t expected =
          slot.optimizer.has_value() ? slot.optimizer->state_dim() : 0;
      if (strides[row] != expected)
        throw std::invalid_argument(
            "checkpoint optimizer state layout does not match store");
    }
  }

  CheckpointCaptureCut capture_checkpoint(
      bool incremental, std::string expected_baseline, bool include_page_images,
      uint64_t expected_page_layout_generation,
      se::Function<void(std::span<const CheckpointMutation>, uint64_t)> prepare,
      se::Function<bool(size_t, PageImageExport<ParamT, OptT> &&)> visitor) {
    std::lock_guard control_plane_lock(control_plane_mutex_);
    [[maybe_unused]] auto retirement_admission =
        retirement_pressure_->control_plane_call();
    invocation_gate_.begin_reconfiguration_and_wait();
    struct InvocationResume {
      InvocationGate &gate;
      ~InvocationResume() { gate.resume_accepting(); }
    } invocation_resume{invocation_gate_};
    if (incremental && (expected_baseline.empty() ||
                        checkpoint_baseline() != expected_baseline))
      throw std::runtime_error(
          "incremental checkpoint requires a matching live store baseline");
    if (!checkpoint_journal_valid_.load(std::memory_order_acquire))
      throw std::runtime_error(
          "checkpoint mutation journal is invalid after a failed mutation");
    const uint64_t page_layout_generation =
        checkpoint_page_layout_generation_.load(std::memory_order_acquire);
    if (incremental && include_page_images &&
        expected_page_layout_generation != page_layout_generation)
      throw std::runtime_error("incremental page checkpoint requires the "
                               "current page layout generation");
    const auto mutation_capture = incremental ? checkpoint_mutations_quiesced()
                                              : CheckpointMutationCapture{};
    const auto &mutations = mutation_capture.mutations;
    const uint64_t watermark = checkpoint_mutation_watermark();
    (*prepare)(mutations, watermark);
    StoreSnapshot *capture_snapshot = snapshot_.load(std::memory_order_acquire);
    if (capture_snapshot == nullptr)
      throw std::runtime_error("store snapshot is unavailable");
    std::unordered_map<uint32_t, std::vector<uint64_t>> dirty_keys;
    ankerl::unordered_dense::map<uint32_t, std::vector<uint64_t>>
        dirty_page_ids;
    se::concurrency::with_epoch([&] {
      if (incremental) {
        if (include_page_images) {
          dirty_page_ids = mutation_capture.deleted_page_ids;
          for (const auto &mutation : mutations) {
            if (!mutation.deleted) {
              const auto &slot =
                  snapshot_slot_for_dim(*capture_snapshot, mutation.dimension);
              const auto page_id = slot.page_id_for_key(mutation.key);
              if (!page_id.has_value())
                throw std::runtime_error(
                    "checkpoint dirty key has no stable physical page");
              dirty_page_ids[mutation.dimension].push_back(*page_id);
            }
          }
          for (auto &[dimension, page_ids] : dirty_page_ids) {
            std::ranges::sort(page_ids);
            page_ids.erase(std::ranges::unique(page_ids).begin(),
                           page_ids.end());
          }
        } else {
          for (const auto &mutation : mutations) {
            if (!mutation.deleted) {
              dirty_keys[mutation.dimension].push_back(mutation.key);
            }
          }
        }
      }
      for (const auto &slot : capture_snapshot->slots) {
        const auto keys_it = dirty_keys.find(static_cast<uint32_t>(slot->dim));
        const auto pages_it =
            dirty_page_ids.find(static_cast<uint32_t>(slot->dim));
        const std::span<const uint64_t> keys =
            keys_it == dirty_keys.end()
                ? std::span<const uint64_t>{}
                : std::span<const uint64_t>(keys_it->second);
        const std::span<const uint64_t> pages =
            pages_it == dirty_page_ids.end()
                ? std::span<const uint64_t>{}
                : std::span<const uint64_t>(pages_it->second);
        if (incremental && ((include_page_images && pages.empty()) ||
                            (!include_page_images && keys.empty())))
          continue;
        if (!slot->export_page_images(
                [&](PageImageExport<ParamT, OptT> &&image) {
                  return (*visitor)(slot->dim, std::move(image));
                },
                keys, incremental && !include_page_images, pages))
          throw std::runtime_error(
              "checkpoint capture visitor stopped before completion");
      }
    });
    std::lock_guard lock(checkpoint_mutations_mutex_);
    if (checkpoint_capture_sequence_ == std::numeric_limits<uint64_t>::max())
      throw std::overflow_error("checkpoint capture identity exhausted");
    const uint64_t capture_id = ++checkpoint_capture_sequence_;
    checkpoint_cuts_.emplace(
        capture_id, PendingCheckpointCut{
                        watermark, std::move(expected_baseline), incremental});
    return {watermark, page_layout_generation, capture_id,
            checkpoint_journal_identity_};
  }

  uint64_t checkpoint_page_layout_generation() const noexcept {
    return checkpoint_page_layout_generation_.load(std::memory_order_acquire);
  }

  std::vector<CheckpointMutation> checkpoint_mutations() const {
    std::lock_guard control_plane_lock(control_plane_mutex_);
    invocation_gate_.begin_reconfiguration_and_wait();
    struct InvocationResume {
      InvocationGate &gate;
      ~InvocationResume() { gate.resume_accepting(); }
    } invocation_resume{invocation_gate_};
    return checkpoint_mutations_quiesced().mutations;
  }

  std::string checkpoint_baseline() const {
    std::lock_guard lock(checkpoint_mutations_mutex_);
    return checkpoint_baseline_;
  }

  bool checkpoint_restore_is_pristine() const {
    std::lock_guard lock(checkpoint_mutations_mutex_);
    return checkpoint_journal_valid_.load(std::memory_order_acquire) &&
           checkpoint_baseline_.empty() && checkpoint_cuts_.empty() &&
           checkpoint_acknowledged_capture_ == 0 &&
           checkpoint_acknowledged_watermark_ == 0;
  }

  // Initializes the store checkpoint journal baseline following physical
  // restore or cold start. Quiesces active invocations via InvocationGate,
  // validates that the mutation journal is clean and unassigned, resets all
  // dirty shard heads and mutation record pools, sets the authoritative
  // baseline token and initial mutation watermark, and mints a
  // cryptographically unique journal incarnation token.
  std::string initialize_checkpoint_baseline(std::string baseline,
                                             uint64_t watermark) {
    if (baseline.empty())
      throw std::invalid_argument("checkpoint baseline must not be empty");
    std::lock_guard control_plane_lock(control_plane_mutex_);
    invocation_gate_.begin_reconfiguration_and_wait();
    struct InvocationResume {
      InvocationGate &gate;
      ~InvocationResume() { gate.resume_accepting(); }
    } invocation_resume{invocation_gate_};
    std::lock_guard lock(checkpoint_mutations_mutex_);
    if (!checkpoint_baseline_.empty() || !checkpoint_cuts_.empty() ||
        checkpoint_acknowledged_capture_ != 0 ||
        checkpoint_acknowledged_watermark_ != 0)
      throw std::logic_error("checkpoint baseline initialization requires a "
                             "fresh checkpoint journal");
    for (size_t shard = 0; shard < storage_topology_.shard_count(); ++shard)
      checkpoint_dirty_heads_[shard].head.store(nullptr,
                                                std::memory_order_release);
    for (const auto &record : checkpoint_record_storage_) {
      if (!record)
        continue;
      record->state.store(0, std::memory_order_release);
      record->queued.store(false, std::memory_order_release);
      record->deleted_page_id = CheckpointMutationRecord::kNoPhysicalPage;
      record->next = nullptr;
    }
    checkpoint_baseline_ = std::move(baseline);
    checkpoint_mutation_sequence_.store(watermark, std::memory_order_release);
    checkpoint_acknowledged_watermark_ = watermark;
    checkpoint_acknowledged_capture_ = 0;
    checkpoint_capture_sequence_ = 0;
    checkpoint_cuts_.clear();
    checkpoint_journal_identity_ = [] {
      std::random_device entropy;
      std::string identity;
      for (size_t index = 0; index < 4; ++index) {
        if (index != 0)
          identity += ":";
        identity += std::to_string(entropy());
      }
      return identity;
    }();
    checkpoint_journal_valid_.store(true, std::memory_order_release);
    return checkpoint_journal_identity_;
  }

  uint64_t checkpoint_mutation_watermark() const {
    std::lock_guard lock(checkpoint_mutations_mutex_);
    return checkpoint_mutation_sequence_.load(std::memory_order_relaxed);
  }

  void cancel_checkpoint_capture(uint64_t capture_id,
                                 std::string journal_identity) {
    std::lock_guard lock(checkpoint_mutations_mutex_);
    if (journal_identity != checkpoint_journal_identity_)
      throw std::invalid_argument(
          "checkpoint capture belongs to a different journal incarnation");
    checkpoint_cuts_.erase(capture_id);
  }

  // Advances the store checkpoint baseline token and commits a previously
  // captured checkpoint cut. Quiesces inflight operations, verifies the journal
  // incarnation identity and matching cut, updates the live baseline token,
  // advances the acknowledged watermark, and prunes mutation journal records
  // with sequence <= watermark across all shards.
  void advance_checkpoint_baseline(uint64_t watermark, std::string baseline,
                                   uint64_t capture_id,
                                   std::string journal_identity) {
    if (baseline.empty())
      throw std::invalid_argument("checkpoint baseline must not be empty");
    std::lock_guard control_plane_lock(control_plane_mutex_);
    invocation_gate_.begin_reconfiguration_and_wait();
    struct InvocationResume {
      InvocationGate &gate;
      ~InvocationResume() { gate.resume_accepting(); }
    } invocation_resume{invocation_gate_};
    std::lock_guard lock(checkpoint_mutations_mutex_);
    if (journal_identity != checkpoint_journal_identity_)
      throw std::invalid_argument("checkpoint acknowledgement belongs to a "
                                  "different journal incarnation");
    if (capture_id == checkpoint_acknowledged_capture_ &&
        baseline == checkpoint_baseline_ &&
        watermark == checkpoint_acknowledged_watermark_)
      return;
    const auto cut = checkpoint_cuts_.find(capture_id);
    if (capture_id <= checkpoint_acknowledged_capture_ ||
        cut == checkpoint_cuts_.end() || cut->second.watermark != watermark)
      throw std::invalid_argument("checkpoint acknowledgement is stale or does "
                                  "not match a captured cut");
    if (cut->second.incremental &&
        cut->second.parent_baseline != checkpoint_baseline_)
      throw std::invalid_argument(
          "checkpoint acknowledgement parent baseline has advanced");
    if (watermark >
        checkpoint_mutation_sequence_.load(std::memory_order_relaxed))
      throw std::invalid_argument(
          "checkpoint mutation watermark is in the future");
    checkpoint_free_record_slots_.reserve(checkpoint_record_storage_.size());
    for (size_t shard = 0; shard < storage_topology_.shard_count(); ++shard) {
      auto &head = checkpoint_dirty_heads_[shard].head;
      CheckpointMutationRecord *current = head.load(std::memory_order_acquire);
      CheckpointMutationRecord *retained = nullptr;
      while (current != nullptr) {
        CheckpointMutationRecord *next = current->next;
        const uint64_t state = current->state.load(std::memory_order_acquire);
        const uint64_t sequence = state & ~CheckpointMutationRecord::kDeleted;
        if (sequence > watermark) {
          current->next = retained;
          retained = current;
        } else {
          current->queued.store(false, std::memory_order_release);
          if ((state & CheckpointMutationRecord::kDeleted) != 0) {
            const size_t owner_slot = current->owner_slot;
            checkpoint_record_storage_[owner_slot].reset();
            checkpoint_free_record_slots_.push_back(owner_slot);
          }
        }
        current = next;
      }
      head.store(retained, std::memory_order_release);
    }
    checkpoint_baseline_ = std::move(baseline);
    checkpoint_acknowledged_capture_ = capture_id;
    checkpoint_acknowledged_watermark_ = watermark;
    std::erase_if(checkpoint_cuts_, [capture_id](const auto &cut) {
      return cut.first <= capture_id;
    });
  }

  std::vector<size_t> table_dims() const {
    ActiveInvocationGuard invocation(invocation_gate_, &storage_);
    StoreSnapshot *snapshot = snapshot_.load(std::memory_order_acquire);
    if (snapshot == nullptr)
      throw std::runtime_error("store snapshot is unavailable");
    std::vector<size_t> result;
    result.reserve(snapshot->slots.size());
    for (SlotStore *slot : snapshot->slots) {
      result.push_back(slot->dim);
    }
    return result;
  }

  // =========================================================================
  // export_batch — visitor-callback snapshot export.
  //
  // Iterates all entries in the hash map via for_each (which is safe to call
  // concurrently with reads; writes may or may not be observed). For each
  // batch of up to batch_size entries, calls visitor(keys, values, scores).
  // If visitor returns false, iteration stops early.
  // =========================================================================

  template <typename K>
  std::error_code export_batch(
      se::Function<bool(size_t, std::span<const K>, std::span<const ParamT>,
                        std::span<const uint64_t>)>
          visitor,
      size_t batch_size) {
    ActiveInvocationGuard invocation(invocation_gate_, &storage_);
    se::concurrency::EpochAnnouncement epoch_announcement(true);
    StoreSnapshot *snapshot = snapshot_.load(std::memory_order_acquire);
    if (snapshot == nullptr)
      throw std::runtime_error("store snapshot is unavailable");
    bool has_score = score_policy_.enabled();

    for (SlotStore *slot : snapshot->slots) {
      size_t dim = slot->dim;
      int num_workers = static_cast<int>(parlay::num_workers());

      struct LocalTinyBuffer {
        std::vector<K> keys;
        std::vector<ParamT> vals;
        std::vector<uint64_t> scores;
      };

      struct GlobalBuffer {
        std::vector<K> keys;
        std::vector<ParamT> vals;
        std::vector<uint64_t> scores;
      };

      std::vector<LocalTinyBuffer> local_buffers(num_workers);
      for (int i = 0; i < num_workers; ++i) {
        local_buffers[i].keys.reserve(256);
        local_buffers[i].vals.reserve(256 * dim);
        if (has_score)
          local_buffers[i].scores.reserve(256);
      }

      GlobalBuffer global_buf;
      global_buf.keys.reserve(batch_size);
      global_buf.vals.reserve(batch_size * dim);
      if (has_score)
        global_buf.scores.reserve(batch_size);

      std::atomic<bool> stopped{false};
      std::mutex m;
      std::mutex visitor_m;

      auto flush_global = [&](GlobalBuffer &buf) -> bool {
        if (buf.keys.empty())
          return true;
        bool cont = (*visitor)(dim, std::span<const K>(buf.keys),
                               std::span<const ParamT>(buf.vals),
                               std::span<const uint64_t>(buf.scores));
        return cont;
      };

      slot->for_each([&](uint64_t key, uint32_t idx) {
        if (stopped.load(std::memory_order_relaxed))
          return;

        int w = parlay::worker_id();
        auto &lb = local_buffers[w];

        lb.keys.push_back(static_cast<K>(key));
        const ParamT *src = slot->arena.data(idx);
        lb.vals.insert(lb.vals.end(), src, src + dim);
        if (has_score) {
          lb.scores.push_back(
              std::atomic_ref<uint64_t>(slot->arena.score_raw(idx))
                  .load(std::memory_order_relaxed));
        }

        if (lb.keys.size() >= 256) {
          bool need_flush = false;
          GlobalBuffer local_flush_buf;

          {
            std::lock_guard<std::mutex> lock(m);
            if (stopped.load(std::memory_order_relaxed))
              return;

            global_buf.keys.insert(global_buf.keys.end(), lb.keys.begin(),
                                   lb.keys.end());
            global_buf.vals.insert(global_buf.vals.end(), lb.vals.begin(),
                                   lb.vals.end());
            if (has_score) {
              global_buf.scores.insert(global_buf.scores.end(),
                                       lb.scores.begin(), lb.scores.end());
            }

            lb.keys.clear();
            lb.vals.clear();
            lb.scores.clear();

            if (global_buf.keys.size() >= batch_size) {
              local_flush_buf = std::move(global_buf);
              global_buf.keys.reserve(batch_size);
              global_buf.vals.reserve(batch_size * dim);
              if (has_score)
                global_buf.scores.reserve(batch_size);
              need_flush = true;
            }
          }

          if (need_flush) {
            std::lock_guard<std::mutex> vlock(visitor_m);
            if (stopped.load(std::memory_order_relaxed))
              return;
            if (!flush_global(local_flush_buf)) {
              stopped.store(true, std::memory_order_relaxed);
            }
          }
        }
      });

      if (stopped.load(std::memory_order_relaxed))
        break;

      for (int w = 0; w < num_workers; ++w) {
        auto &lb = local_buffers[w];
        if (!lb.keys.empty()) {
          std::lock_guard<std::mutex> lock(m);
          if (stopped.load(std::memory_order_relaxed))
            break;
          global_buf.keys.insert(global_buf.keys.end(), lb.keys.begin(),
                                 lb.keys.end());
          global_buf.vals.insert(global_buf.vals.end(), lb.vals.begin(),
                                 lb.vals.end());
          if (has_score) {
            global_buf.scores.insert(global_buf.scores.end(), lb.scores.begin(),
                                     lb.scores.end());
          }
        }
      }

      if (!stopped.load(std::memory_order_relaxed)) {
        bool need_flush = false;
        GlobalBuffer local_flush_buf;
        {
          std::lock_guard<std::mutex> lock(m);
          if (!global_buf.keys.empty()) {
            local_flush_buf = std::move(global_buf);
            need_flush = true;
          }
        }
        if (need_flush) {
          std::lock_guard<std::mutex> vlock(visitor_m);
          flush_global(local_flush_buf);
        }
      }
    }
    return {};
  }

  template <typename K, typename VisitorFn>
  void export_batch_with_opt(VisitorFn &&visitor, size_t batch_size) {
    export_with_opt_impl<K, ExportOwnership::kSpan>(
        std::forward<VisitorFn>(visitor), batch_size);
  }

  template <typename K, typename VisitorFn>
  void export_batch_with_opt_owned(VisitorFn &&visitor, size_t batch_size) {
    export_with_opt_impl<K, ExportOwnership::kMove>(
        std::forward<VisitorFn>(visitor), batch_size);
  }

  template <typename K>
  void assign_opt_states(size_t dim, std::span<const K> keys,
                         std::span<const OptT> opt_states,
                         std::span<const uint32_t> steps) {
    assign_opt_states_span(dim, keys, opt_states, steps);
  }

  template <std::ranges::contiguous_range Keys,
            std::ranges::contiguous_range States,
            std::ranges::contiguous_range Steps>
  void assign_opt_states(size_t dim, const Keys &keys, const States &states,
                         const Steps &steps) {
    using KeyT = std::ranges::range_value_t<Keys>;
    using StateT = std::ranges::range_value_t<States>;
    using StepT = std::ranges::range_value_t<Steps>;
    static_assert(std::is_same_v<StateT, OptT>);
    static_assert(std::is_same_v<StepT, uint32_t>);
    assign_opt_states_span(
        dim,
        std::span<const KeyT>(std::ranges::data(keys), std::ranges::size(keys)),
        std::span<const OptT>(std::ranges::data(states),
                              std::ranges::size(states)),
        std::span<const uint32_t>(std::ranges::data(steps),
                                  std::ranges::size(steps)));
  }

private:
  template <typename K>
  void assign_opt_states_span(size_t dim, std::span<const K> keys,
                              std::span<const OptT> opt_states,
                              std::span<const uint32_t> steps) {
    ActiveInvocationGuard invocation(invocation_gate_, &storage_);
    StoreSnapshot *snapshot = snapshot_.load(std::memory_order_acquire);
    if (snapshot == nullptr)
      throw std::runtime_error("store snapshot is unavailable");
    size_t n = keys.size();
    if (dim >= snapshot->dim_to_slot.size() ||
        snapshot->dim_to_slot[dim] == nullptr) {
      throw std::invalid_argument(
          "assign_opt_states dimension has no physical table");
    }
    auto &slot = *snapshot->dim_to_slot[dim];
    if (!slot.optimizer.has_value())
      throw std::invalid_argument(
          "assign_opt_states dimension has no optimizer");
    size_t opt_dim = slot.optimizer->state_dim();
    if (opt_dim == 0 || opt_states.size() != n * opt_dim)
      throw std::invalid_argument(
          "assign_opt_states payload size does not match dimension");
    if (!steps.empty() && steps.size() != n)
      throw std::invalid_argument(
          "assign_opt_states step count does not match key count");
    if (n == 0)
      return;

    se::concurrency::with_epoch([&] {
      for (size_t i = 0; i < n; ++i) {
        uint64_t key = keys[i];
        const bool found = slot.with_shared_write_entry(
            key, [&](uint32_t index,
                     hogwild_internal::CheckpointRecordRef checkpoint_record) {
              slot.optimizer->set_state(
                  index, std::span<const OptT>(opt_states.data() + i * opt_dim,
                                               opt_dim));
              if (!steps.empty())
                slot.optimizer->set_step(index, steps[i]);
              record_checkpoint_mutation(checkpoint_record);
            });
        if (!found)
          throw std::invalid_argument(
              "assign_opt_states key is absent from dimension table");
      }
    });
  }

public:
  // export_opt_state_batch — type-erased facade-compatible wrapper.
  // Visitor receives (dim, keys, vals, scores, opt_states, steps) with native
  // ParamT/OptT precision.
  template <typename K>
  void export_opt_state_batch(
      se::Function<bool(size_t, std::span<const K>, std::span<const ParamT>,
                        std::span<const uint64_t>, std::span<const OptT>,
                        std::span<const uint32_t>)>
          visitor,
      size_t batch_size) {
    export_batch_with_opt<K>(
        [&](size_t dim, std::span<const K> keys, std::span<const ParamT> vals,
            std::span<const uint64_t> scores, std::span<const OptT> opt_states,
            std::span<const uint32_t> steps) -> bool {
          return (*visitor)(dim, keys, vals, scores, opt_states, steps);
        },
        batch_size);
  }

  // export_opt_state_batch_owned — ownership-transferring facade wrapper.
  // Visitor receives std::vector&& of native ParamT/OptT precision for
  // zero-copy Arrow C Data Interface construction.
  template <typename K>
  void export_opt_state_batch_owned(
      se::Function<bool(size_t, std::vector<K> &&, std::vector<ParamT> &&,
                        std::vector<uint64_t> &&, std::vector<OptT> &&,
                        std::vector<uint32_t> &&)>
          visitor,
      size_t batch_size) {
    export_batch_with_opt_owned<K>(
        [&](size_t dim, std::vector<K> &&keys, std::vector<ParamT> &&vals,
            std::vector<uint64_t> &&scores, std::vector<OptT> &&opt_states,
            std::vector<uint32_t> &&steps) -> bool {
          return (*visitor)(dim, std::move(keys), std::move(vals),
                            std::move(scores), std::move(opt_states),
                            std::move(steps));
        },
        batch_size);
  }

  template <typename K> bool compact_one(size_t dim, K key) {
    ActiveInvocationGuard invocation(invocation_gate_, &storage_);
    StoreSnapshot *snapshot = snapshot_.load(std::memory_order_acquire);
    if (snapshot == nullptr)
      throw std::runtime_error("store snapshot is unavailable");
    ensure_page_layout_generation_available();
    SlotStore &slot = snapshot_slot_for_dim(*snapshot, dim);
    bool compacted = false;
    se::concurrency::with_epoch_reclamation(
        [&] { compacted = slot.compact_key(static_cast<uint64_t>(key)); });
    if (compacted)
      record_page_layout_change();
    return compacted;
  }

  size_t compact_page(size_t dim, uint64_t page_id) {
    ActiveInvocationGuard invocation(invocation_gate_, &storage_);
    StoreSnapshot *snapshot = snapshot_.load(std::memory_order_acquire);
    if (snapshot == nullptr)
      throw std::runtime_error("store snapshot is unavailable");
    ensure_page_layout_generation_available();
    SlotStore &slot = snapshot_slot_for_dim(*snapshot, dim);
    const size_t moved = se::concurrency::with_epoch_reclamation(
        [&] { return slot.compact_page(page_id); });
    if (moved != 0)
      record_page_layout_change();
    return moved;
  }

  size_t compact_pages(size_t dim, std::span<const uint64_t> page_ids) {
    ActiveInvocationGuard invocation(invocation_gate_, &storage_);
    StoreSnapshot *snapshot = snapshot_.load(std::memory_order_acquire);
    if (snapshot == nullptr)
      throw std::runtime_error("store snapshot is unavailable");
    ensure_page_layout_generation_available();
    SlotStore &slot = snapshot_slot_for_dim(*snapshot, dim);
    const size_t moved = se::concurrency::with_epoch_reclamation(
        [&] { return slot.compact_pages(page_ids, false); });
    if (moved != 0)
      record_page_layout_change();
    return moved;
  }

  size_t promote_hot_keys(size_t dim, uint64_t page_id, size_t maximum_keys,
                          uint64_t minimum_score) {
    ActiveInvocationGuard invocation(invocation_gate_, &storage_);
    StoreSnapshot *snapshot = snapshot_.load(std::memory_order_acquire);
    if (snapshot == nullptr)
      throw std::runtime_error("store snapshot is unavailable");
    ensure_page_layout_generation_available();
    SlotStore &slot = snapshot_slot_for_dim(*snapshot, dim);
    const size_t promoted = se::concurrency::with_epoch_reclamation([&] {
      return slot.promote_hot_keys(page_id, maximum_keys, minimum_score);
    });
    if (promoted != 0)
      record_page_layout_change();
    return promoted;
  }

  size_t promoted_hot_page_size(size_t dim) const {
    ActiveInvocationGuard invocation(invocation_gate_, &storage_);
    StoreSnapshot *snapshot = snapshot_.load(std::memory_order_acquire);
    if (snapshot == nullptr)
      throw std::runtime_error("store snapshot is unavailable");
    return snapshot_slot_for_dim(*snapshot, dim).promoted_hot_page_size();
  }

  size_t compact_adaptive_pages(size_t dim, float memory_pressure,
                                const AdaptivePageSelectorConfig &config) {
    ActiveInvocationGuard invocation(invocation_gate_, &storage_);
    StoreSnapshot *snapshot = snapshot_.load(std::memory_order_acquire);
    if (snapshot == nullptr)
      throw std::runtime_error("store snapshot is unavailable");
    ensure_page_layout_generation_available();
    SlotStore &slot = snapshot_slot_for_dim(*snapshot, dim);
    const size_t moved = se::concurrency::with_epoch_reclamation(
        [&] { return slot.compact_adaptive_pages(memory_pressure, config); });
    if (moved != 0)
      record_page_layout_change();
    return moved;
  }

  template <typename Visitor> void export_page_images(Visitor &&visitor) {
    ActiveInvocationGuard invocation(invocation_gate_, &storage_);
    StoreSnapshot *snapshot = snapshot_.load(std::memory_order_acquire);
    if (snapshot == nullptr)
      throw std::runtime_error("store snapshot is unavailable");
    se::concurrency::with_epoch([&] {
      bool keep_exporting = true;
      for (SlotStore *slot : snapshot->slots) {
        if (!keep_exporting)
          break;
        keep_exporting = slot->export_page_images(
            [&](PageImageExport<ParamT, OptT> &&image) {
              return visitor(slot->dim, std::move(image));
            });
      }
    });
  }

  void export_page_images(
      se::Function<bool(size_t, PageImageExport<ParamT, OptT> &&)> visitor) {
    export_page_images(
        [&visitor](size_t dim, PageImageExport<ParamT, OptT> &&image) {
          return (*visitor)(dim, std::move(image));
        });
  }

  void
  import_page_images(std::span<const PageImageImport<ParamT, OptT>> images) {
    import_page_images(page_image_source(images), std::nullopt, true);
  }

  void import_page_images_with_generation(
      std::span<const PageImageImport<ParamT, OptT>> images,
      uint64_t page_layout_generation) {
    import_page_images(page_image_source(images), page_layout_generation, true);
  }

  void import_page_images(PageImageSource<ParamT, OptT> source,
                          std::optional<uint64_t> requested_generation,
                          bool load_optimizer_state) {
    if (!source.has_value())
      throw std::invalid_argument("page-image source is empty");
    std::lock_guard control_plane_lock(control_plane_mutex_);
    ensure_operational();
    if (requested_generation) {
      if (*requested_generation == 0)
        throw std::invalid_argument("page layout generation must be positive");
      if (*requested_generation <
          checkpoint_page_layout_generation_.load(std::memory_order_acquire))
        throw std::invalid_argument(
            "page layout generation regresses during restore");
    }
    [[maybe_unused]] auto retirement_admission =
        retirement_pressure_->control_plane_call();
    invocation_gate_.begin_reconfiguration_and_wait();
    struct InvocationResume {
      InvocationGate &gate;
      ~InvocationResume() { gate.resume_accepting(); }
    } invocation_resume{invocation_gate_};
    for (const auto &slot : slot_storage_) {
      if (slot->arena.allocated_count() != 0 || slot->size() != 0)
        throw std::logic_error("page-image import requires empty slots");
    }
    bool import_attempted = false;
    std::exception_ptr import_failure;
    try {
      checkpoint::ExternalSortedRecords<checkpoint::ActiveKeyIdentity> pages;
      typename SlotStore::PageImportScratch scratch;
      (*source)(pro::make_proxy<FunctionFacade<void(
                    const PageImageImport<ParamT, OptT>
                        &)>>([&](const PageImageImport<ParamT, OptT> &import) {
        if (import_failure)
          std::rethrow_exception(import_failure);
        try {
          if (import.embedding_dim >= dim_to_slot_.size() ||
              dim_to_slot_[import.embedding_dim] == nullptr)
            throw std::invalid_argument(
                "page-image import dimension has no physical table");
          pages.append({import.embedding_dim, import.image.physical_page_id});
          import_attempted = true;
          se::concurrency::with_epoch_reclamation([&] {
            dim_to_slot_[import.embedding_dim]->import_page_image(
                import, load_optimizer_state, scratch);
          });
        } catch (...) {
          import_failure = std::current_exception();
          if (import_attempted)
            mark_terminal_failure();
          throw;
        }
      }));
      if (import_failure)
        std::rethrow_exception(import_failure);
      checkpoint::validate_unique_records(
          pages, "page-image import has duplicate page ids");
      ensure_operational();
      publish_store_snapshot();
      {
        std::lock_guard lock(checkpoint_mutations_mutex_);
        checkpoint_baseline_.clear();
        checkpoint_cuts_.clear();
      }
      if (requested_generation)
        checkpoint_page_layout_generation_.store(*requested_generation,
                                                 std::memory_order_release);
    } catch (...) {
      if (import_attempted)
        mark_terminal_failure();
      if (import_failure)
        std::rethrow_exception(import_failure);
      throw;
    }
  }

  std::vector<std::shared_ptr<const SealedHotPage<ParamT>>>
  seal_adaptive_pages(size_t dim, float memory_pressure,
                      const AdaptivePageSelectorConfig &config,
                      PageThermalTier tier, uint32_t cluster_id,
                      uint64_t page_layout_generation) {
    ActiveInvocationGuard invocation(invocation_gate_, &storage_);
    StoreSnapshot *snapshot = snapshot_.load(std::memory_order_acquire);
    if (snapshot == nullptr)
      throw std::runtime_error("store snapshot is unavailable");
    SlotStore &slot = snapshot_slot_for_dim(*snapshot, dim);
    return se::concurrency::with_epoch([&] {
      return slot.seal_adaptive_pages(memory_pressure, config, tier, cluster_id,
                                      page_layout_generation);
    });
  }

  // =========================================================================
  // evict_by_score — GC operation, decoupled from insertion.
  //
  // Evicts entries with score <= max_score, up to max_evict_count entries.
  // Safe to call from a background thread. Operates per-slot.
  // =========================================================================

  template <typename K>
  std::error_code evict_by_score(uint64_t max_score, size_t max_evict_count,
                                 std::span<K> out_evicted_keys,
                                 std::span<ParamT> out_evicted_values,
                                 size_t &out_evicted_count) {
    if (!score_policy_.enabled())
      throw std::runtime_error(
          "evict_by_score requires a score-enabled policy");
    if (out_evicted_keys.size() < max_evict_count) {
      throw std::invalid_argument("eviction output spans are too small");
    }
    if (max_evict_count == 0)
      return {};

    ActiveInvocationGuard invocation(invocation_gate_, &storage_);
    StoreSnapshot *snapshot = snapshot_.load(std::memory_order_acquire);
    if (snapshot == nullptr)
      throw std::runtime_error("store snapshot is unavailable");
    auto pressure_call = retirement_pressure_->writer_call();
    pressure_call.apply();
    [[maybe_unused]] auto retirement_admission =
        retirement_pressure_->control_plane_call();
    out_evicted_count = 0;

    struct Candidate {
      uint64_t key;
      uint64_t score;
      SlotStore *slot;
    };

    std::vector<Candidate> candidates;
    candidates.reserve(max_evict_count);
    const int num_workers = static_cast<int>(parlay::num_workers());
    std::vector<std::vector<Candidate>> local_candidates(num_workers);
    for (auto &local : local_candidates)
      local.reserve(std::min(max_evict_count, size_t{256}));
    auto add_candidate = [max_evict_count](std::vector<Candidate> &target,
                                           Candidate candidate) {
      auto colder = [](const Candidate &left, const Candidate &right) {
        return left.score < right.score;
      };
      if (target.size() < max_evict_count) {
        target.push_back(candidate);
        if (target.size() == max_evict_count)
          std::make_heap(target.begin(), target.end(), colder);
      } else if (candidate.score < target.front().score) {
        std::pop_heap(target.begin(), target.end(), colder);
        target.back() = candidate;
        std::push_heap(target.begin(), target.end(), colder);
      }
    };
    for (SlotStore *slot : snapshot->slots) {
      slot->for_each([&](uint64_t key, uint32_t idx) {
        auto &local = local_candidates[parlay::worker_id()];
        if (!slot->arena.is_live(idx))
          return;
        uint64_t score = std::atomic_ref<uint64_t>(slot->arena.score_raw(idx))
                             .load(std::memory_order_relaxed);
        if (score > max_score)
          return;
        add_candidate(local, Candidate{key, score, slot});
      });
    }
    for (auto &local : local_candidates) {
      for (const Candidate &candidate : local)
        add_candidate(candidates, candidate);
    }

    size_t required_dim = 0;
    for (const Candidate &candidate : candidates) {
      if (required_dim == 0) {
        required_dim = candidate.slot->dim;
      } else if (candidate.slot->dim != required_dim) {
        throw std::invalid_argument(
            "eviction output cannot represent mixed dimensions");
      }
    }
    if (required_dim != 0 &&
        max_evict_count > std::numeric_limits<size_t>::max() / required_dim) {
      throw std::overflow_error("eviction output size overflow");
    }
    if (out_evicted_values.size() < max_evict_count * required_dim)
      throw std::invalid_argument("eviction value span is too small");

    constexpr size_t kEvictionEpochBatch = 256;
    for (size_t batch_begin = 0;
         batch_begin < candidates.size() && out_evicted_count < max_evict_count;
         batch_begin += kEvictionEpochBatch) {
      const size_t batch_end =
          std::min(batch_begin + kEvictionEpochBatch, candidates.size());
      se::concurrency::with_epoch_reclamation([&] {
        std::sort(candidates.begin() + batch_begin,
                  candidates.begin() + batch_end,
                  [](const Candidate &left, const Candidate &right) {
                    if (left.slot != right.slot)
                      return std::less<SlotStore *>{}(left.slot, right.slot);
                    return left.slot->shard_id_for_key(left.key) <
                           right.slot->shard_id_for_key(right.key);
                  });
        size_t group_begin = batch_begin;
        while (group_begin < batch_end && out_evicted_count < max_evict_count) {
          const Candidate &first = candidates[group_begin];
          SlotStore &slot = *first.slot;
          const size_t shard_id = slot.shard_id_for_key(first.key);
          size_t group_end = group_begin + 1;
          while (group_end < batch_end &&
                 candidates[group_end].slot == first.slot &&
                 slot.shard_id_for_key(candidates[group_end].key) == shard_id) {
            ++group_end;
          }

          auto &shard = *slot.key_shards[shard_id];
          std::unique_lock map_lock(shard.writer_mutex);
          std::vector<uint64_t> keys;
          keys.reserve(group_end - group_begin);
          for (size_t candidate_index = group_begin;
               candidate_index < group_end &&
               out_evicted_count + keys.size() < max_evict_count;
               ++candidate_index) {
            const Candidate &candidate = candidates[candidate_index];
            auto found = shard.find_unlocked(candidate.key);
            if (!found.has_value())
              continue;
            const uint32_t index = *found;
            const uint64_t score =
                std::atomic_ref<uint64_t>(slot.arena.score_raw(index))
                    .load(std::memory_order_relaxed);
            if (score <= max_score && slot.arena.is_live(index))
              keys.push_back(candidate.key);
          }
          shard.erase_indices_unlocked(
              keys, [&](uint64_t key, uint32_t index,
                        hogwild_internal::CheckpointRecordRef checkpoint_record,
                        uint64_t deletion_sequence) noexcept {
                publish_checkpoint_deletion(
                    checkpoint_record, deletion_sequence,
                    index >> slot.arena.entries_shift());
                std::copy_n(slot.arena.data(index), slot.dim,
                            out_evicted_values.data() +
                                out_evicted_count * slot.dim);
                out_evicted_keys[out_evicted_count] = static_cast<K>(key);
                slot.arena.mark_live(index, false);
                ++out_evicted_count;
              });
          group_begin = group_end;
        }
      });
    }
    return {};
  }
};

#ifdef SPARSE_EMBEDDING_TEST_HOOKS
namespace testing {
template <typename ParamT, typename OptT>
void set_lru_clock_for_testing(HogwildEmbeddingStoreCore<ParamT, OptT> &store,
                               uint64_t value) {
  store.score_policy_.set_lru_clock_for_testing(value);
}
} // namespace testing
#endif

template <typename KeyT, typename ParamT, typename OptT,
          typename ScorePolicyT = NoScorePolicy>
using HogwildEmbeddingStore = HogwildEmbeddingStoreCore<ParamT, OptT>;

#ifndef HOGWILD_EXPLICIT_INSTANTIATION
extern template class HogwildEmbeddingStoreCore<float, float>;
extern template class HogwildEmbeddingStoreCore<float, float16_t>;
extern template class HogwildEmbeddingStoreCore<float, bfloat16_t>;
extern template class HogwildEmbeddingStoreCore<float16_t, float>;
extern template class HogwildEmbeddingStoreCore<float16_t, float16_t>;
extern template class HogwildEmbeddingStoreCore<float16_t, bfloat16_t>;
extern template class HogwildEmbeddingStoreCore<bfloat16_t, float>;
extern template class HogwildEmbeddingStoreCore<bfloat16_t, float16_t>;
extern template class HogwildEmbeddingStoreCore<bfloat16_t, bfloat16_t>;
#endif

} // namespace se

#endif // SPARSE_EMBEDDING_HOGWILD_EMBEDDING_STORE_HPP_
