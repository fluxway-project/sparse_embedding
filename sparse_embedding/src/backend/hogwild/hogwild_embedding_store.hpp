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

#include <parlay_hash/unordered_map.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exec/static_thread_pool.hpp>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <ranges>
#include <shared_mutex>
#include <span>
#include <stdexcept>
#include <stdexec/execution.hpp>
#include <string>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "sparse_embedding/src/arena/transient_arena.hpp"
#include "sparse_embedding/src/common/function.hpp"
#include "sparse_embedding/src/eviction/score_policy.hpp"
#include "sparse_embedding/src/interface/initializer_interface.hpp"
#include "sparse_embedding/src/interface/optimizer_interface.hpp"
#include "sparse_embedding/src/simd/simd_ops.hpp"
#include "sparse_embedding/src/storage/page_storage.hpp"

namespace se {

// =============================================================================
// EmbeddingArena: Lock-free, 64-byte cache-line aligned region allocator.
// Index allocation uses atomic fetch_add. Page growth uses mutex
// (amortized rare). Score stored separately to keep data() SIMD-aligned.
// =============================================================================

template <typename ValT> class EmbeddingArena {
  static constexpr size_t kCacheLineSize = 64;
  static constexpr size_t kEntriesPerPage = 4096;
  static constexpr size_t kMaxPages = 8192;

  struct PageTable {
    std::atomic<std::byte *> data[kMaxPages];
    std::atomic<uint64_t *> score[kMaxPages];
    PageTable() {
      for (size_t i = 0; i < kMaxPages; ++i) {
        data[i].store(nullptr, std::memory_order_relaxed);
        score[i].store(nullptr, std::memory_order_relaxed);
      }
    }
  };

public:
  explicit EmbeddingArena(size_t dim, bool enable_score, PageStorage storage)
      : dim_(dim), enable_score_(enable_score),
        pages_(std::make_unique<PageTable>()), storage_(std::move(storage)),
        data_pages_(std::make_unique<StoragePage[]>(kMaxPages)),
        score_pages_(std::make_unique<StoragePage[]>(kMaxPages)) {
    data_stride_ =
        (dim * sizeof(ValT) + kCacheLineSize - 1) & ~(kCacheLineSize - 1);
  }

  ~EmbeddingArena() = default;

  EmbeddingArena(const EmbeddingArena &) = delete;
  EmbeddingArena &operator=(const EmbeddingArena &) = delete;
  EmbeddingArena(EmbeddingArena &&) = delete;
  EmbeddingArena &operator=(EmbeddingArena &&) = delete;

  size_t dim() const { return dim_; }
  bool score_enabled() const { return enable_score_; }

  template <typename Prepare> uint32_t allocate(Prepare &&prepare) {
    uint32_t idx = next_index_.load(std::memory_order_relaxed);
    while (true) {
      ensure_capacity(idx);
      prepare(idx);
      if (next_index_.compare_exchange_weak(idx, idx + 1,
                                            std::memory_order_relaxed,
                                            std::memory_order_relaxed)) {
        if (enable_score_)
          score_raw(idx) = 0;
        return idx;
      }
    }
  }

  uint32_t allocate() {
    return allocate([](uint32_t) {});
  }

  void reserve_entries(size_t capacity) {
    if (capacity == 0)
      return;
    if (capacity > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
      throw std::overflow_error("embedding arena capacity exceeds uint32_t");
    }
    ensure_capacity(static_cast<uint32_t>(capacity - 1));
  }

  [[carries_dependency]] uint64_t &score_raw(uint32_t index) const {
    return pages_->score[index / kEntriesPerPage].load(
        std::memory_order_consume)[index % kEntriesPerPage];
  }

  ValT *data(uint32_t index) const {
    return reinterpret_cast<ValT *>(data_raw(index));
  }

  size_t allocated_count() const {
    return next_index_.load(std::memory_order_relaxed);
  }

  size_t capacity() const {
    return num_pages_.load(std::memory_order_acquire) * kEntriesPerPage;
  }

  void reset() {
    std::lock_guard lock(grow_mtx_);
    for (size_t i = 0; i < kMaxPages; ++i) {
      pages_->data[i].store(nullptr, std::memory_order_relaxed);
      pages_->score[i].store(nullptr, std::memory_order_relaxed);
      data_pages_[i] = StoragePage{};
      score_pages_[i] = StoragePage{};
    }
    num_pages_.store(0, std::memory_order_relaxed);
    next_index_.store(0, std::memory_order_relaxed);
  }

  size_t data_stride() const { return data_stride_; }

  static constexpr size_t max_entries() { return kEntriesPerPage * kMaxPages; }

  void prepare_reused(uint32_t index) {
    if (enable_score_)
      score_raw(index) = 0;
  }

private:
  void ensure_capacity(uint32_t idx) {
    size_t page_needed = idx / kEntriesPerPage;
    if (page_needed >= kMaxPages) {
      throw std::length_error("embedding arena page capacity exceeded");
    }
    if (page_needed < num_pages_.load(std::memory_order_acquire))
      return;
    std::lock_guard lock(grow_mtx_);
    while (page_needed >= num_pages_.load(std::memory_order_relaxed)) {
      size_t pg = num_pages_.load(std::memory_order_relaxed);
      PageEvaluator evaluator;
      if (enable_score_)
        evaluator = [this, pg] { return page_score(pg); };
      const PageEvictionMode eviction_mode = enable_score_
                                                 ? PageEvictionMode::kEnabled
                                                 : PageEvictionMode::kDisabled;
      auto data_page =
          storage_->allocate(kEntriesPerPage * data_stride_, kCacheLineSize,
                             evaluator, eviction_mode);
      StoragePage score_page;
      if (enable_score_) {
        score_page = storage_->allocate(
            kEntriesPerPage * sizeof(uint64_t), kCacheLineSize,
            [] { return uint64_t{0}; }, PageEvictionMode::kEnabled);
      }
      data_pages_[pg] = std::move(data_page);
      score_pages_[pg] = std::move(score_page);
      pages_->data[pg].store(data_pages_[pg].data(), std::memory_order_release);
      if (enable_score_)
        pages_->score[pg].store(
            reinterpret_cast<uint64_t *>(score_pages_[pg].data()),
            std::memory_order_release);
      num_pages_.store(pg + 1, std::memory_order_release);
    }
  }

  uint64_t page_score(size_t page) const {
    size_t first = page * kEntriesPerPage;
    size_t allocated = allocated_count();
    if (first >= allocated)
      return 0;
    size_t count = std::min(kEntriesPerPage, allocated - first);
    uint64_t total = 0;
    for (auto index : std::views::iota(static_cast<uint32_t>(first),
                                       static_cast<uint32_t>(first + count))) {
      uint64_t score = std::atomic_ref<uint64_t>(score_raw(index))
                           .load(std::memory_order_relaxed);
      if (total > std::numeric_limits<uint64_t>::max() - score)
        return std::numeric_limits<uint64_t>::max();
      total += score;
    }
    return total;
  }

  [[carries_dependency]] std::byte *data_raw(uint32_t index) const {
    return pages_->data[index / kEntriesPerPage].load(
               std::memory_order_consume) +
           (index % kEntriesPerPage) * data_stride_;
  }

  size_t dim_;
  size_t data_stride_;
  bool enable_score_;
  std::unique_ptr<PageTable> pages_;
  PageStorage storage_;
  std::unique_ptr<StoragePage[]> data_pages_;
  std::unique_ptr<StoragePage[]> score_pages_;
  std::atomic<uint32_t> next_index_{0};
  std::atomic<size_t> num_pages_{0};
  std::mutex grow_mtx_;
};

// =============================================================================
// HogwildEmbeddingStore — Lock-free architecture via ParlayHash
//
// ParlayHash (parlay::parlay_unordered_map) provides O(1) lock-free
// concurrent Key→uint32_t index mapping. EmbeddingArena provides
// 64-byte aligned contiguous memory for SIMD-friendly dense vectors.
// =============================================================================

template <typename KeyT, typename ParamT, typename OptT,
          typename ScorePolicyT = NoScorePolicy>
class HogwildEmbeddingStore {
public:
  using ValueType = ParamT;
  using Initializer = pro::proxy<InitializerFacade<ParamT>>;
  using OptimizerFactory =
      se::Function<pro::proxy<OptimizerFacade<ParamT, OptT>>(size_t)>;

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

  using MapData =
      parlay::MapData<KeyT, uint32_t, SplitMixHash<KeyT>, std::equal_to<KeyT>>;

  using Key = parlay::IndirectEntries<MapData>;
  using KeyMap = parlay::parlay_hash<Key>;

  struct KeyShard {
    static constexpr size_t kInitialRecycleCapacity = 64;
    static constexpr size_t kParlayHashCapacityMultiplier = 4;

    struct MapBundle {
      Key entries;
      KeyMap key_map;
      explicit MapBundle(size_t capacity)
          : entries(false), key_map(capacity, &entries, false) {}
    };

    size_t reserved_keys;
    size_t count = 0;
    std::unique_ptr<MapBundle> bundle;
    mutable std::shared_mutex map_mutex;
    std::vector<uint32_t> recycled_indices;
    size_t owned_index_count = 0;

    explicit KeyShard(size_t requested_keys)
        : reserved_keys(requested_keys),
          bundle(std::make_unique<MapBundle>(map_capacity(requested_keys))) {}

    static size_t map_capacity(size_t keys_count) {
      if (keys_count >
          std::numeric_limits<size_t>::max() / kParlayHashCapacityMultiplier) {
        throw std::overflow_error("key shard capacity overflow");
      }
      return std::max(size_t(1024), keys_count * kParlayHashCapacityMultiplier);
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

    std::optional<uint32_t> find_unlocked(uint64_t key) {
      return bundle->key_map.Find(
          Key::Entry::make_key(key),
          [](const auto &entry) { return entry.get_entry().second; });
    }

    std::optional<uint32_t> find(uint64_t key) {
      std::shared_lock lock(map_mutex);
      return find_unlocked(key);
    }

    void expand(size_t target_keys) {
      if (target_keys <= reserved_keys)
        return;
      auto new_bundle = std::make_unique<MapBundle>(map_capacity(target_keys));
      bundle->key_map.for_each([&](const auto &entry) {
        auto [k, idx] = entry.get_entry();
        auto map_key = Key::Entry::make_key(k);
        new_bundle->key_map.Insert(
            map_key,
            [&] {
              return new_bundle->entries.make_entry(map_key,
                                                    std::make_pair(k, idx));
            },
            [](const auto &e) { return e.get_entry().second; });
      });
      bundle = std::move(new_bundle);
      reserved_keys = target_keys;
    }

    void prepare_new_index() {
      if (owned_index_count == std::numeric_limits<size_t>::max()) {
        throw std::overflow_error("key shard index count overflow");
      }
      size_t required_capacity = owned_index_count + 1;
      if (recycled_indices.capacity() < required_capacity) {
        size_t next_capacity = compute_next_capacity(
            recycled_indices.capacity(), required_capacity);
        recycled_indices.reserve(next_capacity);
      }
    }

    void insert_existing(uint64_t key, uint32_t idx) {
      auto map_key = Key::Entry::make_key(key);
      auto read_idx = [](const auto &e) { return e.get_entry().second; };
      auto existing = bundle->key_map.Insert(
          map_key,
          [&] {
            return bundle->entries.make_entry(map_key,
                                              std::make_pair(key, idx));
          },
          read_idx);
      if (existing.has_value()) {
        throw std::logic_error("duplicate key while rebuilding key shard");
      }
    }
  };

  struct SlotStore {
    static constexpr size_t kInitialKeysPerShard = 2048;

    size_t dim;
    std::vector<std::unique_ptr<KeyShard>> key_shards;
    size_t shard_mask;
    EmbeddingArena<ParamT> arena;
    pro::proxy<OptimizerFacade<ParamT, OptT>> optimizer;
    std::vector<std::unique_ptr<TransientArena<ParamT>>> transient_shards;
    OptimizerFactory opt_factory;
    uint64_t global_step = 0;

    SlotStore(size_t d, OptimizerFactory opt_factory_in, size_t num_threads,
              bool enable_score, PageStorage storage)
        : dim(d), shard_mask(num_threads - 1), arena(d, enable_score, storage),
          opt_factory(std::move(opt_factory_in)) {
      if (opt_factory.has_value()) {
        optimizer = (*opt_factory)(d);
        if (optimizer->requires_score() && !enable_score) {
          throw std::invalid_argument(
              "dynamic learning-rate policy requires score storage");
        }
      }
      if (optimizer.has_value())
        optimizer->set_page_eviction_enabled(enable_score);
      for (size_t i = 0; i < num_threads; ++i) {
        key_shards.push_back(std::make_unique<KeyShard>(kInitialKeysPerShard));
      }
      for (size_t i = 0; i < num_threads; ++i) {
        transient_shards.push_back(
            std::make_unique<TransientArena<ParamT>>(d, storage, enable_score));
      }
    }

    KeyShard &shard_for_key(uint64_t key) {
      size_t shard_id =
          SplitMixHash<KeyT>{}(static_cast<KeyT>(key)) & shard_mask;
      return *key_shards[shard_id];
    }

    const KeyShard &shard_for_key(uint64_t key) const {
      size_t shard_id =
          SplitMixHash<KeyT>{}(static_cast<KeyT>(key)) & shard_mask;
      return *key_shards[shard_id];
    }

    std::optional<uint32_t> find(uint64_t key) {
      auto &shard = shard_for_key(key);
      return shard.find(key);
    }

    template <typename Fn> void for_each(Fn &&fn) {
      for (auto &shard : key_shards) {
        std::shared_lock lock(shard->map_mutex);
        shard->bundle->key_map.for_each(fn);
      }
    }

    size_t size() {
      size_t total = 0;
      for (auto &shard : key_shards) {
        std::shared_lock lock(shard->map_mutex);
        total += shard->count;
      }
      return total;
    }

    void clear_keys() {
      for (auto &shard : key_shards) {
        shard = std::make_unique<KeyShard>(shard->reserved_keys);
      }
    }

    uint32_t allocate_entry() {
      uint32_t idx = arena.allocate([&](uint32_t candidate) {
        if (optimizer.has_value() && optimizer->capacity() <= candidate) {
          optimizer->reserve(static_cast<size_t>(candidate) + 1);
        }
      });
      if (optimizer.has_value())
        optimizer->allocate(idx);
      return idx;
    }

    template <typename Initialize>
    std::pair<uint32_t, bool> find_or_insert(uint64_t key,
                                             Initialize &&initialize) {
      auto &key_shard = shard_for_key(key);
      if (auto existing = key_shard.find(key))
        return {*existing, false};

      std::unique_lock map_lock(key_shard.map_mutex);
      if (auto existing = key_shard.find_unlocked(key))
        return {*existing, false};

      if (key_shard.count >= key_shard.reserved_keys) {
        size_t new_size = KeyShard::compute_next_capacity(
            key_shard.reserved_keys, key_shard.count + 1);
        key_shard.expand(new_size);
      }

      std::optional<uint32_t> preallocated_idx;
      bool was_inserted = false;
      auto upsert = [&](std::optional<uint32_t> existing) -> uint32_t {
        was_inserted = !existing.has_value();
        if (existing.has_value())
          return *existing;
        uint32_t idx;
        if (preallocated_idx.has_value()) {
          idx = *preallocated_idx;
        } else if (!key_shard.recycled_indices.empty()) {
          idx = key_shard.recycled_indices.back();
          key_shard.recycled_indices.pop_back();
          preallocated_idx = idx;
          arena.prepare_reused(idx);
          if (optimizer.has_value())
            optimizer->allocate(idx);
        } else {
          key_shard.prepare_new_index();
          idx = allocate_entry();
          ++key_shard.owned_index_count;
          preallocated_idx = idx;
        }
        initialize(idx);
        return idx;
      };
      auto update = [&](std::optional<typename Key::Entry> entry) {
        return std::make_pair(
            key, upsert(entry ? std::optional(entry->get_entry().second)
                              : std::nullopt));
      };
      auto construct = [&](const auto &entry) {
        auto map_key = Key::Entry::make_key(key);
        return key_shard.bundle->entries.make_entry(map_key, update(entry));
      };
      std::optional<std::pair<uint64_t, uint32_t>> published;
      try {
        published = key_shard.bundle->key_map.Upsert(Key::Entry::make_key(key),
                                                     construct, update);
      } catch (...) {
        if (preallocated_idx.has_value())
          key_shard.recycled_indices.push_back(*preallocated_idx);
        throw;
      }

      if (was_inserted) {
        ++key_shard.count;
      }

      if (preallocated_idx.has_value() && !was_inserted)
        key_shard.recycled_indices.push_back(*preallocated_idx);
      return {was_inserted ? *preallocated_idx : published->second,
              was_inserted};
    }

    void reserve(size_t keys_count) {
      size_t shard_count = key_shards.size();
      std::vector<size_t> counts(shard_count, keys_count / shard_count);
      for (size_t i = 0; i < keys_count % shard_count; ++i) {
        ++counts[i];
      }
      reserve_per_shard(counts);
    }

    void reserve_per_shard(std::span<const size_t> counts) {
      if (counts.size() != key_shards.size()) {
        throw std::invalid_argument("reserve_per_shard counts size mismatch");
      }
      size_t total_keys = 0;
      for (size_t i = 0; i < counts.size(); ++i) {
        std::shared_lock map_lock(key_shards[i]->map_mutex);
        size_t current_size = key_shards[i]->count;
        size_t target_keys = std::max(counts[i], current_size);
        if (target_keys > std::numeric_limits<size_t>::max() - total_keys) {
          throw std::overflow_error(
              "reserve_per_shard total capacity overflow");
        }
        total_keys += target_keys;
      }
      for (size_t i = 0; i < counts.size(); ++i) {
        std::unique_lock map_lock(key_shards[i]->map_mutex);
        size_t current_size = key_shards[i]->count;
        size_t target_keys = std::max(counts[i], current_size);
        if (target_keys <= key_shards[i]->reserved_keys) {
          continue;
        }
        key_shards[i]->expand(target_keys);
      }
      size_t arena_capacity =
          std::max(total_keys, static_cast<size_t>(arena.allocated_count()));
      arena.reserve_entries(arena_capacity);
      if (optimizer.has_value()) {
        optimizer->reserve(arena_capacity);
      }
    }
  };

  // =========================================================================
  // StorePageEvictionProxy — PageEvictionProxy implementation that aggregates
  // per-entry scores within each 4096-entry Arena page to produce a page-level
  // heat score.  Registered with PageStorage so the background swap thread
  // can determine cold pages without any hot-path overhead.
  //
  // page_idx is the monotonically-assigned index stored in PageRecord. Each
  // arena data and score page registers its own evaluator under that index.
  // =========================================================================
  class StorePageEvictionProxy {
  public:
    explicit StorePageEvictionProxy(
        std::shared_ptr<std::shared_mutex> lifecycle_mutex)
        : lifecycle_mutex_(std::move(lifecycle_mutex)) {}

    bool evaluate_page(size_t page_idx, uint64_t &score) {
      std::function<uint64_t()> evaluator;
      {
        std::lock_guard lock(mutex_);
        auto it = evaluators_.find(page_idx);
        if (it == evaluators_.end())
          return false;
        evaluator = it->second;
      }
      score = evaluator();
      return true;
    }

    void register_page(size_t page_idx, std::function<uint64_t()> evaluator) {
      std::lock_guard lock(mutex_);
      evaluators_[page_idx] = std::move(evaluator);
    }

    void unregister_page(size_t page_idx) noexcept {
      std::lock_guard lock(mutex_);
      evaluators_.erase(page_idx);
    }

    bool score_enabled() const noexcept { return true; }

    void lock_shared() { lifecycle_mutex_->lock_shared(); }
    void unlock_shared() noexcept { lifecycle_mutex_->unlock_shared(); }
    void lock() { lifecycle_mutex_->lock(); }
    void unlock() noexcept { lifecycle_mutex_->unlock(); }

  private:
    std::shared_ptr<std::shared_mutex> lifecycle_mutex_;
    std::mutex mutex_;
    std::unordered_map<size_t, std::function<uint64_t()>> evaluators_;
  };

  explicit HogwildEmbeddingStore(const std::vector<size_t> &table_dims,
                                 size_t num_threads = 0,
                                 Initializer initializer = {},
                                 OptimizerFactory opt_factory = nullptr,
                                 ScorePolicyT score_policy = {},
                                 uint8_t slot_bits = 16,
                                 PageStorage storage = create_page_storage())
      : num_threads_(num_threads > 0
                         ? num_threads
                         : std::max(1u, std::thread::hardware_concurrency())),
        pool_(static_cast<uint32_t>(num_threads_)),
        initializer_(std::move(initializer)),
        score_policy_(std::move(score_policy)), storage_(std::move(storage)),
        background_page_eviction_enabled_(
            ScorePolicyT::kHasScore &&
            storage_->supports_background_page_eviction()),
        lifecycle_mutex_(background_page_eviction_enabled_
                             ? storage_->lifecycle_mutex()
                             : std::make_shared<std::shared_mutex>()) {
    uint32_t p2 = 1;
    while (p2 < num_threads_) {
      p2 <<= 1;
    }
    num_threads_ = p2;

    bool enable_score = ScorePolicyT::kHasScore;

    if (table_dims.empty()) {
      throw std::invalid_argument("table_dims must not be empty");
    }

    size_t max_d = std::ranges::max(table_dims);
    if (max_d > 1048576) {
      throw std::invalid_argument("dim is too large (max 1048576)");
    }
    max_dim_ = max_d;
    dim_to_slot_.resize(max_dim_ + 1, nullptr);

    for (size_t dim : table_dims) {
      if (dim == 0) {
        throw std::invalid_argument("dim must be > 0");
      }
      if (dim_to_slot_[dim] == nullptr) {
        slot_storage_.push_back(std::make_unique<SlotStore>(
            dim, opt_factory, num_threads_, enable_score, storage_));
        dim_to_slot_[dim] = slot_storage_.back().get();
      }
    }

    if (background_page_eviction_enabled_) {
      auto proxy =
          pro::make_proxy_shared<PageEvictionProxyFacade,
                                 StorePageEvictionProxy>(lifecycle_mutex_);
      page_eviction_registration_ =
          storage_->set_page_eviction_proxy(std::move(proxy));
    }
  }

  ~HogwildEmbeddingStore() {
    if (page_eviction_registration_.value != 0)
      storage_->clear_page_eviction_proxy(page_eviction_registration_);
    std::unique_lock lifecycle_lock(*lifecycle_mutex_);
    slot_storage_.clear();
    dim_to_slot_.clear();
  }

  size_t max_dim() const { return max_dim_; }

  std::string_view name() const { return "hogwild"; }

  exec::static_thread_pool &pool() { return pool_; }

  // =========================================================================
  // Forward Pass: Lock-free lookup via ParlayHash
  // =========================================================================

  template <typename K>
  std::error_code forward_lookup(std::span<const K> keys,
                                 std::span<ParamT> out_flat_values,
                                 std::span<const int64_t> row_lengths) {
    std::shared_lock lifecycle_lock(*lifecycle_mutex_);
    size_t n = keys.size();
    auto offsets = compute_offsets(row_lengths);

    auto do_range = [&](size_t begin, size_t end) {
      epoch::with_epoch([&] {
        for (size_t i = begin; i < end; ++i) {
          if (i + 16 < end) {
            __builtin_prefetch(&keys[i + 16], 0, 1);
            __builtin_prefetch(&row_lengths[i + 16], 0, 1);
            __builtin_prefetch(&offsets[i + 16], 0, 1);
            __builtin_prefetch(out_flat_values.data() + offsets[i + 16], 1, 1);
          }
          uint64_t key = static_cast<uint64_t>(keys[i]);
          size_t dim = static_cast<size_t>(row_lengths[i]);
          if (dim >= dim_to_slot_.size() || dim_to_slot_[dim] == nullptr)
              [[unlikely]] {
            throw std::runtime_error(
                "invalid dim access: no physical table for dim " +
                std::to_string(dim));
          }
          auto &slot = *dim_to_slot_[dim];
          ParamT *out = out_flat_values.data() + offsets[i];

          auto found = slot.find(key);
          if (found.has_value()) {
            const ParamT *src = slot.arena.data(*found);
            if constexpr (simd::is_simd_supported_v<ParamT>) {
              simd::simd_copy(src, out, dim);
            } else {
              std::copy_n(src, dim, out);
            }
            if constexpr (ScorePolicyT::kHasScore) {
              score_policy_.on_access(&slot.arena.score_raw(*found));
            }
          } else {
            std::fill_n(out, dim, ParamT{0});
          }
        }
      });
    };

    dispatch_parallel(n, do_range);
    return {};
  }

  // =========================================================================
  // Backward Gradient Accumulation into TransientArena
  // =========================================================================

  template <typename K>
  std::error_code accumulate_gradients(std::span<const K> keys,
                                       std::span<const ParamT> flat_grads,
                                       std::span<const int64_t> row_lengths) {
    std::shared_lock lifecycle_lock(*lifecycle_mutex_);
    std::lock_guard training_lock(training_mutex_);
    size_t n = keys.size();
    auto offsets = compute_offsets(row_lengths);

    size_t num_shards = num_threads_;

    if (num_shards > std::numeric_limits<size_t>::max() / num_shards ||
        num_shards * num_shards >
            std::numeric_limits<size_t>::max() / sizeof(uint32_t)) {
      throw std::overflow_error("shard workspace size overflow");
    }
    size_t workspace_size = num_shards * num_shards;
    gradient_thread_counts_.resize(workspace_size);
    std::fill(gradient_thread_counts_.begin(), gradient_thread_counts_.end(),
              0);

    SplitMixHash<K> hasher;

    // Shared SIMD and scalar visitor keeps both paths identical.
    auto process_keys_simd = [&](size_t start, size_t end, auto &&visitor) {
      size_t i = start;
      if constexpr (sizeof(K) == 8) {
#if defined(__AVX512F__) && defined(__AVX512DQ__)
        __m512i v_mask = _mm512_set1_epi64(num_shards - 1);
        for (; i + 8 <= end; i += 8) {
          __m512i v_keys =
              _mm512_loadu_si512(reinterpret_cast<const void *>(&keys[i]));
          __m512i v_hash = hasher.simd_hash(v_keys);
          __m512i v_shards = _mm512_and_si512(v_hash, v_mask);

          uint64_t extracted[8] __attribute__((aligned(64)));
          _mm512_storeu_si512(extracted, v_shards);

          for (int j = 0; j < 8; ++j) {
            visitor(i + j, static_cast<uint32_t>(extracted[j]));
          }
        }
#endif
      }
      for (; i < end; ++i) {
        uint64_t key = static_cast<uint64_t>(keys[i]);
        uint32_t shard = hasher(key) & (num_shards - 1);
        visitor(i, shard);
      }
    };

    // 1. Parallel Count
    auto do_count = [&](size_t t_begin, size_t t_end) {
      for (size_t t = t_begin; t < t_end; ++t) {
        size_t start = t * n / num_shards;
        size_t end = (t + 1) * n / num_shards;
        process_keys_simd(start, end, [&](size_t /*idx*/, uint32_t shard) {
          gradient_thread_counts_[t * num_shards + shard]++;
        });
      }
    };
    dispatch_parallel(num_shards, do_count, 1);

    // 2. Prefix Sum (Sequential)
    gradient_thread_offsets_.resize(workspace_size);
    gradient_shard_offsets_.assign(num_shards + 1, 0);

    for (size_t s = 0; s < num_shards; ++s) {
      uint32_t current_offset = gradient_shard_offsets_[s];
      for (size_t t = 0; t < num_shards; ++t) {
        gradient_thread_offsets_[t * num_shards + s] = current_offset;
        current_offset += gradient_thread_counts_[t * num_shards + s];
      }
      gradient_shard_offsets_[s + 1] = current_offset;
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
        process_keys_simd(start, end, [&](size_t idx, uint32_t shard) {
          shuffled_indices[gradient_thread_offsets_[t * num_shards + shard]++] =
              idx;
        });
      }
    };
    dispatch_parallel(num_shards, do_scatter, 1);

    auto do_shard = [&](size_t shard_begin, size_t shard_end) {
      for (size_t shard_id = shard_begin; shard_id < shard_end; ++shard_id) {
        size_t start = gradient_shard_offsets_[shard_id];
        size_t end = gradient_shard_offsets_[shard_id + 1];
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
          if (dim >= dim_to_slot_.size() || dim_to_slot_[dim] == nullptr)
              [[unlikely]] {
            throw std::runtime_error(
                "invalid dim access: no physical table for dim " +
                std::to_string(dim));
          }
          auto &slot = *dim_to_slot_[dim];
          const ParamT *grad_src = flat_grads.data() + offsets[i];
          uint32_t t_idx =
              slot.transient_shards[shard_id]->get_or_allocate(key);
          slot.transient_shards[shard_id]->accumulate(t_idx, grad_src);
        }
      }
    };

    dispatch_parallel(num_shards, do_shard, 1);

    // Manual step requirement: optimizer_step() must be called explicitly

    return {};
  }

  // =========================================================================
  // Optimizer Step — Hogwild! lock-free parameter update
  //
  // Per Hogwild! theorem (Niu et al., NeurIPS 2011):
  // For sparse SGD-family optimizers, concurrent lock-free read/write to
  // shared parameters is provably convergent when feature access patterns
  // are sufficiently sparse. Therefore, NO LOCKS are acquired on the global
  // parameter KV table during this update. Forward reads may observe
  // partially-updated parameters, which does not affect convergence.
  // =========================================================================

  std::error_code optimizer_step() {
    std::shared_lock lifecycle_lock(*lifecycle_mutex_);
    std::lock_guard training_lock(training_mutex_);
    for (const auto &slot_ptr : slot_storage_) {
      auto *slot = slot_ptr.get();
      if (slot->optimizer.has_value()) {
        slot->global_step++;
      }
    }

    size_t num_shards = num_threads_;

    auto do_shard = [&](size_t shard_begin, size_t shard_end) {
      epoch::with_epoch([&] {
        for (size_t shard_id = shard_begin; shard_id < shard_end; ++shard_id) {
          for (const auto &slot_ptr : slot_storage_) {
            auto *slot = slot_ptr.get();
            auto &shard = slot->transient_shards[shard_id];
            shard->for_each_accumulated([&](uint64_t feature_id,
                                            const ParamT *accumulated_grad) {
              auto found = slot->find(feature_id);
              if (!found.has_value())
                return;

              uint32_t arena_idx = *found;
              ParamT *param = slot->arena.data(arena_idx);

              if (slot->optimizer.has_value()) {
                if constexpr (ScorePolicyT::kHasScore) {
                  const uint64_t score = std::atomic_ref<uint64_t>(
                                             slot->arena.score_raw(arena_idx))
                                             .load(std::memory_order_relaxed);
                  slot->optimizer->step(
                      arena_idx, std::span<ParamT>(param, slot->dim),
                      std::span<const ParamT>(accumulated_grad, slot->dim),
                      slot->global_step, score);
                } else {
                  slot->optimizer->step(
                      arena_idx, std::span<ParamT>(param, slot->dim),
                      std::span<const ParamT>(accumulated_grad, slot->dim),
                      slot->global_step, 0);
                }
              }
            });
            shard->clear();
          }
        }
      });
    };

    dispatch_parallel(num_shards, do_shard, 1);

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
    std::shared_lock lifecycle_lock(*lifecycle_mutex_);
    size_t n = keys.size();
    auto offsets = compute_offsets(row_lengths);

    auto do_range = [&](size_t begin, size_t end) {
      epoch::with_epoch([&] {
        for (size_t i = begin; i < end; ++i) {
          uint64_t key = static_cast<uint64_t>(keys[i]);
          size_t dim = static_cast<size_t>(row_lengths[i]);
          if (dim >= dim_to_slot_.size() || dim_to_slot_[dim] == nullptr)
              [[unlikely]] {
            throw std::runtime_error("invalid dim access");
          }
          auto &slot = *dim_to_slot_[dim];
          const ParamT *src = flat_values.data() + offsets[i];
          auto [arena_idx, inserted] =
              slot.find_or_insert(key, [&](uint32_t idx) {
                std::copy_n(src, slot.dim, slot.arena.data(idx));
              });
          if (!inserted)
            std::copy_n(src, slot.dim, slot.arena.data(arena_idx));
        }
      });
    };
    dispatch_parallel(n, do_range);
    return {};
  }

  template <typename K>
  std::error_code lookup_or_insert(std::span<const K> keys,
                                   std::span<ParamT> out_flat_values,
                                   std::span<const int64_t> row_lengths,
                                   std::span<uint8_t> out_inserted_mask) {
    std::shared_lock lifecycle_lock(*lifecycle_mutex_);
    size_t n = keys.size();
    auto offsets = compute_offsets(row_lengths);

    auto do_range = [&](size_t begin, size_t end) {
      epoch::with_epoch([&] {
        for (size_t i = begin; i < end; ++i) {
          uint64_t key = static_cast<uint64_t>(keys[i]);
          size_t dim = static_cast<size_t>(row_lengths[i]);
          if (dim >= dim_to_slot_.size() || dim_to_slot_[dim] == nullptr)
              [[unlikely]] {
            throw std::runtime_error("invalid dim access");
          }
          auto &slot = *dim_to_slot_[dim];
          ParamT *out = out_flat_values.data() + offsets[i];
          auto [arena_idx, inserted] =
              slot.find_or_insert(key, [&](uint32_t idx) {
                if (initializer_.has_value()) {
                  initializer_->initialize(
                      std::span<ParamT>(slot.arena.data(idx), dim), key);
                } else {
                  std::fill_n(slot.arena.data(idx), dim, ParamT{0});
                }
              });
          out_inserted_mask[i] = inserted;
          std::copy_n(slot.arena.data(arena_idx), dim, out);
        }
      });
    };
    dispatch_parallel(n, do_range);
    return {};
  }

  template <typename K>
  std::error_code contains(std::span<const K> keys,
                           std::span<uint8_t> out_exists) {
    std::shared_lock lifecycle_lock(*lifecycle_mutex_);
    size_t n = keys.size();
    auto do_range = [&](size_t begin, size_t end) {
      epoch::with_epoch([&] {
        for (size_t i = begin; i < end; ++i) {
          uint64_t key = static_cast<uint64_t>(keys[i]);
          bool found = false;
          for (const auto &slot_ptr : slot_storage_) {
            if (slot_ptr->find(key).has_value()) {
              found = true;
              break;
            }
          }
          out_exists[i] = found;
        }
      });
    };
    dispatch_parallel(n, do_range);
    return {};
  }

  template <typename K> std::error_code erase(std::span<const K> keys) {
    std::unique_lock lifecycle_lock(*lifecycle_mutex_);
    size_t n = keys.size();
    auto do_range = [&](size_t begin, size_t end) {
      epoch::with_epoch([&] {
        for (size_t i = begin; i < end; ++i) {
          uint64_t key = static_cast<uint64_t>(keys[i]);
          for (const auto &slot_ptr : slot_storage_) {
            auto &shard = slot_ptr->shard_for_key(key);
            std::unique_lock map_lock(shard.map_mutex);
            auto removed = shard.bundle->key_map.Remove(
                Key::Entry::make_key(key),
                [](const auto &entry) { return entry.get_entry().second; });
            if (removed.has_value()) {
              shard.recycled_indices.push_back(*removed);
              --shard.count;
            }
          }
        }
      });
    };
    dispatch_parallel(n, do_range);
    return {};
  }

  std::error_code clear() {
    std::unique_lock lifecycle_lock(*lifecycle_mutex_);
    for (const auto &slot_ptr : slot_storage_) {
      auto *slot = slot_ptr.get();
      slot->clear_keys();
      if (slot->optimizer.has_value()) {
        slot->optimizer->clear();
      }
      slot->arena.reset();
      slot->optimizer = slot->opt_factory.has_value()
                            ? (*slot->opt_factory)(slot->dim)
                            : nullptr;
      if (slot->optimizer.has_value() && slot->optimizer->requires_score() &&
          !ScorePolicyT::kHasScore) {
        throw std::invalid_argument(
            "dynamic learning-rate policy requires score storage");
      }
      if (slot->optimizer.has_value())
        slot->optimizer->set_page_eviction_enabled(ScorePolicyT::kHasScore);
    }
    return {};
  }

  std::error_code reserve(size_t keys_count, size_t dim) {
    std::unique_lock lifecycle_lock(*lifecycle_mutex_);
    slot_for_dim(dim).reserve(keys_count);
    return {};
  }

  std::error_code reserve_per_shard(std::span<const size_t> counts,
                                    size_t dim) {
    std::unique_lock lifecycle_lock(*lifecycle_mutex_);
    slot_for_dim(dim).reserve_per_shard(counts);
    return {};
  }

  size_t internal_shard_count() const { return num_threads_; }

  template <typename K>
  std::error_code assign_scores(std::span<const K> keys,
                                std::span<const uint64_t> scores) {
    std::shared_lock lifecycle_lock(*lifecycle_mutex_);
    if constexpr (!ScorePolicyT::kHasScore) {
      throw std::runtime_error("assign_scores requires a score-enabled policy");
    }
    size_t n = keys.size();
    auto do_range = [&](size_t begin, size_t end) {
      for (size_t i = begin; i < end; ++i) {
        uint64_t key = static_cast<uint64_t>(keys[i]);

        for (const auto &slot_ptr : slot_storage_) {
          auto *slot = slot_ptr.get();
          auto found = slot->find(key);

          if (found.has_value()) {
            uint32_t idx = *found;
            std::atomic_ref<uint64_t>(slot->arena.score_raw(idx))
                .store(scores[i], std::memory_order_relaxed);
            break; // Key found and updated, proceed to next key
          }
        }
      }
    };
    dispatch_parallel(n, do_range);
    return {};
  }

  size_t size() const {
    std::shared_lock lifecycle_lock(*lifecycle_mutex_);
    size_t s = 0;
    for (const auto &slot_ptr : slot_storage_) {
      s += slot_ptr->size();
    }
    return s;
  }

  size_t num_slots() const { return slot_storage_.size(); }

private:
  SlotStore &slot_for_dim(size_t dim) {
    if (dim >= dim_to_slot_.size() || dim_to_slot_[dim] == nullptr) {
      throw std::runtime_error(
          "invalid dim access: no physical table for dim " +
          std::to_string(dim));
    }
    return *dim_to_slot_[dim];
  }

  template <typename Fn>
  void dispatch_parallel(size_t n, Fn &&fn,
                         size_t threshold = kParallelThreshold) {
    if (n >= threshold) {
      size_t num_threads = num_threads_;
      size_t chunk_size = (n + num_threads - 1) / num_threads;
      stdexec::sync_wait(stdexec::schedule(pool_.get_scheduler()) |
                         stdexec::bulk(stdexec::par, num_threads,
                                       [&fn, n, chunk_size](size_t i) {
                                         size_t begin =
                                             std::min(i * chunk_size, n);
                                         size_t end =
                                             std::min(begin + chunk_size, n);
                                         if (begin < end) {
                                           fn(begin, end);
                                         }
                                       }));
    } else {
      fn(0, n);
    }
  }

  std::vector<size_t>
  compute_offsets(std::span<const int64_t> row_lengths) const {
    std::vector<size_t> offsets(row_lengths.size());
    auto casted = row_lengths | std::views::transform([](int64_t v) {
                    return static_cast<size_t>(v);
                  });
    std::exclusive_scan(casted.begin(), casted.end(), offsets.begin(),
                        size_t{0});
    return offsets;
  }

  enum class ExportOwnership { kSpan, kMove };

  template <typename K, ExportOwnership Policy, typename VisitorFn>
  void export_with_opt_impl(VisitorFn &&visitor, size_t batch_size) {
    std::shared_lock lifecycle_lock(*lifecycle_mutex_);
    for (const auto &slot_ptr : slot_storage_) {
      auto *slot = slot_ptr.get();
      size_t emb_dim = slot->dim;
      size_t opt_dim =
          slot->optimizer.has_value() ? slot->optimizer->state_dim() : 0;
      int num_workers = static_cast<int>(parlay::num_workers());

      constexpr bool has_score = ScorePolicyT::kHasScore;

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
        if constexpr (has_score)
          local_bufs[i].scores.reserve(256);
        if (opt_dim > 0) {
          local_bufs[i].opts.reserve(256 * opt_dim);
          local_bufs[i].steps.reserve(256);
        }
      }

      Buffer global_buf;
      global_buf.keys.reserve(batch_size);
      global_buf.vals.reserve(batch_size * emb_dim);
      if constexpr (has_score)
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
          if constexpr (has_score)
            buf.scores.reserve(batch_size);
          if (opt_dim > 0) {
            buf.opts.reserve(batch_size * opt_dim);
            buf.steps.reserve(batch_size);
          }
          return cont;
        }
      };

      slot->for_each([&](const auto &entry) {
        if (stopped.load(std::memory_order_relaxed))
          return;

        int w = parlay::worker_id();
        auto [key, idx] = entry.get_entry();
        auto &lb = local_bufs[w];

        lb.keys.push_back(static_cast<K>(key));
        const ParamT *src = slot->arena.data(idx);
        lb.vals.insert(lb.vals.end(), src, src + emb_dim);
        if constexpr (has_score) {
          lb.scores.push_back(
              std::atomic_ref<uint64_t>(slot->arena.score_raw(idx))
                  .load(std::memory_order_relaxed));
        }
        if (opt_dim > 0 && slot->optimizer.has_value()) {
          // OptimizerFacade::state_data fills a caller-provided span (so the
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
            if constexpr (has_score) {
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
              if constexpr (has_score)
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
          if constexpr (has_score) {
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

  std::vector<SlotStore *> dim_to_slot_;
  std::vector<std::unique_ptr<SlotStore>> slot_storage_;
  size_t num_threads_ = 0;
  exec::static_thread_pool pool_;
  Initializer initializer_;
  [[no_unique_address]] ScorePolicyT score_policy_;
  PageStorage storage_;
  bool background_page_eviction_enabled_;
  std::shared_ptr<std::shared_mutex> lifecycle_mutex_;
  PageEvictionRegistration page_eviction_registration_;
  std::mutex training_mutex_;
  std::vector<uint32_t> gradient_thread_counts_;
  std::vector<uint32_t> gradient_thread_offsets_;
  std::vector<uint32_t> gradient_shard_offsets_;
  size_t max_dim_ = 0;
  static constexpr size_t kParallelThreshold = 64;

public:
  bool has_score_policy() const { return ScorePolicyT::kHasScore; }

  std::vector<size_t> table_dims() const {
    std::vector<size_t> result;
    for (const auto &slot : slot_storage_) {
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
    std::shared_lock lifecycle_lock(*lifecycle_mutex_);
    bool has_score = ScorePolicyT::kHasScore;

    for (const auto &slot_ptr : slot_storage_) {
      auto *slot = slot_ptr.get();
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

      slot->for_each([&](const auto &entry) {
        if (stopped.load(std::memory_order_relaxed))
          return;

        int w = parlay::worker_id();
        auto [key, idx] = entry.get_entry();
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

  void assign_opt_states(size_t dim, std::span<const KeyT> keys,
                         std::span<const OptT> opt_states,
                         std::span<const uint32_t> steps) {
    std::shared_lock lifecycle_lock(*lifecycle_mutex_);
    size_t n = keys.size();
    if (dim >= dim_to_slot_.size() || dim_to_slot_[dim] == nullptr)
      throw std::invalid_argument(
          "assign_opt_states dimension has no physical table");
    auto &slot = *dim_to_slot_[dim];
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

    epoch::with_epoch([&] {
      for (size_t i = 0; i < n; ++i) {
        uint64_t key = keys[i];
        auto found = slot.find(key);
        if (!found.has_value())
          throw std::invalid_argument(
              "assign_opt_states key is absent from dimension table");
        slot.optimizer->set_state(
            *found,
            std::span<const OptT>(opt_states.data() + i * opt_dim, opt_dim));
        if (!steps.empty())
          slot.optimizer->set_step(*found, steps[i]);
      }
    });
  }

  // export_opt_state_batch — type-erased facade-compatible wrapper.
  // Visitor receives (dim, keys, vals, scores, opt_states, steps) with native
  // ParamT/OptT precision.
  void export_opt_state_batch(
      se::Function<bool(size_t, std::span<const KeyT>, std::span<const ParamT>,
                        std::span<const uint64_t>, std::span<const OptT>,
                        std::span<const uint32_t>)>
          visitor,
      size_t batch_size) {
    export_batch_with_opt<KeyT>(
        [&](size_t dim, std::span<const KeyT> keys,
            std::span<const ParamT> vals, std::span<const uint64_t> scores,
            std::span<const OptT> opt_states,
            std::span<const uint32_t> steps) -> bool {
          return (*visitor)(dim, keys, vals, scores, opt_states, steps);
        },
        batch_size);
  }

  // export_opt_state_batch_owned — ownership-transferring facade wrapper.
  // Visitor receives std::vector&& of native ParamT/OptT precision for
  // zero-copy Arrow C Data Interface construction.
  void export_opt_state_batch_owned(
      se::Function<bool(size_t, std::vector<KeyT> &&, std::vector<ParamT> &&,
                        std::vector<uint64_t> &&, std::vector<OptT> &&,
                        std::vector<uint32_t> &&)>
          visitor,
      size_t batch_size) {
    export_batch_with_opt_owned<KeyT>(
        [&](size_t dim, std::vector<KeyT> &&keys, std::vector<ParamT> &&vals,
            std::vector<uint64_t> &&scores, std::vector<OptT> &&opt_states,
            std::vector<uint32_t> &&steps) -> bool {
          return (*visitor)(dim, std::move(keys), std::move(vals),
                            std::move(scores), std::move(opt_states),
                            std::move(steps));
        },
        batch_size);
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
    std::unique_lock lifecycle_lock(*lifecycle_mutex_);
    if constexpr (!ScorePolicyT::kHasScore)
      throw std::runtime_error(
          "evict_by_score requires a score-enabled policy");

    out_evicted_count = 0;

    for (const auto &slot_ptr : slot_storage_) {
      auto *slot = slot_ptr.get();
      if (out_evicted_count >= max_evict_count)
        break;

      size_t dim = slot->dim;
      int num_workers = static_cast<int>(parlay::num_workers());
      size_t slot_budget = max_evict_count - out_evicted_count;

      struct LocalEvictBuffer {
        std::vector<std::pair<uint64_t, uint64_t>> candidates;
      };
      std::vector<LocalEvictBuffer> local_buffers(num_workers);
      for (int i = 0; i < num_workers; ++i) {
        local_buffers[i].candidates.reserve(1024);
      }

      std::mutex m;
      std::vector<std::pair<uint64_t, uint64_t>> global_candidates;
      global_candidates.reserve(slot_budget);

      auto merge_global =
          [&](std::vector<std::pair<uint64_t, uint64_t>> &local_cands) {
            std::lock_guard<std::mutex> lock(m);
            for (auto &cand : local_cands) {
              if (global_candidates.size() < slot_budget) {
                global_candidates.push_back(cand);
                if (global_candidates.size() == slot_budget) {
                  std::make_heap(global_candidates.begin(),
                                 global_candidates.end());
                }
              } else if (cand.first < global_candidates.front().first) {
                std::pop_heap(global_candidates.begin(),
                              global_candidates.end());
                global_candidates.back() = cand;
                std::push_heap(global_candidates.begin(),
                               global_candidates.end());
              }
            }
            local_cands.clear();
          };

      slot->for_each([&](const auto &entry) {
        int w = parlay::worker_id();
        auto [key, idx] = entry.get_entry();
        uint64_t s = std::atomic_ref<uint64_t>(slot->arena.score_raw(idx))
                         .load(std::memory_order_relaxed);
        if (s <= max_score) {
          local_buffers[w].candidates.emplace_back(s, key);
          if (local_buffers[w].candidates.size() >= 1024) {
            merge_global(local_buffers[w].candidates);
          }
        }
      });

      for (int w = 0; w < num_workers; ++w) {
        if (!local_buffers[w].candidates.empty()) {
          merge_global(local_buffers[w].candidates);
        }
      }

      // Sort ascending by score — lowest score evicted first.
      std::ranges::sort(global_candidates);

      epoch::with_epoch([&] {
        for (auto &[score, key] : global_candidates) {
          if (out_evicted_count >= max_evict_count)
            break;

          auto found = slot->find(key);
          if (!found.has_value())
            continue;

          uint32_t idx = *found;
          if (out_evicted_count < out_evicted_keys.size()) {
            out_evicted_keys[out_evicted_count] = static_cast<K>(key);
            std::copy_n(slot->arena.data(idx), dim,
                        out_evicted_values.data() + out_evicted_count * dim);
          }
          auto &shard = slot->shard_for_key(key);
          std::unique_lock map_lock(shard.map_mutex);
          auto removed = shard.bundle->key_map.Remove(
              Key::Entry::make_key(key),
              [](const auto &entry) { return entry.get_entry().second; });
          if (removed.has_value()) {
            shard.recycled_indices.push_back(*removed);
            --shard.count;
          }
          ++out_evicted_count;
        }
      });
    }
    return {};
  }
};

// =============================================================================
// Extern template declarations — suppress implicit instantiation in TUs that
// include this header. The explicit instantiation lives in
// hogwild_embedding_store.cpp.
// =============================================================================
#ifndef HOGWILD_EXPLICIT_INSTANTIATION

#define SE_EXTERN_STORE(KEY_T, PARAM_T, OPT_T)                                 \
  extern template class HogwildEmbeddingStore<KEY_T, PARAM_T, OPT_T,           \
                                              NoScorePolicy>;                  \
  extern template class HogwildEmbeddingStore<KEY_T, PARAM_T, OPT_T,           \
                                              LfuScorePolicy>;                 \
  extern template class HogwildEmbeddingStore<KEY_T, PARAM_T, OPT_T,           \
                                              LruScorePolicy>;                 \
  extern template class HogwildEmbeddingStore<KEY_T, PARAM_T, OPT_T,           \
                                              ManualScorePolicy>;

#define SE_EXTERN_OPT_PAIR(PARAM_T, OPT_T)                                     \
  SE_EXTERN_STORE(uint64_t, PARAM_T, OPT_T)                                    \
  SE_EXTERN_STORE(int64_t, PARAM_T, OPT_T)                                     \
  SE_EXTERN_STORE(uint32_t, PARAM_T, OPT_T)                                    \
  SE_EXTERN_STORE(int32_t, PARAM_T, OPT_T)

#define SE_EXTERN_ALL(PARAM_T)                                                 \
  SE_EXTERN_OPT_PAIR(PARAM_T, float)                                           \
  SE_EXTERN_OPT_PAIR(PARAM_T, float16_t)                                       \
  SE_EXTERN_OPT_PAIR(PARAM_T, bfloat16_t)

SE_EXTERN_ALL(float)
SE_EXTERN_ALL(float16_t)
SE_EXTERN_ALL(bfloat16_t)

#undef SE_EXTERN_STORE
#undef SE_EXTERN_OPT_PAIR
#undef SE_EXTERN_ALL

#endif // HOGWILD_EXPLICIT_INSTANTIATION

} // namespace se

#endif // SPARSE_EMBEDDING_HOGWILD_EMBEDDING_STORE_HPP_
