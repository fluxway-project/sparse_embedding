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

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <vector>

#include <exec/static_thread_pool.hpp>
#include <stdexec/execution.hpp>

#include <parlay_hash/unordered_map.h>

#include "sparse_embedding/src/arena/transient_arena.hpp"
#include "sparse_embedding/src/backend/hogwild/parallel_dispatch.hpp"
#include "sparse_embedding/src/common/embedding_error.hpp"
#include "sparse_embedding/src/common/numeric_types.hpp"
#include "sparse_embedding/src/interface/initializer_interface.hpp"
#include "sparse_embedding/src/interface/optimizer_interface.hpp"
#include "sparse_embedding/src/optimizer/optimizer_state.hpp"
#include "sparse_embedding/src/simd/simd_ops.hpp"

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
  explicit EmbeddingArena(size_t dim)
      : dim_(dim), pages_(std::make_unique<PageTable>()) {
    data_stride_ =
        (dim * sizeof(ValT) + kCacheLineSize - 1) & ~(kCacheLineSize - 1);
  }

  ~EmbeddingArena() {
    for (size_t i = 0; i < kMaxPages; ++i) {
      auto *dp = pages_->data[i].load(std::memory_order_relaxed);
      if (dp)
        std::free(dp);
      auto *sp = pages_->score[i].load(std::memory_order_relaxed);
      delete[] sp;
    }
  }

  EmbeddingArena(const EmbeddingArena &) = delete;
  EmbeddingArena &operator=(const EmbeddingArena &) = delete;
  EmbeddingArena(EmbeddingArena &&) = delete;
  EmbeddingArena &operator=(EmbeddingArena &&) = delete;

  size_t dim() const { return dim_; }

  uint32_t allocate() {
    uint32_t idx = next_index_.fetch_add(1, std::memory_order_relaxed);
    ensure_capacity(idx);
    std::memset(data_raw(idx), 0, data_stride_);
    score(idx).store(0, std::memory_order_relaxed);
    return idx;
  }

  std::atomic_ref<uint64_t> score(uint32_t index) const {
    return std::atomic_ref<uint64_t>(
        pages_->score[index / kEntriesPerPage].load(
            std::memory_order_acquire)[index % kEntriesPerPage]);
  }

  ValT *data(uint32_t index) const {
    return reinterpret_cast<ValT *>(data_raw(index));
  }

  size_t allocated_count() const {
    return next_index_.load(std::memory_order_relaxed);
  }

  void reset() {
    std::lock_guard lock(grow_mtx_);
    for (size_t i = 0; i < kMaxPages; ++i) {
      auto *dp = pages_->data[i].exchange(nullptr, std::memory_order_relaxed);
      if (dp)
        std::free(dp);
      auto *sp = pages_->score[i].exchange(nullptr, std::memory_order_relaxed);
      delete[] sp;
    }
    num_pages_.store(0, std::memory_order_relaxed);
    next_index_.store(0, std::memory_order_relaxed);
  }

private:
  void ensure_capacity(uint32_t idx) {
    size_t page_needed = idx / kEntriesPerPage;
    if (page_needed < num_pages_.load(std::memory_order_acquire))
      return;
    std::lock_guard lock(grow_mtx_);
    while (page_needed >= num_pages_.load(std::memory_order_relaxed)) {
      size_t pg = num_pages_.load(std::memory_order_relaxed);
      auto *dp = static_cast<std::byte *>(
          std::aligned_alloc(kCacheLineSize, kEntriesPerPage * data_stride_));
      if (!dp)
        throw std::bad_alloc();
      std::memset(dp, 0, kEntriesPerPage * data_stride_);
      auto *sp = new uint64_t[kEntriesPerPage]();
      pages_->data[pg].store(dp, std::memory_order_release);
      pages_->score[pg].store(sp, std::memory_order_release);
      num_pages_.store(pg + 1, std::memory_order_release);
    }
  }

  std::byte *data_raw(uint32_t index) const {
    return pages_->data[index / kEntriesPerPage].load(
               std::memory_order_acquire) +
           (index % kEntriesPerPage) * data_stride_;
  }

  size_t dim_;
  size_t data_stride_;
  std::unique_ptr<PageTable> pages_;
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

template <typename ValT> class HogwildEmbeddingStore {
public:
  using ValueType = ValT;
  using Initializer = pro::proxy<InitializerFacade<ValT>>;
  using OptimizerFactory =
      std::function<pro::proxy<OptimizerFacade<ValT>>(size_t)>;

  struct SplitMix64Hash {
    size_t operator()(uint64_t x) const {
      x ^= x >> 30;
      x *= 0xbf58476d1ce4e5b9ull;
      x ^= x >> 27;
      x *= 0x94d049bb133111ebull;
      x ^= x >> 31;
      return x;
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

  using MapData = parlay::MapData<uint64_t, uint32_t, SplitMix64Hash,
                                  std::equal_to<uint64_t>>;
  using Key = parlay::IndirectEntries<MapData>;
  using KeyMap = parlay::parlay_hash<Key>;

  struct SlotStore {
    size_t dim;
    Key entries;
    KeyMap key_map;
    EmbeddingArena<ValT> arena;
    pro::proxy<OptimizerFacade<ValT>> optimizer;
    std::vector<std::unique_ptr<TransientArena<ValT>>> transient_shards;
    OptimizerFactory opt_factory;
    uint64_t global_step = 1;

    SlotStore(size_t d, size_t max_active_keys,
              const OptimizerFactory &opt_factory, size_t num_threads)
        : dim(d), entries(false), key_map(1024, &entries, false), arena(d),
          optimizer(opt_factory(d)), opt_factory(opt_factory) {
      for (size_t i = 0; i < num_threads; ++i) {
        transient_shards.push_back(std::make_unique<TransientArena<ValT>>(
            d, max_active_keys / num_threads + 1));
      }
    }
  };

  explicit HogwildEmbeddingStore(std::unordered_map<uint32_t, size_t> slot_dims,
                                 size_t num_threads = 0,
                                 Initializer initializer = {},
                                 size_t max_active_keys = 1 << 20,
                                 OptimizerFactory opt_factory = nullptr)
      : num_threads_(num_threads > 0
                         ? num_threads
                         : std::max(1u, std::thread::hardware_concurrency())),
        pool_(static_cast<uint32_t>(num_threads_)),
        initializer_(std::move(initializer)) {
    uint32_t p2 = 1;
    while (p2 < num_threads_) {
      p2 <<= 1;
    }
    num_threads_ = p2;

    if (slot_dims.empty()) {
      throw std::invalid_argument("slot_dims must not be empty");
    }
    for (auto &[slot_id, dim] : slot_dims) {
      if (dim == 0) {
        throw std::invalid_argument("dim must be > 0 for slot " +
                                    std::to_string(slot_id));
      }
      slots_.emplace(slot_id,
                     std::make_unique<SlotStore>(dim, max_active_keys,
                                                 opt_factory, num_threads_));
      if (dim > max_dim_)
        max_dim_ = dim;
    }
  }

  static uint32_t extract_slot(uint64_t key) {
    return static_cast<uint32_t>((key >> 48) & 0xFFFF);
  }

  size_t max_dim() const { return max_dim_; }

  size_t dim_for_slot(uint32_t slot_id) const {
    auto it = slots_.find(slot_id);
    if (it == slots_.end())
      throw std::invalid_argument("unknown slot " + std::to_string(slot_id));
    return it->second->dim;
  }

  size_t dim_for_key(uint64_t key) const {
    uint32_t slot_id = extract_slot(key);
    return slots_.at(slot_id)->dim;
  }

  std::string_view name() const { return "hogwild"; }

  exec::static_thread_pool &pool() { return pool_; }

  // =========================================================================
  // Forward Pass: Lock-free lookup via ParlayHash
  // =========================================================================

  template <typename KeyT>
  std::error_code forward_lookup(std::span<const KeyT> keys,
                                 std::span<ValT> out_flat_values,
                                 std::span<const int64_t> row_lengths) {
    size_t n = keys.size();
    auto offsets = compute_offsets(row_lengths);

    auto do_range = [&](size_t begin, size_t end) {
      for (size_t i = begin; i < end; ++i) {
        uint64_t key = static_cast<uint64_t>(keys[i]);
        uint32_t slot_id = extract_slot(key);
        auto &slot = *slots_.at(slot_id);
        size_t dim = static_cast<size_t>(row_lengths[i]);
        ValT *out = out_flat_values.data() + offsets[i];

        auto found = epoch::with_epoch([&] {
          return slot.key_map.Find(
              Key::Entry::make_key(key),
              [](const auto &e) { return e.get_entry().second; });
        });
        if (found.has_value()) {
          const ValT *src = slot.arena.data(*found);
          if constexpr (simd::is_simd_supported_v<ValT>) {
            simd::simd_copy(src, out, dim);
          } else {
            std::copy_n(src, dim, out);
          }
        } else {
          std::fill_n(out, dim, ValT{0});
        }
      }
    };

    dispatch_parallel(n, do_range);
    return {};
  }

  // =========================================================================
  // Backward Gradient Accumulation into TransientArena
  // =========================================================================

  template <typename KeyT>
  std::error_code accumulate_gradients(std::span<const KeyT> keys,
                                       std::span<const ValT> flat_grads,
                                       std::span<const int64_t> row_lengths) {
    size_t n = keys.size();
    auto offsets = compute_offsets(row_lengths);

    uint32_t num_shards = num_threads_;
    if (num_shards == 0)
      num_shards = std::thread::hardware_concurrency();

    std::vector<std::vector<uint32_t>> thread_counts(
        num_shards, std::vector<uint32_t>(num_shards, 0));
    SplitMix64Hash hasher;

    // 1. Parallel Count
    auto do_count = [&](size_t t_begin, size_t t_end) {
      for (size_t t = t_begin; t < t_end; ++t) {
        size_t start = t * n / num_shards;
        size_t end = (t + 1) * n / num_shards;
        size_t i = start;

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
            thread_counts[t][extracted[j]]++;
          }
        }
#endif
        for (; i < end; ++i) {
          uint64_t key = static_cast<uint64_t>(keys[i]);
          uint32_t shard = hasher(key) & (num_shards - 1);
          thread_counts[t][shard]++;
        }
      }
    };
    dispatch_parallel(num_shards, do_count);

    // 2. Prefix Sum (Sequential)
    std::vector<std::vector<uint32_t>> thread_offsets(
        num_shards, std::vector<uint32_t>(num_shards, 0));
    std::vector<uint32_t> shard_offsets(num_shards + 1, 0);

    for (size_t s = 0; s < num_shards; ++s) {
      uint32_t current_offset = shard_offsets[s];
      for (size_t t = 0; t < num_shards; ++t) {
        thread_offsets[t][s] = current_offset;
        current_offset += thread_counts[t][s];
      }
      shard_offsets[s + 1] = current_offset;
    }

    // 3. Parallel Scatter
    std::vector<uint32_t> shuffled_indices(n);
    auto do_scatter = [&](size_t t_begin, size_t t_end) {
      for (size_t t = t_begin; t < t_end; ++t) {
        size_t start = t * n / num_shards;
        size_t end = (t + 1) * n / num_shards;
        size_t i = start;

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
            shuffled_indices[thread_offsets[t][extracted[j]]++] = i + j;
          }
        }
#endif
        for (; i < end; ++i) {
          uint64_t key = static_cast<uint64_t>(keys[i]);
          uint32_t shard = hasher(key) & (num_shards - 1);
          shuffled_indices[thread_offsets[t][shard]++] = i;
        }
      }
    };
    dispatch_parallel(num_shards, do_scatter);

    auto do_shard = [&](size_t shard_begin, size_t shard_end) {
      for (size_t shard_id = shard_begin; shard_id < shard_end; ++shard_id) {
        size_t start = shard_offsets[shard_id];
        size_t end = shard_offsets[shard_id + 1];
        for (size_t idx = start; idx < end; ++idx) {
          uint32_t i = shuffled_indices[idx];
          uint64_t key = static_cast<uint64_t>(keys[i]);
          uint32_t slot_id = extract_slot(key);
          auto &slot = *slots_.at(slot_id);
          const ValT *grad_src = flat_grads.data() + offsets[i];
          uint32_t t_idx =
              slot.transient_shards[shard_id]->get_or_allocate(key);
          slot.transient_shards[shard_id]->accumulate(t_idx, grad_src);
        }
      }
    };

    dispatch_parallel(num_shards, do_shard);

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
    for (auto &[_, slot] : slots_) {
      slot->global_step++;
    }

    uint32_t num_shards = num_threads_;
    if (num_shards == 0)
      num_shards = std::thread::hardware_concurrency();

    auto do_shard = [&](size_t shard_begin, size_t shard_end) {
      for (size_t shard_id = shard_begin; shard_id < shard_end; ++shard_id) {
        for (auto &[_, slot] : slots_) {
          auto &shard = slot->transient_shards[shard_id];
          shard->for_each_accumulated(
              [&](uint64_t feature_id, const ValT *accumulated_grad) {
                auto found = epoch::with_epoch([&] {
                  return slot->key_map.Find(
                      Key::Entry::make_key(feature_id),
                      [](const auto &e) { return e.get_entry().second; });
                });
                if (!found.has_value())
                  return;

                uint32_t arena_idx = *found;
                ValT *param = slot->arena.data(arena_idx);

                slot->optimizer->step(
                    arena_idx, std::span<ValT>(param, slot->dim),
                    std::span<const ValT>(accumulated_grad, slot->dim),
                    slot->global_step);
              });
          shard->clear();
        }
      }
    };

    dispatch_parallel(num_shards, do_shard);

    return {};
  }

  // =========================================================================
  // Legacy API (backward compatible)
  // =========================================================================

  template <typename KeyT>
  std::error_code lookup(std::span<const KeyT> keys,
                         std::span<ValT> out_flat_values,
                         std::span<const int64_t> row_lengths) {
    return forward_lookup(keys, out_flat_values, row_lengths);
  }

  template <typename KeyT>
  size_t compute_output_size(std::span<const KeyT> keys,
                             std::span<int64_t> out_row_lengths) const {
    size_t total = 0;
    for (size_t i = 0; i < keys.size(); ++i) {
      uint64_t key = static_cast<uint64_t>(keys[i]);
      auto dim = static_cast<int64_t>(dim_for_key(key));
      out_row_lengths[i] = dim;
      total += dim;
    }
    return total;
  }

  template <typename KeyT>
  std::error_code assign(std::span<const KeyT> keys,
                         std::span<const ValT> flat_values,
                         std::span<const int64_t> row_lengths) {
    size_t n = keys.size();
    auto offsets = compute_offsets(row_lengths);

    auto do_range = [&](size_t begin, size_t end) {
      for (size_t i = begin; i < end; ++i) {
        uint64_t key = static_cast<uint64_t>(keys[i]);
        uint32_t slot_id = extract_slot(key);
        auto &slot = *slots_.at(slot_id);
        const ValT *src = flat_values.data() + offsets[i];

        auto upsert_fn = [&](std::optional<uint32_t> existing) -> uint32_t {
          if (existing.has_value()) {
            std::copy_n(src, slot.dim, slot.arena.data(*existing));
            return *existing;
          }
          uint32_t idx = slot.arena.allocate();
          slot.optimizer->allocate();
          std::copy_n(src, slot.dim, slot.arena.data(idx));
          return idx;
        };

        epoch::with_epoch([&] {
          auto g = [&](std::optional<typename Key::Entry> opt_e)
              -> std::pair<uint64_t, uint32_t> {
            if (opt_e.has_value()) {
              return std::make_pair(key, upsert_fn(opt_e->get_entry().second));
            }
            return std::make_pair(key, upsert_fn(std::nullopt));
          };
          auto constr = [&](const auto &opt_e) {
            return opt_e.has_value()
                       ? slot.entries.make_entry(Key::Entry::make_key(key),
                                                 g(opt_e))
                       : slot.entries.make_entry(Key::Entry::make_key(key),
                                                 g(std::nullopt));
          };
          return slot.key_map.Upsert(Key::Entry::make_key(key), constr, g);
        });
      }
    };
    dispatch_parallel(n, do_range);
    return {};
  }

  template <typename KeyT>
  std::error_code lookup_or_insert(std::span<const KeyT> keys,
                                   std::span<ValT> out_flat_values,
                                   std::span<const int64_t> row_lengths,
                                   std::span<bool> out_inserted_mask) {
    size_t n = keys.size();
    auto offsets = compute_offsets(row_lengths);

    auto do_range = [&](size_t begin, size_t end) {
      for (size_t i = begin; i < end; ++i) {
        uint64_t key = static_cast<uint64_t>(keys[i]);
        uint32_t slot_id = extract_slot(key);
        auto &slot = *slots_.at(slot_id);
        size_t dim = slot.dim;
        ValT *out = out_flat_values.data() + offsets[i];

        bool was_inserted = false;
        auto upsert_fn = [&](std::optional<uint32_t> existing) -> uint32_t {
          if (existing.has_value()) {
            return *existing;
          }
          was_inserted = true;
          uint32_t idx = slot.arena.allocate();
          slot.optimizer->allocate();
          if (initializer_.has_value()) {
            initializer_->initialize(std::span<ValT>(slot.arena.data(idx), dim),
                                     key);
          } else {
            std::fill_n(slot.arena.data(idx), dim, ValT{0});
          }
          return idx;
        };

        auto prev = epoch::with_epoch([&] {
          auto g = [&](std::optional<typename Key::Entry> opt_e)
              -> std::pair<uint64_t, uint32_t> {
            if (opt_e.has_value()) {
              return std::make_pair(key, upsert_fn(opt_e->get_entry().second));
            }
            return std::make_pair(key, upsert_fn(std::nullopt));
          };
          auto constr = [&](const auto &opt_e) {
            return opt_e.has_value()
                       ? slot.entries.make_entry(Key::Entry::make_key(key),
                                                 g(opt_e))
                       : slot.entries.make_entry(Key::Entry::make_key(key),
                                                 g(std::nullopt));
          };
          return slot.key_map.Upsert(Key::Entry::make_key(key), constr, g);
        });
        out_inserted_mask[i] = was_inserted;

        uint32_t arena_idx =
            was_inserted ? epoch::with_epoch([&] {
                             return slot.key_map.Find(
                                 Key::Entry::make_key(key), [](const auto &e) {
                                   return e.get_entry().second;
                                 });
                           }).value()
                         : prev->second;
        std::copy_n(slot.arena.data(arena_idx), dim, out);
      }
    };
    dispatch_parallel(n, do_range);
    return {};
  }

  template <typename KeyT>
  std::error_code contains(std::span<const KeyT> keys,
                           std::span<bool> out_exists) {
    size_t n = keys.size();
    auto do_range = [&](size_t begin, size_t end) {
      for (size_t i = begin; i < end; ++i) {
        uint64_t key = static_cast<uint64_t>(keys[i]);
        uint32_t slot_id = extract_slot(key);
        auto &slot = *slots_.at(slot_id);
        out_exists[i] =
            epoch::with_epoch([&] {
              return slot.key_map.Find(Key::Entry::make_key(key),
                                       [](const auto &e) { return true; });
            }).has_value();
      }
    };
    dispatch_parallel(n, do_range);
    return {};
  }

  template <typename KeyT> std::error_code erase(std::span<const KeyT> keys) {
    size_t n = keys.size();
    auto do_range = [&](size_t begin, size_t end) {
      for (size_t i = begin; i < end; ++i) {
        uint64_t key = static_cast<uint64_t>(keys[i]);
        uint32_t slot_id = extract_slot(key);
        auto &slot = *slots_.at(slot_id);
        epoch::with_epoch([&] {
          slot.key_map.Remove(Key::Entry::make_key(key),
                              [](const auto &e) { return true; });
        });
      }
    };
    dispatch_parallel(n, do_range);
    return {};
  }

  std::error_code clear() {
    for (auto &[_, slot] : slots_) {
      slot->key_map.clear();
      slot->optimizer->clear();
      slot->arena.reset();
      slot->optimizer = slot->opt_factory(slot->dim);
    }
    return {};
  }

  template <typename KeyT>
  std::error_code assign_scores(std::span<const KeyT> keys,
                                std::span<const uint64_t> scores) {
    size_t n = keys.size();
    auto do_range = [&](size_t begin, size_t end) {
      for (size_t i = begin; i < end; ++i) {
        uint64_t key = static_cast<uint64_t>(keys[i]);
        uint32_t slot_id = extract_slot(key);
        auto &slot = *slots_.at(slot_id);

        epoch::with_epoch([&] {
          auto g = [&](std::optional<typename Key::Entry> opt_e)
              -> std::pair<uint64_t, uint32_t> {
            uint32_t idx;
            if (opt_e.has_value()) {
              idx = opt_e->get_entry().second;
            } else {
              idx = slot.arena.allocate();
              slot.optimizer->allocate();
            }
            slot.arena.score(idx).store(scores[i], std::memory_order_relaxed);
            return std::make_pair(key, idx);
          };
          auto constr = [&](const auto &e) {
            return e.has_value()
                       ? slot.entries.make_entry(Key::Entry::make_key(key),
                                                 g(e))
                       : slot.entries.make_entry(Key::Entry::make_key(key),
                                                 g(std::nullopt));
          };
          return slot.key_map.Upsert(Key::Entry::make_key(key), constr, g);
        });
      }
    };
    dispatch_parallel(n, do_range);
    return {};
  }

  size_t size() const {
    size_t s = 0;
    for (auto &[_, slot] : slots_) {
      s += slot->key_map.size();
    }
    return s;
  }

  size_t num_slots() const { return slots_.size(); }

private:
  template <typename Fn> void dispatch_parallel(size_t n, Fn &&fn) {
    if (n >= kParallelThreshold) {
      uint32_t num_threads = num_threads_;
      if (num_threads == 0)
        num_threads = std::thread::hardware_concurrency();
      TaskClosure closure{
          &fn, [](void *ctx, size_t begin, size_t end) {
            auto *f =
                static_cast<std::add_pointer_t<std::remove_reference_t<Fn>>>(
                    ctx);
            (*f)(begin, end);
          }};
      dispatch_parallel_tasks(pool_, num_threads, n, closure);
    } else {
      fn(0, n);
    }
  }

  std::vector<size_t>
  compute_offsets(std::span<const int64_t> row_lengths) const {
    std::vector<size_t> offsets(row_lengths.size());
    size_t total = 0;
    for (size_t i = 0; i < row_lengths.size(); ++i) {
      offsets[i] = total;
      total += static_cast<size_t>(row_lengths[i]);
    }
    return offsets;
  }

  std::unordered_map<uint32_t, std::unique_ptr<SlotStore>> slots_;
  size_t num_threads_ = 0;
  exec::static_thread_pool pool_;
  Initializer initializer_;
  size_t max_dim_ = 0;
  static constexpr size_t kParallelThreshold = 64;
};

} // namespace se

#endif // SPARSE_EMBEDDING_HOGWILD_EMBEDDING_STORE_HPP_
