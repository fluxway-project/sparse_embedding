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
#include <array>
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
#include "sparse_embedding/src/common/function.hpp"
#include "sparse_embedding/src/common/slot_key_codec.hpp"
#include "sparse_embedding/src/eviction/score_policy.hpp"
#include "sparse_embedding/src/interface/initializer_interface.hpp"
#include "sparse_embedding/src/interface/optimizer_interface.hpp"
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
  explicit EmbeddingArena(size_t dim, bool enable_score = true)
      : dim_(dim), enable_score_(enable_score),
        pages_(std::make_unique<PageTable>()) {
    data_stride_ =
        (dim * sizeof(ValT) + kCacheLineSize - 1) & ~(kCacheLineSize - 1);
  }

  ~EmbeddingArena() {
    for (size_t i = 0; i < kMaxPages; ++i) {
      auto *dp = pages_->data[i].load(std::memory_order_relaxed);
      if (dp)
        std::free(dp);
      if (enable_score_) {
        auto *sp = pages_->score[i].load(std::memory_order_relaxed);
        delete[] sp;
      }
    }
  }

  EmbeddingArena(const EmbeddingArena &) = delete;
  EmbeddingArena &operator=(const EmbeddingArena &) = delete;
  EmbeddingArena(EmbeddingArena &&) = delete;
  EmbeddingArena &operator=(EmbeddingArena &&) = delete;

  size_t dim() const { return dim_; }
  bool score_enabled() const { return enable_score_; }

  uint32_t allocate() {
    uint32_t idx = next_index_.fetch_add(1, std::memory_order_relaxed);
    ensure_capacity(idx);
    if (enable_score_)
      score_raw(idx) = 0;
    return idx;
  }

  uint64_t &score_raw(uint32_t index) const {
    return pages_->score[index / kEntriesPerPage].load(
        std::memory_order_acquire)[index % kEntriesPerPage];
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
      if (enable_score_) {
        auto *sp =
            pages_->score[i].exchange(nullptr, std::memory_order_relaxed);
        delete[] sp;
      }
    }
    num_pages_.store(0, std::memory_order_relaxed);
    next_index_.store(0, std::memory_order_relaxed);
  }

  size_t data_stride() const { return data_stride_; }

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
      pages_->data[pg].store(dp, std::memory_order_release);
      if (enable_score_) {
        auto *sp = new uint64_t[kEntriesPerPage]();
        pages_->score[pg].store(sp, std::memory_order_release);
      }
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
  bool enable_score_;
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

template <typename ValT, typename ScorePolicyT = NoScorePolicy>
class HogwildEmbeddingStore {
public:
  using ValueType = ValT;
  using Initializer = pro::proxy<InitializerFacade<ValT>>;
  using OptimizerFactory =
      se::Function<pro::proxy<OptimizerFacade<ValT>>(size_t)>;

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
    uint64_t global_step = 0;

    SlotStore(size_t d, size_t max_active_keys, OptimizerFactory opt_factory,
              size_t num_threads, bool enable_score = true)
        : dim(d), entries(false),
          key_map(std::max(size_t(1024), max_active_keys * 2), &entries, false),
          arena(d, enable_score),
          optimizer(opt_factory.has_value() ? (*opt_factory)(d) : nullptr),
          opt_factory(opt_factory) {
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
                                 OptimizerFactory opt_factory = nullptr,
                                 ScorePolicyT score_policy = {},
                                 uint8_t slot_bits = 16)
      : num_threads_(num_threads > 0
                         ? num_threads
                         : std::max(1u, std::thread::hardware_concurrency())),
        pool_(static_cast<uint32_t>(num_threads_)),
        initializer_(std::move(initializer)),
        score_policy_(std::move(score_policy)), codec_(slot_bits) {
    uint32_t p2 = 1;
    while (p2 < num_threads_) {
      p2 <<= 1;
    }
    num_threads_ = p2;

    bool enable_score = ScorePolicyT::kHasScore;

    if (slot_dims.empty()) {
      throw std::invalid_argument("slot_dims must not be empty");
    }
    slots_.fill(nullptr);
    slot_index_.fill(0);
    uint16_t dense_idx = 0;
    for (auto &[slot_id, dim] : slot_dims) {
      if (dim == 0) {
        throw std::invalid_argument("dim must be > 0 for slot " +
                                    std::to_string(slot_id));
      }
      if (slot_id >= kMaxSlots) {
        throw std::invalid_argument("slot_id must be < " +
                                    std::to_string(kMaxSlots));
      }
      slot_storage_.push_back(std::make_unique<SlotStore>(
          dim, max_active_keys, opt_factory, num_threads_, enable_score));
      slots_[slot_id] = slot_storage_.back().get();
      slot_ids_.push_back(slot_id);
      slot_index_[slot_id] = dense_idx++;
      if (dim > max_dim_)
        max_dim_ = dim;
    }
  }

  uint32_t extract_slot(uint64_t key) const { return codec_.decode_slot(key); }

  const SlotKeyCodec &codec() const { return codec_; }

  size_t max_dim() const { return max_dim_; }

  size_t dim_for_slot(uint32_t slot_id) const {
    if (slot_id >= kMaxSlots || slots_[slot_id] == nullptr)
      throw std::invalid_argument("unknown slot " + std::to_string(slot_id));
    return slots_[slot_id]->dim;
  }

  size_t dim_for_key(uint64_t key) const {
    uint32_t slot_id = extract_slot(key);
    return slots_[slot_id]->dim;
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
      epoch::with_epoch([&] {
        for (size_t i = begin; i < end; ++i) {
          uint64_t key = static_cast<uint64_t>(keys[i]);
          uint32_t slot_id = extract_slot(key);
          auto &slot = *slots_[slot_id];
          size_t dim = static_cast<size_t>(row_lengths[i]);
          ValT *out = out_flat_values.data() + offsets[i];

          auto found =
              slot.key_map.Find(Key::Entry::make_key(key), [](const auto &e) {
                return e.get_entry().second;
              });
          if (found.has_value()) {
            const ValT *src = slot.arena.data(*found);
            if constexpr (simd::is_simd_supported_v<ValT>) {
              simd::simd_copy(src, out, dim);
            } else {
              std::copy_n(src, dim, out);
            }
            if constexpr (ScorePolicyT::kHasScore) {
              score_policy_.on_access(&slot.arena.score_raw(*found));
            }
          } else {
            std::fill_n(out, dim, ValT{0});
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

  template <typename KeyT>
  std::error_code accumulate_gradients(std::span<const KeyT> keys,
                                       std::span<const ValT> flat_grads,
                                       std::span<const int64_t> row_lengths) {
    size_t n = keys.size();
    auto offsets = compute_offsets(row_lengths);

    size_t num_shards = num_threads_;

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
    dispatch_parallel(num_shards, do_count, 1);

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
    thread_local std::vector<uint32_t> shuffled_indices_buf;
    if (shuffled_indices_buf.capacity() > n * 2) {
      shuffled_indices_buf.shrink_to_fit();
    }
    shuffled_indices_buf.resize(n);
    std::span<uint32_t> shuffled_indices(shuffled_indices_buf.data(), n);
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
          _mm512_store_si512(extracted, v_shards);

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
    dispatch_parallel(num_shards, do_scatter, 1);

    auto do_shard = [&](size_t shard_begin, size_t shard_end) {
      for (size_t shard_id = shard_begin; shard_id < shard_end; ++shard_id) {
        size_t start = shard_offsets[shard_id];
        size_t end = shard_offsets[shard_id + 1];
        for (size_t idx = start; idx < end; ++idx) {
          uint32_t i = shuffled_indices[idx];
          uint64_t key = static_cast<uint64_t>(keys[i]);
          uint32_t slot_id = extract_slot(key);
          auto &slot = *slots_[slot_id];
          const ValT *grad_src = flat_grads.data() + offsets[i];
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
    for (uint32_t sid_ : slot_ids_) {
      auto &slot = slot_storage_[slot_index_[sid_]];
      slot->global_step++;
    }

    size_t num_shards = num_threads_;

    auto do_shard = [&](size_t shard_begin, size_t shard_end) {
      epoch::with_epoch([&] {
        for (size_t shard_id = shard_begin; shard_id < shard_end; ++shard_id) {
          for (uint32_t sid_ : slot_ids_) {
            auto *slot = slots_[sid_];
            auto &shard = slot->transient_shards[shard_id];
            shard->for_each_accumulated(
                [&](uint64_t feature_id, const ValT *accumulated_grad) {
                  auto found = slot->key_map.Find(
                      Key::Entry::make_key(feature_id),
                      [](const auto &e) { return e.get_entry().second; });
                  if (!found.has_value())
                    return;

                  uint32_t arena_idx = *found;
                  ValT *param = slot->arena.data(arena_idx);

                  if (slot->optimizer.has_value()) {
                    slot->optimizer->step(
                        arena_idx, std::span<ValT>(param, slot->dim),
                        std::span<const ValT>(accumulated_grad, slot->dim),
                        slot->global_step);
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
      std::vector<std::vector<uint32_t>> local_free_slots(slot_ids_.size());
      epoch::with_epoch([&] {
        for (size_t i = begin; i < end; ++i) {
          uint64_t key = static_cast<uint64_t>(keys[i]);
          uint32_t slot_id = extract_slot(key);
          auto &slot = *slots_[slot_id];
          auto &free_slots = local_free_slots[slot_index_[slot_id]];
          const ValT *src = flat_values.data() + offsets[i];

          std::optional<uint32_t> preallocated_idx = std::nullopt;
          bool was_inserted = false;

          auto upsert_fn = [&](std::optional<uint32_t> existing) -> uint32_t {
            was_inserted = false;
            if (existing.has_value()) {
              return *existing;
            }
            was_inserted = true;
            uint32_t idx;
            if (preallocated_idx.has_value()) {
              idx = *preallocated_idx;
            } else if (!free_slots.empty()) {
              idx = free_slots.back();
              free_slots.pop_back();
              preallocated_idx = idx;
              std::copy_n(src, slot.dim, slot.arena.data(idx));
            } else {
              idx = slot.arena.allocate();
              if (slot.optimizer.has_value())
                slot.optimizer->allocate();
              preallocated_idx = idx;
              std::copy_n(src, slot.dim, slot.arena.data(idx));
            }
            return idx;
          };

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
          auto prev = slot.key_map.Upsert(Key::Entry::make_key(key), constr, g);

          if (preallocated_idx.has_value() && !was_inserted) {
            free_slots.push_back(*preallocated_idx);
          }
          if (!was_inserted) {
            std::copy_n(src, slot.dim, slot.arena.data(prev->second));
          }
        }
      });
    };
    dispatch_parallel(n, do_range);
    return {};
  }

  template <typename KeyT>
  std::error_code lookup_or_insert(std::span<const KeyT> keys,
                                   std::span<ValT> out_flat_values,
                                   std::span<const int64_t> row_lengths,
                                   std::span<uint8_t> out_inserted_mask) {
    size_t n = keys.size();
    auto offsets = compute_offsets(row_lengths);

    auto do_range = [&](size_t begin, size_t end) {
      std::vector<std::vector<uint32_t>> local_free_slots(slot_ids_.size());
      epoch::with_epoch([&] {
        for (size_t i = begin; i < end; ++i) {
          uint64_t key = static_cast<uint64_t>(keys[i]);
          uint32_t slot_id = extract_slot(key);
          auto &slot = *slots_[slot_id];
          auto &free_slots = local_free_slots[slot_index_[slot_id]];
          size_t dim = slot.dim;
          ValT *out = out_flat_values.data() + offsets[i];

          std::optional<uint32_t> preallocated_idx = std::nullopt;
          bool was_inserted = false;

          auto upsert_fn = [&](std::optional<uint32_t> existing) -> uint32_t {
            was_inserted = false;
            if (existing.has_value()) {
              return *existing;
            }
            was_inserted = true;
            uint32_t idx;
            if (preallocated_idx.has_value()) {
              idx = *preallocated_idx;
            } else if (!free_slots.empty()) {
              idx = free_slots.back();
              free_slots.pop_back();
              preallocated_idx = idx;
              if (initializer_.has_value()) {
                initializer_->initialize(
                    std::span<ValT>(slot.arena.data(idx), dim), key);
              } else {
                std::fill_n(slot.arena.data(idx), dim, ValT{0});
              }
            } else {
              idx = slot.arena.allocate();
              if (slot.optimizer.has_value())
                slot.optimizer->allocate();
              preallocated_idx = idx;
              if (initializer_.has_value()) {
                initializer_->initialize(
                    std::span<ValT>(slot.arena.data(idx), dim), key);
              } else {
                std::fill_n(slot.arena.data(idx), dim, ValT{0});
              }
            }
            return idx;
          };

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
          auto prev = slot.key_map.Upsert(Key::Entry::make_key(key), constr, g);

          if (preallocated_idx.has_value() && !was_inserted) {
            free_slots.push_back(*preallocated_idx);
          }
          uint32_t arena_idx = was_inserted ? *preallocated_idx : prev->second;
          out_inserted_mask[i] = was_inserted;
          std::copy_n(slot.arena.data(arena_idx), dim, out);
        }
      });
    };
    dispatch_parallel(n, do_range);
    return {};
  }

  template <typename KeyT>
  std::error_code contains(std::span<const KeyT> keys,
                           std::span<uint8_t> out_exists) {
    size_t n = keys.size();
    auto do_range = [&](size_t begin, size_t end) {
      epoch::with_epoch([&] {
        for (size_t i = begin; i < end; ++i) {
          uint64_t key = static_cast<uint64_t>(keys[i]);
          uint32_t slot_id = extract_slot(key);
          auto &slot = *slots_[slot_id];
          out_exists[i] = slot.key_map
                              .Find(Key::Entry::make_key(key),
                                    [](const auto &e) { return true; })
                              .has_value();
        }
      });
    };
    dispatch_parallel(n, do_range);
    return {};
  }

  template <typename KeyT> std::error_code erase(std::span<const KeyT> keys) {
    size_t n = keys.size();
    auto do_range = [&](size_t begin, size_t end) {
      epoch::with_epoch([&] {
        for (size_t i = begin; i < end; ++i) {
          uint64_t key = static_cast<uint64_t>(keys[i]);
          uint32_t slot_id = extract_slot(key);
          auto &slot = *slots_[slot_id];
          slot.key_map.Remove(Key::Entry::make_key(key),
                              [](const auto &e) { return true; });
        }
      });
    };
    dispatch_parallel(n, do_range);
    return {};
  }

  std::error_code clear() {
    for (uint32_t sid : slot_ids_) {
      auto *slot = slots_[sid];
      slot->key_map.clear();
      if (slot->optimizer.has_value()) {
        slot->optimizer->clear();
      }
      slot->arena.reset();
      slot->optimizer = slot->opt_factory.has_value()
                            ? (*slot->opt_factory)(slot->dim)
                            : nullptr;
    }
    return {};
  }

  template <typename KeyT>
  std::error_code assign_scores(std::span<const KeyT> keys,
                                std::span<const uint64_t> scores) {
    if constexpr (!ScorePolicyT::kHasScore) {
      throw std::runtime_error("assign_scores requires a score-enabled policy");
    }
    size_t n = keys.size();
    auto do_range = [&](size_t begin, size_t end) {
      std::vector<std::vector<uint32_t>> local_free_slots(slot_ids_.size());
      for (size_t i = begin; i < end; ++i) {
        uint64_t key = static_cast<uint64_t>(keys[i]);
        uint32_t slot_id = extract_slot(key);
        auto &slot = *slots_[slot_id];
        auto &free_slots = local_free_slots[slot_index_[slot_id]];

        epoch::with_epoch([&] {
          std::optional<uint32_t> preallocated_idx = std::nullopt;
          bool was_inserted = false;

          auto g = [&](std::optional<typename Key::Entry> opt_e)
              -> std::pair<uint64_t, uint32_t> {
            was_inserted = false;
            uint32_t idx;
            if (opt_e.has_value()) {
              idx = opt_e->get_entry().second;
            } else {
              was_inserted = true;
              if (preallocated_idx.has_value()) {
                idx = *preallocated_idx;
              } else if (!free_slots.empty()) {
                idx = free_slots.back();
                free_slots.pop_back();
                preallocated_idx = idx;
                if (initializer_.has_value()) {
                  initializer_->initialize(
                      std::span<ValT>(slot.arena.data(idx), slot.dim), key);
                } else {
                  std::fill_n(slot.arena.data(idx), slot.dim, ValT{0});
                }
              } else {
                idx = slot.arena.allocate();
                if (slot.optimizer.has_value())
                  slot.optimizer->allocate();
                preallocated_idx = idx;
                if (initializer_.has_value()) {
                  initializer_->initialize(
                      std::span<ValT>(slot.arena.data(idx), slot.dim), key);
                } else {
                  std::fill_n(slot.arena.data(idx), slot.dim, ValT{0});
                }
              }
            }
            std::atomic_ref<uint64_t>(slot.arena.score_raw(idx))
                .store(scores[i], std::memory_order_relaxed);
            return std::make_pair(key, idx);
          };
          auto constr = [&](const auto &e) {
            return e.has_value()
                       ? slot.entries.make_entry(Key::Entry::make_key(key),
                                                 g(e))
                       : slot.entries.make_entry(Key::Entry::make_key(key),
                                                 g(std::nullopt));
          };
          slot.key_map.Upsert(Key::Entry::make_key(key), constr, g);

          if (preallocated_idx.has_value() && !was_inserted) {
            free_slots.push_back(*preallocated_idx);
          }
        });
      }
    };
    dispatch_parallel(n, do_range);
    return {};
  }

  size_t size() const {
    size_t s = 0;
    for (uint32_t sid : slot_ids_) {
      s += slots_[sid]->key_map.size();
    }
    return s;
  }

  size_t num_slots() const { return slot_ids_.size(); }

private:
  template <typename Fn>
  void dispatch_parallel(size_t n, Fn &&fn,
                         size_t threshold = kParallelThreshold) {
    if (n >= threshold) {
      size_t num_threads = num_threads_;
      size_t chunk_size = (n + num_threads - 1) / num_threads;
      stdexec::sync_wait(stdexec::schedule(pool_.get_scheduler()) |
                         stdexec::bulk(stdexec::par_unseq, num_threads,
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
    size_t total = 0;
    for (size_t i = 0; i < row_lengths.size(); ++i) {
      offsets[i] = total;
      total += static_cast<size_t>(row_lengths[i]);
    }
    return offsets;
  }

  static constexpr size_t kMaxSlots = 65536;
  std::array<SlotStore *, kMaxSlots> slots_{};
  std::vector<std::unique_ptr<SlotStore>> slot_storage_;
  std::vector<uint32_t> slot_ids_;
  std::array<uint16_t, kMaxSlots> slot_index_{};
  size_t num_threads_ = 0;
  exec::static_thread_pool pool_;
  Initializer initializer_;
  ScorePolicyT score_policy_;
  size_t max_dim_ = 0;
  SlotKeyCodec codec_;
  static constexpr size_t kParallelThreshold = 64;

public:
  bool has_score_policy() const { return ScorePolicyT::kHasScore; }

  std::unordered_map<uint32_t, size_t> slot_dims() const {
    std::unordered_map<uint32_t, size_t> result;
    for (uint32_t id : slot_ids_)
      result[id] = slots_[id]->dim;
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

  template <typename KeyT>
  std::error_code
  export_batch(se::Function<bool(std::span<const KeyT>, std::span<const ValT>,
                                 std::span<const uint64_t>)>
                   visitor,
               size_t batch_size) {
    bool has_score = ScorePolicyT::kHasScore;

    for (uint32_t slot_id : slot_ids_) {
      auto *slot = slots_[slot_id];
      size_t dim = slot->dim;
      int num_workers = static_cast<int>(parlay::num_workers());

      struct LocalTinyBuffer {
        std::vector<KeyT> keys;
        std::vector<ValT> vals;
        std::vector<uint64_t> scores;
      };

      struct GlobalBuffer {
        std::vector<KeyT> keys;
        std::vector<ValT> vals;
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
        bool cont = (*visitor)(std::span<const KeyT>(buf.keys),
                               std::span<const ValT>(buf.vals),
                               std::span<const uint64_t>(buf.scores));
        return cont;
      };

      slot->key_map.for_each([&](const auto &entry) {
        if (stopped.load(std::memory_order_relaxed))
          return;

        int w = parlay::worker_id();
        auto [key, idx] = entry.get_entry();
        auto &lb = local_buffers[w];

        lb.keys.push_back(static_cast<KeyT>(key));
        const ValT *src = slot->arena.data(idx);
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

  // export_batch_with_opt — same as export_batch but additionally collects
  // optimizer state (e.g. Adam m/v moments) alongside parameters.
  // visitor signature: (keys, vals, opt_states) -> bool
  // opt_states is a flat float32 buffer of length n * opt_dim;
  // opt_dim == 0 when the optimizer has no state (e.g. plain SGD).
  template <typename VisitorFn>
  void export_batch_with_opt(VisitorFn &&visitor, size_t batch_size) {
    for (uint32_t slot_id : slot_ids_) {
      auto *slot = slots_[slot_id];
      size_t emb_dim = slot->dim;
      size_t opt_dim =
          slot->optimizer.has_value() ? slot->optimizer->state_dim() : 0;

      std::vector<uint64_t> key_buf;
      std::vector<ValT> val_buf;
      std::vector<float> opt_buf;
      key_buf.reserve(batch_size);
      val_buf.reserve(batch_size * emb_dim);
      opt_buf.reserve(batch_size * opt_dim);

      auto flush = [&]() -> bool {
        if (key_buf.empty())
          return true;
        bool cont = visitor(std::span<const uint64_t>(key_buf),
                            std::span<const ValT>(val_buf),
                            std::span<const float>(opt_buf));
        key_buf.clear();
        val_buf.clear();
        opt_buf.clear();
        return cont;
      };

      bool stopped = false;
      slot->key_map.for_each([&](const auto &entry) {
        if (stopped)
          return;
        auto [key, idx] = entry.get_entry();
        key_buf.push_back(key);
        const ValT *src = slot->arena.data(idx);
        val_buf.insert(val_buf.end(), src, src + emb_dim);
        if (opt_dim > 0 && slot->optimizer.has_value()) {
          const float *os = slot->optimizer->state_data(idx);
          opt_buf.insert(opt_buf.end(), os, os + opt_dim);
        }
        if (key_buf.size() >= batch_size) {
          if (!flush())
            stopped = true;
        }
      });

      if (!stopped)
        flush();
    }
  }

  // export_batch_with_opt_owned — ownership-transferring variant.
  // Visitor receives std::vector&& instead of std::span, enabling the
  // consumer to take ownership of the heap buffers (e.g. for zero-copy
  // Arrow C Data Interface construction).
  template <typename VisitorFn>
  void export_batch_with_opt_owned(VisitorFn &&visitor, size_t batch_size) {
    for (uint32_t slot_id : slot_ids_) {
      auto *slot = slots_[slot_id];
      size_t emb_dim = slot->dim;
      size_t opt_dim =
          slot->optimizer.has_value() ? slot->optimizer->state_dim() : 0;

      std::vector<uint64_t> key_buf;
      std::vector<ValT> val_buf;
      std::vector<float> opt_buf;
      key_buf.reserve(batch_size);
      val_buf.reserve(batch_size * emb_dim);
      opt_buf.reserve(batch_size * opt_dim);

      auto flush = [&]() -> bool {
        if (key_buf.empty())
          return true;
        bool cont =
            visitor(std::move(key_buf), std::move(val_buf), std::move(opt_buf));
        key_buf.clear();
        val_buf.clear();
        opt_buf.clear();
        key_buf.reserve(batch_size);
        val_buf.reserve(batch_size * emb_dim);
        opt_buf.reserve(batch_size * opt_dim);
        return cont;
      };

      bool stopped = false;
      slot->key_map.for_each([&](const auto &entry) {
        if (stopped)
          return;
        auto [key, idx] = entry.get_entry();
        key_buf.push_back(key);
        const ValT *src = slot->arena.data(idx);
        val_buf.insert(val_buf.end(), src, src + emb_dim);
        if (opt_dim > 0 && slot->optimizer.has_value()) {
          const float *os = slot->optimizer->state_data(idx);
          opt_buf.insert(opt_buf.end(), os, os + opt_dim);
        }
        if (key_buf.size() >= batch_size) {
          if (!flush())
            stopped = true;
        }
      });

      if (!stopped)
        flush();
    }
  }

  // assign_opt_states — restore optimizer state for a batch of existing keys.
  // Keys must already exist in the store (i.e. assign() called first).
  // opt_states is a flat float32 buffer of length n * opt_dim.
  void assign_opt_states(std::span<const uint64_t> keys,
                         std::span<const float> opt_states) {
    size_t n = keys.size();
    if (n == 0)
      return;

    for (uint32_t sid : slot_ids_) {
      auto *slot = slots_[sid];
      if (!slot->optimizer.has_value())
        continue;
      size_t opt_dim = slot->optimizer->state_dim();
      if (opt_dim == 0)
        continue;

      epoch::with_epoch([&] {
        for (size_t i = 0; i < n; ++i) {
          uint64_t key = keys[i];
          auto found =
              slot->key_map.Find(Key::Entry::make_key(key), [](const auto &e) {
                return e.get_entry().second;
              });
          if (!found.has_value())
            continue;
          uint32_t idx = *found;
          slot->optimizer->set_state(
              idx,
              std::span<const float>(opt_states.data() + i * opt_dim, opt_dim));
        }
      });
    }
  }

  // export_opt_state_batch — type-erased facade-compatible wrapper.
  // Visitor receives (keys, flat_f32_vals, opt_states) for checkpointing.
  // ValT values are widened to float32 to provide a uniform interface
  // regardless of the store's native precision.
  void export_opt_state_batch(
      se::Function<bool(std::span<const uint64_t>, std::span<const float>,
                        std::span<const float>)>
          visitor,
      size_t batch_size) {
    export_batch_with_opt(
        [&](std::span<const uint64_t> keys, std::span<const ValT> vals,
            std::span<const float> opt_states) -> bool {
          thread_local std::vector<float> val_f32_buf;
          val_f32_buf.resize(vals.size());
          for (size_t i = 0; i < vals.size(); ++i)
            val_f32_buf[i] = static_cast<float>(vals[i]);
          return (*visitor)(keys, std::span<const float>(val_f32_buf),
                            opt_states);
        },
        batch_size);
  }

  // export_opt_state_batch_owned — ownership-transferring facade wrapper.
  // Same as export_opt_state_batch but visitor receives std::vector&&
  // for zero-copy Arrow C Data Interface construction.
  void export_opt_state_batch_owned(
      se::Function<bool(std::vector<uint64_t> &&, std::vector<float> &&,
                        std::vector<float> &&)>
          visitor,
      size_t batch_size) {
    export_batch_with_opt_owned(
        [&](std::vector<uint64_t> &&keys, std::vector<ValT> &&vals,
            std::vector<float> &&opt_states) -> bool {
          std::vector<float> val_f32(vals.size());
          for (size_t i = 0; i < vals.size(); ++i)
            val_f32[i] = static_cast<float>(vals[i]);
          return (*visitor)(std::move(keys), std::move(val_f32),
                            std::move(opt_states));
        },
        batch_size);
  }

  // =========================================================================
  // evict_by_score — GC operation, decoupled from insertion.
  //
  // Evicts entries with score <= max_score, up to max_evict_count entries.
  // Safe to call from a background thread. Operates per-slot.
  // =========================================================================

  template <typename KeyT>
  std::error_code evict_by_score(uint64_t max_score, size_t max_evict_count,
                                 std::span<KeyT> out_evicted_keys,
                                 std::span<ValT> out_evicted_values,
                                 size_t &out_evicted_count) {
    if constexpr (!ScorePolicyT::kHasScore)
      throw std::runtime_error(
          "evict_by_score requires a score-enabled policy");

    out_evicted_count = 0;

    for (uint32_t slot_id : slot_ids_) {
      auto *slot = slots_[slot_id];
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

      slot->key_map.for_each([&](const auto &entry) {
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
      std::sort(global_candidates.begin(), global_candidates.end());

      epoch::with_epoch([&] {
        for (auto &[score, key] : global_candidates) {
          if (out_evicted_count >= max_evict_count)
            break;

          auto found =
              slot->key_map.Find(Key::Entry::make_key(key), [](const auto &e) {
                return e.get_entry().second;
              });
          if (!found.has_value())
            continue;

          uint32_t idx = *found;
          if (out_evicted_count < out_evicted_keys.size()) {
            out_evicted_keys[out_evicted_count] = static_cast<KeyT>(key);
            std::copy_n(slot->arena.data(idx), dim,
                        out_evicted_values.data() + out_evicted_count * dim);
          }
          slot->key_map.Remove(Key::Entry::make_key(key),
                               [](const auto &) { return true; });
          ++out_evicted_count;
        }
      });
    }
    return {};
  }
};

} // namespace se

#endif // SPARSE_EMBEDDING_HOGWILD_EMBEDDING_STORE_HPP_
