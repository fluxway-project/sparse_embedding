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
#include <random>
#include <shared_mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <exec/static_thread_pool.hpp>
#include <stdexec/execution.hpp>

#include <ankerl/unordered_dense.h>
#include <tsl/robin_map.h>

#include "arena/transient_arena.hpp"
#include "common/embedding_error.hpp"
#include "common/numeric_types.hpp"
#include "interface/initializer_interface.hpp"
#include "interface/optimizer_interface.hpp"
#include "optimizer/optimizer_state.hpp"
#include "simd/simd_ops.hpp"

namespace se {

// =============================================================================
// EmbeddingArena: 64-byte cache-line aligned region allocator.
// Score stored separately to keep data() pointer SIMD-aligned.
// =============================================================================

template <typename ValT> class EmbeddingArena {
  static constexpr size_t kCacheLineSize = 64;
  static constexpr size_t kEntriesPerPage = 4096;

  struct AlignedDeleter {
    void operator()(std::byte *p) const { std::free(p); }
  };
  using AlignedPage = std::unique_ptr<std::byte[], AlignedDeleter>;

public:
  explicit EmbeddingArena(size_t dim) : dim_(dim) {
    data_stride_ =
        (dim * sizeof(ValT) + kCacheLineSize - 1) & ~(kCacheLineSize - 1);
  }

  EmbeddingArena(const EmbeddingArena &) = delete;
  EmbeddingArena &operator=(const EmbeddingArena &) = delete;
  EmbeddingArena(EmbeddingArena &&) = default;
  EmbeddingArena &operator=(EmbeddingArena &&) = default;

  size_t dim() const { return dim_; }

  uint32_t allocate() {
    uint32_t idx;
    if (!free_list_.empty()) {
      idx = free_list_.back();
      free_list_.pop_back();
    } else {
      idx = next_index_++;
      if (idx / kEntriesPerPage >= data_pages_.size()) {
        auto *dp = static_cast<std::byte *>(
            std::aligned_alloc(kCacheLineSize, kEntriesPerPage * data_stride_));
        if (!dp)
          throw std::bad_alloc();
        std::memset(dp, 0, kEntriesPerPage * data_stride_);
        data_pages_.emplace_back(dp);
        score_pages_.push_back(std::make_unique<uint64_t[]>(kEntriesPerPage));
      }
    }
    std::memset(data_raw(idx), 0, data_stride_);
    score(idx).store(0, std::memory_order_relaxed);
    return idx;
  }

  void deallocate(uint32_t index) { free_list_.push_back(index); }

  std::atomic_ref<uint64_t> score(uint32_t index) const {
    return std::atomic_ref<uint64_t>(
        score_pages_[index / kEntriesPerPage][index % kEntriesPerPage]);
  }

  ValT *data(uint32_t index) const {
    return reinterpret_cast<ValT *>(data_raw(index));
  }

  size_t allocated_count() const { return next_index_ - free_list_.size(); }

private:
  std::byte *data_raw(uint32_t index) const {
    return data_pages_[index / kEntriesPerPage].get() +
           (index % kEntriesPerPage) * data_stride_;
  }

  size_t dim_;
  size_t data_stride_;
  std::vector<AlignedPage> data_pages_;
  std::vector<std::unique_ptr<uint64_t[]>> score_pages_;
  std::vector<uint32_t> free_list_;
  uint32_t next_index_ = 0;
};

// =============================================================================
// ArenaHandle
// =============================================================================

template <typename ValT> struct ArenaHandle {
  uint32_t index = UINT32_MAX;

  ArenaHandle() = default;
  explicit ArenaHandle(uint32_t i) : index(i) {}

  bool valid() const { return index != UINT32_MAX; }
};

// =============================================================================
// HogwildEmbeddingStore
// =============================================================================

template <typename ValT> class HogwildEmbeddingStore {
public:
  using ValueType = ValT;
  using Initializer = pro::proxy<InitializerFacade<ValT>>;
  using OptimizerFactory =
      std::function<pro::proxy<OptimizerFacade<ValT>>(size_t)>;

  static constexpr size_t kNumShards = 256;

  struct Shard {
    std::shared_mutex mtx;
    tsl::robin_map<uint64_t, ArenaHandle<ValT>,
                   ankerl::unordered_dense::hash<uint64_t>>
        map;
    EmbeddingArena<ValT> arena;
    pro::proxy<OptimizerFacade<ValT>> optimizer;

    explicit Shard(size_t d, const OptimizerFactory &opt_factory)
        : arena(d), optimizer(opt_factory(d)) {}
  };

  struct SlotStore {
    size_t dim;
    std::vector<std::unique_ptr<Shard>> shards;
    std::unique_ptr<TransientArena<ValT>> transient;
    OptimizerFactory opt_factory;
    uint64_t global_step = 1;

    SlotStore(size_t d, size_t max_active_keys, uint32_t grad_accum_steps,
              const OptimizerFactory &opt_factory)
        : dim(d), opt_factory(opt_factory) {
      for (size_t i = 0; i < kNumShards; ++i)
        shards.push_back(std::make_unique<Shard>(d, opt_factory));
      transient = std::make_unique<TransientArena<ValT>>(d, max_active_keys,
                                                         grad_accum_steps);
    }

    Shard &shard_for_key(uint64_t key) {
      uint64_t hash = ankerl::unordered_dense::hash<uint64_t>{}(key);
      return *shards[hash % kNumShards];
    }
  };

  explicit HogwildEmbeddingStore(std::unordered_map<uint32_t, size_t> slot_dims,
                                 size_t num_threads = 0,
                                 Initializer initializer = {},
                                 uint32_t grad_accum_steps = 1,
                                 size_t max_active_keys = 1 << 20,
                                 OptimizerFactory opt_factory = nullptr)
      : pool_(static_cast<uint32_t>(
            num_threads > 0
                ? num_threads
                : std::max(1u, std::thread::hardware_concurrency()))),
        initializer_(std::move(initializer)) {
    if (slot_dims.empty()) {
      throw std::invalid_argument("slot_dims must not be empty");
    }
    for (auto &[slot_id, dim] : slot_dims) {
      if (dim == 0) {
        throw std::invalid_argument("dim must be > 0 for slot " +
                                    std::to_string(slot_id));
      }
      slots_.emplace(slot_id, std::make_unique<SlotStore>(dim, max_active_keys,
                                                          grad_accum_steps,
                                                          opt_factory));
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
  // Forward Pass: SIMD gather + software prefetch from Wait-Free map
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
        auto &shard = slot.shard_for_key(key);
        size_t dim = static_cast<size_t>(row_lengths[i]);
        ValT *out = out_flat_values.data() + offsets[i];

        if (i + 1 < end) {
          uint64_t next_key = static_cast<uint64_t>(keys[i + 1]);
          auto &next_shard =
              slots_.at(extract_slot(next_key))->shard_for_key(next_key);
          simd::prefetch_l2(&next_shard.map);
        }

        std::shared_lock lock(shard.mtx);
        auto it = shard.map.find(key);
        if (it != shard.map.end()) {
          const ValT *src = shard.arena.data(it->second.index);
          simd::prefetch_l1(src);
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

    auto do_range = [&](size_t begin, size_t end) {
      for (size_t i = begin; i < end; ++i) {
        uint64_t key = static_cast<uint64_t>(keys[i]);
        uint32_t slot_id = extract_slot(key);
        auto &slot = *slots_.at(slot_id);
        const ValT *grad_src = flat_grads.data() + offsets[i];
        uint32_t t_idx = slot.transient->get_or_allocate(key);
        slot.transient->accumulate(t_idx, grad_src);
      }
    };

    dispatch_parallel(n, do_range);

    for (auto &[_, slot] : slots_) {
      if (slot->transient->advance_batch()) {
        optimizer_step_for_slot(*slot);
      }
    }

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
      optimizer_step_for_slot(*slot);
    }
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
        auto &shard = slot.shard_for_key(key);
        const ValT *src = flat_values.data() + offsets[i];

        std::unique_lock lock(shard.mtx);
        auto it = shard.map.find(key);
        if (it == shard.map.end()) {
          uint32_t idx = shard.arena.allocate();
          shard.optimizer->allocate();
          std::copy_n(src, slot.dim, shard.arena.data(idx));
          shard.map.emplace(key, ArenaHandle<ValT>(idx));
        } else {
          std::copy_n(src, slot.dim, shard.arena.data(it->second.index));
        }
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
        auto &shard = slot.shard_for_key(key);
        size_t dim = slot.dim;
        ValT *out = out_flat_values.data() + offsets[i];

        std::unique_lock lock(shard.mtx);
        auto it = shard.map.find(key);
        bool inserted = false;
        if (it == shard.map.end()) {
          uint32_t idx = shard.arena.allocate();
          shard.optimizer->allocate();
          if (initializer_.has_value()) {
            initializer_->initialize(
                std::span<ValT>(shard.arena.data(idx), dim), key);
          } else {
            std::fill_n(shard.arena.data(idx), dim, ValT{0});
          }
          it = shard.map.emplace(key, ArenaHandle<ValT>(idx)).first;
          inserted = true;
        }

        std::copy_n(shard.arena.data(it->second.index), dim, out);
        out_inserted_mask[i] = inserted;
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
        auto &shard = slot.shard_for_key(key);

        std::shared_lock lock(shard.mtx);
        out_exists[i] = (shard.map.find(key) != shard.map.end());
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
        auto &shard = slot.shard_for_key(key);

        std::unique_lock lock(shard.mtx);
        auto it = shard.map.find(key);
        if (it != shard.map.end()) {
          uint32_t idx = it->second.index;
          shard.arena.deallocate(idx);
          shard.optimizer->deallocate(idx);
          shard.map.erase(it);
        }
      }
    };
    dispatch_parallel(n, do_range);
    return {};
  }

  std::error_code clear() {
    for (auto &[_, slot] : slots_) {
      for (auto &shard : slot->shards) {
        std::unique_lock lock(shard->mtx);
        shard->map.clear();
        shard->optimizer->clear();
        shard = std::make_unique<Shard>(slot->dim, slot->opt_factory);
      }
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
        auto &shard = slot.shard_for_key(key);

        {
          std::shared_lock lock(shard.mtx);
          auto it = shard.map.find(key);
          if (it != shard.map.end()) {
            shard.arena.score(it->second.index)
                .store(scores[i], std::memory_order_relaxed);
            continue;
          }
        }
        {
          std::unique_lock lock(shard.mtx);
          auto it = shard.map.find(key);
          if (it == shard.map.end()) {
            uint32_t idx = shard.arena.allocate();
            shard.optimizer->allocate();
            shard.arena.score(idx).store(scores[i], std::memory_order_relaxed);
            shard.map.emplace(key, ArenaHandle<ValT>(idx));
          } else {
            shard.arena.score(it->second.index)
                .store(scores[i], std::memory_order_relaxed);
          }
        }
      }
    };
    dispatch_parallel(n, do_range);
    return {};
  }

  size_t size() const {
    size_t s = 0;
    for (auto &[_, slot] : slots_) {
      for (const auto &shard : slot->shards) {
        std::shared_lock lock(shard->mtx);
        s += shard->map.size();
      }
    }
    return s;
  }

  size_t num_slots() const { return slots_.size(); }

private:
  void optimizer_step_for_slot(SlotStore &slot) {
    // =========================================================================
    slot.global_step++;

    slot.transient->for_each_accumulated([&](uint64_t feature_id,
                                             const ValT *accumulated_grad) {
      auto &shard = slot.shard_for_key(feature_id);
      auto it = shard.map.find(feature_id);
      if (it == shard.map.end())
        return;

      uint32_t arena_idx = it->second.index;
      ValT *param = shard.arena.data(arena_idx);

      shard.optimizer->step(arena_idx, std::span<ValT>(param, slot.dim),
                            std::span<const ValT>(accumulated_grad, slot.dim),
                            slot.transient->grad_accum_steps(),
                            slot.global_step);
    });

    slot.transient->clear();
  }

  template <typename Fn> void dispatch_parallel(size_t n, Fn &&fn) {
    if (n >= kParallelThreshold) {
      stdexec::sync_wait(stdexec::bulk_chunked(
          stdexec::schedule(pool_.get_scheduler()), stdexec::par, n,
          [&fn](size_t begin, size_t end) { fn(begin, end); }));
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
  exec::static_thread_pool pool_;
  Initializer initializer_;
  size_t max_dim_ = 0;
  static constexpr size_t kParallelThreshold = 64;
};

} // namespace se

#endif // SPARSE_EMBEDDING_HOGWILD_EMBEDDING_STORE_HPP_
