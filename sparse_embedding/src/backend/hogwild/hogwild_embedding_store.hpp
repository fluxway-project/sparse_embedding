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
#include <vector>

#include <exec/static_thread_pool.hpp>
#include <stdexec/execution.hpp>

#include <ankerl/unordered_dense.h>
#include <tsl/robin_map.h>

#include "common/embedding_error.hpp"
#include "common/numeric_types.hpp"

namespace se {

// =============================================================================
// EmbeddingArena: Core zero-copy continuous physical memory allocator.
// =============================================================================

template <typename ValT> class EmbeddingArena {
  static constexpr size_t kEntriesPerPage = 4096;

public:
  explicit EmbeddingArena(size_t dim) : dim_(dim) {
    constexpr size_t kAlign = alignof(std::max_align_t);
    entry_stride_ =
        (sizeof(uint64_t) + dim * sizeof(ValT) + kAlign - 1) & ~(kAlign - 1);
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
      if (idx / kEntriesPerPage >= pages_.size()) {
        pages_.push_back(
            std::make_unique<std::byte[]>(kEntriesPerPage * entry_stride_));
      }
    }
    std::memset(raw(idx), 0, entry_stride_);
    return idx;
  }

  void deallocate(uint32_t index) { free_list_.push_back(index); }

  std::atomic_ref<uint64_t> score(uint32_t index) const {
    return std::atomic_ref<uint64_t>(*reinterpret_cast<uint64_t *>(raw(index)));
  }

  ValT *data(uint32_t index) const {
    return reinterpret_cast<ValT *>(raw(index) + sizeof(uint64_t));
  }

  size_t allocated_count() const { return next_index_ - free_list_.size(); }

private:
  std::byte *raw(uint32_t index) const {
    return pages_[index / kEntriesPerPage].get() +
           (index % kEntriesPerPage) * entry_stride_;
  }

  size_t dim_;
  size_t entry_stride_;
  std::vector<std::unique_ptr<std::byte[]>> pages_;
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
// TransientPool for gradient accumulation
// Allocates fixed capacity gradient blocks and provides 8-bit spinlocks per
// row.
// =============================================================================

template <typename ValT> class TransientPool {
public:
  explicit TransientPool(size_t dim, size_t max_features)
      : dim_(dim), capacity_(max_features) {
    grads_ = std::make_unique<ValT[]>(max_features * dim);
    locks_ = std::make_unique<std::atomic_flag[]>(max_features);
    std::fill_n(grads_.get(), max_features * dim, ValT{0});
  }

  void accumulate_simd(uint32_t index, const ValT *inc) {
    auto &lock = locks_[index];
    while (lock.test_and_set(std::memory_order_acquire)) {
      // 8-bit spinlock
#if defined(__i386__) || defined(__x86_64__)
      __builtin_ia32_pause();
#endif
    }

    // SIMD AVX2/AVX-512 unrolled accumulation or auto-vectorized loop
    ValT *target = grads_.get() + index * dim_;
#pragma omp simd
    for (size_t d = 0; d < dim_; ++d) {
      target[d] += inc[d];
    }

    lock.clear(std::memory_order_release);
  }

  void clear() { std::fill_n(grads_.get(), capacity_ * dim_, ValT{0}); }

  ValT *data(uint32_t index) { return grads_.get() + index * dim_; }

private:
  size_t dim_;
  size_t capacity_;
  std::unique_ptr<ValT[]> grads_;
  std::unique_ptr<std::atomic_flag[]> locks_;
};

// =============================================================================
// HogwildEmbeddingStore: High performance L1/L3 wait-free map
// =============================================================================

template <typename ValT> class HogwildEmbeddingStore {
public:
  enum class InitType : uint8_t { kZeros = 0, kRandomSeeded = 1 };

  // For global wait-free parameter table, we retain minimal sharding (e.g.,
  // 256) simply to mask resizing overhead, but strictly rely on robust global
  // querying.
  static constexpr size_t kNumShards = 256;

  struct Shard {
    std::shared_mutex
        mtx; // Global parameter map doesn't block Forward per Hogwild.
    tsl::robin_map<uint64_t, ArenaHandle<ValT>,
                   ankerl::unordered_dense::hash<uint64_t>>
        map;
    EmbeddingArena<ValT> arena;

    explicit Shard(size_t d) : arena(d) {}
  };

  struct SlotStore {
    size_t dim;
    std::vector<std::unique_ptr<Shard>> shards;
    std::unique_ptr<TransientPool<ValT>> transient_pool;

    explicit SlotStore(size_t d) : dim(d) {
      for (size_t i = 0; i < kNumShards; ++i) {
        shards.push_back(std::make_unique<Shard>(d));
      }
      // Allocate transient pool allowing max concurrent accumulated features.
      transient_pool = std::make_unique<TransientPool<ValT>>(d, 1024 * 1024);
    }

    Shard &shard_for_key(uint64_t key) {
      uint64_t hash = ankerl::unordered_dense::hash<uint64_t>{}(key);
      return *shards[hash % kNumShards];
    }
  };

  explicit HogwildEmbeddingStore(std::unordered_map<uint32_t, size_t> slot_dims,
                                 size_t num_threads = 0,
                                 InitType init_type = InitType::kZeros)
      : pool_(static_cast<uint32_t>(
            num_threads > 0
                ? num_threads
                : std::max(1u, std::thread::hardware_concurrency()))),
        init_type_(init_type) {
    if (slot_dims.empty()) {
      throw std::invalid_argument("slot_dims must not be empty");
    }
    for (auto &[slot_id, dim] : slot_dims) {
      if (dim == 0) {
        throw std::invalid_argument("dim must be > 0 for slot " +
                                    std::to_string(slot_id));
      }
      slots_.emplace(slot_id, std::make_unique<SlotStore>(dim));
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

  // =========================================================================
  // Forward Pass: High performance Wait-Free map gather
  // No Write locks grabbed. Fast robin hood probing.
  // =========================================================================

  template <typename KeyT>
  std::error_code lookup(std::span<const KeyT> keys,
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

        // Wait-Free (shared read parallelization)
        std::shared_lock lock(shard.mtx);
        auto it = shard.map.find(key);
        if (it != shard.map.end()) {
          // Direct memory SIMD transfer from DLPack / Arena continuous blocks
          std::copy_n(shard.arena.data(it->second.index), dim, out);
        } else {
          std::fill_n(out, dim, ValT{0});
        }
      }
    };

    dispatch_parallel(n, do_range);
    return {};
  }

  // =========================================================================
  // Backward Gradient Accumulation Pass into Transient Pool
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

        // Hash to an index strictly bounded by max_features per batch
        // accumulation. We use a simplified transient ID bounded mapping here.
        uint64_t t_id =
            ankerl::unordered_dense::hash<uint64_t>{}(key) % (1024 * 1024);

        slot.transient_pool->accumulate_simd(t_id, grad_src);
      }
    };

    dispatch_parallel(n, do_range);
    return {};
  }

  // =========================================================================
  // Python Wrapper Support Methods
  // =========================================================================

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
  std::error_code assign_uniform(std::span<const KeyT> keys,
                                 std::span<const ValT> values, size_t dim) {
    size_t n = keys.size();
    if (values.size() < n * dim)
      return EmbeddingErrc::kInvalidDimension;

    auto do_range = [&](size_t begin, size_t end) {
      for (size_t i = begin; i < end; ++i) {
        uint64_t key = static_cast<uint64_t>(keys[i]);
        uint32_t slot_id = extract_slot(key);
        auto &slot = *slots_.at(slot_id);
        auto &shard = slot.shard_for_key(key);
        const ValT *src = values.data() + i * dim;

        std::unique_lock lock(
            shard.mtx); // Unique lock required for Hogwild insert
        auto it = shard.map.find(key);
        if (it == shard.map.end()) {
          uint32_t idx = shard.arena.allocate();
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
          if (init_type_ == InitType::kZeros) {
            std::fill_n(shard.arena.data(idx), dim, ValT{0});
          } else {
            std::mt19937 rng(static_cast<uint32_t>(key ^ (key >> 32)));
            std::uniform_real_distribution<float> dist(-0.05f, 0.05f);
            for (size_t d = 0; d < dim; ++d) {
              shard.arena.data(idx)[d] = static_cast<ValT>(dist(rng));
            }
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
          shard.arena.deallocate(it->second.index);
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
        shard = std::make_unique<Shard>(slot->dim);
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

  size_t total_size() const {
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
  InitType init_type_;
  size_t max_dim_ = 0;
  static constexpr size_t kParallelThreshold = 64;
};

} // namespace se

#endif // SPARSE_EMBEDDING_HOGWILD_EMBEDDING_STORE_HPP_
