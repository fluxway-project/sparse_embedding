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

#ifndef SPARSE_EMBEDDING_TRANSIENT_ARENA_HPP_
#define SPARSE_EMBEDDING_TRANSIENT_ARENA_HPP_

#include <ankerl/unordered_dense.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "sparse_embedding/src/memory/seqlock.hpp"
#include "sparse_embedding/src/simd/memory_ops.hpp"
#include "sparse_embedding/src/simd/prefetch.hpp"
#include "sparse_embedding/src/storage/page_storage.hpp"

namespace se {

#include <bit>

template <typename ValT> class TransientArena {
  static constexpr size_t kCacheLineSize = 64;

public:
  class AccumulatedBatchView {
  public:
    size_t size() const noexcept { return feature_ids->size(); }
    uint64_t key(size_t position) const noexcept {
      return (*feature_ids)[position];
    }
    uint64_t hash(size_t position) const noexcept {
      return (*feature_hashes)[position];
    }
    const ValT *gradient(size_t position) const noexcept {
      return arena->grad_data(static_cast<uint32_t>(position));
    }
    const std::vector<uint64_t> &keys() const noexcept { return *feature_ids; }
    const std::vector<uint64_t> &hashes() const noexcept {
      return *feature_hashes;
    }
    bool contiguous() const noexcept { return true; }

  private:
    friend class TransientArena;
    const TransientArena *arena = nullptr;
    const std::vector<uint64_t> *feature_ids = nullptr;
    const std::vector<uint64_t> *feature_hashes = nullptr;
  };

  TransientArena(size_t dim, PageStorage storage,
                 bool page_eviction_enabled = false, size_t chunk_size = 8192)
      : dim_(dim), storage_(std::move(storage)),
        page_eviction_enabled_(page_eviction_enabled), chunk_size_(chunk_size),
        chunk_shift_(static_cast<unsigned>(std::countr_zero(chunk_size))),
        chunk_mask_(chunk_size - 1) {
    if (chunk_size_ == 0 || !std::has_single_bit(chunk_size_))
      throw std::invalid_argument(
          "transient arena chunk size must be a power of two");
    grad_stride_ =
        (dim * sizeof(ValT) + kCacheLineSize - 1) & ~(kCacheLineSize - 1);
    id_map_.reserve(chunk_size_);
  }

  ~TransientArena() = default;

  TransientArena(const TransientArena &) = delete;
  TransientArena &operator=(const TransientArena &) = delete;

  uint32_t get_or_allocate(uint64_t feature_id) {
    {
      se::memory::PackedReadGuard metadata_lock(metadata_state_,
                                                metadata_readers_);
      const auto existing = id_map_.find(feature_id);
      if (existing != id_map_.end())
        return existing->second;
    }
    se::memory::PackedWriteGuard metadata_lock(metadata_state_,
                                               metadata_readers_);
    return get_or_allocate_impl(feature_id, 0, false);
  }

  uint32_t get_or_allocate_hashed(uint64_t feature_id, uint64_t feature_hash) {
    {
      se::memory::PackedReadGuard metadata_lock(metadata_state_,
                                                metadata_readers_);
      const auto existing = id_map_.find(feature_id);
      if (existing != id_map_.end())
        return existing->second;
    }
    se::memory::PackedWriteGuard metadata_lock(metadata_state_,
                                               metadata_readers_);
    return get_or_allocate_impl(feature_id, feature_hash, true);
  }

  void accumulate_hashed(uint64_t feature_id, uint64_t feature_hash,
                         const ValT *grad) {
    uint32_t slot_index;
    bool needs_allocate = false;
    {
      se::memory::PackedReadGuard metadata_lock(metadata_state_,
                                                metadata_readers_);
      const auto existing = id_map_.find(feature_id);
      if (existing != id_map_.end()) {
        slot_index = existing->second;
      } else {
        needs_allocate = true;
      }
      if (!needs_allocate) {
        std::lock_guard stripe_lock(stripe_mutexes_[slot_index % kStripeCount]);
        accumulate_impl(slot_index, grad);
        return;
      }
    }
    {
      se::memory::PackedWriteGuard write_lock(metadata_state_,
                                              metadata_readers_);
      slot_index = get_or_allocate_impl(feature_id, feature_hash, true);
      std::lock_guard stripe_lock(stripe_mutexes_[slot_index % kStripeCount]);
      accumulate_impl(slot_index, grad);
    }
  }

  void accumulate_hashed_exclusive(uint64_t feature_id, uint64_t feature_hash,
                                   const ValT *grad) {
    auto [it, inserted] =
        id_map_.try_emplace(feature_id, static_cast<uint32_t>(next_slot_));
    if (!inserted) {
      accumulate_impl(it->second, grad);
      return;
    }
    if (next_slot_ == capacity_) [[unlikely]] {
      allocate_chunk();
    }
    const uint32_t slot_index = static_cast<uint32_t>(next_slot_);
    if constexpr (simd::is_simd_supported_v<ValT>) {
      simd::simd_copy(grad, reinterpret_cast<ValT *>(grad_data(slot_index)),
                      dim_);
    } else {
      std::copy_n(grad, dim_, reinterpret_cast<ValT *>(grad_data(slot_index)));
    }
    feature_ids_.push_back(feature_id);
    feature_hashes_.push_back(feature_hash);
    ++next_slot_;
  }

  template <typename Fn> void for_each_accumulated(Fn &&fn) {
    se::memory::PackedWriteGuard metadata_lock(metadata_state_,
                                               metadata_readers_);
    if (feature_ids_.size() != next_slot_ ||
        feature_hashes_.size() != next_slot_)
      throw std::logic_error(
          "transient arena feature-id layout is inconsistent");
    constexpr size_t kPrefetchDist = 8;
    for (size_t slot_index = 0; slot_index < next_slot_; ++slot_index) {
      if (slot_index + kPrefetchDist < next_slot_) {
        __builtin_prefetch(&feature_ids_[slot_index + kPrefetchDist], 0, 1);
        __builtin_prefetch(&feature_hashes_[slot_index + kPrefetchDist], 0, 1);
      }
      invoke_accumulated(fn, slot_index);
    }
  }

  template <typename Fn> void with_accumulated_batch(Fn &&fn) {
    se::memory::PackedWriteGuard metadata_lock(metadata_state_,
                                               metadata_readers_);
    if (feature_ids_.size() != next_slot_ ||
        feature_hashes_.size() != next_slot_)
      throw std::logic_error(
          "transient arena feature-id layout is inconsistent");
    AccumulatedBatchView view;
    view.arena = this;
    view.feature_ids = &feature_ids_;
    view.feature_hashes = &feature_hashes_;
    std::forward<Fn>(fn)(view);
  }

  void clear() {
    se::memory::PackedWriteGuard metadata_lock(metadata_state_,
                                               metadata_readers_);
    id_map_.clear();
    feature_ids_.clear();
    feature_hashes_.clear();
    next_slot_ = 0;
  }

  void reserve(size_t expected_unique_keys) {
    if (expected_unique_keys == 0)
      return;
    se::memory::PackedWriteGuard metadata_lock(metadata_state_,
                                               metadata_readers_);
    if (expected_unique_keys > std::numeric_limits<size_t>::max() - next_slot_)
      throw std::overflow_error("transient arena reserve size overflows");
    const size_t required = next_slot_ + expected_unique_keys;
    id_map_.reserve(expected_unique_keys);
    feature_ids_.reserve(expected_unique_keys);
    feature_hashes_.reserve(expected_unique_keys);
    while (capacity_ < required)
      allocate_chunk();
  }

  size_t dim() const { return dim_; }

  size_t active_slots() const {
    se::memory::PackedReadGuard metadata_lock(metadata_state_,
                                              metadata_readers_);
    return next_slot_;
  }

private:
  uint32_t get_or_allocate_impl(uint64_t feature_id, uint64_t feature_hash,
                                bool hash_valid) {
    if (next_slot_ > std::numeric_limits<uint32_t>::max() ||
        next_slot_ == capacity_) {
      auto existing = id_map_.find(feature_id);
      if (existing != id_map_.end())
        return existing->second;
      if (next_slot_ > std::numeric_limits<uint32_t>::max())
        throw std::length_error("transient arena index capacity exceeded");
      allocate_chunk();
    }
    const auto existing = id_map_.find(feature_id);
    if (existing != id_map_.end())
      return existing->second;
    const auto slot_index = static_cast<uint32_t>(next_slot_);
    auto [it_new, inserted] = id_map_.try_emplace(feature_id, slot_index);
    if (inserted) {
      std::memset(grad_data(slot_index), 0, dim_ * sizeof(ValT));
      feature_ids_.push_back(feature_id);
      feature_hashes_.push_back(hash_valid ? feature_hash : 0);
      ++next_slot_;
    }
    return it_new->second;
  }

  void accumulate_impl(uint32_t slot_index, const ValT *grad) {
    if constexpr (simd::is_simd_supported_v<ValT>) {
      simd::simd_accumulate(
          grad, reinterpret_cast<ValT *>(grad_data(slot_index)), dim_);
    } else {
      ValT *target = reinterpret_cast<ValT *>(grad_data(slot_index));
      for (size_t d = 0; d < dim_; ++d) {
        target[d] += grad[d];
      }
    }
  }

public:
  void accumulate(uint32_t slot_index, const ValT *grad) {
    se::memory::PackedReadGuard metadata_lock(metadata_state_,
                                              metadata_readers_);
    std::lock_guard stripe_lock(stripe_mutexes_[slot_index % kStripeCount]);
    accumulate_impl(slot_index, grad);
  }

private:
  template <typename Fn> void invoke_accumulated(Fn &fn, size_t slot_index) {
    const uint64_t key = feature_ids_[slot_index];
    const uint64_t hash = feature_hashes_[slot_index];
    const ValT *gradient = reinterpret_cast<const ValT *>(
        grad_data(static_cast<uint32_t>(slot_index)));
    if constexpr (std::is_invocable_v<Fn &, uint64_t, uint64_t, const ValT *>)
      fn(key, hash, gradient);
    else
      fn(key, gradient);
  }

  void allocate_chunk() {
    PageEvaluator evaluator;
    if (page_eviction_enabled_)
      evaluator = []() constexpr -> uint64_t {
        return std::numeric_limits<uint64_t>::max();
      };
    const PageEvictionMode eviction_mode = page_eviction_enabled_
                                               ? PageEvictionMode::kEnabled
                                               : PageEvictionMode::kDisabled;
    grad_chunks_.push_back(
        storage_->allocate(chunk_size_ * grad_stride_, kCacheLineSize,
                           std::move(evaluator), eviction_mode));
    capacity_ += chunk_size_;
  }

  ValT *grad_data(uint32_t slot_index) {
    size_t chunk_idx = slot_index >> chunk_shift_;
    size_t offset = slot_index & chunk_mask_;
    return reinterpret_cast<ValT *>(grad_chunks_[chunk_idx].data() +
                                    offset * grad_stride_);
  }

  const ValT *grad_data(uint32_t slot_index) const {
    size_t chunk_idx = slot_index >> chunk_shift_;
    size_t offset = slot_index & chunk_mask_;
    return reinterpret_cast<const ValT *>(grad_chunks_[chunk_idx].data() +
                                          offset * grad_stride_);
  }

  size_t dim_;
  size_t grad_stride_;
  PageStorage storage_;
  bool page_eviction_enabled_;
  size_t chunk_size_{8192};
  unsigned chunk_shift_{13};
  size_t chunk_mask_{8191};
  std::vector<StoragePage> grad_chunks_;
  size_t capacity_{0};
  static constexpr size_t kStripeCount = 64;
  mutable std::atomic<uint64_t> metadata_state_{0};
  mutable std::atomic<uint32_t> metadata_readers_{0};
  mutable std::array<std::mutex, kStripeCount> stripe_mutexes_;
  ankerl::unordered_dense::map<uint64_t, uint32_t> id_map_;
  std::vector<uint64_t> feature_ids_;
  std::vector<uint64_t> feature_hashes_;
  size_t next_slot_{0};
};

extern template class TransientArena<float>;
extern template class TransientArena<float16_t>;
extern template class TransientArena<bfloat16_t>;

} // namespace se

#endif // SPARSE_EMBEDDING_TRANSIENT_ARENA_HPP_
