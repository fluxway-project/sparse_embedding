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

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "sparse_embedding/src/simd/simd_ops.hpp"
#include "sparse_embedding/src/storage/page_storage.hpp"

namespace se {

template <typename ValT> class TransientArena {
  static constexpr size_t kCacheLineSize = 64;

  static constexpr size_t kChunkSize = 8192; // Number of elements per page

public:
  TransientArena(size_t dim, PageStorage storage,
                 bool page_eviction_enabled = false)
      : dim_(dim), storage_(std::move(storage)),
        page_eviction_enabled_(page_eviction_enabled) {
    grad_stride_ =
        (dim * sizeof(ValT) + kCacheLineSize - 1) & ~(kCacheLineSize - 1);
    id_map_.reserve(kChunkSize);
  }

  ~TransientArena() = default;

  TransientArena(const TransientArena &) = delete;
  TransientArena &operator=(const TransientArena &) = delete;

  uint32_t get_or_allocate(uint64_t feature_id) {
    return get_or_allocate_impl(feature_id, 0, false);
  }

  uint32_t get_or_allocate_hashed(uint64_t feature_id, uint64_t feature_hash) {
    return get_or_allocate_impl(feature_id, feature_hash, true);
  }

  template <typename Fn> void for_each_accumulated(Fn &&fn) {
    if (feature_ids_.size() != next_slot_ ||
        feature_hashes_.size() != next_slot_)
      throw std::logic_error(
          "transient arena feature-id layout is inconsistent");
    for (size_t slot_index = 0; slot_index < next_slot_; ++slot_index)
      invoke_accumulated(fn, slot_index);
  }

  void clear() {
    id_map_.clear();
    feature_ids_.clear();
    feature_hashes_.clear();
    next_slot_ = 0;
  }

  void reserve(size_t expected_unique_keys) {
    if (expected_unique_keys == 0)
      return;
    id_map_.reserve(expected_unique_keys);
    feature_ids_.reserve(expected_unique_keys);
    feature_hashes_.reserve(expected_unique_keys);
  }

  size_t dim() const { return dim_; }

  size_t active_slots() const { return next_slot_; }

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

public:
  void accumulate(uint32_t slot_index, const ValT *grad) {
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
        storage_->allocate(kChunkSize * grad_stride_, kCacheLineSize,
                           std::move(evaluator), eviction_mode));
    capacity_ += kChunkSize;
  }

  ValT *grad_data(uint32_t slot_index) {
    size_t chunk_idx = slot_index / kChunkSize;
    size_t offset = slot_index % kChunkSize;
    return reinterpret_cast<ValT *>(grad_chunks_[chunk_idx].data() +
                                    offset * grad_stride_);
  }

  const ValT *grad_data(uint32_t slot_index) const {
    size_t chunk_idx = slot_index / kChunkSize;
    size_t offset = slot_index % kChunkSize;
    return reinterpret_cast<const ValT *>(grad_chunks_[chunk_idx].data() +
                                          offset * grad_stride_);
  }

  size_t dim_;
  size_t grad_stride_;
  PageStorage storage_;
  bool page_eviction_enabled_;
  std::vector<StoragePage> grad_chunks_;
  size_t capacity_{0};
  ankerl::unordered_dense::map<uint64_t, uint32_t> id_map_;
  std::vector<uint64_t> feature_ids_;
  std::vector<uint64_t> feature_hashes_;
  size_t next_slot_{0};
};

} // namespace se

#endif // SPARSE_EMBEDDING_TRANSIENT_ARENA_HPP_
