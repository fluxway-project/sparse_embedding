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

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <shared_mutex>
#include <stdexcept>
#include <type_traits>

#include <ankerl/unordered_dense.h>
#include <tsl/robin_map.h>

#include "sparse_embedding/src/simd/simd_ops.hpp"

namespace se {

template <typename ValT> class TransientArena {
  static constexpr size_t kCacheLineSize = 64;

public:
  TransientArena(size_t dim, size_t max_active_keys, uint32_t grad_accum_steps)
      : dim_(dim), capacity_(max_active_keys),
        grad_accum_steps_(grad_accum_steps) {
    grad_stride_ =
        (dim * sizeof(ValT) + kCacheLineSize - 1) & ~(kCacheLineSize - 1);
    grad_buffer_ = static_cast<std::byte *>(
        std::aligned_alloc(kCacheLineSize, capacity_ * grad_stride_));
    if (!grad_buffer_) {
      throw std::bad_alloc();
    }
    std::memset(grad_buffer_, 0, capacity_ * grad_stride_);
    locks_ = std::make_unique<std::atomic_flag[]>(capacity_);
    id_map_.reserve(capacity_ / 2);
  }

  ~TransientArena() { std::free(grad_buffer_); }

  TransientArena(const TransientArena &) = delete;
  TransientArena &operator=(const TransientArena &) = delete;

  uint32_t get_or_allocate(uint64_t feature_id) {
    {
      std::shared_lock lock(id_map_mtx_);
      auto it = id_map_.find(feature_id);
      if (it != id_map_.end())
        return it->second;
    }
    {
      std::unique_lock lock(id_map_mtx_);
      auto [it, inserted] = id_map_.try_emplace(
          feature_id, next_slot_.load(std::memory_order_relaxed));
      if (inserted) {
        uint32_t slot = next_slot_.fetch_add(1, std::memory_order_relaxed);
        if (slot >= capacity_) {
          throw std::overflow_error("transient arena capacity exceeded");
        }
        std::memset(grad_data(slot), 0, dim_ * sizeof(ValT));
        it.value() = slot;
      }
      return it->second;
    }
  }

  void accumulate(uint32_t slot_index, const ValT *grad) {
    auto &lock = locks_[slot_index];
    while (lock.test_and_set(std::memory_order_acquire)) {
#if defined(__x86_64__)
      _mm_pause();
#endif
    }

    if constexpr (simd::is_simd_supported_v<ValT>) {
      simd::simd_accumulate(
          grad, reinterpret_cast<ValT *>(grad_data(slot_index)), dim_);
    } else {
      ValT *target = reinterpret_cast<ValT *>(grad_data(slot_index));
      for (size_t d = 0; d < dim_; ++d) {
        target[d] += grad[d];
      }
    }

    lock.clear(std::memory_order_release);
  }

  bool advance_batch() {
    uint32_t prev = batch_counter_.fetch_add(1, std::memory_order_acq_rel);
    return (prev + 1) >= grad_accum_steps_;
  }

  template <typename Fn> void for_each_accumulated(Fn &&fn) {
    std::shared_lock lock(id_map_mtx_);
    for (const auto &[feature_id, slot_index] : id_map_) {
      fn(feature_id, reinterpret_cast<const ValT *>(grad_data(slot_index)));
    }
  }

  void clear() {
    std::unique_lock lock(id_map_mtx_);
    id_map_.clear();
    next_slot_.store(0, std::memory_order_relaxed);
    batch_counter_.store(0, std::memory_order_relaxed);
    std::memset(grad_buffer_, 0, capacity_ * grad_stride_);
  }

  uint32_t grad_accum_steps() const { return grad_accum_steps_; }
  size_t dim() const { return dim_; }

  size_t active_slots() const {
    return next_slot_.load(std::memory_order_relaxed);
  }

private:
  ValT *grad_data(uint32_t slot_index) {
    return reinterpret_cast<ValT *>(grad_buffer_ + slot_index * grad_stride_);
  }

  const ValT *grad_data(uint32_t slot_index) const {
    return reinterpret_cast<const ValT *>(grad_buffer_ +
                                          slot_index * grad_stride_);
  }

  size_t dim_;
  size_t capacity_;
  uint32_t grad_accum_steps_;
  size_t grad_stride_;

  std::byte *grad_buffer_;
  std::unique_ptr<std::atomic_flag[]> locks_;

  tsl::robin_map<uint64_t, uint32_t, ankerl::unordered_dense::hash<uint64_t>>
      id_map_;
  std::shared_mutex id_map_mtx_;

  std::atomic<uint32_t> next_slot_{0};
  std::atomic<uint32_t> batch_counter_{0};
};

} // namespace se

#endif // SPARSE_EMBEDDING_TRANSIENT_ARENA_HPP_
