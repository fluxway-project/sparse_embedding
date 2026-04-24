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
#include <stdexcept>
#include <type_traits>

#include <ankerl/unordered_dense.h>
#include <tsl/robin_map.h>

#include "sparse_embedding/src/simd/simd_ops.hpp"

namespace se {

template <typename ValT> class TransientArena {
  static constexpr size_t kCacheLineSize = 64;

public:
  TransientArena(size_t dim, size_t max_active_keys)
      : dim_(dim), capacity_(max_active_keys) {
    grad_stride_ =
        (dim * sizeof(ValT) + kCacheLineSize - 1) & ~(kCacheLineSize - 1);
    grad_buffer_ = static_cast<std::byte *>(
        std::aligned_alloc(kCacheLineSize, capacity_ * grad_stride_));
    if (!grad_buffer_) {
      throw std::bad_alloc();
    }
    std::memset(grad_buffer_, 0, capacity_ * grad_stride_);
    id_map_.reserve(capacity_ / 2);
  }

  ~TransientArena() { std::free(grad_buffer_); }

  TransientArena(const TransientArena &) = delete;
  TransientArena &operator=(const TransientArena &) = delete;

  uint32_t get_or_allocate(uint64_t feature_id) {
    auto [it_new, inserted] = id_map_.try_emplace(feature_id, next_slot_);
    if (inserted) {
      if (next_slot_ >= capacity_) {
        throw std::overflow_error("transient arena capacity exceeded");
      }
      std::memset(grad_data(next_slot_), 0, dim_ * sizeof(ValT));
      ++next_slot_;
    }
    return it_new->second;
  }

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

  template <typename Fn> void for_each_accumulated(Fn &&fn) {
    for (const auto &[feature_id, slot_index] : id_map_) {
      fn(feature_id, reinterpret_cast<const ValT *>(grad_data(slot_index)));
    }
  }

  void clear() {
    id_map_.clear();
    next_slot_ = 0;
  }

  size_t dim() const { return dim_; }

  size_t active_slots() const { return next_slot_; }

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
  size_t grad_stride_;
  std::byte *grad_buffer_;
  tsl::robin_map<uint64_t, uint32_t, ankerl::unordered_dense::hash<uint64_t>>
      id_map_;
  uint32_t next_slot_{0};
};

} // namespace se

#endif // SPARSE_EMBEDDING_TRANSIENT_ARENA_HPP_
