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
#include <vector>

#include "sparse_embedding/src/simd/simd_ops.hpp"

namespace se {

template <typename ValT> class TransientArena {
  static constexpr size_t kCacheLineSize = 64;

  static constexpr size_t kChunkSize = 8192; // Number of elements per page

public:
  TransientArena(size_t dim, size_t initial_capacity = kChunkSize)
      : dim_(dim) {
    grad_stride_ =
        (dim * sizeof(ValT) + kCacheLineSize - 1) & ~(kCacheLineSize - 1);
    allocate_chunk();
    id_map_.reserve(initial_capacity);
  }

  ~TransientArena() {
    for (auto* ptr : grad_chunks_) {
      std::free(ptr);
    }
  }

  TransientArena(const TransientArena &) = delete;
  TransientArena &operator=(const TransientArena &) = delete;

  uint32_t get_or_allocate(uint64_t feature_id) {
    auto [it_new, inserted] = id_map_.try_emplace(feature_id, next_slot_);
    if (inserted) {
      if (next_slot_ >= capacity_) {
        allocate_chunk();
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
  void allocate_chunk() {
    std::byte *new_chunk = static_cast<std::byte *>(
        std::aligned_alloc(kCacheLineSize, kChunkSize * grad_stride_));
    if (!new_chunk) {
      throw std::bad_alloc();
    }
    grad_chunks_.push_back(new_chunk);
    capacity_ += kChunkSize;
  }

  ValT *grad_data(uint32_t slot_index) {
    size_t chunk_idx = slot_index / kChunkSize;
    size_t offset = slot_index % kChunkSize;
    return reinterpret_cast<ValT *>(grad_chunks_[chunk_idx] + offset * grad_stride_);
  }

  const ValT *grad_data(uint32_t slot_index) const {
    size_t chunk_idx = slot_index / kChunkSize;
    size_t offset = slot_index % kChunkSize;
    return reinterpret_cast<const ValT *>(grad_chunks_[chunk_idx] +
                                          offset * grad_stride_);
  }

  size_t dim_;
  size_t grad_stride_;
  std::vector<std::byte *> grad_chunks_;
  size_t capacity_{0};
  ankerl::unordered_dense::map<uint64_t, uint32_t> id_map_;
  uint32_t next_slot_{0};
};

} // namespace se

#endif // SPARSE_EMBEDDING_TRANSIENT_ARENA_HPP_
