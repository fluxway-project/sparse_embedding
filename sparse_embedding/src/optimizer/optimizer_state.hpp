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

#ifndef SPARSE_EMBEDDING_OPTIMIZER_STATE_HPP_
#define SPARSE_EMBEDDING_OPTIMIZER_STATE_HPP_

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <vector>

namespace se {

class OptimizerStateArena {
  static constexpr size_t kCacheLineSize = 64;
  static constexpr size_t kEntriesPerPage = 4096;

  struct AlignedDeleter {
    void operator()(std::byte *p) const { std::free(p); }
  };

public:
  explicit OptimizerStateArena(size_t dim) : dim_(dim) {
    row_stride_ =
        (dim * sizeof(float) + kCacheLineSize - 1) & ~(kCacheLineSize - 1);
  }

  OptimizerStateArena(const OptimizerStateArena &) = delete;
  OptimizerStateArena &operator=(const OptimizerStateArena &) = delete;
  OptimizerStateArena(OptimizerStateArena &&) = default;
  OptimizerStateArena &operator=(OptimizerStateArena &&) = default;

  uint32_t allocate() {
    std::lock_guard<std::mutex> lock(alloc_mtx_);
    uint32_t idx;
    if (!free_list_.empty()) {
      idx = free_list_.back();
      free_list_.pop_back();
    } else {
      idx = next_index_++;
      if (idx / kEntriesPerPage >= pages_.size()) {
        auto *p = static_cast<std::byte *>(
            std::aligned_alloc(kCacheLineSize, kEntriesPerPage * row_stride_));
        if (!p)
          throw std::bad_alloc();
        std::memset(p, 0, kEntriesPerPage * row_stride_);
        pages_.emplace_back(p);
      }
    }
    std::memset(raw(idx), 0, row_stride_);
    return idx;
  }

  void deallocate(uint32_t index) {
    std::lock_guard<std::mutex> lock(alloc_mtx_);
    free_list_.push_back(index);
  }

  void clear() {
    std::lock_guard<std::mutex> lock(alloc_mtx_);
    pages_.clear();
    free_list_.clear();
    next_index_ = 0;
  }

  float *data(uint32_t index) { return reinterpret_cast<float *>(raw(index)); }

  const float *data(uint32_t index) const {
    return reinterpret_cast<const float *>(raw(index));
  }

  size_t dim() const { return dim_; }

private:
  std::byte *raw(uint32_t index) const {
    return pages_[index / kEntriesPerPage].get() +
           (index % kEntriesPerPage) * row_stride_;
  }

  size_t dim_;
  size_t row_stride_;
  std::vector<std::unique_ptr<std::byte[], AlignedDeleter>> pages_;
  std::vector<uint32_t> free_list_;
  uint32_t next_index_ = 0;
  std::mutex alloc_mtx_;
};

} // namespace se

#endif // SPARSE_EMBEDDING_OPTIMIZER_STATE_HPP_
