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

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <vector>

#include "sparse_embedding/src/storage/page_storage.hpp"

namespace se {

template <typename T> class OptimizerStateArena {
  static constexpr size_t kCacheLineSize = 64;
  static constexpr size_t kEntriesPerPage = 4096;

public:
  explicit OptimizerStateArena(size_t dim, PageStorage storage,
                               bool page_eviction_enabled = false)
      : dim_(dim), storage_(std::move(storage)),
        page_eviction_enabled_(page_eviction_enabled) {
    row_stride_ =
        (dim * sizeof(T) + kCacheLineSize - 1) & ~(kCacheLineSize - 1);
  }

  OptimizerStateArena(const OptimizerStateArena &) = delete;
  OptimizerStateArena &operator=(const OptimizerStateArena &) = delete;
  OptimizerStateArena(OptimizerStateArena &&) = default;
  OptimizerStateArena &operator=(OptimizerStateArena &&) = default;

  void allocate(uint32_t idx) {
    if (idx >= capacity_.load(std::memory_order_acquire)) {
      throw std::logic_error("optimizer state capacity was not reserved");
    }
    std::memset(raw(idx), 0, row_stride_);
  }

  void reserve(size_t capacity) {
    if (capacity == 0) {
      return;
    }
    std::lock_guard<std::mutex> lock(alloc_mtx_);
    size_t required_pages = (capacity + kEntriesPerPage - 1) / kEntriesPerPage;
    while (pages_.size() < required_pages) {
      PageEvaluator evaluator;
      if (page_eviction_enabled_)
        evaluator = []() constexpr -> uint64_t {
          return std::numeric_limits<uint64_t>::max();
        };
      const PageEvictionMode eviction_mode = page_eviction_enabled_
                                                 ? PageEvictionMode::kEnabled
                                                 : PageEvictionMode::kDisabled;
      pages_.push_back(storage_->allocate(kEntriesPerPage * row_stride_,
                                          kCacheLineSize, std::move(evaluator),
                                          eviction_mode));
      capacity_.store(pages_.size() * kEntriesPerPage,
                      std::memory_order_release);
    }
  }

  size_t capacity() const { return capacity_.load(std::memory_order_acquire); }

  void set_page_eviction_enabled(bool enabled) {
    std::lock_guard<std::mutex> lock(alloc_mtx_);
    if (!pages_.empty() && enabled != page_eviction_enabled_)
      throw std::logic_error(
          "optimizer page eviction policy cannot change after allocation");
    page_eviction_enabled_ = enabled;
  }

  void deallocate(uint32_t index) { std::memset(raw(index), 0, row_stride_); }

  void clear() {
    std::lock_guard<std::mutex> lock(alloc_mtx_);
    pages_.clear();
    capacity_.store(0, std::memory_order_release);
  }

  T *data(uint32_t index) { return reinterpret_cast<T *>(raw(index)); }

  const T *data(uint32_t index) const {
    return reinterpret_cast<const T *>(raw(index));
  }

  constexpr size_t dim() const { return dim_; }

private:
  std::byte *raw(uint32_t index) const {
    return pages_[index / kEntriesPerPage].data() +
           (index % kEntriesPerPage) * row_stride_;
  }

  size_t dim_;
  size_t row_stride_;
  PageStorage storage_;
  bool page_eviction_enabled_;
  std::vector<StoragePage> pages_;
  std::atomic<size_t> capacity_{0};
  std::mutex alloc_mtx_;
};

} // namespace se

#endif // SPARSE_EMBEDDING_OPTIMIZER_STATE_HPP_
