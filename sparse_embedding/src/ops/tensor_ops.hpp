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

#ifndef SPARSE_EMBEDDING_TENSOR_OPS_HPP_
#define SPARSE_EMBEDDING_TENSOR_OPS_HPP_

#include <cstdint>
#include <memory>
#include <span>

#include <ankerl/unordered_dense.h>
#include <exec/static_thread_pool.hpp>
#include <stdexec/execution.hpp>

namespace se {
namespace ops {

template <typename KeyT> struct UniqueResult {
  std::shared_ptr<KeyT[]> unique_keys;
  size_t unique_count;
  std::shared_ptr<int32_t[]> inverse_indices;
  size_t inverse_count;
  std::shared_ptr<int32_t[]> first_indices;
  std::shared_ptr<int32_t[]> counts;
};

// High-performance Unique operator using ankerl::unordered_dense map.
// Designed to be run inside the stdexec thread pool for heavy ops, but
// internally limits contention by fast thread-local/hash approaches. For
// simplicity and maximizing L1 caches on moderately sized batches (e.g. 1M
// elements), a tight sequential pass on ankerl map often outperforms complex
// parallel-sorts due to prefetching.
template <typename KeyT>
UniqueResult<KeyT> unique_keys_inverse(std::span<const KeyT> keys,
                                       bool compute_first_indices = false,
                                       bool compute_counts = false) {
  size_t n = keys.size();

  auto inverse = std::shared_ptr<int32_t[]>(new int32_t[n]);
  std::shared_ptr<int32_t[]> first_indices =
      compute_first_indices ? std::shared_ptr<int32_t[]>(new int32_t[n])
                            : nullptr;
  std::shared_ptr<int32_t[]> counts =
      compute_counts ? std::shared_ptr<int32_t[]>(new int32_t[n]) : nullptr;

  // Fast hash map to deduce uniqueness and contiguous IDs
  ankerl::unordered_dense::map<KeyT, int32_t> unique_map;
  unique_map.reserve(n);

  for (size_t i = 0; i < n; ++i) {
    KeyT key = keys[i];
    auto [it, inserted] = unique_map.try_emplace(key, unique_map.size());
    int32_t uid = it->second;
    inverse[i] = uid;
    if (inserted) {
      if (compute_first_indices)
        first_indices[uid] = i;
      if (compute_counts)
        counts[uid] = 1;
    } else {
      if (compute_counts)
        counts[uid]++;
    }
  }

  size_t unique_count = unique_map.size();
  auto unique_keys = std::shared_ptr<KeyT[]>(new KeyT[unique_count]);

  // ankerl::unordered_dense ensures dense contiguous iteration
  // so we can extract keys extremely fast
  size_t idx = 0;
  for (const auto &kv : unique_map) {
    unique_keys[idx++] = kv.first;
  }

  return {unique_keys, unique_count, inverse, n, first_indices, counts};
}

} // namespace ops
} // namespace se

#endif // SPARSE_EMBEDDING_TENSOR_OPS_HPP_
