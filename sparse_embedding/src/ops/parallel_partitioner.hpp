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

#ifndef SPARSE_EMBEDDING_PARALLEL_PARTITIONER_HPP_
#define SPARSE_EMBEDDING_PARALLEL_PARTITIONER_HPP_

#include <algorithm>
#include <cstdint>
#include <exec/static_thread_pool.hpp>
#include <span>
#include <stdexcept>
#include <stdexec/execution.hpp>
#include <thread>

namespace se::ops {

constexpr size_t kPartitionerParallelThreshold = 16384;

inline exec::static_thread_pool &global_partition_pool() {
  static exec::static_thread_pool pool(
      std::max(1u, std::thread::hardware_concurrency()));
  return pool;
}

template <typename Pool, typename Fn>
inline void parallel_dispatch_chunks(Pool &pool, size_t n, size_t num_threads,
                                     size_t min_chunk_size, Fn &&fn) {
  if (n == 0)
    return;
  size_t effective_workers =
      (num_threads == 0) ? std::max(1u, std::thread::hardware_concurrency())
                         : num_threads;
  if (effective_workers <= 1 || n < min_chunk_size) {
    fn(0, n);
    return;
  }
  effective_workers =
      std::min(effective_workers, (n + min_chunk_size - 1) / min_chunk_size);
  if (effective_workers <= 1) {
    fn(0, n);
    return;
  }
  const size_t chunk_size = (n + effective_workers - 1) / effective_workers;
  stdexec::sync_wait(stdexec::schedule(pool.get_scheduler()) |
                     stdexec::bulk(stdexec::par, effective_workers,
                                   [&fn, n, chunk_size](size_t i) {
                                     const size_t begin =
                                         std::min(i * chunk_size, n);
                                     const size_t end =
                                         std::min(begin + chunk_size, n);
                                     if (begin < end) {
                                       fn(begin, end - begin);
                                     }
                                   }));
}

template <typename P, typename K>
inline void invoke_partition_span(P &p, std::span<const K> keys,
                                  std::span<uint32_t> out) {
  if constexpr (requires { p->partition_span(keys, out); }) {
    p->partition_span(keys, out);
  } else {
    p.partition_span(keys, out);
  }
}

template <typename P, typename K>
inline void invoke_filter_span(P &p, std::span<const K> keys, uint32_t shard,
                               std::span<bool> out) {
  if constexpr (requires { p->filter_span(keys, shard, out); }) {
    p->filter_span(keys, shard, out);
  } else {
    p.filter_span(keys, shard, out);
  }
}

template <typename Pool, typename PartitionerT, typename KeyT>
inline void parallel_partition(Pool &pool, PartitionerT &&partitioner,
                               std::span<const KeyT> keys,
                               std::span<uint32_t> out,
                               size_t num_threads = 0) {
  if (keys.size() != out.size()) {
    throw std::invalid_argument("keys and out size mismatch");
  }
  parallel_dispatch_chunks(
      pool, keys.size(), num_threads, kPartitionerParallelThreshold,
      [&](size_t offset, size_t count) {
        invoke_partition_span(partitioner, keys.subspan(offset, count),
                              out.subspan(offset, count));
      });
}

template <typename Pool, typename PartitionerT, typename KeyT>
inline void parallel_filter(Pool &pool, PartitionerT &&partitioner,
                            std::span<const KeyT> keys, uint32_t target_shard,
                            std::span<bool> out, size_t num_threads = 0) {
  if (keys.size() != out.size()) {
    throw std::invalid_argument("keys and out size mismatch");
  }
  parallel_dispatch_chunks(
      pool, keys.size(), num_threads, kPartitionerParallelThreshold,
      [&](size_t offset, size_t count) {
        invoke_filter_span(partitioner, keys.subspan(offset, count),
                           target_shard, out.subspan(offset, count));
      });
}

} // namespace se::ops

#endif // SPARSE_EMBEDDING_PARALLEL_PARTITIONER_HPP_
