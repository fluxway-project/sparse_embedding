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

#ifndef SPARSE_EMBEDDING_UNIQUE_OPS_HPP_
#define SPARSE_EMBEDDING_UNIQUE_OPS_HPP_

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <vector>

#include <ankerl/unordered_dense.h>
#include <parlay/parallel.h>

namespace se::ops {

constexpr size_t kUniqueParallelThreshold = 32768;
constexpr size_t kDefaultPartitions = 256;
constexpr unsigned kDefaultPartitionBits = 8;
constexpr size_t kDefaultSweetSpotThreads = 24;

inline void validate_unique_input_size(size_t size) {
  if (size > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
    throw std::invalid_argument(
        "Input size exceeds maximum supported 32-bit element count (2^31 - 1)");
  }
}

template <typename KeyT> struct UniqueResult {
  std::shared_ptr<KeyT[]> unique_keys;
  size_t unique_count = 0;
  std::shared_ptr<int32_t[]> inverse_indices;
  size_t inverse_count = 0;
  std::shared_ptr<int32_t[]> first_indices;
  std::shared_ptr<int32_t[]> counts;
};

struct SplitMix64Hasher {
  template <typename T>
  [[nodiscard]] constexpr uint64_t operator()(T key) const noexcept {
    uint64_t x =
        static_cast<uint64_t>(static_cast<std::make_unsigned_t<T>>(key));
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ull;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebull;
    x ^= x >> 31;
    return x;
  }
};

template <typename KeyT>
UniqueResult<KeyT> unique_keys_inverse_sequential(std::span<const KeyT> keys,
                                                  bool compute_first_indices,
                                                  bool compute_counts,
                                                  bool compute_inverse) {
  const size_t n = keys.size();
  validate_unique_input_size(n);
  if (n == 0) {
    return {};
  }

  std::shared_ptr<int32_t[]> inverse =
      compute_inverse ? std::shared_ptr<int32_t[]>(new int32_t[n]) : nullptr;
  std::shared_ptr<int32_t[]> first_indices =
      compute_first_indices ? std::shared_ptr<int32_t[]>(new int32_t[n])
                            : nullptr;
  std::shared_ptr<int32_t[]> counts =
      compute_counts ? std::shared_ptr<int32_t[]>(new int32_t[n]) : nullptr;

  ankerl::unordered_dense::map<KeyT, int32_t> unique_map;
  unique_map.reserve(std::min(n, size_t{1048576}));

  for (size_t i = 0; i < n; ++i) {
    const KeyT key = keys[i];
    auto [it, inserted] =
        unique_map.try_emplace(key, static_cast<int32_t>(unique_map.size()));
    const int32_t uid = it->second;
    if (compute_inverse) {
      inverse[i] = uid;
    }
    if (inserted) {
      if (compute_first_indices) {
        first_indices[uid] = static_cast<int32_t>(i);
      }
      if (compute_counts) {
        counts[uid] = 1;
      }
    } else {
      if (compute_counts) {
        ++counts[uid];
      }
    }
  }

  const size_t unique_count = unique_map.size();
  auto unique_keys = std::shared_ptr<KeyT[]>(new KeyT[unique_count]);
  size_t idx = 0;
  for (const auto &kv : unique_map) {
    unique_keys[idx++] = kv.first;
  }

  return {unique_keys,   unique_count, inverse, compute_inverse ? n : 0,
          first_indices, counts};
}

template <typename KeyT, typename HasherT = SplitMix64Hasher>
UniqueResult<KeyT> unique_keys_inverse_radix(
    std::span<const KeyT> keys, bool compute_first_indices, bool compute_counts,
    size_t num_threads, bool compute_inverse, HasherT hasher = HasherT{}) {
  const size_t n = keys.size();
  validate_unique_input_size(n);
  if (n < kUniqueParallelThreshold || num_threads == 1) {
    return unique_keys_inverse_sequential(keys, compute_first_indices,
                                          compute_counts, compute_inverse);
  }

  size_t effective_workers =
      (num_threads == 0) ? std::min(kDefaultSweetSpotThreads,
                                    std::max(size_t{1}, parlay::num_workers()))
                         : num_threads;
  effective_workers = std::max(
      size_t{1}, std::min({effective_workers, n / 1024, kDefaultPartitions}));

  if (effective_workers <= 1) {
    return unique_keys_inverse_sequential(keys, compute_first_indices,
                                          compute_counts, compute_inverse);
  }

  const size_t chunk_size = (n + effective_workers - 1) / effective_workers;

  // Pass 1: Thread-local partition histograms
  std::vector<std::array<size_t, kDefaultPartitions>> local_hists(
      effective_workers);
  for (auto &hist : local_hists) {
    hist.fill(0);
  }

  parlay::parallel_for(
      0, effective_workers,
      [&](size_t t) {
        const size_t start = std::min(t * chunk_size, n);
        const size_t end = std::min(start + chunk_size, n);
        auto &hist = local_hists[t];
        for (size_t i = start; i < end; ++i) {
          const uint64_t h = hasher(keys[i]);
          const size_t p =
              (h >> (64 - kDefaultPartitionBits)) & (kDefaultPartitions - 1);
          ++hist[p];
        }
      },
      1);

  // Pass 2: Prefix sums for partition offsets and thread write offsets
  std::array<size_t, kDefaultPartitions + 1> partition_offsets{};
  std::vector<std::array<size_t, kDefaultPartitions>> thread_offsets(
      effective_workers);
  for (size_t p = 0; p < kDefaultPartitions; ++p) {
    size_t sum = 0;
    for (size_t t = 0; t < effective_workers; ++t) {
      thread_offsets[t][p] = partition_offsets[p] + sum;
      sum += local_hists[t][p];
    }
    partition_offsets[p + 1] = partition_offsets[p] + sum;
  }

  // Pass 3: Parallel allocate partition buffers and scatter
  auto part_keys = std::make_unique_for_overwrite<KeyT[]>(n);
  const bool need_orig_idx = compute_inverse || compute_first_indices;
  std::unique_ptr<int32_t[]> part_orig_idx;
  if (need_orig_idx) {
    part_orig_idx = std::make_unique_for_overwrite<int32_t[]>(n);
  }

  // Pre-fault pages in parallel chunks to avoid concurrent page fault lock
  // contention
  constexpr size_t kPageStrideKeys = 4096 / sizeof(KeyT);
  constexpr size_t kPageStrideIdx = 4096 / sizeof(int32_t);

  parlay::parallel_for(
      0, effective_workers,
      [&](size_t t) {
        const size_t start = std::min(t * chunk_size, n);
        const size_t end = std::min(start + chunk_size, n);
        for (size_t i = start; i < end; i += kPageStrideKeys) {
          part_keys[i] = 0;
        }
        if (end > start) {
          part_keys[end - 1] = 0;
        }
        if (need_orig_idx) {
          for (size_t i = start; i < end; i += kPageStrideIdx) {
            part_orig_idx[i] = 0;
          }
          if (end > start) {
            part_orig_idx[end - 1] = 0;
          }
        }
      },
      1);

  parlay::parallel_for(
      0, effective_workers,
      [&](size_t t) {
        const size_t start = std::min(t * chunk_size, n);
        const size_t end = std::min(start + chunk_size, n);
        auto cursors = thread_offsets[t];
        if (need_orig_idx) {
          for (size_t i = start; i < end; ++i) {
            const KeyT k = keys[i];
            const uint64_t h = hasher(k);
            const size_t p =
                (h >> (64 - kDefaultPartitionBits)) & (kDefaultPartitions - 1);
            const size_t pos = cursors[p]++;
            part_keys[pos] = k;
            part_orig_idx[pos] = static_cast<int32_t>(i);
          }
        } else {
          for (size_t i = start; i < end; ++i) {
            const KeyT k = keys[i];
            const uint64_t h = hasher(k);
            const size_t p =
                (h >> (64 - kDefaultPartitionBits)) & (kDefaultPartitions - 1);
            const size_t pos = cursors[p]++;
            part_keys[pos] = k;
          }
        }
      },
      1);

  // Pass 4: Partition-level concurrent deduplication (lock-free, L2-resident)
  struct PartitionOutput {
    std::vector<KeyT> ukeys;
    std::vector<int32_t> ufirst;
    std::vector<int32_t> ucounts;
  };
  std::vector<PartitionOutput> part_results(kDefaultPartitions);
  std::unique_ptr<int32_t[]> part_local_uids;
  if (compute_inverse) {
    part_local_uids = std::make_unique_for_overwrite<int32_t[]>(n);
    parlay::parallel_for(
        0, effective_workers,
        [&](size_t t) {
          const size_t start = std::min(t * chunk_size, n);
          const size_t end = std::min(start + chunk_size, n);
          for (size_t i = start; i < end; i += kPageStrideIdx) {
            part_local_uids[i] = 0;
          }
          if (end > start) {
            part_local_uids[end - 1] = 0;
          }
        },
        1);
  }

  parlay::parallel_for(
      0, effective_workers,
      [&](size_t t) {
        const size_t p_start = t * kDefaultPartitions / effective_workers;
        const size_t p_end = (t + 1) * kDefaultPartitions / effective_workers;
        for (size_t p = p_start; p < p_end; ++p) {
          const size_t start = partition_offsets[p];
          const size_t end = partition_offsets[p + 1];
          const size_t len = end - start;
          if (len == 0) {
            continue;
          }

          ankerl::unordered_dense::map<KeyT, int32_t> lmap;
          lmap.reserve(std::min(len, size_t{65536}));

          if (compute_counts) {
            part_results[p].ucounts.reserve(std::min(len, size_t{65536}));
          }
          if (compute_first_indices) {
            part_results[p].ufirst.reserve(std::min(len, size_t{65536}));
          }

          for (size_t j = start; j < end; ++j) {
            const KeyT k = part_keys[j];
            auto [it, inserted] =
                lmap.try_emplace(k, static_cast<int32_t>(lmap.size()));
            const int32_t uid = it->second;
            if (compute_inverse) {
              part_local_uids[j] = uid;
            }

            if (inserted) {
              if (compute_first_indices) {
                part_results[p].ufirst.push_back(part_orig_idx[j]);
              }
              if (compute_counts) {
                part_results[p].ucounts.push_back(1);
              }
            } else {
              if (compute_counts) {
                ++part_results[p].ucounts[uid];
              }
            }
          }

          part_results[p].ukeys.resize(lmap.size());
          for (const auto &kv : lmap) {
            part_results[p].ukeys[kv.second] = kv.first;
          }
        }
      },
      1);

  // Pass 5: Prefix sum of unique counts across partitions
  std::array<int32_t, kDefaultPartitions + 1> global_uoffsets{};
  for (size_t p = 0; p < kDefaultPartitions; ++p) {
    global_uoffsets[p + 1] =
        global_uoffsets[p] + static_cast<int32_t>(part_results[p].ukeys.size());
  }

  const size_t total_unique = global_uoffsets[kDefaultPartitions];
  auto unique_keys = std::shared_ptr<KeyT[]>(new KeyT[total_unique]);
  std::shared_ptr<int32_t[]> inverse =
      compute_inverse ? std::shared_ptr<int32_t[]>(new int32_t[n]) : nullptr;
  std::shared_ptr<int32_t[]> first_indices =
      compute_first_indices
          ? std::shared_ptr<int32_t[]>(new int32_t[total_unique])
          : nullptr;
  std::shared_ptr<int32_t[]> counts =
      compute_counts ? std::shared_ptr<int32_t[]>(new int32_t[total_unique])
                     : nullptr;

  if (compute_inverse) {
    parlay::parallel_for(
        0, effective_workers,
        [&](size_t t) {
          const size_t start = std::min(t * chunk_size, n);
          const size_t end = std::min(start + chunk_size, n);
          for (size_t i = start; i < end; i += kPageStrideIdx) {
            inverse[i] = 0;
          }
          if (end > start) {
            inverse[end - 1] = 0;
          }
        },
        1);
  }

  // Pass 6: Parallel gather unique keys, metadata, and scatter inverse indices
  parlay::parallel_for(
      0, effective_workers,
      [&](size_t t) {
        const size_t p_start = t * kDefaultPartitions / effective_workers;
        const size_t p_end = (t + 1) * kDefaultPartitions / effective_workers;
        for (size_t p = p_start; p < p_end; ++p) {
          const int32_t uoff = global_uoffsets[p];
          const size_t p_count = part_results[p].ukeys.size();
          if (p_count == 0) {
            continue;
          }

          std::copy(part_results[p].ukeys.begin(), part_results[p].ukeys.end(),
                    unique_keys.get() + uoff);

          if (compute_first_indices) {
            std::copy(part_results[p].ufirst.begin(),
                      part_results[p].ufirst.end(), first_indices.get() + uoff);
          }
          if (compute_counts) {
            std::copy(part_results[p].ucounts.begin(),
                      part_results[p].ucounts.end(), counts.get() + uoff);
          }

          if (compute_inverse) {
            const size_t start = partition_offsets[p];
            const size_t end = partition_offsets[p + 1];
            for (size_t j = start; j < end; ++j) {
              inverse[part_orig_idx[j]] = part_local_uids[j] + uoff;
            }
          }
        }
      },
      1);

  return {unique_keys,   total_unique, inverse, compute_inverse ? n : 0,
          first_indices, counts};
}

template <typename KeyT>
inline UniqueResult<KeyT>
unique_keys_inverse(std::span<const KeyT> keys,
                    bool compute_first_indices = false,
                    bool compute_counts = false, size_t num_threads = 0,
                    bool compute_inverse = true) {
  if (keys.size() < kUniqueParallelThreshold || num_threads == 1) {
    return unique_keys_inverse_sequential(keys, compute_first_indices,
                                          compute_counts, compute_inverse);
  }
  return unique_keys_inverse_radix(keys, compute_first_indices, compute_counts,
                                   num_threads, compute_inverse);
}

} // namespace se::ops

#endif // SPARSE_EMBEDDING_UNIQUE_OPS_HPP_
