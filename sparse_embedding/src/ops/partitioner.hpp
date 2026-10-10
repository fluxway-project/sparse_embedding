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

#ifndef SPARSE_EMBEDDING_PARTITIONER_HPP_
#define SPARSE_EMBEDDING_PARTITIONER_HPP_

#include <proxy/proxy.h>

#include <cstdint>
#include <span>
#include <stdexcept>

#include "sparse_embedding/src/common/dtypes.hpp"

#if defined(__AVX512F__) && defined(__AVX512DQ__)
#include <immintrin.h>
#endif

namespace se {
namespace ops {

inline uint32_t fast_range_64(uint64_t word, uint32_t p) {
  return static_cast<uint32_t>((static_cast<unsigned __int128>(word) * p) >>
                               64);
}

inline uint32_t fast_range_32(uint32_t word, uint32_t p) {
  return static_cast<uint32_t>((static_cast<uint64_t>(word) * p) >> 32);
}

PRO_DEF_MEM_DISPATCH(partition_span_dispatch, partition_span);
PRO_DEF_MEM_DISPATCH(filter_span_dispatch, filter_span);

struct PartitionerFacade
    : pro::facade_builder //
      ::add_convention<partition_span_dispatch, void(std::span<const uint64_t>,
                                                     std::span<uint32_t>)> //
      ::add_convention<partition_span_dispatch,
                       void(std::span<const int64_t>, std::span<uint32_t>)> //
      ::add_convention<partition_span_dispatch, void(std::span<const uint32_t>,
                                                     std::span<uint32_t>)> //
      ::add_convention<partition_span_dispatch, void(std::span<const int32_t>,
                                                     std::span<uint32_t>)> //
      ::add_convention<filter_span_dispatch,
                       void(std::span<const uint64_t>, uint32_t,
                            std::span<bool>)> //
      ::add_convention<filter_span_dispatch,
                       void(std::span<const int64_t>, uint32_t,
                            std::span<bool>)> //
      ::add_convention<filter_span_dispatch,
                       void(std::span<const uint32_t>, uint32_t,
                            std::span<bool>)> //
      ::add_convention<filter_span_dispatch,
                       void(std::span<const int32_t>, uint32_t,
                            std::span<bool>)> //
      ::build {};

class ModuloPartitioner {
  uint32_t num_shards_;
  bool is_power_of_two_;

  template <typename KeyT> uint32_t normalized_modulo(KeyT key) const {
    return static_cast<uint32_t>(modulo_partition(key, num_shards_));
  }

  template <typename KeyT> uint32_t shard_for(KeyT key) const {
    return normalized_modulo(key);
  }

public:
  ModuloPartitioner(uint32_t num_shards) : num_shards_(num_shards) {
    if (num_shards_ == 0) {
      throw std::invalid_argument("num_shards must be > 0");
    }
    is_power_of_two_ = (num_shards_ & (num_shards_ - 1)) == 0;
  }

  template <typename KeyT>
  void partition_span(std::span<const KeyT> keys,
                      std::span<uint32_t> out_shard_ids) const {
    size_t n = keys.size();
    size_t i = 0;
#if defined(__AVX512F__) && defined(__AVX512DQ__)
    if (is_power_of_two_) {
      if constexpr (sizeof(KeyT) == 8) {
        __m512i v_mask = _mm512_set1_epi64(num_shards_ - 1);
        for (; i + 8 <= n; i += 8) {
          __m512i v_keys =
              _mm512_loadu_si512(reinterpret_cast<const void *>(&keys[i]));
          __m512i v_mod = _mm512_and_si512(v_keys, v_mask);
          __m256i v_res = _mm512_cvtepi64_epi32(v_mod);
          _mm256_storeu_si256(reinterpret_cast<__m256i *>(&out_shard_ids[i]),
                              v_res);
        }
      } else {
        __m512i v_mask = _mm512_set1_epi32(num_shards_ - 1);
        for (; i + 16 <= n; i += 16) {
          __m512i v_keys =
              _mm512_loadu_si512(reinterpret_cast<const void *>(&keys[i]));
          __m512i v_mod = _mm512_and_si512(v_keys, v_mask);
          _mm512_storeu_si512(reinterpret_cast<void *>(&out_shard_ids[i]),
                              v_mod);
        }
      }
    }
#endif
    if (is_power_of_two_) {
      uint64_t mask = num_shards_ - 1;
      for (; i < n; ++i) {
        out_shard_ids[i] =
            static_cast<uint32_t>(static_cast<uint64_t>(keys[i]) & mask);
      }
    } else {
      for (; i < n; ++i) {
        out_shard_ids[i] = shard_for(keys[i]);
      }
    }
  }

  template <typename KeyT>
  void filter_span(std::span<const KeyT> keys, uint32_t target_shard,
                   std::span<bool> out_mask) const {
    size_t n = keys.size();
    size_t i = 0;
#if defined(__AVX512F__) && defined(__AVX512DQ__)
    if (is_power_of_two_) {
      if constexpr (sizeof(KeyT) == 8) {
        __m512i v_shard_id = _mm512_set1_epi64(target_shard);
        __m512i v_mask = _mm512_set1_epi64(num_shards_ - 1);
        for (; i + 8 <= n; i += 8) {
          __m512i v_keys =
              _mm512_loadu_si512(reinterpret_cast<const void *>(&keys[i]));
          __m512i v_mod = _mm512_and_si512(v_keys, v_mask);
          __mmask8 cmp = _mm512_cmpeq_epi64_mask(v_mod, v_shard_id);
          for (int j = 0; j < 8; ++j) {
            out_mask[i + j] = (cmp & (1 << j)) != 0;
          }
        }
      } else {
        __m512i v_shard_id = _mm512_set1_epi32(target_shard);
        __m512i v_mask = _mm512_set1_epi32(num_shards_ - 1);
        for (; i + 16 <= n; i += 16) {
          __m512i v_keys =
              _mm512_loadu_si512(reinterpret_cast<const void *>(&keys[i]));
          __m512i v_mod = _mm512_and_si512(v_keys, v_mask);
          __mmask16 cmp = _mm512_cmpeq_epi32_mask(v_mod, v_shard_id);
          for (int j = 0; j < 16; ++j) {
            out_mask[i + j] = (cmp & (1 << j)) != 0;
          }
        }
      }
    }
#endif
    if (is_power_of_two_) {
      uint64_t mask = num_shards_ - 1;
      for (; i < n; ++i) {
        out_mask[i] = (static_cast<uint64_t>(keys[i]) & mask) == target_shard;
      }
    } else {
      for (; i < n; ++i) {
        out_mask[i] = shard_for(keys[i]) == target_shard;
      }
    }
  }
};

class FastRangePartitioner {
  uint32_t num_shards_;

public:
  FastRangePartitioner(uint32_t num_shards) : num_shards_(num_shards) {
    if (num_shards_ == 0) {
      throw std::invalid_argument("num_shards must be > 0");
    }
  }

  template <typename KeyT>
  void partition_span(std::span<const KeyT> keys,
                      std::span<uint32_t> out_shard_ids) const {
    size_t n = keys.size();
    for (size_t i = 0; i < n; ++i) {
      if constexpr (sizeof(KeyT) == 8) {
        out_shard_ids[i] =
            fast_range_64(static_cast<uint64_t>(keys[i]), num_shards_);
      } else {
        out_shard_ids[i] =
            fast_range_32(static_cast<uint32_t>(keys[i]), num_shards_);
      }
    }
  }

  template <typename KeyT>
  void filter_span(std::span<const KeyT> keys, uint32_t target_shard,
                   std::span<bool> out_mask) const {
    size_t n = keys.size();
    for (size_t i = 0; i < n; ++i) {
      if constexpr (sizeof(KeyT) == 8) {
        out_mask[i] = fast_range_64(static_cast<uint64_t>(keys[i]),
                                    num_shards_) == target_shard;
      } else {
        out_mask[i] = fast_range_32(static_cast<uint32_t>(keys[i]),
                                    num_shards_) == target_shard;
      }
    }
  }
};

class HashPartitioner {
  uint32_t num_shards_;
  bool is_power_of_two_;

  struct SplitMix64Hash {
    size_t operator()(uint64_t x) const {
      x ^= x >> 30;
      x *= 0xbf58476d1ce4e5b9ull;
      x ^= x >> 27;
      x *= 0x94d049bb133111ebull;
      x ^= x >> 31;
      return x;
    }

#if defined(__AVX512F__) && defined(__AVX512DQ__)
    __m512i simd_hash(__m512i x) const {
      x = _mm512_xor_si512(x, _mm512_srli_epi64(x, 30));
      __m512i c1 = _mm512_set1_epi64(0xbf58476d1ce4e5b9ull);
      x = _mm512_mullo_epi64(x, c1);
      x = _mm512_xor_si512(x, _mm512_srli_epi64(x, 27));
      __m512i c2 = _mm512_set1_epi64(0x94d049bb133111ebull);
      x = _mm512_mullo_epi64(x, c2);
      x = _mm512_xor_si512(x, _mm512_srli_epi64(x, 31));
      return x;
    }
#endif
  };

  struct Murmur32Hash {
    uint32_t operator()(uint32_t x) const {
      x ^= x >> 16;
      x *= 0x85ebca6b;
      x ^= x >> 13;
      x *= 0xc2b2ae35;
      x ^= x >> 16;
      return x;
    }

#if defined(__AVX512F__) && defined(__AVX512DQ__)
    __m512i simd_hash(__m512i x) const {
      x = _mm512_xor_si512(x, _mm512_srli_epi32(x, 16));
      __m512i c1 = _mm512_set1_epi32(0x85ebca6b);
      x = _mm512_mullo_epi32(x, c1);
      x = _mm512_xor_si512(x, _mm512_srli_epi32(x, 13));
      __m512i c2 = _mm512_set1_epi32(0xc2b2ae35);
      x = _mm512_mullo_epi32(x, c2);
      x = _mm512_xor_si512(x, _mm512_srli_epi32(x, 16));
      return x;
    }
#endif
  };

public:
  HashPartitioner(uint32_t num_shards) : num_shards_(num_shards) {
    if (num_shards_ == 0) {
      throw std::invalid_argument("num_shards must be > 0");
    }
    is_power_of_two_ = (num_shards_ & (num_shards_ - 1)) == 0;
  }

  template <typename KeyT>
  void partition_span(std::span<const KeyT> keys,
                      std::span<uint32_t> out_shard_ids) const {
    size_t n = keys.size();
    size_t i = 0;
    SplitMix64Hash hasher;

#if defined(__AVX512F__) && defined(__AVX512DQ__)
    if (is_power_of_two_) {
      if constexpr (sizeof(KeyT) == 8) {
        __m512i v_mask = _mm512_set1_epi64(num_shards_ - 1);
        for (; i + 8 <= n; i += 8) {
          __m512i v_keys =
              _mm512_loadu_si512(reinterpret_cast<const void *>(&keys[i]));
          __m512i v_hash = hasher.simd_hash(v_keys);
          __m512i v_mod = _mm512_and_si512(v_hash, v_mask);
          __m256i v_res = _mm512_cvtepi64_epi32(v_mod);
          _mm256_storeu_si256(reinterpret_cast<__m256i *>(&out_shard_ids[i]),
                              v_res);
        }
      } else {
        Murmur32Hash hasher32;
        __m512i v_mask = _mm512_set1_epi32(num_shards_ - 1);
        for (; i + 16 <= n; i += 16) {
          __m512i v_keys =
              _mm512_loadu_si512(reinterpret_cast<const void *>(&keys[i]));
          __m512i v_hash = hasher32.simd_hash(v_keys);
          __m512i v_mod = _mm512_and_si512(v_hash, v_mask);
          _mm512_storeu_si512(reinterpret_cast<void *>(&out_shard_ids[i]),
                              v_mod);
        }
      }
    }
#endif
    if (is_power_of_two_) {
      uint64_t mask = num_shards_ - 1;
      for (; i < n; ++i) {
        if constexpr (sizeof(KeyT) == 8) {
          out_shard_ids[i] = static_cast<uint32_t>(
              hasher(static_cast<uint64_t>(keys[i])) & mask);
        } else {
          Murmur32Hash hasher32;
          out_shard_ids[i] = static_cast<uint32_t>(
              hasher32(static_cast<uint32_t>(keys[i])) & mask);
        }
      }
    } else {
      for (; i < n; ++i) {
        if constexpr (sizeof(KeyT) == 8) {
          out_shard_ids[i] = fast_range_64(
              hasher(static_cast<uint64_t>(keys[i])), num_shards_);
        } else {
          Murmur32Hash hasher32;
          out_shard_ids[i] = fast_range_32(
              hasher32(static_cast<uint32_t>(keys[i])), num_shards_);
        }
      }
    }
  }

  template <typename KeyT>
  void filter_span(std::span<const KeyT> keys, uint32_t target_shard,
                   std::span<bool> out_mask) const {
    size_t n = keys.size();
    size_t i = 0;
    SplitMix64Hash hasher;

#if defined(__AVX512F__) && defined(__AVX512DQ__)
    if (is_power_of_two_) {
      if constexpr (sizeof(KeyT) == 8) {
        __m512i v_shard_id = _mm512_set1_epi64(target_shard);
        __m512i v_mask = _mm512_set1_epi64(num_shards_ - 1);
        for (; i + 8 <= n; i += 8) {
          __m512i v_keys =
              _mm512_loadu_si512(reinterpret_cast<const void *>(&keys[i]));
          __m512i v_hash = hasher.simd_hash(v_keys);
          __m512i v_mod = _mm512_and_si512(v_hash, v_mask);
          __mmask8 cmp = _mm512_cmpeq_epi64_mask(v_mod, v_shard_id);
          for (int j = 0; j < 8; ++j) {
            out_mask[i + j] = (cmp & (1 << j)) != 0;
          }
        }
      } else {
        Murmur32Hash hasher32;
        __m512i v_shard_id = _mm512_set1_epi32(target_shard);
        __m512i v_mask = _mm512_set1_epi32(num_shards_ - 1);
        for (; i + 16 <= n; i += 16) {
          __m512i v_keys =
              _mm512_loadu_si512(reinterpret_cast<const void *>(&keys[i]));
          __m512i v_hash = hasher32.simd_hash(v_keys);
          __m512i v_mod = _mm512_and_si512(v_hash, v_mask);
          __mmask16 cmp = _mm512_cmpeq_epi32_mask(v_mod, v_shard_id);
          for (int j = 0; j < 16; ++j) {
            out_mask[i + j] = (cmp & (1 << j)) != 0;
          }
        }
      }
    }
#endif
    if (is_power_of_two_) {
      uint64_t mask = num_shards_ - 1;
      for (; i < n; ++i) {
        if constexpr (sizeof(KeyT) == 8) {
          out_mask[i] =
              (hasher(static_cast<uint64_t>(keys[i])) & mask) == target_shard;
        } else {
          Murmur32Hash hasher32;
          out_mask[i] =
              (hasher32(static_cast<uint32_t>(keys[i])) & mask) == target_shard;
        }
      }
    } else {
      for (; i < n; ++i) {
        if constexpr (sizeof(KeyT) == 8) {
          out_mask[i] = fast_range_64(hasher(static_cast<uint64_t>(keys[i])),
                                      num_shards_) == target_shard;
        } else {
          Murmur32Hash hasher32;
          out_mask[i] = fast_range_32(hasher32(static_cast<uint32_t>(keys[i])),
                                      num_shards_) == target_shard;
        }
      }
    }
  }
};

} // namespace ops
} // namespace se

#endif // SPARSE_EMBEDDING_PARTITIONER_HPP_
