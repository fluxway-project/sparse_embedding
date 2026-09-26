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

#ifndef SPARSE_EMBEDDING_SIMD_UNIFORM_INIT_OPS_HPP_
#define SPARSE_EMBEDDING_SIMD_UNIFORM_INIT_OPS_HPP_

#include <bit>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "sparse_embedding/src/simd/intrinsics.hpp"
#include "sparse_embedding/src/simd/traits.hpp"

namespace se::simd {

template <typename ValT>
inline void simd_uniform_init(ValT *__restrict__ data, size_t count,
                              float min_val, float max_val, uint64_t key) {

  static_cast<void>(static_cast<ValT>(min_val));
  static_cast<void>(static_cast<ValT>(max_val));
  size_t i = 0;
  [[assume(count > 0)]];
#if defined(__AVX512F__)
  if constexpr (is_simd_supported_v<ValT>) {
    {
      uint32_t seed = static_cast<uint32_t>(key ^ (key >> 32));
      __m512i base_idx = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5,
                                          4, 3, 2, 1, 0);
      __m512i vseed = _mm512_set1_epi32(seed);
      __m512i stride = _mm512_set1_epi32(16);
      __m512i current_idx = _mm512_add_epi32(vseed, base_idx);

      __m512i m1 = _mm512_set1_epi32(0x85ebca6b);
      __m512i m2 = _mm512_set1_epi32(0xc2b2ae35);

      __m512 diff = _mm512_set1_ps(max_val - min_val);
      __m512 min_v = _mm512_set1_ps(min_val);
      __m512i magic = _mm512_set1_epi32(0x3f800000);

      for (; i + 16 <= count; i += 16) {
        __m512i h = current_idx;
        h = _mm512_xor_epi32(h, _mm512_srli_epi32(h, 16));
        h = _mm512_mullo_epi32(h, m1);
        h = _mm512_xor_epi32(h, _mm512_srli_epi32(h, 13));
        h = _mm512_mullo_epi32(h, m2);
        h = _mm512_xor_epi32(h, _mm512_srli_epi32(h, 16));

        __m512i shf = _mm512_srli_epi32(h, 9);
        __m512i float_bits = _mm512_or_epi32(shf, magic);
        __m512 floats = _mm512_castsi512_ps(float_bits);
        floats = _mm512_sub_ps(floats, _mm512_set1_ps(1.0f));

        __m512 result = _mm512_fmadd_ps(floats, diff, min_v);
        store_from_ps(data + i, result);

        current_idx = _mm512_add_epi32(current_idx, stride);
      }
    }
  }
#elif defined(__AVX2__)
  if constexpr (std::is_same_v<ValT, float>
#if defined(__F16C__)
                || std::is_same_v<ValT, float16_t>
#endif
  ) {
    {
      uint32_t seed = static_cast<uint32_t>(key ^ (key >> 32));
      __m256i base_idx = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
      __m256i vseed = _mm256_set1_epi32(seed);
      __m256i stride = _mm256_set1_epi32(8);
      __m256i current_idx = _mm256_add_epi32(vseed, base_idx);

      __m256i m1 = _mm256_set1_epi32(0x85ebca6b);
      __m256i m2 = _mm256_set1_epi32(0xc2b2ae35);

      __m256 diff = _mm256_set1_ps(max_val - min_val);
      __m256 min_v = _mm256_set1_ps(min_val);
      __m256i magic = _mm256_set1_epi32(0x3f800000);

      for (; i + 8 <= count; i += 8) {
        __m256i h = current_idx;
        h = _mm256_xor_si256(h, _mm256_srli_epi32(h, 16));
        h = _mm256_mullo_epi32(h, m1);
        h = _mm256_xor_si256(h, _mm256_srli_epi32(h, 13));
        h = _mm256_mullo_epi32(h, m2);
        h = _mm256_xor_si256(h, _mm256_srli_epi32(h, 16));

        __m256i shf = _mm256_srli_epi32(h, 9);
        __m256i float_bits = _mm256_or_si256(shf, magic);
        __m256 floats = _mm256_castsi256_ps(float_bits);
        floats = _mm256_sub_ps(floats, _mm256_set1_ps(1.0f));

        __m256 result = _mm256_fmadd_ps(floats, diff, min_v);
        store256_from_ps(data + i, result);

        current_idx = _mm256_add_epi32(current_idx, stride);
      }
    }
  }
#endif
  for (; i < count; ++i) {
    uint32_t h = static_cast<uint32_t>(key ^ (key >> 32)) + i;
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    uint32_t shf = h >> 9;
    uint32_t float_bits = shf | 0x3f800000;
    float floats = std::bit_cast<float>(float_bits);
    if constexpr (std::is_same_v<ValT, float>) {
      data[i] = (floats - 1.0f) * (max_val - min_val) + min_val;
    } else {
      data[i] =
          static_cast<ValT>((floats - 1.0f) * (max_val - min_val) + min_val);
    }
  }
}

} // namespace se::simd

#endif // SPARSE_EMBEDDING_SIMD_UNIFORM_INIT_OPS_HPP_
