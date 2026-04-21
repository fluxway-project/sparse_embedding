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

#ifndef SPARSE_EMBEDDING_SIMD_OPS_HPP_
#define SPARSE_EMBEDDING_SIMD_OPS_HPP_

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#include "common/numeric_types.hpp"

namespace se::simd {

static constexpr size_t kCacheLineSize = 64;

#if defined(__AVX512F__)
static constexpr size_t kFloatSimdWidth = 16;
#elif defined(__AVX2__)
static constexpr size_t kFloatSimdWidth = 8;
#else
static constexpr size_t kFloatSimdWidth = 4;
#endif

template <typename ValT>
inline constexpr bool is_simd_supported_v = std::is_same_v<ValT, float>
#if defined(__AVX512BF16__)
                                            || std::is_same_v<ValT, bfloat16_t>
#endif
#if defined(__AVX512FP16__) || defined(__F16C__)
                                            || std::is_same_v<ValT, float16_t>
#endif
    ;

// SIMD Intrinsic Abstractions for Loading/Storing Float equivalents:
#if defined(__AVX512F__)
inline __m512 load_to_ps(const float *ptr) { return _mm512_loadu_ps(ptr); }
inline void store_from_ps(float *ptr, __m512 val) {
  _mm512_storeu_ps(ptr, val);
}

#if defined(__AVX512BF16__)
inline __m512 load_to_ps(const bfloat16_t *ptr) {
  __m256i packed = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(ptr));
  __m512i extended = _mm512_cvtepu16_epi32(packed);
  __m512i shifted = _mm512_slli_epi32(extended, 16);
  return _mm512_castsi512_ps(shifted);
}
inline void store_from_ps(bfloat16_t *ptr, __m512 val) {
  __m256bh bh = _mm512_cvtneps_pbh(val);
  _mm256_storeu_si256(reinterpret_cast<__m256i *>(ptr),
                      reinterpret_cast<__m256i>(bh));
}
#endif

#if defined(__AVX512FP16__) || defined(__F16C__)
inline __m512 load_to_ps(const float16_t *ptr) {
  return _mm512_cvtph_ps(
      _mm256_loadu_si256(reinterpret_cast<const __m256i *>(ptr)));
}
inline void store_from_ps(float16_t *ptr, __m512 val) {
  _mm256_storeu_si256(
      reinterpret_cast<__m256i *>(ptr),
      _mm512_cvtps_ph(val, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
}
#endif
#endif // __AVX512F__

#if defined(__AVX2__)
inline __m256 load256_to_ps(const float *ptr) { return _mm256_loadu_ps(ptr); }
inline void store256_from_ps(float *ptr, __m256 val) {
  _mm256_storeu_ps(ptr, val);
}

#if defined(__F16C__)
inline __m256 load256_to_ps(const float16_t *ptr) {
  return _mm256_cvtph_ps(
      _mm_loadu_si128(reinterpret_cast<const __m128i *>(ptr)));
}
inline void store256_from_ps(float16_t *ptr, __m256 val) {
  _mm_storeu_si128(
      reinterpret_cast<__m128i *>(ptr),
      _mm256_cvtps_ph(val, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
}
#endif
#endif // __AVX2__

inline void prefetch_l1(const void *addr) {
#if defined(__x86_64__)
  _mm_prefetch(reinterpret_cast<const char *>(addr), _MM_HINT_T0);
#endif
}

inline void prefetch_l2(const void *addr) {
#if defined(__x86_64__)
  _mm_prefetch(reinterpret_cast<const char *>(addr), _MM_HINT_T1);
#endif
}

template <typename ValT>
inline void simd_copy(const ValT *__restrict__ src, ValT *__restrict__ dst,
                      size_t count) {
  size_t i = 0;
#if defined(__AVX512F__)
  if constexpr (is_simd_supported_v<ValT>) {
    for (; i + 16 <= count; i += 16) {
      store_from_ps(dst + i, load_to_ps(src + i));
    }
  }
#elif defined(__AVX2__)
  if constexpr (std::is_same_v<ValT, float>
#if defined(__F16C__)
                || std::is_same_v<ValT, float16_t>
#endif
  ) {
    for (; i + 8 <= count; i += 8) {
      store256_from_ps(dst + i, load256_to_ps(src + i));
    }
  }
#endif
  for (; i < count; ++i) {
    dst[i] = src[i];
  }
}

template <typename ValT>
inline void simd_accumulate(const ValT *__restrict__ src,
                            ValT *__restrict__ dst, size_t count) {
  size_t i = 0;
#if defined(__AVX512F__)
  if constexpr (is_simd_supported_v<ValT>) {
    for (; i + 16 <= count; i += 16) {
      store_from_ps(dst + i,
                    _mm512_add_ps(load_to_ps(dst + i), load_to_ps(src + i)));
    }
  }
#elif defined(__AVX2__)
  if constexpr (std::is_same_v<ValT, float>
#if defined(__F16C__)
                || std::is_same_v<ValT, float16_t>
#endif
  ) {
    for (; i + 8 <= count; i += 8) {
      store256_from_ps(dst + i, _mm256_add_ps(load256_to_ps(dst + i),
                                              load256_to_ps(src + i)));
    }
  }
#endif
  for (; i < count; ++i) {
    if constexpr (std::is_same_v<ValT, float>) {
      dst[i] += src[i];
    } else {
      dst[i] = static_cast<ValT>(static_cast<float>(dst[i]) +
                                 static_cast<float>(src[i]));
    }
  }
}

template <typename ValT>
inline void simd_adam_update(ValT *__restrict__ param,
                             const float *__restrict__ grad,
                             float *__restrict__ m, float *__restrict__ v,
                             float lr, float beta1, float beta2, float eps,
                             float bc1, float bc2, size_t count) {
  size_t i = 0;
#if defined(__AVX512F__)
  if constexpr (is_simd_supported_v<ValT>) {
    __m512 vb1 = _mm512_set1_ps(beta1);
    __m512 vb2 = _mm512_set1_ps(beta2);
    __m512 v1mb1 = _mm512_set1_ps(1.0f - beta1);
    __m512 v1mb2 = _mm512_set1_ps(1.0f - beta2);
    __m512 vlr = _mm512_set1_ps(-lr);
    __m512 veps = _mm512_set1_ps(eps);
    __m512 vbc1 = _mm512_set1_ps(bc1);
    __m512 vbc2 = _mm512_set1_ps(bc2);

    for (; i + 16 <= count; i += 16) {
      __m512 g = _mm512_loadu_ps(grad + i);
      __m512 mi = _mm512_loadu_ps(m + i);
      __m512 vi = _mm512_loadu_ps(v + i);

      mi = _mm512_fmadd_ps(vb1, mi, _mm512_mul_ps(v1mb1, g));
      vi = _mm512_fmadd_ps(vb2, vi, _mm512_mul_ps(v1mb2, _mm512_mul_ps(g, g)));

      _mm512_storeu_ps(m + i, mi);
      _mm512_storeu_ps(v + i, vi);

      __m512 m_hat = _mm512_mul_ps(mi, vbc1);
      __m512 v_hat = _mm512_mul_ps(vi, vbc2);
      __m512 denom = _mm512_add_ps(_mm512_sqrt_ps(v_hat), veps);
      __m512 p = load_to_ps(param + i);
      p = _mm512_fmadd_ps(vlr, _mm512_div_ps(m_hat, denom), p);
      store_from_ps(param + i, p);
    }
  }
#elif defined(__AVX2__)
  if constexpr (std::is_same_v<ValT, float>
#if defined(__F16C__)
                || std::is_same_v<ValT, float16_t>
#endif
  ) {
    __m256 vb1 = _mm256_set1_ps(beta1);
    __m256 vb2 = _mm256_set1_ps(beta2);
    __m256 v1mb1 = _mm256_set1_ps(1.0f - beta1);
    __m256 v1mb2 = _mm256_set1_ps(1.0f - beta2);
    __m256 vlr = _mm256_set1_ps(-lr);
    __m256 veps = _mm256_set1_ps(eps);
    __m256 vbc1 = _mm256_set1_ps(bc1);
    __m256 vbc2 = _mm256_set1_ps(bc2);

    for (; i + 8 <= count; i += 8) {
      __m256 g = _mm256_loadu_ps(grad + i);
      __m256 mi = _mm256_loadu_ps(m + i);
      __m256 vi = _mm256_loadu_ps(v + i);

      mi = _mm256_fmadd_ps(vb1, mi, _mm256_mul_ps(v1mb1, g));
      vi = _mm256_fmadd_ps(vb2, vi, _mm256_mul_ps(v1mb2, _mm256_mul_ps(g, g)));

      _mm256_storeu_ps(m + i, mi);
      _mm256_storeu_ps(v + i, vi);

      __m256 m_hat = _mm256_mul_ps(mi, vbc1);
      __m256 v_hat = _mm256_mul_ps(vi, vbc2);
      __m256 denom = _mm256_add_ps(_mm256_sqrt_ps(v_hat), veps);
      __m256 p = load256_to_ps(param + i);
      p = _mm256_fmadd_ps(vlr, _mm256_div_ps(m_hat, denom), p);
      store256_from_ps(param + i, p);
    }
  }
#endif
  for (; i < count; ++i) {
    m[i] = beta1 * m[i] + (1.0f - beta1) * grad[i];
    v[i] = beta2 * v[i] + (1.0f - beta2) * grad[i] * grad[i];
    float m_hat = m[i] * bc1;
    float v_hat = v[i] * bc2;
    if constexpr (std::is_same_v<ValT, float>) {
      param[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);
    } else {
      param[i] = static_cast<ValT>(static_cast<float>(param[i]) -
                                   lr * m_hat / (std::sqrt(v_hat) + eps));
    }
  }
}

template <typename ValT>
inline void simd_uniform_init(ValT *__restrict__ data, size_t count,
                              float min_val, float max_val, uint64_t key) {
  size_t i = 0;
#if defined(__AVX512F__)
  if constexpr (is_simd_supported_v<ValT>) {
    uint32_t seed = static_cast<uint32_t>(key ^ (key >> 32));
    __m512i base_idx =
        _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
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
#elif defined(__AVX2__)
  if constexpr (std::is_same_v<ValT, float>
#if defined(__F16C__)
                || std::is_same_v<ValT, float16_t>
#endif
  ) {
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
      h = _mm256_xor_epi32(h, _mm256_srli_epi32(h, 16));
      h = _mm256_mullo_epi32(h, m1);
      h = _mm256_xor_epi32(h, _mm256_srli_epi32(h, 13));
      h = _mm256_mullo_epi32(h, m2);
      h = _mm256_xor_epi32(h, _mm256_srli_epi32(h, 16));

      __m256i shf = _mm256_srli_epi32(h, 9);
      __m256i float_bits = _mm256_or_epi32(shf, magic);
      __m256 floats = _mm256_castsi256_ps(float_bits);
      floats = _mm256_sub_ps(floats, _mm256_set1_ps(1.0f));

      __m256 result = _mm256_fmadd_ps(floats, diff, min_v);
      store256_from_ps(data + i, result);

      current_idx = _mm256_add_epi32(current_idx, stride);
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
    float floats;
    std::memcpy(&floats, &float_bits, sizeof(float));
    if constexpr (std::is_same_v<ValT, float>) {
      data[i] = (floats - 1.0f) * (max_val - min_val) + min_val;
    } else {
      data[i] =
          static_cast<ValT>((floats - 1.0f) * (max_val - min_val) + min_val);
    }
  }
}

} // namespace se::simd

#endif // SPARSE_EMBEDDING_SIMD_OPS_HPP_
