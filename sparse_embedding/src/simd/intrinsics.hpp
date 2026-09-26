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

#ifndef SPARSE_EMBEDDING_SIMD_INTRINSICS_HPP_
#define SPARSE_EMBEDDING_SIMD_INTRINSICS_HPP_

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#include <array>

#include "sparse_embedding/src/common/numeric_types.hpp"

namespace se::simd {

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

#if defined(__AVX512F__)
template <typename ValT> inline void store_from_ps(ValT *ptr, __m512 val) {
  alignas(64) std::array<float, 16> lanes{};
  _mm512_store_ps(lanes.data(), val);
  for (float lane : lanes)
    static_cast<void>(static_cast<ValT>(lane));
  store_from_ps(ptr, val);
}
#endif

#if defined(__AVX2__)
template <typename ValT> inline void store256_from_ps(ValT *ptr, __m256 val) {
  alignas(32) std::array<float, 8> lanes{};
  _mm256_store_ps(lanes.data(), val);
  for (float lane : lanes)
    static_cast<void>(static_cast<ValT>(lane));
  store256_from_ps(ptr, val);
}
#endif

} // namespace se::simd

#endif // SPARSE_EMBEDDING_SIMD_INTRINSICS_HPP_
