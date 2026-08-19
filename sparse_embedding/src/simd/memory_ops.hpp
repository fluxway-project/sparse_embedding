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

#ifndef SPARSE_EMBEDDING_SIMD_MEMORY_OPS_HPP_
#define SPARSE_EMBEDDING_SIMD_MEMORY_OPS_HPP_

#include <cstddef>
#include <cstring>
#include <type_traits>

#include "sparse_embedding/src/simd/intrinsics.hpp"
#include "sparse_embedding/src/simd/traits.hpp"

namespace se::simd {

template <typename ValT>
inline void simd_copy(const ValT *__restrict__ src, ValT *__restrict__ dst,
                      size_t count) {
  static_assert(std::is_trivially_copyable_v<ValT>);
  std::memcpy(dst, src, count * sizeof(ValT));
}

template <typename ValT>
inline void simd_accumulate(const ValT *__restrict__ src,
                            ValT *__restrict__ dst, size_t count) {
  size_t i = 0;
  [[assume(count > 0)]];
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

} // namespace se::simd

#endif // SPARSE_EMBEDDING_SIMD_MEMORY_OPS_HPP_
