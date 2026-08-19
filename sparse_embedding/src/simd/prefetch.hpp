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

#ifndef SPARSE_EMBEDDING_SIMD_PREFETCH_HPP_
#define SPARSE_EMBEDDING_SIMD_PREFETCH_HPP_

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace se::simd {

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

} // namespace se::simd

#endif // SPARSE_EMBEDDING_SIMD_PREFETCH_HPP_
