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

#ifndef SPARSE_EMBEDDING_SIMD_TRAITS_HPP_
#define SPARSE_EMBEDDING_SIMD_TRAITS_HPP_

#include <cstddef>
#include <type_traits>

#include "sparse_embedding/src/common/numeric_types.hpp"

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

} // namespace se::simd

#endif // SPARSE_EMBEDDING_SIMD_TRAITS_HPP_
