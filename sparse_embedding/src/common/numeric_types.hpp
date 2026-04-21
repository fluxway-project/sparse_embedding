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

#ifndef SPARSE_EMBEDDING_NUMERIC_TYPES_HPP_
#define SPARSE_EMBEDDING_NUMERIC_TYPES_HPP_

// ---------------------------------------------------------------------------
// Half/bfloat16 type abstraction
//
// Priority order:
//   1. C++23 <stdfloat>  — std::float16_t / std::bfloat16_t
//      Guard: __has_include(<stdfloat>) && C++ standard draft >= C++20
//      GCC 13 reports __cplusplus = 202100L for -std=c++23 (draft value),
//      so we use >= 202002L (C++20 published) as the floor: if the compiler
//      is new enough to ship <stdfloat> it is at least targeting C++20.
//   2. GCC/Clang extension builtins  — _Float16 / __bf16
//      Detection: __FLT16_MAX__ for float16, __BFLT16_MAX__ for bfloat16.
//      Available even when targeting C++17/20 on GCC ≥ 12 / Clang ≥ 15.
//   3. Hard error — forces an actionable upgrade message.
//
// Downstream code always uses se::float16_t / se::bfloat16_t.
// ---------------------------------------------------------------------------

#if __has_include(<stdfloat>) && __cplusplus >= 202002L
#include <stdfloat>
namespace se {
using float16_t = std::float16_t;
using bfloat16_t = std::bfloat16_t;
} // namespace se

#elif defined(__FLT16_MAX__) && (defined(__GNUC__) || defined(__clang__))
namespace se {
using float16_t = _Float16;
#if defined(__BFLT16_MAX__)
using bfloat16_t = __bf16;
#else
#error "bfloat16 not supported by this compiler; upgrade to GCC≥12 / Clang≥15"
#endif
} // namespace se

#else
#error                                                                         \
    "Half-precision types require C++23 <stdfloat> or GCC≥12/Clang≥15 compiler builtins"
#endif

#endif // SPARSE_EMBEDDING_NUMERIC_TYPES_HPP_
