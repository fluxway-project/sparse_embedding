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

#ifndef SPARSE_EMBEDDING_BINDINGS_DTYPE_DISPATCH_HPP_
#define SPARSE_EMBEDDING_BINDINGS_DTYPE_DISPATCH_HPP_

#include <dlpack/dlpack.h>

#include <cstdint>
#include <stdexcept>

#include "sparse_embedding/src/common/dtypes.hpp"

namespace se::py::detail {

// Reuse core types from domain SSOT
using KeyDType = se::KeyDType;
using ValueDType = se::ValueDType;

using se::dispatch_dtype_pair;
using se::dispatch_key_type;
using se::dispatch_value_type;
using se::key_dtype_name;
using se::parse_key_dtype;
using se::parse_value_dtype;
using se::value_dtype_name;

// Legacy alias for value dispatch
template <typename Fn>
decltype(auto) dispatch_value_dtype(ValueDType dtype, Fn &&fn) {
  return se::dispatch_value_type(dtype, std::forward<Fn>(fn));
}

inline KeyDType detect_key_dtype(const DLManagedTensor *dlm) {
  const auto code = dlm->dl_tensor.dtype.code;
  const auto bits = dlm->dl_tensor.dtype.bits;
  if (code == static_cast<uint8_t>(kDLInt) && bits == 32)
    return KeyDType::kInt32;
  if (code == static_cast<uint8_t>(kDLUInt) && bits == 32)
    return KeyDType::kUInt32;
  if (code == static_cast<uint8_t>(kDLInt) && bits == 64)
    return KeyDType::kInt64;
  if (code == static_cast<uint8_t>(kDLUInt) && bits == 64)
    return KeyDType::kUInt64;
  throw std::invalid_argument("keys must be int32, uint32, int64 or uint64");
}

inline ValueDType detect_value_dtype(const DLManagedTensor *dlm) {
  const auto code = dlm->dl_tensor.dtype.code;
  const auto bits = dlm->dl_tensor.dtype.bits;
  if (code == static_cast<uint8_t>(kDLFloat) && bits == 32)
    return ValueDType::kFloat32;
  if (code == static_cast<uint8_t>(kDLFloat) && bits == 16)
    return ValueDType::kFloat16;
  if (code == static_cast<uint8_t>(kDLBfloat) && bits == 16)
    return ValueDType::kBFloat16;
  throw std::invalid_argument("values must be float32, float16 or bfloat16");
}

} // namespace se::py::detail

#endif // SPARSE_EMBEDDING_BINDINGS_DTYPE_DISPATCH_HPP_
