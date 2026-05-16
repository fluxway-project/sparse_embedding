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

#ifndef SPARSE_EMBEDDING_INITIALIZER_INTERFACE_HPP_
#define SPARSE_EMBEDDING_INITIALIZER_INTERFACE_HPP_

#include <proxy/proxy.h>

#include <cstdint>
#include <span>

#include "sparse_embedding/src/common/numeric_types.hpp"

namespace se {

PRO_DEF_MEM_DISPATCH(MemInitialize, initialize);

template <typename ValT>
struct InitializerFacade
    : pro::facade_builder //
      ::add_convention<MemInitialize,
                       void(std::span<ValT> param, uint64_t key)> //
      ::build {};

PRO_DEF_MEM_DISPATCH(MemBuildInitF32, build_f32);
PRO_DEF_MEM_DISPATCH(MemBuildInitF16, build_f16);
PRO_DEF_MEM_DISPATCH(MemBuildInitBF16, build_bf16);

struct InitializerBuilderFacade
    : pro::facade_builder //
      ::add_convention<MemBuildInitF32,
                       pro::proxy<InitializerFacade<float>>()> //
      ::add_convention<MemBuildInitF16,
                       pro::proxy<InitializerFacade<se::float16_t>>()> //
      ::add_convention<MemBuildInitBF16,
                       pro::proxy<InitializerFacade<se::bfloat16_t>>()> //
      ::build {};

} // namespace se

#endif // SPARSE_EMBEDDING_INITIALIZER_INTERFACE_HPP_
