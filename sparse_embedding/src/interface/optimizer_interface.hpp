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

#ifndef SPARSE_EMBEDDING_OPTIMIZER_INTERFACE_HPP_
#define SPARSE_EMBEDDING_OPTIMIZER_INTERFACE_HPP_

#include <proxy/proxy.h>

#include <cstdint>
#include <span>

#include "sparse_embedding/src/common/numeric_types.hpp"

namespace se {

PRO_DEF_MEM_DISPATCH(OptMemAllocate, allocate);
PRO_DEF_MEM_DISPATCH(OptMemDeallocate, deallocate);
PRO_DEF_MEM_DISPATCH(OptMemClear, clear);
PRO_DEF_MEM_DISPATCH(OptMemStep, step);

template <typename ValT>
struct OptimizerFacade
    : pro::facade_builder ::add_convention<OptMemAllocate, void()>::
          add_convention<pro::weak_dispatch<OptMemDeallocate>,
                         void(uint32_t idx)>::
              add_convention<pro::weak_dispatch<OptMemClear>, void()>::
                  add_convention<OptMemStep,
                                 void(uint32_t idx, std::span<ValT> param,
                                      std::span<const ValT> grad,
                                      uint64_t global_step)>::build {};

PRO_DEF_MEM_DISPATCH(MemBuildOptF32, build_f32);
PRO_DEF_MEM_DISPATCH(MemBuildOptF16, build_f16);
PRO_DEF_MEM_DISPATCH(MemBuildOptBF16, build_bf16);

struct OptimizerBuilderFacade
    : pro::facade_builder ::add_convention<
          MemBuildOptF32, pro::proxy<OptimizerFacade<float>>(size_t)>::
          add_convention<MemBuildOptF16,
                         pro::proxy<OptimizerFacade<se::float16_t>>(size_t)>::
              add_convention<MemBuildOptBF16,
                             pro::proxy<OptimizerFacade<se::bfloat16_t>>(
                                 size_t)>::build {};

} // namespace se

#endif // SPARSE_EMBEDDING_OPTIMIZER_INTERFACE_HPP_
