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
PRO_DEF_MEM_DISPATCH(OptMemReserve, reserve);
PRO_DEF_MEM_DISPATCH(OptMemDeallocate, deallocate);
PRO_DEF_MEM_DISPATCH(OptMemClear, clear);
PRO_DEF_MEM_DISPATCH(OptMemStep, step);
PRO_DEF_MEM_DISPATCH(OptMemStateData, state_data);
PRO_DEF_MEM_DISPATCH(OptMemStateDim, state_dim);
PRO_DEF_MEM_DISPATCH(OptMemSetState, set_state);
PRO_DEF_MEM_DISPATCH(OptMemStepData, step_data);
PRO_DEF_MEM_DISPATCH(OptMemSetStep, set_step);

template <typename ParamT, typename OptT>
struct OptimizerFacade
    : pro::facade_builder                           //
      ::add_convention<OptMemAllocate, void()>      //
      ::add_convention<OptMemReserve, void(size_t)> //
      ::add_convention<pro::weak_dispatch<OptMemDeallocate>,
                       void(uint32_t idx)>::
          add_convention<pro::weak_dispatch<OptMemClear>, void()> //
      ::add_convention<OptMemStep, void(uint32_t idx, std::span<ParamT> param,
                                        std::span<const ParamT> grad,
                                        uint64_t global_step)> //
      ::add_convention<pro::weak_dispatch<OptMemStateData>,
                       void(uint32_t idx, std::span<OptT> out) const> //
      ::add_convention<pro::weak_dispatch<OptMemStateDim>,
                       size_t() const> //
      ::add_convention<pro::weak_dispatch<OptMemSetState>,
                       void(uint32_t idx, std::span<const OptT> state)> //
      ::add_convention<pro::weak_dispatch<OptMemStepData>,
                       uint32_t(uint32_t idx) const> //
      ::add_convention<pro::weak_dispatch<OptMemSetStep>,
                       void(uint32_t idx, uint32_t step)> //
      ::build {};

PRO_DEF_MEM_DISPATCH(MemBuildOpt_f32_f32, build_f32_f32);
PRO_DEF_MEM_DISPATCH(MemBuildOpt_f32_f16, build_f32_f16);
PRO_DEF_MEM_DISPATCH(MemBuildOpt_f32_bf16, build_f32_bf16);
PRO_DEF_MEM_DISPATCH(MemBuildOpt_f16_f32, build_f16_f32);
PRO_DEF_MEM_DISPATCH(MemBuildOpt_f16_f16, build_f16_f16);
PRO_DEF_MEM_DISPATCH(MemBuildOpt_f16_bf16, build_f16_bf16);
PRO_DEF_MEM_DISPATCH(MemBuildOpt_bf16_f32, build_bf16_f32);
PRO_DEF_MEM_DISPATCH(MemBuildOpt_bf16_f16, build_bf16_f16);
PRO_DEF_MEM_DISPATCH(MemBuildOpt_bf16_bf16, build_bf16_bf16);

struct OptimizerBuilderFacade
    : pro::facade_builder //
      ::add_convention<MemBuildOpt_f32_f32,
                       pro::proxy<OptimizerFacade<float, float>>(size_t)> //
      ::add_convention<MemBuildOpt_f32_f16,
                       pro::proxy<OptimizerFacade<float, se::float16_t>>(
                           size_t)> //
      ::add_convention<MemBuildOpt_f32_bf16,
                       pro::proxy<OptimizerFacade<float, se::bfloat16_t>>(
                           size_t)> //
      ::add_convention<MemBuildOpt_f16_f32,
                       pro::proxy<OptimizerFacade<se::float16_t, float>>(
                           size_t)> //
      ::add_convention<
          MemBuildOpt_f16_f16,
          pro::proxy<OptimizerFacade<se::float16_t, se::float16_t>>(size_t)> //
      ::add_convention<
          MemBuildOpt_f16_bf16,
          pro::proxy<OptimizerFacade<se::float16_t, se::bfloat16_t>>(size_t)> //
      ::add_convention<MemBuildOpt_bf16_f32,
                       pro::proxy<OptimizerFacade<se::bfloat16_t, float>>(
                           size_t)> //
      ::add_convention<
          MemBuildOpt_bf16_f16,
          pro::proxy<OptimizerFacade<se::bfloat16_t, se::float16_t>>(size_t)> //
      ::add_convention<
          MemBuildOpt_bf16_bf16,
          pro::proxy<OptimizerFacade<se::bfloat16_t, se::bfloat16_t>>(
              size_t)> //
      ::build {};

} // namespace se

#endif // SPARSE_EMBEDDING_OPTIMIZER_INTERFACE_HPP_
