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

#include <span>

#include "sparse_embedding/src/common/numeric_types.hpp"

namespace se {

PRO_DEF_MEM_DISPATCH(OptMemAllocate, allocate);
PRO_DEF_MEM_DISPATCH(OptMemSetPageEvictionEnabled, set_page_eviction_enabled);
PRO_DEF_MEM_DISPATCH(OptMemCapacity, capacity);
PRO_DEF_MEM_DISPATCH(OptMemReserve, reserve);
PRO_DEF_MEM_DISPATCH(OptMemDeallocate, deallocate);
PRO_DEF_MEM_DISPATCH(OptMemClear, clear);
PRO_DEF_MEM_DISPATCH(OptMemStep, step);
PRO_DEF_MEM_DISPATCH(OptMemStateData, state_data);
PRO_DEF_MEM_DISPATCH(OptMemStateDim, state_dim);
PRO_DEF_MEM_DISPATCH(OptMemAllocationBytesPerEntry, allocation_bytes_per_entry);
PRO_DEF_MEM_DISPATCH(OptMemSetState, set_state);
PRO_DEF_MEM_DISPATCH(OptMemStepData, step_data);
PRO_DEF_MEM_DISPATCH(OptMemSetStep, set_step);
PRO_DEF_MEM_DISPATCH(OptMemRequiresScore, requires_score);
PRO_DEF_MEM_DISPATCH(OptMemPrefetch, prefetch);

template <typename ParamT, typename OptT>
struct OptimizerFacade
    : pro::facade_builder                              //
      ::add_convention<OptMemAllocate, void(uint32_t)> //
      ::add_convention<pro::weak_dispatch<OptMemSetPageEvictionEnabled>,
                       void(bool)>                     //
      ::add_convention<OptMemCapacity, size_t() const> //
      ::add_convention<OptMemReserve, void(size_t)>    //
      ::add_convention<pro::weak_dispatch<OptMemDeallocate>,
                       void(uint32_t idx)>::
          add_convention<pro::weak_dispatch<OptMemClear>, void()> //
      ::add_convention<OptMemStep,
                       void(uint32_t idx, std::span<ParamT> param,
                            std::span<const ParamT> grad, uint64_t global_step,
                            uint64_t score)>              //
      ::add_convention<OptMemRequiresScore, bool() const> //
      ::add_convention<pro::weak_dispatch<OptMemPrefetch>,
                       void(uint32_t idx)> //
      ::add_convention<pro::weak_dispatch<OptMemStateData>,
                       void(uint32_t idx, std::span<OptT> out) const> //
      ::add_convention<pro::weak_dispatch<OptMemStateDim>,
                       size_t() const> //
      ::add_convention<pro::weak_dispatch<OptMemAllocationBytesPerEntry>,
                       size_t() const> //
      ::add_convention<pro::weak_dispatch<OptMemSetState>,
                       void(uint32_t idx, std::span<const OptT> state)> //
      ::add_convention<pro::weak_dispatch<OptMemStepData>,
                       uint32_t(uint32_t idx) const> //
      ::add_convention<pro::weak_dispatch<OptMemSetStep>,
                       void(uint32_t idx, uint32_t step)> //
      ::build {};

} // namespace se

#endif // SPARSE_EMBEDDING_OPTIMIZER_INTERFACE_HPP_
