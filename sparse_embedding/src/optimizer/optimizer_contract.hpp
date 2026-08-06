#pragma once

#include <proxy/proxy.h>

#include <cstddef>
#include <cstdint>
#include <span>

#include "sparse_embedding/src/optimizer/optimizer_codec.hpp"

namespace se {

struct OptimizerBatchView {
  std::span<const uint32_t> indices;
  std::span<std::byte> parameters;
  std::span<const std::byte> gradients;
  size_t parameter_stride = 0;
  size_t gradient_stride = 0;
  size_t dimension = 0;
};

PRO_DEF_MEM_DISPATCH(OptimizerAllocate, allocate);
PRO_DEF_MEM_DISPATCH(OptimizerReserve, reserve);
PRO_DEF_MEM_DISPATCH(OptimizerDeallocate, deallocate);
PRO_DEF_MEM_DISPATCH(OptimizerClear, clear);
PRO_DEF_MEM_DISPATCH(OptimizerKernelUpdate, update);
PRO_DEF_MEM_DISPATCH(OptimizerKernelStateLayout, state_layout);
PRO_DEF_MEM_DISPATCH(OptimizerKernelStateCodec, state_codec);

struct OptimizerExtensionFacade
    : pro                                                   //
      ::facade_builder                                      //
      ::add_convention<OptimizerAllocate, void(uint32_t)>   //
      ::add_convention<OptimizerReserve, void(size_t)>      //
      ::add_convention<OptimizerDeallocate, void(uint32_t)> //
      ::add_convention<OptimizerClear, void()>              //
      ::add_convention<OptimizerKernelUpdate,
                       void(const OptimizerBatchView &, uint64_t, uint64_t)> //
      ::add_convention<OptimizerKernelStateLayout,
                       OptimizerStateLayout() const> //
      ::add_convention<OptimizerKernelStateCodec,
                       OptimizerStateCodec() const> //
      ::build {};

using OptimizerExtension = pro::proxy<OptimizerExtensionFacade>;

} // namespace se
