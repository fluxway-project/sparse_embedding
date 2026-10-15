#pragma once

#include <proxy/proxy.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <ranges>
#include <span>
#include <stdexcept>

namespace se {

template <typename ParamT> struct OptimizerBatchRow {
  uint32_t index = 0;
  std::span<ParamT> parameters;
  std::span<const ParamT> gradients;
  uint64_t score = 0;
};

template <typename ParamT> struct OptimizerBatchView {
  std::span<const OptimizerBatchRow<ParamT>> rows;
  size_t dimension = 0;

  [[nodiscard]] size_t size() const noexcept { return rows.size(); }

  [[nodiscard]] bool valid() const noexcept {
    return dimension != 0 && std::ranges::all_of(rows, [this](const auto &row) {
             return row.parameters.size() == dimension &&
                    row.gradients.size() == dimension;
           });
  }

  [[nodiscard]] const OptimizerBatchRow<ParamT> &row(size_t position) const {
    if (position >= size() || !valid())
      throw std::out_of_range("optimizer batch parameter row is out of range");
    return rows[position];
  }
};

PRO_DEF_MEM_DISPATCH(OptimizerAllocate, allocate);
PRO_DEF_MEM_DISPATCH(OptimizerReserve, reserve);
PRO_DEF_MEM_DISPATCH(OptimizerDeallocate, deallocate);
PRO_DEF_MEM_DISPATCH(OptimizerClear, clear);
PRO_DEF_MEM_DISPATCH(OptimizerKernelUpdate, update);
PRO_DEF_MEM_DISPATCH(OptimizerKernelSetPageEvictionEnabled,
                     set_page_eviction_enabled);
PRO_DEF_MEM_DISPATCH(OptimizerKernelCapacity, capacity);
PRO_DEF_MEM_DISPATCH(OptimizerKernelStepData, step_data);
PRO_DEF_MEM_DISPATCH(OptimizerKernelSetStep, set_step);
PRO_DEF_MEM_DISPATCH(OptimizerKernelRequiresScore, requires_score);
PRO_DEF_MEM_DISPATCH(OptimizerKernelPrefetchState, prefetch_state);
PRO_DEF_MEM_DISPATCH(OptimizerKernelStep, step);
PRO_DEF_MEM_DISPATCH(OptimizerKernelStateData, state_data);
PRO_DEF_MEM_DISPATCH(OptimizerKernelStateDim, state_dim);
PRO_DEF_MEM_DISPATCH(OptimizerKernelAllocationBytesPerEntry,
                     allocation_bytes_per_entry);
PRO_DEF_MEM_DISPATCH(OptimizerKernelSetState, set_state);

template <typename ParamT, typename OptT>
struct OptimizerExtensionFacade
    : pro                                                   //
      ::facade_builder                                      //
      ::add_convention<OptimizerAllocate, void(uint32_t)>   //
      ::add_convention<OptimizerReserve, void(size_t)>      //
      ::add_convention<OptimizerDeallocate, void(uint32_t)> //
      ::add_convention<OptimizerClear, void()>              //
      ::add_convention<OptimizerKernelUpdate,
                       void(const OptimizerBatchView<ParamT> &, uint64_t)> //
      ::template add_convention<
          pro::weak_dispatch<OptimizerKernelSetPageEvictionEnabled>,
          void(bool)> //
      ::template add_convention<pro::weak_dispatch<OptimizerKernelCapacity>,
                                size_t() const> //
      ::template add_convention<pro::weak_dispatch<OptimizerKernelStepData>,
                                uint32_t(uint32_t) const> //
      ::template add_convention<pro::weak_dispatch<OptimizerKernelSetStep>,
                                void(uint32_t, uint32_t)> //
      ::template add_convention<
          pro::weak_dispatch<OptimizerKernelRequiresScore>,
          bool() const> //
      ::template add_convention<
          pro::weak_dispatch<OptimizerKernelPrefetchState>,
          void(uint32_t) const> //
      ::template add_convention<pro::weak_dispatch<OptimizerKernelStep>,
                                void(uint32_t, std::span<ParamT>,
                                     std::span<const ParamT>, uint64_t,
                                     uint64_t)> //
      ::template add_convention<pro::weak_dispatch<OptimizerKernelStateData>,
                                void(uint32_t, std::span<OptT>) const> //
      ::template add_convention<pro::weak_dispatch<OptimizerKernelStateDim>,
                                size_t() const> //
      ::template add_convention<
          pro::weak_dispatch<OptimizerKernelAllocationBytesPerEntry>,
          size_t() const> //
      ::template add_convention<pro::weak_dispatch<OptimizerKernelSetState>,
                                void(uint32_t, std::span<const OptT>)> //
      ::build {};

template <typename ParamT, typename OptT>
using OptimizerExtension = pro::proxy<OptimizerExtensionFacade<ParamT, OptT>>;

} // namespace se
