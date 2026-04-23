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

#ifndef SPARSE_EMBEDDING_EMBEDDING_TABLE_INTERFACE_HPP_
#define SPARSE_EMBEDDING_EMBEDDING_TABLE_INTERFACE_HPP_

#include <proxy/proxy.h>

#include <system_error>

namespace se {

struct NotImplemented {
  explicit NotImplemented(auto &&...) {
    throw std::runtime_error{
        "Not implemented: this EmbeddingTable backend does not support "
        "the requested operation"};
  }

  template <class T> operator T() const noexcept { std::unreachable(); }
};

// --- Lifecycle ---------------------------------------------------------------
PRO_DEF_MEM_DISPATCH(MemInit, init);
PRO_DEF_MEM_DISPATCH(MemShutdown, shutdown);

// --- Metadata ----------------------------------------------------------------
PRO_DEF_MEM_DISPATCH(MemName, name);
PRO_DEF_MEM_DISPATCH(MemDim, dim);
PRO_DEF_MEM_DISPATCH(MemSize, size);
PRO_DEF_MEM_DISPATCH(MemCapacity, capacity);
PRO_DEF_MEM_DISPATCH(MemReserve, reserve);

// --- Core read/write ---------------------------------------------------------
PRO_DEF_MEM_DISPATCH(MemLookup, lookup);
PRO_DEF_MEM_DISPATCH(MemLookupOrInsert, lookup_or_insert);
PRO_DEF_MEM_DISPATCH(MemAssign, assign);
PRO_DEF_MEM_DISPATCH(MemContains, contains);
PRO_DEF_MEM_DISPATCH(MemErase, erase);
PRO_DEF_MEM_DISPATCH(MemClear, clear);

// --- Score / eviction (optional) ---------------------------------------------
PRO_DEF_MEM_DISPATCH(MemAssignScores, assign_scores);
PRO_DEF_MEM_DISPATCH(MemInsertAndEvict, insert_and_evict);

// --- Export / snapshot -------------------------------------------------------
PRO_DEF_MEM_DISPATCH(MemExportBatch, export_batch);

// --- Training: Forward -------------------------------------------------------
PRO_DEF_MEM_DISPATCH(MemForwardLookup, forward_lookup);

// --- Training: Backward ------------------------------------------------------
PRO_DEF_MEM_DISPATCH(MemAccumulateGradients, accumulate_gradients);
PRO_DEF_MEM_DISPATCH(MemOptimizerStep, optimizer_step);

// ---------------------------------------------------------------------------
// Weak wrappers (optional)
// ---------------------------------------------------------------------------
using WeakMemInit = pro::weak_dispatch<MemInit>;
using WeakMemShutdown = pro::weak_dispatch<MemShutdown>;
using WeakMemName = pro::weak_dispatch<MemName>;
using WeakMemDim = pro::weak_dispatch<MemDim>;
using WeakMemSize = pro::weak_dispatch<MemSize>;
using WeakMemCapacity = pro::weak_dispatch<MemCapacity>;
using WeakMemReserve = pro::weak_dispatch<MemReserve>;
using WeakMemLookup = pro::weak_dispatch<MemLookup>;
using WeakMemLookupOrInsert = pro::weak_dispatch<MemLookupOrInsert>;
using WeakMemAssign = pro::weak_dispatch<MemAssign>;
using WeakMemContains = pro::weak_dispatch<MemContains>;
using WeakMemErase = pro::weak_dispatch<MemErase>;
using WeakMemClear = pro::weak_dispatch<MemClear>;
using WeakMemAssignScores = pro::weak_dispatch<MemAssignScores>;
using WeakMemInsertAndEvict = pro::weak_dispatch<MemInsertAndEvict>;
using WeakMemExportBatch = pro::weak_dispatch<MemExportBatch>;
using WeakMemForwardLookup = pro::weak_dispatch<MemForwardLookup>;
using WeakMemAccumulateGradients = pro::weak_dispatch<MemAccumulateGradients>;
using WeakMemOptimizerStep = pro::weak_dispatch<MemOptimizerStep>;

// ---------------------------------------------------------------------------
// Signature macros
// ---------------------------------------------------------------------------

#define SE_LOOKUP_FACADE_SIG(key_t)                                            \
  std::error_code(std::span<const key_t> keys, std::span<ValT> out_values,     \
                  std::span<const int64_t> row_lengths)

#define SE_LOOKUP_OR_INSERT_FACADE_SIG(key_t)                                  \
  std::error_code(std::span<const key_t> keys, std::span<ValT> out_values,     \
                  std::span<const int64_t> row_lengths,                        \
                  std::span<uint8_t> out_inserted_mask)

#define SE_ASSIGN_FACADE_SIG(key_t)                                            \
  std::error_code(std::span<const key_t> keys, std::span<const ValT> values,   \
                  std::span<const int64_t> row_lengths)

#define SE_CONTAINS_FACADE_SIG(key_t)                                          \
  std::error_code(std::span<const key_t> keys, std::span<bool> out_exists)

#define SE_ERASE_FACADE_SIG(key_t) std::error_code(std::span<const key_t> keys)

#define SE_ASSIGN_SCORES_FACADE_SIG(key_t)                                     \
  std::error_code(std::span<const key_t> keys, std::span<const uint64_t> scores)

#define SE_FORWARD_LOOKUP_FACADE_SIG(key_t)                                    \
  std::error_code(std::span<const key_t> keys, std::span<ValT> out_values,     \
                  std::span<const int64_t> row_lengths)

#define SE_ACCUMULATE_GRADIENTS_FACADE_SIG(key_t)                              \
  std::error_code(std::span<const key_t> keys,                                 \
                  std::span<const ValT> grad_values,                           \
                  std::span<const int64_t> row_lengths)

#define SE_INSERT_AND_EVICT_FACADE_SIG(key_t)                                  \
  std::error_code(                                                             \
      std::size_t n_keys, std::span<const key_t> keys,                         \
      std::span<const ValT> values, std::span<const uint64_t> scores,          \
      std::span<key_t> out_evicted_keys, std::span<ValT> out_evicted_values,   \
      std::size_t &out_evicted_count)

#define SE_EXPORT_BATCH_FACADE_SIG(key_t)                                      \
  std::error_code(std::size_t max_n, std::size_t cursor_offset,                \
                  std::size_t &out_exported_count, std::span<key_t> out_keys,  \
                  std::span<ValT> out_values, std::span<uint64_t> out_scores)

// clang-format off
// NOLINTBEGIN(whitespace/indent_namespace)
template <typename ValT>
struct EmbeddingTableFacade : pro::facade_builder
  // --- Lifecycle -------------------------------------------------------------
  ::add_convention<WeakMemInit,
                   std::error_code(std::string_view config_json)>
  ::add_convention<WeakMemShutdown,
                   std::error_code()>

  // --- Metadata -------------------------------------------------------------
  ::add_convention<WeakMemName,
                   std::string_view() const>
  ::add_convention<WeakMemDim,
                   std::size_t() const>
  ::add_convention<WeakMemSize,
                   std::size_t() const>
  ::add_convention<WeakMemCapacity,
                   std::size_t() const>
  ::add_convention<WeakMemReserve,
                   std::error_code(std::size_t new_capacity)>

  // --- Lookup ---------------------------------------------------------------
  ::add_convention<WeakMemLookup,
                   SE_LOOKUP_FACADE_SIG(int64_t), SE_LOOKUP_FACADE_SIG(uint64_t)>

  // --- Lookup-or-insert -----------------------------------------------------
  ::add_convention<WeakMemLookupOrInsert,
                   SE_LOOKUP_OR_INSERT_FACADE_SIG(int64_t), SE_LOOKUP_OR_INSERT_FACADE_SIG(uint64_t)>

  // --- Assign ---------------------------------------------------------------
  ::add_convention<WeakMemAssign,
                   SE_ASSIGN_FACADE_SIG(int64_t), SE_ASSIGN_FACADE_SIG(uint64_t)>

  // --- Contains -------------------------------------------------------------
  ::add_convention<WeakMemContains,
                   SE_CONTAINS_FACADE_SIG(int64_t), SE_CONTAINS_FACADE_SIG(uint64_t)>

  // --- Erase / clear --------------------------------------------------------
  ::add_convention<WeakMemErase,
                   SE_ERASE_FACADE_SIG(int64_t), SE_ERASE_FACADE_SIG(uint64_t)>
  ::add_convention<WeakMemClear,
                   std::error_code()>

  // --- Score / eviction -----------------------------------------------------
  ::add_convention<WeakMemAssignScores,
                   SE_ASSIGN_SCORES_FACADE_SIG(int64_t), SE_ASSIGN_SCORES_FACADE_SIG(uint64_t)>
  ::add_convention<WeakMemInsertAndEvict,
                   SE_INSERT_AND_EVICT_FACADE_SIG(int64_t), SE_INSERT_AND_EVICT_FACADE_SIG(uint64_t)>

  // --- Export / snapshot ----------------------------------------------------
  ::add_convention<WeakMemExportBatch,
                   SE_EXPORT_BATCH_FACADE_SIG(int64_t), SE_EXPORT_BATCH_FACADE_SIG(uint64_t)>

  // --- Training: Forward ----------------------------------------------------
  ::add_convention<WeakMemForwardLookup,
                   SE_FORWARD_LOOKUP_FACADE_SIG(int64_t), SE_FORWARD_LOOKUP_FACADE_SIG(uint64_t)>

  // --- Training: Backward ---------------------------------------------------
  ::add_convention<WeakMemAccumulateGradients,
                   SE_ACCUMULATE_GRADIENTS_FACADE_SIG(int64_t), SE_ACCUMULATE_GRADIENTS_FACADE_SIG(uint64_t)>
  ::add_convention<WeakMemOptimizerStep,
                   std::error_code()>

  ::build {};
// NOLINTEND(whitespace/indent_namespace)
// clang-format on

#undef SE_LOOKUP_FACADE_SIG
#undef SE_LOOKUP_OR_INSERT_FACADE_SIG
#undef SE_ASSIGN_FACADE_SIG
#undef SE_CONTAINS_FACADE_SIG
#undef SE_ERASE_FACADE_SIG
#undef SE_ASSIGN_SCORES_FACADE_SIG
#undef SE_FORWARD_LOOKUP_FACADE_SIG
#undef SE_ACCUMULATE_GRADIENTS_FACADE_SIG
#undef SE_INSERT_AND_EVICT_FACADE_SIG
#undef SE_EXPORT_BATCH_FACADE_SIG

} // namespace se

#endif // SPARSE_EMBEDDING_EMBEDDING_TABLE_INTERFACE_HPP_
