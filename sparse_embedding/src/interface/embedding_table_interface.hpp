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

#include <span>
#include <system_error>
#include <vector>

#include "sparse_embedding/src/common/function.hpp"

namespace se {

template <typename ParamT, typename OptT> struct PageImageExport {
  uint64_t physical_page_id;
  uint32_t embedding_dim;
  uint32_t page_slot_stride_bytes;
  std::vector<uint32_t> slot_ordinals;
  std::vector<uint64_t> keys;
  std::vector<uint64_t> seq_counts;
  std::vector<ParamT> values;
  std::vector<uint64_t> scores;
  std::vector<OptT> optimizer_states;
  std::vector<uint32_t> steps;
};

template <typename ParamT, typename OptT> struct PageImageImport {
  size_t embedding_dim;
  PageImageExport<ParamT, OptT> image;
};

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
PRO_DEF_MEM_DISPATCH(MemReservePerShard, reserve_per_shard);
PRO_DEF_MEM_DISPATCH(MemInternalShardCount, internal_shard_count);

// --- Core read/write ---------------------------------------------------------
PRO_DEF_MEM_DISPATCH(MemLookup, lookup);
PRO_DEF_MEM_DISPATCH(MemLookupOrInsert, lookup_or_insert);
PRO_DEF_MEM_DISPATCH(MemAssign, assign);
PRO_DEF_MEM_DISPATCH(MemContains, contains);
PRO_DEF_MEM_DISPATCH(MemErase, erase);
PRO_DEF_MEM_DISPATCH(MemClear, clear);

// --- Score / eviction (optional) ---------------------------------------------
PRO_DEF_MEM_DISPATCH(MemHasScorePolicy, has_score_policy);
PRO_DEF_MEM_DISPATCH(MemAssignScores, assign_scores);
PRO_DEF_MEM_DISPATCH(MemEvictByScore, evict_by_score);

// --- Export / snapshot -------------------------------------------------------
// export_batch uses a visitor callback — backend iterates all entries via
// for_each and calls visitor(keys, values, scores) per batch.
// visitor returns false to stop early.
PRO_DEF_MEM_DISPATCH(MemExportBatch, export_batch);
PRO_DEF_MEM_DISPATCH(MemCompactPage, compact_page);
PRO_DEF_MEM_DISPATCH(MemPromoteHotKeys, promote_hot_keys);
PRO_DEF_MEM_DISPATCH(MemPromotedHotPageSize, promoted_hot_page_size);
PRO_DEF_MEM_DISPATCH(MemExportPageImages, export_page_images);
PRO_DEF_MEM_DISPATCH(MemImportPageImages, import_page_images);

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
using WeakMemReservePerShard = pro::weak_dispatch<MemReservePerShard>;
using WeakMemInternalShardCount = pro::weak_dispatch<MemInternalShardCount>;
using WeakMemLookup = pro::weak_dispatch<MemLookup>;
using WeakMemLookupOrInsert = pro::weak_dispatch<MemLookupOrInsert>;
using WeakMemAssign = pro::weak_dispatch<MemAssign>;
using WeakMemContains = pro::weak_dispatch<MemContains>;
using WeakMemErase = pro::weak_dispatch<MemErase>;
using WeakMemClear = pro::weak_dispatch<MemClear>;
using WeakMemHasScorePolicy = pro::weak_dispatch<MemHasScorePolicy>;
using WeakMemAssignScores = pro::weak_dispatch<MemAssignScores>;
using WeakMemEvictByScore = pro::weak_dispatch<MemEvictByScore>;
using WeakMemExportBatch = pro::weak_dispatch<MemExportBatch>;
using WeakMemCompactPage = pro::weak_dispatch<MemCompactPage>;
using WeakMemPromoteHotKeys = pro::weak_dispatch<MemPromoteHotKeys>;
using WeakMemPromotedHotPageSize = pro::weak_dispatch<MemPromotedHotPageSize>;
using WeakMemExportPageImages = pro::weak_dispatch<MemExportPageImages>;
using WeakMemImportPageImages = pro::weak_dispatch<MemImportPageImages>;
using WeakMemForwardLookup = pro::weak_dispatch<MemForwardLookup>;
using WeakMemAccumulateGradients = pro::weak_dispatch<MemAccumulateGradients>;
using WeakMemOptimizerStep = pro::weak_dispatch<MemOptimizerStep>;

// ---------------------------------------------------------------------------
// Signature macros
// ---------------------------------------------------------------------------

#define SE_LOOKUP_FACADE_SIG(key_t)                                            \
  std::error_code(std::span<const key_t> keys, std::span<ParamT> out_values,   \
                  std::span<const int64_t> row_lengths)

#define SE_LOOKUP_OR_INSERT_FACADE_SIG(key_t)                                  \
  std::error_code(std::span<const key_t> keys, std::span<ParamT> out_values,   \
                  std::span<const int64_t> row_lengths)

#define SE_ASSIGN_FACADE_SIG(key_t)                                            \
  std::error_code(std::span<const key_t> keys, std::span<const ParamT> values, \
                  std::span<const int64_t> row_lengths)

#define SE_CONTAINS_FACADE_SIG(key_t)                                          \
  std::error_code(std::span<const key_t> keys, std::span<uint8_t> out_exists)

#define SE_ERASE_FACADE_SIG(key_t) std::error_code(std::span<const key_t> keys)

#define SE_ASSIGN_SCORES_FACADE_SIG(key_t)                                     \
  std::error_code(std::span<const key_t> keys, std::span<const uint64_t> scores)

#define SE_FORWARD_LOOKUP_FACADE_SIG(key_t)                                    \
  std::error_code(std::span<const key_t> keys, std::span<ParamT> out_values,   \
                  std::span<const int64_t> row_lengths)

#define SE_ACCUMULATE_GRADIENTS_FACADE_SIG(key_t)                              \
  std::error_code(std::span<const key_t> keys,                                 \
                  std::span<const ParamT> grad_values,                         \
                  std::span<const int64_t> row_lengths)

// evict_by_score: asynchronous-safe GC operation.
// Evicts entries with score <= max_score, up to max_evict_count entries.
// out_evicted_* are filled with evicted (key, value) pairs.
#define SE_EVICT_BY_SCORE_FACADE_SIG(key_t)                                    \
  std::error_code(uint64_t max_score, std::size_t max_evict_count,             \
                  std::span<key_t> out_evicted_keys,                           \
                  std::span<ParamT> out_evicted_values,                        \
                  std::size_t &out_evicted_count)

// export_batch: visitor-callback snapshot export.
// Backend calls visitor(keys, values, scores) for each batch.
// visitor returns false to stop early.
// batch_size controls the maximum number of entries per visitor call.
#define SE_EXPORT_BATCH_FACADE_SIG(key_t)                                      \
  std::error_code(                                                             \
      se::Function<bool(std::size_t, std::span<const key_t>,                   \
                        std::span<const ParamT>, std::span<const uint64_t>)>   \
          visitor,                                                             \
      std::size_t batch_size)

#define SE_COMPACT_PAGE_FACADE_SIG                                             \
  std::size_t(std::size_t dim, uint64_t page_id)

#define SE_PROMOTE_HOT_KEYS_FACADE_SIG                                         \
  std::size_t(std::size_t dim, uint64_t page_id, std::size_t maximum_keys,     \
              uint64_t minimum_score)

#define SE_PROMOTED_HOT_PAGE_SIZE_FACADE_SIG std::size_t(std::size_t dim) const

#define SE_EXPORT_PAGE_IMAGES_FACADE_SIG                                       \
  void(se::Function<bool(std::size_t, PageImageExport<ParamT, OptT> &&)>       \
           visitor)

#define SE_IMPORT_PAGE_IMAGES_FACADE_SIG                                       \
  void(std::span<const PageImageImport<ParamT, OptT>> images)

#define SE_ASSIGN_OPT_STATES_FACADE_SIG(key_t)                                 \
  void(std::size_t, std::span<const key_t>, std::span<const OptT>,             \
       std::span<const uint32_t>)

#define SE_EXPORT_OPT_STATE_BATCH_FACADE_SIG(key_t)                            \
  void(se::Function<bool(std::size_t, std::span<const key_t>,                  \
                         std::span<const ParamT>, std::span<const uint64_t>,   \
                         std::span<const OptT>, std::span<const uint32_t>)>,   \
       std::size_t)

#define SE_EXPORT_OPT_STATE_BATCH_OWNED_FACADE_SIG(key_t)                      \
  void(se::Function<bool(std::size_t, std::vector<key_t> &&,                   \
                         std::vector<ParamT> &&, std::vector<uint64_t> &&,     \
                         std::vector<OptT> &&, std::vector<uint32_t> &&)>,     \
       std::size_t)

PRO_DEF_MEM_DISPATCH(MemAssignOptStates, assign_opt_states);
PRO_DEF_MEM_DISPATCH(MemExportOptStateBatch, export_opt_state_batch);
PRO_DEF_MEM_DISPATCH(MemExportOptStateBatchOwned, export_opt_state_batch_owned);

// clang-format off
// NOLINTBEGIN(whitespace/indent_namespace)
template <typename ParamT, typename OptT>
struct EmbeddingTableFacade : pro //
  ::facade_builder //
  // --- Lifecycle -------------------------------------------------------------
  ::add_convention<WeakMemInit,
                   std::error_code(std::string_view config_json)> //
  ::add_convention<WeakMemShutdown,
                   std::error_code()>

  // --- Metadata -------------------------------------------------------------
  ::add_convention<WeakMemName,
                   std::string_view() const> //
  ::add_convention<WeakMemDim,
                   std::size_t() const> //
  ::add_convention<WeakMemSize,
                   std::size_t() const> //
  ::add_convention<WeakMemCapacity,
                   std::size_t() const> //
  ::add_convention<WeakMemReserve,
                   std::error_code(std::size_t keys_count, std::size_t dim)> //
  ::add_convention<WeakMemReservePerShard,
                   std::error_code(std::span<const std::size_t> counts, std::size_t dim)> //
  ::add_convention<WeakMemInternalShardCount,
                   std::size_t() const> //

  // --- Lookup ---------------------------------------------------------------
  ::add_convention<WeakMemLookup,
                   SE_LOOKUP_FACADE_SIG(int32_t), SE_LOOKUP_FACADE_SIG(uint32_t),
                   SE_LOOKUP_FACADE_SIG(int64_t), SE_LOOKUP_FACADE_SIG(uint64_t)> //

  // --- Lookup-or-insert -----------------------------------------------------
  ::add_convention<WeakMemLookupOrInsert,
                   SE_LOOKUP_OR_INSERT_FACADE_SIG(int32_t), SE_LOOKUP_OR_INSERT_FACADE_SIG(uint32_t),
                   SE_LOOKUP_OR_INSERT_FACADE_SIG(int64_t), SE_LOOKUP_OR_INSERT_FACADE_SIG(uint64_t)> //

  // --- Assign ---------------------------------------------------------------
  ::add_convention<WeakMemAssign,
                   SE_ASSIGN_FACADE_SIG(int32_t), SE_ASSIGN_FACADE_SIG(uint32_t),
                   SE_ASSIGN_FACADE_SIG(int64_t), SE_ASSIGN_FACADE_SIG(uint64_t)> //

  // --- Contains -------------------------------------------------------------
  ::add_convention<WeakMemContains,
                   SE_CONTAINS_FACADE_SIG(int32_t), SE_CONTAINS_FACADE_SIG(uint32_t),
                   SE_CONTAINS_FACADE_SIG(int64_t), SE_CONTAINS_FACADE_SIG(uint64_t)> //

  // --- Erase / clear --------------------------------------------------------
  ::add_convention<WeakMemErase,
                   SE_ERASE_FACADE_SIG(int32_t), SE_ERASE_FACADE_SIG(uint32_t),
                   SE_ERASE_FACADE_SIG(int64_t), SE_ERASE_FACADE_SIG(uint64_t)> //
  ::add_convention<WeakMemClear,
                   std::error_code()> //

  // --- Score / eviction -----------------------------------------------------
  ::add_convention<WeakMemHasScorePolicy,
                   bool() const> //
  ::add_convention<WeakMemAssignScores,
                   SE_ASSIGN_SCORES_FACADE_SIG(int32_t), SE_ASSIGN_SCORES_FACADE_SIG(uint32_t),
                   SE_ASSIGN_SCORES_FACADE_SIG(int64_t), SE_ASSIGN_SCORES_FACADE_SIG(uint64_t)> //
  ::add_convention<WeakMemEvictByScore,
                   SE_EVICT_BY_SCORE_FACADE_SIG(int32_t), SE_EVICT_BY_SCORE_FACADE_SIG(uint32_t),
                   SE_EVICT_BY_SCORE_FACADE_SIG(int64_t), SE_EVICT_BY_SCORE_FACADE_SIG(uint64_t)> //

  // --- Export / snapshot ----------------------------------------------------
  ::add_convention<WeakMemExportBatch,
                   SE_EXPORT_BATCH_FACADE_SIG(int32_t), SE_EXPORT_BATCH_FACADE_SIG(uint32_t),
                   SE_EXPORT_BATCH_FACADE_SIG(int64_t), SE_EXPORT_BATCH_FACADE_SIG(uint64_t)> //
  ::add_convention<WeakMemCompactPage, SE_COMPACT_PAGE_FACADE_SIG> //
  ::add_convention<WeakMemPromoteHotKeys, SE_PROMOTE_HOT_KEYS_FACADE_SIG> //
  ::add_convention<WeakMemPromotedHotPageSize,                                \
                   SE_PROMOTED_HOT_PAGE_SIZE_FACADE_SIG> //
  ::add_convention<WeakMemExportPageImages, SE_EXPORT_PAGE_IMAGES_FACADE_SIG> //
  ::add_convention<WeakMemImportPageImages, SE_IMPORT_PAGE_IMAGES_FACADE_SIG> //

  // --- Training: Forward ----------------------------------------------------
  ::add_convention<WeakMemForwardLookup,
                   SE_FORWARD_LOOKUP_FACADE_SIG(int32_t), SE_FORWARD_LOOKUP_FACADE_SIG(uint32_t),
                   SE_FORWARD_LOOKUP_FACADE_SIG(int64_t), SE_FORWARD_LOOKUP_FACADE_SIG(uint64_t)> //

  // --- Training: Backward ---------------------------------------------------
  ::add_convention<WeakMemAccumulateGradients,
                   SE_ACCUMULATE_GRADIENTS_FACADE_SIG(int32_t), SE_ACCUMULATE_GRADIENTS_FACADE_SIG(uint32_t),
                   SE_ACCUMULATE_GRADIENTS_FACADE_SIG(int64_t), SE_ACCUMULATE_GRADIENTS_FACADE_SIG(uint64_t)> //
  ::add_convention<WeakMemOptimizerStep,
                   std::error_code()> //

  // --- Optimizer state checkpoint -------------------------------------------
  ::add_convention<pro::weak_dispatch<MemAssignOptStates>,
                   SE_ASSIGN_OPT_STATES_FACADE_SIG(int32_t),
                   SE_ASSIGN_OPT_STATES_FACADE_SIG(uint32_t),
                   SE_ASSIGN_OPT_STATES_FACADE_SIG(int64_t),
                   SE_ASSIGN_OPT_STATES_FACADE_SIG(uint64_t)> //
  ::add_convention<pro::weak_dispatch<MemExportOptStateBatch>,
                   SE_EXPORT_OPT_STATE_BATCH_FACADE_SIG(int32_t),
                   SE_EXPORT_OPT_STATE_BATCH_FACADE_SIG(uint32_t),
                   SE_EXPORT_OPT_STATE_BATCH_FACADE_SIG(int64_t),
                   SE_EXPORT_OPT_STATE_BATCH_FACADE_SIG(uint64_t)> //
  ::add_convention<pro::weak_dispatch<MemExportOptStateBatchOwned>,
                   SE_EXPORT_OPT_STATE_BATCH_OWNED_FACADE_SIG(int32_t),
                   SE_EXPORT_OPT_STATE_BATCH_OWNED_FACADE_SIG(uint32_t),
                   SE_EXPORT_OPT_STATE_BATCH_OWNED_FACADE_SIG(int64_t),
                   SE_EXPORT_OPT_STATE_BATCH_OWNED_FACADE_SIG(uint64_t)> //

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
#undef SE_EVICT_BY_SCORE_FACADE_SIG
#undef SE_EXPORT_BATCH_FACADE_SIG
#undef SE_COMPACT_PAGE_FACADE_SIG
#undef SE_PROMOTE_HOT_KEYS_FACADE_SIG
#undef SE_PROMOTED_HOT_PAGE_SIZE_FACADE_SIG
#undef SE_EXPORT_PAGE_IMAGES_FACADE_SIG
#undef SE_IMPORT_PAGE_IMAGES_FACADE_SIG
#undef SE_ASSIGN_OPT_STATES_FACADE_SIG
#undef SE_EXPORT_OPT_STATE_BATCH_FACADE_SIG
#undef SE_EXPORT_OPT_STATE_BATCH_OWNED_FACADE_SIG

} // namespace se

#endif // SPARSE_EMBEDDING_EMBEDDING_TABLE_INTERFACE_HPP_
