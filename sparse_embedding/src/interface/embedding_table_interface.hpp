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

// ms proxy — zero-overhead polymorphism for plugin backends
// See: https://github.com/ngcpp/proxy
#include <proxy.h> // from @proxy

#include <system_error>

#include "common/numeric_types.hpp"

// =============================================================================
// Interface semantics
// =============================================================================
//
// KV Embedding backends in recommendation/search systems share a narrow set of
// operations.  Each backend (in-process hash table, Redis, RocksDB, …) is
// provided as a *plugin* — a concrete type that is never inherited from an
// abstract base class.  ms proxy acts as the type-erased handle, eliminating
// both vtable inheritance overhead and the "fragile base class" problem.
//
// Key type  : uint64_t or int64_t  (feature ID)
// Value type: dense float vector   (embedding row), typed via template
//             Supported scalar elements: float, se::float16_t, se::bfloat16_t
// Score type: uint64_t             (priority / LRU timestamp for eviction)
//
// All mutating operations write through immediately; async flushing is the
// backend's responsibility.
// =============================================================================

namespace se {

// ---------------------------------------------------------------------------
// Dispatch declarations
// Each PRO_DEF_MEM_DISPATCH / PRO_DEF_WEAK_DISPATCH pair maps a method name
// to one or more overloaded call signatures.  WEAK variants fall back to
// NotImplemented so that backends only override what they actually support.
// ---------------------------------------------------------------------------

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

// Weak (optional) wrappers
PRO_DEF_WEAK_DISPATCH(WeakMemInit, MemInit, NotImplemented);
PRO_DEF_WEAK_DISPATCH(WeakMemShutdown, MemShutdown, NotImplemented);
PRO_DEF_WEAK_DISPATCH(WeakMemName, MemName, NotImplemented);
PRO_DEF_WEAK_DISPATCH(WeakMemDim, MemDim, NotImplemented);
PRO_DEF_WEAK_DISPATCH(WeakMemSize, MemSize, NotImplemented);
PRO_DEF_WEAK_DISPATCH(WeakMemCapacity, MemCapacity, NotImplemented);
PRO_DEF_WEAK_DISPATCH(WeakMemReserve, MemReserve, NotImplemented);
PRO_DEF_WEAK_DISPATCH(WeakMemLookup, MemLookup, NotImplemented);
PRO_DEF_WEAK_DISPATCH(WeakMemLookupOrInsert, MemLookupOrInsert, NotImplemented);
PRO_DEF_WEAK_DISPATCH(WeakMemAssign, MemAssign, NotImplemented);
PRO_DEF_WEAK_DISPATCH(WeakMemContains, MemContains, NotImplemented);
PRO_DEF_WEAK_DISPATCH(WeakMemErase, MemErase, NotImplemented);
PRO_DEF_WEAK_DISPATCH(WeakMemClear, MemClear, NotImplemented);
PRO_DEF_WEAK_DISPATCH(WeakMemAssignScores, MemAssignScores, NotImplemented);
PRO_DEF_WEAK_DISPATCH(WeakMemInsertAndEvict, MemInsertAndEvict, NotImplemented);
PRO_DEF_WEAK_DISPATCH(WeakMemExportBatch, MemExportBatch, NotImplemented);

// ---------------------------------------------------------------------------
// Signature expansion macros
// All tensor arguments are passed as std::span<const T> (read) or
// std::span<T> (write) to remain allocation-free at the call boundary.
// The backend owns its own memory / arena.
// ---------------------------------------------------------------------------

// lookup(n_keys, keys, out_values[n_keys * dim])
#define SE_LOOKUP_SIG(key_t, val_t)                                            \
  std::error_code(std::size_t n_keys, std::span<const key_t> keys,             \
                  std::span<val_t> out_values)

// lookup_or_insert(n_keys, keys, out_values[n_keys * dim], out_inserted_mask)
#define SE_LOOKUP_OR_INSERT_SIG(key_t, val_t)                                  \
  std::error_code(std::size_t n_keys, std::span<const key_t> keys,             \
                  std::span<val_t> out_values,                                 \
                  std::span<bool> out_inserted_mask)

// assign(n_keys, keys, values[n_keys * dim])
#define SE_ASSIGN_SIG(key_t, val_t)                                            \
  std::error_code(std::size_t n_keys, std::span<const key_t> keys,             \
                  std::span<const val_t> values)

// contains(n_keys, keys, out_exists)
#define SE_CONTAINS_SIG(key_t)                                                 \
  std::error_code(std::size_t n_keys, std::span<const key_t> keys,             \
                  std::span<bool> out_exists)

// erase(n_keys, keys)
#define SE_ERASE_SIG(key_t)                                                    \
  std::error_code(std::size_t n_keys, std::span<const key_t> keys)

// assign_scores(n_keys, keys, scores)
#define SE_ASSIGN_SCORES_SIG(key_t)                                            \
  std::error_code(std::size_t n_keys, std::span<const key_t> keys,             \
                  std::span<const uint64_t> scores)

// insert_and_evict(n_keys, keys, values, scores,
//                  out_evicted_keys, out_evicted_values, out_evicted_count)
#define SE_INSERT_AND_EVICT_SIG(key_t, val_t)                                  \
  std::error_code(                                                             \
      std::size_t n_keys, std::span<const key_t> keys,                         \
      std::span<const val_t> values, std::span<const uint64_t> scores,         \
      std::span<key_t> out_evicted_keys, std::span<val_t> out_evicted_values,  \
      std::size_t &out_evicted_count)

// export_batch(max_n, cursor_offset, out_exported_count, out_keys,
//              out_values, out_scores)
#define SE_EXPORT_BATCH_SIG(key_t, val_t)                                      \
  std::error_code(std::size_t max_n, std::size_t cursor_offset,                \
                  std::size_t &out_exported_count, std::span<key_t> out_keys,  \
                  std::span<val_t> out_values, std::span<uint64_t> out_scores)

// ---------------------------------------------------------------------------
// Type-list expansion helpers
// Expand each operation over the Cartesian product of supported key and value
// element types.  Adding a new type only requires touching this section.
// ---------------------------------------------------------------------------

#define SE_FOREACH_KEY_VAL(OP)                                                 \
  OP(int64_t, float), OP(int64_t, se::float16_t), OP(int64_t, se::bfloat16_t), \
      OP(uint64_t, float), OP(uint64_t, se::float16_t),                        \
      OP(uint64_t, se::bfloat16_t)

#define SE_FOREACH_KEY(OP) OP(int64_t), OP(uint64_t)

// ---------------------------------------------------------------------------
// EmbeddingTable — the ms proxy facade
//
// A backend plugin exposes its concrete type to this facade by implementing
// the requisite methods; it never derives from any base class.
//
// Usage:
//   pro::proxy<EmbeddingTable> table = pro::make_proxy<EmbeddingTable,
//   MyBackend>(args...); table->lookup(n, keys, out);
// ---------------------------------------------------------------------------

// clang-format off
// NOLINTBEGIN(whitespace/indent_namespace)
struct EmbeddingTable : pro::facade_builder
  // --- Lifecycle -------------------------------------------------------------
  ::add_convention<WeakMemInit,
                   std::error_code(std::string_view config_json)>
  ::add_convention<WeakMemShutdown,
                   std::error_code()>

  // --- Metadata -------------------------------------------------------------
  ::add_convention<WeakMemName,
                   std::string_view()>
  ::add_convention<WeakMemDim,
                   std::size_t()>
  ::add_convention<WeakMemSize,
                   std::size_t()>
  ::add_convention<WeakMemCapacity,
                   std::size_t()>
  ::add_convention<WeakMemReserve,
                   std::error_code(std::size_t new_capacity)>

  // --- Lookup ---------------------------------------------------------------
  ::add_convention<WeakMemLookup,
                   SE_FOREACH_KEY_VAL(SE_LOOKUP_SIG)>

  // --- Lookup-or-insert -----------------------------------------------------
  ::add_convention<WeakMemLookupOrInsert,
                   SE_FOREACH_KEY_VAL(SE_LOOKUP_OR_INSERT_SIG)>

  // --- Assign ---------------------------------------------------------------
  ::add_convention<WeakMemAssign,
                   SE_FOREACH_KEY_VAL(SE_ASSIGN_SIG)>

  // --- Contains -------------------------------------------------------------
  ::add_convention<WeakMemContains,
                   SE_FOREACH_KEY(SE_CONTAINS_SIG)>

  // --- Erase / clear --------------------------------------------------------
  ::add_convention<WeakMemErase,
                   SE_FOREACH_KEY(SE_ERASE_SIG)>
  ::add_convention<WeakMemClear,
                   std::error_code()>

  // --- Score / eviction (optional, backends may leave unimplemented) --------
  ::add_convention<WeakMemAssignScores,
                   SE_FOREACH_KEY(SE_ASSIGN_SCORES_SIG)>
  ::add_convention<WeakMemInsertAndEvict,
                   SE_FOREACH_KEY_VAL(SE_INSERT_AND_EVICT_SIG)>

  // --- Export / snapshot ----------------------------------------------------
  ::add_convention<WeakMemExportBatch,
                   SE_FOREACH_KEY_VAL(SE_EXPORT_BATCH_SIG)>

  ::build {};
// NOLINTEND(whitespace/indent_namespace)
// clang-format on

// Clean up internal macros — only the se:: types and the facade itself leak
// out of this header.
#undef SE_LOOKUP_SIG
#undef SE_LOOKUP_OR_INSERT_SIG
#undef SE_ASSIGN_SIG
#undef SE_CONTAINS_SIG
#undef SE_ERASE_SIG
#undef SE_ASSIGN_SCORES_SIG
#undef SE_INSERT_AND_EVICT_SIG
#undef SE_EXPORT_BATCH_SIG
#undef SE_FOREACH_KEY_VAL
#undef SE_FOREACH_KEY

} // namespace se

#endif // SPARSE_EMBEDDING_EMBEDDING_TABLE_INTERFACE_HPP_
