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

#ifndef SPARSE_EMBEDDING_MODULE_WRAPPER_HPP_
#define SPARSE_EMBEDDING_MODULE_WRAPPER_HPP_

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <dlpack/dlpack.h>
#include <proxy/proxy.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <variant>
#include <vector>

#include <exec/static_thread_pool.hpp>

#include "sparse_embedding/src/bindings/dlpack_utils.hpp"
#include "sparse_embedding/src/checkpoint/checkpoint_interface.hpp"
#include "sparse_embedding/src/common/function.hpp"
#include "sparse_embedding/src/common/numeric_types.hpp"
#include "sparse_embedding/src/initializer/uniform_initializer.hpp"
#include "sparse_embedding/src/initializer/zeros_initializer.hpp"
#include "sparse_embedding/src/interface/embedding_config.hpp"
#include "sparse_embedding/src/interface/embedding_table_interface.hpp"
#include "sparse_embedding/src/interface/initializer_interface.hpp"
#include "sparse_embedding/src/interface/optimizer_interface.hpp"
#include "sparse_embedding/src/ops/partitioner.hpp"
#include "sparse_embedding/src/optimizer/adam_optimizer.hpp"
#include "sparse_embedding/src/optimizer/optimizer_config.hpp"
#include "sparse_embedding/src/optimizer/sgd_optimizer.hpp"

namespace se::py::detail {

namespace nb = nanobind;

// =============================================================================
// Utilities
// =============================================================================

inline void check_error(std::error_code ec) {
  if (ec)
    throw std::runtime_error(ec.message());
}

enum class KeyDType { kInt32, kUInt32, kInt64, kUInt64 };

inline KeyDType detect_key_dtype(const DLManagedTensor *dlm) {
  auto code = dlm->dl_tensor.dtype.code;
  auto bits = dlm->dl_tensor.dtype.bits;
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

template <typename Fn>
auto dispatch_on_key(const DLManagedTensor *dlm,
                     const std::string & /*expected_key_dtype*/, Fn &&fn) {
  auto kdt = detect_key_dtype(dlm);
  switch (kdt) {
  case KeyDType::kInt32:
    return fn(se::py::dlpack_to_span<const int32_t>(dlm));
  case KeyDType::kUInt32:
    return fn(se::py::dlpack_to_span<const uint32_t>(dlm));
  case KeyDType::kInt64:
    return fn(se::py::dlpack_to_span<const int64_t>(dlm));
  case KeyDType::kUInt64:
    return fn(se::py::dlpack_to_span<const uint64_t>(dlm));
  }
  throw std::invalid_argument("unsupported key_dtype");
}

template <typename Fn>
auto dispatch_partition_key(const DLManagedTensor *dlm, Fn &&fn) {
  auto kdt = detect_key_dtype(dlm);
  if (kdt == KeyDType::kInt32)
    return fn(se::py::dlpack_to_span<const int32_t>(dlm));
  if (kdt == KeyDType::kUInt32)
    return fn(se::py::dlpack_to_span<const uint32_t>(dlm));
  if (kdt == KeyDType::kInt64)
    return fn(se::py::dlpack_to_span<const int64_t>(dlm));
  return fn(se::py::dlpack_to_span<const uint64_t>(dlm));
}

inline size_t get_dlm_size(const DLManagedTensor *dlm) {
  size_t n = 1;
  for (int i = 0; i < dlm->dl_tensor.ndim; ++i) {
    n *= static_cast<size_t>(dlm->dl_tensor.shape[i]);
  }
  return n;
}

inline void fill_uniform_row_lengths(std::span<int64_t> out, int64_t dim) {
  std::fill(out.begin(), out.end(), dim);
}

// =============================================================================
// Builder Proxies for Initializers and Optimizers
// =============================================================================

struct PyOptimizer {
  se::OptimizerConfigVariant config;
};

struct PyInitializer {
  std::shared_ptr<pro::proxy<se::InitializerBuilderFacade>> builder;
};

struct ZerosInitializerBuilder {
  pro::proxy<se::InitializerFacade<float>> build_f32() {
    return pro::make_proxy<se::InitializerFacade<float>,
                           se::ZerosInitializer<float>>();
  }
  pro::proxy<se::InitializerFacade<se::float16_t>> build_f16() {
    return pro::make_proxy<se::InitializerFacade<se::float16_t>,
                           se::ZerosInitializer<se::float16_t>>();
  }
  pro::proxy<se::InitializerFacade<se::bfloat16_t>> build_bf16() {
    return pro::make_proxy<se::InitializerFacade<se::bfloat16_t>,
                           se::ZerosInitializer<se::bfloat16_t>>();
  }
};

struct UniformInitializerBuilder {
  float min_val, max_val;
  pro::proxy<se::InitializerFacade<float>> build_f32() {
    return pro::make_proxy<se::InitializerFacade<float>,
                           se::UniformInitializer<float>>(min_val, max_val);
  }
  pro::proxy<se::InitializerFacade<se::float16_t>> build_f16() {
    return pro::make_proxy<se::InitializerFacade<se::float16_t>,
                           se::UniformInitializer<se::float16_t>>(min_val,
                                                                  max_val);
  }
  pro::proxy<se::InitializerFacade<se::bfloat16_t>> build_bf16() {
    return pro::make_proxy<se::InitializerFacade<se::bfloat16_t>,
                           se::UniformInitializer<se::bfloat16_t>>(min_val,
                                                                   max_val);
  }
};

struct PyPartitioner {
  std::shared_ptr<pro::proxy<se::ops::PartitionerFacade>> facade;
  // Methods defined in module.cpp
  nb::object partition(nb::object keys);
  nb::object async_partition(nb::object keys);
  nb::object filter(nb::object keys, uint32_t target_shard);
  nb::object async_filter(nb::object keys, uint32_t target_shard);
};

// =============================================================================
// Forward declaration
// =============================================================================

class EmbeddingStoreWrapper;

// =============================================================================
// Impl struct declarations (methods defined in .inl, instantiated in per-type
// .cpp). Static methods take EmbeddingStoreWrapper& to access private members
// via friend.
// =============================================================================

template <typename ParamT, typename OptT> struct ConstructOpsImpl {
  static void construct(EmbeddingStoreWrapper &self,
                        const PyInitializer &init_cfg,
                        const PyOptimizer &opt_cfg,
                        const std::string &score_policy_name,
                        size_t num_threads);
};

template <typename ParamT, typename OptT> struct SyncOpsImpl {
  static nb::tuple lookup(EmbeddingStoreWrapper &self, nb::object keys,
                          int64_t dim);
  static nb::tuple lookup_or_insert(EmbeddingStoreWrapper &self,
                                    nb::object keys, int64_t dim);
  static void assign(EmbeddingStoreWrapper &self, nb::object keys,
                     nb::object values);
  static nb::object contains(EmbeddingStoreWrapper &self, nb::object keys);
  static void accumulate_gradients(EmbeddingStoreWrapper &self, nb::object keys,
                                   nb::object grads);
  static void optimizer_step(EmbeddingStoreWrapper &self);
  static void erase(EmbeddingStoreWrapper &self, nb::object keys);
  static void clear(EmbeddingStoreWrapper &self);
  static void assign_scores(EmbeddingStoreWrapper &self, nb::object keys,
                            nb::object scores);
};

template <typename ParamT, typename OptT> struct AsyncOpsImpl {
  static nb::object async_lookup(EmbeddingStoreWrapper &self, nb::object keys,
                                 int64_t dim);
  static nb::object async_assign(EmbeddingStoreWrapper &self, nb::object keys,
                                 nb::object values);
  static nb::object async_lookup_or_insert(EmbeddingStoreWrapper &self,
                                           nb::object keys, int64_t dim);
  static nb::object async_accumulate_gradients(EmbeddingStoreWrapper &self,
                                               nb::object keys,
                                               nb::object grads);
  static nb::object async_optimizer_step(EmbeddingStoreWrapper &self);
  static nb::object async_contains(EmbeddingStoreWrapper &self,
                                   nb::object keys);
  static nb::object async_erase(EmbeddingStoreWrapper &self, nb::object keys);
  static nb::object async_clear(EmbeddingStoreWrapper &self);
  static nb::object async_assign_scores(EmbeddingStoreWrapper &self,
                                        nb::object keys, nb::object scores);
};

template <typename ParamT, typename OptT> struct CheckpointOpsImpl {
#ifdef SE_ENABLE_LANCE_CHECKPOINTER
  static void save_checkpoint(EmbeddingStoreWrapper &self,
                              const std::string &uri, const std::string &mode,
                              uint64_t min_score, bool include_optimizer_state,
                              bool use_compression);
  static void load_checkpoint(EmbeddingStoreWrapper &self,
                              const std::string &uri, uint64_t version,
                              bool clear, bool load_optimizer_state,
                              nb::object filter_fn, uint32_t target_shard_id,
                              size_t batch_size);
#endif
  static void export_batches_dlpack(EmbeddingStoreWrapper &self,
                                    nb::callable visitor, size_t batch_size,
                                    bool include_optimizer_state);
  static void export_batches_arrow(EmbeddingStoreWrapper &self,
                                   nb::callable visitor, size_t batch_size,
                                   bool include_optimizer_state);
  static void assign_opt_state(EmbeddingStoreWrapper &self, nb::object keys_obj,
                               nb::object opt_states_obj, size_t dimension,
                               nb::object steps_obj);
};

// =============================================================================
// EmbeddingStoreWrapper
// =============================================================================

class EmbeddingStoreWrapper {
  template <typename, typename> friend struct ConstructOpsImpl;
  template <typename, typename> friend struct SyncOpsImpl;
  template <typename, typename> friend struct AsyncOpsImpl;
  template <typename, typename> friend struct CheckpointOpsImpl;

  using StoreProxyVariant = std::variant<
      pro::proxy<se::EmbeddingTableFacade<float, float>>,
      pro::proxy<se::EmbeddingTableFacade<float, se::float16_t>>,
      pro::proxy<se::EmbeddingTableFacade<float, se::bfloat16_t>>,
      pro::proxy<se::EmbeddingTableFacade<se::float16_t, float>>,
      pro::proxy<se::EmbeddingTableFacade<se::float16_t, se::float16_t>>,
      pro::proxy<se::EmbeddingTableFacade<se::float16_t, se::bfloat16_t>>,
      pro::proxy<se::EmbeddingTableFacade<se::bfloat16_t, float>>,
      pro::proxy<se::EmbeddingTableFacade<se::bfloat16_t, se::float16_t>>,
      pro::proxy<se::EmbeddingTableFacade<se::bfloat16_t, se::bfloat16_t>>>;
  StoreProxyVariant store_;
  exec::static_thread_pool async_pool_;
  std::string value_dtype_;
  std::string optimizer_dtype_;
  std::string key_dtype_;
  KeyDType key_type_enum_;
  std::vector<size_t> dims_;
  se::PageStorage storage_;

#ifdef SE_ENABLE_LANCE_CHECKPOINTER
  se::CheckpointWriterFactory checkpoint_writer_factory_;
  se::CheckpointReaderFactory checkpoint_reader_factory_;
#endif

  template <typename ParamT, typename OptT> auto &get_store() {
    return std::get<pro::proxy<se::EmbeddingTableFacade<ParamT, OptT>>>(store_);
  }

  template <typename Fn> auto dispatch_val(Fn &&fn) {
    if (value_dtype_ == "float32" || value_dtype_ == "f32")
      return fn.template operator()<float>();
    if (value_dtype_ == "float16" || value_dtype_ == "f16")
      return fn.template operator()<se::float16_t>();
    if (value_dtype_ == "bfloat16" || value_dtype_ == "bf16")
      return fn.template operator()<se::bfloat16_t>();
    throw std::invalid_argument("unsupported dtype: " + value_dtype_);
  }

  template <typename Fn> auto dispatch_types(Fn &&fn) {
    auto code = [](const std::string &s) -> int {
      if (s == "float32" || s == "f32")
        return 0;
      if (s == "float16" || s == "f16")
        return 1;
      if (s == "bfloat16" || s == "bf16")
        return 2;
      return -1;
    };
    int p = code(value_dtype_);
    int o = code(optimizer_dtype_);
    if (p < 0)
      throw std::invalid_argument("unsupported dtype: " + value_dtype_);
    if (o < 0)
      throw std::invalid_argument("unsupported opt_dtype: " + optimizer_dtype_);
    if (p == 0 && o == 0)
      return fn.template operator()<float, float>();
    if (p == 0 && o == 1)
      return fn.template operator()<float, se::float16_t>();
    if (p == 0 && o == 2)
      return fn.template operator()<float, se::bfloat16_t>();
    if (p == 1 && o == 0)
      return fn.template operator()<se::float16_t, float>();
    if (p == 1 && o == 1)
      return fn.template operator()<se::float16_t, se::float16_t>();
    if (p == 1 && o == 2)
      return fn.template operator()<se::float16_t, se::bfloat16_t>();
    if (p == 2 && o == 0)
      return fn.template operator()<se::bfloat16_t, float>();
    if (p == 2 && o == 1)
      return fn.template operator()<se::bfloat16_t, se::float16_t>();
    return fn.template operator()<se::bfloat16_t, se::bfloat16_t>();
  }

public:
  // ---- Constructor (defined in module.cpp to avoid cross-TU instantiation)
  // ----
  EmbeddingStoreWrapper(const se::EmbeddingConfig &emb_cfg,
                        std::optional<PyInitializer> init_cfg_opt,
                        std::optional<PyOptimizer> opt_cfg_opt,
                        std::string score_policy_name);

  // ---- Metadata (inline, trivial) ----

  size_t total_size() {
    size_t sz = 0;
    dispatch_types(
        [&]<typename P, typename O>() { sz = get_store<P, O>()->size(); });
    return sz;
  }

  size_t max_dim() const {
    if (dims_.empty())
      return 0;
    return std::ranges::max(dims_);
  }

  size_t num_slots() const { return dims_.size(); }

  std::vector<size_t> table_dims() const { return dims_; }

  size_t internal_shard_count() {
    size_t count = 0;
    dispatch_types([&]<typename P, typename O>() {
      count = get_store<P, O>()->internal_shard_count();
    });
    return count;
  }

  nb::dict storage_stats() {
    if (storage_->consume_spill_warning() &&
        PyErr_WarnEx(PyExc_RuntimeWarning,
                     "SparseEmbeddingSpillWarning: embedding pages are using "
                     "file-backed storage",
                     1) < 0) {
      throw nb::python_error();
    }
    auto stats = storage_->stats();
    nb::dict result;
    result["memory_bytes"] = stats.memory_bytes;
    result["spill_bytes"] = stats.spill_bytes;
    result["memory_limit_bytes"] = stats.memory_limit_bytes;
    result["spill_limit_bytes"] = stats.spill_limit_bytes;
    result["spill_count"] = stats.spill_count;
    result["spill_directory"] = storage_->spill_directory();
    result["swap_out_count"] = stats.swap_out_count;
    result["eviction_selection_count"] = stats.eviction_selection_count;
    result["eviction_evaluated_pages"] = stats.eviction_evaluated_pages;
    result["eviction_selection_ns"] = stats.eviction_selection_ns;
    result["eviction_candidate_invalidated_count"] =
        stats.eviction_candidate_invalidated_count;
    return result;
  }

  void reserve(size_t keys_count, int64_t dim) {
    if (dim <= 0)
      throw std::invalid_argument("dim must be > 0");
    std::error_code ec;
    dispatch_types([&]<typename P, typename O>() {
      ec = get_store<P, O>()->reserve(keys_count, static_cast<size_t>(dim));
    });
    check_error(ec);
  }

  void reserve_per_shard(const std::vector<size_t> &counts, int64_t dim) {
    if (dim <= 0)
      throw std::invalid_argument("dim must be > 0");
    std::error_code ec;
    dispatch_types([&]<typename P, typename O>() {
      ec = get_store<P, O>()->reserve_per_shard(
          std::span<const size_t>(counts.data(), counts.size()),
          static_cast<size_t>(dim));
    });
    check_error(ec);
  }

  // ---- Sync Operations (thin dispatchers) ----

  nb::tuple lookup(nb::object keys, int64_t dim) {
    if (dim <= 0)
      throw std::invalid_argument("dim must be > 0");
    return dispatch_types([&]<typename P, typename O>() {
      return SyncOpsImpl<P, O>::lookup(*this, keys, dim);
    });
  }

  nb::tuple lookup_or_insert(nb::object keys, int64_t dim) {
    if (dim <= 0)
      throw std::invalid_argument("dim must be > 0");
    return dispatch_types([&]<typename P, typename O>() {
      return SyncOpsImpl<P, O>::lookup_or_insert(*this, keys, dim);
    });
  }

  void assign(nb::object keys, nb::object values) {
    dispatch_types([&]<typename P, typename O>() {
      SyncOpsImpl<P, O>::assign(*this, keys, values);
    });
  }

  nb::object contains(nb::object keys) {
    return dispatch_types([&]<typename P, typename O>() {
      return SyncOpsImpl<P, O>::contains(*this, keys);
    });
  }

  void accumulate_gradients(nb::object keys, nb::object grads) {
    dispatch_types([&]<typename P, typename O>() {
      SyncOpsImpl<P, O>::accumulate_gradients(*this, keys, grads);
    });
  }

  void optimizer_step() {
    dispatch_types([&]<typename P, typename O>() {
      SyncOpsImpl<P, O>::optimizer_step(*this);
    });
  }

  void erase(nb::object keys) {
    dispatch_types([&]<typename P, typename O>() {
      SyncOpsImpl<P, O>::erase(*this, keys);
    });
  }

  void clear() {
    dispatch_types(
        [&]<typename P, typename O>() { SyncOpsImpl<P, O>::clear(*this); });
  }

  void assign_scores(nb::object keys, nb::object scores) {
    dispatch_types([&]<typename P, typename O>() {
      SyncOpsImpl<P, O>::assign_scores(*this, keys, scores);
    });
  }

  // ---- Async Operations (thin dispatchers) ----

  nb::object async_lookup(nb::object keys, int64_t dim) {
    if (dim <= 0)
      throw std::invalid_argument("dim must be > 0");
    return dispatch_types([&]<typename P, typename O>() -> nb::object {
      return AsyncOpsImpl<P, O>::async_lookup(*this, keys, dim);
    });
  }

  nb::object async_assign(nb::object keys, nb::object values) {
    return dispatch_types([&]<typename P, typename O>() -> nb::object {
      return AsyncOpsImpl<P, O>::async_assign(*this, keys, values);
    });
  }

  nb::object async_lookup_or_insert(nb::object keys, int64_t dim) {
    if (dim <= 0)
      throw std::invalid_argument("dim must be > 0");
    return dispatch_types([&]<typename P, typename O>() -> nb::object {
      return AsyncOpsImpl<P, O>::async_lookup_or_insert(*this, keys, dim);
    });
  }

  nb::object async_accumulate_gradients(nb::object keys, nb::object grads) {
    return dispatch_types([&]<typename P, typename O>() -> nb::object {
      return AsyncOpsImpl<P, O>::async_accumulate_gradients(*this, keys, grads);
    });
  }

  nb::object async_optimizer_step() {
    return dispatch_types([&]<typename P, typename O>() -> nb::object {
      return AsyncOpsImpl<P, O>::async_optimizer_step(*this);
    });
  }

  nb::object async_contains(nb::object keys) {
    return dispatch_types([&]<typename P, typename O>() -> nb::object {
      return AsyncOpsImpl<P, O>::async_contains(*this, keys);
    });
  }

  nb::object async_erase(nb::object keys) {
    return dispatch_types([&]<typename P, typename O>() -> nb::object {
      return AsyncOpsImpl<P, O>::async_erase(*this, keys);
    });
  }

  nb::object async_clear() {
    return dispatch_types([&]<typename P, typename O>() -> nb::object {
      return AsyncOpsImpl<P, O>::async_clear(*this);
    });
  }

  nb::object async_assign_scores(nb::object keys, nb::object scores) {
    return dispatch_types([&]<typename P, typename O>() -> nb::object {
      return AsyncOpsImpl<P, O>::async_assign_scores(*this, keys, scores);
    });
  }

  // ---- Checkpoint Operations ----

#ifdef SE_ENABLE_LANCE_CHECKPOINTER
  void save_checkpoint(const std::string &uri, const std::string &mode,
                       uint64_t min_score, bool include_optimizer_state,
                       bool use_compression) {
    dispatch_types([&]<typename P, typename O>() {
      CheckpointOpsImpl<P, O>::save_checkpoint(*this, uri, mode, min_score,
                                               include_optimizer_state,
                                               use_compression);
    });
  }

  void load_checkpoint(const std::string &uri, uint64_t version, bool clear,
                       bool load_optimizer_state,
                       nb::object filter_fn = nb::none(),
                       uint32_t target_shard_id = 0,
                       size_t batch_size = 65536) {
    dispatch_types([&]<typename P, typename O>() {
      CheckpointOpsImpl<P, O>::load_checkpoint(*this, uri, version, clear,
                                               load_optimizer_state, filter_fn,
                                               target_shard_id, batch_size);
    });
  }

  void set_checkpoint_factories(se::CheckpointWriterFactory wf,
                                se::CheckpointReaderFactory rf) {
    checkpoint_writer_factory_ = std::move(wf);
    checkpoint_reader_factory_ = std::move(rf);
  }
#endif // SE_ENABLE_LANCE_CHECKPOINTER

  void export_batches_dlpack(nb::callable visitor, size_t batch_size = 65536,
                             bool include_optimizer_state = false) {
    dispatch_types([&]<typename P, typename O>() {
      CheckpointOpsImpl<P, O>::export_batches_dlpack(*this, visitor, batch_size,
                                                     include_optimizer_state);
    });
  }

  void export_batches_arrow(nb::callable visitor, size_t batch_size = 65536,
                            bool include_optimizer_state = false) {
    dispatch_types([&]<typename P, typename O>() {
      CheckpointOpsImpl<P, O>::export_batches_arrow(*this, visitor, batch_size,
                                                    include_optimizer_state);
    });
  }

  void assign_opt_state(nb::object keys_obj, nb::object opt_states_obj,
                        size_t dimension, nb::object steps_obj) {
    dispatch_types([&]<typename P, typename O>() {
      CheckpointOpsImpl<P, O>::assign_opt_state(*this, keys_obj, opt_states_obj,
                                                dimension, steps_obj);
    });
  }
};

} // namespace se::py::detail

#endif // SPARSE_EMBEDDING_MODULE_WRAPPER_HPP_
