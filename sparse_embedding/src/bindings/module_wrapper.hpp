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

#include <dlpack/dlpack.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <proxy/proxy.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <exec/static_thread_pool.hpp>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <system_error>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "sparse_embedding/src/bindings/construct_api.hpp"
#include "sparse_embedding/src/bindings/dlpack_utils.hpp"
#include "sparse_embedding/src/bindings/dtype_dispatch.hpp"
#if defined(SE_ENABLE_LANCE_C)
#include "sparse_embedding/src/checkpoint/lance/storage_options.hpp"
#endif
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
#include "sparse_embedding/src/optimizer/optimizer_config.hpp"

namespace se::py::detail {

namespace nb = nanobind;

// =============================================================================
// Utilities
// =============================================================================

inline void check_error(std::error_code ec) {
  if (ec)
    throw std::runtime_error(ec.message());
}

template <typename Fn>
auto dispatch_on_key(const DLManagedTensor *dlm,
                     KeyDType /*expected_key_dtype*/, Fn &&fn) {
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

inline std::vector<int64_t> &get_tls_row_lengths() {
  thread_local std::vector<int64_t> tls_row_lengths;
  return tls_row_lengths;
}

// =============================================================================
// Builder Proxies for Initializers and Optimizers
// =============================================================================

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
#if defined(SE_ENABLE_LANCE_C)
  static void save_checkpoint(EmbeddingStoreWrapper &self,
                              const std::string &uri, const std::string &mode,
                              uint64_t min_score, bool include_optimizer_state,
                              bool use_compression);
  static void load_checkpoint(EmbeddingStoreWrapper &self,
                              const std::string &uri, uint64_t version,
                              bool clear, bool load_optimizer_state,
                              nb::object filter_fn, uint32_t target_shard_id,
                              size_t batch_size);
  static nb::dict write_native_fragment(
      EmbeddingStoreWrapper &self, const std::string &uri,
      uint32_t participant_index, const std::vector<uint32_t> &dimensions,
      const std::string &embedding_dtype, const std::string &optimizer_dtype,
      const std::string &score_policy, uint32_t protocol_version,
      const std::string &schema_digest, bool include_page_images = false);
  static void
  restore_native_entries(EmbeddingStoreWrapper &self, const std::string &uri,
                         uint64_t version, nb::object selected_keys,
                         const se::LanceStorageOptions &storage_options,
                         bool load_optimizer_state);
  static void
  restore_native_page_images(EmbeddingStoreWrapper &self,
                             const std::vector<std::string> &uris,
                             uint64_t version, uint32_t participant_index,
                             const se::LanceStorageOptions &storage_options,
                             bool load_optimizer_state);
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

#define SE_DECLARE_TYPED_BINDING_INSTANCES(PARAM_T, OPT_T)                     \
  extern template struct SyncOpsImpl<PARAM_T, OPT_T>;                          \
  extern template struct AsyncOpsImpl<PARAM_T, OPT_T>;

SE_DECLARE_TYPED_BINDING_INSTANCES(float, float)
SE_DECLARE_TYPED_BINDING_INSTANCES(float, se::float16_t)
SE_DECLARE_TYPED_BINDING_INSTANCES(float, se::bfloat16_t)
SE_DECLARE_TYPED_BINDING_INSTANCES(se::float16_t, float)
SE_DECLARE_TYPED_BINDING_INSTANCES(se::float16_t, se::float16_t)
SE_DECLARE_TYPED_BINDING_INSTANCES(se::float16_t, se::bfloat16_t)
SE_DECLARE_TYPED_BINDING_INSTANCES(se::bfloat16_t, float)
SE_DECLARE_TYPED_BINDING_INSTANCES(se::bfloat16_t, se::float16_t)
SE_DECLARE_TYPED_BINDING_INSTANCES(se::bfloat16_t, se::bfloat16_t)

#define SE_DECLARE_TYPED_CHECKPOINT_INSTANCE(PARAM_T, OPT_T)                   \
  extern template struct CheckpointOpsImpl<PARAM_T, OPT_T>;

SE_DECLARE_TYPED_CHECKPOINT_INSTANCE(float, float)
SE_DECLARE_TYPED_CHECKPOINT_INSTANCE(float, se::float16_t)
SE_DECLARE_TYPED_CHECKPOINT_INSTANCE(float, se::bfloat16_t)
SE_DECLARE_TYPED_CHECKPOINT_INSTANCE(se::float16_t, float)
SE_DECLARE_TYPED_CHECKPOINT_INSTANCE(se::float16_t, se::float16_t)
SE_DECLARE_TYPED_CHECKPOINT_INSTANCE(se::float16_t, se::bfloat16_t)
SE_DECLARE_TYPED_CHECKPOINT_INSTANCE(se::bfloat16_t, float)
SE_DECLARE_TYPED_CHECKPOINT_INSTANCE(se::bfloat16_t, se::float16_t)
SE_DECLARE_TYPED_CHECKPOINT_INSTANCE(se::bfloat16_t, se::bfloat16_t)

#undef SE_DECLARE_TYPED_CHECKPOINT_INSTANCE

#undef SE_DECLARE_TYPED_BINDING_INSTANCES

// =============================================================================
// EmbeddingStoreWrapper
// =============================================================================

class EmbeddingStoreWrapper {
  friend void construct_store(EmbeddingStoreWrapper &, const PyInitializer &,
                              const std::optional<PyOptimizer> &, size_t,
                              se::StoreConstructionKind,
                              const se::EmbeddingConfig &);
  template <typename, typename> friend struct SyncOpsImpl;
  template <typename, typename> friend struct AsyncOpsImpl;
  template <typename, typename> friend struct CheckpointOpsImpl;

  using StoreProxyVariant = se::StoreProxyVariant;
  StoreProxyVariant store_;
  exec::static_thread_pool async_pool_;
  ValueDType value_dtype_code_;
  ValueDType optimizer_dtype_code_;
  ValueDType gradient_dtype_code_;
  float gradient_scale_;
  bool optimizer_enabled_;
  KeyDType key_type_enum_;
  std::string score_policy_name_;
  std::vector<size_t> dims_;
  se::PageStorage storage_;

  se::CheckpointWriterFactory checkpoint_writer_factory_;
  se::CheckpointReaderFactory checkpoint_reader_factory_;

  template <typename ParamT, typename OptT> auto &get_store() {
    return std::get<pro::proxy<se::EmbeddingTableFacade<ParamT, OptT>>>(store_);
  }

  template <typename Fn> auto dispatch_val(Fn &&fn) {
    return dispatch_value_dtype(value_dtype_code_, std::forward<Fn>(fn));
  }

  template <typename Fn> auto dispatch_types(Fn &&fn) {
    return dispatch_dtype_pair(value_dtype_code_, optimizer_dtype_code_,
                               std::forward<Fn>(fn));
  }

public:
  // ---- Constructor (defined in module.cpp to avoid cross-TU instantiation)
  // ----
  EmbeddingStoreWrapper(const se::EmbeddingConfig &emb_cfg,
                        std::optional<PyInitializer> init_cfg_opt,
                        std::optional<PyOptimizer> opt_cfg_opt,
                        std::string score_policy_name);

  size_t total_size();
  size_t max_dim() const;
  size_t num_slots() const;
  std::vector<size_t> table_dims() const;
  std::string value_dtype() const;
  std::string optimizer_dtype() const;
  bool optimizer_enabled() const;
  std::string score_policy() const;
  size_t internal_shard_count();
  nb::dict storage_stats();
  void reserve(size_t keys_count, int64_t dim);
  void reserve_per_shard(const std::vector<size_t> &counts, int64_t dim);
  nb::tuple lookup(nb::object keys, int64_t dim);
  nb::tuple lookup_or_insert(nb::object keys, int64_t dim);
  void assign(nb::object keys, nb::object values);
  nb::object contains(nb::object keys);
  void accumulate_gradients(nb::object keys, nb::object grads);
  void optimizer_step();
  void erase(nb::object keys);
  void clear();
  void assign_scores(nb::object keys, nb::object scores);
  size_t compact_page(size_t dim, uint64_t page_id);
  size_t promote_hot_keys(size_t dim, uint64_t page_id, size_t maximum_keys,
                          uint64_t minimum_score);
  size_t promoted_hot_page_size(size_t dim);
  nb::list export_page_images();
  void import_page_images(nb::list pages);
  nb::object async_lookup(nb::object keys, int64_t dim);
  nb::object async_assign(nb::object keys, nb::object values);
  nb::object async_lookup_or_insert(nb::object keys, int64_t dim);
  nb::object async_accumulate_gradients(nb::object keys, nb::object grads);
  nb::object async_optimizer_step();
  nb::object async_contains(nb::object keys);
  nb::object async_erase(nb::object keys);
  nb::object async_clear();
  nb::object async_assign_scores(nb::object keys, nb::object scores);

  // ---- Checkpoint Operations ----
#ifdef SE_ENABLE_LANCE_C
  void save_checkpoint(const std::string &uri, const std::string &mode,
                       uint64_t min_score, bool include_optimizer_state,
                       bool use_compression);
  void load_checkpoint(const std::string &uri, uint64_t version, bool clear,
                       bool load_optimizer_state,
                       nb::object filter_fn = nb::none(),
                       uint32_t target_shard_id = 0, size_t batch_size = 65536);
  nb::dict write_native_fragment(
      const std::string &uri, uint32_t participant_index,
      const std::vector<uint32_t> &dimensions,
      const std::string &embedding_dtype, const std::string &optimizer_dtype,
      const std::string &score_policy, uint32_t protocol_version,
      const std::string &schema_digest, bool include_page_images = false);
  void restore_native_entries(const std::string &uri, uint64_t version,
                              nb::object selected_keys,
                              const se::LanceStorageOptions &storage_options,
                              bool load_optimizer_state);
  void
  restore_native_page_images(const std::vector<std::string> &uris,
                             uint64_t version, uint32_t participant_index,
                             const se::LanceStorageOptions &storage_options,
                             bool load_optimizer_state);

  void set_checkpoint_factories(se::CheckpointWriterFactory wf,
                                se::CheckpointReaderFactory rf);
#endif

  void export_batches_dlpack(nb::callable visitor, size_t batch_size = 65536,
                             bool include_optimizer_state = false);
  void export_batches_arrow(nb::callable visitor, size_t batch_size = 65536,
                            bool include_optimizer_state = false);
  void assign_opt_state(nb::object keys_obj, nb::object opt_states_obj,
                        size_t dimension, nb::object steps_obj);
};

} // namespace se::py::detail

#endif // SPARSE_EMBEDDING_MODULE_WRAPPER_HPP_
