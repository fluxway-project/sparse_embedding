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

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <dlpack/dlpack.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <utility>
#include <variant>

#include "sparse_embedding/src/backend/hogwild/hogwild_embedding_store.hpp"
#include "sparse_embedding/src/bindings/async_bridge.hpp"
#include "sparse_embedding/src/bindings/dlpack_utils.hpp"
#include "sparse_embedding/src/bindings/python_wake_manager.hpp"
#include "sparse_embedding/src/checkpoint/checkpoint_interface.hpp"
#include "sparse_embedding/src/checkpoint/lance/lance_checkpoint.hpp"
#include "sparse_embedding/src/common/function.hpp"
#include "sparse_embedding/src/eviction/score_policy.hpp"
#include "sparse_embedding/src/initializer/initializer_config.hpp"
#include "sparse_embedding/src/initializer/uniform_initializer.hpp"
#include "sparse_embedding/src/initializer/zeros_initializer.hpp"
#include "sparse_embedding/src/interface/embedding_config.hpp"
#include "sparse_embedding/src/interface/embedding_table_interface.hpp"
#include "sparse_embedding/src/ops/tensor_ops.hpp"
#include "sparse_embedding/src/optimizer/adam_optimizer.hpp"
#include "sparse_embedding/src/optimizer/optimizer_config.hpp"
#include "sparse_embedding/src/optimizer/sgd_optimizer.hpp"

namespace nb = nanobind;

namespace {

inline void check_error(std::error_code ec) {
  if (ec)
    throw std::runtime_error(ec.message());
}

// KeyDType detection
// =============================================================================

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
auto dispatch_on_key(const DLManagedTensor *dlm, Fn &&fn) {
  auto kdt = detect_key_dtype(dlm);
  if (kdt == KeyDType::kInt32 || kdt == KeyDType::kUInt32)
    throw std::invalid_argument(
        "embedding table operations only support 64-bit keys");
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

// =============================================================================
// Builder Proxies for Initializers and Optimizers
// =============================================================================

struct PyOptimizer {
  std::shared_ptr<pro::proxy<se::OptimizerBuilderFacade>> builder;
};

struct PyInitializer {
  std::shared_ptr<pro::proxy<se::InitializerBuilderFacade>> builder;
};

template <template <typename> class OptT, typename ConfigT>
struct DefaultOptimizerBuilder {
  ConfigT config;

  pro::proxy<se::OptimizerFacade<float>> build_f32(size_t dim) {
    return pro::make_proxy<se::OptimizerFacade<float>, OptT<float>>(dim,
                                                                    config);
  }
  pro::proxy<se::OptimizerFacade<se::float16_t>> build_f16(size_t dim) {
    return pro::make_proxy<se::OptimizerFacade<se::float16_t>,
                           OptT<se::float16_t>>(dim, config);
  }
  pro::proxy<se::OptimizerFacade<se::bfloat16_t>> build_bf16(size_t dim) {
    return pro::make_proxy<se::OptimizerFacade<se::bfloat16_t>,
                           OptT<se::bfloat16_t>>(dim, config);
  }
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

// =============================================================================
// EmbeddingStoreWrapper: proxy wrapper for orthogonality
// =============================================================================

class EmbeddingStoreWrapper {
  using StoreProxyVariant =
      std::variant<pro::proxy<se::EmbeddingTableFacade<float>>,
                   pro::proxy<se::EmbeddingTableFacade<se::float16_t>>,
                   pro::proxy<se::EmbeddingTableFacade<se::bfloat16_t>>>;
  StoreProxyVariant store_;
  exec::static_thread_pool async_pool_;
  std::unordered_map<uint32_t, size_t> slot_dims_;
  size_t max_dim_ = 0;
  std::string dtype_;

  template <typename ValT> auto &get_store() {
    return std::get<pro::proxy<se::EmbeddingTableFacade<ValT>>>(store_);
  }

  template <typename Fn> auto dispatch_val(Fn &&fn) {
    if (dtype_ == "float32" || dtype_ == "f32")
      return fn.template operator()<float>();
    if (dtype_ == "float16" || dtype_ == "f16")
      return fn.template operator()<se::float16_t>();
    if (dtype_ == "bfloat16" || dtype_ == "bf16")
      return fn.template operator()<se::bfloat16_t>();
    throw std::invalid_argument("unsupported dtype: " + dtype_);
  }

  size_t compute_output_size(auto ks, std::span<int64_t> out_row_lengths) {
    size_t total = 0;
    for (size_t i = 0; i < ks.size(); ++i) {
      uint64_t key = static_cast<uint64_t>(ks[i]);
      uint32_t slot_id = static_cast<uint32_t>((key >> 48) & 0xFFFF);
      auto dim = static_cast<int64_t>(slot_dims_.at(slot_id));
      out_row_lengths[i] = dim;
      total += dim;
    }
    return total;
  }

public:
  EmbeddingStoreWrapper(const se::EmbeddingConfig &emb_cfg,
                        PyInitializer init_cfg, PyOptimizer opt_cfg,
                        std::string score_policy_name)
      : async_pool_((emb_cfg.num_threads == 0)
                        ? std::max(1u, std::thread::hardware_concurrency())
                        : emb_cfg.num_threads),
        slot_dims_(emb_cfg.slot_dims), dtype_(emb_cfg.dtype) {
    for (auto &[slot, dim] : slot_dims_) {
      if (dim > max_dim_)
        max_dim_ = dim;
    }

    size_t nt = (emb_cfg.num_threads == 0)
                    ? std::max(1u, std::thread::hardware_concurrency())
                    : emb_cfg.num_threads;

    auto construct = [&]<typename ValT>() {
      pro::proxy<se::InitializerFacade<ValT>> init_proxy;

      if constexpr (std::is_same_v<ValT, float>) {
        init_proxy = (*init_cfg.builder)->build_f32();
      } else if constexpr (std::is_same_v<ValT, se::float16_t>) {
        init_proxy = (*init_cfg.builder)->build_f16();
      } else if constexpr (std::is_same_v<ValT, se::bfloat16_t>) {
        init_proxy = (*init_cfg.builder)->build_bf16();
      }

      using OptFacade = se::OptimizerFacade<ValT>;
      using OptFactory = se::Function<pro::proxy<OptFacade>(size_t)>;

      OptFactory opt_factory;

      if constexpr (std::is_same_v<ValT, float>) {
        opt_factory =
            pro::make_proxy<se::FunctionFacade<pro::proxy<OptFacade>(size_t)>>(
                [builder = opt_cfg.builder](size_t d) mutable {
                  return (*builder)->build_f32(d);
                });
      } else if constexpr (std::is_same_v<ValT, se::float16_t>) {
        opt_factory =
            pro::make_proxy<se::FunctionFacade<pro::proxy<OptFacade>(size_t)>>(
                [builder = opt_cfg.builder](size_t d) mutable {
                  return (*builder)->build_f16(d);
                });
      } else if constexpr (std::is_same_v<ValT, se::bfloat16_t>) {
        opt_factory =
            pro::make_proxy<se::FunctionFacade<pro::proxy<OptFacade>(size_t)>>(
                [builder = opt_cfg.builder](size_t d) mutable {
                  return (*builder)->build_bf16(d);
                });
      }

      auto construct_with_policy = [&]<typename Policy>(Policy policy) {
        using StoreType = se::HogwildEmbeddingStore<ValT, Policy>;
        store_ = pro::make_proxy<se::EmbeddingTableFacade<ValT>, StoreType>(
            slot_dims_, nt, std::move(init_proxy), emb_cfg.max_active_keys,
            std::move(opt_factory), std::move(policy));
      };

      if (score_policy_name == "none") {
        construct_with_policy(se::NoScorePolicy{});
      } else if (score_policy_name == "lfu") {
        construct_with_policy(se::LfuScorePolicy{});
      } else if (score_policy_name == "lru") {
        construct_with_policy(se::LruScorePolicy{});
      } else if (score_policy_name == "manual") {
        construct_with_policy(se::ManualScorePolicy{});
      } else {
        throw std::invalid_argument("unknown score policy: " +
                                    score_policy_name);
      }
    };

    dispatch_val(construct);
  }

  // ---- Metadata ----

  size_t total_size() {
    size_t sz = 0;
    dispatch_val([&]<typename ValT>() { sz = get_store<ValT>()->size(); });
    return sz;
  }

  size_t max_dim() { return max_dim_; }

  size_t num_slots() { return slot_dims_.size(); }

  size_t dim_for_slot(uint32_t slot_id) { return slot_dims_.at(slot_id); }

  // ---- get_dims ----

  nb::object get_dims(nb::object keys) {
    auto [dlm_keys, cap_keys] = se::py::extract_dlpack_tensor(keys);
    size_t n = get_dlm_size(dlm_keys);
    int64_t shape[1] = {static_cast<int64_t>(n)};
    DLManagedTensor *dlm =
        se::py::create_output_tensor<int64_t>(std::span(shape));
    auto *out = static_cast<int64_t *>(dlm->dl_tensor.data);

    dispatch_on_key(dlm_keys, [&](auto ks) {
      for (size_t i = 0; i < n; ++i) {
        uint64_t key = static_cast<uint64_t>(ks[i]);
        uint32_t slot_id = static_cast<uint32_t>((key >> 48) & 0xFFFF);
        out[i] = static_cast<int64_t>(slot_dims_.at(slot_id));
      }
    });

    return se::py::wrap_output_tensor(dlm);
  }

  // ---- Sync Lookup ----

  nb::tuple lookup(nb::object keys) {
    auto [dlm_keys, cap_keys] = se::py::extract_dlpack_tensor(keys);
    size_t n = get_dlm_size(dlm_keys);
    DLManagedTensor *vals_dlm = nullptr;
    DLManagedTensor *lens_dlm = nullptr;
    std::error_code ec;

    {
      nb::gil_scoped_release release;
      dispatch_val([&]<typename ValT>() {
        dispatch_on_key(dlm_keys, [&](auto ks) {
          auto row_lengths = std::make_unique_for_overwrite<int64_t[]>(n);
          size_t total = compute_output_size(ks, {row_lengths.get(), n});

          int64_t vals_shape[] = {static_cast<int64_t>(total)};
          vals_dlm = se::py::create_output_tensor<ValT>(vals_shape);
          lens_dlm =
              se::py::wrap_unique_ptr_as_tensor(std::move(row_lengths), n);

          auto vals_span = std::span<ValT>(
              static_cast<ValT *>(vals_dlm->dl_tensor.data), total);
          ec = get_store<ValT>()->forward_lookup(
              ks, vals_span,
              {static_cast<const int64_t *>(lens_dlm->dl_tensor.data), n});
        });
      });
    }

    if (ec) {
      if (vals_dlm)
        vals_dlm->deleter(vals_dlm);
      if (lens_dlm)
        lens_dlm->deleter(lens_dlm);
      throw std::runtime_error(ec.message());
    }

    return nb::make_tuple(se::py::wrap_output_tensor(vals_dlm),
                          se::py::wrap_output_tensor(lens_dlm));
  }

  // ---- Sync lookup_or_insert ----

  nb::tuple lookup_or_insert(nb::object keys) {
    auto [dlm_keys, cap_keys] = se::py::extract_dlpack_tensor(keys);
    size_t n = get_dlm_size(dlm_keys);
    DLManagedTensor *vals_dlm = nullptr;
    DLManagedTensor *lens_dlm = nullptr;
    DLManagedTensor *mask_dlm = nullptr;
    std::error_code ec;

    {
      nb::gil_scoped_release release;
      dispatch_val([&]<typename ValT>() {
        dispatch_on_key(dlm_keys, [&](auto ks) {
          int64_t lens_shape[] = {static_cast<int64_t>(n)};
          lens_dlm = se::py::create_output_tensor<int64_t>(lens_shape);
          auto *lens_out = static_cast<int64_t *>(lens_dlm->dl_tensor.data);
          size_t total = compute_output_size(ks, {lens_out, n});

          int64_t vals_shape[] = {static_cast<int64_t>(total)};
          vals_dlm = se::py::create_output_tensor<ValT>(vals_shape);
          int64_t mask_shape[] = {static_cast<int64_t>(n)};
          mask_dlm = se::py::create_output_tensor<uint8_t>(mask_shape);

          auto vals_span = std::span<ValT>(
              static_cast<ValT *>(vals_dlm->dl_tensor.data), total);

          auto *mask_out = static_cast<uint8_t *>(mask_dlm->dl_tensor.data);
          ec = get_store<ValT>()->lookup_or_insert(
              ks, vals_span, {lens_out, n}, std::span<uint8_t>(mask_out, n));
        });
      });
    }

    if (ec) {
      if (vals_dlm)
        vals_dlm->deleter(vals_dlm);
      if (lens_dlm)
        lens_dlm->deleter(lens_dlm);
      if (mask_dlm)
        mask_dlm->deleter(mask_dlm);
      throw std::runtime_error(ec.message());
    }

    return nb::make_tuple(se::py::wrap_output_tensor(vals_dlm),
                          se::py::wrap_output_tensor(lens_dlm),
                          se::py::wrap_output_tensor(mask_dlm));
  }

  // ---- Sync assign ----

  void assign(nb::object keys, nb::object values) {
    auto [dlm_keys, cap_keys] = se::py::extract_dlpack_tensor(keys);
    auto [dlm_values, cap_values] = se::py::extract_dlpack_tensor(values);
    size_t n = get_dlm_size(dlm_keys);
    std::error_code ec;
    {
      nb::gil_scoped_release release;
      dispatch_val([&]<typename ValT>() {
        dispatch_on_key(dlm_keys, [&](auto ks) {
          auto val_span = se::py::dlpack_to_span<const ValT>(dlm_values);
          if (dlm_values->dl_tensor.ndim == 2) {
            size_t dim = dlm_values->dl_tensor.shape[1];
            auto row_lengths = std::make_unique_for_overwrite<int64_t[]>(n);
            std::fill_n(row_lengths.get(), n, dim);
            ec =
                get_store<ValT>()->assign(ks, val_span, {row_lengths.get(), n});
          } else if (dlm_values->dl_tensor.ndim == 1) {
            auto row_lengths = std::make_unique_for_overwrite<int64_t[]>(n);
            compute_output_size(ks, {row_lengths.get(), n});
            ec =
                get_store<ValT>()->assign(ks, val_span, {row_lengths.get(), n});
          } else {
            throw std::invalid_argument("values must be 1D (ragged) or 2D");
          }
        });
      });
    }
    check_error(ec);
  }

  // ---- Sync contains ----

  nb::object contains(nb::object keys) {
    auto [dlm_keys, cap_keys] = se::py::extract_dlpack_tensor(keys);
    size_t n = get_dlm_size(dlm_keys);
    int64_t shape[1] = {static_cast<int64_t>(n)};
    DLManagedTensor *dlm =
        se::py::create_output_tensor<uint8_t>(std::span(shape));
    auto *out = static_cast<uint8_t *>(dlm->dl_tensor.data);

    std::error_code ec;
    {
      nb::gil_scoped_release release;
      dispatch_val([&]<typename ValT>() {
        dispatch_on_key(dlm_keys, [&](auto ks) {
          ec = get_store<ValT>()->contains(ks, std::span<uint8_t>(out, n));
        });
      });
    }

    check_error(ec);
    return se::py::wrap_output_tensor(dlm);
  }

  // ---- Accumulate gradients ----

  void accumulate_gradients(nb::object keys, nb::object grads) {
    auto [dlm_keys, cap_keys] = se::py::extract_dlpack_tensor(keys);
    auto [dlm_grads, cap_grads] = se::py::extract_dlpack_tensor(grads);
    size_t n = get_dlm_size(dlm_keys);

    std::error_code ec;
    {
      nb::gil_scoped_release release;
      dispatch_val([&]<typename ValT>() {
        auto grads_span = se::py::dlpack_to_span<const ValT>(dlm_grads);
        dispatch_on_key(dlm_keys, [&](auto ks) {
          auto row_lengths = std::make_unique_for_overwrite<int64_t[]>(n);
          compute_output_size(ks, {row_lengths.get(), n});
          ec = get_store<ValT>()->accumulate_gradients(ks, grads_span,
                                                       {row_lengths.get(), n});
        });
      });
    }
    check_error(ec);
  }

  // ---- Optimizer Step ----

  void optimizer_step() {
    nb::gil_scoped_release release;
    std::error_code ec;
    dispatch_val(
        [&]<typename ValT>() { ec = get_store<ValT>()->optimizer_step(); });
    check_error(ec);
  }

  // ---- Sync erase ----

  void erase(nb::object keys) {
    auto [dlm_keys, cap_keys] = se::py::extract_dlpack_tensor(keys);
    std::error_code ec;
    {
      nb::gil_scoped_release release;
      dispatch_val([&]<typename ValT>() {
        dispatch_on_key(dlm_keys,
                        [&](auto ks) { ec = get_store<ValT>()->erase(ks); });
      });
    }
    check_error(ec);
  }

  // ---- Sync clear ----

  void clear() {
    nb::gil_scoped_release release;
    dispatch_val([&]<typename ValT>() { get_store<ValT>()->clear(); });
  }

  // ---- Assign scores ----

  void assign_scores(nb::object keys, nb::object scores) {
    auto [dlm_keys, cap_keys] = se::py::extract_dlpack_tensor(keys);
    auto [dlm_scores, cap_scores] = se::py::extract_dlpack_tensor(scores);
    std::error_code ec;
    auto scores_span = se::py::dlpack_to_span<const uint64_t>(dlm_scores);
    {
      nb::gil_scoped_release release;
      dispatch_val([&]<typename ValT>() {
        dispatch_on_key(dlm_keys, [&](auto ks) {
          ec = get_store<ValT>()->assign_scores(ks, scores_span);
        });
      });
    }
    check_error(ec);
  }

  // ---- Async lookup ----

  nb::object async_lookup(nb::object keys) {
    auto [dlm_keys, cap_keys] = se::py::extract_dlpack_tensor(keys);
    size_t n = get_dlm_size(dlm_keys);
    auto kdt = detect_key_dtype(dlm_keys);

    const void *keys_data =
        reinterpret_cast<const char *>(dlm_keys->dl_tensor.data) +
        dlm_keys->dl_tensor.byte_offset;

    auto wrapper_ptr = this;

    return dispatch_val([&]<typename ValT>() -> nb::object {
      return se::py::async_dispatch(
          async_pool_,
          [wrapper_ptr, n, kdt, keys_data,
           keys_keeper = std::make_shared<se::py::AsyncPyObjectKeeper>(
               cap_keys)]() -> std::pair<DLManagedTensor *, DLManagedTensor *> {
            DLManagedTensor *vals_dlm = nullptr;
            DLManagedTensor *lens_dlm = nullptr;

            try {
              auto do_lookup = [&](auto ks) {
                auto row_lengths = std::make_unique_for_overwrite<int64_t[]>(n);
                size_t total = wrapper_ptr->compute_output_size(
                    ks, {row_lengths.get(), n});

                int64_t vals_shape[] = {static_cast<int64_t>(total)};
                vals_dlm = se::py::create_output_tensor<ValT>(vals_shape);
                lens_dlm = se::py::wrap_unique_ptr_as_tensor(
                    std::move(row_lengths), n);

                auto vals_span = std::span<ValT>(
                    static_cast<ValT *>(vals_dlm->dl_tensor.data), total);
                auto ec = wrapper_ptr->get_store<ValT>()->forward_lookup(
                    ks, vals_span,
                    {static_cast<const int64_t *>(lens_dlm->dl_tensor.data),
                     n});
                if (ec)
                  throw std::runtime_error(ec.message());
              };

              if (kdt == KeyDType::kInt64) {
                auto ks = std::span<const int64_t>(
                    static_cast<const int64_t *>(keys_data), n);
                do_lookup(ks);
              } else {
                auto ks = std::span<const uint64_t>(
                    static_cast<const uint64_t *>(keys_data), n);
                do_lookup(ks);
              }
            } catch (...) {
              if (vals_dlm)
                vals_dlm->deleter(vals_dlm);
              if (lens_dlm)
                lens_dlm->deleter(lens_dlm);
              throw;
            }
            return {vals_dlm, lens_dlm};
          },
          [](nb::object future,
             std::pair<DLManagedTensor *, DLManagedTensor *> result) {
            auto [vals_dlm, lens_dlm] = result;
            future.attr("set_result")(
                nb::make_tuple(se::py::wrap_output_tensor(vals_dlm),
                               se::py::wrap_output_tensor(lens_dlm)));
          });
    });
  }

  // ---- Async assign ----

  nb::object async_assign(nb::object keys, nb::object values) {
    auto [dlm_keys, cap_keys] = se::py::extract_dlpack_tensor(keys);
    auto [dlm_values, cap_values] = se::py::extract_dlpack_tensor(values);
    size_t n = get_dlm_size(dlm_keys);
    auto kdt = detect_key_dtype(dlm_keys);
    auto val_ndim = dlm_values->dl_tensor.ndim;
    auto val_dim1 =
        (val_ndim == 2) ? dlm_values->dl_tensor.shape[1] : size_t(0);

    const void *keys_data =
        reinterpret_cast<const char *>(dlm_keys->dl_tensor.data) +
        dlm_keys->dl_tensor.byte_offset;
    const void *values_data =
        reinterpret_cast<const char *>(dlm_values->dl_tensor.data) +
        dlm_values->dl_tensor.byte_offset;

    auto wrapper_ptr = this;

    return dispatch_val([&]<typename ValT>() -> nb::object {
      return se::py::async_dispatch(
          async_pool_,
          [wrapper_ptr, n, kdt, keys_data, values_data, val_ndim, val_dim1,
           keys_keeper =
               std::make_shared<se::py::AsyncPyObjectKeeper>(cap_keys),
           values_keeper = std::make_shared<se::py::AsyncPyObjectKeeper>(
               cap_values)]() -> std::error_code {
            auto do_assign = [&](auto ks) -> std::error_code {
              if (val_ndim == 2) {
                auto row_lengths = std::make_unique_for_overwrite<int64_t[]>(n);
                std::fill_n(row_lengths.get(), n, val_dim1);
                return wrapper_ptr->get_store<ValT>()->assign(
                    ks,
                    std::span<const ValT>(
                        static_cast<const ValT *>(values_data), n * val_dim1),
                    {row_lengths.get(), n});
              }
              auto row_lengths = std::make_unique_for_overwrite<int64_t[]>(n);
              wrapper_ptr->compute_output_size(ks, {row_lengths.get(), n});
              size_t total = 0;
              for (size_t i = 0; i < n; ++i)
                total += row_lengths[i];
              return wrapper_ptr->get_store<ValT>()->assign(
                  ks,
                  std::span<const ValT>(static_cast<const ValT *>(values_data),
                                        total),
                  {row_lengths.get(), n});
            };

            if (kdt == KeyDType::kInt64) {
              return do_assign(std::span<const int64_t>(
                  static_cast<const int64_t *>(keys_data), n));
            }
            return do_assign(std::span<const uint64_t>(
                static_cast<const uint64_t *>(keys_data), n));
          },
          [](nb::object future, std::error_code ec) {
            if (!ec) {
              future.attr("set_result")(nb::none());
            } else {
              nb::module_ builtins = nb::module_::import_("builtins");
              future.attr("set_exception")(
                  builtins.attr("RuntimeError")(ec.message()));
            }
          });
    });
  }

  void save_checkpoint(const std::string &uri, const std::string &mode,
                       uint64_t min_score, bool include_optimizer_state,
                       bool use_compression) {
    se::CheckpointExportOptions options;
    options.mode = (mode == "incremental")
                       ? se::CheckpointExportOptions::Mode::kIncremental
                       : se::CheckpointExportOptions::Mode::kFull;
    options.min_score = min_score;
    options.include_optimizer_state = include_optimizer_state;
    options.use_compression = use_compression;

    auto writer = checkpoint_writer_factory_;
    if (!writer.has_value()) {
      throw std::runtime_error("checkpoint backend not configured");
    }
    bool has_score_policy = false;
    dispatch_val([&]<typename ValT>() {
      has_score_policy = get_store<ValT>()->has_score_policy();
    });

    {
      nb::gil_scoped_release release;
      dispatch_val([&]<typename ValT>() {
        writer->begin(uri, 0, 0, has_score_policy, options);

        for (auto &[slot_id, dim] : slot_dims_) {
          // begin this slot
          writer->begin(uri, static_cast<uint32_t>(slot_id), 0,
                        has_score_policy, options);

          get_store<ValT>()->export_batch(
              pro::make_proxy<se::FunctionFacade<bool(
                  std::span<const uint64_t>, std::span<const ValT>,
                  std::span<const uint64_t>)>>(
                  [&](std::span<const uint64_t> keys,
                      std::span<const ValT> vals,
                      std::span<const uint64_t> scores) -> bool {
                    // Filter by min_score for incremental mode
                    if (options.mode ==
                            se::CheckpointExportOptions::Mode::kIncremental &&
                        has_score_policy) {
                      std::vector<uint64_t> fkeys;
                      std::vector<ValT> fvals;
                      std::vector<uint64_t> fscores;
                      for (size_t i = 0; i < keys.size(); ++i) {
                        if (scores[i] >= options.min_score) {
                          fkeys.push_back(keys[i]);
                          fvals.insert(fvals.end(), vals.data() + i * dim,
                                       vals.data() + (i + 1) * dim);
                          if (!scores.empty())
                            fscores.push_back(scores[i]);
                        }
                      }
                      if (!fkeys.empty()) {
                        writer->write_batch(fkeys, fvals.data(),
                                            fvals.size() * sizeof(ValT),
                                            fscores, nullptr, 0);
                      }
                    } else {
                      std::vector<uint64_t> ukeys(keys.begin(), keys.end());
                      writer->write_batch(ukeys, vals.data(),
                                          vals.size() * sizeof(ValT), scores,
                                          nullptr, 0);
                    }
                    return true;
                  }),
              options.batch_size);

          writer->commit();
        }
      });
    }
  }

  void load_checkpoint(const std::string &uri, uint64_t version, bool clear,
                       bool load_optimizer_state) {
    se::CheckpointLoadOptions options;
    options.version = version;
    options.clear_before_load = clear;
    options.load_optimizer_state = load_optimizer_state;

    auto reader = checkpoint_reader_factory_;
    if (!reader.has_value()) {
      throw std::runtime_error("checkpoint backend not configured");
    }
    {
      nb::gil_scoped_release release;
      if (clear)
        dispatch_val([&]<typename ValT>() { get_store<ValT>()->clear(); });

      dispatch_val([&]<typename ValT>() {
        bool has_score = get_store<ValT>()->has_score_policy();

        for (auto &[slot_id, dim] : slot_dims_) {
          reader->open(uri, static_cast<uint32_t>(slot_id), version,
                       static_cast<int64_t>(options.batch_size));

          std::vector<uint64_t> keys;
          std::vector<std::byte> embs;
          std::vector<uint64_t> scores;
          std::vector<std::byte> opt_states;

          while (reader->read_batch(options.batch_size, keys, embs, scores,
                                    opt_states)) {
            size_t n = keys.size();
            auto vals = std::span<const ValT>(
                reinterpret_cast<const ValT *>(embs.data()), n * dim);
            auto row_lengths = std::make_unique<int64_t[]>(n);
            std::fill_n(row_lengths.get(), n, static_cast<int64_t>(dim));
            get_store<ValT>()->assign(
                std::span<const uint64_t>(keys), vals,
                std::span<const int64_t>(row_lengths.get(), n));
            if (has_score && !scores.empty()) {
              get_store<ValT>()->assign_scores(
                  std::span<const uint64_t>(keys),
                  std::span<const uint64_t>(scores));
            }
          }
        }
      });
    }
  }

  void set_checkpoint_factories(se::CheckpointWriterFactory wf,
                                se::CheckpointReaderFactory rf) {
    checkpoint_writer_factory_ = std::move(wf);
    checkpoint_reader_factory_ = std::move(rf);
  }

private:
  se::CheckpointWriterFactory checkpoint_writer_factory_;
  se::CheckpointReaderFactory checkpoint_reader_factory_;
};

} // namespace

// =============================================================================
// NB_MODULE
// =============================================================================

NB_MODULE(_core, m) {
  m.doc() = "sparse_embedding C++ backend";

  nb::class_<se::EmbeddingConfig>(m, "EmbeddingConfig")
      .def(nb::init<std::unordered_map<uint32_t, size_t>, size_t, std::string,
                    size_t>(),
           nb::arg("slot_dims"), nb::arg("num_threads") = 0,
           nb::arg("dtype") = "float32", nb::arg("max_active_keys") = 1 << 20)
      .def_rw("slot_dims", &se::EmbeddingConfig::slot_dims)
      .def_rw("num_threads", &se::EmbeddingConfig::num_threads)
      .def_rw("dtype", &se::EmbeddingConfig::dtype)
      .def_rw("max_active_keys", &se::EmbeddingConfig::max_active_keys);

  nb::class_<PyInitializer>(m, "Initializer");

  struct ZerosInitializerPy : PyInitializer {};
  nb::class_<ZerosInitializerPy, PyInitializer>(m, "Zeros")
      .def("__init__", [](ZerosInitializerPy *self) {
        new (self) ZerosInitializerPy();
        self->builder =
            std::make_shared<pro::proxy<se::InitializerBuilderFacade>>(
                pro::make_proxy<se::InitializerBuilderFacade,
                                ZerosInitializerBuilder>());
      });

  struct UniformInitializerPy : PyInitializer {};
  nb::class_<UniformInitializerPy, PyInitializer>(m, "Uniform")
      .def(
          "__init__",
          [](UniformInitializerPy *self, float min_val, float max_val) {
            new (self) UniformInitializerPy();
            self->builder =
                std::make_shared<pro::proxy<se::InitializerBuilderFacade>>(
                    pro::make_proxy<se::InitializerBuilderFacade,
                                    UniformInitializerBuilder>(min_val,
                                                               max_val));
          },
          nb::arg("min_val") = -0.05f, nb::arg("max_val") = 0.05f);

  nb::class_<PyOptimizer>(m, "Optimizer");

  struct AdamOptimizerPy : PyOptimizer {};
  nb::class_<AdamOptimizerPy, PyOptimizer>(m, "Adam").def(
      "__init__",
      [](AdamOptimizerPy *self, float lr, float beta1, float beta2, float eps,
         float weight_decay) {
        new (self) AdamOptimizerPy();
        self->builder =
            std::make_shared<pro::proxy<se::OptimizerBuilderFacade>>(
                pro::make_proxy<
                    se::OptimizerBuilderFacade,
                    DefaultOptimizerBuilder<se::AdamOptimizer, se::AdamConfig>>(
                    se::AdamConfig{lr, beta1, beta2, eps, weight_decay}));
      },
      nb::arg("lr") = 1e-3f, nb::arg("beta1") = 0.9f, nb::arg("beta2") = 0.999f,
      nb::arg("eps") = 1e-8f, nb::arg("weight_decay") = 0.0f);

  struct SgdOptimizerPy : PyOptimizer {};
  nb::class_<SgdOptimizerPy, PyOptimizer>(m, "Sgd").def(
      "__init__",
      [](SgdOptimizerPy *self, float lr, float weight_decay) {
        new (self) SgdOptimizerPy();
        self->builder =
            std::make_shared<pro::proxy<se::OptimizerBuilderFacade>>(
                pro::make_proxy<
                    se::OptimizerBuilderFacade,
                    DefaultOptimizerBuilder<se::SgdOptimizer, se::SgdConfig>>(
                    se::SgdConfig{lr, weight_decay}));
      },
      nb::arg("lr") = 1e-2f, nb::arg("weight_decay") = 0.0f);

  nb::class_<se::py::DLPackCapsuleWrapper>(m, "DLPackCapsuleWrapper")
      .def("__dlpack__", &se::py::DLPackCapsuleWrapper::dlpack, nb::kw_only(),
           nb::arg("stream") = nb::none(), nb::arg("max_version") = nb::none(),
           nb::arg("dl_device") = nb::none(), nb::arg("copy") = nb::none())
      .def("__dlpack_device__", &se::py::DLPackCapsuleWrapper::dlpack_device);

  nb::class_<EmbeddingStoreWrapper>(m, "EmbeddingStore")
      .def(
          nb::init<const se::EmbeddingConfig &, PyInitializer, PyOptimizer,
                   std::string>(),
          nb::arg("embedding_config"),
          nb::arg("initializer_config") =
              PyInitializer{
                  std::make_shared<pro::proxy<se::InitializerBuilderFacade>>(
                      pro::make_proxy<se::InitializerBuilderFacade,
                                      UniformInitializerBuilder>(-0.05f,
                                                                 0.05f))},
          nb::arg("optimizer_config") =
              PyOptimizer{
                  std::make_shared<pro::proxy<se::OptimizerBuilderFacade>>(
                      pro::make_proxy<se::OptimizerBuilderFacade,
                                      DefaultOptimizerBuilder<se::AdamOptimizer,
                                                              se::AdamConfig>>(
                          se::AdamConfig{}))},
          nb::arg("score_policy") = "none")

      .def("total_size", &EmbeddingStoreWrapper::total_size)
      .def("max_dim", &EmbeddingStoreWrapper::max_dim)
      .def("num_slots", &EmbeddingStoreWrapper::num_slots)
      .def("dim_for_slot", &EmbeddingStoreWrapper::dim_for_slot,
           nb::arg("slot_id"))
      .def("get_dims", &EmbeddingStoreWrapper::get_dims, nb::arg("keys"))

      .def("lookup", &EmbeddingStoreWrapper::lookup, nb::arg("keys"))
      .def("lookup_or_insert", &EmbeddingStoreWrapper::lookup_or_insert,
           nb::arg("keys"))
      .def("assign", &EmbeddingStoreWrapper::assign, nb::arg("keys"),
           nb::arg("values"))
      .def("contains", &EmbeddingStoreWrapper::contains, nb::arg("keys"))
      .def("erase", &EmbeddingStoreWrapper::erase, nb::arg("keys"))
      .def("clear", &EmbeddingStoreWrapper::clear)
      .def("assign_scores", &EmbeddingStoreWrapper::assign_scores,
           nb::arg("keys"), nb::arg("scores"))
      .def("accumulate_gradients", &EmbeddingStoreWrapper::accumulate_gradients,
           nb::arg("keys"), nb::arg("grads"))
      .def("optimizer_step", &EmbeddingStoreWrapper::optimizer_step)

      .def("async_lookup", &EmbeddingStoreWrapper::async_lookup,
           nb::arg("keys"))
      .def("async_assign", &EmbeddingStoreWrapper::async_assign,
           nb::arg("keys"), nb::arg("values"))

      .def("save_checkpoint", &EmbeddingStoreWrapper::save_checkpoint,
           nb::arg("uri"), nb::arg("mode") = "full", nb::arg("min_score") = 0,
           nb::arg("include_optimizer_state") = false,
           nb::arg("use_compression") = false)
      .def("load_checkpoint", &EmbeddingStoreWrapper::load_checkpoint,
           nb::arg("uri"), nb::arg("version") = 0, nb::arg("clear") = true,
           nb::arg("load_optimizer_state") = false)
      .def("set_checkpoint_factories",
           [](EmbeddingStoreWrapper &self, const std::string &backend) {
             if (backend == "lance") {
               auto wf = pro::make_proxy<se::CheckpointWriterFacade,
                                         se::LanceCheckpointWriter>();
               auto rf = pro::make_proxy<se::CheckpointReaderFacade,
                                         se::LanceCheckpointReader>();
               self.set_checkpoint_factories(std::move(wf), std::move(rf));
             } else {
               throw std::runtime_error("Unsupported backend: " + backend);
             }
           });

  m.def(
      "async_unique",
      [](nb::object keys, bool return_inverse, bool return_first_indices,
         bool return_counts) -> nb::object {
        auto [dlm_keys, cap_keys] = se::py::extract_dlpack_tensor(keys);
        size_t n = get_dlm_size(dlm_keys);
        auto kdt = detect_key_dtype(dlm_keys);
        const void *keys_data =
            reinterpret_cast<const char *>(dlm_keys->dl_tensor.data) +
            dlm_keys->dl_tensor.byte_offset;

        static exec::static_thread_pool global_pool(
            std::max(1u, std::thread::hardware_concurrency()));

        struct ResultWrapper {
          DLManagedTensor *u_dlm = nullptr;
          DLManagedTensor *inv_dlm = nullptr;
          DLManagedTensor *first_dlm = nullptr;
          DLManagedTensor *counts_dlm = nullptr;
          bool return_inverse;
          bool return_first_indices;
          bool return_counts;
        };

        return se::py::async_dispatch(
            global_pool,
            [n, kdt, keys_data, return_inverse, return_first_indices,
             return_counts,
             keys_keeper = std::make_shared<se::py::AsyncPyObjectKeeper>(
                 cap_keys)]() -> ResultWrapper {
              ResultWrapper wrap;
              wrap.return_inverse = return_inverse;
              wrap.return_first_indices = return_first_indices;
              wrap.return_counts = return_counts;

              auto process_res = [&](auto &res) {
                wrap.u_dlm = se::py::wrap_shared_ptr_as_tensor(
                    res.unique_keys, res.unique_count);
                if (return_inverse)
                  wrap.inv_dlm = se::py::wrap_shared_ptr_as_tensor(
                      res.inverse_indices, res.inverse_count);
                if (return_first_indices)
                  wrap.first_dlm = se::py::wrap_shared_ptr_as_tensor(
                      res.first_indices, res.unique_count);
                if (return_counts)
                  wrap.counts_dlm = se::py::wrap_shared_ptr_as_tensor(
                      res.counts, res.unique_count);
              };

              auto dispatch = [&](auto dummy) {
                using T = std::decay_t<decltype(dummy)>;
                auto span =
                    std::span<const T>(static_cast<const T *>(keys_data), n);
                auto res = se::ops::unique_keys_inverse(
                    span, return_first_indices, return_counts);
                process_res(res);
              };

              if (kdt == KeyDType::kInt32)
                dispatch(int32_t{});
              else if (kdt == KeyDType::kUInt32)
                dispatch(uint32_t{});
              else if (kdt == KeyDType::kInt64)
                dispatch(int64_t{});
              else
                dispatch(uint64_t{});

              return wrap;
            },
            [](nb::object future, ResultWrapper wrap) {
              std::vector<nb::object> ret;
              ret.push_back(se::py::wrap_output_tensor(wrap.u_dlm));
              if (wrap.return_inverse)
                ret.push_back(se::py::wrap_output_tensor(wrap.inv_dlm));
              if (wrap.return_first_indices)
                ret.push_back(se::py::wrap_output_tensor(wrap.first_dlm));
              if (wrap.return_counts)
                ret.push_back(se::py::wrap_output_tensor(wrap.counts_dlm));

              nb::tuple res_tuple =
                  nb::steal<nb::tuple>(PySequence_Tuple(nb::cast(ret).ptr()));
              future.attr("set_result")(res_tuple);
            });
      },
      nb::arg("keys"), nb::arg("return_inverse") = true,
      nb::arg("return_first_indices") = false,
      nb::arg("return_counts") = false);

  m.def("wake_fd", []() { return se::py::GetPythonWakeManager().fd(); });
  m.def("drain_wake_queue", []() { se::py::GetPythonWakeManager().Drain(); });
}
