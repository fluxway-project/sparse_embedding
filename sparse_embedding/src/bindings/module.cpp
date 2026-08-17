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

#include <dlpack/dlpack.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>

#include "sparse_embedding/src/bindings/async_bridge.hpp"
#include "sparse_embedding/src/bindings/dlpack_utils.hpp"
#include "sparse_embedding/src/bindings/module_wrapper.hpp"
#include "sparse_embedding/src/bindings/module_wrapper_methods.inl"
#include "sparse_embedding/src/ops/partitioner.hpp"
#include "sparse_embedding/src/ops/tensor_ops.hpp"
#ifdef SE_ENABLE_LANCE_CHECKPOINTER
#include "sparse_embedding/src/bindings/module_checkpoint_methods.inl"
#include "sparse_embedding/src/checkpoint/checkpoint_registry.hpp"
#include "sparse_embedding/src/checkpoint/lance/layout_schema.hpp"
#include "sparse_embedding/src/checkpoint/lance/native_fragment_writer.hpp"
#endif

namespace nb = nanobind;

using namespace se::py::detail;

// =============================================================================
// EmbeddingStoreWrapper constructor.
// =============================================================================

namespace se::py::detail {

void construct_store(EmbeddingStoreWrapper &self, const PyInitializer &init_cfg,
                     const PyOptimizer &opt_cfg, size_t thread_budget,
                     se::StoreConstructionKind kind) {
  se::StoreConstructionTarget target{&self.store_, self.dims_, self.storage_};
  se::StoreConstructionRequest request{init_cfg.builder, opt_cfg.config};
  se::create_hogwild_store(kind, std::move(target), std::move(request),
                           thread_budget);
}

EmbeddingStoreWrapper::EmbeddingStoreWrapper(
    const se::EmbeddingConfig &emb_cfg,
    std::optional<PyInitializer> init_cfg_opt,
    std::optional<PyOptimizer> opt_cfg_opt, std::string score_policy_name)
    : async_pool_((emb_cfg.thread_budget == 0)
                      ? std::max(1u, std::thread::hardware_concurrency())
                      : emb_cfg.thread_budget),
      value_dtype_code_(parse_value_dtype(emb_cfg.value_dtype, "dtype")),
      optimizer_dtype_code_(parse_value_dtype(emb_cfg.optimizer_dtype.empty()
                                                  ? emb_cfg.value_dtype
                                                  : emb_cfg.optimizer_dtype,
                                              "opt_dtype")),
      key_type_enum_(parse_key_dtype(emb_cfg.key_dtype)),
      score_policy_name_(score_policy_name),
      storage_(se::create_page_storage(emb_cfg.storage)) {
  PyInitializer init_cfg = init_cfg_opt.value_or(
      PyInitializer{std::make_shared<pro::proxy<se::InitializerBuilderFacade>>(
          pro::make_proxy<se::InitializerBuilderFacade,
                          UniformInitializerBuilder>(-0.05f, 0.05f))});

  PyOptimizer opt_cfg = opt_cfg_opt.value_or(PyOptimizer{se::AdamConfig{}});

  dims_ = emb_cfg.dims;
  if (dims_.empty())
    throw std::invalid_argument("dims must not be empty");

  size_t thread_budget = (emb_cfg.thread_budget == 0)
                             ? std::max(1u, std::thread::hardware_concurrency())
                             : emb_cfg.thread_budget;

  const auto score_policy = se::parse_store_score_policy(score_policy_name);
  const bool requires_score = std::visit(
      [](const auto &config) {
        return config.dynamic_lr_policy.has_value() &&
               std::visit(
                   [](const auto &policy) { return policy.requires_score(); },
                   *config.dynamic_lr_policy);
      },
      opt_cfg.config);
  if (requires_score && score_policy == se::StoreScorePolicy::kNone)
    throw std::invalid_argument(
        "dynamic learning-rate policy requires a score policy");

  const auto to_store_dtype = [](ValueDType dtype) {
    switch (dtype) {
    case ValueDType::kFloat32:
      return se::StoreParamDType::kFloat32;
    case ValueDType::kFloat16:
      return se::StoreParamDType::kFloat16;
    case ValueDType::kBFloat16:
      return se::StoreParamDType::kBFloat16;
    }
    throw std::invalid_argument("unsupported value dtype");
  };
  construct_store(*this, init_cfg, opt_cfg, thread_budget,
                  se::StoreConstructionKind{
                      to_store_dtype(value_dtype_code_),
                      to_store_dtype(optimizer_dtype_code_), score_policy});
}

} // namespace se::py::detail

// =============================================================================
// PyPartitioner method implementations
// =============================================================================

namespace se::py::detail {

nb::object PyPartitioner::partition(nb::object keys) {
  auto [dlm_keys, cap_keys] = se::py::extract_dlpack_tensor(keys);
  size_t n = dlm_keys->dl_tensor.shape[0];

  DLManagedTensor *out_dlm = se::py::create_output_tensor<uint32_t>(
      std::span<const int64_t>({static_cast<int64_t>(n)}));
  auto out_span = se::py::dlpack_to_span<uint32_t>(out_dlm);

  dispatch_partition_key(dlm_keys, [&](auto keys_span) {
    (*facade)->partition_span(keys_span, out_span);
  });
  return se::py::wrap_output_tensor(out_dlm);
}

nb::object PyPartitioner::async_partition(nb::object keys) {
  auto [dlm_keys, cap_keys] = se::py::extract_dlpack_tensor(keys);
  size_t n = dlm_keys->dl_tensor.shape[0];
  auto kdt = detect_key_dtype(dlm_keys);
  const void *keys_data =
      reinterpret_cast<const char *>(dlm_keys->dl_tensor.data) +
      dlm_keys->dl_tensor.byte_offset;

  static exec::static_thread_pool pool(
      std::max(1u, std::thread::hardware_concurrency()));
  auto facade_copy = facade;

  return se::py::async_dispatch(
      pool,
      [facade_copy, n, kdt, keys_data,
       keys_keeper = std::make_shared<se::py::AsyncPyObjectKeeper>(
           cap_keys)]() -> DLManagedTensor * {
        DLManagedTensor *out_dlm = se::py::create_output_tensor<uint32_t>(
            std::span<const int64_t>({static_cast<int64_t>(n)}));
        uint32_t *out_ptr =
            reinterpret_cast<uint32_t *>(out_dlm->dl_tensor.data);

        auto do_partition = [&](auto ks) {
          (*facade_copy)->partition_span(ks, std::span<uint32_t>(out_ptr, n));
        };
        if (kdt == KeyDType::kInt32)
          do_partition(std::span<const int32_t>(
              static_cast<const int32_t *>(keys_data), n));
        else if (kdt == KeyDType::kUInt32)
          do_partition(std::span<const uint32_t>(
              static_cast<const uint32_t *>(keys_data), n));
        else if (kdt == KeyDType::kInt64)
          do_partition(std::span<const int64_t>(
              static_cast<const int64_t *>(keys_data), n));
        else
          do_partition(std::span<const uint64_t>(
              static_cast<const uint64_t *>(keys_data), n));

        return out_dlm;
      },
      [](nb::object future, DLManagedTensor *mask_dlm) {
        future.attr("set_result")(se::py::wrap_output_tensor(mask_dlm));
      });
}

nb::object PyPartitioner::filter(nb::object keys, uint32_t target_shard) {
  auto [dlm_keys, cap_keys] = se::py::extract_dlpack_tensor(keys);
  size_t n = dlm_keys->dl_tensor.shape[0];

  DLManagedTensor *out_dlm = se::py::create_output_tensor<bool>(
      std::span<const int64_t>({static_cast<int64_t>(n)}));
  auto out_span = se::py::dlpack_to_span<bool>(out_dlm);

  dispatch_partition_key(dlm_keys, [&](auto keys_span) {
    (*facade)->filter_span(keys_span, target_shard, out_span);
  });
  return se::py::wrap_output_tensor(out_dlm);
}

nb::object PyPartitioner::async_filter(nb::object keys, uint32_t target_shard) {
  auto [dlm_keys, cap_keys] = se::py::extract_dlpack_tensor(keys);
  size_t n = dlm_keys->dl_tensor.shape[0];
  auto kdt = detect_key_dtype(dlm_keys);
  const void *keys_data =
      reinterpret_cast<const char *>(dlm_keys->dl_tensor.data) +
      dlm_keys->dl_tensor.byte_offset;

  static exec::static_thread_pool pool(
      std::max(1u, std::thread::hardware_concurrency()));
  auto facade_copy = facade;

  return se::py::async_dispatch(
      pool,
      [facade_copy, n, target_shard, kdt, keys_data,
       keys_keeper = std::make_shared<se::py::AsyncPyObjectKeeper>(
           cap_keys)]() -> DLManagedTensor * {
        DLManagedTensor *out_dlm = se::py::create_output_tensor<bool>(
            std::span<const int64_t>({static_cast<int64_t>(n)}));
        bool *out_ptr = reinterpret_cast<bool *>(out_dlm->dl_tensor.data);

        auto do_filter = [&](auto ks) {
          (*facade_copy)
              ->filter_span(ks, target_shard, std::span<bool>(out_ptr, n));
        };
        if (kdt == KeyDType::kInt32)
          do_filter(std::span<const int32_t>(
              static_cast<const int32_t *>(keys_data), n));
        else if (kdt == KeyDType::kUInt32)
          do_filter(std::span<const uint32_t>(
              static_cast<const uint32_t *>(keys_data), n));
        else if (kdt == KeyDType::kInt64)
          do_filter(std::span<const int64_t>(
              static_cast<const int64_t *>(keys_data), n));
        else
          do_filter(std::span<const uint64_t>(
              static_cast<const uint64_t *>(keys_data), n));

        return out_dlm;
      },
      [](nb::object future, DLManagedTensor *mask_dlm) {
        future.attr("set_result")(se::py::wrap_output_tensor(mask_dlm));
      });
}

} // namespace se::py::detail

// =============================================================================
// NB_MODULE
// =============================================================================

NB_MODULE(_core, m) {
  m.doc() = "sparse_embedding C++ backend";

  nb::class_<se::StorageOptions>(m, "_StorageOptions")
      .def_static(
          "auto",
          [](std::optional<size_t> memory_limit_bytes,
             std::optional<std::string> spill_directory,
             std::optional<size_t> max_spill_bytes,
             std::optional<double> high_watermark,
             std::optional<double> low_watermark,
             std::optional<double> backpressure_fraction,
             std::optional<double> circuit_fraction,
             std::optional<int64_t> epoch_stall_timeout_ms) {
            se::StorageOptions options;
            options.mode = se::StorageMode::kAuto;
            options.memory_limit_bytes = memory_limit_bytes;
            if (spill_directory)
              options.spill_directory = *spill_directory;
            options.max_spill_bytes = max_spill_bytes;
            if (high_watermark || low_watermark) {
              se::WatermarkConfig wm;
              if (high_watermark)
                wm.high = *high_watermark;
              if (low_watermark)
                wm.low = *low_watermark;
              options.watermark = wm;
            }
            if (backpressure_fraction)
              options.retirement_pressure.backpressure_fraction =
                  *backpressure_fraction;
            if (circuit_fraction)
              options.retirement_pressure.circuit_fraction = *circuit_fraction;
            if (epoch_stall_timeout_ms) {
              if (*epoch_stall_timeout_ms <= 0) {
                throw std::invalid_argument(
                    "epoch_stall_timeout_ms must be positive");
              }
              options.retirement_pressure.epoch_stall_timeout =
                  std::chrono::milliseconds(*epoch_stall_timeout_ms);
            }
            return options;
          },
          nb::arg("memory_limit_bytes") = nb::none(),
          nb::arg("spill_directory") = nb::none(),
          nb::arg("max_spill_bytes") = nb::none(),
          nb::arg("high_watermark") = nb::none(),
          nb::arg("low_watermark") = nb::none(),
          nb::arg("backpressure_fraction") = nb::none(),
          nb::arg("circuit_fraction") = nb::none(),
          nb::arg("epoch_stall_timeout_ms") = nb::none())
      .def_static("memory_only", []() {
        se::StorageOptions options;
        options.mode = se::StorageMode::kMemoryOnly;
        return options;
      });

  nb::class_<se::EmbeddingConfig>(m, "_EmbeddingConfig")
      .def(nb::init<std::vector<size_t>, size_t, std::string, std::string,
                    std::string, se::StorageOptions>(),
           nb::arg("dims"), nb::arg("thread_budget") = 0,
           nb::arg("value_dtype") = "float32", nb::arg("optimizer_dtype") = "",
           nb::arg("key_dtype") = "int64",
           nb::arg("storage") = se::StorageOptions{})
      .def_rw("dims", &se::EmbeddingConfig::dims)
      .def_rw("thread_budget", &se::EmbeddingConfig::thread_budget)
      .def_rw("value_dtype", &se::EmbeddingConfig::value_dtype)
      .def_rw("optimizer_dtype", &se::EmbeddingConfig::optimizer_dtype)
      .def_rw("key_dtype", &se::EmbeddingConfig::key_dtype);

  nb::class_<PyInitializer>(m, "Initializer");

  struct PyLrPolicy {
    std::optional<se::DynamicLrPolicyVariant> policy;
  };
  nb::class_<PyLrPolicy>(m, "LrPolicy");

  struct ConstantLrPolicyPy : PyLrPolicy {};
  nb::class_<ConstantLrPolicyPy, PyLrPolicy>(m, "ConstantLrPolicy")
      .def("__init__", [](ConstantLrPolicyPy *self) {
        new (self) ConstantLrPolicyPy();
        self->policy.emplace(se::ConstantLrPolicy{});
      });

  struct SqrtDecayLrPolicyPy : PyLrPolicy {};
  nb::class_<SqrtDecayLrPolicyPy, PyLrPolicy>(m, "SqrtDecayLrPolicy")
      .def("__init__", [](SqrtDecayLrPolicyPy *self) {
        new (self) SqrtDecayLrPolicyPy();
        self->policy.emplace(se::SqrtDecayLrPolicy{});
      });

  struct LogDecayLrPolicyPy : PyLrPolicy {};
  nb::class_<LogDecayLrPolicyPy, PyLrPolicy>(m, "LogDecayLrPolicy")
      .def("__init__", [](LogDecayLrPolicyPy *self) {
        new (self) LogDecayLrPolicyPy();
        self->policy.emplace(se::LogDecayLrPolicy{});
      });

  struct InverseDecayLrPolicyPy : PyLrPolicy {};
  nb::class_<InverseDecayLrPolicyPy, PyLrPolicy>(m, "InverseDecayLrPolicy")
      .def("__init__", [](InverseDecayLrPolicyPy *self) {
        new (self) InverseDecayLrPolicyPy();
        self->policy.emplace(se::InverseDecayLrPolicy{});
      });

  struct CosineFrequencyLrPolicyPy : PyLrPolicy {};
  nb::class_<CosineFrequencyLrPolicyPy, PyLrPolicy>(m,
                                                    "CosineFrequencyLrPolicy")
      .def(
          "__init__",
          [](CosineFrequencyLrPolicyPy *self, float min_lr_ratio,
             uint64_t max_score) {
            new (self) CosineFrequencyLrPolicyPy();
            self->policy.emplace(
                se::CosineFrequencyLrPolicy(min_lr_ratio, max_score));
          },
          nb::arg("min_lr_ratio") = 0.1f, nb::arg("max_score") = 1000);

  struct CustomLutLrPolicyPy : PyLrPolicy {};
  nb::class_<CustomLutLrPolicyPy, PyLrPolicy>(m, "CustomLutLrPolicy")
      .def(
          "__init__",
          [](CustomLutLrPolicyPy *self, std::vector<float> lut) {
            new (self) CustomLutLrPolicyPy();
            self->policy.emplace(se::CustomLutLrPolicy(std::move(lut)));
          },
          nb::arg("lut"));

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
         float weight_decay, std::optional<PyLrPolicy> policy) {
        new (self) AdamOptimizerPy();
        std::optional<se::DynamicLrPolicyVariant> lr_policy;
        if (policy) {
          lr_policy = *policy->policy;
        }
        self->config = se::AdamConfig{lr,  beta1,        beta2,
                                      eps, weight_decay, std::move(lr_policy)};
      },
      nb::arg("lr") = 1e-3f, nb::arg("beta1") = 0.9f, nb::arg("beta2") = 0.999f,
      nb::arg("eps") = 1e-8f, nb::arg("weight_decay") = 0.0f,
      nb::arg("dynamic_lr_policy") = nb::none());

  struct SgdOptimizerPy : PyOptimizer {};
  nb::class_<SgdOptimizerPy, PyOptimizer>(m, "Sgd").def(
      "__init__",
      [](SgdOptimizerPy *self, float lr, float weight_decay,
         std::optional<PyLrPolicy> policy) {
        new (self) SgdOptimizerPy();
        std::optional<se::DynamicLrPolicyVariant> lr_policy;
        if (policy) {
          lr_policy = *policy->policy;
        }
        self->config = se::SgdConfig{lr, weight_decay, std::move(lr_policy)};
      },
      nb::arg("lr") = 1e-2f, nb::arg("weight_decay") = 0.0f,
      nb::arg("dynamic_lr_policy") = nb::none());

  nb::class_<se::py::DLPackCapsuleWrapper>(m, "DLPackCapsuleWrapper")
      .def("__dlpack__", &se::py::DLPackCapsuleWrapper::dlpack, nb::kw_only(),
           nb::arg("stream") = nb::none(), nb::arg("max_version") = nb::none(),
           nb::arg("dl_device") = nb::none(), nb::arg("copy") = nb::none())
      .def("__dlpack_device__", &se::py::DLPackCapsuleWrapper::dlpack_device);

  nb::class_<EmbeddingStoreWrapper>(m, "EmbeddingStore")
      .def(nb::init<const se::EmbeddingConfig &, std::optional<PyInitializer>,
                    std::optional<PyOptimizer>, std::string>(),
           nb::arg("embedding_config"),
           nb::arg("initializer_config") = nb::none(),
           nb::arg("optimizer_config") = nb::none(),
           nb::arg("score_policy") = "none")

      .def("total_size", &EmbeddingStoreWrapper::total_size)
      .def("max_dim", &EmbeddingStoreWrapper::max_dim)
      .def("num_slots", &EmbeddingStoreWrapper::num_slots)
      .def("table_dims", &EmbeddingStoreWrapper::table_dims)
      .def("internal_shard_count", &EmbeddingStoreWrapper::internal_shard_count)
      .def("storage_stats", &EmbeddingStoreWrapper::storage_stats)
      .def("reserve", &EmbeddingStoreWrapper::reserve, nb::arg("keys_count"),
           nb::arg("dim"))
      .def("reserve_per_shard", &EmbeddingStoreWrapper::reserve_per_shard,
           nb::arg("counts"), nb::arg("dim"))

      .def("lookup", &EmbeddingStoreWrapper::lookup, nb::arg("keys"),
           nb::arg("dim"))
      .def("lookup_or_insert", &EmbeddingStoreWrapper::lookup_or_insert,
           nb::arg("keys"), nb::arg("dim"))
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
           nb::arg("keys"), nb::arg("dim"),
           nb::sig("def async_lookup(self, keys: object, dim: int) -> "
                   "typing.Awaitable[typing.Any]"))
      .def("async_assign", &EmbeddingStoreWrapper::async_assign,
           nb::arg("keys"), nb::arg("values"),
           nb::sig("def async_assign(self, keys: object, values: object) -> "
                   "typing.Awaitable[typing.Any]"))
      .def("async_lookup_or_insert",
           &EmbeddingStoreWrapper::async_lookup_or_insert, nb::arg("keys"),
           nb::arg("dim"),
           nb::sig("def async_lookup_or_insert(self, keys: object, dim: int) "
                   "-> typing.Awaitable[typing.Any]"))
      .def("async_accumulate_gradients",
           &EmbeddingStoreWrapper::async_accumulate_gradients, nb::arg("keys"),
           nb::arg("grads"),
           nb::sig("def async_accumulate_gradients(self, keys: object, grads: "
                   "object) -> typing.Awaitable[typing.Any]"))
      .def(
          "async_optimizer_step", &EmbeddingStoreWrapper::async_optimizer_step,
          nb::sig(
              "def async_optimizer_step(self) -> typing.Awaitable[typing.Any]"))
      .def("async_contains", &EmbeddingStoreWrapper::async_contains,
           nb::arg("keys"),
           nb::sig("def async_contains(self, keys: object) -> "
                   "typing.Awaitable[typing.Any]"))
      .def("async_erase", &EmbeddingStoreWrapper::async_erase, nb::arg("keys"),
           nb::sig("def async_erase(self, keys: object) -> "
                   "typing.Awaitable[typing.Any]"))
      .def("async_clear", &EmbeddingStoreWrapper::async_clear,
           nb::sig("def async_clear(self) -> typing.Awaitable[typing.Any]"))
      .def("async_assign_scores", &EmbeddingStoreWrapper::async_assign_scores,
           nb::arg("keys"), nb::arg("scores"),
           nb::sig("def async_assign_scores(self, keys: object, scores: "
                   "object) -> typing.Awaitable[typing.Any]"))

      .def("compact_page", &EmbeddingStoreWrapper::compact_page, nb::arg("dim"),
           nb::arg("page_id"))
      .def("promote_hot_keys", &EmbeddingStoreWrapper::promote_hot_keys,
           nb::arg("dim"), nb::arg("page_id"), nb::arg("maximum_keys"),
           nb::arg("minimum_score") = 0)
      .def("promoted_hot_page_size",
           &EmbeddingStoreWrapper::promoted_hot_page_size, nb::arg("dim"))
      .def("export_page_images", &EmbeddingStoreWrapper::export_page_images)
      .def("import_page_images", &EmbeddingStoreWrapper::import_page_images,
           nb::arg("pages"))
      .def("value_dtype", &EmbeddingStoreWrapper::value_dtype)
      .def("optimizer_dtype", &EmbeddingStoreWrapper::optimizer_dtype)
      .def("score_policy", &EmbeddingStoreWrapper::score_policy)

      ;

#ifdef SE_ENABLE_LANCE_CHECKPOINTER
  auto checkpoint = m.def_submodule("checkpoint");
  checkpoint.def(
      "save_checkpoint",
      [](EmbeddingStoreWrapper &self, const std::string &uri,
         const std::string &mode, uint64_t min_score,
         bool include_optimizer_state, bool use_compression) {
        self.save_checkpoint(uri, mode, min_score, include_optimizer_state,
                             use_compression);
      },
      nb::arg("store"), nb::arg("uri"), nb::arg("mode") = "full",
      nb::arg("min_score") = 0, nb::arg("include_optimizer_state") = false,
      nb::arg("use_compression") = false);
  checkpoint.def(
      "load_checkpoint",
      [](EmbeddingStoreWrapper &self, const std::string &uri, uint64_t version,
         bool clear, bool load_optimizer_state, nb::object filter_fn,
         uint32_t target_shard_id, size_t batch_size) {
        self.load_checkpoint(uri, version, clear, load_optimizer_state,
                             filter_fn, target_shard_id, batch_size);
      },
      nb::arg("store"), nb::arg("uri"), nb::arg("version") = 0,
      nb::arg("clear") = true, nb::arg("load_optimizer_state") = false,
      nb::arg("filter_fn") = nb::none(), nb::arg("target_shard_id") = 0,
      nb::arg("batch_size") = 65536);
  checkpoint.def(
      "write_native_fragment",
      [](EmbeddingStoreWrapper &self, const std::string &uri,
         uint32_t participant_index, const std::vector<uint32_t> &dimensions,
         const std::string &embedding_dtype, const std::string &optimizer_dtype,
         const std::string &score_policy, uint32_t protocol_version,
         const std::string &schema_digest) {
        return self.write_native_fragment(
            uri, participant_index, dimensions, embedding_dtype,
            optimizer_dtype, score_policy, protocol_version, schema_digest);
      },
      nb::arg("store"), nb::arg("uri"), nb::arg("participant_index"),
      nb::arg("dimensions"), nb::arg("embedding_dtype"),
      nb::arg("optimizer_dtype"), nb::arg("score_policy"),
      nb::arg("protocol_version"), nb::arg("schema_digest"));
  checkpoint.def(
      "set_checkpoint_factories",
      [](EmbeddingStoreWrapper &self, const std::string &backend) {
        auto factories = se::make_checkpoint_factories(backend);
        self.set_checkpoint_factories(std::move(factories.writer),
                                      std::move(factories.reader));
      },
      nb::arg("store"), nb::arg("backend"));
  checkpoint.def(
      "export_batches_dlpack",
      [](EmbeddingStoreWrapper &self, nb::callable visitor, size_t batch_size,
         bool include_optimizer_state) {
        self.export_batches_dlpack(visitor, batch_size,
                                   include_optimizer_state);
      },
      nb::arg("store"), nb::arg("visitor"), nb::arg("batch_size") = 65536,
      nb::arg("include_optimizer_state") = false);
  checkpoint.def(
      "export_batches_arrow",
      [](EmbeddingStoreWrapper &self, nb::callable visitor, size_t batch_size,
         bool include_optimizer_state) {
        self.export_batches_arrow(visitor, batch_size, include_optimizer_state);
      },
      nb::arg("store"), nb::arg("visitor"), nb::arg("batch_size") = 65536,
      nb::arg("include_optimizer_state") = false);
  checkpoint.def(
      "assign_opt_state",
      [](EmbeddingStoreWrapper &self, nb::object keys, nb::object opt_states,
         size_t dimension, nb::object steps) {
        self.assign_opt_state(keys, opt_states, dimension, steps);
      },
      nb::arg("store"), nb::arg("keys"), nb::arg("opt_states"),
      nb::arg("dimension"), nb::arg("steps") = nb::none());
  checkpoint.def(
      "restore_native_entries",
      [](EmbeddingStoreWrapper &self, const std::string &uri, uint64_t version,
         nb::object selected_keys) {
        self.restore_native_entries(uri, version, selected_keys);
      },
      nb::arg("store"), nb::arg("uri"), nb::arg("version"),
      nb::arg("selected_keys"));
  checkpoint.def(
      "restore_native_page_images",
      [](EmbeddingStoreWrapper &self, const std::string &uri, uint64_t version,
         uint32_t participant_index) {
        self.restore_native_page_images(uri, version, participant_index);
      },
      nb::arg("store"), nb::arg("uri"), nb::arg("version"),
      nb::arg("participant_index"));
  checkpoint.def(
      "active_entry_keys",
      [](const std::string &uri, uint64_t version) {
        std::vector<uint64_t> keys;
        se::scan_native_active_entry_keys(
            uri, version, [&](std::span<const uint64_t> batch) {
              keys.insert(keys.end(), batch.begin(), batch.end());
              return true;
            });
        auto output = se::py::create_output_tensor<uint64_t>(
            std::array<int64_t, 1>{static_cast<int64_t>(keys.size())});
        std::copy(keys.begin(), keys.end(),
                  static_cast<uint64_t *>(output->dl_tensor.data));
        return se::py::wrap_output_tensor(output);
      },
      nb::arg("uri"), nb::arg("version"));
  checkpoint.def(
      "merge_native_lance_sources",
      [](const std::string &target_uri,
         const std::vector<std::string> &source_uris,
         const std::vector<std::string> &keys, uint64_t checkpoint_version) {
        std::vector<std::string_view> key_views;
        key_views.reserve(keys.size());
        for (const auto &key : keys)
          key_views.emplace_back(key);
        const auto artifact = se::merge_native_lance_sources(
            target_uri, source_uris, key_views, checkpoint_version);
        nb::dict result;
        result["locator"] = artifact.locator;
        result["target_version"] = artifact.target_version;
        result["inserted_rows"] = artifact.inserted_rows;
        result["updated_rows"] = artifact.updated_rows;
        result["deleted_rows"] = artifact.deleted_rows;
        return result;
      },
      nb::arg("target_uri"), nb::arg("source_uris"), nb::arg("keys"),
      nb::arg("checkpoint_version") = 0);

  checkpoint.def("lance_entries_schema_capsule", []() {
    auto *s = new ArrowSchema(se::lance_layout::make_entries_schema());
    return nb::capsule(s, "arrow_schema", [](void *p) noexcept {
      auto *schema = static_cast<ArrowSchema *>(p);
      if (schema) {
        if (schema->release)
          schema->release(schema);
        delete schema;
      }
    });
  });

  checkpoint.def("lance_page_images_schema_capsule", []() {
    auto *s = new ArrowSchema(se::lance_layout::make_page_images_schema());
    return nb::capsule(s, "arrow_schema", [](void *p) noexcept {
      auto *schema = static_cast<ArrowSchema *>(p);
      if (schema) {
        if (schema->release)
          schema->release(schema);
        delete schema;
      }
    });
  });

  checkpoint.def("lance_manifest_schema_capsule", []() {
    auto *s = new ArrowSchema(se::lance_layout::make_manifest_schema());
    return nb::capsule(s, "arrow_schema", [](void *p) noexcept {
      auto *schema = static_cast<ArrowSchema *>(p);
      if (schema) {
        if (schema->release)
          schema->release(schema);
        delete schema;
      }
    });
  });
#endif

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
      nb::arg("return_first_indices") = false, nb::arg("return_counts") = false,
      nb::sig("def async_unique(keys: object, return_inverse: bool = True, "
              "return_first_indices: bool = False, return_counts: bool = "
              "False) -> typing.Awaitable[typing.Any]"));

  m.def("wake_fd", []() { return se::py::GetPythonWakeManager().fd(); });
  m.def("drain_wake_queue", []() { se::py::GetPythonWakeManager().Drain(); });
  m.def("_clear_wake_queue", []() { se::py::GetPythonWakeManager().Clear(); });
  m.def("set_leak_warnings",
        [](bool enable) { nb::set_leak_warnings(enable); });

  nb::class_<PyPartitioner>(m, "Partitioner")
      .def("partition", &PyPartitioner::partition, nb::arg("keys"))
      .def("async_partition", &PyPartitioner::async_partition, nb::arg("keys"),
           nb::sig("def async_partition(self, keys: object) -> "
                   "typing.Awaitable[typing.Any]"))
      .def("filter", &PyPartitioner::filter, nb::arg("keys"),
           nb::arg("target_shard"))
      .def("async_filter", &PyPartitioner::async_filter, nb::arg("keys"),
           nb::arg("target_shard"),
           nb::sig("def async_filter(self, keys: object, target_shard: int) -> "
                   "typing.Awaitable[typing.Any]"));

  m.def(
      "ModuloPartitioner",
      [](uint32_t num_shards) -> PyPartitioner {
        auto proxy = pro::make_proxy<se::ops::PartitionerFacade,
                                     se::ops::ModuloPartitioner>(num_shards);
        return PyPartitioner{
            std::make_shared<decltype(proxy)>(std::move(proxy))};
      },
      nb::arg("num_shards"));

  m.def(
      "HashPartitioner",
      [](uint32_t num_shards) -> PyPartitioner {
        auto proxy = pro::make_proxy<se::ops::PartitionerFacade,
                                     se::ops::HashPartitioner>(num_shards);
        return PyPartitioner{
            std::make_shared<decltype(proxy)>(std::move(proxy))};
      },
      nb::arg("num_shards"));

  m.def(
      "FastRangePartitioner",
      [](uint32_t num_shards) -> PyPartitioner {
        auto proxy = pro::make_proxy<se::ops::PartitionerFacade,
                                     se::ops::FastRangePartitioner>(num_shards);
        return PyPartitioner{
            std::make_shared<decltype(proxy)>(std::move(proxy))};
      },
      nb::arg("num_shards"));

  // ---------------------------------------------------------------------------
  // Key codec utilities
  // ---------------------------------------------------------------------------

  m.def(
      "encode_keys",
      [](nb::object raw_keys_obj, uint32_t slot_id,
         int slot_bits) -> nb::object {
        if (slot_bits <= 0 || slot_bits >= 64)
          throw std::invalid_argument("slot_bits must be in [1, 63]");
        if (slot_id >= (1ULL << slot_bits))
          throw std::invalid_argument(
              "slot_id is too large for the given slot_bits");

        auto [dlm, cap] = se::py::extract_dlpack_tensor(raw_keys_obj);
        if (dlm->dl_tensor.dtype.bits != 64)
          throw std::invalid_argument(
              "encode_keys requires a 64-bit key tensor; received " +
              std::to_string(dlm->dl_tensor.dtype.bits) + "-bit input");
        size_t n = static_cast<size_t>(dlm->dl_tensor.shape[0]);
        const uint64_t slot_prefix = static_cast<uint64_t>(slot_id)
                                     << (64 - slot_bits);
        const uint64_t raw_mask = (slot_bits == 64)
                                      ? ~uint64_t{0}
                                      : ((uint64_t{1} << (64 - slot_bits)) - 1);

        const void *src_data =
            reinterpret_cast<const char *>(dlm->dl_tensor.data) +
            dlm->dl_tensor.byte_offset;

        auto out_dlm = se::py::create_output_tensor<uint64_t>(
            std::array<int64_t, 1>{static_cast<int64_t>(n)});
        uint64_t *out = static_cast<uint64_t *>(out_dlm->dl_tensor.data);

        const uint64_t *src = static_cast<const uint64_t *>(src_data);

        size_t i = 0;
#if defined(__AVX512F__) && defined(__AVX512DQ__)
        __m512i v_prefix = _mm512_set1_epi64(static_cast<int64_t>(slot_prefix));
        __m512i v_mask = _mm512_set1_epi64(static_cast<int64_t>(raw_mask));
        for (; i + 8 <= n; i += 8) {
          __m512i v_raw = _mm512_loadu_si512(src + i);
          __m512i v_enc =
              _mm512_or_si512(v_prefix, _mm512_and_si512(v_raw, v_mask));
          _mm512_storeu_si512(out + i, v_enc);
        }
#endif
        for (; i < n; ++i)
          out[i] = slot_prefix | (src[i] & raw_mask);

        return se::py::wrap_output_tensor(out_dlm);
      },
      nb::arg("raw_keys"), nb::arg("slot_id"), nb::arg("slot_bits") = 16,
      "Encode raw keys by packing slot_id into the top slot_bits bits.\n"
      "Returns a uint64 tensor of the same length.");

  m.def(
      "decode_keys",
      [](nb::object encoded_keys_obj, int slot_bits) -> nb::object {
        if (slot_bits <= 0 || slot_bits >= 64)
          throw std::invalid_argument("slot_bits must be in [1, 63]");

        auto [dlm, cap] = se::py::extract_dlpack_tensor(encoded_keys_obj);
        if (dlm->dl_tensor.dtype.bits != 64)
          throw std::invalid_argument(
              "decode_keys requires a 64-bit encoded-key tensor; received " +
              std::to_string(dlm->dl_tensor.dtype.bits) + "-bit input");
        size_t n = static_cast<size_t>(dlm->dl_tensor.shape[0]);
        const uint64_t raw_mask = (slot_bits == 64)
                                      ? ~uint64_t{0}
                                      : ((uint64_t{1} << (64 - slot_bits)) - 1);
        const int shift = 64 - slot_bits;

        const uint64_t *src = reinterpret_cast<const uint64_t *>(
            reinterpret_cast<const char *>(dlm->dl_tensor.data) +
            dlm->dl_tensor.byte_offset);

        auto raw_dlm = se::py::create_output_tensor<uint64_t>(
            std::array<int64_t, 1>{static_cast<int64_t>(n)});
        auto slot_dlm = se::py::create_output_tensor<uint32_t>(
            std::array<int64_t, 1>{static_cast<int64_t>(n)});
        uint64_t *raw_out = static_cast<uint64_t *>(raw_dlm->dl_tensor.data);
        uint32_t *slot_out = static_cast<uint32_t *>(slot_dlm->dl_tensor.data);

        size_t i = 0;
#if defined(__AVX512F__) && defined(__AVX512DQ__)
        __m512i v_mask = _mm512_set1_epi64(static_cast<int64_t>(raw_mask));
        for (; i + 8 <= n; i += 8) {
          __m512i v_enc = _mm512_loadu_si512(src + i);
          __m512i v_raw = _mm512_and_si512(v_enc, v_mask);
          __m512i v_slot = _mm512_srli_epi64(v_enc, shift);
          _mm512_storeu_si512(raw_out + i, v_raw);
          // compress 8×uint64 slot_ids down to 8×uint32
          uint64_t tmp[8];
          _mm512_storeu_si512(tmp, v_slot);
          for (int j = 0; j < 8; ++j)
            slot_out[i + j] = static_cast<uint32_t>(tmp[j]);
        }
#endif
        for (; i < n; ++i) {
          raw_out[i] = src[i] & raw_mask;
          slot_out[i] = static_cast<uint32_t>(src[i] >> shift);
        }

        return nb::make_tuple(se::py::wrap_output_tensor(raw_dlm),
                              se::py::wrap_output_tensor(slot_dlm));
      },
      nb::arg("encoded_keys"), nb::arg("slot_bits") = 16,
      "Decode encoded keys into (raw_keys: uint64, slot_ids: uint32).\n"
      "Inverse of encode_keys.");
}
