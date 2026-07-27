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
#include <format>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>

#include "sparse_embedding/src/bindings/async_bridge.hpp"
#include "sparse_embedding/src/bindings/dlpack_utils.hpp"
#include "sparse_embedding/src/bindings/module_wrapper.hpp"
#include "sparse_embedding/src/common/slot_key_codec.hpp"
#include "sparse_embedding/src/ops/partitioner.hpp"
#include "sparse_embedding/src/ops/tensor_ops.hpp"
#ifdef SE_ENABLE_LANCE_CHECKPOINTER
#include "sparse_embedding/src/checkpoint/lance/lance_checkpoint.hpp"
#endif

namespace nb = nanobind;

using namespace se::py::detail;

// =============================================================================
// EmbeddingStoreWrapper constructor (out-of-line to avoid cross-TU
// instantiation of ConstructOpsImpl in every .inl translation unit)
// =============================================================================

namespace se::py::detail {

EmbeddingStoreWrapper::EmbeddingStoreWrapper(
    const se::EmbeddingConfig &emb_cfg,
    std::optional<PyInitializer> init_cfg_opt,
    std::optional<PyOptimizer> opt_cfg_opt, std::string score_policy_name)
    : async_pool_((emb_cfg.num_threads == 0)
                      ? std::max(1u, std::thread::hardware_concurrency())
                      : emb_cfg.num_threads),
      value_dtype_(emb_cfg.value_dtype),
      optimizer_dtype_(emb_cfg.optimizer_dtype.empty()
                           ? emb_cfg.value_dtype
                           : emb_cfg.optimizer_dtype),
      key_dtype_(emb_cfg.key_dtype),
      storage_(se::create_page_storage(emb_cfg.storage)) {
  if (key_dtype_ == "int32" || key_dtype_ == "i32")
    key_type_enum_ = KeyDType::kInt32;
  else if (key_dtype_ == "uint32" || key_dtype_ == "u32")
    key_type_enum_ = KeyDType::kUInt32;
  else if (key_dtype_ == "int64" || key_dtype_ == "i64")
    key_type_enum_ = KeyDType::kInt64;
  else if (key_dtype_ == "uint64" || key_dtype_ == "u64" || key_dtype_.empty())
    key_type_enum_ = KeyDType::kUInt64;
  else
    throw std::invalid_argument(
        std::format("unsupported key_dtype: {}", key_dtype_));
  PyInitializer init_cfg = init_cfg_opt.value_or(
      PyInitializer{std::make_shared<pro::proxy<se::InitializerBuilderFacade>>(
          pro::make_proxy<se::InitializerBuilderFacade,
                          UniformInitializerBuilder>(-0.05f, 0.05f))});

  PyOptimizer opt_cfg = opt_cfg_opt.value_or(PyOptimizer{se::AdamConfig{}});

  dims_ = emb_cfg.dims;
  if (dims_.empty())
    throw std::invalid_argument("dims must not be empty");

  size_t num_threads = (emb_cfg.num_threads == 0)
                           ? std::max(1u, std::thread::hardware_concurrency())
                           : emb_cfg.num_threads;

  dispatch_types([&]<typename P, typename O>() {
    ConstructOpsImpl<P, O>::construct(*this, init_cfg, opt_cfg,
                                      score_policy_name, num_threads);
  });
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
             std::optional<double> low_watermark) {
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
            return options;
          },
          nb::arg("memory_limit_bytes") = nb::none(),
          nb::arg("spill_directory") = nb::none(),
          nb::arg("max_spill_bytes") = nb::none(),
          nb::arg("high_watermark") = nb::none(),
          nb::arg("low_watermark") = nb::none())
      .def_static("memory_only",
                  []() {
                    se::StorageOptions options;
                    options.mode = se::StorageMode::kMemoryOnly;
                    return options;
                  })
      .def_static(
          "disk_spill",
          [](const std::string &directory, size_t max_bytes) {
            se::StorageOptions options;
            options.mode = se::StorageMode::kDiskOnly;
            options.memory_limit_bytes = 0;
            options.spill_directory = directory;
            options.max_spill_bytes = max_bytes;
            return options;
          },
          nb::arg("directory"), nb::arg("max_bytes"));

  nb::class_<se::EmbeddingConfig>(m, "_EmbeddingConfig")
      .def(nb::init<std::vector<size_t>, size_t, std::string, std::string,
                    std::string, se::StorageOptions>(),
           nb::arg("dims"), nb::arg("num_threads") = 0,
           nb::arg("value_dtype") = "float32", nb::arg("optimizer_dtype") = "",
           nb::arg("key_dtype") = "int64",
           nb::arg("storage") = se::StorageOptions{})
      .def_rw("dims", &se::EmbeddingConfig::dims)
      .def_rw("num_threads", &se::EmbeddingConfig::num_threads)
      .def_rw("value_dtype", &se::EmbeddingConfig::value_dtype)
      .def_rw("optimizer_dtype", &se::EmbeddingConfig::optimizer_dtype)
      .def_rw("key_dtype", &se::EmbeddingConfig::key_dtype);

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
        self->config = se::AdamConfig{lr, beta1, beta2, eps, weight_decay};
      },
      nb::arg("lr") = 1e-3f, nb::arg("beta1") = 0.9f, nb::arg("beta2") = 0.999f,
      nb::arg("eps") = 1e-8f, nb::arg("weight_decay") = 0.0f);

  struct SgdOptimizerPy : PyOptimizer {};
  nb::class_<SgdOptimizerPy, PyOptimizer>(m, "Sgd").def(
      "__init__",
      [](SgdOptimizerPy *self, float lr, float weight_decay) {
        new (self) SgdOptimizerPy();
        self->config = se::SgdConfig{lr, weight_decay};
      },
      nb::arg("lr") = 1e-2f, nb::arg("weight_decay") = 0.0f);

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

      .def("export_batches_dlpack",
           &EmbeddingStoreWrapper::export_batches_dlpack, nb::arg("visitor"),
           nb::arg("batch_size") = 65536,
           nb::arg("include_optimizer_state") = false)
      .def("export_batches_arrow", &EmbeddingStoreWrapper::export_batches_arrow,
           nb::arg("visitor"), nb::arg("batch_size") = 65536,
           nb::arg("include_optimizer_state") = false)
      .def("assign_opt_state", &EmbeddingStoreWrapper::assign_opt_state,
           nb::arg("keys"), nb::arg("opt_states"), nb::arg("dimension"),
           nb::arg("steps") = nb::none())

#ifdef SE_ENABLE_LANCE_CHECKPOINTER
      .def("save_checkpoint", &EmbeddingStoreWrapper::save_checkpoint,
           nb::arg("uri"), nb::arg("mode") = "full", nb::arg("min_score") = 0,
           nb::arg("include_optimizer_state") = false,
           nb::arg("use_compression") = false)
      .def("load_checkpoint", &EmbeddingStoreWrapper::load_checkpoint,
           nb::arg("uri"), nb::arg("version") = 0, nb::arg("clear") = true,
           nb::arg("load_optimizer_state") = false,
           nb::arg("filter_fn") = nb::none(), nb::arg("target_shard_id") = 0,
           nb::arg("batch_size") = 65536)
      .def("set_checkpoint_factories",
           [](EmbeddingStoreWrapper &self, const std::string &backend) {
             if (backend == "lance") {
               auto wf = pro::make_proxy<se::CheckpointWriterFacade,
                                         se::LanceCheckpointWriter>();
               auto rf = pro::make_proxy<se::CheckpointReaderFacade,
                                         se::LanceCheckpointReader>();
               self.set_checkpoint_factories(std::move(wf), std::move(rf));
               return;
             }
             throw std::runtime_error("Unsupported backend: " + backend);
           })
#endif // SE_ENABLE_LANCE_CHECKPOINTER
      ;

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
