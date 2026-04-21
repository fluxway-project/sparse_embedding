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
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>

#include <dlpack/dlpack.h>

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

#include "backend/hogwild/hogwild_embedding_store.hpp"
#include "bindings/async_bridge.hpp"
#include "bindings/dlpack_utils.hpp"
#include "bindings/python_wake_manager.hpp"
#include "ops/tensor_ops.hpp"

namespace nb = nanobind;

namespace {

inline void check_error(std::error_code ec) {
  if (ec)
    throw std::runtime_error(ec.message());
}

// =============================================================================
// KeyDType detection from ndarray
// =============================================================================

enum class KeyDType { kInt64, kUInt64 };

inline KeyDType
detect_key_dtype(const nb::ndarray<nb::c_contig, nb::device::cpu> &arr) {
  auto code = arr.dtype().code;
  auto bits = arr.dtype().bits;
  if (code == static_cast<uint8_t>(kDLInt) && bits == 64)
    return KeyDType::kInt64;
  if (code == static_cast<uint8_t>(kDLUInt) && bits == 64)
    return KeyDType::kUInt64;
  if (code == static_cast<uint8_t>(kDLInt) && bits == 32)
    return KeyDType::kInt64;
  throw std::invalid_argument("keys must be int32, int64 or uint64");
}

template <typename KeyT>
std::span<const KeyT>
keys_span(const nb::ndarray<nb::c_contig, nb::device::cpu> &arr) {
  return {static_cast<const KeyT *>(arr.data()), arr.shape(0)};
}

// =============================================================================
// dispatch_on_key: run a lambda with the correct KeyT deduced from ndarray
// =============================================================================

template <typename Fn>
auto dispatch_on_key(const nb::ndarray<nb::c_contig, nb::device::cpu> &keys,
                     Fn &&fn) {
  auto kdt = detect_key_dtype(keys);
  if (kdt == KeyDType::kInt64)
    return fn(keys_span<int64_t>(keys));
  return fn(keys_span<uint64_t>(keys));
}

// =============================================================================
// EmbeddingStoreWrapper: type-erased wrapper (variant over ValT)
// =============================================================================

class EmbeddingStoreWrapper {
  using StoreVar =
      std::variant<std::unique_ptr<se::HogwildEmbeddingStore<float>>,
                   std::unique_ptr<se::HogwildEmbeddingStore<se::float16_t>>,
                   std::unique_ptr<se::HogwildEmbeddingStore<se::bfloat16_t>>>;

  StoreVar store_;
  se::py::AsyncTracker async_tracker_;

  template <typename Fn> auto visit_store(Fn &&fn) {
    return std::visit([&](auto &ptr) -> decltype(auto) { return fn(*ptr); },
                      store_);
  }

public:
  EmbeddingStoreWrapper(std::unordered_map<uint32_t, size_t> slot_dims,
                        size_t num_threads, int init_type,
                        const std::string &dtype) {
    auto itype =
        (init_type == 0)
            ? se::HogwildEmbeddingStore<float>::InitType::kZeros
            : se::HogwildEmbeddingStore<float>::InitType::kRandomSeeded;
    size_t nt = (num_threads == 0)
                    ? std::max(1u, std::thread::hardware_concurrency())
                    : num_threads;

    if (dtype == "float32" || dtype == "f32") {
      store_ = std::make_unique<se::HogwildEmbeddingStore<float>>(
          std::move(slot_dims), nt, itype);
    } else if (dtype == "float16" || dtype == "f16") {
      store_ = std::make_unique<se::HogwildEmbeddingStore<se::float16_t>>(
          std::move(slot_dims), nt,
          static_cast<se::HogwildEmbeddingStore<se::float16_t>::InitType>(
              itype));
    } else if (dtype == "bfloat16" || dtype == "bf16") {
      store_ = std::make_unique<se::HogwildEmbeddingStore<se::bfloat16_t>>(
          std::move(slot_dims), nt,
          static_cast<se::HogwildEmbeddingStore<se::bfloat16_t>::InitType>(
              itype));
    } else {
      throw std::invalid_argument("unsupported dtype: " + dtype);
    }
  }

  // ---- Metadata ----

  size_t total_size() {
    return visit_store([](auto &s) { return s.total_size(); });
  }

  size_t max_dim() {
    return visit_store([](auto &s) { return s.max_dim(); });
  }

  size_t num_slots() {
    return visit_store([](auto &s) { return s.num_slots(); });
  }

  size_t dim_for_slot(uint32_t slot_id) {
    return visit_store([=](auto &s) { return s.dim_for_slot(slot_id); });
  }

  // ---- get_dims: return per-key dims ----

  nb::object get_dims(nb::ndarray<nb::c_contig, nb::device::cpu> keys) {
    size_t n = keys.shape(0);
    int64_t shape[1] = {static_cast<int64_t>(n)};
    DLManagedTensor *dlm =
        se::py::create_output_tensor<int64_t>(std::span(shape));
    auto *out = static_cast<int64_t *>(dlm->dl_tensor.data);

    visit_store([&](auto &s) {
      dispatch_on_key(keys, [&](auto ks) {
        for (size_t i = 0; i < n; ++i) {
          out[i] =
              static_cast<int64_t>(s.dim_for_key(static_cast<uint64_t>(ks[i])));
        }
      });
    });

    return se::py::wrap_output_tensor(dlm);
  }

  // ---- Sync Lookup (ragged) ----

  nb::tuple lookup(nb::ndarray<nb::c_contig, nb::device::cpu> keys) {
    return visit_store(
        [&](auto &store) -> nb::tuple { return lookup_impl(store, keys); });
  }

  // ---- Sync lookup_or_insert ----

  nb::tuple lookup_or_insert(nb::ndarray<nb::c_contig, nb::device::cpu> keys) {
    return visit_store([&](auto &store) -> nb::tuple {
      return lookup_or_insert_impl(store, keys);
    });
  }

  // ---- Sync assign ----

  void assign(nb::ndarray<nb::c_contig, nb::device::cpu> keys,
              nb::ndarray<nb::c_contig, nb::device::cpu> values) {
    visit_store([&](auto &store) { assign_impl(store, keys, values); });
  }

  // ---- Sync contains ----

  nb::object contains(nb::ndarray<nb::c_contig, nb::device::cpu> keys) {
    size_t n = keys.shape(0);
    int64_t shape[1] = {static_cast<int64_t>(n)};
    DLManagedTensor *dlm =
        se::py::create_output_tensor<uint8_t>(std::span(shape));
    auto buf = std::make_unique<bool[]>(n);
    std::fill_n(buf.get(), n, false);

    std::error_code ec;
    {
      nb::gil_scoped_release release;
      auto exists_span = std::span<bool>(buf.get(), n);
      dispatch_on_key(keys, [&](auto ks) {
        visit_store([&](auto &s) { ec = s.contains(ks, exists_span); });
      });

      auto *out = static_cast<uint8_t *>(dlm->dl_tensor.data);
      for (size_t i = 0; i < n; ++i)
        out[i] = buf[i] ? 1 : 0;
    }

    check_error(ec);
    return se::py::wrap_output_tensor(dlm);
  }

  // ---- Sync erase ----

  void erase(nb::ndarray<nb::c_contig, nb::device::cpu> keys) {
    std::error_code ec;
    {
      nb::gil_scoped_release release;
      dispatch_on_key(keys, [&](auto ks) {
        visit_store([&](auto &s) { ec = s.erase(ks); });
      });
    }
    check_error(ec);
  }

  // ---- Sync clear ----

  void clear() {
    nb::gil_scoped_release release;
    visit_store([](auto &s) { s.clear(); });
  }

  // ---- Assign scores ----

  void assign_scores(nb::ndarray<nb::c_contig, nb::device::cpu> keys,
                     nb::ndarray<nb::c_contig, nb::device::cpu> scores) {
    std::error_code ec;
    auto scores_span = std::span<const uint64_t>(
        static_cast<const uint64_t *>(scores.data()), scores.shape(0));
    {
      nb::gil_scoped_release release;
      dispatch_on_key(keys, [&](auto ks) {
        visit_store([&](auto &s) { ec = s.assign_scores(ks, scores_span); });
      });
    }
    check_error(ec);
  }

  // ---- Async lookup ----

  nb::object async_lookup(nb::ndarray<nb::c_contig, nb::device::cpu> keys) {
    return visit_store([&](auto &store) -> nb::object {
      return async_lookup_impl(store, keys);
    });
  }

  // ---- Async assign ----

  nb::object async_assign(nb::ndarray<nb::c_contig, nb::device::cpu> keys,
                          nb::ndarray<nb::c_contig, nb::device::cpu> values) {
    return visit_store([&](auto &store) -> nb::object {
      return async_assign_impl(store, keys, values);
    });
  }

private:
  // ---- lookup_impl (typed) ----

  template <typename ValT>
  nb::tuple lookup_impl(se::HogwildEmbeddingStore<ValT> &store,
                        nb::ndarray<nb::c_contig, nb::device::cpu> keys) {
    size_t n = keys.shape(0);

    DLManagedTensor *vals_dlm = nullptr;
    DLManagedTensor *lens_dlm = nullptr;
    std::error_code ec;

    {
      nb::gil_scoped_release release;

      dispatch_on_key(keys, [&](auto ks) {
        auto row_lengths = std::make_unique<int64_t[]>(n);
        size_t total = store.compute_output_size(ks, {row_lengths.get(), n});

        int64_t vals_shape[] = {static_cast<int64_t>(total)};
        vals_dlm = se::py::create_output_tensor<ValT>(vals_shape);
        int64_t lens_shape[] = {static_cast<int64_t>(n)};
        lens_dlm = se::py::create_output_tensor<int64_t>(lens_shape);
        std::memcpy(lens_dlm->dl_tensor.data, row_lengths.get(),
                    n * sizeof(int64_t));

        auto vals_span = std::span<ValT>(
            static_cast<ValT *>(vals_dlm->dl_tensor.data), total);
        ec = store.lookup(ks, vals_span, {row_lengths.get(), n});
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

  // ---- lookup_or_insert_impl ----

  template <typename ValT>
  nb::tuple
  lookup_or_insert_impl(se::HogwildEmbeddingStore<ValT> &store,
                        nb::ndarray<nb::c_contig, nb::device::cpu> keys) {
    size_t n = keys.shape(0);

    DLManagedTensor *vals_dlm = nullptr;
    DLManagedTensor *lens_dlm = nullptr;
    DLManagedTensor *mask_dlm = nullptr;
    std::error_code ec;

    {
      nb::gil_scoped_release release;

      dispatch_on_key(keys, [&](auto ks) {
        auto row_lengths = std::make_unique<int64_t[]>(n);
        size_t total = store.compute_output_size(ks, {row_lengths.get(), n});

        int64_t vals_shape[] = {static_cast<int64_t>(total)};
        vals_dlm = se::py::create_output_tensor<ValT>(vals_shape);
        int64_t lens_shape[] = {static_cast<int64_t>(n)};
        lens_dlm = se::py::create_output_tensor<int64_t>(lens_shape);
        int64_t mask_shape[] = {static_cast<int64_t>(n)};
        mask_dlm = se::py::create_output_tensor<uint8_t>(mask_shape);

        std::memcpy(lens_dlm->dl_tensor.data, row_lengths.get(),
                    n * sizeof(int64_t));

        auto vals_span = std::span<ValT>(
            static_cast<ValT *>(vals_dlm->dl_tensor.data), total);

        auto mask_buf = std::make_unique<bool[]>(n);
        std::fill_n(mask_buf.get(), n, false);
        ec = store.lookup_or_insert(ks, vals_span, {row_lengths.get(), n},
                                    std::span<bool>(mask_buf.get(), n));

        auto *mask_out = static_cast<uint8_t *>(mask_dlm->dl_tensor.data);
        for (size_t i = 0; i < n; ++i)
          mask_out[i] = mask_buf[i] ? 1 : 0;
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

  // ---- assign_impl ----

  template <typename ValT>
  void assign_impl(se::HogwildEmbeddingStore<ValT> &store,
                   nb::ndarray<nb::c_contig, nb::device::cpu> keys,
                   nb::ndarray<nb::c_contig, nb::device::cpu> values) {
    size_t n = keys.shape(0);

    std::error_code ec;
    {
      nb::gil_scoped_release release;

      dispatch_on_key(keys, [&](auto ks) {
        if (values.ndim() == 2) {
          size_t dim = values.shape(1);
          auto val_span = std::span<const ValT>(
              static_cast<const ValT *>(values.data()), n * dim);
          ec = store.assign_uniform(ks, val_span, dim);
        } else if (values.ndim() == 1) {
          auto row_lengths = std::make_unique<int64_t[]>(n);
          store.compute_output_size(ks, {row_lengths.get(), n});
          auto val_span = std::span<const ValT>(
              static_cast<const ValT *>(values.data()), values.shape(0));
          ec = store.assign(ks, val_span, {row_lengths.get(), n});
        } else {
          throw std::invalid_argument("values must be 1D (ragged) or 2D");
        }
      });
    }

    check_error(ec);
  }

  // ---- async_lookup_impl ----

  template <typename ValT>
  nb::object
  async_lookup_impl(se::HogwildEmbeddingStore<ValT> &store,
                    nb::ndarray<nb::c_contig, nb::device::cpu> keys) {
    size_t n = keys.shape(0);
    auto kdt = detect_key_dtype(keys);

    std::shared_ptr<void> keys_copy;
    if (kdt == KeyDType::kInt64) {
      auto *buf = new int64_t[n];
      std::memcpy(buf, keys.data(), n * sizeof(int64_t));
      keys_copy.reset(buf, [](void *p) { delete[] static_cast<int64_t *>(p); });
    } else {
      auto *buf = new uint64_t[n];
      std::memcpy(buf, keys.data(), n * sizeof(uint64_t));
      keys_copy.reset(buf,
                      [](void *p) { delete[] static_cast<uint64_t *>(p); });
    }

    auto *store_ptr = &store;

    return se::py::async_dispatch(
        async_tracker_,
        [store_ptr, n, kdt,
         keys_copy]() -> std::pair<DLManagedTensor *, DLManagedTensor *> {
          DLManagedTensor *vals_dlm = nullptr;
          DLManagedTensor *lens_dlm = nullptr;

          auto do_lookup = [&](auto ks) {
            auto row_lengths = std::make_unique<int64_t[]>(n);
            size_t total =
                store_ptr->compute_output_size(ks, {row_lengths.get(), n});

            int64_t vals_shape[] = {static_cast<int64_t>(total)};
            vals_dlm = se::py::create_output_tensor<ValT>(vals_shape);
            int64_t lens_shape[] = {static_cast<int64_t>(n)};
            lens_dlm = se::py::create_output_tensor<int64_t>(lens_shape);
            std::memcpy(lens_dlm->dl_tensor.data, row_lengths.get(),
                        n * sizeof(int64_t));

            auto vals_span = std::span<ValT>(
                static_cast<ValT *>(vals_dlm->dl_tensor.data), total);
            auto ec = store_ptr->lookup(ks, vals_span, {row_lengths.get(), n});
            if (ec)
              throw std::runtime_error(ec.message());
          };

          if (kdt == KeyDType::kInt64) {
            auto ks = std::span<const int64_t>(
                static_cast<const int64_t *>(keys_copy.get()), n);
            do_lookup(ks);
          } else {
            auto ks = std::span<const uint64_t>(
                static_cast<const uint64_t *>(keys_copy.get()), n);
            do_lookup(ks);
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
  }

  // ---- async_assign_impl ----

  template <typename ValT>
  nb::object
  async_assign_impl(se::HogwildEmbeddingStore<ValT> &store,
                    nb::ndarray<nb::c_contig, nb::device::cpu> keys,
                    nb::ndarray<nb::c_contig, nb::device::cpu> values) {
    size_t n = keys.shape(0);
    auto kdt = detect_key_dtype(keys);

    std::shared_ptr<void> keys_copy;
    if (kdt == KeyDType::kInt64) {
      auto *buf = new int64_t[n];
      std::memcpy(buf, keys.data(), n * sizeof(int64_t));
      keys_copy.reset(buf, [](void *p) { delete[] static_cast<int64_t *>(p); });
    } else {
      auto *buf = new uint64_t[n];
      std::memcpy(buf, keys.data(), n * sizeof(uint64_t));
      keys_copy.reset(buf,
                      [](void *p) { delete[] static_cast<uint64_t *>(p); });
    }

    size_t val_count = 1;
    for (size_t d = 0; d < values.ndim(); ++d)
      val_count *= values.shape(d);
    auto vals_copy = std::make_shared<std::vector<ValT>>(val_count);
    std::memcpy(vals_copy->data(), values.data(), val_count * sizeof(ValT));
    auto val_ndim = values.ndim();
    auto val_dim1 = (val_ndim == 2) ? values.shape(1) : size_t(0);

    auto *store_ptr = &store;

    return se::py::async_dispatch(
        async_tracker_,
        [store_ptr, n, kdt, keys_copy, vals_copy, val_ndim,
         val_dim1]() -> std::error_code {
          auto do_assign = [&](auto ks) -> std::error_code {
            if (val_ndim == 2) {
              return store_ptr->assign_uniform(
                  ks, std::span<const ValT>(*vals_copy), val_dim1);
            }
            auto row_lengths = std::make_unique<int64_t[]>(n);
            store_ptr->compute_output_size(ks, {row_lengths.get(), n});
            return store_ptr->assign(ks, std::span<const ValT>(*vals_copy),
                                     {row_lengths.get(), n});
          };

          if (kdt == KeyDType::kInt64) {
            return do_assign(std::span<const int64_t>(
                static_cast<const int64_t *>(keys_copy.get()), n));
          }
          return do_assign(std::span<const uint64_t>(
              static_cast<const uint64_t *>(keys_copy.get()), n));
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
  }
};

} // namespace

// =============================================================================
// NB_MODULE
// =============================================================================

NB_MODULE(_core, m) {
  m.doc() = "sparse_embedding C++ backend: multi-slot, arena-allocated, "
            "DLPack zero-copy, folly CHM, stdexec parallelism";

  nb::class_<se::py::DLPackCapsuleWrapper>(m, "DLPackCapsuleWrapper")
      .def("__dlpack__", &se::py::DLPackCapsuleWrapper::dlpack, nb::kw_only(),
           nb::arg("stream") = nb::none(), nb::arg("max_version") = nb::none(),
           nb::arg("dl_device") = nb::none(), nb::arg("copy") = nb::none())
      .def("__dlpack_device__", &se::py::DLPackCapsuleWrapper::dlpack_device);

  nb::class_<EmbeddingStoreWrapper>(m, "EmbeddingStore")
      .def(nb::init<std::unordered_map<uint32_t, size_t>, size_t, int,
                    std::string>(),
           nb::arg("slot_dims"), nb::arg("num_threads") = 0,
           nb::arg("init_type") = 0, nb::arg("dtype") = "float32")

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

      .def("async_lookup", &EmbeddingStoreWrapper::async_lookup,
           nb::arg("keys"))
      .def("async_assign", &EmbeddingStoreWrapper::async_assign,
           nb::arg("keys"), nb::arg("values"));

  m.def("async_unique", [](nb::ndarray<nb::c_contig, nb::device::cpu> keys) {
    size_t n = keys.shape(0);
    auto kdt = detect_key_dtype(keys);

    std::shared_ptr<void> keys_copy;
    if (kdt == KeyDType::kInt64) {
      auto *buf = new int64_t[n];
      std::memcpy(buf, keys.data(), n * sizeof(int64_t));
      keys_copy.reset(buf, [](void *p) { delete[] static_cast<int64_t *>(p); });
    } else {
      auto *buf = new uint64_t[n];
      std::memcpy(buf, keys.data(), n * sizeof(uint64_t));
      keys_copy.reset(buf,
                      [](void *p) { delete[] static_cast<uint64_t *>(p); });
    }

    static se::py::AsyncTracker global_tracker;

    return se::py::async_dispatch(
        global_tracker,
        [n, kdt,
         keys_copy]() -> std::pair<DLManagedTensor *, DLManagedTensor *> {
          size_t u_count = 0;
          DLManagedTensor *u_dlm = nullptr;
          DLManagedTensor *inv_dlm = nullptr;

          if (kdt == KeyDType::kInt64) {
            auto span = std::span<const int64_t>(
                static_cast<const int64_t *>(keys_copy.get()), n);
            auto res = se::ops::unique_keys_inverse(span);

            int64_t shape_u[] = {static_cast<int64_t>(res.unique_count)};
            u_dlm = se::py::create_output_tensor<int64_t>(shape_u);
            std::memcpy(u_dlm->dl_tensor.data, res.unique_keys.get(),
                        res.unique_count * sizeof(int64_t));

            int64_t shape_inv[] = {static_cast<int64_t>(res.inverse_count)};
            inv_dlm = se::py::create_output_tensor<int64_t>(shape_inv);
            std::memcpy(inv_dlm->dl_tensor.data, res.inverse_indices.get(),
                        res.inverse_count * sizeof(int64_t));
          } else {
            auto span = std::span<const uint64_t>(
                static_cast<const uint64_t *>(keys_copy.get()), n);
            auto res = se::ops::unique_keys_inverse(span);

            int64_t shape_u[] = {static_cast<int64_t>(res.unique_count)};
            u_dlm = se::py::create_output_tensor<uint64_t>(shape_u);
            std::memcpy(u_dlm->dl_tensor.data, res.unique_keys.get(),
                        res.unique_count * sizeof(uint64_t));

            int64_t shape_inv[] = {static_cast<int64_t>(res.inverse_count)};
            inv_dlm = se::py::create_output_tensor<int64_t>(shape_inv);
            std::memcpy(inv_dlm->dl_tensor.data, res.inverse_indices.get(),
                        res.inverse_count * sizeof(int64_t));
          }

          return {u_dlm, inv_dlm};
        },
        [](nb::object future,
           std::pair<DLManagedTensor *, DLManagedTensor *> result) {
          auto [u_dlm, inv_dlm] = result;
          future.attr("set_result")(
              nb::make_tuple(se::py::wrap_output_tensor(u_dlm),
                             se::py::wrap_output_tensor(inv_dlm)));
        });
  });

  m.def("wake_fd", []() { return se::py::GetPythonWakeManager().fd(); });
  m.def("drain_wake_queue", []() { se::py::GetPythonWakeManager().Drain(); });
}
