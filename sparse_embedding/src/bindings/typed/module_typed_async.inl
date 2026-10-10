// Copyright 2026 The Fluxway Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0
// =============================================================================
// Implementation of AsyncOpsImpl<ParamT, OptT> methods
// =============================================================================

#include "sparse_embedding/src/bindings/module_wrapper.hpp"
#include "sparse_embedding/src/bindings/async_bridge.hpp"

namespace se::py::detail {

template <typename ParamT, typename OptT>
nb::object AsyncOpsImpl<ParamT, OptT>::async_lookup(
    EmbeddingStoreWrapper &self, nb::object keys, int64_t dim) {
  auto [dlm_keys, cap_keys] = se::py::extract_dlpack_tensor(keys);
  size_t n = get_dlm_size(dlm_keys);
  auto kdt = detect_key_dtype(dlm_keys);
  validate_key_dtype(kdt, self.key_type_enum_);
  const void *keys_data =
      reinterpret_cast<const char *>(dlm_keys->dl_tensor.data) +
      dlm_keys->dl_tensor.byte_offset;
  auto *wrapper_ptr = &self;
  return se::py::async_dispatch(
      self.async_pool_,
      [wrapper_ptr, n, kdt, keys_data, dim,
       keys_keeper = std::make_shared<se::py::AsyncPyObjectKeeper>(
           cap_keys)]() -> std::pair<DLManagedTensor *, DLManagedTensor *> {
        DLManagedTensor *vals_dlm = nullptr;
        DLManagedTensor *lens_dlm = nullptr;
        try {
          auto do_lookup = [&](auto ks) {
            size_t total = static_cast<size_t>(dim) * n;
            auto row_lengths = std::make_unique<int64_t[]>(n);
            fill_uniform_row_lengths({row_lengths.get(), n}, dim);
            int64_t vals_shape[] = {static_cast<int64_t>(total)};
            vals_dlm = se::py::create_output_tensor<ParamT>(vals_shape);
            lens_dlm =
                se::py::wrap_unique_ptr_as_tensor(std::move(row_lengths), n);
            auto vals_span = std::span<ParamT>(
                static_cast<ParamT *>(vals_dlm->dl_tensor.data), total);
            auto ec = wrapper_ptr->get_store<ParamT, OptT>()->forward_lookup(
                ks, vals_span,
                {static_cast<const int64_t *>(lens_dlm->dl_tensor.data), n});
            if (ec)
              throw std::runtime_error(ec.message());
          };
          dispatch_key_span(kdt, keys_data, n, do_lookup);
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
}

template <typename ParamT, typename OptT>
nb::object AsyncOpsImpl<ParamT, OptT>::async_assign(
    EmbeddingStoreWrapper &self, nb::object keys, nb::object values) {
  auto [dlm_keys, cap_keys] = se::py::extract_dlpack_tensor(keys);
  auto [dlm_values, cap_values] = se::py::extract_dlpack_tensor(values);
  size_t n = get_dlm_size(dlm_keys);
  int64_t val_dim = 0;
  if (dlm_values->dl_tensor.ndim == 2) {
    if (dlm_values->dl_tensor.shape[0] != static_cast<int64_t>(n))
      throw std::invalid_argument("values shape[0] must match keys length");
    val_dim = dlm_values->dl_tensor.shape[1];
  } else if (dlm_values->dl_tensor.ndim == 1) {
    if (n > 0) {
      if (dlm_values->dl_tensor.shape[0] % n != 0)
        throw std::invalid_argument(
            "1D values length must be a multiple of keys length");
      val_dim = dlm_values->dl_tensor.shape[0] / n;
    }
  } else {
    throw std::invalid_argument("values must be a 1D or 2D tensor");
  }
  auto kdt = detect_key_dtype(dlm_keys);
  validate_key_dtype(kdt, self.key_type_enum_);
  const void *keys_data =
      reinterpret_cast<const char *>(dlm_keys->dl_tensor.data) +
      dlm_keys->dl_tensor.byte_offset;
  const void *values_data =
      reinterpret_cast<const char *>(dlm_values->dl_tensor.data) +
      dlm_values->dl_tensor.byte_offset;
  auto *wrapper_ptr = &self;
  return se::py::async_dispatch(
      self.async_pool_,
      [wrapper_ptr, n, kdt, keys_data, values_data, val_dim,
       keys_keeper =
           std::make_shared<se::py::AsyncPyObjectKeeper>(cap_keys),
       values_keeper = std::make_shared<se::py::AsyncPyObjectKeeper>(
           cap_values)]() -> std::error_code {
        auto do_assign = [&](auto ks) -> std::error_code {
          auto& tls_row_lengths = get_tls_row_lengths();
          tls_row_lengths.resize(n);
          fill_uniform_row_lengths(tls_row_lengths, val_dim);
          return wrapper_ptr->get_store<ParamT, OptT>()->assign(
              ks,
              std::span<const ParamT>(
                  static_cast<const ParamT *>(values_data), n * val_dim),
              tls_row_lengths);
        };
        return dispatch_key_span(kdt, keys_data, n, do_assign);
      },
      [](nb::object future, std::error_code ec) {
        if (!ec)
          future.attr("set_result")(nb::none());
        else
          future.attr("set_exception")(
              nb::module_::import_("builtins")
                  .attr("RuntimeError")(ec.message()));
      });
}

template <typename ParamT, typename OptT>
nb::object AsyncOpsImpl<ParamT, OptT>::async_lookup_or_insert(
    EmbeddingStoreWrapper &self, nb::object keys, int64_t dim) {
  auto [dlm_keys, cap_keys] = se::py::extract_dlpack_tensor(keys);
  size_t n = get_dlm_size(dlm_keys);
  auto kdt = detect_key_dtype(dlm_keys);
  validate_key_dtype(kdt, self.key_type_enum_);
  const void *keys_data =
      reinterpret_cast<const char *>(dlm_keys->dl_tensor.data) +
      dlm_keys->dl_tensor.byte_offset;
  auto *wrapper_ptr = &self;
  return se::py::async_dispatch(
      self.async_pool_,
      [wrapper_ptr, n, kdt, keys_data, dim,
       keys_keeper = std::make_shared<se::py::AsyncPyObjectKeeper>(
           cap_keys)]()
          -> std::pair<DLManagedTensor *, DLManagedTensor *> {
        DLManagedTensor *vals_dlm = nullptr, *lens_dlm = nullptr;
        try {
          auto do_loi = [&](auto ks) {
            size_t total = static_cast<size_t>(dim) * n;
            auto row_lengths = std::make_unique<int64_t[]>(n);
            fill_uniform_row_lengths({row_lengths.get(), n}, dim);
            lens_dlm =
                se::py::wrap_unique_ptr_as_tensor(std::move(row_lengths), n);
            auto *lens_out =
                static_cast<int64_t *>(lens_dlm->dl_tensor.data);
            int64_t vals_shape[] = {static_cast<int64_t>(total)};
            vals_dlm = se::py::create_output_tensor<ParamT>(vals_shape);
            auto vals_span = std::span<ParamT>(
                static_cast<ParamT *>(vals_dlm->dl_tensor.data), total);
            auto ec =
                wrapper_ptr->get_store<ParamT, OptT>()->lookup_or_insert(
                    ks, vals_span, {lens_out, n});
            if (ec)
              throw std::runtime_error(ec.message());
          };
          dispatch_key_span(kdt, keys_data, n, do_loi);
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
        auto [v, l] = result;
        future.attr("set_result")(
            nb::make_tuple(se::py::wrap_output_tensor(v),
                           se::py::wrap_output_tensor(l)));
      });
}

template <typename ParamT, typename OptT>
nb::object AsyncOpsImpl<ParamT, OptT>::async_accumulate_gradients(
    EmbeddingStoreWrapper &self, nb::object keys, nb::object grads) {
  auto [dlm_keys, cap_keys] = se::py::extract_dlpack_tensor(keys);
  auto [dlm_grads, cap_grads] = se::py::extract_dlpack_tensor(grads);
  size_t n = get_dlm_size(dlm_keys);
  int64_t dim = 0;
  if (dlm_grads->dl_tensor.ndim == 2) {
    if (dlm_grads->dl_tensor.shape[0] != static_cast<int64_t>(n))
      throw std::invalid_argument("grads shape[0] must match keys length");
    dim = dlm_grads->dl_tensor.shape[1];
  } else if (dlm_grads->dl_tensor.ndim == 1) {
    if (n > 0) {
      if (dlm_grads->dl_tensor.shape[0] % n != 0)
        throw std::invalid_argument(
            "1D grads length must be a multiple of keys length");
      dim = dlm_grads->dl_tensor.shape[0] / n;
    }
  } else {
    throw std::invalid_argument("grads must be a 1D or 2D tensor");
  }
  auto kdt = detect_key_dtype(dlm_keys);
  validate_key_dtype(kdt, self.key_type_enum_);
  auto *wrapper_ptr = &self;
  return dispatch_value_dtype(self.gradient_dtype_code_,
                             [&]<typename GradT>() -> nb::object {
    auto grads_span = se::py::dlpack_to_span<const GradT>(dlm_grads);
    const GradT *grads_data = grads_span.data();
    const void *keys_data =
        reinterpret_cast<const char *>(dlm_keys->dl_tensor.data) +
        dlm_keys->dl_tensor.byte_offset;
    return se::py::async_dispatch(
        self.async_pool_,
        [wrapper_ptr, n, kdt, keys_data, grads_data, dim,
         keys_keeper =
             std::make_shared<se::py::AsyncPyObjectKeeper>(cap_keys),
         grads_keeper = std::make_shared<se::py::AsyncPyObjectKeeper>(
             cap_grads)]() -> std::error_code {
          auto do_acc = [&](auto ks) -> std::error_code {
            auto& tls_row_lengths = get_tls_row_lengths();
            tls_row_lengths.resize(n);
            fill_uniform_row_lengths(tls_row_lengths, dim);
            return wrapper_ptr->get_store<ParamT, OptT>()
                ->accumulate_gradients(
                    ks, std::span<const GradT>(grads_data, n * dim),
                    tls_row_lengths);
          };
          return dispatch_key_span(kdt, keys_data, n, do_acc);
        },
        [](nb::object future, std::error_code ec) {
          if (!ec)
            future.attr("set_result")(nb::none());
          else
            future.attr("set_exception")(
                nb::module_::import_("builtins")
                  .attr("RuntimeError")(ec.message()));
      });
  });
}

template <typename ParamT, typename OptT>
nb::object
AsyncOpsImpl<ParamT, OptT>::async_optimizer_step(EmbeddingStoreWrapper &self) {
  auto *wrapper_ptr = &self;
  return se::py::async_dispatch(
      self.async_pool_,
      [wrapper_ptr]() -> std::error_code {
        return wrapper_ptr->get_store<ParamT, OptT>()->optimizer_step();
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

template <typename ParamT, typename OptT>
nb::object AsyncOpsImpl<ParamT, OptT>::async_contains(
    EmbeddingStoreWrapper &self, nb::object keys) {
  auto [dlm_keys, cap_keys] = se::py::extract_dlpack_tensor(keys);
  size_t n = get_dlm_size(dlm_keys);
  auto kdt = detect_key_dtype(dlm_keys);
  validate_key_dtype(kdt, self.key_type_enum_);
  const void *keys_data =
      reinterpret_cast<const char *>(dlm_keys->dl_tensor.data) +
      dlm_keys->dl_tensor.byte_offset;
  auto *wrapper_ptr = &self;
  return se::py::async_dispatch(
      self.async_pool_,
      [wrapper_ptr, n, kdt, keys_data,
       keys_keeper = std::make_shared<se::py::AsyncPyObjectKeeper>(
           cap_keys)]() -> DLManagedTensor * {
        DLManagedTensor *out_dlm = nullptr;
        try {
          auto do_contains = [&](auto ks) {
            int64_t shape[1] = {static_cast<int64_t>(n)};
            out_dlm =
                se::py::create_output_tensor<uint8_t>(std::span(shape));
            auto *out = static_cast<uint8_t *>(out_dlm->dl_tensor.data);
            auto ec = wrapper_ptr->get_store<ParamT, OptT>()->contains(
                ks, std::span<uint8_t>(out, n));
            if (ec)
              throw std::runtime_error(ec.message());
          };
          dispatch_key_span(kdt, keys_data, n, do_contains);
        } catch (...) {
          if (out_dlm)
            out_dlm->deleter(out_dlm);
          throw;
        }
        return out_dlm;
      },
      [](nb::object future, DLManagedTensor *mask_dlm) {
        future.attr("set_result")(se::py::wrap_output_tensor(mask_dlm));
      });
}

template <typename ParamT, typename OptT>
nb::object AsyncOpsImpl<ParamT, OptT>::async_erase(
    EmbeddingStoreWrapper &self, nb::object keys) {
  auto [dlm_keys, cap_keys] = se::py::extract_dlpack_tensor(keys);
  size_t n = get_dlm_size(dlm_keys);
  auto kdt = detect_key_dtype(dlm_keys);
  validate_key_dtype(kdt, self.key_type_enum_);
  const void *keys_data =
      reinterpret_cast<const char *>(dlm_keys->dl_tensor.data) +
      dlm_keys->dl_tensor.byte_offset;
  auto *wrapper_ptr = &self;
  return se::py::async_dispatch(
      self.async_pool_,
      [wrapper_ptr, n, kdt, keys_data,
       keys_keeper = std::make_shared<se::py::AsyncPyObjectKeeper>(
           cap_keys)]() -> std::error_code {
        auto erase = [&](auto keys_span) {
          return wrapper_ptr->get_store<ParamT, OptT>()->erase(keys_span);
        };
        return dispatch_key_span(kdt, keys_data, n, erase);
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

template <typename ParamT, typename OptT>
nb::object
AsyncOpsImpl<ParamT, OptT>::async_clear(EmbeddingStoreWrapper &self) {
  auto *wrapper_ptr = &self;
  return se::py::async_dispatch(
      self.async_pool_,
      [wrapper_ptr]() -> std::error_code {
        wrapper_ptr->get_store<ParamT, OptT>()->clear();
        return std::error_code{};
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

template <typename ParamT, typename OptT>
nb::object AsyncOpsImpl<ParamT, OptT>::async_assign_scores(
    EmbeddingStoreWrapper &self, nb::object keys, nb::object scores) {
  auto [dlm_keys, cap_keys] = se::py::extract_dlpack_tensor(keys);
  auto [dlm_scores, cap_scores] = se::py::extract_dlpack_tensor(scores);
  size_t n = get_dlm_size(dlm_keys);
  auto kdt = detect_key_dtype(dlm_keys);
  validate_key_dtype(kdt, self.key_type_enum_);
  const void *keys_data =
      reinterpret_cast<const char *>(dlm_keys->dl_tensor.data) +
      dlm_keys->dl_tensor.byte_offset;
  const void *scores_data =
      reinterpret_cast<const char *>(dlm_scores->dl_tensor.data) +
      dlm_scores->dl_tensor.byte_offset;
  auto *wrapper_ptr = &self;
  return se::py::async_dispatch(
      self.async_pool_,
      [wrapper_ptr, n, kdt, keys_data, scores_data,
       keys_keeper =
           std::make_shared<se::py::AsyncPyObjectKeeper>(cap_keys),
       scores_keeper = std::make_shared<se::py::AsyncPyObjectKeeper>(
           cap_scores)]() -> std::error_code {
        auto scores_span = std::span<const uint64_t>(
            static_cast<const uint64_t *>(scores_data), n);
        auto assign_scores = [&](auto keys_span) {
          return wrapper_ptr->get_store<ParamT, OptT>()->assign_scores(
              keys_span, scores_span);
        };
        return dispatch_key_span(kdt, keys_data, n, assign_scores);
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

} // namespace se::py::detail
