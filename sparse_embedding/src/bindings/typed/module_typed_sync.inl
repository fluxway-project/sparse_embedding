// Copyright 2026 The Fluxway Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0
// =============================================================================
// Implementation of SyncOpsImpl<ParamT, OptT> methods
// =============================================================================

#include "sparse_embedding/src/bindings/module_wrapper.hpp"

namespace se::py::detail {

template <typename ParamT, typename OptT>
nb::tuple SyncOpsImpl<ParamT, OptT>::lookup(EmbeddingStoreWrapper &self,
                                            nb::object keys, int64_t dim) {
  auto [dlm_keys, cap_keys] = se::py::extract_dlpack_tensor(keys);
  size_t n = get_dlm_size(dlm_keys);
  DLManagedTensor *vals_dlm = nullptr;
  DLManagedTensor *lens_dlm = nullptr;
  std::error_code ec;

  {
    nb::gil_scoped_release release;
    dispatch_on_key(dlm_keys, self.key_type_enum_, [&](auto ks) {
      size_t total = static_cast<size_t>(dim) * n;
      auto row_lengths = std::make_unique<int64_t[]>(n);
      fill_uniform_row_lengths({row_lengths.get(), n}, dim);

      int64_t vals_shape[] = {static_cast<int64_t>(total)};
      vals_dlm = se::py::create_output_tensor<ParamT>(vals_shape);
      lens_dlm = se::py::wrap_unique_ptr_as_tensor(std::move(row_lengths), n);

      auto vals_span = std::span<ParamT>(
          static_cast<ParamT *>(vals_dlm->dl_tensor.data), total);
      ec = self.get_store<ParamT, OptT>()->forward_lookup(
          ks, vals_span,
          {static_cast<const int64_t *>(lens_dlm->dl_tensor.data), n});
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

template <typename ParamT, typename OptT>
nb::tuple SyncOpsImpl<ParamT, OptT>::lookup_or_insert(
    EmbeddingStoreWrapper &self, nb::object keys, int64_t dim) {
  auto [dlm_keys, cap_keys] = se::py::extract_dlpack_tensor(keys);
  size_t n = get_dlm_size(dlm_keys);
  DLManagedTensor *vals_dlm = nullptr;
  DLManagedTensor *lens_dlm = nullptr;
  std::error_code ec;

  {
    nb::gil_scoped_release release;
    dispatch_on_key(dlm_keys, self.key_type_enum_, [&](auto ks) {
      size_t total = static_cast<size_t>(dim) * n;
      auto row_lengths_buf = std::make_unique<int64_t[]>(n);
      fill_uniform_row_lengths({row_lengths_buf.get(), n}, dim);

      lens_dlm =
          se::py::wrap_unique_ptr_as_tensor(std::move(row_lengths_buf), n);
      auto *lens_out = static_cast<int64_t *>(lens_dlm->dl_tensor.data);

      int64_t vals_shape[] = {static_cast<int64_t>(total)};
      vals_dlm = se::py::create_output_tensor<ParamT>(vals_shape);
      auto vals_span = std::span<ParamT>(
          static_cast<ParamT *>(vals_dlm->dl_tensor.data), total);
      ec = self.get_store<ParamT, OptT>()->lookup_or_insert(
          ks, vals_span, {lens_out, n});
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

template <typename ParamT, typename OptT>
void SyncOpsImpl<ParamT, OptT>::assign(EmbeddingStoreWrapper &self,
                                       nb::object keys, nb::object values) {
  auto [dlm_keys, cap_keys] = se::py::extract_dlpack_tensor(keys);
  auto [dlm_values, cap_values] = se::py::extract_dlpack_tensor(values);
  size_t n = get_dlm_size(dlm_keys);
  std::error_code ec;
  {
    nb::gil_scoped_release release;
    dispatch_on_key(dlm_keys, self.key_type_enum_, [&](auto ks) {
      auto val_span = se::py::dlpack_to_span<const ParamT>(dlm_values);
      int64_t dim = 0;
      if (dlm_values->dl_tensor.ndim == 2) {
        if (dlm_values->dl_tensor.shape[0] != static_cast<int64_t>(n)) {
          throw std::invalid_argument(
              "values shape[0] must match keys length");
        }
        dim = dlm_values->dl_tensor.shape[1];
      } else if (dlm_values->dl_tensor.ndim == 1) {
        if (n > 0) {
          if (dlm_values->dl_tensor.shape[0] % n != 0) {
            throw std::invalid_argument(
                "1D values length must be a multiple of keys length");
          }
          dim = dlm_values->dl_tensor.shape[0] / n;
        }
      } else {
        throw std::invalid_argument("values must be a 1D or 2D tensor");
      }
      auto& tls_row_lengths = get_tls_row_lengths();
      tls_row_lengths.resize(n);
      fill_uniform_row_lengths(tls_row_lengths, dim);
      ec = self.get_store<ParamT, OptT>()->assign(ks, val_span,
                                                   tls_row_lengths);
    });
  }
  check_error(ec);
}

template <typename ParamT, typename OptT>
nb::object SyncOpsImpl<ParamT, OptT>::contains(EmbeddingStoreWrapper &self,
                                               nb::object keys) {
  auto [dlm_keys, cap_keys] = se::py::extract_dlpack_tensor(keys);
  size_t n = get_dlm_size(dlm_keys);
  int64_t shape[1] = {static_cast<int64_t>(n)};
  DLManagedTensor *dlm =
      se::py::create_output_tensor<uint8_t>(std::span(shape));
  auto *out = static_cast<uint8_t *>(dlm->dl_tensor.data);

  std::error_code ec;
  {
    nb::gil_scoped_release release;
    dispatch_on_key(dlm_keys, self.key_type_enum_, [&](auto ks) {
      ec = self.get_store<ParamT, OptT>()->contains(
          ks, std::span<uint8_t>(out, n));
    });
  }

  check_error(ec);
  return se::py::wrap_output_tensor(dlm);
}

template <typename ParamT, typename OptT>
void SyncOpsImpl<ParamT, OptT>::accumulate_gradients(
    EmbeddingStoreWrapper &self, nb::object keys, nb::object grads) {
  auto [dlm_keys, cap_keys] = se::py::extract_dlpack_tensor(keys);
  auto [dlm_grads, cap_grads] = se::py::extract_dlpack_tensor(grads);
  size_t n = get_dlm_size(dlm_keys);
  int64_t dim = 0;
  if (dlm_grads->dl_tensor.ndim == 2) {
    if (dlm_grads->dl_tensor.shape[0] != static_cast<int64_t>(n)) {
      throw std::invalid_argument("grads shape[0] must match keys length");
    }
    dim = dlm_grads->dl_tensor.shape[1];
  } else if (dlm_grads->dl_tensor.ndim == 1) {
    if (n > 0) {
      if (dlm_grads->dl_tensor.shape[0] % n != 0) {
        throw std::invalid_argument(
            "1D grads length must be a multiple of keys length");
      }
      dim = dlm_grads->dl_tensor.shape[0] / n;
    }
  } else {
    throw std::invalid_argument("grads must be a 1D or 2D tensor");
  }

  std::error_code ec;
  {
    nb::gil_scoped_release release;
    dispatch_value_dtype(self.gradient_dtype_code_, [&]<typename GradT>() {
      auto grads_span = se::py::dlpack_to_span<const GradT>(dlm_grads);
      dispatch_on_key(dlm_keys, self.key_type_enum_, [&](auto ks) {
        auto& tls_row_lengths = get_tls_row_lengths();
        tls_row_lengths.resize(n);
        fill_uniform_row_lengths(tls_row_lengths, dim);
        ec = self.get_store<ParamT, OptT>()->accumulate_gradients(
            ks, grads_span, tls_row_lengths);
      });
    });
  }
  check_error(ec);
}

template <typename ParamT, typename OptT>
void SyncOpsImpl<ParamT, OptT>::optimizer_step(EmbeddingStoreWrapper &self) {
  nb::gil_scoped_release release;
  std::error_code ec;
  ec = self.get_store<ParamT, OptT>()->optimizer_step();
  check_error(ec);
}

template <typename ParamT, typename OptT>
void SyncOpsImpl<ParamT, OptT>::erase(EmbeddingStoreWrapper &self,
                                      nb::object keys) {
  auto [dlm_keys, cap_keys] = se::py::extract_dlpack_tensor(keys);
  std::error_code ec;
  {
    nb::gil_scoped_release release;
    dispatch_on_key(dlm_keys, self.key_type_enum_, [&](auto ks) {
      ec = self.get_store<ParamT, OptT>()->erase(ks);
    });
  }
  check_error(ec);
}

template <typename ParamT, typename OptT>
void SyncOpsImpl<ParamT, OptT>::clear(EmbeddingStoreWrapper &self) {
  nb::gil_scoped_release release;
  self.get_store<ParamT, OptT>()->clear();
}

template <typename ParamT, typename OptT>
void SyncOpsImpl<ParamT, OptT>::assign_scores(EmbeddingStoreWrapper &self,
                                              nb::object keys,
                                              nb::object scores) {
  auto [dlm_keys, cap_keys] = se::py::extract_dlpack_tensor(keys);
  auto [dlm_scores, cap_scores] = se::py::extract_dlpack_tensor(scores);
  std::error_code ec;
  auto scores_span = se::py::dlpack_to_span<const uint64_t>(dlm_scores);
  {
    nb::gil_scoped_release release;
    dispatch_on_key(dlm_keys, self.key_type_enum_, [&](auto ks) {
      ec = self.get_store<ParamT, OptT>()->assign_scores(ks, scores_span);
    });
  }
  check_error(ec);
}

} // namespace se::py::detail
