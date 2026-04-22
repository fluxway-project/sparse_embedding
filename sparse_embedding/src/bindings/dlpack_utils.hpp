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

#ifndef SPARSE_EMBEDDING_DLPACK_UTILS_HPP_
#define SPARSE_EMBEDDING_DLPACK_UTILS_HPP_

#include <dlpack/dlpack.h>
#include <nanobind/nanobind.h>

#include <atomic>
#include <cstring>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "sparse_embedding/src/common/numeric_types.hpp"

namespace se::py {

namespace nb = nanobind;

// =============================================================================
// Compile-time DLDataType mapping
// =============================================================================

template <typename T> constexpr DLDataType to_dl_dtype();
template <> constexpr DLDataType to_dl_dtype<float>() {
  return {kDLFloat, 32, 1};
}
template <> constexpr DLDataType to_dl_dtype<int64_t>() {
  return {kDLInt, 64, 1};
}
template <> constexpr DLDataType to_dl_dtype<uint64_t>() {
  return {kDLUInt, 64, 1};
}
template <> constexpr DLDataType to_dl_dtype<int32_t>() {
  return {kDLInt, 32, 1};
}
template <> constexpr DLDataType to_dl_dtype<uint8_t>() {
  return {kDLUInt, 8, 1};
}
template <> constexpr DLDataType to_dl_dtype<bool>() { return {kDLBool, 8, 1}; }
template <> constexpr DLDataType to_dl_dtype<se::float16_t>() {
  return {kDLFloat, 16, 1};
}
template <> constexpr DLDataType to_dl_dtype<se::bfloat16_t>() {
  return {kDLBfloat, 16, 1};
}

// =============================================================================
// ExtractDlpackTensor: call __dlpack__() on a Python object, get raw pointer
// =============================================================================

inline std::pair<DLManagedTensor *, nb::object>
extract_dlpack_tensor(const nb::object &py_tensor) {
  if (!nb::hasattr(py_tensor, "__dlpack__")) {
    throw std::invalid_argument("object has no __dlpack__");
  }
  nb::object capsule = py_tensor.attr("__dlpack__")();
  if (!PyCapsule_CheckExact(capsule.ptr())) {
    throw std::invalid_argument("__dlpack__ must return a PyCapsule");
  }
  void *raw = PyCapsule_GetPointer(capsule.ptr(), "dltensor");
  if (!raw) {
    PyErr_Clear();
    throw std::invalid_argument("invalid or consumed dltensor capsule");
  }
  return {static_cast<DLManagedTensor *>(raw), std::move(capsule)};
}

// =============================================================================
// DlpackToSpan: zero-copy view into a DLTensor as std::span<T>
// =============================================================================

template <typename T> std::span<T> dlpack_to_span(const DLManagedTensor *dlm) {
  const DLTensor &t = dlm->dl_tensor;

  if (t.device.device_type != kDLCPU) {
    throw std::invalid_argument("only CPU tensors are supported");
  }

  DLDataType expected = to_dl_dtype<std::remove_const_t<T>>();
  if (t.dtype.code != expected.code || t.dtype.bits != expected.bits) {
    throw std::invalid_argument("dtype mismatch in dlpack_to_span");
  }

  if (t.strides != nullptr) {
    int64_t stride = 1;
    for (int i = t.ndim - 1; i >= 0; --i) {
      if (t.strides[i] != stride) {
        throw std::invalid_argument("tensor must be C-contiguous");
      }
      stride *= t.shape[i];
    }
  }

  size_t total = 1;
  for (int i = 0; i < t.ndim; ++i)
    total *= static_cast<size_t>(t.shape[i]);

  auto *base =
      reinterpret_cast<T *>(static_cast<char *>(t.data) + t.byte_offset);
  return std::span<T>(base, total);
}

// =============================================================================
// CreateOutputTensor: allocate a DLManagedTensor owning its data buffer
// =============================================================================

template <typename T> struct OutputTensorCtx {
  std::unique_ptr<T[]> data;
  std::vector<int64_t> shape;
};

template <typename T>
DLManagedTensor *create_output_tensor(std::span<const int64_t> shape) {
  size_t total = 1;
  for (auto s : shape)
    total *= static_cast<size_t>(s);

  auto *ctx = new OutputTensorCtx<T>();
  ctx->data = std::make_unique<T[]>(total);
  ctx->shape.assign(shape.begin(), shape.end());

  auto *dlm = new DLManagedTensor;
  dlm->dl_tensor.data = ctx->data.get();
  dlm->dl_tensor.device = {kDLCPU, 0};
  dlm->dl_tensor.ndim = static_cast<int>(shape.size());
  dlm->dl_tensor.dtype = to_dl_dtype<T>();
  dlm->dl_tensor.shape = ctx->shape.data();
  dlm->dl_tensor.strides = nullptr;
  dlm->dl_tensor.byte_offset = 0;
  dlm->manager_ctx = ctx;
  dlm->deleter = [](DLManagedTensor *self) {
    delete static_cast<OutputTensorCtx<T> *>(self->manager_ctx);
    delete self;
  };

  return dlm;
}

// =============================================================================
// Safe capsule destructor for unconsumed PyCapsules
// =============================================================================

inline void safe_capsule_destructor(PyObject *obj) {
  auto *dlm =
      static_cast<DLManagedTensor *>(PyCapsule_GetPointer(obj, "dltensor"));
  if (dlm == nullptr) {
    PyErr_Clear();
    return;
  }
  if (dlm->deleter) {
    dlm->deleter(dlm);
  }
}

// =============================================================================
// DLPackCapsuleWrapper: __dlpack__ / __dlpack_device__ for Python consumers
// =============================================================================

class __attribute__((visibility("hidden"))) DLPackCapsuleWrapper {
public:
  DLPackCapsuleWrapper(nb::object capsule, DLDevice device)
      : capsule_(std::move(capsule)), device_(device), consumed_(false) {}

  DLPackCapsuleWrapper(DLPackCapsuleWrapper &&other) noexcept
      : capsule_(std::move(other.capsule_)), device_(other.device_),
        consumed_(other.consumed_.load(std::memory_order_relaxed)) {
    other.consumed_.store(true, std::memory_order_relaxed);
  }

  DLPackCapsuleWrapper &operator=(DLPackCapsuleWrapper &&other) noexcept {
    if (this != &other) {
      capsule_ = std::move(other.capsule_);
      device_ = other.device_;
      consumed_.store(other.consumed_.load(std::memory_order_relaxed),
                      std::memory_order_relaxed);
      other.consumed_.store(true, std::memory_order_relaxed);
    }
    return *this;
  }

  DLPackCapsuleWrapper(const DLPackCapsuleWrapper &) = delete;
  DLPackCapsuleWrapper &operator=(const DLPackCapsuleWrapper &) = delete;

  nb::object dlpack(nb::object stream = nb::none(),
                    nb::object max_version = nb::none(),
                    nb::object dl_device = nb::none(),
                    nb::object copy = nb::none()) {
    bool expected = false;
    if (!consumed_.compare_exchange_strong(expected, true,
                                           std::memory_order_acq_rel)) {
      throw std::runtime_error(
          "DLPack capsule has already been consumed. "
          "__dlpack__() can only be called once per capsule.");
    }

    if (device_.device_type == kDLCPU && !stream.is_none()) {
      throw std::invalid_argument("stream must be None for CPU device");
    }

    if (!dl_device.is_none()) {
      auto target = nb::cast<nb::tuple>(dl_device);
      int target_type = nb::cast<int>(target[0]);
      if (target_type != static_cast<int>(device_.device_type)) {
        if (!copy.is_none() && !nb::cast<bool>(copy)) {
          throw std::runtime_error(
              "Cross-device export requires copy=True or copy=None");
        }
        throw std::runtime_error("Cross-device DLPack export not supported");
      }
    }

    if (!copy.is_none() && nb::cast<bool>(copy)) {
      throw std::runtime_error("copy=True not supported; zero-copy only");
    }

    return std::move(capsule_);
  }

  nb::tuple dlpack_device() const {
    return nb::make_tuple(static_cast<int>(device_.device_type),
                          device_.device_id);
  }

private:
  nb::object capsule_;
  DLDevice device_;
  mutable std::atomic<bool> consumed_;
};

// =============================================================================
// Helpers: wrap a DLManagedTensor* into a DLPackCapsuleWrapper
// =============================================================================

inline nb::object wrap_output_tensor(DLManagedTensor *dlm) {
  DLDevice dev = dlm->dl_tensor.device;
  nb::object capsule = nb::steal<nb::object>(
      PyCapsule_New(dlm, "dltensor", safe_capsule_destructor));
  return nb::cast(DLPackCapsuleWrapper(std::move(capsule), dev));
}

} // namespace se::py

#endif // SPARSE_EMBEDDING_DLPACK_UTILS_HPP_
