/* Standalone excerpt of DLPackCapsuleWrapper from dlpack_helpers.hpp — no axon/rpc deps. */

#pragma once

#include <dlpack/dlpack.h>
#include <nanobind/nanobind.h>

#include <atomic>
#include <stdexcept>
#include <utility>

namespace emb_dlpack {

namespace nb = nanobind;

/// Wraps a raw ``dltensor`` PyCapsule with ``__dlpack__`` / ``__dlpack_device__`` for JAX/NumPy/PyTorch.
class DLPackCapsuleWrapper {
public:
    DLPackCapsuleWrapper(nb::object capsule, DLDevice device)
        : capsule_(std::move(capsule)), device_(device), consumed_(false) {}

    DLPackCapsuleWrapper(DLPackCapsuleWrapper&& other) noexcept
        : capsule_(std::move(other.capsule_)),
          device_(other.device_),
          consumed_(other.consumed_.load(std::memory_order_relaxed)) {
        other.consumed_.store(true, std::memory_order_relaxed);
    }

    DLPackCapsuleWrapper& operator=(DLPackCapsuleWrapper&& other) noexcept {
        if (this != &other) {
            capsule_ = std::move(other.capsule_);
            device_ = other.device_;
            consumed_.store(
                other.consumed_.load(std::memory_order_relaxed),
                std::memory_order_relaxed);
            other.consumed_.store(true, std::memory_order_relaxed);
        }
        return *this;
    }

    DLPackCapsuleWrapper(const DLPackCapsuleWrapper&) = delete;
    DLPackCapsuleWrapper& operator=(const DLPackCapsuleWrapper&) = delete;

    nb::object dlpack(
        nb::object stream = nb::none(), nb::object max_version = nb::none(),
        nb::object dl_device = nb::none(), nb::object copy = nb::none()) {
        bool expected = false;
        if (!consumed_.compare_exchange_strong(
                expected, true, std::memory_order_acq_rel)) {
            throw std::runtime_error(
                "DLPack capsule has already been consumed. "
                "__dlpack__() can only be called once per capsule.");
        }

        if (device_.device_type == kDLCPU && !stream.is_none()) {
            throw std::invalid_argument(
                "stream must be None for CPU device (kDLCPU)");
        }

        if (!dl_device.is_none()) {
            auto target = nb::cast<nb::tuple>(dl_device);
            int target_type = nb::cast<int>(target[0]);
            if (target_type != static_cast<int>(device_.device_type)) {
                if (!copy.is_none() && !nb::cast<bool>(copy)) {
                    throw std::runtime_error(
                        "Cross-device export requires copy=True or copy=None, "
                        "but copy=False was specified");
                }
                throw std::runtime_error(
                    "Cross-device DLPack export is not yet supported");
            }
        }

        if (!copy.is_none() && nb::cast<bool>(copy)) {
            throw std::runtime_error(
                "copy=True is not supported; this implementation is zero-copy only");
        }

        return std::move(capsule_);
    }

    nb::tuple dlpack_device() const {
        return nb::make_tuple(
            static_cast<int>(device_.device_type), device_.device_id);
    }

private:
    nb::object capsule_;
    DLDevice device_;
    mutable std::atomic<bool> consumed_;
};

}  // namespace emb_dlpack
