#pragma once

#include <utils/epoch.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>

#include "sparse_embedding/src/storage/retirement_pressure.hpp"

namespace se::concurrency {

template <typename T> class EpochRetirement {
public:
  EpochRetirement(std::shared_ptr<RetirementPressure> pressure, size_t bytes)
      : pressure_(std::move(pressure)), deferred_(epoch::New<Deferred>()) {
    if (!pressure_) {
      epoch::Delete(deferred_);
      throw std::invalid_argument("epoch retirement pressure domain is null");
    }
    try {
      deferred_->retirement_id = pressure_->begin_retirement(bytes);
      deferred_->pressure = pressure_;
    } catch (...) {
      epoch::Delete(deferred_);
      throw;
    }
  }
  ~EpochRetirement() {
    if (deferred_ != nullptr)
      epoch::Delete(deferred_);
  }
  EpochRetirement(const EpochRetirement &) = delete;
  EpochRetirement &operator=(const EpochRetirement &) = delete;
  EpochRetirement(EpochRetirement &&other) noexcept
      : pressure_(std::move(other.pressure_)),
        deferred_(std::exchange(other.deferred_, nullptr)) {}
  EpochRetirement &operator=(EpochRetirement &&) = delete;
  template <typename... Args> void emplace(Args &&...args) {
    if (deferred_ == nullptr)
      throw std::logic_error("epoch retirement is already published");
    deferred_->value.emplace(std::forward<Args>(args)...);
  }
  void publish() {
    if (deferred_ == nullptr || !deferred_->value.has_value())
      throw std::logic_error("epoch retirement has no deferred value");
    epoch::Retire(std::exchange(deferred_, nullptr));
  }

private:
  struct Deferred {
    std::shared_ptr<RetirementPressure> pressure;
    uint64_t retirement_id = 0;
    std::optional<T> value;
    ~Deferred() noexcept {
      value.reset();
      if (pressure)
        pressure->complete_retirement_noexcept(retirement_id);
    }
  };
  std::shared_ptr<RetirementPressure> pressure_;
  Deferred *deferred_;
};

} // namespace se::concurrency
