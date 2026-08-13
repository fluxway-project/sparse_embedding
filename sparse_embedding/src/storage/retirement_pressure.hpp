#pragma once

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <immintrin.h>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>

namespace se {

enum class RetirementPressureState : uint8_t {
  kNormal,
  kBackpressure,
  kCircuitOpen,
};

struct RetirementPressureConfig {
  double backpressure_fraction = 0.15;
  double circuit_fraction = 0.25;
  std::chrono::milliseconds epoch_stall_timeout = std::chrono::seconds(30);
};

struct RetirementPressureStats {
  size_t retired_bytes = 0;
  RetirementPressureState state = RetirementPressureState::kNormal;
  uint64_t backpressure_count = 0;
  uint64_t oldest_unreclaimed_age_ns = 0;
  uint64_t epoch_stall_count = 0;
};

class RetirementPressure {
public:
  class ControlPlaneCall {
  public:
    ~ControlPlaneCall() noexcept {
      if (!admitted_)
        return;
      if (active_control_plane_call_ != this)
        std::terminate();
      active_control_plane_call_ = previous_;
    }

    ControlPlaneCall(const ControlPlaneCall &) = delete;
    ControlPlaneCall &operator=(const ControlPlaneCall &) = delete;
    ControlPlaneCall(ControlPlaneCall &&) = delete;
    ControlPlaneCall &operator=(ControlPlaneCall &&) = delete;

    bool admitted() const noexcept { return admitted_; }

    void require_admitted() const {
      if (!admitted_)
        throw std::runtime_error(
            "retirement pressure circuit is open for control-plane mutation");
    }

  private:
    friend class RetirementPressure;

    ControlPlaneCall(RetirementPressure &pressure, bool reject_if_open)
        : pressure_(&pressure) {
      std::lock_guard lock(pressure.mutex_);
      pressure.refresh_stall_state_locked(std::chrono::steady_clock::now());
      if (pressure.state_ == RetirementPressureState::kCircuitOpen &&
          !pressure.has_active_admission()) {
        if (reject_if_open) {
          throw std::runtime_error(
              "retirement pressure circuit is open for control-plane mutation");
        }
        return;
      }
      previous_ = active_control_plane_call_;
      active_control_plane_call_ = this;
      admitted_ = true;
    }

    RetirementPressure *pressure_ = nullptr;
    ControlPlaneCall *previous_ = nullptr;
    bool admitted_ = false;
  };

  class WriterCall {
  public:
    explicit WriterCall(RetirementPressure &pressure) noexcept
        : pressure_(&pressure) {}

    bool apply() {
      if (applied_)
        return false;
      applied_ = true;
      return pressure_->apply_writer_backpressure();
    }

  private:
    RetirementPressure *pressure_;
    bool applied_ = false;
  };

  explicit RetirementPressure(size_t effective_memory_limit_bytes,
                              RetirementPressureConfig config = {})
      : effective_memory_limit_bytes_(effective_memory_limit_bytes),
        config_(config),
        backpressure_bytes_(threshold_bytes(config.backpressure_fraction)),
        circuit_bytes_(threshold_bytes(config.circuit_fraction)) {
    validate_config();
  }

  uint64_t begin_retirement(size_t bytes) {
    if (bytes == 0)
      throw std::invalid_argument("retirement byte count must be nonzero");
    std::lock_guard lock(mutex_);
    refresh_stall_state_locked(std::chrono::steady_clock::now());
    if (state_ == RetirementPressureState::kCircuitOpen &&
        !has_active_admission()) {
      throw std::runtime_error(
          "retirement pressure circuit is open for control-plane mutation");
    }
    if (bytes > std::numeric_limits<size_t>::max() - retired_bytes_)
      throw std::overflow_error("retirement byte accounting overflow");
    const uint64_t id = next_retirement_id_++;
    records_.emplace(id, Record{bytes, std::chrono::steady_clock::now()});
    retired_bytes_ += bytes;
    update_state_locked();
    return id;
  }

  void complete_retirement(uint64_t id) {
    std::lock_guard lock(mutex_);
    auto it = records_.find(id);
    if (it == records_.end())
      throw std::logic_error("retirement completion token is unknown");
    retired_bytes_ -= it->second.bytes;
    records_.erase(it);
    update_state_locked();
    refresh_stall_state_locked(std::chrono::steady_clock::now());
  }

  void complete_retirement_noexcept(uint64_t id) noexcept {
    try {
      complete_retirement(id);
    } catch (...) {
      std::terminate();
    }
  }

  WriterCall writer_call() noexcept { return WriterCall(*this); }

  ControlPlaneCall control_plane_call() {
    return ControlPlaneCall(*this, true);
  }

  ControlPlaneCall try_control_plane_call() {
    return ControlPlaneCall(*this, false);
  }

  bool circuit_open() const noexcept {
    std::lock_guard lock(mutex_);
    return state_ == RetirementPressureState::kCircuitOpen;
  }

  void refresh_stall_state() noexcept {
    std::lock_guard lock(mutex_);
    refresh_stall_state_locked(std::chrono::steady_clock::now());
  }

  RetirementPressureStats stats() const noexcept {
    std::lock_guard lock(mutex_);
    const auto now = std::chrono::steady_clock::now();
    RetirementPressureStats result;
    result.retired_bytes = retired_bytes_;
    result.state = state_;
    result.backpressure_count = backpressure_count_;
    result.epoch_stall_count = epoch_stall_count_;
    if (!records_.empty()) {
      result.oldest_unreclaimed_age_ns = static_cast<uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              now - records_.begin()->second.retired_at)
              .count());
    }
    return result;
  }

private:
  bool has_active_admission() const noexcept {
    for (const ControlPlaneCall *call = active_control_plane_call_;
         call != nullptr; call = call->previous_) {
      if (call->pressure_ == this)
        return true;
    }
    return false;
  }

  struct Record {
    size_t bytes;
    std::chrono::steady_clock::time_point retired_at;
  };

  size_t threshold_bytes(double fraction) const noexcept {
    if (effective_memory_limit_bytes_ == 0)
      return 0;
    return static_cast<size_t>(std::ceil(
        static_cast<long double>(effective_memory_limit_bytes_) * fraction));
  }

  void validate_config() const {
    if (!std::isfinite(config_.backpressure_fraction) ||
        !std::isfinite(config_.circuit_fraction) ||
        config_.backpressure_fraction <= 0.0 ||
        config_.circuit_fraction >= 1.0 ||
        config_.backpressure_fraction >= config_.circuit_fraction) {
      throw std::invalid_argument(
          "retirement pressure fractions must satisfy 0 < backpressure < "
          "circuit < 1");
    }
    if (config_.epoch_stall_timeout <= std::chrono::milliseconds::zero()) {
      throw std::invalid_argument(
          "retirement pressure epoch stall timeout must be positive");
    }
  }

  bool apply_writer_backpressure() {
    bool should_yield = false;
    {
      std::lock_guard lock(mutex_);
      refresh_stall_state_locked(std::chrono::steady_clock::now());
      if (state_ == RetirementPressureState::kBackpressure) {
        ++backpressure_count_;
        should_yield = true;
      }
    }
    if (should_yield)
      _mm_pause();
    return should_yield;
  }

  void update_state_locked() noexcept {
    if (state_ == RetirementPressureState::kCircuitOpen) {
      if (retired_bytes_ < backpressure_bytes_)
        state_ = RetirementPressureState::kNormal;
      return;
    }
    if (retired_bytes_ >= circuit_bytes_ && circuit_bytes_ != 0) {
      state_ = RetirementPressureState::kCircuitOpen;
    } else if (retired_bytes_ >= backpressure_bytes_ &&
               backpressure_bytes_ != 0) {
      state_ = RetirementPressureState::kBackpressure;
    } else {
      state_ = RetirementPressureState::kNormal;
    }
  }

  void refresh_stall_state_locked(
      std::chrono::steady_clock::time_point now) noexcept {
    if (records_.empty()) {
      stall_active_ = false;
      return;
    }
    const auto age = now - records_.begin()->second.retired_at;
    if (age < config_.epoch_stall_timeout) {
      stall_active_ = false;
      return;
    }
    if (stall_active_)
      return;
    stall_active_ = true;
    ++epoch_stall_count_;
    std::cerr << std::format(
        "{{\"severity\":\"CRITICAL\",\"event\":\"epoch_stall\","
        "\"retired_bytes\":{},\"oldest_unreclaimed_age_ns\":{}}}\n",
        retired_bytes_,
        std::chrono::duration_cast<std::chrono::nanoseconds>(age).count());
  }

  size_t effective_memory_limit_bytes_;
  RetirementPressureConfig config_;
  size_t backpressure_bytes_;
  size_t circuit_bytes_;
  mutable std::mutex mutex_;
  std::map<uint64_t, Record> records_;
  size_t retired_bytes_ = 0;
  uint64_t next_retirement_id_ = 1;
  uint64_t backpressure_count_ = 0;
  uint64_t epoch_stall_count_ = 0;
  RetirementPressureState state_ = RetirementPressureState::kNormal;
  bool stall_active_ = false;
  inline static thread_local ControlPlaneCall *active_control_plane_call_ =
      nullptr;
};

} // namespace se
