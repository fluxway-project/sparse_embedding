#pragma once

#include <utils/epoch.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "sparse_embedding/src/storage/retirement_pressure.hpp"

namespace se::concurrency {

inline thread_local size_t epoch_reclamation_scope_depth = 0;

class EpochRetirementNode {
public:
  virtual ~EpochRetirementNode() = default;

private:
  friend class EpochRetirementDomain;
  EpochRetirementNode *next = nullptr;
  long retired_epoch = 0;
};

class EpochRetirementDomain {
public:
  static constexpr long kReclamationEpochLag = 2;

  static void retire(EpochRetirementNode *node) noexcept {
    if (node == nullptr)
      std::terminate();
    node->retired_epoch = epoch::internal::get_epoch().get_current();
    push(node);
    if (epoch_reclamation_scope_depth == 0)
      request_progress();
  }

  static void request_progress() noexcept {
    if (pending_.load(std::memory_order_acquire) == nullptr)
      return;
    progress_requested_.store(true, std::memory_order_release);
    if (progressing_.test_and_set(std::memory_order_acquire))
      return;
    do {
      progress_requested_.store(false, std::memory_order_release);
      for (long attempt = 0; attempt < kReclamationEpochLag; ++attempt)
        epoch::internal::get_epoch().update_epoch();
      const long current_epoch = epoch::internal::get_epoch().get_current();
      EpochRetirementNode *nodes =
          pending_.exchange(nullptr, std::memory_order_acq_rel);
      EpochRetirementNode *deferred = nullptr;
      EpochRetirementNode *deferred_tail = nullptr;
      while (nodes != nullptr) {
        EpochRetirementNode *node = nodes;
        nodes = nodes->next;
        if (node->retired_epoch + kReclamationEpochLag <= current_epoch) {
          delete node;
        } else {
          node->next = deferred;
          deferred = node;
          if (deferred_tail == nullptr)
            deferred_tail = node;
        }
      }
      if (deferred != nullptr)
        push(deferred, deferred_tail);
    } while (progress_requested_.load(std::memory_order_acquire));
    progressing_.clear(std::memory_order_release);
    if (progress_requested_.load(std::memory_order_acquire))
      request_progress();
  }

private:
  static void push(EpochRetirementNode *head,
                   EpochRetirementNode *tail = nullptr) noexcept {
    if (tail == nullptr)
      tail = head;
    EpochRetirementNode *observed = pending_.load(std::memory_order_relaxed);
    do {
      tail->next = observed;
    } while (!pending_.compare_exchange_weak(
        observed, head, std::memory_order_release, std::memory_order_relaxed));
  }

  inline static std::atomic<EpochRetirementNode *> pending_{nullptr};
  inline static std::atomic<bool> progress_requested_{false};
  inline static std::atomic_flag progressing_ = ATOMIC_FLAG_INIT;
};

class EpochAnnouncement {
public:
  explicit EpochAnnouncement(bool progress_retirement = false)
      : id_(epoch::internal::get_epoch().announce()),
        progress_retirement_(progress_retirement) {
    if (progress_retirement_)
      ++epoch_reclamation_scope_depth;
  }
  ~EpochAnnouncement() {
    epoch::internal::get_epoch().unannounce(id_);
    if (!progress_retirement_)
      return;
    if (epoch_reclamation_scope_depth == 0)
      std::terminate();
    --epoch_reclamation_scope_depth;
    if (epoch_reclamation_scope_depth == 0)
      EpochRetirementDomain::request_progress();
  }
  EpochAnnouncement(const EpochAnnouncement &) = delete;
  EpochAnnouncement &operator=(const EpochAnnouncement &) = delete;

private:
  int id_;
  bool progress_retirement_;
};

template <typename Thunk> decltype(auto) with_epoch(Thunk &&thunk) {
  EpochAnnouncement announcement;
  if constexpr (std::is_void_v<std::invoke_result_t<Thunk>>) {
    std::invoke(std::forward<Thunk>(thunk));
  } else {
    return std::invoke(std::forward<Thunk>(thunk));
  }
}

template <typename Thunk> decltype(auto) with_epoch_reclamation(Thunk &&thunk) {
  EpochAnnouncement announcement(true);
  if constexpr (std::is_void_v<std::invoke_result_t<Thunk>>) {
    std::invoke(std::forward<Thunk>(thunk));
  } else {
    return std::invoke(std::forward<Thunk>(thunk));
  }
}

template <typename T> class EpochRetirement {
public:
  EpochRetirement(std::shared_ptr<RetirementPressure> pressure, size_t bytes)
      : pressure_(std::move(pressure)), deferred_(new Deferred()) {
    if (!pressure_) {
      delete deferred_;
      throw std::invalid_argument("epoch retirement pressure domain is null");
    }
    try {
      deferred_->retirement_id = pressure_->begin_retirement(bytes);
      deferred_->pressure = pressure_;
    } catch (...) {
      delete deferred_;
      throw;
    }
  }
  ~EpochRetirement() {
    if (deferred_ != nullptr)
      delete deferred_;
  }
  EpochRetirement(const EpochRetirement &) = delete;
  EpochRetirement &operator=(const EpochRetirement &) = delete;
  EpochRetirement(EpochRetirement &&other) noexcept
      : pressure_(std::move(other.pressure_)),
        deferred_(std::exchange(other.deferred_, nullptr)) {}
  EpochRetirement &operator=(EpochRetirement &&) = delete;
  template <typename... Args> T &emplace(Args &&...args) {
    if (deferred_ == nullptr)
      throw std::logic_error("epoch retirement is already published");
    return deferred_->value.emplace(std::forward<Args>(args)...);
  }
  void publish() {
    if (deferred_ == nullptr || !deferred_->value.has_value())
      throw std::logic_error("epoch retirement has no deferred value");
    EpochRetirementDomain::retire(std::exchange(deferred_, nullptr));
  }

private:
  struct Deferred final : EpochRetirementNode {
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
