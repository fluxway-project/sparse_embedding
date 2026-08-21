#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
#include <stdexcept>
#include <thread>

#include "sparse_embedding/src/execution/topology/thread_affinity.hpp"
#include "sparse_embedding/src/observability/diagnostic_event.hpp"

namespace se {

class TelemetryRing {
public:
  static constexpr size_t kMaxSamples = 16;

  enum class EventKind : uint8_t {
    kPageTelemetry,
    kDiagnostic,
  };

  struct Event {
    EventKind kind = EventKind::kPageTelemetry;
    void *target = nullptr;
    void (*consume)(void *, const Event &) noexcept = nullptr;
    std::array<uint32_t, kMaxSamples> page_ids{};
    std::array<uint64_t, kMaxSamples> keys{};
    uint32_t page_count = 0;
    uint32_t key_count = 0;
    uint32_t stripe = 0;
    uint32_t weight = 1;
    DiagnosticEvent diagnostic;
  };

  explicit TelemetryRing(size_t capacity, std::optional<int> cpu = std::nullopt,
                         DiagnosticSink diagnostic_sink = nullptr)
      : capacity_(checked_capacity(capacity)),
        slots_(std::make_unique<Slot[]>(capacity_)), cpu_(cpu),
        diagnostic_sink_(diagnostic_sink) {
    for (size_t i = 0; i < capacity_; ++i)
      slots_[i].sequence.store(i, std::memory_order_relaxed);
  }

  ~TelemetryRing() { stop(); }

  TelemetryRing(const TelemetryRing &) = delete;
  TelemetryRing &operator=(const TelemetryRing &) = delete;

  void start() {
    bool expected = false;
    if (!running_.compare_exchange_strong(expected, true,
                                          std::memory_order_acq_rel))
      throw std::logic_error("telemetry ring was already started");
    consumer_ = std::thread([this] { consume_loop(); });
  }

  void stop() noexcept {
    if (!running_.exchange(false, std::memory_order_acq_rel))
      return;
    wake_sequence_.fetch_add(1, std::memory_order_release);
    wake_sequence_.notify_all();
    if (consumer_.joinable())
      consumer_.join();
  }

  bool try_publish(const Event &event) noexcept {
    if (event.kind == EventKind::kPageTelemetry) {
      if (event.target == nullptr || event.consume == nullptr ||
          event.page_count > kMaxSamples || event.key_count > kMaxSamples ||
          event.weight == 0)
        std::terminate();
    } else if (event.kind != EventKind::kDiagnostic) {
      std::terminate();
    } else if (diagnostic_sink_ == nullptr) {
      std::terminate();
    }

    size_t position = enqueue_position_.load(std::memory_order_relaxed);
    while (true) {
      Slot &slot = slots_[position % capacity_];
      const size_t sequence = slot.sequence.load(std::memory_order_acquire);
      const intptr_t difference =
          static_cast<intptr_t>(sequence) - static_cast<intptr_t>(position);
      if (difference == 0) {
        if (enqueue_position_.compare_exchange_weak(
                position, position + 1, std::memory_order_relaxed,
                std::memory_order_relaxed)) {
          slot.event = event;
          slot.sequence.store(position + 1, std::memory_order_release);
          wake_sequence_.fetch_add(1, std::memory_order_release);
          wake_sequence_.notify_one();
          return true;
        }
        continue;
      }
      if (difference < 0) {
        if (event.kind == EventKind::kDiagnostic) {
          dropped_diagnostic_events_.fetch_add(1, std::memory_order_relaxed);
        } else {
          dropped_batches_.fetch_add(1, std::memory_order_relaxed);
        }
        return false;
      }
      position = enqueue_position_.load(std::memory_order_relaxed);
    }
  }

  uint64_t dropped_batches() const noexcept {
    return dropped_batches_.load(std::memory_order_relaxed);
  }

  uint64_t dropped_diagnostic_events() const noexcept {
    return dropped_diagnostic_events_.load(std::memory_order_relaxed);
  }

  bool try_publish_diagnostic(DiagnosticEvent event) noexcept {
    if (diagnostic_sink_ == nullptr)
      std::terminate();
    Event envelope;
    envelope.kind = EventKind::kDiagnostic;
    envelope.diagnostic = event;
    return try_publish(envelope);
  }

private:
  struct alignas(64) Slot {
    std::atomic<size_t> sequence{0};
    Event event;
  };

  static size_t checked_capacity(size_t capacity) {
    if (capacity < 2 || (capacity & (capacity - 1)) != 0)
      throw std::invalid_argument(
          "telemetry ring capacity must be a power of two >= 2");
    return capacity;
  }

  bool try_consume(Event &event) noexcept {
    size_t position = dequeue_position_.load(std::memory_order_relaxed);
    while (true) {
      Slot &slot = slots_[position % capacity_];
      const size_t sequence = slot.sequence.load(std::memory_order_acquire);
      const intptr_t difference =
          static_cast<intptr_t>(sequence) - static_cast<intptr_t>(position + 1);
      if (difference == 0) {
        if (dequeue_position_.compare_exchange_weak(
                position, position + 1, std::memory_order_relaxed,
                std::memory_order_relaxed)) {
          event = slot.event;
          slot.sequence.store(position + capacity_, std::memory_order_release);
          return true;
        }
        continue;
      }
      if (difference < 0)
        return false;
      position = dequeue_position_.load(std::memory_order_relaxed);
    }
  }

  void consume_loop() noexcept {
    if (cpu_.has_value())
      ThreadAffinity::bind_current_thread_to_cpu(*cpu_);
    Event event;
    while (running_.load(std::memory_order_acquire) ||
           dequeue_position_.load(std::memory_order_relaxed) !=
               enqueue_position_.load(std::memory_order_acquire)) {
      if (try_consume(event)) {
        if (event.kind == EventKind::kDiagnostic) {
          if (diagnostic_sink_ == nullptr)
            std::terminate();
          diagnostic_sink_(event.diagnostic);
        } else {
          event.consume(event.target, event);
        }
        continue;
      }
      const uint64_t observed = wake_sequence_.load(std::memory_order_acquire);
      if (dequeue_position_.load(std::memory_order_relaxed) ==
              enqueue_position_.load(std::memory_order_acquire) &&
          running_.load(std::memory_order_acquire))
        wake_sequence_.wait(observed, std::memory_order_acquire);
    }
  }

  const size_t capacity_;
  std::unique_ptr<Slot[]> slots_;
  std::atomic<size_t> enqueue_position_{0};
  std::atomic<size_t> dequeue_position_{0};
  std::atomic<uint64_t> dropped_batches_{0};
  std::atomic<uint64_t> dropped_diagnostic_events_{0};
  std::atomic<uint64_t> wake_sequence_{0};
  std::atomic<bool> running_{false};
  std::thread consumer_;
  std::optional<int> cpu_;
  DiagnosticSink diagnostic_sink_ = nullptr;
};

} // namespace se
