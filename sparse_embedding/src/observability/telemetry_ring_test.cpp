#include "sparse_embedding/src/observability/telemetry_ring.hpp"

#include <atomic>
#include <chrono>
#include <stdexcept>
#include <thread>

namespace {

std::atomic<uint64_t> consumed_count{0};
std::atomic<uint64_t> consumed_key{0};

void record_diagnostic(const se::DiagnosticEvent &event) noexcept {
  consumed_key.store(event.key, std::memory_order_relaxed);
  consumed_count.fetch_add(1, std::memory_order_release);
}

void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

} // namespace

int main() {
  consumed_count.store(0, std::memory_order_relaxed);
  consumed_key.store(0, std::memory_order_relaxed);
  {
    se::TelemetryRing ring(8, std::nullopt, &record_diagnostic);
    ring.start();
    require(ring.try_publish_diagnostic(
                {se::DiagnosticKind::kOptimizerLocatorRevalidation, 42}),
            "diagnostic event was not published");
    for (size_t attempt = 0;
         attempt < 1000 && consumed_count.load(std::memory_order_acquire) == 0;
         ++attempt)
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    require(consumed_count.load(std::memory_order_acquire) == 1,
            "diagnostic event was not consumed asynchronously");
    require(consumed_key.load(std::memory_order_relaxed) == 42,
            "diagnostic event key was corrupted");
    ring.stop();
  }

  {
    se::TelemetryRing draining_ring(8, std::nullopt, &record_diagnostic);
    draining_ring.start();
    require(draining_ring.try_publish_diagnostic(
                {se::DiagnosticKind::kOptimizerLocatorRevalidation, 84}),
            "drain test diagnostic event was not published");
    draining_ring.stop();
    require(consumed_count.load(std::memory_order_acquire) == 2,
            "ring stop did not drain a published diagnostic event");
    require(consumed_key.load(std::memory_order_relaxed) == 84,
            "drained diagnostic event key was corrupted");
  }

  se::TelemetryRing full_ring(2, std::nullopt, &record_diagnostic);
  require(full_ring.try_publish_diagnostic(
              {se::DiagnosticKind::kOptimizerLocatorRevalidation, 1}),
          "first bounded diagnostic event was rejected");
  require(full_ring.try_publish_diagnostic(
              {se::DiagnosticKind::kOptimizerLocatorRevalidation, 2}),
          "second bounded diagnostic event was rejected");
  require(!full_ring.try_publish_diagnostic(
              {se::DiagnosticKind::kOptimizerLocatorRevalidation, 3}),
          "full diagnostic ring blocked instead of dropping");
  require(full_ring.dropped_diagnostic_events() == 1,
          "diagnostic drop counter was not incremented");
}
