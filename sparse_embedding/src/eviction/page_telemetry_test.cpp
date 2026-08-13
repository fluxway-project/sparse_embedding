#include "sparse_embedding/src/eviction/page_telemetry.hpp"

#include <array>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

template <typename Function> void require_throws(Function &&function) {
  try {
    function();
  } catch (const std::exception &) {
    return;
  }
  throw std::runtime_error("expected operation to throw");
}

} // namespace

int main() {
  se::PageResidencyTelemetry telemetry(4);
  telemetry.register_resident(0);
  telemetry.advance_residency_epoch();
  telemetry.advance_residency_epoch();
  telemetry.register_resident(1);
  require(telemetry.residency_age(0) == 2.0F &&
              telemetry.residency_age(1) == 0.0F,
          "page telemetry did not distinguish residency age");
  telemetry.advance_residency_epoch();
  require(telemetry.residency_age(0) == 3.0F &&
              telemetry.residency_age(1) == 1.0F,
          "page telemetry residency epoch did not advance");
  telemetry.mark_nonresident(0);
  require_throws([&] { telemetry.residency_age(0); });
  telemetry.register_resident(0);
  require(telemetry.residency_age(0) == 0.0F,
          "page telemetry did not restart residency after page reuse");
  se::PageAccessTelemetry access(4, 64);
  constexpr std::array<uint64_t, 4> keys{10, 20, 30, 40};
  constexpr std::array<uint32_t, 4> pages{0, 1, 0, 0};
  for (size_t iteration = 0; iteration < 8; ++iteration)
    access.observe_batch(keys, pages);
  require(access.hits(0) == 24 && access.hits(1) == 8 && access.hits(2) == 0,
          "page telemetry did not preserve per-page heat");
  access.observe(3, 8);
  require(access.hits(3) == 8,
          "weighted page telemetry did not preserve page heat");
  require(access.coaccess_density(keys) >= 8.0F,
          "page telemetry did not record bounded co-access");
  access.decay(1);
  require(access.hits(0) == 12 && access.hits(1) == 4,
          "page telemetry heat decay is incorrect");
  require_throws([&] { access.hits(4); });
  require_throws([&] { telemetry.residency_age(2); });
  require_throws([&] { access.decay(0); });
  telemetry.reset();
  access.reset();
  require(access.hits(0) == 0, "page telemetry reset did not clear heat");
  require_throws([&] { telemetry.residency_age(0); });

  se::StripedPageAccessTelemetry striped(4, 64, 2);
  striped.observe(0, 0);
  striped.observe(1, 0);
  striped.observe(1, 1);
  striped.observe_coaccess_batch(0, std::array<uint64_t, 2>{10, 20});
  striped.observe_coaccess_batch(1, std::array<uint64_t, 2>{10, 20});
  require(striped.hits(0) == 2 && striped.hits(1) == 1,
          "striped telemetry did not merge page heat");
  require(striped.coaccess_estimate(10, 20) == 2,
          "striped telemetry did not merge coaccess");
  striped.decay(1);
  require(striped.hits(0) == 0 && striped.coaccess_estimate(10, 20) == 0,
          "striped telemetry decay is incorrect");

  constexpr size_t worker_count = 4;
  constexpr size_t iterations = 4096;
  se::StripedPageAccessTelemetry concurrent(4, 64, worker_count);
  std::vector<std::thread> workers;
  workers.reserve(worker_count);
  for (size_t worker = 0; worker < worker_count; ++worker) {
    workers.emplace_back([&concurrent, worker] {
      constexpr std::array<uint64_t, 2> batch{100, 200};
      for (size_t iteration = 0; iteration < iterations; ++iteration) {
        concurrent.observe(worker, 2);
        concurrent.observe_coaccess_batch(worker, batch);
      }
    });
  }
  for (auto &worker : workers)
    worker.join();
  require(concurrent.hits(2) == worker_count * iterations,
          "striped telemetry lost concurrent page heat");
  require(concurrent.coaccess_estimate(100, 200) >= worker_count * iterations,
          "striped telemetry lost concurrent coaccess");
}
