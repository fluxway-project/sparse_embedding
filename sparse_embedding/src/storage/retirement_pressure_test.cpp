#include "sparse_embedding/src/storage/page_storage.hpp"
#include "sparse_embedding/src/storage/retirement_pressure.hpp"

#include <chrono>
#include <stdexcept>
#include <thread>

namespace {

void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

} // namespace

int main() {
  bool invalid_fraction_rejected = false;
  try {
    se::RetirementPressure invalid(
        1024, se::RetirementPressureConfig{.backpressure_fraction = 0.25,
                                           .circuit_fraction = 0.15});
  } catch (const std::invalid_argument &) {
    invalid_fraction_rejected = true;
  }
  require(invalid_fraction_rejected,
          "invalid retirement pressure fractions were accepted");

  se::StorageOptions invalid_auto_options;
  invalid_auto_options.memory_limit_bytes = 4096;
  invalid_auto_options.watermark =
      se::WatermarkConfig{.high = 0.25, .low = 0.1};
  invalid_auto_options.retirement_pressure.circuit_fraction = 0.25;
  bool invalid_auto_rejected = false;
  try {
    (void)se::create_page_storage(std::move(invalid_auto_options));
  } catch (const std::invalid_argument &) {
    invalid_auto_rejected = true;
  }
  require(invalid_auto_rejected,
          "Auto storage accepted circuit at spill high watermark");

  se::RetirementPressureConfig config;
  config.epoch_stall_timeout = std::chrono::milliseconds(1);
  se::RetirementPressure pressure(1000, config);
  const uint64_t first = pressure.begin_retirement(149);
  require(pressure.stats().state == se::RetirementPressureState::kNormal,
          "retirement below backpressure threshold was throttled");
  const uint64_t second = pressure.begin_retirement(1);
  auto writer = pressure.writer_call();
  require(writer.apply(), "backpressured writer did not yield");
  require(!writer.apply(), "writer yielded more than once per invocation");

  uint64_t third = 0;
  uint64_t fourth = 0;
  {
    auto admission = pressure.control_plane_call();
    third = pressure.begin_retirement(100);
    require(pressure.stats().state == se::RetirementPressureState::kCircuitOpen,
            "retirement circuit did not open");
    fourth = pressure.begin_retirement(1);
  }
  bool circuit_rejected = false;
  try {
    (void)pressure.begin_retirement(1);
  } catch (const std::runtime_error &) {
    circuit_rejected = true;
  }
  require(circuit_rejected, "circuit admitted a control-plane retirement");
  pressure.record_sink_hit();
  pressure.complete_retirement(fourth);
  pressure.complete_retirement(third);
  require(pressure.stats().state == se::RetirementPressureState::kCircuitOpen,
          "circuit closed before reaching hysteresis low threshold");
  pressure.complete_retirement(second);
  require(pressure.stats().state == se::RetirementPressureState::kNormal,
          "circuit did not close below backpressure threshold");

  std::this_thread::sleep_for(std::chrono::milliseconds(2));
  pressure.refresh_stall_state();
  pressure.refresh_stall_state();
  require(pressure.stats().epoch_stall_count == 1,
          "continuous epoch stall emitted more than one event");
  pressure.complete_retirement(first);
  const auto stats = pressure.stats();
  require(stats.retired_bytes == 0, "retired byte accounting did not drain");
  require(stats.sink_hit_count == 1, "sink hit accounting mismatch");
  return 0;
}
