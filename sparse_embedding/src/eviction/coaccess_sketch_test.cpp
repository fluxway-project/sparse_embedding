#include "sparse_embedding/src/eviction/coaccess_sketch.hpp"

#include <array>
#include <limits>
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
  require_throws(
      [] { se::CoaccessSketch(std::numeric_limits<size_t>::max() / 2 + 1); });
  se::CoaccessSketch sketch(1 << 16);
  constexpr std::array<uint64_t, 4> keys{10, 20, 30, 40};
  std::vector<std::thread> workers;
  for (int worker = 0; worker < 8; ++worker) {
    workers.emplace_back([&] {
      for (int iteration = 0; iteration < 1000; ++iteration)
        sketch.observe_batch<uint64_t>(keys);
    });
  }
  for (auto &worker : workers)
    worker.join();
  require(sketch.estimate(10, 20) >= 8000,
          "co-access sketch lost concurrent observations");
  const uint32_t before_decay = sketch.estimate(10, 20);
  sketch.decay_range(0, sketch.columns(), 1);
  const uint32_t after_decay = sketch.estimate(10, 20);
  require(after_decay == before_decay / 2, "co-access sketch decay mismatch");

  constexpr std::array<std::array<uint64_t, 2>, 4> worker_samples{{
      {{101, 102}},
      {{201, 202}},
      {{301, 302}},
      {{401, 402}},
  }};
  se::CoaccessSketch full(1 << 16);
  se::CoaccessSketch sampled(1 << 16);
  for (size_t invocation = 0; invocation < worker_samples.size();
       ++invocation) {
    for (const auto &worker_sample : worker_samples)
      full.observe_batch<uint64_t>(worker_sample);
    sampled.observe_batch<uint64_t>(
        worker_samples[invocation],
        static_cast<uint32_t>(worker_samples.size()));
  }
  for (const auto &worker_sample : worker_samples) {
    require(full.estimate(worker_sample[0], worker_sample[1]) ==
                sampled.estimate(worker_sample[0], worker_sample[1]),
            "weighted coaccess sampling did not conserve observations");
  }
}
