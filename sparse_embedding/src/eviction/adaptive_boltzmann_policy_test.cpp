#include "sparse_embedding/src/eviction/adaptive_boltzmann_policy.hpp"

#include <array>
#include <cmath>
#include <stdexcept>

namespace {

void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

} // namespace

int main() {
  std::array signals{
      se::PageThermalSignal{1.0F, 0.0F, 0.0F},
      se::PageThermalSignal{4.0F, 1.0F, 0.0F},
      se::PageThermalSignal{8.0F, 2.0F, 1.0F},
  };
  std::array<float, signals.size()> weights{};
  se::AdaptiveBoltzmannPolicy policy;
  const auto summary = policy.compute_eviction_weights(signals, 0.75F, weights);
  require(summary.temperature >= 0.125F && summary.temperature <= 8.0F,
          "adaptive temperature escaped bounds");
  require(weights[0] > weights[1] && weights[1] > weights[2],
          "colder pages must receive larger eviction weights");
  require(std::abs(weights[0] + weights[1] + weights[2] - 1.0F) < 1.0e-5F,
          "Boltzmann weights must sum to one");

  alignas(64) std::array<float, 32> heat{};
  alignas(64) std::array<float, 32> coaccess{};
  alignas(64) std::array<float, 32> age{};
  alignas(64) std::array<float, 32> utilities{};
  for (size_t index = 0; index < heat.size(); ++index) {
    heat[index] = static_cast<float>(index);
    coaccess[index] = 2.0F;
    age[index] = 4.0F;
  }
  policy.compute_utilities(heat, coaccess, age, utilities);
  for (size_t index = 0; index < utilities.size(); ++index) {
    require(utilities[index] == static_cast<float>(index) + 3.0F,
            "batched thermal utility mismatch");
  }

  for (int index = 0; index <= 16000; ++index) {
    const float value = -static_cast<float>(index) / 1000.0F;
    const float exact = std::exp(value);
    const float approximate =
        se::AdaptiveBoltzmannPolicy::approximate_exp(value);
    const float relative_error = std::abs(approximate - exact) / exact;
    require(relative_error < 0.001F,
            "approximate exponential exceeded relative-error bound");
  }
}
