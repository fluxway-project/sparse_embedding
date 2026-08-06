#pragma once

#ifndef SPARSE_EMBEDDING_ADAPTIVE_BOLTZMANN_POLICY_HPP_
#define SPARSE_EMBEDDING_ADAPTIVE_BOLTZMANN_POLICY_HPP_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace se {

struct AdaptiveBoltzmannConfig {
  float base_temperature = 1.0F;
  float minimum_temperature = 0.125F;
  float maximum_temperature = 8.0F;
  float heat_weight = 1.0F;
  float coaccess_weight = 1.0F;
  float residency_weight = 0.25F;
  float pressure_gain = 2.0F;
};

struct PageThermalSignal {
  float decayed_heat;
  float coaccess_density;
  float residency_age;
};

struct BoltzmannSummary {
  float temperature;
  float minimum_utility;
  float maximum_utility;
  float weight_sum;
};

class AdaptiveBoltzmannPolicy {
public:
  explicit AdaptiveBoltzmannPolicy(AdaptiveBoltzmannConfig config = {})
      : config_(config) {
    validate_config();
  }

  BoltzmannSummary
  compute_eviction_weights(std::span<const PageThermalSignal> signals,
                           float memory_pressure,
                           std::span<float> out_weights) const {
    if (signals.size() != out_weights.size())
      throw std::invalid_argument("Boltzmann signal/output size mismatch");
    if (signals.empty())
      throw std::invalid_argument("Boltzmann signals must not be empty");
    if (!std::isfinite(memory_pressure) || memory_pressure < 0.0F ||
        memory_pressure > 1.0F)
      throw std::invalid_argument("memory pressure must be in [0, 1]");

    float minimum = std::numeric_limits<float>::infinity();
    float maximum = -std::numeric_limits<float>::infinity();
    double mean = 0.0;
    for (size_t index = 0; index < signals.size(); ++index) {
      const float utility = page_utility(signals[index]);
      if (!std::isfinite(utility))
        throw std::invalid_argument("page thermal signal must be finite");
      out_weights[index] = utility;
      minimum = std::min(minimum, utility);
      maximum = std::max(maximum, utility);
      mean += utility;
    }
    mean /= static_cast<double>(signals.size());

    double variance = 0.0;
    for (float utility : out_weights) {
      const double delta = static_cast<double>(utility) - mean;
      variance += delta * delta;
    }
    variance /= static_cast<double>(signals.size());
    const float dispersion = static_cast<float>(std::sqrt(variance));
    const float normalized_dispersion =
        dispersion / std::max(std::abs(static_cast<float>(mean)), 1.0F);
    const float temperature =
        std::clamp(config_.base_temperature *
                       (1.0F + config_.pressure_gain * memory_pressure) /
                       (1.0F + normalized_dispersion),
                   config_.minimum_temperature, config_.maximum_temperature);

    std::vector<float> exponents(out_weights.size());
    for (size_t index = 0; index < out_weights.size(); ++index)
      exponents[index] = -(out_weights[index] - minimum) / temperature;
    approximate_exp_batch(exponents, out_weights);
    const float weight_sum = normalize_weights(out_weights);
    return {temperature, minimum, maximum, weight_sum};
  }

  void compute_utilities(std::span<const float> decayed_heat,
                         std::span<const float> coaccess_density,
                         std::span<const float> residency_age,
                         std::span<float> out_utilities) const {
    const size_t count = decayed_heat.size();
    if (coaccess_density.size() != count || residency_age.size() != count ||
        out_utilities.size() != count)
      throw std::invalid_argument("thermal signal column size mismatch");
    const float heat_weight = config_.heat_weight;
    const float coaccess_weight = config_.coaccess_weight;
    const float residency_weight = config_.residency_weight;
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC ivdep
#endif
    for (size_t index = 0; index < count; ++index) {
      out_utilities[index] = heat_weight * decayed_heat[index] +
                             coaccess_weight * coaccess_density[index] +
                             residency_weight * residency_age[index];
    }
  }

  void approximate_exp_batch(std::span<const float> values,
                             std::span<float> out) const {
    if (values.size() != out.size())
      throw std::invalid_argument("approximate exponential columns differ");
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC ivdep
#endif
    for (size_t index = 0; index < values.size(); ++index)
      out[index] = approximate_exp(values[index]);
  }

  static float normalize_weights(std::span<float> weights) {
    float sum = 0.0F;
    for (float weight : weights)
      sum += weight;
    if (!(sum > 0.0F) || !std::isfinite(sum))
      throw std::runtime_error("Boltzmann weight normalization failed");
    const float inverse_sum = 1.0F / sum;
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC ivdep
#endif
    for (float &weight : weights)
      weight *= inverse_sum;
    return sum;
  }

  float page_utility(const PageThermalSignal &signal) const noexcept {
    return config_.heat_weight * signal.decayed_heat +
           config_.coaccess_weight * signal.coaccess_density +
           config_.residency_weight * signal.residency_age;
  }

  static inline float approximate_exp(float value) noexcept {
    constexpr float kMinimumExponent = -16.0F;
    constexpr float kInverseLn2 = 1.4426950408889634F;
    constexpr float kLn2 = 0.6931471805599453F;
    value = std::clamp(value, kMinimumExponent, 0.0F);
    const int power = static_cast<int>(std::floor(value * kInverseLn2));
    const float remainder = value - static_cast<float>(power) * kLn2;
    const float remainder2 = remainder * remainder;
    const float polynomial =
        1.0F + remainder +
        remainder2 *
            (0.5F + remainder * (0.1666666716F +
                                 remainder * (0.0416666679F +
                                              remainder * 0.0083333338F)));
    return std::ldexp(polynomial, power);
  }

private:
  void validate_config() const {
    const float values[] = {
        config_.base_temperature,    config_.minimum_temperature,
        config_.maximum_temperature, config_.heat_weight,
        config_.coaccess_weight,     config_.residency_weight,
        config_.pressure_gain};
    for (float value : values) {
      if (!std::isfinite(value) || value < 0.0F)
        throw std::invalid_argument(
            "Boltzmann configuration must be finite and non-negative");
    }
    if (!(config_.minimum_temperature > 0.0F) ||
        config_.minimum_temperature > config_.base_temperature ||
        config_.base_temperature > config_.maximum_temperature)
      throw std::invalid_argument("invalid Boltzmann temperature bounds");
  }

  AdaptiveBoltzmannConfig config_;
};

} // namespace se

#endif
