#include "sparse_embedding/src/simd/simd_ops.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>

int main() {
  alignas(64) std::array<float, 37> parameters{};
  alignas(64) std::array<float, 37> gradients{};
  for (size_t i = 0; i < parameters.size(); ++i) {
    parameters[i] = static_cast<float>(i + 1);
    gradients[i] = 0.5f;
  }

  se::simd::simd_sgd_update(parameters.data(), gradients.data(), 0.1f, -0.01f,
                            parameters.size());

  for (size_t i = 0; i < parameters.size(); ++i) {
    float original = static_cast<float>(i + 1);
    float expected = original - 0.1f * (0.5f - 0.01f * original);
    if (std::abs(parameters[i] - expected) > 1e-6f)
      throw std::runtime_error("SIMD SGD result mismatch");
  }
}
