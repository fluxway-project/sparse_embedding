// Copyright 2026 The Fluxway Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#pragma once

#ifndef SPARSE_EMBEDDING_DYNAMIC_LR_POLICY_HPP_
#define SPARSE_EMBEDDING_DYNAMIC_LR_POLICY_HPP_

#include <cmath>
#include <cstdint>
#include <numbers>
#include <stdexcept>
#include <variant>
#include <vector>

#include "sparse_embedding/src/optimizer/learning_rate.hpp"

namespace se {

struct ConstantLrPolicy {
  constexpr float compute_lr(float base_lr, uint64_t /*score*/) const {
    return base_lr;
  }
  constexpr bool requires_score() const { return false; }
};

struct SqrtDecayLrPolicy {
  constexpr float compute_lr(float base_lr, uint64_t score) const {
    if (score == 0)
      return base_lr;
    return base_lr / std::sqrt(static_cast<float>(score));
  }
  constexpr bool requires_score() const { return true; }
};

struct LogDecayLrPolicy {
  constexpr float compute_lr(float base_lr, uint64_t score) const {
    return base_lr / std::log2(2.0f + static_cast<float>(score));
  }
  constexpr bool requires_score() const { return true; }
};

struct InverseDecayLrPolicy {
  constexpr float compute_lr(float base_lr, uint64_t score) const {
    if (score == 0)
      return base_lr;
    return base_lr / static_cast<float>(score);
  }
  constexpr bool requires_score() const { return true; }
};

struct CosineFrequencyLrPolicy {
  std::vector<float> lut;
  uint64_t max_score;
  float min_lr_ratio;

  CosineFrequencyLrPolicy(float min_ratio = 0.1f, uint64_t max_s = 1024)
      : max_score(max_s), min_lr_ratio(min_ratio) {
    if (!std::isfinite(min_lr_ratio) || min_lr_ratio < 0.0f ||
        min_lr_ratio > 1.0f) {
      throw std::invalid_argument("min_lr_ratio must be finite and in [0, 1]");
    }
    if (max_score == 0) {
      throw std::invalid_argument("max_score must be greater than zero");
    }
    lut.resize(max_score + 1);
    for (uint64_t i = 0; i <= max_score; ++i) {
      float ratio = static_cast<float>(i) / static_cast<float>(max_score);
      float cos_val = std::cos(std::numbers::pi_v<float> * ratio);
      float multiplier = (1.0f + cos_val) / 2.0f;
      lut[i] = multiplier;
    }
  }

  constexpr float compute_lr(float base_lr, uint64_t score) const {
    float min_lr = base_lr * min_lr_ratio;
    if (score >= max_score)
      return min_lr;
    return min_lr + (base_lr - min_lr) * lut[score];
  }
  constexpr bool requires_score() const { return true; }
};

struct CustomLutLrPolicy {
  std::vector<float> lut;
  uint64_t max_score;

  CustomLutLrPolicy(std::vector<float> custom_lut)
      : lut(std::move(custom_lut)), max_score(0) {
    if (lut.empty()) {
      throw std::invalid_argument("lut must not be empty");
    }
    for (float multiplier : lut) {
      if (!std::isfinite(multiplier) || multiplier < 0.0f) {
        throw std::invalid_argument(
            "lut multipliers must be finite and non-negative");
      }
    }
    max_score = lut.size() - 1;
  }

  constexpr float compute_lr(float base_lr, uint64_t score) const {
    if (score >= max_score)
      return base_lr * lut[max_score];
    return base_lr * lut[score];
  }
  constexpr bool requires_score() const { return true; }
};

using DynamicLrPolicyVariant =
    std::variant<ConstantLrPolicy, SqrtDecayLrPolicy, LogDecayLrPolicy,
                 InverseDecayLrPolicy, CosineFrequencyLrPolicy,
                 CustomLutLrPolicy>;

static_assert(DynamicLrPolicy<ConstantLrPolicy>);
static_assert(DynamicLrPolicy<SqrtDecayLrPolicy>);
static_assert(DynamicLrPolicy<LogDecayLrPolicy>);
static_assert(DynamicLrPolicy<InverseDecayLrPolicy>);
static_assert(DynamicLrPolicy<CosineFrequencyLrPolicy>);
static_assert(DynamicLrPolicy<CustomLutLrPolicy>);

} // namespace se

#endif // SPARSE_EMBEDDING_DYNAMIC_LR_POLICY_HPP_
