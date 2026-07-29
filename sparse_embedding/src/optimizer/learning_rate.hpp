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

#ifndef SPARSE_EMBEDDING_LEARNING_RATE_HPP_
#define SPARSE_EMBEDDING_LEARNING_RATE_HPP_

#include <concepts>
#include <cstdint>
#include <utility>

namespace se {

template <typename Policy>
concept DynamicLrPolicy =
    std::copy_constructible<Policy> &&
    requires(const Policy &policy, float base_lr, uint64_t score) {
      { policy.compute_lr(base_lr, score) } -> std::same_as<float>;
      { policy.requires_score() } -> std::same_as<bool>;
    };

struct StaticLearningRate {
  explicit constexpr StaticLearningRate(float base_lr) : base_lr_(base_lr) {}

  constexpr float compute(uint64_t /*score*/) const { return base_lr_; }
  constexpr bool requires_score() const { return false; }

private:
  float base_lr_;
};

template <DynamicLrPolicy PolicyT> struct DynamicLearningRate {
  constexpr DynamicLearningRate(float base_lr, PolicyT policy)
      : base_lr_(base_lr), policy_(std::move(policy)) {}

  constexpr float compute(uint64_t score) const {
    return policy_.compute_lr(base_lr_, score);
  }

  constexpr bool requires_score() const { return policy_.requires_score(); }

private:
  float base_lr_;
  PolicyT policy_;
};

} // namespace se

#endif // SPARSE_EMBEDDING_LEARNING_RATE_HPP_
