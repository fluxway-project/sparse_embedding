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

#ifndef SPARSE_EMBEDDING_OPTIMIZER_CONFIG_HPP_
#define SPARSE_EMBEDDING_OPTIMIZER_CONFIG_HPP_

#include <variant>

namespace se {

struct AdamConfig {
  float lr = 1e-3f;
  float beta1 = 0.9f;
  float beta2 = 0.999f;
  float eps = 1e-8f;
  float weight_decay = 0.0f;
};

struct SgdConfig {
  float lr = 1e-2f;
  float weight_decay = 0.0f;
};

// Extensible variant for python bindings
using OptimizerConfigVariant = std::variant<AdamConfig, SgdConfig>;

} // namespace se

#endif // SPARSE_EMBEDDING_OPTIMIZER_CONFIG_HPP_
