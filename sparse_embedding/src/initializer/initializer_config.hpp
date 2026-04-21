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

#ifndef SPARSE_EMBEDDING_INITIALIZER_CONFIG_HPP_
#define SPARSE_EMBEDDING_INITIALIZER_CONFIG_HPP_

#include <variant>

namespace se {

struct ZerosInitializerConfig {};

struct UniformInitializerConfig {
  float min_val = -0.05f;
  float max_val = 0.05f;
};

// Extensible variant for python bindings
using InitializerConfigVariant =
    std::variant<ZerosInitializerConfig, UniformInitializerConfig>;

} // namespace se

#endif // SPARSE_EMBEDDING_INITIALIZER_CONFIG_HPP_
