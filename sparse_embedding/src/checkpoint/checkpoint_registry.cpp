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

#include "sparse_embedding/src/checkpoint/checkpoint_registry.hpp"

#include <algorithm>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace se {
namespace {

struct CheckpointBackendEntry {
  std::string name;
  CheckpointFactoryBuilder builder;
};

struct CheckpointBackendRegistry {
  std::mutex mutex;
  std::vector<CheckpointBackendEntry> entries;
};

CheckpointBackendRegistry &registry() {
  static CheckpointBackendRegistry instance;
  return instance;
}

} // namespace

void register_checkpoint_backend(std::string_view name,
                                 CheckpointFactoryBuilder builder) {
  if (name.empty())
    throw std::invalid_argument("checkpoint backend name must not be empty");
  if (builder == nullptr)
    throw std::invalid_argument("checkpoint backend builder must not be null");

  auto &state = registry();
  std::lock_guard lock(state.mutex);
  const auto duplicate =
      std::find_if(state.entries.begin(), state.entries.end(),
                   [name](const CheckpointBackendEntry &entry) {
                     return entry.name == name;
                   });
  if (duplicate != state.entries.end())
    throw std::logic_error("checkpoint backend already registered: " +
                           std::string(name));
  state.entries.push_back({std::string(name), builder});
}

CheckpointFactories make_checkpoint_factories(std::string_view name) {
  if (name.empty())
    throw std::invalid_argument("checkpoint backend name must not be empty");

  auto &state = registry();
  std::lock_guard lock(state.mutex);
  const auto match = std::find_if(state.entries.begin(), state.entries.end(),
                                  [name](const CheckpointBackendEntry &entry) {
                                    return entry.name == name;
                                  });
  if (match == state.entries.end())
    throw std::invalid_argument("unsupported checkpoint backend: " +
                                std::string(name));
  return match->builder();
}

} // namespace se
