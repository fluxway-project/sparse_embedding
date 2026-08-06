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

#ifndef SPARSE_EMBEDDING_CHECKPOINT_REGISTRY_HPP_
#define SPARSE_EMBEDDING_CHECKPOINT_REGISTRY_HPP_

#include <string_view>

#include "sparse_embedding/src/checkpoint/checkpoint_interface.hpp"

namespace se {

struct CheckpointFactories {
  CheckpointWriterFactory writer;
  CheckpointReaderFactory reader;
};

using CheckpointFactoryBuilder = CheckpointFactories (*)();

void register_checkpoint_backend(std::string_view name,
                                 CheckpointFactoryBuilder builder);

CheckpointFactories make_checkpoint_factories(std::string_view name);

} // namespace se

#endif // SPARSE_EMBEDDING_CHECKPOINT_REGISTRY_HPP_
