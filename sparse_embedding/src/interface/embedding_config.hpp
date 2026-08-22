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

#ifndef SPARSE_EMBEDDING_EMBEDDING_CONFIG_HPP_
#define SPARSE_EMBEDDING_EMBEDDING_CONFIG_HPP_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "sparse_embedding/src/storage/page_storage.hpp"
#include "sparse_embedding/src/storage/topology/storage_topology.hpp"

namespace se {

struct EmbeddingConfig {
  std::vector<size_t>
      dims; // physical table dims, each unique value creates one SlotStore
  size_t thread_budget = 0; // 0 means automatic local execution budget
  std::string value_dtype = "float32";
  std::string optimizer_dtype = ""; // if empty, falls back to value_dtype
  std::string key_dtype = "int64";  // int64, uint64, int32, uint32
  StorageOptions storage;
  storage::StorageTuningOptions storage_tuning;
};

} // namespace se

#endif // SPARSE_EMBEDDING_EMBEDDING_CONFIG_HPP_
