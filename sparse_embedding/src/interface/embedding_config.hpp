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

namespace se {

struct EmbeddingConfig {
  std::vector<size_t>
      dims; // physical table dims, each unique value creates one SlotStore
  size_t num_threads = 0; // 0 means auto
  std::string dtype = "float32";
  std::string opt_dtype = "";       // if empty, falls back to dtype
  std::string key_dtype = "int64";  // int64, uint64, int32, uint32
  size_t max_active_keys = 1048576; // 1 << 20
};

} // namespace se

#endif // SPARSE_EMBEDDING_EMBEDDING_CONFIG_HPP_
