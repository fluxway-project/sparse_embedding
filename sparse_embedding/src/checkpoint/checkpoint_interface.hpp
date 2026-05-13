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

#ifndef SPARSE_EMBEDDING_CHECKPOINT_INTERFACE_HPP_
#define SPARSE_EMBEDDING_CHECKPOINT_INTERFACE_HPP_

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <proxy/proxy.h>

namespace se {

// ---------------------------------------------------------------------------
// Checkpoint mode / options
// ---------------------------------------------------------------------------

struct CheckpointExportOptions {
  enum class Mode : uint8_t { kFull, kIncremental };
  Mode mode = Mode::kFull;
  uint64_t min_score = 0;
  bool include_optimizer_state = false;
  size_t batch_size = 65536;
  bool use_compression = false;
};

struct CheckpointLoadOptions {
  bool clear_before_load = true;
  bool load_optimizer_state = false;
  uint64_t version = 0;
  size_t batch_size = 65536;
};

struct CheckpointVersion {
  uint64_t version_id;
  uint64_t timestamp_ms;
  uint64_t num_entries;
};

struct CheckpointMetadata {
  std::vector<size_t> table_dims;
  uint64_t global_step = 0;
  uint64_t timestamp_ms = 0;
  std::string score_policy;
  std::string dtype;
  bool has_optimizer_state = false;
};

// ---------------------------------------------------------------------------
// CheckpointWriterFacade — duck-typed proxy.
//
// Any struct satisfying the three methods qualifies.
// ---------------------------------------------------------------------------

PRO_DEF_MEM_DISPATCH(MemCkptWriterBegin, begin);
PRO_DEF_MEM_DISPATCH(MemCkptWriteBatch, write_batch);
PRO_DEF_MEM_DISPATCH(MemCkptCommit, commit);

struct CheckpointWriterFacade
    : pro::facade_builder ::add_convention<
          MemCkptWriterBegin,
          void(std::string_view uri, size_t dim, size_t optimizer_state_dim,
               bool has_score, CheckpointExportOptions options)>::
          add_convention<
              MemCkptWriteBatch,
              void(std::span<const uint64_t> keys, const void *embeddings,
                   size_t embedding_bytes, std::span<const uint64_t> scores,
                   const void *optimizer_states, size_t opt_state_bytes)>::
              add_convention<MemCkptCommit, uint64_t()>::support_copy<
                  pro::constraint_level::nontrivial>::build {};

// ---------------------------------------------------------------------------
// CheckpointReaderFacade — duck-typed proxy.
// ---------------------------------------------------------------------------

PRO_DEF_MEM_DISPATCH(MemCkptReaderOpen, open);
PRO_DEF_MEM_DISPATCH(MemCkptReadBatch, read_batch);
PRO_DEF_MEM_DISPATCH(MemCkptVersionInfo, version_info);

struct CheckpointReaderFacade
    : pro::facade_builder ::add_convention<
          MemCkptReaderOpen, void(std::string_view uri, size_t dim,
                                  uint64_t version, int64_t batch_size)>::
          add_convention<MemCkptReadBatch,
                         bool(size_t max_n, std::vector<uint64_t> &out_keys,
                              std::vector<std::byte> &out_embeddings,
                              std::vector<uint64_t> &out_scores,
                              std::vector<std::byte> &out_optimizer_states)>::
              add_convention<MemCkptVersionInfo, CheckpointVersion() const>::
                  support_copy<pro::constraint_level::nontrivial>::build {};

// ---------------------------------------------------------------------------
// Factory types — injected into the checkpoint coordinator
// ---------------------------------------------------------------------------

using CheckpointWriterFactory = pro::proxy<CheckpointWriterFacade>;
using CheckpointReaderFactory = pro::proxy<CheckpointReaderFacade>;

} // namespace se

#endif // SPARSE_EMBEDDING_CHECKPOINT_INTERFACE_HPP_
