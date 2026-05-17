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

#ifndef SPARSE_EMBEDDING_LANCE_CHECKPOINT_HPP_
#define SPARSE_EMBEDDING_LANCE_CHECKPOINT_HPP_

#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

#include <lance/lance.h>
#include <lance/lance.hpp>

#include "sparse_embedding/src/checkpoint/checkpoint_interface.hpp"

namespace se {

// Plain struct — satisfies CheckpointWriterFacade via duck typing (no virtual).
class LanceCheckpointWriter {
  std::string uri_;
  size_t dim_ = 0;
  size_t opt_state_dim_ = 0;
  bool has_score_ = false;
  CheckpointExportOptions options_;

  std::vector<uint64_t> buf_keys_;
  std::vector<std::byte> buf_embeddings_;
  std::vector<uint64_t> buf_scores_;
  std::vector<std::byte> buf_opt_states_;
  std::vector<uint32_t> buf_steps_;
  size_t entry_byte_size_ = 0;
  bool is_first_batch_ = true;
  uint64_t committed_version_ = 0;

  struct BatchStreamState {
    const LanceCheckpointWriter *writer;
    bool consumed;
  };

  static int stream_get_schema(ArrowArrayStream *stream, ArrowSchema *out);
  static int stream_get_next(ArrowArrayStream *stream, ArrowArray *out);
  static const char *stream_get_last_error(ArrowArrayStream *);
  static void stream_release(ArrowArrayStream *stream);

  void build_schema(ArrowSchema *out) const;
  void build_array(ArrowArray *out) const;
  void flush_to_lance();

  static void release_schema(ArrowSchema *schema);
  static void release_array(ArrowArray *array);

public:
  void begin(std::string_view uri, size_t dim, size_t optimizer_state_dim,
             bool has_score, CheckpointExportOptions options);

  void write_batch(std::span<const uint64_t> keys, const void *embeddings,
                   size_t embedding_bytes, std::span<const uint64_t> scores,
                   const void *optimizer_states, size_t opt_state_bytes,
                   std::span<const uint32_t> steps);

  uint64_t commit();
};

// Plain struct — satisfies CheckpointReaderFacade via duck typing (no virtual).
class LanceCheckpointReader {
  std::string uri_;
  size_t dim_ = 0;
  uint64_t version_ = 0;
  CheckpointVersion version_info_{};

  ArrowArrayStream stream_{};
  ArrowSchema schema_{};
  bool stream_open_ = false;
  size_t opt_state_dim_ = 0;
  bool has_score_ = false;
  bool has_step_ = false;
  int64_t batch_size_ = 65536;

  void close_stream();

public:
  ~LanceCheckpointReader();

  void open(std::string_view uri, size_t dim, uint64_t version,
            int64_t batch_size = 65536);

  bool read_batch(size_t max_n, std::vector<uint64_t> &out_keys,
                  std::vector<std::byte> &out_embeddings,
                  std::vector<uint64_t> &out_scores,
                  std::vector<std::byte> &out_optimizer_states,
                  std::vector<uint32_t> &out_steps);

  CheckpointVersion version_info() const { return version_info_; }
};

} // namespace se

#endif // SPARSE_EMBEDDING_LANCE_CHECKPOINT_HPP_
