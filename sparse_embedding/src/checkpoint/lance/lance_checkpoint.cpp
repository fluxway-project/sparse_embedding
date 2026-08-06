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

#include "sparse_embedding/src/checkpoint/lance/lance_checkpoint.hpp"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <span>
#include <stdexcept>

namespace se {

// =============================================================================
// Arrow helpers
// =============================================================================

namespace {

void no_op_release_schema(ArrowSchema *s) {
  if (!s)
    return;
  for (auto *child :
       std::span{s->children, static_cast<size_t>(s->n_children)}) {
    if (child && child->release) {
      child->release(child);
    }
  }
  delete[] s->children;
  s->children = nullptr;
  s->release = nullptr;
}

void no_op_release_array(ArrowArray *a) {
  if (!a)
    return;
  for (auto *child :
       std::span{a->children, static_cast<size_t>(a->n_children)}) {
    if (child && child->release) {
      child->release(child);
    }
  }
  for (int64_t i = 0; i < a->n_buffers; ++i) {
    // buffers are borrowed, don't free
  }
  delete[] a->children;
  delete[] a->buffers;
  a->children = nullptr;
  a->buffers = nullptr;
  a->release = nullptr;
}

void make_child_schema(ArrowSchema *child, const char *name,
                       const char *format) {
  *child = ArrowSchema{
      .format = format,
      .name = name,
      .metadata = nullptr,
      .flags = 0,
      .n_children = 0,
      .children = nullptr,
      .dictionary = nullptr,
      .release = no_op_release_schema,
      .private_data = nullptr,
  };
}

void make_child_array(ArrowArray *child, int64_t length, const void *buf0,
                      const void *buf1) {
  *child = ArrowArray{
      .length = length,
      .null_count = 0,
      .offset = 0,
      .n_buffers = 2,
      .n_children = 0,
      .buffers = new const void *[2] { buf0, buf1 },
      .children = nullptr,
      .dictionary = nullptr,
      .release = no_op_release_array,
      .private_data = nullptr,
  };
}

std::string dim_uri(std::string_view base, size_t dim) {
  if (base.empty() || base.back() == '/') {
    return std::format("{}dim_{:04}", base, dim);
  } else {
    return std::format("{}/dim_{:04}", base, dim);
  }
}

const char *get_zstd_metadata() {
  static const std::string meta = [] {
    std::string m;
    int32_t num_kvs = 1;
    std::string key = "lance-encoding:compression";
    std::string val = "zstd";
    int32_t key_len = static_cast<int32_t>(key.size());
    int32_t val_len = static_cast<int32_t>(val.size());

    m.append(reinterpret_cast<const char *>(&num_kvs), 4);
    m.append(reinterpret_cast<const char *>(&key_len), 4);
    m.append(key);
    m.append(reinterpret_cast<const char *>(&val_len), 4);
    m.append(val);
    return m;
  }();
  return meta.c_str();
}

} // namespace

// =============================================================================
// LanceCheckpointWriter
// =============================================================================

void LanceCheckpointWriter::begin(std::string_view uri, size_t dim,
                                  size_t optimizer_state_dim, bool has_score,
                                  CheckpointExportOptions options) {
  uri_ = dim_uri(uri, dim);
  dim_ = dim;
  opt_state_dim_ = optimizer_state_dim;
  has_score_ = has_score;
  options_ = options;
  is_first_batch_ = true;
  committed_version_ = 0;
  num_keys_ = 0;

  buf_keys_.clear();
  buf_embeddings_.clear();
  buf_scores_.clear();
  buf_opt_states_.clear();
  buf_steps_.clear();
}

void LanceCheckpointWriter::write_batch(
    size_t num_keys, const void *keys, size_t key_bytes, const void *embeddings,
    size_t embedding_bytes, std::span<const uint64_t> scores,
    const void *optimizer_states, size_t opt_state_bytes,
    std::span<const uint32_t> steps) {
  if (num_keys == 0)
    return;
  if (keys == nullptr || embeddings == nullptr || key_bytes == 0 ||
      embedding_bytes == 0 || key_bytes % num_keys != 0 ||
      embedding_bytes % num_keys != 0)
    throw std::invalid_argument("invalid Lance checkpoint batch buffers");
  if (has_score_ && scores.size() != num_keys)
    throw std::invalid_argument("Lance checkpoint score count mismatch");
  if (opt_state_dim_ > 0 && (optimizer_states == nullptr ||
                             opt_state_bytes != num_keys * opt_state_dim_))
    throw std::invalid_argument(
        "Lance checkpoint optimizer state size mismatch");
  if (!steps.empty() && steps.size() != num_keys)
    throw std::invalid_argument("Lance checkpoint step count mismatch");

  auto *k = static_cast<const std::byte *>(keys);
  buf_keys_.insert(buf_keys_.end(), k, k + key_bytes);
  num_keys_ += num_keys;

  auto *e = static_cast<const std::byte *>(embeddings);
  buf_embeddings_.insert(buf_embeddings_.end(), e, e + embedding_bytes);

  if (has_score_ && !scores.empty()) {
    buf_scores_.insert(buf_scores_.end(), scores.begin(), scores.end());
  } else {
    buf_scores_.clear();
  }

  if (opt_state_dim_ > 0 && optimizer_states && opt_state_bytes > 0) {
    auto *p = static_cast<const std::byte *>(optimizer_states);
    buf_opt_states_.insert(buf_opt_states_.end(), p, p + opt_state_bytes);
  } else {
    buf_opt_states_.clear();
  }

  if (!steps.empty()) {
    buf_steps_.insert(buf_steps_.end(), steps.begin(), steps.end());
  } else {
    buf_steps_.clear();
  }

  flush_to_lance();
}

void LanceCheckpointWriter::build_schema(ArrowSchema *out) const {
  int n_cols = 2; // key + embedding
  if (has_score_)
    ++n_cols;
  if (opt_state_dim_ > 0)
    ++n_cols;
  if (!buf_steps_.empty())
    ++n_cols;

  *out = ArrowSchema{
      .format = "+s", // struct
      .name = "",
      .metadata = nullptr,
      .flags = 0,
      .n_children = n_cols,
      .children = new ArrowSchema *[n_cols],
      .dictionary = nullptr,
      .release = no_op_release_schema,
      .private_data = nullptr,
  };

  int col = 0;

  // key
  out->children[col] = new ArrowSchema();
  make_child_schema(out->children[col], "key", options_.key_schema.c_str());
  ++col;

  // embedding: fixed_size_binary(dim * val_size)
  // We use fixed_size_binary since we don't know the ValT at this layer
  size_t entry_bytes = num_keys_ == 0 ? 0 : buf_embeddings_.size() / num_keys_;
  thread_local std::string emb_format;
  emb_format = "w:" + std::to_string(entry_bytes);
  out->children[col] = new ArrowSchema();
  make_child_schema(out->children[col], "embedding_params", emb_format.c_str());
  if (options_.use_compression) {
    out->children[col]->metadata = get_zstd_metadata();
  }
  ++col;

  if (has_score_) {
    out->children[col] = new ArrowSchema();
    make_child_schema(out->children[col], "score", "L");
    ++col;
  }

  if (opt_state_dim_ > 0) {
    thread_local std::string opt_format;
    opt_format = "w:" + std::to_string(opt_state_dim_);
    out->children[col] = new ArrowSchema();
    make_child_schema(out->children[col], "optimizer_states",
                      opt_format.c_str());
    if (options_.use_compression) {
      out->children[col]->metadata = get_zstd_metadata();
    }
    ++col;
  }

  // step: uint32
  if (!buf_steps_.empty()) {
    out->children[col] = new ArrowSchema();
    make_child_schema(out->children[col], "step", "I");
    ++col;
  }
}

void LanceCheckpointWriter::build_array(ArrowArray *out) const {
  int64_t n = static_cast<int64_t>(num_keys_);
  int n_cols = 2;
  if (has_score_)
    ++n_cols;
  if (opt_state_dim_ > 0)
    ++n_cols;
  if (!buf_steps_.empty())
    ++n_cols;

  *out = ArrowArray{
      .length = n,
      .null_count = 0,
      .offset = 0,
      .n_buffers = 1,
      .n_children = n_cols,
      .buffers = new const void *[1] { nullptr }, // validity bitmap (none)
      .children = new ArrowArray *[n_cols],
      .dictionary = nullptr,
      .release = no_op_release_array,
      .private_data = nullptr,
  };

  int col = 0;

  out->children[col] = new ArrowArray();
  make_child_array(out->children[col], n, nullptr, buf_keys_.data());
  ++col;

  out->children[col] = new ArrowArray();
  make_child_array(out->children[col], n, nullptr, buf_embeddings_.data());
  ++col;

  if (has_score_) {
    out->children[col] = new ArrowArray();
    make_child_array(out->children[col], n, nullptr, buf_scores_.data());
    ++col;
  }

  if (opt_state_dim_ > 0) {
    out->children[col] = new ArrowArray();
    make_child_array(out->children[col], n, nullptr, buf_opt_states_.data());
    ++col;
  }

  if (!buf_steps_.empty()) {
    out->children[col] = new ArrowArray();
    make_child_array(out->children[col], n, nullptr, buf_steps_.data());
    ++col;
  }
}

int LanceCheckpointWriter::stream_get_schema(ArrowArrayStream *stream,
                                             ArrowSchema *out) {
  auto *state = static_cast<BatchStreamState *>(stream->private_data);
  state->writer->build_schema(out);
  return 0;
}

int LanceCheckpointWriter::stream_get_next(ArrowArrayStream *stream,
                                           ArrowArray *out) {
  auto *state = static_cast<BatchStreamState *>(stream->private_data);
  if (state->consumed) {
    *out = ArrowArray{
        .length = 0,
        .null_count = 0,
        .offset = 0,
        .n_buffers = 0,
        .n_children = 0,
        .buffers = nullptr,
        .children = nullptr,
        .dictionary = nullptr,
        .release = nullptr,
        .private_data = nullptr,
    };
    return 0;
  }
  state->consumed = true;
  state->writer->build_array(out);
  return 0;
}

const char *LanceCheckpointWriter::stream_get_last_error(ArrowArrayStream *) {
  return nullptr;
}

void LanceCheckpointWriter::stream_release(ArrowArrayStream *stream) {
  auto *state = static_cast<BatchStreamState *>(stream->private_data);
  delete state;
  stream->private_data = nullptr;
  stream->release = nullptr;
}

void LanceCheckpointWriter::flush_to_lance() {
  if (buf_keys_.empty())
    return;

  ArrowSchema schema{};
  build_schema(&schema);

  BatchStreamState *state = new BatchStreamState{this, false};

  ArrowArrayStream stream{};
  stream.get_schema = &stream_get_schema;
  stream.get_next = &stream_get_next;
  stream.get_last_error = &stream_get_last_error;
  stream.release = &stream_release;
  stream.private_data = state;

  std::filesystem::create_directories(uri_);

  int32_t mode;
  if (is_first_batch_) {
    bool exists = std::filesystem::exists(uri_ + "/_versions");
    if (exists) {
      if (options_.mode == CheckpointExportOptions::Mode::kIncremental) {
        mode = LANCE_WRITE_APPEND;
      } else {
        mode = LANCE_WRITE_OVERWRITE;
      }
    } else {
      mode = LANCE_WRITE_CREATE;
    }
    is_first_batch_ = false;
  } else {
    mode = LANCE_WRITE_APPEND;
  }

  LanceDataset *out_ds = nullptr;
  int32_t rc;

  if (options_.use_compression) {
    LanceWriteParams params{};
    params.data_storage_version =
        "next"; // Use next version for compression support
    rc = lance_dataset_write_with_params(uri_.c_str(), &schema, &stream, mode,
                                         &params, nullptr, &out_ds);
  } else {
    rc = lance_dataset_write(uri_.c_str(), &schema, &stream, mode, nullptr,
                             &out_ds);
  }

  if (schema.release)
    schema.release(&schema);

  if (rc != 0) {
    const char *msg = lance_last_error_message();
    std::string err = msg ? msg : "unknown lance error";
    if (msg)
      lance_free_string(msg);
    throw std::runtime_error("lance checkpoint write failed: " + err);
  }

  if (out_ds) {
    committed_version_ = lance_dataset_version(out_ds);
    lance_dataset_close(out_ds);
  }

  buf_keys_.clear();
  num_keys_ = 0;
  buf_embeddings_.clear();
  buf_scores_.clear();
  buf_opt_states_.clear();
  buf_steps_.clear();
}

uint64_t LanceCheckpointWriter::commit() { return committed_version_; }

// =============================================================================
// LanceCheckpointReader
// =============================================================================

LanceCheckpointReader::~LanceCheckpointReader() { close_stream(); }

void LanceCheckpointReader::close_stream() {
  if (stream_open_ && stream_.release) {
    stream_.release(&stream_);
  }
  if (schema_.release) {
    schema_.release(&schema_);
  }
  stream_open_ = false;
}

void LanceCheckpointReader::open(std::string_view uri, size_t dim,
                                 uint64_t version, int64_t batch_size) {
  close_stream();
  uri_ = dim_uri(uri, dim);
  dim_ = dim;
  version_ = version;
  batch_size_ = batch_size > 0 ? batch_size : 65536;

  LanceDataset *ds = lance_dataset_open(uri_.c_str(), nullptr, version);
  if (!ds) {
    const char *msg = lance_last_error_message();
    std::string err = msg ? msg : "unknown";
    if (msg)
      lance_free_string(msg);
    throw std::runtime_error("failed to open lance checkpoint: " + err);
  }

  version_info_.version_id = lance_dataset_version(ds);
  version_info_.num_entries = lance_dataset_count_rows(ds);
  version_info_.timestamp_ms = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count());

  LanceScanner *scanner = lance_scanner_new(ds, nullptr, nullptr);
  if (!scanner) {
    lance_dataset_close(ds);
    const char *msg = lance_last_error_message();
    std::string err = msg ? msg : "unknown";
    if (msg)
      lance_free_string(msg);
    throw std::runtime_error("failed to create lance scanner: " + err);
  }

  stream_ = ArrowArrayStream{};
  lance_scanner_set_batch_size(scanner, batch_size_);
  if (lance_scanner_to_arrow_stream(scanner, &stream_) != 0) {
    lance_scanner_close(scanner);
    lance_dataset_close(ds);
    const char *msg = lance_last_error_message();
    std::string err = msg ? msg : "unknown";
    if (msg)
      lance_free_string(msg);
    throw std::runtime_error("failed to create arrow stream: " + err);
  }
  stream_open_ = true;

  schema_ = ArrowSchema{};
  stream_.get_schema(&stream_, &schema_);

  has_score_ = false;
  opt_state_dim_ = 0;
  has_step_ = false;
  dim_ = 0;
  key_dim_ = 8;
  for (int64_t i = 0; i < schema_.n_children; ++i) {
    auto *child = schema_.children[i];
    if (std::strcmp(child->name, "key") == 0) {
      if (child->format && (std::strcmp(child->format, "i") == 0 ||
                            std::strcmp(child->format, "I") == 0)) {
        key_dim_ = 4;
      } else {
        key_dim_ = 8;
      }
    } else if (std::strcmp(child->name, "embedding_params") == 0) {
      // format is "w:<nbytes>"
      if (child->format && child->format[0] == 'w' && child->format[1] == ':') {
        size_t bytes = std::stoul(child->format + 2);
        dim_ = bytes; // raw bytes per entry, caller interprets by ValT
      }
    } else if (std::strcmp(child->name, "score") == 0) {
      has_score_ = true;
    } else if (std::strcmp(child->name, "optimizer_states") == 0) {
      if (child->format && child->format[0] == 'w' && child->format[1] == ':') {
        size_t bytes = std::stoul(child->format + 2);
        opt_state_dim_ = bytes;
      }
    } else if (std::strcmp(child->name, "step") == 0) {
      has_step_ = true;
    }
  }

  lance_scanner_close(scanner);
  lance_dataset_close(ds);
}

bool LanceCheckpointReader::read_batch(
    size_t max_n, std::vector<std::byte> &out_keys,
    std::vector<std::byte> &out_embeddings, std::vector<uint64_t> &out_scores,
    std::vector<std::byte> &out_optimizer_states,
    std::vector<uint32_t> &out_steps) {
  if (!stream_open_)
    return false;

  ArrowArray batch{};
  int rc = stream_.get_next(&stream_, &batch);
  if (rc != 0 || !batch.release)
    return false;

  int64_t n = batch.length;
  if (n == 0) {
    if (batch.release)
      batch.release(&batch);
    return false;
  }

  int col = 0;

  // key column
  {
    auto *child = batch.children[col];
    auto *data = static_cast<const std::byte *>(
        static_cast<const void *>(child->buffers[1]));
    size_t total_bytes = static_cast<size_t>(n) * key_dim_;
    auto *start = data + child->offset * key_dim_;
    out_keys.assign(start, start + total_bytes);
    ++col;
  }

  // embedding column
  {
    auto *child = batch.children[col];
    auto *data = static_cast<const std::byte *>(
        static_cast<const void *>(child->buffers[1]));
    size_t total_bytes = static_cast<size_t>(n) * dim_;
    auto *start = data + child->offset * dim_;
    out_embeddings.assign(start, start + total_bytes);
    ++col;
  }

  // score column (optional)
  if (has_score_ && col < batch.n_children) {
    auto *child = batch.children[col];
    auto *data = static_cast<const uint64_t *>(child->buffers[1]);
    out_scores.assign(data + child->offset, data + child->offset + n);
    ++col;
  } else {
    out_scores.clear();
  }

  // optimizer_states column (optional)
  if (opt_state_dim_ > 0 && col < batch.n_children) {
    auto *child = batch.children[col];
    auto *data = static_cast<const std::byte *>(
        static_cast<const void *>(child->buffers[1]));
    size_t total_bytes = static_cast<size_t>(n) * opt_state_dim_;
    auto *start = data + child->offset * opt_state_dim_;
    out_optimizer_states.assign(start, start + total_bytes);
    ++col;
  } else {
    out_optimizer_states.clear();
  }

  // step column (optional)
  if (has_step_ && col < batch.n_children) {
    auto *child = batch.children[col];
    auto *data = static_cast<const uint32_t *>(child->buffers[1]);
    out_steps.assign(data + child->offset, data + child->offset + n);
    ++col;
  } else {
    out_steps.clear();
  }

  if (batch.release)
    batch.release(&batch);

  return true;
}

// =============================================================================
// LanceZeroCopyWriter
// =============================================================================

void LanceZeroCopyWriter::write(
    std::string_view uri, size_t dim, size_t opt_state_dim, bool has_score,
    CheckpointExportOptions options,
    std::function<bool(ChunkData &)> get_next_chunk) {
  uri_ = dim_uri(uri, dim);
  dim_ = dim;
  opt_state_dim_ = opt_state_dim;
  has_score_ = has_score;
  options_ = options;
  get_next_chunk_ = std::move(get_next_chunk);

  ArrowSchema schema{};
  build_schema(&schema);

  StreamState *state = new StreamState{this};

  ArrowArrayStream stream{};
  stream.get_schema = &stream_get_schema;
  stream.get_next = &stream_get_next;
  stream.get_last_error = &stream_get_last_error;
  stream.release = &stream_release;
  stream.private_data = state;

  std::filesystem::create_directories(uri_);

  int32_t mode;
  bool exists = std::filesystem::exists(uri_ + "/_versions");
  if (exists) {
    if (options_.mode == CheckpointExportOptions::Mode::kIncremental) {
      mode = LANCE_WRITE_APPEND;
    } else {
      mode = LANCE_WRITE_OVERWRITE;
    }
  } else {
    mode = LANCE_WRITE_CREATE;
  }

  LanceDataset *out_ds = nullptr;
  int32_t rc;

  if (options_.use_compression) {
    LanceWriteParams params{};
    params.data_storage_version = "next";
    rc = lance_dataset_write_with_params(uri_.c_str(), &schema, &stream, mode,
                                         &params, nullptr, &out_ds);
  } else {
    rc = lance_dataset_write(uri_.c_str(), &schema, &stream, mode, nullptr,
                             &out_ds);
  }

  if (rc != 0) {
    std::string err_msg = "LanceZeroCopyWriter flush error: ";
    err_msg += (out_ds ? "unknown error" : "failed to write dataset");
    if (out_ds)
      lance_dataset_close(out_ds);
    throw std::runtime_error(err_msg);
  } else if (out_ds) {
    lance_dataset_close(out_ds);
  }
}

void LanceZeroCopyWriter::build_schema(ArrowSchema *out) const {
  int n_cols = 2; // key + embedding
  if (has_score_)
    ++n_cols;
  if (opt_state_dim_ > 0)
    ++n_cols;
  if (options_.include_optimizer_state)
    ++n_cols;

  *out = ArrowSchema{
      .format = "+s", // struct
      .name = "",
      .metadata = nullptr,
      .flags = 0,
      .n_children = n_cols,
      .children = new ArrowSchema *[n_cols],
      .dictionary = nullptr,
      .release = no_op_release_schema,
      .private_data = nullptr,
  };

  int col = 0;

  out->children[col] = new ArrowSchema();
  make_child_schema(out->children[col], "key", options_.key_schema.c_str());
  ++col;

  thread_local std::string emb_format;
  emb_format =
      "w:" +
      std::to_string(
          dim_ *
          sizeof(float)); // we assume element is float or data stride is dim_*4
  out->children[col] = new ArrowSchema();
  make_child_schema(out->children[col], "embedding_params", emb_format.c_str());
  if (options_.use_compression) {
    out->children[col]->metadata = get_zstd_metadata();
  }
  ++col;

  if (has_score_) {
    out->children[col] = new ArrowSchema();
    make_child_schema(out->children[col], "score", "L");
    ++col;
  }

  if (opt_state_dim_ > 0) {
    thread_local std::string opt_format;
    opt_format = "w:" + std::to_string(opt_state_dim_);
    out->children[col] = new ArrowSchema();
    make_child_schema(out->children[col], "optimizer_states",
                      opt_format.c_str());
    if (options_.use_compression) {
      out->children[col]->metadata = get_zstd_metadata();
    }
    ++col;
  }

  if (options_.include_optimizer_state) {
    out->children[col] = new ArrowSchema();
    make_child_schema(out->children[col], "step", "I");
    ++col;
  }
}

void LanceZeroCopyWriter::build_array(ArrowArray *out,
                                      const ChunkData &chunk) const {
  int64_t n = static_cast<int64_t>(chunk.num_keys);
  int n_cols = 2;
  if (has_score_)
    ++n_cols;
  if (opt_state_dim_ > 0)
    ++n_cols;
  if (options_.include_optimizer_state)
    ++n_cols;

  *out = ArrowArray{
      .length = n,
      .null_count = 0,
      .offset = 0,
      .n_buffers = 1,
      .n_children = n_cols,
      .buffers = new const void *[1] { nullptr },
      .children = new ArrowArray *[n_cols],
      .dictionary = nullptr,
      .release = no_op_release_array,
      .private_data = nullptr,
  };

  int col = 0;

  out->children[col] = new ArrowArray();
  make_child_array(out->children[col], n, nullptr, chunk.keys);
  ++col;

  out->children[col] = new ArrowArray();
  make_child_array(out->children[col], n, nullptr, chunk.embeddings);
  ++col;

  if (has_score_) {
    out->children[col] = new ArrowArray();
    make_child_array(out->children[col], n, nullptr, chunk.scores);
    ++col;
  }

  if (opt_state_dim_ > 0) {
    out->children[col] = new ArrowArray();
    make_child_array(out->children[col], n, nullptr, chunk.opt_states);
    ++col;
  }

  if (options_.include_optimizer_state) {
    out->children[col] = new ArrowArray();
    make_child_array(out->children[col], n, nullptr, chunk.steps);
    ++col;
  }
}

int LanceZeroCopyWriter::stream_get_schema(ArrowArrayStream *stream,
                                           ArrowSchema *out) {
  auto *state = static_cast<StreamState *>(stream->private_data);
  state->writer->build_schema(out);
  return 0;
}

int LanceZeroCopyWriter::stream_get_next(ArrowArrayStream *stream,
                                         ArrowArray *out) {
  auto *state = static_cast<StreamState *>(stream->private_data);
  ChunkData chunk;
  if (state->writer->get_next_chunk_(chunk)) {
    state->writer->build_array(out, chunk);
  } else {
    *out = ArrowArray{
        .length = 0,
        .null_count = 0,
        .offset = 0,
        .n_buffers = 0,
        .n_children = 0,
        .buffers = nullptr,
        .children = nullptr,
        .dictionary = nullptr,
        .release = nullptr,
        .private_data = nullptr,
    };
  }
  return 0;
}

const char *LanceZeroCopyWriter::stream_get_last_error(ArrowArrayStream *) {
  return nullptr;
}

void LanceZeroCopyWriter::stream_release(ArrowArrayStream *stream) {
  auto *state = static_cast<StreamState *>(stream->private_data);
  delete state;
  stream->private_data = nullptr;
  stream->release = nullptr;
}
} // namespace se
