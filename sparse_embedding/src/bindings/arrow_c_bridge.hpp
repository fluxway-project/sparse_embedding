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
//
// Arrow C Data Interface bridge for zero-copy export of embedding data.
//
// Constructs ArrowArray / ArrowSchema structs directly from owned C++ vectors.
// The ArrowArray::release callback transfers memory lifetime to the consumer
// (PyArrow), eliminating all copies in the C++ → Python → Lance pipeline.
//
// Reference: https://arrow.apache.org/docs/format/CDataInterface.html

#pragma once

#ifndef SPARSE_EMBEDDING_ARROW_C_BRIDGE_HPP_
#define SPARSE_EMBEDDING_ARROW_C_BRIDGE_HPP_

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

// ============================================================================
// Arrow C Data Interface struct definitions (from the Arrow spec).
// Self-contained — no dependency on any Arrow C++ library.
// ============================================================================

#ifndef ARROW_C_DATA_INTERFACE
#define ARROW_C_DATA_INTERFACE

struct ArrowSchema {
  const char *format;
  const char *name;
  const char *metadata;
  int64_t flags;
  int64_t n_children;
  struct ArrowSchema **children;
  struct ArrowSchema *dictionary;
  void (*release)(struct ArrowSchema *);
  void *private_data;
};

struct ArrowArray {
  int64_t length;
  int64_t null_count;
  int64_t offset;
  int64_t n_buffers;
  int64_t n_children;
  const void **buffers;
  struct ArrowArray **children;
  struct ArrowArray *dictionary;
  void (*release)(struct ArrowArray *);
  void *private_data;
};

struct ArrowArrayStream {
  int (*get_schema)(struct ArrowArrayStream *, struct ArrowSchema *out);
  int (*get_next)(struct ArrowArrayStream *, struct ArrowArray *out);
  const char *(*get_last_error)(struct ArrowArrayStream *);
  void (*release)(struct ArrowArrayStream *);
  void *private_data;
};

#endif // ARROW_C_DATA_INTERFACE

namespace se::arrow {

constexpr int64_t kArrowFlagNullable = 2;

// ============================================================================
// SchemaBuilder: constructs ArrowSchema trees for our checkpoint schemas.
//
// Two schemas are supported:
//   - EmbeddingParams:  struct<slot_id: uint32, key: int64, embedding_params:
//   list<f32>>
//   - OptimizerStates:   struct<slot_id: uint32, key: int64, optimizer_states:
//   list<f32>>
// ============================================================================

namespace detail {

inline ArrowSchema *make_leaf_schema(const char *format, const char *name) {
  auto *s = new ArrowSchema;
  s->format = format;
  s->name = name;
  s->metadata = nullptr;
  s->flags = kArrowFlagNullable;
  s->n_children = 0;
  s->children = nullptr;
  s->dictionary = nullptr;
  s->private_data = nullptr;
  s->release = [](ArrowSchema *self) { delete self; };
  return s;
}

inline ArrowSchema *make_list_schema(const char *name) {
  auto *item = make_leaf_schema("f", "item"); // float32

  auto *s = new ArrowSchema;
  s->format = "+l"; // list
  s->name = name;
  s->metadata = nullptr;
  s->flags = kArrowFlagNullable;
  s->n_children = 1;
  s->children = new ArrowSchema *[1] { item };
  s->dictionary = nullptr;
  s->private_data = nullptr;
  s->release = [](ArrowSchema *self) {
    for (int64_t i = 0; i < self->n_children; ++i) {
      if (self->children[i] && self->children[i]->release) {
        self->children[i]->release(self->children[i]);
      }
    }
    delete[] self->children;
    delete self;
  };
  return s;
}

} // namespace detail

// Build record batch schema with required (dim, key, list_col) and optional
// uint32 "step" column. Per OCP, callers that don't need per-ID step pass
// with_step=false (default) and observe the legacy 3-column layout.
inline void build_record_batch_schema(ArrowSchema *out,
                                      const char *list_col_name,
                                      const char *key_format,
                                      bool with_step = false) {
  auto *dim = detail::make_leaf_schema("I", "dim"); // uint32
  auto *key = detail::make_leaf_schema(key_format, "key");
  auto *list_col = detail::make_list_schema(list_col_name);
  ArrowSchema *step =
      with_step ? detail::make_leaf_schema("I", "step") : nullptr;

  const int64_t n_children = with_step ? 4 : 3;
  auto **children = new ArrowSchema *[static_cast<size_t>(n_children)];
  children[0] = dim;
  children[1] = key;
  children[2] = list_col;
  if (with_step)
    children[3] = step;

  out->format = "+s"; // struct (record batch)
  out->name = nullptr;
  out->metadata = nullptr;
  out->flags = kArrowFlagNullable;
  out->n_children = n_children;
  out->children = children;
  out->dictionary = nullptr;
  out->private_data = nullptr;
  out->release = [](ArrowSchema *self) {
    for (int64_t i = 0; i < self->n_children; ++i) {
      if (self->children[i] && self->children[i]->release) {
        self->children[i]->release(self->children[i]);
      }
    }
    delete[] self->children;
    self->release = nullptr;
  };
}

// ============================================================================
// ArrayBuilder: constructs ArrowArray trees from owned C++ vectors.
//
// Takes ownership of the vectors via std::move. The release callback frees
// the heap memory when PyArrow's GC collects the RecordBatch.
// ============================================================================

namespace detail {

struct LeafArrayCtx {
  const void *buffers[2]; // [null_bitmap, data]
  virtual ~LeafArrayCtx() = default;
};

template <typename T> struct OwnedLeafArrayCtx : LeafArrayCtx {
  std::vector<T> data;
  explicit OwnedLeafArrayCtx(std::vector<T> &&d) : data(std::move(d)) {
    buffers[0] = nullptr; // no null bitmap
    buffers[1] = this->data.data();
  }
};

struct ListArrayCtx {
  std::vector<int32_t> offsets;
  const void *buffers[2]; // [null_bitmap, offsets]

  explicit ListArrayCtx(std::vector<int32_t> &&o) : offsets(std::move(o)) {
    buffers[0] = nullptr;
    buffers[1] = this->offsets.data();
  }
};

// Holds up to 4 child arrays: [dim, key, list_col, step].
// step slot remains nullptr when the schema does not include it.
struct RecordBatchCtx {
  const void *buffers[1]; // [null_bitmap]
  ArrowArray *children[4];

  RecordBatchCtx() {
    buffers[0] = nullptr;
    children[0] = nullptr;
    children[1] = nullptr;
    children[2] = nullptr;
    children[3] = nullptr;
  }
};

template <typename T> ArrowArray *make_leaf_array(std::vector<T> &&data) {
  auto *ctx = new OwnedLeafArrayCtx<T>(std::move(data));
  auto *arr = new ArrowArray;
  arr->length = static_cast<int64_t>(ctx->data.size());
  arr->null_count = 0;
  arr->offset = 0;
  arr->n_buffers = 2;
  arr->n_children = 0;
  arr->buffers = ctx->buffers;
  arr->children = nullptr;
  arr->dictionary = nullptr;
  arr->private_data = ctx;
  arr->release = [](ArrowArray *self) {
    delete static_cast<LeafArrayCtx *>(self->private_data);
    delete self;
  };
  return arr;
}

inline ArrowArray *make_list_array(std::vector<int32_t> &&offsets,
                                   ArrowArray *values_child) {
  auto *ctx = new ListArrayCtx(std::move(offsets));
  auto *arr = new ArrowArray;
  // list length = number of offsets - 1
  arr->length = static_cast<int64_t>(ctx->offsets.size()) - 1;
  arr->null_count = 0;
  arr->offset = 0;
  arr->n_buffers = 2;
  arr->n_children = 1;
  arr->buffers = ctx->buffers;
  arr->children = new ArrowArray *[1] { values_child };
  arr->dictionary = nullptr;
  arr->private_data = ctx;
  arr->release = [](ArrowArray *self) {
    for (int64_t i = 0; i < self->n_children; ++i) {
      if (self->children[i] && self->children[i]->release) {
        self->children[i]->release(self->children[i]);
      }
    }
    delete[] self->children;
    delete static_cast<ListArrayCtx *>(self->private_data);
    delete self;
  };
  return arr;
}

} // namespace detail

// ============================================================================
// build_record_batch_array: construct the top-level struct ArrowArray.
//
// Takes ownership of keys, slot_ids, list offsets, and flat values.
// The list column uses (offsets, flat_values) to represent variable-length
// embedding params or optimizer states.
//
// After this call, all input vectors are in a moved-from state.
// ============================================================================

// Build the top-level struct array. When `steps` is non-empty, an extra
// uint32 child column is appended (must match the schema built with
// with_step=true). Empty steps preserve the legacy 3-column layout.
template <typename KeyT>
inline void build_record_batch_array(ArrowArray *out,
                                     std::vector<uint32_t> &&dims,
                                     std::vector<KeyT> &&keys,
                                     std::vector<int32_t> &&offsets,
                                     std::vector<float> &&flat_values,
                                     std::vector<uint32_t> &&steps = {}) {
  int64_t n = static_cast<int64_t>(keys.size());
  const bool with_step = !steps.empty();

  auto *dim_arr = detail::make_leaf_array(std::move(dims));
  auto *key_arr = detail::make_leaf_array(std::move(keys));
  auto *val_child = detail::make_leaf_array(std::move(flat_values));
  auto *list_arr = detail::make_list_array(std::move(offsets), val_child);
  ArrowArray *step_arr =
      with_step ? detail::make_leaf_array(std::move(steps)) : nullptr;

  auto *ctx = new detail::RecordBatchCtx();
  ctx->children[0] = dim_arr;
  ctx->children[1] = key_arr;
  ctx->children[2] = list_arr;
  if (with_step)
    ctx->children[3] = step_arr;

  out->length = n;
  out->null_count = 0;
  out->offset = 0;
  out->n_buffers = 1;
  out->n_children = with_step ? 4 : 3;
  out->buffers = ctx->buffers;
  out->children = ctx->children;
  out->dictionary = nullptr;
  out->private_data = ctx;
  out->release = [](ArrowArray *self) {
    auto *c = static_cast<detail::RecordBatchCtx *>(self->private_data);
    for (int i = 0; i < 4; ++i) {
      if (c->children[i] && c->children[i]->release) {
        c->children[i]->release(c->children[i]);
      }
    }
    delete c;
    self->release = nullptr;
  };
}

// ============================================================================
// build_batch_from_export: high-level helper called from the binding layer.
//
// Takes raw export vectors (keys with slot_id encoded in high bits, flat vals)
// and produces a ready-to-import (ArrowArray*, ArrowSchema*) pair.
//
// Computes:
//   - slot_id column by extracting (key >> 48) & 0xFFFF
//   - list offsets from (n, dim) where dim = vals.size() / keys.size()
// ============================================================================

struct ArrowBatchPair {
  ArrowArray array;
  ArrowSchema schema;
};

template <typename KeyT> inline const char *arrow_key_format() {
  if constexpr (std::is_same_v<KeyT, int32_t>)
    return "i";
  if constexpr (std::is_same_v<KeyT, uint32_t>)
    return "I";
  if constexpr (std::is_same_v<KeyT, int64_t>)
    return "l";
  if constexpr (std::is_same_v<KeyT, uint64_t>)
    return "L";
  throw std::invalid_argument("Unsupported KeyT for Arrow schema");
}

// Build an embedding record batch.
// Note on PyArrow Compatibility: We hardcode `std::vector<float>` instead of a
// generic `ParamT`. PyArrow does not natively support bfloat16. To ensure the
// exported C Data Interface struct can be parsed in Python, any bfloat16
// backend buffers must be cast to float32 before reaching here.
template <typename KeyT>
inline ArrowBatchPair build_embedding_batch(size_t emb_dim,
                                            std::vector<KeyT> &&keys,
                                            std::vector<float> &&vals) {
  size_t n = keys.size();
  size_t dim = n > 0 ? vals.size() / n : 0;

  std::vector<uint32_t> dims(n, static_cast<uint32_t>(emb_dim));

  std::vector<int32_t> offsets(n + 1);
  for (size_t i = 0; i <= n; ++i) {
    offsets[i] = static_cast<int32_t>(i * dim);
  }

  ArrowBatchPair pair;
  build_record_batch_schema(&pair.schema, "embedding_params",
                            arrow_key_format<KeyT>());
  build_record_batch_array(&pair.array, std::move(dims), std::move(keys),
                           std::move(offsets), std::move(vals));
  return pair;
}

// Build an opt_state record batch with an optional per-ID step column.
// Empty `steps` produces the legacy 3-column layout (used for optimizers
// that do not track step state, e.g. SGD).
template <typename KeyT>
inline ArrowBatchPair build_opt_state_batch(
    size_t emb_dim, std::vector<KeyT> &&keys, std::vector<float> &&vals,
    std::vector<float> &&opt_states, std::vector<uint32_t> &&steps = {}) {
  size_t n = keys.size();
  size_t opt_dim = n > 0 ? opt_states.size() / n : 0;
  const bool with_step = !steps.empty();

  std::vector<uint32_t> dims(n, static_cast<uint32_t>(emb_dim));

  std::vector<int32_t> offsets(n + 1);
  for (size_t i = 0; i <= n; ++i) {
    offsets[i] = static_cast<int32_t>(i * opt_dim);
  }

  ArrowBatchPair pair;
  build_record_batch_schema(&pair.schema, "optimizer_states",
                            arrow_key_format<KeyT>(), with_step);
  build_record_batch_array(&pair.array, std::move(dims), std::move(keys),
                           std::move(offsets), std::move(opt_states),
                           std::move(steps));
  (void)vals;
  return pair;
}

} // namespace se::arrow

#endif // SPARSE_EMBEDDING_ARROW_C_BRIDGE_HPP_
