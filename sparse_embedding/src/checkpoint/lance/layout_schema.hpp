#pragma once

#ifndef SPARSE_EMBEDDING_LANCE_LAYOUT_SCHEMA_HPP_
#define SPARSE_EMBEDDING_LANCE_LAYOUT_SCHEMA_HPP_

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

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

#endif // ARROW_C_DATA_INTERFACE

namespace se::lance_layout {

struct ColumnDef {
  const char *name;
  const char *arrow_format;
};

inline constexpr std::array<ColumnDef, 15> kEntryColumns = {{
    {"dim", "I"},
    {"key", "L"},
    {"embedding_values", "z"},
    {"embedding_dtype", "u"},
    {"optimizer_state_values", "z"},
    {"optimizer_state_dtype", "u"},
    {"optimizer_state_stride", "I"},
    {"step", "I"},
    {"score", "L"},
    {"sequence", "L"},
    {"physical_page_id", "L"},
    {"page_slot", "I"},
    {"row_state", "u"},
    {"source_participant_index", "I"},
    {"mutation_sequence", "L"},
}};

inline constexpr std::array<ColumnDef, 17> kPageImageColumns = {{
    {"source_participant_index", "I"},
    {"embedding_dim", "I"},
    {"physical_page_id", "L"},
    {"page_slot_stride_bytes", "I"},
    {"embedding_dtype", "u"},
    {"slot_ordinals", "z"},
    {"keys", "z"},
    {"sequences", "z"},
    {"embedding_values", "z"},
    {"scores", "z"},
    {"optimizer_state_values", "z"},
    {"optimizer_state_dtype", "u"},
    {"optimizer_state_stride", "I"},
    {"steps", "z"},
    {"checksum", "u"},
    {"row_state", "u"},
    {"entries_per_page", "I"},
}};

inline constexpr std::string_view kEntriesDataset = "entries.lance";
inline constexpr std::string_view kPageImagesDataset = "page_images.lance";
inline constexpr std::string_view kSnapshotManifestDataset =
    "snapshot_manifest.lance";

inline void release_arrow_schema(ArrowSchema *schema) noexcept {
  if (!schema || !schema->release)
    return;
  for (int64_t i = 0; i < schema->n_children; ++i) {
    if (schema->children && schema->children[i]) {
      if (schema->children[i]->release) {
        schema->children[i]->release(schema->children[i]);
      }
      delete schema->children[i];
      schema->children[i] = nullptr;
    }
  }
  delete[] schema->children;
  schema->children = nullptr;
  schema->release = nullptr;
}

inline ArrowSchema *make_leaf_schema(const char *name, const char *format) {
  return new ArrowSchema{
      .format = format,
      .name = name,
      .metadata = nullptr,
      .flags = 0,
      .n_children = 0,
      .children = nullptr,
      .dictionary = nullptr,
      .release =
          [](ArrowSchema *self) noexcept {
            if (self) {
              self->release = nullptr;
            }
          },
      .private_data = nullptr,
  };
}

inline ArrowSchema *make_list_schema(const char *name, const char *child_name,
                                     const char *child_format) {
  auto *child = make_leaf_schema(child_name, child_format);
  auto **children = new ArrowSchema *[1] { child };
  return new ArrowSchema{
      .format = "+l",
      .name = name,
      .metadata = nullptr,
      .flags = 0,
      .n_children = 1,
      .children = children,
      .dictionary = nullptr,
      .release =
          [](ArrowSchema *self) noexcept {
            if (!self || !self->release)
              return;
            if (self->children && *self->children) {
              if ((*self->children)->release) {
                (*self->children)->release(*self->children);
              }
              delete *self->children;
            }
            delete[] self->children;
            self->children = nullptr;
            self->release = nullptr;
          },
      .private_data = nullptr,
  };
}

template <size_t N>
inline ArrowSchema make_columns_schema(const std::array<ColumnDef, N> &columns,
                                       const char *metadata = nullptr) {
  auto **children = new ArrowSchema *[N];
  for (size_t i = 0; i < N; ++i) {
    children[i] = make_leaf_schema(columns[i].name, columns[i].arrow_format);
  }
  return ArrowSchema{
      .format = "+s",
      .name = "",
      .metadata = metadata,
      .flags = 0,
      .n_children = static_cast<int64_t>(N),
      .children = children,
      .dictionary = nullptr,
      .release = &release_arrow_schema,
      .private_data = nullptr,
  };
}

inline const char *arrow_format_for_key_dtype(std::string_view dtype) {
  if (dtype == "int32")
    return "i";
  if (dtype == "uint32")
    return "I";
  if (dtype == "int64")
    return "l";
  if (dtype == "uint64")
    return "L";
  throw std::invalid_argument("unsupported key dtype: " + std::string(dtype));
}

inline ArrowSchema make_entries_schema(std::string_view key_dtype = "uint64",
                                       const char *metadata = nullptr) {
  auto columns = kEntryColumns;
  const auto key_column =
      std::ranges::find_if(columns, [](const ColumnDef &column) {
        return std::string_view(column.name) == "key";
      });
  if (key_column == columns.end())
    throw std::logic_error("entry schema has no key column");
  key_column->arrow_format = arrow_format_for_key_dtype(key_dtype);
  return make_columns_schema(columns, metadata);
}

inline ArrowSchema make_page_images_schema(const char *metadata = nullptr) {
  return make_columns_schema(kPageImageColumns, metadata);
}

inline ArrowSchema make_manifest_schema(const char *metadata = nullptr) {
  constexpr size_t kManifestFieldCount = 32;
  std::array<ArrowSchema *, kManifestFieldCount> fields{};
  size_t field_count = 0;
  const auto append = [&](ArrowSchema *field) {
    if (field_count == fields.size()) {
      release_arrow_schema(field);
      delete field;
      throw std::logic_error("manifest schema has too many fields");
    }
    fields[field_count++] = field;
  };
  const auto append_leaf = [&](const char *name, const char *format) {
    append(make_leaf_schema(name, format));
  };
  const auto append_list = [&](const char *name, const char *item_name,
                               const char *item_format) {
    append(make_list_schema(name, item_name, item_format));
  };
  append_leaf("checkpoint_name", "u");
  append_leaf("checkpoint_version", "L");
  append_leaf("parent_checkpoint_version", "L");
  append_leaf("participant_count", "I");
  append_leaf("checkpoint_mode", "u");
  append_leaf("schema_digest", "u");
  append_leaf("fragment_digest", "u");
  append_leaf("save_attempt", "u");
  append_leaf("authority_scope", "u");
  append_leaf("authority_generation", "L");
  append_leaf("commit_authority", "u");
  append_leaf("created_at", "tsu:UTC");
  append_list("dimensions", "item", "I");
  append_leaf("embedding_dtype", "u");
  append_leaf("optimizer_dtype", "u");
  append_leaf("score_policy", "u");
  append_leaf("key_dtype", "u");
  append_leaf("reference_id", "u");
  append_list("fragment_ids", "item", "u");
  append_leaf("compacted_version", "L");
  append_leaf("commit_state", "u");
  append_leaf("storage_mode", "u");
  append_leaf("format_version", "I");
  append_leaf("parent_checkpoint_name", "u");
  append_leaf("parent_reference_id", "u");
  append_list("participant_watermarks", "item", "L");
  append_list("fragment_descriptors", "item", "u");
  append_leaf("include_page_images", "b");
  append_list("participant_page_capacities", "item", "I");
  append_list("participant_page_datasets", "item", "u");
  append_list("participant_page_versions", "item", "L");
  append_list("participant_page_layout_generations", "item", "L");
  if (field_count != fields.size()) {
    for (size_t index = 0; index < field_count; ++index) {
      release_arrow_schema(fields[index]);
      delete fields[index];
    }
    throw std::logic_error("manifest schema field count is incomplete");
  }
  auto **children = new ArrowSchema *[fields.size()];
  for (size_t index = 0; index < fields.size(); ++index)
    children[index] = fields[index];

  return ArrowSchema{
      .format = "+s",
      .name = "",
      .metadata = metadata,
      .flags = 0,
      .n_children = static_cast<int64_t>(kManifestFieldCount),
      .children = children,
      .dictionary = nullptr,
      .release = &release_arrow_schema,
      .private_data = nullptr,
  };
}

} // namespace se::lance_layout

#endif // SPARSE_EMBEDDING_LANCE_LAYOUT_SCHEMA_HPP_
