#pragma once

#ifndef SPARSE_EMBEDDING_LANCE_LAYOUT_SCHEMA_HPP_
#define SPARSE_EMBEDDING_LANCE_LAYOUT_SCHEMA_HPP_

#include <array>
#include <cstdint>
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
    {"last_updated_checkpoint_version", "L"},
    {"source_participant_index", "I"},
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
    {"last_updated_checkpoint_version", "L"},
}};

inline constexpr std::string_view kEntriesDataset = "entries";
inline constexpr std::string_view kPageImagesDataset = "page_images";
inline constexpr std::string_view kSnapshotManifestDataset =
    "snapshot_manifest";

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
            if (self->children && self->children[0]) {
              if (self->children[0]->release) {
                self->children[0]->release(self->children[0]);
              }
              delete self->children[0];
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

inline ArrowSchema make_entries_schema(const char *metadata = nullptr) {
  return make_columns_schema(kEntryColumns, metadata);
}

inline ArrowSchema make_page_images_schema(const char *metadata = nullptr) {
  return make_columns_schema(kPageImageColumns, metadata);
}

inline ArrowSchema make_manifest_schema(const char *metadata = nullptr) {
  constexpr size_t kManifestFieldCount = 22;
  auto **children = new ArrowSchema *[kManifestFieldCount];
  children[0] = make_leaf_schema("protocol_version", "I");
  children[1] = make_leaf_schema("checkpoint_name", "u");
  children[2] = make_leaf_schema("checkpoint_version", "L");
  children[3] = make_leaf_schema("parent_checkpoint_version", "L");
  children[4] = make_leaf_schema("participant_count", "I");
  children[5] = make_leaf_schema("checkpoint_mode", "u");
  children[6] = make_leaf_schema("schema_digest", "u");
  children[7] = make_leaf_schema("fragment_digest", "u");
  children[8] = make_leaf_schema("save_attempt", "u");
  children[9] = make_leaf_schema("authority_scope", "u");
  children[10] = make_leaf_schema("authority_generation", "L");
  children[11] = make_leaf_schema("commit_authority", "u");
  children[12] = make_leaf_schema("created_at", "tsu:UTC");
  children[13] = make_list_schema("dimensions", "item", "I");
  children[14] = make_leaf_schema("embedding_dtype", "u");
  children[15] = make_leaf_schema("optimizer_dtype", "u");
  children[16] = make_leaf_schema("score_policy", "u");
  children[17] = make_leaf_schema("reference_id", "u");
  children[18] = make_list_schema("fragment_ids", "item", "u");
  children[19] = make_leaf_schema("compacted_version", "L");
  children[20] = make_leaf_schema("commit_state", "u");
  children[21] = make_leaf_schema("storage_mode", "u");

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
