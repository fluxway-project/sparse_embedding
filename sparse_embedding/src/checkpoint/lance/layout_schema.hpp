#pragma once

#ifndef SPARSE_EMBEDDING_LANCE_LAYOUT_SCHEMA_HPP_
#define SPARSE_EMBEDDING_LANCE_LAYOUT_SCHEMA_HPP_

#include <array>
#include <cstdint>
#include <string_view>

namespace se::lance_layout {

inline constexpr uint32_t kSchemaVersion = 2;

struct Column {
  std::string_view name;
  std::string_view arrow_format;
};

inline constexpr std::array kEntryColumns{
    Column{"schema_version", "I"},
    Column{"snapshot_id", "L"},
    Column{"page_layout_generation", "L"},
    Column{"worker_id", "I"},
    Column{"shard_id", "I"},
    Column{"shard_sequence", "L"},
    Column{"operation", "C"},
    Column{"key", "L"},
    Column{"seq_count", "L"},
    Column{"tier", "C"},
    Column{"cluster_id", "I"},
    Column{"page_ordinal", "L"},
    Column{"slot_ordinal", "I"},
    Column{"embedding_params", "z"},
    Column{"score", "L"},
    Column{"optimizer_states", "z"},
    Column{"step", "I"},
};

inline constexpr std::array kPageImageColumns{
    Column{"schema_version", "I"},
    Column{"snapshot_id", "L"},
    Column{"page_layout_generation", "L"},
    Column{"worker_id", "I"},
    Column{"shard_id", "I"},
    Column{"page_ordinal", "L"},
    Column{"tier", "C"},
    Column{"entry_count", "I"},
    Column{"vector_stride", "I"},
    Column{"checksum", "L"},
    Column{"page_payload", "Z"},
};

inline constexpr std::array kManifestColumns{
    Column{"schema_version", "I"},
    Column{"snapshot_id", "L"},
    Column{"page_layout_generation", "L"},
    Column{"worker_count", "I"},
    Column{"shard_count", "I"},
    Column{"entries_lance_version", "L"},
    Column{"page_images_lance_version", "L"},
    Column{"shard_high_watermarks", "Z"},
};

inline constexpr std::string_view kEntriesDataset = "entries";
inline constexpr std::string_view kPageImagesDataset = "page_images";
inline constexpr std::string_view kSnapshotManifestDataset =
    "snapshot_manifest";

} // namespace se::lance_layout

#endif
