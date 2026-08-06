#include "sparse_embedding/src/checkpoint/lance/layout_schema.hpp"

#include <set>
#include <stdexcept>
#include <string_view>

int main() {
  std::set<std::string_view> entry_names;
  for (const auto &column : se::lance_layout::kEntryColumns) {
    if (!entry_names.insert(column.name).second)
      throw std::runtime_error("duplicate Lance entry layout column");
  }
  for (std::string_view required :
       {"snapshot_id", "worker_id", "shard_id", "shard_sequence", "operation",
        "page_ordinal", "slot_ordinal"}) {
    if (!entry_names.contains(required))
      throw std::runtime_error("missing Lance entry layout column");
  }

  std::set<std::string_view> page_names;
  for (const auto &column : se::lance_layout::kPageImageColumns) {
    if (!page_names.insert(column.name).second)
      throw std::runtime_error("duplicate Lance page-image column");
  }
  if (!page_names.contains("page_payload") || !page_names.contains("checksum"))
    throw std::runtime_error("Lance page-image schema is incomplete");

  std::set<std::string_view> manifest_names;
  for (const auto &column : se::lance_layout::kManifestColumns)
    manifest_names.insert(column.name);
  for (std::string_view required :
       {"entries_lance_version", "page_images_lance_version",
        "shard_high_watermarks"}) {
    if (!manifest_names.contains(required))
      throw std::runtime_error("Lance snapshot manifest schema is incomplete");
  }
}
