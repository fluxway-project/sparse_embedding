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
       {"dim", "key", "embedding_values", "embedding_dtype",
        "optimizer_state_values", "optimizer_state_dtype", "step",
        "row_state"}) {
    if (!entry_names.contains(required))
      throw std::runtime_error("missing Lance entry layout column");
  }

  std::set<std::string_view> page_names;
  for (const auto &column : se::lance_layout::kPageImageColumns) {
    if (!page_names.insert(column.name).second)
      throw std::runtime_error("duplicate Lance page-image column");
  }
  for (std::string_view required :
       {"embedding_dim", "page_slot_stride_bytes", "slot_ordinals", "keys",
        "sequences", "embedding_values", "optimizer_state_values", "checksum",
        "row_state"}) {
    if (!page_names.contains(required))
      throw std::runtime_error("Lance page-image schema is incomplete");
  }

  std::set<std::string_view> manifest_names;
  constexpr std::array<std::string_view, 22> manifest_columns = {
      "protocol_version",     "checkpoint_name",
      "checkpoint_version",   "parent_checkpoint_version",
      "participant_count",    "checkpoint_mode",
      "schema_digest",        "fragment_digest",
      "save_attempt",         "authority_scope",
      "authority_generation", "commit_authority",
      "created_at",           "dimensions",
      "embedding_dtype",      "optimizer_dtype",
      "score_policy",         "reference_id",
      "fragment_ids",         "compacted_version",
      "commit_state",         "storage_mode"};
  manifest_names.insert(manifest_columns.begin(), manifest_columns.end());
  for (std::string_view required :
       {"protocol_version", "fragment_ids", "compacted_version", "commit_state",
        "storage_mode", "schema_digest", "dimensions"}) {
    if (!manifest_names.contains(required))
      throw std::runtime_error("Lance snapshot manifest schema is incomplete");
  }
}
