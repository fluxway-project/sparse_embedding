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
  auto manifest_schema = se::lance_layout::make_manifest_schema();
  for (int64_t index = 0; index < manifest_schema.n_children; ++index) {
    if (!manifest_names.insert(manifest_schema.children[index]->name).second)
      throw std::runtime_error("duplicate Lance manifest column");
  }
  manifest_schema.release(&manifest_schema);
  for (std::string_view required :
       {"fragment_ids", "compacted_version", "commit_state", "storage_mode",
        "schema_digest", "dimensions", "key_dtype"}) {
    if (!manifest_names.contains(required))
      throw std::runtime_error("Lance snapshot manifest schema is incomplete");
  }

  for (const auto &[dtype, fmt] : {std::pair{"int32", "i"},
                                   {"uint32", "I"},
                                   {"int64", "l"},
                                   {"uint64", "L"}}) {
    auto schema = se::lance_layout::make_entries_schema(dtype);
    if (std::string_view(schema.children[1]->format) != fmt)
      throw std::runtime_error("entry key format mismatch");
    if (schema.release)
      schema.release(&schema);
  }
}
