#pragma once

#include "sparse_embedding/src/checkpoint/lance/native_fragment_writer.hpp"
#include "sparse_embedding/src/interface/embedding_table_interface.hpp"

namespace se {

template <typename ParamT, typename OptT> struct NativeRestore {
  static void entries(pro::proxy<EmbeddingTableFacade<ParamT, OptT>> &store,
                      std::span<const std::string> uris, uint64_t version,
                      pro::proxy<ops::PartitionerFacade> partitioner,
                      uint32_t partition_id, std::string_view key_dtype,
                      bool load_optimizer_state,
                      const LanceStorageOptions &storage_options,
                      std::optional<uint32_t> source_participant_index);

  static void pages(pro::proxy<EmbeddingTableFacade<ParamT, OptT>> &store,
                    std::span<const std::string> uris, uint64_t version,
                    uint32_t participant_index, std::string_view key_dtype,
                    std::string_view optimizer_dtype,
                    const LanceStorageOptions &storage_options,
                    bool load_optimizer_state, uint64_t page_layout_generation);
};

} // namespace se
