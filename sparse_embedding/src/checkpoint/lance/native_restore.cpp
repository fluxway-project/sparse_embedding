#include "sparse_embedding/src/checkpoint/lance/native_restore.hpp"

#include "sparse_embedding/src/checkpoint/lance/native_entry_restore.hpp"

namespace se {

template <typename ParamT, typename OptT>
void NativeRestore<ParamT, OptT>::entries(
    pro::proxy<EmbeddingTableFacade<ParamT, OptT>> &store,
    std::span<const std::string> uris, uint64_t version,
    pro::proxy<ops::PartitionerFacade> partitioner, uint32_t partition_id,
    std::string_view key_dtype, bool load_optimizer_state,
    const LanceStorageOptions &storage_options,
    std::optional<uint32_t> source_participant_index) {
  restore_native_entry_batches<uint64_t, ParamT, OptT>(
      *store,
      [&](NativeEntryBatchVisitor visitor) {
        scan_native_checkpoint_entries(
            uris, version, std::move(partitioner), partition_id, key_dtype,
            [&](const NativeEntryBatch &batch) {
              if (batch.embedding_dtype !=
                      value_dtype_name(ValueDTypeOf<ParamT>::value) ||
                  (load_optimizer_state && !batch.optimizer_dtype.empty() &&
                   batch.optimizer_dtype !=
                       value_dtype_name(ValueDTypeOf<OptT>::value)))
                throw std::invalid_argument(
                    "native checkpoint payload dtype does not match store");
              if (load_optimizer_state)
                store->validate_optimizer_state_layout(batch.dimensions,
                                                       batch.optimizer_strides);
              return visitor(batch);
            },
            load_optimizer_state, storage_options, source_participant_index);
      },
      load_optimizer_state);
}

template <typename ParamT, typename OptT>
void NativeRestore<ParamT, OptT>::pages(
    pro::proxy<EmbeddingTableFacade<ParamT, OptT>> &store,
    std::span<const std::string> uris, uint64_t version,
    uint32_t participant_index, std::string_view key_dtype,
    std::string_view optimizer_dtype,
    const LanceStorageOptions &storage_options, bool load_optimizer_state,
    uint64_t page_layout_generation) {
  const size_t entries_per_page = store->entries_per_page();
  auto source =
      pro::make_proxy<FunctionFacade<void(PageImageVisitor<ParamT, OptT>)>>(
          [&](PageImageVisitor<ParamT, OptT> visitor) {
            PageImageImport<ParamT, OptT> import{};
            for (const auto &uri : uris) {
              scan_native_page_images(
                  uri, version, participant_index, key_dtype,
                  [&](NativePageImageView &&page) {
                    if (page.entries_per_page != entries_per_page)
                      throw std::invalid_argument(
                          "page-image geometry does not match store topology");
                    if (page.embedding_dtype !=
                            value_dtype_name(ValueDTypeOf<ParamT>::value) ||
                        (load_optimizer_state &&
                         page.optimizer_dtype != optimizer_dtype))
                      throw std::invalid_argument(
                          "page-image dtype does not match store");
                    import.embedding_dim = page.embedding_dim;
                    auto &image = import.image;
                    image.physical_page_id = page.physical_page_id;
                    image.embedding_dim = page.embedding_dim;
                    image.page_slot_stride_bytes = page.page_slot_stride_bytes;
                    image.slot_ordinals.swap(page.slot_ordinals);
                    image.keys.swap(page.keys);
                    image.seq_counts.swap(page.sequences);
                    image.scores.swap(page.scores);
                    image.steps.swap(page.steps);
                    copy_native_values(std::span<const std::byte>(page.values),
                                       import.image.values);
                    if (load_optimizer_state)
                      copy_native_values(
                          std::span<const std::byte>(page.optimizer_values),
                          import.image.optimizer_states);
                    (*visitor)(import);
                    return true;
                  },
                  load_optimizer_state, storage_options);
            }
          });
  store->import_page_images(std::move(source), page_layout_generation,
                            load_optimizer_state);
}

template struct NativeRestore<float, float>;
template struct NativeRestore<float, float16_t>;
template struct NativeRestore<float, bfloat16_t>;
template struct NativeRestore<float16_t, float>;
template struct NativeRestore<float16_t, float16_t>;
template struct NativeRestore<float16_t, bfloat16_t>;
template struct NativeRestore<bfloat16_t, float>;
template struct NativeRestore<bfloat16_t, float16_t>;
template struct NativeRestore<bfloat16_t, bfloat16_t>;

} // namespace se
