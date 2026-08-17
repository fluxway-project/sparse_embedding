namespace se::py::detail {

inline void EmbeddingStoreWrapper::save_checkpoint(
    const std::string &uri, const std::string &mode, uint64_t min_score,
    bool include_optimizer_state, bool use_compression) {
  dispatch_types([&]<typename P, typename O>() {
    CheckpointOpsImpl<P, O>::save_checkpoint(*this, uri, mode, min_score,
                                             include_optimizer_state,
                                             use_compression);
  });
}

inline void EmbeddingStoreWrapper::load_checkpoint(
    const std::string &uri, uint64_t version, bool clear,
    bool load_optimizer_state, nb::object filter_fn, uint32_t target_shard_id,
    size_t batch_size) {
  dispatch_types([&]<typename P, typename O>() {
    CheckpointOpsImpl<P, O>::load_checkpoint(*this, uri, version, clear,
                                             load_optimizer_state, filter_fn,
                                             target_shard_id, batch_size);
  });
}

inline nb::dict EmbeddingStoreWrapper::write_native_fragment(
    const std::string &uri, uint32_t participant_index,
    const std::vector<uint32_t> &dimensions, const std::string &embedding_dtype,
    const std::string &optimizer_dtype, const std::string &score_policy,
    uint32_t protocol_version, const std::string &schema_digest) {
  return dispatch_types([&]<typename P, typename O>() {
    return CheckpointOpsImpl<P, O>::write_native_fragment(
        *this, uri, participant_index, dimensions, embedding_dtype,
        optimizer_dtype, score_policy, protocol_version, schema_digest);
  });
}

inline void EmbeddingStoreWrapper::restore_native_entries(
    const std::string &uri, uint64_t version, nb::object selected_keys) {
  dispatch_types([&]<typename P, typename O>() {
    CheckpointOpsImpl<P, O>::restore_native_entries(*this, uri, version,
                                                     selected_keys);
  });
}

inline void EmbeddingStoreWrapper::restore_native_page_images(
    const std::string &uri, uint64_t version, uint32_t participant_index) {
  dispatch_types([&]<typename P, typename O>() {
    CheckpointOpsImpl<P, O>::restore_native_page_images(
        *this, uri, version, participant_index);
  });
}

inline void EmbeddingStoreWrapper::set_checkpoint_factories(
    se::CheckpointWriterFactory wf, se::CheckpointReaderFactory rf) {
  checkpoint_writer_factory_ = std::move(wf);
  checkpoint_reader_factory_ = std::move(rf);
}

} // namespace se::py::detail
