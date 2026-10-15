namespace se::py::detail {

#if defined(SE_ENABLE_LANCE_C)
inline nb::dict EmbeddingStoreWrapper::write_native_fragment(
    const std::string &uri, uint32_t participant_index,
    const std::vector<uint32_t> &dimensions, const std::string &embedding_dtype,
    const std::string &optimizer_dtype, const std::string &score_policy,
    const std::string &key_dtype, const std::string &schema_digest,
    bool include_page_images, bool incremental_mode,
    const std::string &expected_baseline, const std::string &page_dataset_uri,
    uint64_t page_dataset_version, uint64_t expected_page_layout_generation) {
  validate_key_dtype(key_type_enum_, se::parse_key_dtype(key_dtype));
  return dispatch_types([&]<typename P, typename O>() {
    return CheckpointOpsImpl<P, O>::write_native_fragment(
        *this, uri, participant_index, dimensions, embedding_dtype,
        optimizer_dtype, score_policy, key_dtype, schema_digest,
        include_page_images, incremental_mode, expected_baseline,
        page_dataset_uri, page_dataset_version,
        expected_page_layout_generation);
  });
}

inline void EmbeddingStoreWrapper::restore_native_partition_entries(
    const std::vector<std::string> &uris, uint64_t version,
    uint64_t partition_count, uint64_t partition_id,
    const std::string &key_dtype, const std::string &assignment_name,
    const std::unordered_map<std::string, std::string> &assignment_parameters,
    const se::LanceStorageOptions &storage_options, bool load_optimizer_state,
    std::optional<uint32_t> source_participant_index) {
  validate_key_dtype(key_type_enum_, se::parse_key_dtype(key_dtype));
  dispatch_types([&]<typename P, typename O>() {
    CheckpointOpsImpl<P, O>::restore_native_partition_entries(
        *this, uris, version, partition_count, partition_id, key_dtype,
        assignment_name, assignment_parameters, storage_options,
        load_optimizer_state, source_participant_index);
  });
}

inline void EmbeddingStoreWrapper::restore_native_page_images(
    const std::vector<std::string> &uris, uint64_t version,
    uint32_t participant_index, const std::string &key_dtype,
    const se::LanceStorageOptions &storage_options, bool load_optimizer_state,
    uint64_t page_layout_generation) {
  validate_key_dtype(key_type_enum_, se::parse_key_dtype(key_dtype));
  dispatch_types([&]<typename P, typename O>() {
    CheckpointOpsImpl<P, O>::restore_native_page_images(
        *this, uris, version, participant_index, key_dtype, storage_options,
        load_optimizer_state, page_layout_generation);
  });
}
#endif

} // namespace se::py::detail
