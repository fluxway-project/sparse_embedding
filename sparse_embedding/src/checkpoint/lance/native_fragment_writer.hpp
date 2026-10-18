#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "sparse_embedding/src/checkpoint/checkpoint_stream_statistics.hpp"
#include "sparse_embedding/src/checkpoint/lance/arrow/c/abi.h"
#include "sparse_embedding/src/checkpoint/lance/native_entry_batch.hpp"
#include "sparse_embedding/src/checkpoint/lance/storage_options.hpp"
#include "sparse_embedding/src/ops/partitioner.hpp"

namespace se {

struct CheckpointMutation;

struct NativeFragmentArtifact {
  std::string locator;
  uint64_t entries_version = 0;
  uint64_t page_images_version = 0;
  uint64_t page_layout_generation = 0;
  uint64_t entry_count = 0;
  uint64_t page_count = 0;
  uint64_t physical_bytes = 0;
  uint32_t entries_per_page = 0;
  std::string schema_digest;
  std::string content_digest;
};

struct NativeCompactionArtifact {
  std::string locator;
  uint64_t target_version = 0;
  uint64_t inserted_rows = 0;
  uint64_t updated_rows = 0;
  uint64_t deleted_rows = 0;
  std::string content_digest;
};

struct PipelinedWriterState;

class NativeFragmentWriter {
  std::unique_ptr<PipelinedWriterState> state_;

  void start_entries();
  void start_pages();

public:
  NativeFragmentWriter();
  ~NativeFragmentWriter();
  NativeFragmentWriter(const NativeFragmentWriter &) = delete;
  NativeFragmentWriter &operator=(const NativeFragmentWriter &) = delete;
  NativeFragmentWriter(NativeFragmentWriter &&) noexcept;
  NativeFragmentWriter &operator=(NativeFragmentWriter &&) noexcept;

  static void release_schema(ArrowSchema *schema);
  static void release_array(ArrowArray *array);

  void begin_fragment(std::string_view staging_uri, uint32_t participant_index,
                      std::span<const uint32_t> dimensions,
                      std::string_view embedding_dtype,
                      std::string_view optimizer_dtype,
                      std::string_view score_policy, std::string_view key_dtype,
                      std::string_view schema_digest, bool has_score,
                      bool include_page_images, uint32_t entries_per_page,
                      std::string_view page_dataset_uri,
                      uint64_t page_dataset_version = 0);

  void write_page_image(
      size_t embedding_dim, uint64_t physical_page_id,
      uint32_t page_slot_stride_bytes, size_t embedding_row_bytes,
      std::span<const uint32_t> slots, std::span<const uint64_t> keys,
      std::span<const uint64_t> sequences, std::span<const std::byte> values,
      std::span<const uint64_t> scores, std::span<const std::byte> optimizer,
      size_t optimizer_stride, size_t optimizer_itemsize,
      std::span<const uint32_t> steps);

  void write_page_tombstone(size_t embedding_dim, uint64_t physical_page_id,
                            uint32_t page_slot_stride_bytes,
                            uint32_t entries_per_page);

  void set_incremental_mutations(std::span<const CheckpointMutation> mutations);
  void write_tombstone(uint32_t dimension, uint64_t key,
                       uint64_t mutation_sequence);

  NativeFragmentArtifact finalize();
};

// Validate the chain and, when a target assignment is supplied, prove that all
// historical identities retain their source participant. An absent target
// performs chain validation only and returns false (no layout proof).
bool validate_native_checkpoint_chain(
    std::span<const std::vector<std::string>> source_uris,
    std::span<const std::vector<uint64_t>> participant_watermarks,
    std::span<const uint32_t> dimensions, std::string_view key_dtype,
    const LanceStorageOptions &storage_options,
    pro::proxy<se::ops::PartitionerFacade> target_partitioner = {},
    uint32_t target_partition_count = 0);

NativeFragmentArtifact compact_native_checkpoint(
    std::string_view target_uri, std::span<const std::string> source_uris,
    uint32_t participant_index, std::span<const uint32_t> dimensions,
    std::string_view embedding_dtype, std::string_view optimizer_dtype,
    std::string_view score_policy, std::string_view key_dtype,
    std::string_view schema_digest, bool include_page_images,
    uint32_t entries_per_page, std::string_view page_dataset_uri,
    const LanceStorageOptions &storage_options);

NativeCompactionArtifact compact_native_lance_sources(
    std::string_view target_uri, std::span<const std::string> source_uris,
    const LanceStorageOptions &storage_options = LanceStorageOptions{});

struct NativePageImageView {
  uint32_t entries_per_page = 0;
  uint32_t embedding_dim = 0;
  uint64_t physical_page_id = 0;
  uint32_t page_slot_stride_bytes = 0;
  std::string embedding_dtype;
  std::vector<uint32_t> slot_ordinals;
  std::vector<uint64_t> keys;
  std::vector<uint64_t> sequences;
  std::span<const std::byte> values;
  std::vector<uint64_t> scores;
  std::span<const std::byte> optimizer_values;
  std::string optimizer_dtype;
  uint32_t optimizer_stride = 0;
  std::vector<uint32_t> steps;
  std::string checksum;
};

using NativeEntryBatchVisitor = std::function<bool(const NativeEntryBatch &)>;
using NativePageImageVisitor = std::function<bool(NativePageImageView &&)>;

checkpoint::NativeEntryScanStatistics scan_native_checkpoint_entries(
    std::span<const std::string> uris, uint64_t version,
    pro::proxy<se::ops::PartitionerFacade> partitioner, uint32_t partition_id,
    std::string_view key_dtype, NativeEntryBatchVisitor visitor,
    bool load_optimizer_state, const LanceStorageOptions &storage_options,
    std::optional<uint32_t> source_participant_index);
void scan_native_page_images(
    std::string_view uri, uint64_t version, uint32_t participant_index,
    std::string_view key_dtype, NativePageImageVisitor visitor,
    bool load_optimizer_state = true,
    const LanceStorageOptions &storage_options = LanceStorageOptions{});

} // namespace se
