#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "sparse_embedding/src/checkpoint/lance/arrow/c/abi.h"

namespace se {

struct NativeFragmentArtifact {
  std::string locator;
  uint64_t entries_version = 0;
  uint64_t page_images_version = 0;
  uint64_t entry_count = 0;
  uint64_t page_count = 0;
  uint64_t physical_bytes = 0;
  std::string schema_digest;
  std::string content_digest;
};

struct NativeMergeArtifact {
  std::string locator;
  uint64_t target_version = 0;
  uint64_t inserted_rows = 0;
  uint64_t updated_rows = 0;
  uint64_t deleted_rows = 0;
};

class NativeFragmentWriter {
  std::string root_;
  std::string entries_uri_;
  std::string pages_uri_;
  std::string embedding_dtype_;
  std::string optimizer_dtype_;
  uint32_t participant_index_ = 0;
  bool has_score_ = false;
  bool entries_created_ = false;
  bool pages_created_ = false;
  std::vector<char> schema_metadata_;
  NativeFragmentArtifact artifact_;
  uint64_t content_hash_ = 1469598103934665603ULL;

public:
  static void release_schema(ArrowSchema *schema);
  static void release_array(ArrowArray *array);

private:
  void write_entries(size_t dim, std::span<const uint64_t> keys,
                     std::span<const std::byte> values, size_t value_stride,
                     std::span<const uint64_t> scores,
                     std::span<const std::byte> optimizer,
                     size_t optimizer_stride, std::span<const uint32_t> steps,
                     std::span<const uint64_t> sequences,
                     uint64_t physical_page_id,
                     std::span<const uint32_t> slots);
  void write_page(size_t embedding_dim, uint64_t physical_page_id,
                  uint32_t page_slot_stride_bytes,
                  std::span<const uint32_t> slots,
                  std::span<const uint64_t> keys,
                  std::span<const uint64_t> sequences,
                  std::span<const std::byte> values,
                  std::span<const uint64_t> scores,
                  std::span<const std::byte> optimizer, size_t optimizer_stride,
                  std::span<const uint32_t> steps, std::string_view checksum);

public:
  void begin_fragment(std::string_view staging_uri, uint32_t participant_index,
                      std::span<const uint32_t> dimensions,
                      std::string_view embedding_dtype,
                      std::string_view optimizer_dtype,
                      std::string_view score_policy, uint32_t protocol_version,
                      std::string_view schema_digest, bool has_score);

  void write_page_image(
      size_t embedding_dim, uint64_t physical_page_id,
      uint32_t page_slot_stride_bytes, size_t embedding_row_bytes,
      std::span<const uint32_t> slots, std::span<const uint64_t> keys,
      std::span<const uint64_t> sequences, std::span<const std::byte> values,
      std::span<const uint64_t> scores, std::span<const std::byte> optimizer,
      size_t optimizer_stride, size_t optimizer_itemsize,
      std::span<const uint32_t> steps);

  NativeFragmentArtifact finalize();
};

NativeMergeArtifact merge_native_lance_sources(
    std::string_view target_uri, std::span<const std::string> source_uris,
    std::span<const std::string_view> keys, uint64_t checkpoint_version = 0);

struct NativeEntryBatch {
  std::vector<uint32_t> dimensions;
  std::vector<uint64_t> keys;
  std::vector<std::byte> embedding_values;
  std::vector<uint32_t> embedding_offsets;
  std::string embedding_dtype;
  std::vector<std::byte> optimizer_values;
  std::vector<uint32_t> optimizer_offsets;
  std::string optimizer_dtype;
  std::vector<uint32_t> optimizer_strides;
  std::vector<uint32_t> steps;
  std::vector<uint64_t> scores;
};

struct NativePageImage {
  uint32_t embedding_dim = 0;
  uint64_t physical_page_id = 0;
  uint32_t page_slot_stride_bytes = 0;
  std::string embedding_dtype;
  std::vector<uint32_t> slot_ordinals;
  std::vector<uint64_t> keys;
  std::vector<uint64_t> sequences;
  std::vector<std::byte> values;
  std::vector<uint64_t> scores;
  std::vector<std::byte> optimizer_values;
  std::string optimizer_dtype;
  uint32_t optimizer_stride = 0;
  std::vector<uint32_t> steps;
  std::string checksum;
};

using NativeKeyBatchVisitor = std::function<bool(std::span<const uint64_t>)>;
using NativeEntryBatchVisitor = std::function<bool(const NativeEntryBatch &)>;
using NativePageImageVisitor = std::function<bool(NativePageImage &&)>;

void scan_native_active_entry_keys(std::string_view uri, uint64_t version,
                                   NativeKeyBatchVisitor visitor);
void scan_native_entries(std::string_view uri, uint64_t version,
                         std::span<const uint64_t> selected_keys,
                         NativeEntryBatchVisitor visitor);
void scan_native_page_images(std::string_view uri, uint64_t version,
                             uint32_t participant_index,
                             NativePageImageVisitor visitor);

} // namespace se
