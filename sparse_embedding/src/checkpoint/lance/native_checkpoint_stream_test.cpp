#include "sparse_embedding/src/checkpoint/checkpoint_test_support.hpp"
#include "sparse_embedding/src/checkpoint/lance/layout_schema.hpp"
#include "sparse_embedding/src/checkpoint/lance/native_fragment_writer.hpp"
#include "sparse_embedding/src/interface/embedding_table_interface.hpp"

#include <cstring>

namespace {

using namespace se::checkpoint::test;
constexpr std::array<uint32_t, 1> kDimensions{2};
constexpr uint32_t kEntriesPerPage = 4096;

std::string entries(const std::filesystem::path &root) {
  return (root / se::lance_layout::kEntriesDataset).string();
}

void begin(se::NativeFragmentWriter &writer, const std::filesystem::path &root,
           uint32_t participant = 0, bool pages = false) {
  writer.begin_fragment(
      root.string(), participant, kDimensions, "float32", "", "lfu", "uint64",
      "stream-test", true, pages, kEntriesPerPage,
      pages ? (root / se::lance_layout::kPageImagesDataset).string() : "");
}

template <size_t Count>
void write(se::NativeFragmentWriter &writer,
           const std::array<uint64_t, Count> &keys,
           const std::array<uint64_t, Count> &sequences,
           const std::array<float, Count * 2> &values,
           const std::array<uint64_t, Count> &scores) {
  std::array<uint32_t, Count> slots;
  for (size_t index = 0; index < Count; ++index)
    slots[index] = index;
  writer.write_page_image(2, 0, 64, 2 * sizeof(float), slots, keys, sequences,
                          std::as_bytes(std::span(values)), scores, {}, 0, 0,
                          {});
}

void check_single_source_pass_and_chain_order() {
  se::checkpoint::TemporaryRunDirectory root;
  const auto full = root.file("full");
  const auto delta = root.file("delta");
  se::NativeFragmentWriter writer;
  begin(writer, full);
  write<2>(writer, {0, 2}, {12, 14}, {1, 2, 3, 4}, {4, 5});
  (void)writer.finalize();
  begin(writer, delta);
  const std::array<se::CheckpointMutation, 2> mutations{
      se::CheckpointMutation{2, 0, 1, false}, {2, 2, 2, true}};
  writer.set_incremental_mutations(mutations);
  write<1>(writer, {0}, {6}, {9, 10}, {17});
  (void)writer.finalize();

  const std::array<std::string, 2> uris{entries(full), entries(delta)};
  const std::array<std::vector<std::string>, 2> chain{
      std::vector<std::string>{uris[0]}, {uris[1]}};
  const std::array<std::vector<uint64_t>, 2> watermarks{
      std::vector<uint64_t>{0}, {2}};
  (void)se::validate_native_checkpoint_chain(chain, watermarks, kDimensions,
                                             "uint64", {});
  size_t visits = 0;
  const auto statistics = se::scan_native_checkpoint_entries(
      uris, 0,
      pro::make_proxy<se::ops::PartitionerFacade, se::ops::ModuloPartitioner>(
          1),
      0, "uint64",
      [&](const se::NativeEntryBatch &batch) {
        ++visits;
        require(batch.keys == std::vector<uint64_t>{0} &&
                    batch.sequences == std::vector<uint64_t>{6} &&
                    batch.scores == std::vector<uint64_t>{17},
                "winner followed physical sequence instead of source order");
        const std::array<float, 2> expected{9, 10};
        require(batch.embedding_values.size() == sizeof(expected) &&
                    std::memcmp(batch.embedding_values.data(), expected.data(),
                                sizeof(expected)) == 0,
                "winner values were not preserved");
        return true;
      },
      true, {}, 0);
  require(visits == 1 && statistics.sources_scanned == 2 &&
              statistics.rows_scanned == 4 &&
              statistics.payload.rows_staged == 3 &&
              statistics.payload.rows_selected == 1 &&
              statistics.payload.logical_bytes_written == 168 &&
              statistics.payload.logical_bytes_read == 56,
          "native reader rescanned sources or replayed obsolete payload");

  const auto compacted = root.file("compacted");
  const auto artifact = se::compact_native_checkpoint(
      compacted.string(), uris, 0, kDimensions, "float32", "", "lfu", "uint64",
      "stream-test", true, kEntriesPerPage,
      (compacted / se::lance_layout::kPageImagesDataset).string(), {});
  require(artifact.entry_count == 1 && artifact.page_count == 1,
          "compaction retained an obsolete identity");
  const std::array<std::vector<std::string>, 1> compacted_chain{
      std::vector<std::string>{entries(compacted)}};
  const std::array<std::vector<uint64_t>, 1> compacted_watermark{
      std::vector<uint64_t>{0}};
  (void)se::validate_native_checkpoint_chain(
      compacted_chain, compacted_watermark, kDimensions, "uint64", {});
  size_t pages = 0;
  se::scan_native_page_images(
      (compacted / se::lance_layout::kPageImagesDataset).string(), 0, 0,
      "uint64", [&](se::NativePageImageView &&page) {
        ++pages;
        require(page.keys == std::vector<uint64_t>{0} &&
                    page.sequences == std::vector<uint64_t>{6} &&
                    page.scores == std::vector<uint64_t>{17},
                "FULL compaction changed winning physical state");
        return true;
      });
  require(pages == 1, "compacted page was not readable");
}

void check_non_target_metadata_rejected_before_visitation() {
  se::checkpoint::TemporaryRunDirectory root;
  const auto full = root.file("duplicate");
  se::NativeFragmentWriter writer;
  begin(writer, full);
  write<2>(writer, {0, 0}, {1, 3}, {1, 2, 3, 4}, {4, 5});
  (void)writer.finalize();
  const std::array<std::string, 1> uris{entries(full)};
  size_t visits = 0;
  require_throws<std::invalid_argument>(
      [&] {
        (void)se::scan_native_checkpoint_entries(
            uris, 0,
            pro::make_proxy<se::ops::PartitionerFacade,
                            se::ops::ModuloPartitioner>(2),
            1, "uint64",
            [&](const auto &) {
              ++visits;
              return true;
            },
            false, {}, 0);
      },
      "non-target duplicate identity was ignored");
  require(visits == 0, "invalid history reached target materialization");
}

void check_page_scan_across_batches() {
  se::checkpoint::TemporaryRunDirectory root;
  const auto fragment = root.file("pages");
  se::NativeFragmentWriter writer;
  begin(writer, fragment, 0, true);
  const size_t count = se::CheckpointStreamPolicy::kPageImportBatchPages + 1;
  const std::array<uint32_t, 1> slots{0};
  const std::array<uint64_t, 1> sequences{2}, scores{0};
  const std::array<float, 2> values{1, 2};
  for (uint64_t index = 0; index < count; ++index) {
    const std::array<uint64_t, 1> keys{index};
    writer.write_page_image(2, index, 64, sizeof(values), slots, keys,
                            sequences, std::as_bytes(std::span(values)), scores,
                            {}, 0, 0, {});
  }
  (void)writer.finalize();
  size_t seen = 0;
  se::scan_native_page_images(
      (fragment / se::lance_layout::kPageImagesDataset).string(), 0, 0,
      "uint64", [&](se::NativePageImageView &&image) {
        require(
            image.physical_page_id == seen &&
                image.keys == std::vector<uint64_t>{seen},
            "multi-batch page scanner reordered or retained the wrong page");
        ++seen;
        return true;
      });
  require(seen == count, "page scanner stopped at its first bounded batch");
}

} // namespace

int main() {
  check_single_source_pass_and_chain_order();
  check_non_target_metadata_rejected_before_visitation();
  check_page_scan_across_batches();
}
