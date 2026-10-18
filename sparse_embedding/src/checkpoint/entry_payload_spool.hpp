#pragma once

#include <optional>

#include "sparse_embedding/src/checkpoint/external_sort.hpp"
#include "sparse_embedding/src/checkpoint/lance/native_entry_batch.hpp"
#include "sparse_embedding/src/common/dtypes.hpp"

namespace se::checkpoint {

inline constexpr uint16_t kNoPayloadDtype =
    std::numeric_limits<uint16_t>::max();

inline ValueDType canonical_payload_dtype(std::string_view name) {
  const auto dtype = parse_value_dtype(name);
  if (value_dtype_name(dtype) != name)
    throw std::invalid_argument("checkpoint payload dtype must be canonical");
  return dtype;
}

inline uint16_t payload_dtype_code(std::string_view name) {
  return name.empty() ? kNoPayloadDtype
                      : static_cast<uint16_t>(canonical_payload_dtype(name));
}

inline ValueDType payload_dtype(uint16_t code) {
  if (!std::in_range<std::underlying_type_t<ValueDType>>(code))
    throw std::invalid_argument("checkpoint payload dtype code is invalid");
  const auto dtype = static_cast<ValueDType>(code);
  (void)value_dtype_name(dtype);
  return dtype;
}

struct EntryPayloadHeader {
  uint64_t key;
  uint64_t score;
  uint64_t sequence;
  uint32_t dimension;
  uint32_t optimizer_stride;
  uint32_t step;
  uint32_t embedding_bytes;
  uint32_t optimizer_bytes;
  uint16_t embedding_dtype;
  uint16_t optimizer_dtype;
};

static_assert(sizeof(EntryPayloadHeader) == 48);
static_assert(std::has_unique_object_representations_v<EntryPayloadHeader>);

// Source payload is staged once; only validated winners are replayed, in
// increasing file-offset order. Obsolete keys never consume final store slots.
class EntryPayloadSpool {
  TemporaryRunDirectory directory_;
  std::ofstream output_;
  PayloadSpoolStatistics statistics_;
  const bool load_optimizer_state_;
  bool sealed_ = false;
  bool consumed_ = false;
  std::optional<uint16_t> embedding_dtype_;
  std::optional<uint16_t> optimizer_dtype_;

  void validate(const EntryPayloadHeader &header) const {
    const auto embedding_dtype = payload_dtype(header.embedding_dtype);
    if (header.dimension == 0 ||
        header.embedding_bytes !=
            checked_byte_count(header.dimension,
                               value_dtype_itemsize(embedding_dtype)))
      throw std::runtime_error("checkpoint embedding payload shape is invalid");
    if (!load_optimizer_state_) {
      if (header.optimizer_bytes != 0 || header.optimizer_stride != 0)
        throw std::runtime_error(
            "weight-only checkpoint staged optimizer state");
      return;
    }
    const uint64_t optimizer_bytes =
        header.optimizer_dtype == kNoPayloadDtype
            ? 0
            : checked_byte_count(
                  header.optimizer_stride,
                  value_dtype_itemsize(payload_dtype(header.optimizer_dtype)));
    if (header.optimizer_bytes != optimizer_bytes ||
        (header.optimizer_dtype == kNoPayloadDtype &&
         header.optimizer_stride != 0))
      throw std::runtime_error("checkpoint optimizer payload shape is invalid");
  }

  static void reset_batch(NativeEntryBatch &batch) {
    batch.dimensions.clear();
    batch.keys.clear();
    batch.embedding_values.clear();
    batch.embedding_offsets.assign(1, 0);
    batch.optimizer_values.clear();
    batch.optimizer_offsets.assign(1, 0);
    batch.optimizer_strides.clear();
    batch.steps.clear();
    batch.scores.clear();
    batch.sequences.clear();
  }

  static void read_payload(std::ifstream &input, uint32_t bytes,
                           std::vector<std::byte> &values,
                           std::vector<uint32_t> &offsets) {
    const size_t begin = values.size();
    const uint64_t end = checked_sum(begin, bytes);
    if (!std::in_range<uint32_t>(end))
      throw std::overflow_error("checkpoint batch binary offsets overflow");
    values.resize(end);
    if (bytes != 0) {
      input.read(reinterpret_cast<char *>(values.data() + begin), bytes);
      if (input.gcount() != bytes || !input)
        throw std::runtime_error("checkpoint staged payload is truncated");
    }
    offsets.push_back(static_cast<uint32_t>(end));
  }

public:
  explicit EntryPayloadSpool(bool load_optimizer_state)
      : load_optimizer_state_(load_optimizer_state) {}
  EntryPayloadSpool(const EntryPayloadSpool &) = delete;
  EntryPayloadSpool &operator=(const EntryPayloadSpool &) = delete;

  const PayloadSpoolStatistics &statistics() const noexcept {
    return statistics_;
  }

  uint64_t append(const EntryPayloadHeader &header,
                  std::span<const std::byte> embedding,
                  std::span<const std::byte> optimizer) {
    if (sealed_)
      throw std::logic_error("checkpoint payload spool is already sealed");
    validate(header);
    if (header.embedding_bytes != embedding.size() ||
        header.optimizer_bytes != optimizer.size())
      throw std::runtime_error("checkpoint staged payload lengths differ");
    if ((embedding_dtype_ && *embedding_dtype_ != header.embedding_dtype) ||
        (load_optimizer_state_ && optimizer_dtype_ &&
         *optimizer_dtype_ != header.optimizer_dtype))
      throw std::runtime_error(
          "checkpoint payload dtype changed across sources");
    embedding_dtype_ = header.embedding_dtype;
    optimizer_dtype_ = header.optimizer_dtype;
    const uint64_t offset = statistics_.logical_bytes_written;
    const uint64_t bytes =
        checked_sum(sizeof(EntryPayloadHeader),
                    checked_sum(embedding.size(), optimizer.size()));
    const uint64_t end = checked_sum(offset, bytes);
    if (!std::in_range<std::streamoff>(end))
      throw std::overflow_error("checkpoint payload file offset overflows");
    if (!output_.is_open()) {
      output_.open(directory_.file("payload.bin"),
                   std::ios::binary | std::ios::trunc);
      if (!output_)
        throw std::runtime_error("checkpoint payload spool open failed");
    }
    write_records(output_, std::span<const EntryPayloadHeader>(&header, 1));
    write_records(output_, embedding);
    write_records(output_, optimizer);
    statistics_.logical_bytes_written = end;
    statistics_.rows_staged = checked_sum(statistics_.rows_staged, 1);
    return offset;
  }

  void seal() {
    if (sealed_)
      throw std::logic_error("checkpoint payload spool is already sealed");
    if (output_.is_open())
      finish_output(output_);
    sealed_ = true;
  }

  template <typename SortedOffsets, typename Visitor>
  bool consume_selected(SortedOffsets &offsets, Visitor &&visitor) {
    if (!sealed_ || consumed_)
      throw std::logic_error(
          "checkpoint payload spool is not ready for consumption");
    consumed_ = true;
    try {
      bool complete = true;
      {
        std::ifstream input;
        if (statistics_.logical_bytes_written != 0) {
          input.open(directory_.file("payload.bin"), std::ios::binary);
          if (!input)
            throw std::runtime_error(
                "checkpoint payload spool read open failed");
          if (std::filesystem::file_size(directory_.file("payload.bin")) !=
              statistics_.logical_bytes_written)
            throw std::runtime_error("checkpoint payload spool length differs");
        }
        NativeEntryBatch batch;
        reset_batch(batch);
        auto flush = [&] {
          const bool keep_reading = visitor(batch);
          reset_batch(batch);
          return keep_reading;
        };
        uint64_t next_offset = 0;
        complete = offsets.consume_sorted([&](uint64_t offset) {
          if (offset < next_offset ||
              offset > statistics_.logical_bytes_written ||
              statistics_.logical_bytes_written - offset <
                  sizeof(EntryPayloadHeader))
            throw std::runtime_error(
                "checkpoint selected payload offset is invalid");
          if (offset != next_offset) {
            input.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
            if (!input)
              throw std::runtime_error("checkpoint staged payload seek failed");
            statistics_.seek_operations =
                checked_sum(statistics_.seek_operations, 1);
          }
          EntryPayloadHeader header;
          if (read_records(input, std::span<EntryPayloadHeader>(&header, 1)) !=
              1)
            throw std::runtime_error(
                "checkpoint staged payload header is truncated");
          statistics_.logical_bytes_read =
              checked_sum(statistics_.logical_bytes_read, sizeof(header));
          validate(header);
          const uint64_t bytes =
              checked_sum(header.embedding_bytes, header.optimizer_bytes);
          const uint64_t record_bytes = checked_sum(sizeof(header), bytes);
          if (record_bytes > statistics_.logical_bytes_written - offset)
            throw std::runtime_error(
                "checkpoint staged payload length is invalid");
          next_offset = checked_sum(offset, record_bytes);
          const size_t batch_bytes =
              batch.embedding_values.size() + batch.optimizer_values.size();
          if (!batch.keys.empty() &&
              (batch.keys.size() == CheckpointStreamPolicy::kScanBatchRows ||
               checked_sum(batch_bytes, bytes) >
                   CheckpointStreamPolicy::kEntryImportBatchBytes)) {
            if (!flush())
              return false;
          }
          read_payload(input, header.embedding_bytes, batch.embedding_values,
                       batch.embedding_offsets);
          read_payload(input, header.optimizer_bytes, batch.optimizer_values,
                       batch.optimizer_offsets);
          batch.dimensions.push_back(header.dimension);
          batch.keys.push_back(header.key);
          batch.optimizer_strides.push_back(header.optimizer_stride);
          batch.steps.push_back(header.step);
          batch.scores.push_back(header.score);
          batch.sequences.push_back(header.sequence);
          batch.embedding_dtype =
              value_dtype_name(payload_dtype(header.embedding_dtype));
          batch.optimizer_dtype =
              header.optimizer_dtype == kNoPayloadDtype
                  ? ""
                  : value_dtype_name(payload_dtype(header.optimizer_dtype));
          statistics_.logical_bytes_read =
              checked_sum(statistics_.logical_bytes_read, bytes);
          statistics_.rows_selected = checked_sum(statistics_.rows_selected, 1);
          statistics_.peak_batch_rows =
              std::max(statistics_.peak_batch_rows, batch.keys.size());
          statistics_.peak_batch_bytes = std::max(
              statistics_.peak_batch_bytes,
              batch.embedding_values.size() + batch.optimizer_values.size());
          return true;
        });
        if (complete && !batch.keys.empty())
          complete = flush();
      }
      directory_.cleanup();
      return complete;
    } catch (...) {
      directory_.cleanup_after_error();
      throw;
    }
  }
};

} // namespace se::checkpoint
