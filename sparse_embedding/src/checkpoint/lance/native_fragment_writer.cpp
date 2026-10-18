#include "sparse_embedding/src/checkpoint/lance/native_fragment_writer.hpp"
#include "sparse_embedding/src/checkpoint/lance/native_payload.hpp"

#include <lance/lance.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <condition_variable>
#include <cstring>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <type_traits>
#include <unordered_map>

#include "sparse_embedding/src/checkpoint/checkpoint_stream_policy.hpp"
#include "sparse_embedding/src/checkpoint/entry_payload_spool.hpp"
#include "sparse_embedding/src/checkpoint/lance/layout_schema.hpp"
#include "sparse_embedding/src/checkpoint/ordered_checkpoint.hpp"
#include "sparse_embedding/src/common/dtypes.hpp"
#include "sparse_embedding/src/interface/embedding_table_interface.hpp"
#include "sparse_embedding/src/ops/partitioner_registry.hpp"
#include "sparse_embedding/src/storage/topology/storage_topology.hpp"

namespace se {
namespace {

constexpr std::string_view kActiveRowState = "active";

struct DatasetDeleter {
  void operator()(LanceDataset *dataset) const noexcept {
    if (dataset)
      lance_dataset_close(dataset);
  }
};

struct ScannerDeleter {
  void operator()(LanceScanner *scanner) const noexcept {
    if (scanner)
      lance_scanner_close(scanner);
  }
};

class ArrowStreamGuard {
  ArrowArrayStream stream_{};

public:
  ArrowStreamGuard() = default;
  ArrowArrayStream *get() noexcept { return &stream_; }
  ~ArrowStreamGuard() {
    if (stream_.release)
      stream_.release(&stream_);
  }
  ArrowStreamGuard(const ArrowStreamGuard &) = delete;
  ArrowStreamGuard &operator=(const ArrowStreamGuard &) = delete;
};

class ArrowSchemaGuard {
  ArrowSchema schema_{};

public:
  ArrowSchemaGuard() = default;
  ArrowSchema *get() noexcept { return &schema_; }
  ~ArrowSchemaGuard() {
    if (schema_.release)
      schema_.release(&schema_);
  }
  ArrowSchemaGuard(const ArrowSchemaGuard &) = delete;
  ArrowSchemaGuard &operator=(const ArrowSchemaGuard &) = delete;
};

class ArrowArrayGuard {
  ArrowArray array_{};

public:
  ArrowArrayGuard() = default;
  ~ArrowArrayGuard() { reset(); }
  ArrowArrayGuard(const ArrowArrayGuard &) = delete;
  ArrowArrayGuard &operator=(const ArrowArrayGuard &) = delete;
  ArrowArrayGuard(ArrowArrayGuard &&other) noexcept : array_(other.array_) {
    other.array_ = {};
  }
  ArrowArrayGuard &operator=(ArrowArrayGuard &&other) noexcept {
    if (this == &other)
      return *this;
    reset();
    array_ = other.array_;
    other.array_ = {};
    return *this;
  }

  ArrowArray *get() noexcept { return &array_; }
  const ArrowArray *get() const noexcept { return &array_; }
  ArrowArray *operator->() noexcept { return &array_; }
  const ArrowArray *operator->() const noexcept { return &array_; }
  ArrowArray &operator*() noexcept { return array_; }

  void dismiss() noexcept { array_.release = nullptr; }
  void reset() noexcept {
    if (array_.release)
      array_.release(&array_);
    array_ = {};
  }
};

class BinaryColumnView {
  const ArrowArray *array_;
  const int32_t *offsets_ = nullptr;
  const std::byte *data_ = nullptr;
  size_t data_bytes_ = 0;

public:
  explicit BinaryColumnView(const ArrowArray *array) : array_(array) {
    if (!array || array->n_buffers < 3 || !array->buffers ||
        !array->buffers[1] || array->length < 0 || array->offset < 0 ||
        array->offset > std::numeric_limits<int64_t>::max() - array->length)
      throw std::runtime_error("invalid Arrow binary column");
    offsets_ = static_cast<const int32_t *>(array->buffers[1]);
    data_ = static_cast<const std::byte *>(array->buffers[2]);
    const int64_t first = array->offset;
    const int64_t last = first + array->length;
    if (offsets_[first] < 0 || offsets_[last] < offsets_[first])
      throw std::runtime_error("invalid Arrow binary offsets");
    data_bytes_ = static_cast<size_t>(offsets_[last]);
    if (data_bytes_ != 0 && !data_)
      throw std::runtime_error("Arrow binary column has no data buffer");
  }

  std::span<const std::byte> row(int64_t row_index) const {
    if (row_index < 0 || row_index >= array_->length)
      throw std::out_of_range("Arrow binary row is outside the record batch");
    const int64_t index = array_->offset + row_index;
    const int32_t begin = offsets_[index];
    const int32_t end = offsets_[index + 1];
    if (begin < 0 || end < begin || static_cast<size_t>(end) > data_bytes_)
      throw std::runtime_error("Arrow binary offsets are invalid");
    if (end == begin)
      return {};
    return {data_ + begin, static_cast<size_t>(end - begin)};
  }
};

class ArrowRecordBatchView {
  const ArrowArray *array_;
  const ArrowSchema *schema_;

public:
  ArrowRecordBatchView(const ArrowArray *array, const ArrowSchema *schema)
      : array_(array), schema_(schema) {
    if (!array_ || !schema_ || array_->length < 0 || schema_->n_children < 0 ||
        array_->n_children != schema_->n_children)
      throw std::runtime_error("invalid Arrow record batch");
  }

  const ArrowArray *field(size_t index) const {
    if (index >= static_cast<size_t>(array_->n_children) || !array_->children ||
        !array_->children[index])
      throw std::out_of_range("Arrow record batch field is outside the schema");
    return array_->children[index];
  }

  const ArrowSchema *field_schema(size_t index) const {
    if (index >= static_cast<size_t>(schema_->n_children) ||
        !schema_->children || !schema_->children[index])
      throw std::out_of_range("Arrow schema field is outside the schema");
    return schema_->children[index];
  }

  template <typename T> std::span<const T> fixed(size_t index) const {
    const auto *field_array = field(index);
    if (field_array->n_buffers < 2 || !field_array->buffers[1] ||
        field_array->length < 0 || field_array->offset < 0)
      throw std::runtime_error("invalid Arrow fixed-width column");
    const auto *values = static_cast<const T *>(field_array->buffers[1]);
    return {values + field_array->offset,
            static_cast<size_t>(field_array->length)};
  }

  BinaryColumnView binary(size_t index) const {
    return BinaryColumnView(field(index));
  }

  int64_t length() const noexcept { return array_->length; }
};

class ArrowArrayView {
  const ArrowArray *array_;

public:
  explicit ArrowArrayView(const ArrowArray *array) : array_(array) {
    if (!array_ || array_->length < 0 || array_->offset < 0)
      throw std::runtime_error("invalid Arrow array");
  }

  BinaryColumnView binary() const { return BinaryColumnView(array_); }

  template <typename T> T value(int64_t row) const {
    if (row < 0 || row >= array_->length || array_->n_buffers < 2 ||
        !array_->buffers[1])
      throw std::out_of_range("Arrow fixed-width row is outside the array");
    const auto *values = static_cast<const T *>(array_->buffers[1]);
    return values[array_->offset + row];
  }

  template <typename T> std::span<const T> values() const {
    if (array_->n_buffers < 2 || !array_->buffers[1])
      throw std::runtime_error("invalid Arrow fixed-width column");
    const auto *values = static_cast<const T *>(array_->buffers[1]);
    return {values + array_->offset, static_cast<size_t>(array_->length)};
  }

  std::span<const std::byte> bytes(size_t width) const {
    if (array_->n_buffers < 2 || !array_->buffers[1] ||
        width > std::numeric_limits<size_t>::max() /
                    static_cast<size_t>(array_->length))
      throw std::runtime_error("invalid Arrow fixed-width column");
    const auto *data = static_cast<const std::byte *>(array_->buffers[1]);
    return {data + static_cast<size_t>(array_->offset) * width,
            static_cast<size_t>(array_->length) * width};
  }
};

inline void release_arrow_children(ArrowArray &array) noexcept {
  if (!array.children)
    return;
  for (int64_t index = 0; index < array.n_children; ++index) {
    auto *child = array.children[index];
    if (child && child->release)
      child->release(child);
  }
  delete[] array.children;
  array.children = nullptr;
}

inline void release_arrow_schema_children(ArrowSchema &schema) noexcept {
  if (!schema.children)
    return;
  for (int64_t index = 0; index < schema.n_children; ++index) {
    auto *child = schema.children[index];
    if (child->release)
      child->release(child);
  }
  delete[] schema.children;
  schema.children = nullptr;
}

inline const ArrowSchema *arrow_schema_field(const ArrowSchema &schema,
                                             size_t index) {
  if (index >= static_cast<size_t>(schema.n_children) || !schema.children ||
      !schema.children[index])
    throw std::out_of_range("Arrow schema field is outside the schema");
  return schema.children[index];
}

inline const ArrowArray *arrow_array_field(const ArrowArray &array,
                                           size_t index) {
  if (index >= static_cast<size_t>(array.n_children) || !array.children ||
      !array.children[index])
    throw std::out_of_range("Arrow record batch field is outside the schema");
  return array.children[index];
}

struct ColumnOwner {
  std::vector<std::byte> data;
  std::vector<int32_t> offsets;
};

void release_column(ArrowArray *array) {
  if (!array || !array->release)
    return;
  delete static_cast<ColumnOwner *>(array->private_data);
  delete[] array->buffers;
  array->release = nullptr;
  array->private_data = nullptr;
}

ArrowArray *make_fixed_column(const char *format, const char *name,
                              std::vector<std::byte> data, size_t width,
                              size_t length) {
  auto *owner = new ColumnOwner{std::move(data), {}};
  auto *schema = static_cast<void *>(nullptr);
  (void)schema;
  auto *array =
      new ArrowArray{static_cast<int64_t>(length),
                     0,
                     0,
                     2,
                     0,
                     new const void *[2] { nullptr, owner->data.data() },
                     nullptr,
                     nullptr,
                     &release_column,
                     owner};
  (void)format;
  (void)name;
  (void)width;
  return array;
}

ArrowArray *make_single_binary_column(std::span<const std::byte> value) {
  if (value.size() > static_cast<size_t>(INT32_MAX))
    throw std::length_error("native Lance binary column exceeds Arrow range");
  auto *owner = new ColumnOwner;
  owner->offsets.reserve(2);
  owner->offsets.push_back(0);
  owner->data.assign(value.begin(), value.end());
  owner->offsets.push_back(static_cast<int32_t>(value.size()));
  auto *array = new ArrowArray{1,
                               0,
                               0,
                               3,
                               0,
                               new const void *[3] {
                                 nullptr, owner->offsets.data(),
                                     owner->data.data()
                               },
                               nullptr,
                               nullptr,
                               &release_column,
                               owner};
  return array;
}

ArrowArray *make_repeated_binary_column(std::span<const std::byte> row,
                                        size_t count) {
  if (row.size() > static_cast<size_t>(INT32_MAX) ||
      count > static_cast<size_t>(INT32_MAX) ||
      (count != 0 && row.size() > static_cast<size_t>(INT32_MAX) / count))
    throw std::length_error("native Lance binary column exceeds Arrow range");
  auto *owner = new ColumnOwner;
  owner->data.resize(row.size() * count);
  for (size_t index = 0; index < count; ++index)
    std::copy(row.begin(), row.end(),
              owner->data.begin() +
                  static_cast<std::ptrdiff_t>(index * row.size()));
  owner->offsets.resize(count + 1);
  for (size_t index = 0; index <= count; ++index)
    owner->offsets[index] = static_cast<int32_t>(index * row.size());
  return new ArrowArray{static_cast<int64_t>(count),
                        0,
                        0,
                        3,
                        0,
                        new const void *[3] {
                          nullptr, owner->offsets.data(), owner->data.data()
                        },
                        nullptr,
                        nullptr,
                        &release_column,
                        owner};
}

ArrowArray *make_strided_binary_column(std::span<const std::byte> data,
                                       size_t stride, size_t count) {
  if (stride != 0 && (count > std::numeric_limits<size_t>::max() / stride ||
                      data.size() != stride * count))
    throw std::invalid_argument("native Lance payload stride mismatch");
  if (count == 0)
    return make_repeated_binary_column({}, 0);
  if (stride == 0)
    return make_repeated_binary_column({}, count);
  if (stride > static_cast<size_t>(INT32_MAX) ||
      count > static_cast<size_t>(INT32_MAX) ||
      stride > static_cast<size_t>(INT32_MAX) / count)
    throw std::length_error("native Lance binary column exceeds Arrow range");
  auto *owner = new ColumnOwner;
  owner->data.assign(data.begin(), data.end());
  owner->offsets.resize(count + 1);
  for (size_t index = 0; index <= count; ++index)
    owner->offsets[index] = static_cast<int32_t>(index * stride);
  return new ArrowArray{static_cast<int64_t>(count),
                        0,
                        0,
                        3,
                        0,
                        new const void *[3] {
                          nullptr, owner->offsets.data(), owner->data.data()
                        },
                        nullptr,
                        nullptr,
                        &release_column,
                        owner};
}

ArrowArray *make_binary_column(std::vector<std::byte> data,
                               std::vector<int32_t> offsets) {
  if (offsets.empty() || offsets.front() != 0 ||
      offsets.back() != static_cast<int32_t>(data.size()))
    throw std::invalid_argument("native Lance binary offsets are invalid");
  auto *owner = new ColumnOwner{std::move(data), std::move(offsets)};
  return new ArrowArray{static_cast<int64_t>(owner->offsets.size() - 1),
                        0,
                        0,
                        3,
                        0,
                        new const void *[3] {
                          nullptr, owner->offsets.data(), owner->data.data()
                        },
                        nullptr,
                        nullptr,
                        &release_column,
                        owner};
}

std::vector<std::byte> string_bytes(std::string_view value) {
  auto first = reinterpret_cast<const std::byte *>(value.data());
  return std::vector<std::byte>(first, first + value.size());
}

std::span<const std::byte> binary_value(const ArrowArray *array, int64_t row) {
  return ArrowArrayView(array).binary().row(row);
}

std::string binary_string(const ArrowArray *array, int64_t row) {
  const auto bytes = binary_value(array, row);
  return std::string(reinterpret_cast<const char *>(bytes.data()),
                     bytes.size());
}

template <typename T> T fixed_value(const ArrowArray *array, int64_t row) {
  return ArrowArrayView(array).value<T>(row);
}

inline size_t find_column_index(const ArrowSchema &schema,
                                std::string_view name) {
  for (int64_t index = 0; index < schema.n_children; ++index) {
    if (arrow_schema_field(schema, index)->name &&
        name == arrow_schema_field(schema, index)->name) {
      return static_cast<size_t>(index);
    }
  }
  throw std::runtime_error("native Lance schema is missing required field: " +
                           std::string(name));
}

struct EntryScanIndices {
  size_t dim = 0;
  size_t key = 0;
  size_t embedding_values = 0;
  size_t embedding_dtype = 0;
  size_t optimizer_state_values = 0;
  size_t optimizer_state_dtype = 0;
  size_t optimizer_state_stride = 0;
  size_t step = 0;
  size_t score = 0;
  size_t sequence = 0;
  size_t mutation_sequence = 0;
  size_t row_state = 0;
  size_t source_participant_index = 0;
};

inline EntryScanIndices resolve_entry_scan_indices(const ArrowSchema &schema,
                                                   bool load_optimizer_state) {
  return {
      .dim = find_column_index(schema, "dim"),
      .key = find_column_index(schema, "key"),
      .embedding_values = find_column_index(schema, "embedding_values"),
      .embedding_dtype = find_column_index(schema, "embedding_dtype"),
      .optimizer_state_values =
          load_optimizer_state
              ? find_column_index(schema, "optimizer_state_values")
              : 0,
      .optimizer_state_dtype =
          load_optimizer_state
              ? find_column_index(schema, "optimizer_state_dtype")
              : 0,
      .optimizer_state_stride =
          load_optimizer_state
              ? find_column_index(schema, "optimizer_state_stride")
              : 0,
      .step = load_optimizer_state ? find_column_index(schema, "step") : 0,
      .score = find_column_index(schema, "score"),
      .sequence = find_column_index(schema, "sequence"),
      .mutation_sequence = find_column_index(schema, "mutation_sequence"),
      .row_state = find_column_index(schema, "row_state"),
      .source_participant_index =
          find_column_index(schema, "source_participant_index"),
  };
}

struct PageImageScanIndices {
  size_t entries_per_page = 0;
  size_t source_participant_index = 0;
  size_t embedding_dim = 0;
  size_t physical_page_id = 0;
  size_t page_slot_stride_bytes = 0;
  size_t embedding_dtype = 0;
  size_t slot_ordinals = 0;
  size_t keys = 0;
  size_t sequences = 0;
  size_t embedding_values = 0;
  size_t scores = 0;
  size_t optimizer_state_values = 0;
  size_t optimizer_state_dtype = 0;
  size_t optimizer_state_stride = 0;
  size_t steps = 0;
  size_t checksum = 0;
  size_t row_state = 0;
};

inline PageImageScanIndices
resolve_page_image_scan_indices(const ArrowSchema &schema) {
  return {
      .entries_per_page = find_column_index(schema, "entries_per_page"),
      .source_participant_index =
          find_column_index(schema, "source_participant_index"),
      .embedding_dim = find_column_index(schema, "embedding_dim"),
      .physical_page_id = find_column_index(schema, "physical_page_id"),
      .page_slot_stride_bytes =
          find_column_index(schema, "page_slot_stride_bytes"),
      .embedding_dtype = find_column_index(schema, "embedding_dtype"),
      .slot_ordinals = find_column_index(schema, "slot_ordinals"),
      .keys = find_column_index(schema, "keys"),
      .sequences = find_column_index(schema, "sequences"),
      .embedding_values = find_column_index(schema, "embedding_values"),
      .scores = find_column_index(schema, "scores"),
      .optimizer_state_values =
          find_column_index(schema, "optimizer_state_values"),
      .optimizer_state_dtype =
          find_column_index(schema, "optimizer_state_dtype"),
      .optimizer_state_stride =
          find_column_index(schema, "optimizer_state_stride"),
      .steps = find_column_index(schema, "steps"),
      .checksum = find_column_index(schema, "checksum"),
      .row_state = find_column_index(schema, "row_state"),
  };
}

size_t dtype_itemsize(std::string_view dtype) {
  return value_dtype_itemsize(checkpoint::canonical_payload_dtype(dtype));
}

std::vector<char> make_arrow_metadata(
    std::span<const std::pair<std::string, std::string>> entries) {
  if (entries.empty())
    return {};
  if (entries.size() > static_cast<size_t>(INT32_MAX))
    throw std::length_error("native Lance metadata has too many entries");
  size_t total_size = sizeof(int32_t);
  for (const auto &[key, value] : entries) {
    if (key.size() > static_cast<size_t>(INT32_MAX) ||
        value.size() > static_cast<size_t>(INT32_MAX))
      throw std::length_error(
          "native Lance metadata entry exceeds Arrow range");
    total_size += sizeof(int32_t) + key.size() + sizeof(int32_t) + value.size();
  }
  const int32_t num_pairs = static_cast<int32_t>(entries.size());
  std::vector<char> buffer(total_size);
  char *ptr = buffer.data();
  std::memcpy(ptr, &num_pairs, sizeof(int32_t));
  ptr += sizeof(int32_t);
  for (const auto &[key, value] : entries) {
    const int32_t key_len = static_cast<int32_t>(key.size());
    const int32_t val_len = static_cast<int32_t>(value.size());
    std::memcpy(ptr, &key_len, sizeof(int32_t));
    ptr += sizeof(int32_t);
    std::memcpy(ptr, key.data(), key.size());
    ptr += key.size();
    std::memcpy(ptr, &val_len, sizeof(int32_t));
    ptr += sizeof(int32_t);
    std::memcpy(ptr, value.data(), value.size());
    ptr += value.size();
  }
  return buffer;
}

std::vector<char> clone_arrow_metadata(const char *metadata) {
  if (!metadata)
    return {};
  int32_t num_pairs = 0;
  std::memcpy(&num_pairs, metadata, sizeof(int32_t));
  if (num_pairs < 0)
    return {};
  const char *ptr = metadata + sizeof(int32_t);
  for (int32_t i = 0; i < num_pairs; ++i) {
    int32_t klen = 0;
    std::memcpy(&klen, ptr, sizeof(int32_t));
    ptr += sizeof(int32_t) + klen;
    int32_t vlen = 0;
    std::memcpy(&vlen, ptr, sizeof(int32_t));
    ptr += sizeof(int32_t) + vlen;
  }
  return std::vector<char>(metadata, ptr);
}

std::string arrow_metadata_value(const char *metadata,
                                 std::string_view wanted) {
  if (!metadata)
    return {};
  int32_t num_pairs = 0;
  std::memcpy(&num_pairs, metadata, sizeof(int32_t));
  if (num_pairs < 0)
    throw std::invalid_argument("native Lance schema metadata is malformed");
  const char *ptr = metadata + sizeof(int32_t);
  for (int32_t i = 0; i < num_pairs; ++i) {
    int32_t key_len = 0;
    std::memcpy(&key_len, ptr, sizeof(int32_t));
    ptr += sizeof(int32_t);
    if (key_len < 0)
      throw std::invalid_argument("native Lance schema metadata is malformed");
    const std::string_view key(ptr, static_cast<size_t>(key_len));
    ptr += key_len;
    int32_t value_len = 0;
    std::memcpy(&value_len, ptr, sizeof(int32_t));
    ptr += sizeof(int32_t);
    if (value_len < 0)
      throw std::invalid_argument("native Lance schema metadata is malformed");
    const std::string_view value(ptr, static_cast<size_t>(value_len));
    ptr += value_len;
    if (key == wanted)
      return std::string(value);
  }
  return {};
}

void require_schema_key_dtype(const ArrowSchema &schema,
                              std::string_view expected) {
  const auto actual =
      arrow_metadata_value(schema.metadata, "sparse_embedding.key_dtype");
  if (actual.empty() || actual != expected)
    throw std::invalid_argument(
        "native Lance schema key dtype does not match restore request");
  // Entry schemas expose a scalar key column; page schemas encode keys in
  // their page payload instead. Check the physical entry format before any
  // typed span can trust metadata for its width and signedness.
  for (int64_t column = 0; column < schema.n_children; ++column) {
    const auto *field = arrow_schema_field(schema, column);
    if (field && field->name && std::string_view(field->name) == "key" &&
        (!field->format ||
         std::string_view(field->format) !=
             lance_layout::arrow_format_for_key_dtype(expected)))
      throw std::invalid_argument(
          "native Lance schema key dtype does not match restore request");
  }
}

template <typename KeyT>
bool key_matches_logical_partition(KeyT key, uint64_t partition_count,
                                   uint64_t partition_id) {
  return modulo_partition(key, partition_count) == partition_id;
}

std::string join_dimensions(std::span<const uint32_t> dimensions) {
  if (dimensions.empty())
    throw std::invalid_argument(
        "native Lance specification dimensions are empty");
  std::string result;
  for (size_t index = 0; index < dimensions.size(); ++index) {
    if (index != 0)
      result.push_back(',');
    std::array<char, 11> buffer{};
    auto [end, error] = std::to_chars(
        buffer.data(), buffer.data() + buffer.size(), dimensions[index]);
    if (error != std::errc{})
      throw std::runtime_error("failed to encode native Lance dimensions");
    result.append(buffer.data(), end);
  }
  return result;
}

void check_lance(int32_t rc, std::string_view operation) {
  if (rc == 0)
    return;
  const char *message = lance_last_error_message();
  std::string detail = message ? message : "unknown Lance error";
  if (message)
    lance_free_string(message);
  throw std::runtime_error(std::string(operation) + ": " + detail);
}

uint64_t fnv1a(std::span<const std::byte> bytes,
               uint64_t seed = 1469598103934665603ULL) {
  uint64_t hash = seed;
  for (std::byte byte : bytes) {
    hash ^= std::to_integer<unsigned char>(byte);
    hash *= 1099511628211ULL;
  }
  return hash;
}

template <typename T>
void append_little_endian(std::vector<std::byte> &output, T value) {
  static_assert(std::is_unsigned_v<T>);
  for (size_t byte = 0; byte < sizeof(T); ++byte)
    output.push_back(static_cast<std::byte>(value >> (byte * 8)));
}

void append_length_prefixed(std::vector<std::byte> &output,
                            std::string_view value) {
  if (value.size() > std::numeric_limits<uint32_t>::max())
    throw std::length_error("canonical tombstone field is too large");
  append_little_endian<uint32_t>(output, static_cast<uint32_t>(value.size()));
  const auto bytes = string_bytes(value);
  output.insert(output.end(), bytes.begin(), bytes.end());
}

std::vector<std::byte>
canonical_entry_tombstone(uint32_t participant_index, uint32_t dimension,
                          uint64_t key, uint64_t mutation_sequence,
                          std::string_view embedding_dtype,
                          std::string_view optimizer_dtype) {
  std::vector<std::byte> record;
  append_little_endian<uint8_t>(record, 1);
  append_length_prefixed(record, "entry_tombstone");
  append_little_endian<uint32_t>(record, participant_index);
  append_little_endian<uint32_t>(record, dimension);
  append_little_endian<uint64_t>(record, key);
  append_little_endian<uint64_t>(record, mutation_sequence);
  append_length_prefixed(record, embedding_dtype);
  append_length_prefixed(record, optimizer_dtype);
  append_little_endian<uint8_t>(record, 1);
  return record;
}

std::string checksum(std::span<const std::byte> slots,
                     std::span<const std::byte> keys,
                     std::span<const std::byte> sequences,
                     std::span<const std::byte> values,
                     std::span<const std::byte> scores,
                     std::span<const std::byte> optimizer,
                     std::span<const std::byte> steps) {
  uint64_t hash = fnv1a(slots);
  hash = fnv1a(keys, hash);
  hash = fnv1a(sequences, hash);
  hash = fnv1a(values, hash);
  hash = fnv1a(scores, hash);
  hash = fnv1a(optimizer, hash);
  hash = fnv1a(steps, hash);
  std::array<char, 17> output{};
  auto [end, error] =
      std::to_chars(output.data(), output.data() + 16, hash, 16);
  if (error != std::errc{})
    throw std::runtime_error("failed to format native page checksum");
  return std::string(output.data(), end);
}

template <typename T>
std::vector<std::byte> bytes_of(std::span<const T> values) {
  auto bytes = std::as_bytes(values);
  return std::vector<std::byte>(bytes.begin(), bytes.end());
}

ArrowArray make_struct_array(std::vector<ArrowArray *> children,
                             size_t length) {
  auto **owned = new ArrowArray *[children.size()];
  std::copy(children.begin(), children.end(), owned);
  return ArrowArray{static_cast<int64_t>(length),
                    0,
                    0,
                    1,
                    static_cast<int64_t>(children.size()),
                    new const void *[1] { nullptr },
                    owned,
                    nullptr,
                    [](ArrowArray *array) {
                      release_arrow_children(*array);
                      delete[] array->buffers;
                      array->release = nullptr;
                    },
                    nullptr};
}

ArrowArray *numeric_column(std::span<const int32_t> values) {
  return make_fixed_column("i", "", bytes_of(values), sizeof(int32_t),
                           values.size());
}
ArrowArray *numeric_column(std::span<const uint32_t> values) {
  return make_fixed_column("I", "", bytes_of(values), sizeof(uint32_t),
                           values.size());
}
ArrowArray *numeric_column(std::span<const int64_t> values) {
  return make_fixed_column("l", "", bytes_of(values), sizeof(int64_t),
                           values.size());
}
ArrowArray *numeric_column(std::span<const uint64_t> values) {
  return make_fixed_column("L", "", bytes_of(values), sizeof(uint64_t),
                           values.size());
}

inline size_t fixed_column_width(const ArrowSchema *child_schema) {
  if (!child_schema || !child_schema->format)
    return 0;
  switch (child_schema->format[0]) {
  case 'i':
    return sizeof(int32_t);
  case 'I':
    return sizeof(uint32_t);
  case 'l':
    return sizeof(int64_t);
  case 'L':
    return sizeof(uint64_t);
  default:
    return 0;
  }
}

inline uint64_t read_polymorphic_key(const ArrowArray *array,
                                     const ArrowSchema *schema, int64_t row) {
  const char *fmt = schema->format;
  if (!fmt)
    throw std::invalid_argument(
        "missing format in Arrow schema for key column");
  switch (fmt[0]) {
  case 'i':
    return static_cast<uint64_t>(fixed_value<int32_t>(array, row));
  case 'I':
    return static_cast<uint64_t>(fixed_value<uint32_t>(array, row));
  case 'l':
    return static_cast<uint64_t>(fixed_value<int64_t>(array, row));
  case 'L':
    return fixed_value<uint64_t>(array, row);
  default:
    throw std::invalid_argument("unsupported Arrow format for key column: " +
                                std::string(fmt));
  }
}

ArrowArray copy_selected_rows(const ArrowSchema &schema,
                              const ArrowArray &source,
                              std::span<const int64_t> selected) {
  std::vector<ArrowArray *> children;
  children.reserve(static_cast<size_t>(schema.n_children));
  for (int64_t child_index = 0; child_index < schema.n_children;
       ++child_index) {
    const auto *child_schema = arrow_schema_field(schema, child_index);
    const auto *child = arrow_array_field(source, child_index);
    if (child_schema->format &&
        (child_schema->format[0] == 'u' || child_schema->format[0] == 'z')) {
      std::vector<std::byte> data;
      std::vector<int32_t> offsets{0};
      for (const int64_t row : selected) {
        const auto value = binary_value(child, row);
        if (data.size() > static_cast<size_t>(INT32_MAX) - value.size())
          throw std::length_error(
              "native Lance compaction binary column exceeds Arrow range");
        data.insert(data.end(), value.begin(), value.end());
        offsets.push_back(static_cast<int32_t>(data.size()));
      }
      children.push_back(
          make_binary_column(std::move(data), std::move(offsets)));
      continue;
    }
    const size_t width = fixed_column_width(child_schema);
    if (width == 0)
      throw std::runtime_error(
          "native Lance compaction column format is unsupported");
    std::vector<std::byte> data;
    data.reserve(selected.size() * width);
    for (const int64_t row : selected) {
      const auto bytes = ArrowArrayView(child).bytes(width);
      const auto *begin = bytes.data() + static_cast<size_t>(row) * width;
      data.insert(data.end(), begin, begin + width);
    }
    children.push_back(make_fixed_column(child_schema->format,
                                         child_schema->name, std::move(data),
                                         width, selected.size()));
  }
  return make_struct_array(std::move(children), selected.size());
}

ArrowArray make_empty_array(const ArrowSchema &schema) {
  std::vector<ArrowArray *> children;
  children.reserve(static_cast<size_t>(schema.n_children));
  for (int64_t child_index = 0; child_index < schema.n_children;
       ++child_index) {
    const auto *child_schema = arrow_schema_field(schema, child_index);
    if (child_schema->format &&
        (child_schema->format[0] == 'u' || child_schema->format[0] == 'z')) {
      children.push_back(make_binary_column({}, {0}));
      continue;
    }
    const auto width = fixed_column_width(child_schema);
    if (width == 0)
      throw std::runtime_error(
          "native Lance compaction column format is unsupported");
    children.push_back(make_fixed_column(child_schema->format,
                                         child_schema->name, {}, width, 0));
  }
  return make_struct_array(std::move(children), 0);
}

template <typename T> class BoundedChannel {
  std::mutex mutex_;
  std::condition_variable cv_push_;
  std::condition_variable cv_pop_;
  std::queue<T> queue_;
  size_t capacity_;
  bool closed_ = false;
  std::exception_ptr error_ = nullptr;

public:
  explicit BoundedChannel(size_t capacity = 8) : capacity_(capacity) {}

  void push(T item) {
    std::unique_lock lock(mutex_);
    cv_push_.wait(lock,
                  [this] { return queue_.size() < capacity_ || closed_; });
    if (error_)
      std::rethrow_exception(error_);
    if (closed_)
      throw std::runtime_error("push to closed channel");
    queue_.push(std::move(item));
    cv_pop_.notify_one();
  }

  bool pop(T &item) {
    std::unique_lock lock(mutex_);
    cv_pop_.wait(lock, [this] { return !queue_.empty() || closed_; });
    if (!queue_.empty()) {
      item = std::move(queue_.front());
      queue_.pop();
      cv_push_.notify_one();
      return true;
    }
    if (error_)
      std::rethrow_exception(error_);
    return false;
  }

  void close() {
    std::unique_lock lock(mutex_);
    closed_ = true;
    cv_push_.notify_all();
    cv_pop_.notify_all();
  }

  void set_error(std::exception_ptr err) {
    std::unique_lock lock(mutex_);
    error_ = err;
    closed_ = true;
    cv_push_.notify_all();
    cv_pop_.notify_all();
  }

  void clear_and_release() {
    std::unique_lock lock(mutex_);
    closed_ = true;
    while (!queue_.empty()) {
      auto &val = queue_.front();
      if (val.has_value() && val->release) {
        val->release(&*val);
      }
      queue_.pop();
    }
    cv_push_.notify_all();
    cv_pop_.notify_all();
  }
};

void stop_stream(BoundedChannel<std::optional<ArrowArray>> &channel,
                 std::thread &worker, LanceDataset *&dataset) noexcept {
  channel.clear_and_release();
  if (worker.joinable())
    worker.join();
  if (dataset) {
    lance_dataset_close(dataset);
    dataset = nullptr;
  }
}

struct ChannelStreamState {
  BoundedChannel<std::optional<ArrowArray>> *channel = nullptr;
  ArrowSchema schema{};
};

struct CompactionWriterGuard {
  BoundedChannel<std::optional<ArrowArray>> &channel;
  std::thread &worker;
  LanceDataset *&dataset;

  ~CompactionWriterGuard() noexcept { stop_stream(channel, worker, dataset); }
};

int channel_stream_schema(ArrowArrayStream *stream, ArrowSchema *out) {
  auto *state = static_cast<ChannelStreamState *>(stream->private_data);
  *out = state->schema;
  state->schema.release = nullptr;
  return 0;
}

int channel_stream_next(ArrowArrayStream *stream, ArrowArray *out) {
  auto *state = static_cast<ChannelStreamState *>(stream->private_data);
  std::optional<ArrowArray> batch;
  if (state->channel->pop(batch) && batch.has_value()) {
    *out = *batch;
    return 0;
  }
  *out = ArrowArray{};
  return 0;
}

const char *channel_stream_error(ArrowArrayStream *) { return nullptr; }

void channel_stream_release(ArrowArrayStream *stream) {
  auto *state = static_cast<ChannelStreamState *>(stream->private_data);
  if (state) {
    if (state->schema.release)
      state->schema.release(&state->schema);
    delete state;
  }
  stream->private_data = nullptr;
  stream->release = nullptr;
}

} // namespace

struct PipelinedWriterState {
  std::string root;
  std::string entries_uri;
  std::string pages_uri;
  uint64_t page_dataset_version = 0;
  std::string embedding_dtype;
  std::string optimizer_dtype;
  std::string key_dtype;
  uint32_t participant_index = 0;
  bool has_score = false;
  bool include_page_images = false;
  std::vector<char> schema_metadata;
  NativeFragmentArtifact artifact;
  uint64_t content_hash = 1469598103934665603ULL;
  struct MutationKey {
    uint32_t dimension;
    uint64_t key;
    bool operator==(const MutationKey &) const = default;
  };
  struct MutationKeyHash {
    size_t operator()(const MutationKey &value) const noexcept {
      size_t hash = std::hash<uint64_t>{}(value.key);
      hash ^= std::hash<uint32_t>{}(value.dimension) +
              static_cast<size_t>(0x9e3779b9) + (hash << 6) + (hash >> 2);
      return hash;
    }
  };
  std::unordered_map<MutationKey, uint64_t, MutationKeyHash>
      incremental_sequences;
  bool incremental_filter_enabled = false;

  BoundedChannel<std::optional<ArrowArray>> entries_channel{
      CheckpointStreamPolicy::kChannelCapacity};
  std::thread entries_worker;
  LanceDataset *entries_dataset = nullptr;
  std::exception_ptr entries_error = nullptr;

  std::unique_ptr<BoundedChannel<std::optional<ArrowArray>>> pages_channel;
  std::thread pages_worker;
  LanceDataset *pages_dataset = nullptr;
  std::exception_ptr pages_error = nullptr;

  ~PipelinedWriterState() { abort(); }

  void abort() {
    stop_stream(entries_channel, entries_worker, entries_dataset);

    if (pages_channel) {
      stop_stream(*pages_channel, pages_worker, pages_dataset);
    }
  }
};

void NativeFragmentWriter::release_schema(ArrowSchema *schema) {
  if (!schema || !schema->release)
    return;
  release_arrow_schema_children(*schema);
  schema->release = nullptr;
}

void NativeFragmentWriter::release_array(ArrowArray *array) {
  if (!array || !array->release)
    return;
  release_arrow_children(*array);
  delete[] array->buffers;
  array->release = nullptr;
}

NativeFragmentWriter::NativeFragmentWriter() = default;
NativeFragmentWriter::~NativeFragmentWriter() = default;
NativeFragmentWriter::NativeFragmentWriter(NativeFragmentWriter &&) noexcept =
    default;
NativeFragmentWriter &
NativeFragmentWriter::operator=(NativeFragmentWriter &&) noexcept = default;

void NativeFragmentWriter::set_incremental_mutations(
    std::span<const CheckpointMutation> mutations) {
  if (!state_)
    throw std::runtime_error(
        "incremental mutations require an active fragment");
  state_->incremental_sequences.clear();
  state_->incremental_filter_enabled = true;
  std::vector<CheckpointMutation> ordered(mutations.begin(), mutations.end());
  std::ranges::sort(ordered, [](const auto &left, const auto &right) {
    return std::tie(left.dimension, left.key, left.mutation_sequence,
                    left.deleted) < std::tie(right.dimension, right.key,
                                             right.mutation_sequence,
                                             right.deleted);
  });
  for (const auto &mutation : ordered) {
    if (mutation.deleted) {
      write_tombstone(mutation.dimension, mutation.key,
                      mutation.mutation_sequence);
    } else if (!state_->incremental_sequences
                    .emplace(
                        PipelinedWriterState::MutationKey{mutation.dimension,
                                                          mutation.key},
                        mutation.mutation_sequence)
                    .second) {
      throw std::invalid_argument("duplicate checkpoint mutation identity");
    }
  }
}

void NativeFragmentWriter::write_tombstone(uint32_t dimension, uint64_t key,
                                           uint64_t mutation_sequence) {
  if (!state_)
    throw std::runtime_error("tombstone write requires an active fragment");
  std::vector<ArrowArray *> children;
  children.reserve(se::lance_layout::kEntryColumns.size());
  children.push_back(numeric_column(std::span<const uint32_t>(&dimension, 1)));
  const auto kdt = parse_key_dtype(state_->key_dtype);
  se::dispatch_key_type(kdt, [&]<typename KeyT>() {
    const KeyT typed_key = static_cast<KeyT>(key);
    children.push_back(numeric_column(std::span<const KeyT>(&typed_key, 1)));
  });
  children.push_back(make_strided_binary_column({}, 0, 1));
  children.push_back(
      make_repeated_binary_column(string_bytes(state_->embedding_dtype), 1));
  children.push_back(make_strided_binary_column({}, 0, 1));
  children.push_back(
      make_repeated_binary_column(string_bytes(state_->optimizer_dtype), 1));
  const uint32_t optimizer_stride = 0;
  children.push_back(
      numeric_column(std::span<const uint32_t>(&optimizer_stride, 1)));
  const uint32_t step = 0;
  children.push_back(numeric_column(std::span<const uint32_t>(&step, 1)));
  const uint64_t zero = 0;
  children.push_back(numeric_column(std::span<const uint64_t>(&zero, 1)));
  children.push_back(numeric_column(std::span<const uint64_t>(&zero, 1)));
  children.push_back(numeric_column(std::span<const uint64_t>(&zero, 1)));
  const uint32_t slot = 0;
  children.push_back(numeric_column(std::span<const uint32_t>(&slot, 1)));
  children.push_back(make_repeated_binary_column(string_bytes("deleted"), 1));
  children.push_back(
      numeric_column(std::span<const uint32_t>(&state_->participant_index, 1)));
  children.push_back(
      numeric_column(std::span<const uint64_t>(&mutation_sequence, 1)));
  start_entries();
  state_->entries_channel.push(make_struct_array(std::move(children), 1));
  ++state_->artifact.entry_count;
  const auto canonical = canonical_entry_tombstone(
      state_->participant_index, dimension, key, mutation_sequence,
      state_->embedding_dtype, state_->optimizer_dtype);
  state_->content_hash = fnv1a(canonical, state_->content_hash);
}

void NativeFragmentWriter::begin_fragment(
    std::string_view staging_uri, uint32_t participant_index,
    std::span<const uint32_t> dimensions, std::string_view embedding_dtype,
    std::string_view optimizer_dtype, std::string_view score_policy,
    std::string_view key_dtype, std::string_view schema_digest, bool has_score,
    bool include_page_images, uint32_t entries_per_page,
    std::string_view page_dataset_uri, uint64_t page_dataset_version) {
  const storage::PageGeometry geometry(entries_per_page);
  if (include_page_images && page_dataset_uri.empty())
    throw std::invalid_argument(
        "page dataset URI is required when page images are enabled");
  if (!include_page_images && !page_dataset_uri.empty())
    throw std::invalid_argument(
        "page dataset URI is invalid when page images are disabled");
  state_ = std::make_unique<PipelinedWriterState>();
  state_->root = std::string(staging_uri);
  state_->entries_uri =
      state_->root + "/" + std::string(lance_layout::kEntriesDataset);
  state_->pages_uri = std::string(page_dataset_uri);
  state_->page_dataset_version = page_dataset_version;
  state_->embedding_dtype = std::string(embedding_dtype);
  state_->optimizer_dtype = std::string(optimizer_dtype);
  state_->key_dtype = std::string(key_dtype);
  state_->participant_index = participant_index;
  state_->has_score = has_score;
  state_->include_page_images = include_page_images;
  state_->artifact.entries_per_page =
      static_cast<uint32_t>(geometry.entries_per_page());
  state_->artifact.locator = state_->root;
  state_->artifact.schema_digest = std::string(schema_digest);

  if (state_->root.find("://") == std::string::npos)
    std::filesystem::create_directories(state_->root);

  const std::string dimensions_value = join_dimensions(dimensions);
  (void)parse_key_dtype(key_dtype);
  const std::array<std::pair<std::string, std::string>, 5> metadata = {{
      {"sparse_embedding.dimensions", dimensions_value},
      {"sparse_embedding.embedding_dtype", std::string(embedding_dtype)},
      {"sparse_embedding.optimizer_dtype", std::string(optimizer_dtype)},
      {"sparse_embedding.score_policy", std::string(score_policy)},
      {"sparse_embedding.key_dtype", std::string(key_dtype)},
  }};
  state_->schema_metadata = make_arrow_metadata(metadata);
}

void NativeFragmentWriter::start_entries() {
  if (state_->entries_worker.joinable())
    return;
  state_->entries_worker = std::thread([state = state_.get()]() {
    try {
      ArrowSchema schema = se::lance_layout::make_entries_schema(
          state->key_dtype, state->schema_metadata.empty()
                                ? nullptr
                                : state->schema_metadata.data());
      ArrowSchema stream_sc = se::lance_layout::make_entries_schema(
          state->key_dtype, state->schema_metadata.empty()
                                ? nullptr
                                : state->schema_metadata.data());
      auto *stream_state =
          new ChannelStreamState{&state->entries_channel, stream_sc};
      ArrowArrayStream stream{&channel_stream_schema, &channel_stream_next,
                              &channel_stream_error, &channel_stream_release,
                              stream_state};
      check_lance(lance_dataset_write(state->entries_uri.c_str(), &schema,
                                      &stream, LANCE_WRITE_CREATE, nullptr,
                                      &state->entries_dataset),
                  "native Lance fragment write entries");
      if (schema.release)
        schema.release(&schema);
    } catch (...) {
      state->entries_error = std::current_exception();
      state->entries_channel.set_error(state->entries_error);
    }
  });
}

void NativeFragmentWriter::start_pages() {
  if (state_->pages_worker.joinable())
    return;
  if (!state_->include_page_images)
    throw std::logic_error("page writer is disabled");
  {
    state_->pages_channel =
        std::make_unique<BoundedChannel<std::optional<ArrowArray>>>(
            CheckpointStreamPolicy::kChannelCapacity);
    state_->pages_worker = std::thread([state = state_.get()]() {
      try {
        ArrowSchema schema = se::lance_layout::make_page_images_schema(
            state->schema_metadata.empty() ? nullptr
                                           : state->schema_metadata.data());
        ArrowSchema stream_sc = se::lance_layout::make_page_images_schema(
            state->schema_metadata.empty() ? nullptr
                                           : state->schema_metadata.data());
        auto *stream_state =
            new ChannelStreamState{state->pages_channel.get(), stream_sc};
        ArrowArrayStream stream{&channel_stream_schema, &channel_stream_next,
                                &channel_stream_error, &channel_stream_release,
                                stream_state};
        if (state->page_dataset_version == 0) {
          check_lance(lance_dataset_write(state->pages_uri.c_str(), &schema,
                                          &stream, LANCE_WRITE_CREATE, nullptr,
                                          &state->pages_dataset),
                      "native Lance fragment write pages");
        } else {
          state->pages_dataset = lance_dataset_open(
              state->pages_uri.c_str(), nullptr, state->page_dataset_version);
          if (state->pages_dataset == nullptr)
            check_lance(-1, "native Lance page parent open");
          const char *const on_columns[] = {
              "source_participant_index", "embedding_dim", "physical_page_id"};
          const LanceMergeInsertParams params{
              LANCE_MERGE_WHEN_MATCHED_UPDATE_ALL, nullptr,
              LANCE_MERGE_WHEN_NOT_MATCHED_INSERT_ALL,
              LANCE_MERGE_WHEN_NOT_MATCHED_BY_SOURCE_KEEP, nullptr};
          check_lance(lance_dataset_merge_insert(state->pages_dataset,
                                                 on_columns, 3, &stream,
                                                 &params, nullptr),
                      "native Lance page snapshot merge");
        }
        if (schema.release)
          schema.release(&schema);
      } catch (...) {
        state->pages_error = std::current_exception();
        state->pages_channel->set_error(state->pages_error);
      }
    });
  }
}

void NativeFragmentWriter::write_page_image(
    size_t embedding_dim, uint64_t physical_page_id,
    uint32_t page_slot_stride_bytes, size_t embedding_row_bytes,
    std::span<const uint32_t> slots, std::span<const uint64_t> keys,
    std::span<const uint64_t> sequences, std::span<const std::byte> values,
    std::span<const uint64_t> scores, std::span<const std::byte> optimizer,
    size_t optimizer_stride, size_t optimizer_itemsize,
    std::span<const uint32_t> steps) {
  if (!state_)
    throw std::runtime_error("write_page_image called without begin_fragment");
  if (keys.empty())
    throw std::invalid_argument("native Lance page image must not be empty");
  if (slots.size() != keys.size() || keys.size() != sequences.size())
    throw std::invalid_argument("native Lance page index shape mismatch");
  if (page_slot_stride_bytes == 0 || embedding_row_bytes == 0 ||
      values.size() != keys.size() * embedding_row_bytes)
    throw std::invalid_argument("native Lance page value shape mismatch");
  if (!scores.empty() && scores.size() != keys.size())
    throw std::invalid_argument("native Lance page score shape mismatch");
  if (!steps.empty() && steps.size() != keys.size())
    throw std::invalid_argument("native Lance page step shape mismatch");

  const auto slot_bytes = std::as_bytes(slots);
  const auto key_bytes = std::as_bytes(keys);
  const auto sequence_bytes = std::as_bytes(sequences);
  const auto value_bytes = values;
  const auto score_bytes = std::as_bytes(scores);
  const auto optimizer_bytes = optimizer;
  const auto step_bytes = std::as_bytes(steps);
  std::string page_digest =
      checksum(slot_bytes, key_bytes, sequence_bytes, value_bytes, score_bytes,
               optimizer_bytes, step_bytes);
  std::vector<uint32_t> selected_slots;
  std::vector<uint64_t> selected_keys;
  std::vector<uint64_t> selected_sequences;
  std::vector<uint64_t> selected_scores;
  std::vector<uint32_t> selected_steps;
  std::vector<std::byte> selected_values;
  std::vector<std::byte> selected_optimizer;
  if (state_->incremental_filter_enabled) {
    const size_t value_stride = embedding_row_bytes;
    const size_t optimizer_stride_bytes = optimizer_stride * optimizer_itemsize;
    for (size_t index = 0; index < keys.size(); ++index) {
      if (!state_->incremental_sequences.contains(
              PipelinedWriterState::MutationKey{
                  static_cast<uint32_t>(embedding_dim), keys[index]}))
        continue;
      selected_slots.push_back(slots[index]);
      selected_keys.push_back(keys[index]);
      selected_sequences.push_back(sequences[index]);
      if (!scores.empty())
        selected_scores.push_back(scores[index]);
      if (!steps.empty())
        selected_steps.push_back(steps[index]);
      selected_values.insert(selected_values.end(),
                             values.begin() + index * value_stride,
                             values.begin() + (index + 1) * value_stride);
      if (optimizer_stride_bytes != 0)
        selected_optimizer.insert(
            selected_optimizer.end(),
            optimizer.begin() + index * optimizer_stride_bytes,
            optimizer.begin() + (index + 1) * optimizer_stride_bytes);
    }
    slots = std::span<const uint32_t>(selected_slots);
    keys = std::span<const uint64_t>(selected_keys);
    sequences = std::span<const uint64_t>(selected_sequences);
    scores = std::span<const uint64_t>(selected_scores);
    steps = std::span<const uint32_t>(selected_steps);
    values = std::span<const std::byte>(selected_values);
    optimizer = std::span<const std::byte>(selected_optimizer);
  }

  const size_t optimizer_stride_bytes = optimizer_stride * optimizer_itemsize;
  if (optimizer_stride_bytes != 0 &&
      optimizer.size() != keys.size() * optimizer_stride_bytes)
    throw std::invalid_argument(
        "native Lance optimizer payload shape mismatch");

  if (!keys.empty()) {
    const size_t count = keys.size();
    std::vector<ArrowArray *> children;
    children.reserve(se::lance_layout::kEntryColumns.size());
    children.push_back(numeric_column(
        std::vector<uint32_t>(count, static_cast<uint32_t>(embedding_dim))));

    const auto kdt = parse_key_dtype(state_->key_dtype);
    se::dispatch_key_type(kdt, [&]<typename KeyT>() {
      std::vector<KeyT> typed_keys;
      typed_keys.reserve(count);
      for (uint64_t k : keys) {
        typed_keys.push_back(static_cast<KeyT>(k));
      }
      children.push_back(numeric_column(typed_keys));
    });
    children.push_back(
        make_strided_binary_column(values, embedding_row_bytes, count));
    const auto embedding_dtype_bytes = string_bytes(state_->embedding_dtype);
    children.push_back(
        make_repeated_binary_column(embedding_dtype_bytes, count));
    children.push_back(
        make_strided_binary_column(optimizer, optimizer_stride_bytes, count));
    const auto optimizer_dtype_bytes = string_bytes(state_->optimizer_dtype);
    children.push_back(
        make_repeated_binary_column(optimizer_dtype_bytes, count));
    const uint32_t opt_stride_val =
        optimizer_stride_bytes == 0 || state_->optimizer_dtype.empty()
            ? 0
            : static_cast<uint32_t>(optimizer_stride_bytes /
                                    dtype_itemsize(state_->optimizer_dtype));
    std::vector<uint32_t> opt_strides(count, opt_stride_val);
    children.push_back(numeric_column(opt_strides));
    std::vector<uint32_t> default_steps(count, 0);
    children.push_back(numeric_column(
        steps.empty() ? std::span<const uint32_t>{default_steps} : steps));
    std::vector<uint64_t> default_scores(count, 0ULL);
    children.push_back(numeric_column(
        scores.empty() ? std::span<const uint64_t>{default_scores} : scores));
    children.push_back(numeric_column(sequences));
    std::vector<uint64_t> page_ids(count, physical_page_id);
    children.push_back(numeric_column(page_ids));
    children.push_back(numeric_column(slots));
    const auto active_row_state = string_bytes("active");
    children.push_back(make_repeated_binary_column(active_row_state, count));
    std::vector<uint32_t> part_indices(count, state_->participant_index);
    children.push_back(numeric_column(part_indices));
    std::vector<uint64_t> mutation_sequences(count, 0ULL);
    if (state_->incremental_filter_enabled)
      for (size_t index = 0; index < count; ++index)
        mutation_sequences[index] =
            state_->incremental_sequences.at(PipelinedWriterState::MutationKey{
                static_cast<uint32_t>(embedding_dim), keys[index]});
    children.push_back(numeric_column(mutation_sequences));

    auto entries_array = make_struct_array(std::move(children), count);
    state_->artifact.entry_count += count;
    start_entries();
    state_->entries_channel.push(std::move(entries_array));
  }

  if (state_->include_page_images) {
    start_pages();
    const size_t p_count = 1;
    std::vector<ArrowArray *> p_children;
    p_children.reserve(se::lance_layout::kPageImageColumns.size());
    std::vector<uint32_t> part_col(p_count, state_->participant_index);
    p_children.push_back(numeric_column(part_col));
    std::vector<uint32_t> dim_col(p_count,
                                  static_cast<uint32_t>(embedding_dim));
    p_children.push_back(numeric_column(dim_col));
    std::vector<uint64_t> page_col(p_count, physical_page_id);
    p_children.push_back(numeric_column(page_col));
    std::vector<uint32_t> stride_col(p_count, page_slot_stride_bytes);
    p_children.push_back(numeric_column(stride_col));
    auto single_row = [](std::span<const std::byte> val) {
      return p_count == 0 ? make_strided_binary_column({}, 0, 0)
                          : make_single_binary_column(val);
    };
    auto scalar = [&](std::string_view val) {
      const auto bytes = string_bytes(val);
      return single_row(bytes);
    };
    p_children.push_back(scalar(state_->embedding_dtype));
    p_children.push_back(single_row(slot_bytes));
    p_children.push_back(single_row(key_bytes));
    p_children.push_back(single_row(sequence_bytes));
    p_children.push_back(single_row(value_bytes));
    p_children.push_back(single_row(score_bytes));
    p_children.push_back(single_row(optimizer_bytes));
    p_children.push_back(scalar(state_->optimizer_dtype));
    std::vector<uint32_t> opt_stride_col(
        p_count, static_cast<uint32_t>(optimizer_stride_bytes));
    p_children.push_back(numeric_column(opt_stride_col));
    p_children.push_back(single_row(step_bytes));
    p_children.push_back(scalar(page_digest));
    p_children.push_back(scalar("active"));
    p_children.push_back(numeric_column(
        std::vector<uint32_t>(p_count, state_->artifact.entries_per_page)));
    auto page_array = make_struct_array(std::move(p_children), p_count);
    state_->artifact.page_count += 1;
    state_->pages_channel->push(std::move(page_array));
  }

  state_->content_hash = fnv1a(slot_bytes, state_->content_hash);
  state_->content_hash = fnv1a(key_bytes, state_->content_hash);
  state_->content_hash = fnv1a(sequence_bytes, state_->content_hash);
  state_->content_hash = fnv1a(value_bytes, state_->content_hash);
  state_->content_hash = fnv1a(score_bytes, state_->content_hash);
  state_->content_hash = fnv1a(optimizer_bytes, state_->content_hash);
  state_->content_hash = fnv1a(step_bytes, state_->content_hash);

  std::array<char, 17> digest{};
  auto [end, error] = std::to_chars(digest.data(), digest.data() + 16,
                                    state_->content_hash, 16);
  if (error != std::errc{})
    throw std::runtime_error("failed to format native fragment digest");
  state_->artifact.content_digest.assign(digest.data(), end);
}

void NativeFragmentWriter::write_page_tombstone(size_t embedding_dim,
                                                uint64_t physical_page_id,
                                                uint32_t page_slot_stride_bytes,
                                                uint32_t entries_per_page) {
  if (!state_)
    throw std::runtime_error(
        "write_page_tombstone called without begin_fragment");
  if (!state_->include_page_images)
    throw std::logic_error("page tombstone requires page images");
  if (page_slot_stride_bytes == 0 || entries_per_page == 0)
    throw std::invalid_argument(
        "native Lance page tombstone geometry is invalid");

  start_pages();
  const size_t count = 1;
  std::vector<ArrowArray *> children;
  children.reserve(se::lance_layout::kPageImageColumns.size());
  children.push_back(
      numeric_column(std::vector<uint32_t>(count, state_->participant_index)));
  children.push_back(
      numeric_column(std::vector<uint32_t>(count, embedding_dim)));
  children.push_back(
      numeric_column(std::vector<uint64_t>(count, physical_page_id)));
  children.push_back(
      numeric_column(std::vector<uint32_t>(count, page_slot_stride_bytes)));
  const auto scalar = [](std::string_view value) {
    const auto bytes = string_bytes(value);
    return count == 0 ? make_strided_binary_column({}, 0, 0)
                      : make_single_binary_column(bytes);
  };
  const auto empty = [] { return make_single_binary_column({}); };
  children.push_back(scalar(state_->embedding_dtype));
  children.push_back(empty());
  children.push_back(empty());
  children.push_back(empty());
  children.push_back(empty());
  children.push_back(empty());
  children.push_back(empty());
  children.push_back(scalar(state_->optimizer_dtype));
  children.push_back(numeric_column(std::vector<uint32_t>(count, 0)));
  children.push_back(empty());
  children.push_back(scalar(checksum({}, {}, {}, {}, {}, {}, {})));
  children.push_back(scalar("deleted"));
  children.push_back(
      numeric_column(std::vector<uint32_t>(count, entries_per_page)));
  auto page_array = make_struct_array(std::move(children), count);
  ++state_->artifact.page_count;
  state_->pages_channel->push(std::move(page_array));
}

NativeFragmentArtifact NativeFragmentWriter::finalize() {
  if (!state_)
    throw std::runtime_error("finalize called without active fragment");

  if (state_->entries_worker.joinable()) {
    state_->entries_channel.push(std::nullopt);
    state_->entries_channel.close();
  }
  if (state_->entries_worker.joinable())
    state_->entries_worker.join();
  if (state_->entries_error)
    std::rethrow_exception(state_->entries_error);

  if (state_->entries_dataset) {
    state_->artifact.entries_version =
        lance_dataset_version(state_->entries_dataset);
    lance_dataset_close(state_->entries_dataset);
    state_->entries_dataset = nullptr;
  }

  if (state_->pages_worker.joinable()) {
    state_->pages_channel->push(std::nullopt);
    state_->pages_channel->close();
    if (state_->pages_worker.joinable())
      state_->pages_worker.join();
    if (state_->pages_error)
      std::rethrow_exception(state_->pages_error);

    if (state_->pages_dataset) {
      state_->artifact.page_images_version =
          lance_dataset_version(state_->pages_dataset);
      lance_dataset_close(state_->pages_dataset);
      state_->pages_dataset = nullptr;
    }
  } else {
    state_->artifact.page_images_version = 0;
  }

  if (state_->artifact.content_digest.empty()) {
    std::array<char, 17> digest{};
    auto [end, error] = std::to_chars(digest.data(), digest.data() + 16,
                                      state_->content_hash, 16);
    if (error != std::errc{})
      throw std::runtime_error("failed to format empty native fragment digest");
    state_->artifact.content_digest.assign(digest.data(), end);
  }

  state_->artifact.physical_bytes = 0;
  if (state_->root.find("://") == std::string::npos) {
    for (const auto &path :
         std::filesystem::recursive_directory_iterator(state_->root))
      if (path.is_regular_file())
        state_->artifact.physical_bytes += path.file_size();
  }

  return std::move(state_->artifact);
}

NativeCompactionArtifact
compact_native_lance_sources(std::string_view target_uri,
                             std::span<const std::string> source_uris,
                             const LanceStorageOptions &storage_options) {
  if (target_uri.empty() || source_uris.empty())
    throw std::invalid_argument(
        "native Lance compaction requires target and sources");
  NativeCompactionArtifact result;
  result.locator = std::string(target_uri);
  uint64_t content_hash = 1469598103934665603ULL;
  const char *const *options = storage_options.lance_options();

  std::vector<char> output_metadata;
  bool is_entries_dataset = false;
  bool schema_initialized = false;
  std::string compaction_key_dtype = "uint64";

  BoundedChannel<std::optional<ArrowArray>> channel{
      CheckpointStreamPolicy::kChannelCapacity};
  std::exception_ptr writer_error;
  LanceDataset *target_dataset = nullptr;
  std::thread writer_thread;
  CompactionWriterGuard writer_guard{channel, writer_thread, target_dataset};

  for (const auto &source_uri : source_uris) {
    std::unique_ptr<LanceDataset, DatasetDeleter> source(
        lance_dataset_open(source_uri.c_str(), options, 0));
    if (!source) {
      const char *message = lance_last_error_message();
      std::string detail = message ? message : "unknown Lance error";
      if (message)
        lance_free_string(message);
      throw std::runtime_error(
          "native Lance compaction source open failed for " + source_uri +
          ": " + detail);
    }
    std::unique_ptr<LanceScanner, ScannerDeleter> scanner(
        lance_scanner_new(source.get(), nullptr, nullptr));
    if (!scanner) {
      throw std::runtime_error(
          "native Lance compaction scanner creation failed");
    }
    lance_scanner_set_batch_size(scanner.get(),
                                 CheckpointStreamPolicy::kScanBatchRows);
    ArrowStreamGuard stream;
    if (lance_scanner_to_arrow_stream(scanner.get(), stream.get()) != 0) {
      check_lance(-1, "native Lance compaction scanner");
    }
    ArrowSchemaGuard schema;
    if (stream.get()->get_schema(stream.get(), schema.get()) != 0)
      throw std::runtime_error("native Lance compaction schema read failed");
    std::optional<size_t> dim_index;
    std::optional<size_t> key_index;
    std::optional<size_t> state_index;
    for (size_t index = 0;
         index < static_cast<size_t>(schema.get()->n_children); ++index) {
      const char *name = arrow_schema_field(*schema.get(), index)->name;
      if (!name)
        continue;
      if (std::string_view(name) == "dim")
        dim_index = index;
      else if (std::string_view(name) == "key")
        key_index = index;
      else if (std::string_view(name) == "row_state")
        state_index = index;
    }
    const bool is_entries = dim_index.has_value() && key_index.has_value() &&
                            state_index.has_value();
    if (!is_entries &&
        schema.get()->n_children !=
            static_cast<int64_t>(lance_layout::kPageImageColumns.size()))
      throw std::runtime_error(
          "native Lance compaction source schema is unsupported");

    if (!schema_initialized) {
      if (schema.get()->metadata)
        output_metadata = clone_arrow_metadata(schema.get()->metadata);
      is_entries_dataset = is_entries;
      if (is_entries && key_index.has_value()) {
        const char *fmt = arrow_schema_field(*schema.get(), *key_index)->format;
        if (fmt) {
          if (std::string_view(fmt) == "i")
            compaction_key_dtype = "int32";
          else if (std::string_view(fmt) == "I")
            compaction_key_dtype = "uint32";
          else if (std::string_view(fmt) == "l")
            compaction_key_dtype = "int64";
          else if (std::string_view(fmt) == "L")
            compaction_key_dtype = "uint64";
        }
      }
      schema_initialized = true;

      writer_thread = std::thread([&, target_uri_str = std::string(target_uri),
                                   compaction_key_dtype]() {
        try {
          ArrowSchema output_schema =
              is_entries_dataset
                  ? se::lance_layout::make_entries_schema(
                        compaction_key_dtype, output_metadata.empty()
                                                  ? nullptr
                                                  : output_metadata.data())
                  : se::lance_layout::make_page_images_schema(
                        output_metadata.empty() ? nullptr
                                                : output_metadata.data());
          ArrowSchema stream_sc =
              is_entries_dataset
                  ? se::lance_layout::make_entries_schema(
                        compaction_key_dtype, output_metadata.empty()
                                                  ? nullptr
                                                  : output_metadata.data())
                  : se::lance_layout::make_page_images_schema(
                        output_metadata.empty() ? nullptr
                                                : output_metadata.data());
          auto *stream_state = new ChannelStreamState{&channel, stream_sc};
          ArrowArrayStream out_stream{
              &channel_stream_schema, &channel_stream_next,
              &channel_stream_error, &channel_stream_release, stream_state};
          check_lance(lance_dataset_write(
                          target_uri_str.c_str(), &output_schema, &out_stream,
                          LANCE_WRITE_CREATE, options, &target_dataset),
                      "native Lance compaction write");
          if (output_schema.release)
            output_schema.release(&output_schema);
        } catch (...) {
          writer_error = std::current_exception();
          channel.set_error(writer_error);
        }
      });
    }

    for (;;) {
      ArrowArray batch{};
      if (stream.get()->get_next(stream.get(), &batch) != 0)
        throw std::runtime_error(
            "native Lance compaction batch read failed for " + source_uri);
      if (!batch.release)
        break;
      std::vector<int64_t> selected_rows;
      selected_rows.reserve(static_cast<size_t>(batch.length));
      if (is_entries) {
        for (int64_t row = 0; row < batch.length; ++row) {
          if (binary_string(arrow_array_field(batch, *state_index), row) !=
              kActiveRowState)
            continue;
          selected_rows.push_back(row);
        }
      } else {
        for (int64_t row = 0; row < batch.length; ++row)
          selected_rows.push_back(row);
      }
      ArrowArray selected =
          copy_selected_rows(*schema.get(), batch, selected_rows);
      const size_t row_count = selected_rows.size();
      for (int64_t col = 0; col < selected.n_children; ++col) {
        const ArrowArray *child = arrow_array_field(selected, col);
        const ArrowSchema *child_schema =
            arrow_schema_field(*schema.get(), col);
        if (child_schema->format && (child_schema->format[0] == 'z' ||
                                     child_schema->format[0] == 'u')) {
          for (int64_t row = 0; row < child->length; ++row) {
            content_hash =
                fnv1a(ArrowArrayView(child).binary().row(row), content_hash);
          }
          continue;
        }
        for (int64_t row = 0; row < child->length; ++row) {
          const size_t width = fixed_column_width(child_schema);
          if (width == 0)
            throw std::runtime_error(
                "native Lance compaction column format is unsupported");
          const auto bytes = ArrowArrayView(child).bytes(width);
          content_hash =
              fnv1a(std::span<const std::byte>(
                        bytes.data() + static_cast<size_t>(row) * width, width),
                    content_hash);
        }
      }
      batch.release(&batch);
      if (selected_rows.empty()) {
        if (selected.release)
          selected.release(&selected);
        continue;
      }
      channel.push(std::move(selected));
      result.inserted_rows += row_count;
    }
  }

  if (schema_initialized) {
    if (result.inserted_rows == 0) {
      ArrowSchema output_schema =
          is_entries_dataset
              ? se::lance_layout::make_entries_schema(
                    compaction_key_dtype,
                    output_metadata.empty() ? nullptr : output_metadata.data())
              : se::lance_layout::make_page_images_schema(
                    output_metadata.empty() ? nullptr : output_metadata.data());
      channel.push(make_empty_array(output_schema));
      if (output_schema.release)
        output_schema.release(&output_schema);
    }
    channel.push(std::nullopt);
    writer_thread.join();
    if (writer_error)
      std::rethrow_exception(writer_error);
    if (target_dataset) {
      result.target_version = lance_dataset_version(target_dataset);
      lance_dataset_close(target_dataset);
      target_dataset = nullptr;
    }
  }

  std::array<char, 17> digest{};
  auto [end, error] =
      std::to_chars(digest.data(), digest.data() + 16, content_hash, 16);
  if (error != std::errc{})
    throw std::runtime_error("failed to format native compaction digest");
  result.content_digest.assign(digest.data(), end);
  return result;
}

namespace {

template <typename Callback>
void scan_dataset(
    std::string_view uri, uint64_t version, Callback &&callback,
    const LanceStorageOptions &storage_options = LanceStorageOptions{},
    const char *const *columns = nullptr,
    int64_t batch_rows = CheckpointStreamPolicy::kScanBatchRows) {
  const char *const *options = storage_options.lance_options();
  std::unique_ptr<LanceDataset, DatasetDeleter> dataset(
      lance_dataset_open(std::string(uri).c_str(), options, version));
  if (!dataset)
    check_lance(-1, "native Lance restore dataset open");
  std::unique_ptr<LanceScanner, ScannerDeleter> scanner(
      lance_scanner_new(dataset.get(), columns, nullptr));
  if (!scanner)
    throw std::runtime_error("native Lance restore scanner creation failed");
  lance_scanner_set_batch_size(scanner.get(), batch_rows);
  ArrowStreamGuard stream;
  if (lance_scanner_to_arrow_stream(scanner.get(), stream.get()) != 0)
    check_lance(-1, "native Lance restore scanner");
  ArrowSchemaGuard schema;
  if (stream.get()->get_schema(stream.get(), schema.get()) != 0)
    throw std::runtime_error("native Lance restore schema read failed");
  callback(*schema.get(), *stream.get());
}

} // namespace

bool validate_native_checkpoint_chain(
    std::span<const std::vector<std::string>> source_uris,
    std::span<const std::vector<uint64_t>> participant_watermarks,
    std::span<const uint32_t> dimensions, std::string_view key_dtype,
    const LanceStorageOptions &storage_options,
    pro::proxy<se::ops::PartitionerFacade> target_partitioner,
    uint32_t target_partition_count) {
  if (source_uris.empty() ||
      source_uris.size() != participant_watermarks.size() ||
      source_uris.front().empty())
    throw std::invalid_argument(
        "checkpoint chain participant cuts are invalid");
  const size_t participant_count = source_uris.front().size();
  if (target_partitioner.has_value() != (target_partition_count != 0))
    throw std::invalid_argument("checkpoint target assignment is incomplete");
  const auto key_type = parse_key_dtype(key_dtype);
  bool preserves_source_ownership = target_partitioner.has_value() &&
                                    target_partition_count == participant_count;
  std::array<uint32_t, CheckpointStreamPolicy::kScanBatchRows>
      target_partitions;
  std::vector<uint64_t> previous_watermarks(participant_count, 0);
  checkpoint::ExternalSortedRecords<checkpoint::CheckpointEntryRecord>
      identities;
  checkpoint::ExternalSortedRecords<checkpoint::MutationSequenceIdentity>
      sequences;
  const char *const columns[] = {"dim",
                                 "key",
                                 "row_state",
                                 "mutation_sequence",
                                 "source_participant_index",
                                 nullptr};
  for (size_t checkpoint_index = 0; checkpoint_index < source_uris.size();
       ++checkpoint_index) {
    if (source_uris[checkpoint_index].size() != participant_count ||
        participant_watermarks[checkpoint_index].size() != participant_count)
      throw std::runtime_error("checkpoint chain participant topology changed");
    for (uint32_t participant = 0; participant < participant_count;
         ++participant) {
      const uint64_t watermark =
          participant_watermarks[checkpoint_index][participant];
      if (watermark < previous_watermarks[participant])
        throw std::runtime_error(
            "checkpoint participant watermark is retrograde");
      const auto &uri = source_uris[checkpoint_index][participant];
      if (uri.empty())
        continue;
      const uint64_t source = checkpoint::checked_sum(
          checkpoint::checked_byte_count(checkpoint_index, participant_count),
          participant);
      uint64_t source_row = 0;
      scan_dataset(
          uri, 0,
          [&](const ArrowSchema &schema, ArrowArrayStream &stream) {
            require_schema_key_dtype(schema, key_dtype);
            const auto dim_col = find_column_index(schema, "dim");
            const auto key_col = find_column_index(schema, "key");
            const auto state_col = find_column_index(schema, "row_state");
            const auto mutation_col =
                find_column_index(schema, "mutation_sequence");
            const auto participant_col =
                find_column_index(schema, "source_participant_index");
            for (;;) {
              ArrowArrayGuard batch;
              auto &array = *batch;
              if (stream.get_next(&stream, &array) != 0)
                throw std::runtime_error(
                    "checkpoint chain validation scan failed");
              if (!array.release)
                break;
              CheckpointStreamPolicy::validate_scan_batch_rows(array.length);
              if (preserves_source_ownership) {
                dispatch_key_type(key_type, [&]<typename KeyT>() {
                  target_partitioner->partition_span(
                      ArrowArrayView(arrow_array_field(array, key_col))
                          .values<KeyT>(),
                      std::span<uint32_t>(target_partitions.data(),
                                          array.length));
                });
              }
              for (int64_t row = 0; row < array.length; ++row) {
                const uint32_t dim = fixed_value<uint32_t>(
                    arrow_array_field(array, dim_col), row);
                const uint64_t key = read_polymorphic_key(
                    arrow_array_field(array, key_col),
                    arrow_schema_field(schema, key_col), row);
                const uint64_t sequence = fixed_value<uint64_t>(
                    arrow_array_field(array, mutation_col), row);
                const auto state =
                    binary_string(arrow_array_field(array, state_col), row);
                if (std::find(dimensions.begin(), dimensions.end(), dim) ==
                        dimensions.end() ||
                    fixed_value<uint32_t>(
                        arrow_array_field(array, participant_col), row) !=
                        participant)
                  throw std::runtime_error("checkpoint entry dimension or "
                                           "participant identity mismatch");
                // Include tombstones: historical identities cannot change
                // participant even when their latest state is deleted.
                if (preserves_source_ownership &&
                    target_partitions[row] != participant)
                  preserves_source_ownership = false;
                if (state != "active" && state != "deleted")
                  throw std::runtime_error("checkpoint row state is invalid");
                if (checkpoint_index == 0) {
                  if (sequence != 0 || state != "active")
                    throw std::runtime_error(
                        "FULL checkpoint entry is invalid");
                } else if (!(previous_watermarks[participant] < sequence &&
                             sequence <= watermark)) {
                  throw std::runtime_error("checkpoint mutation sequence is "
                                           "outside its participant cut");
                }
                identities.append(
                    {key, checkpoint_index, source, source_row, sequence,
                     checkpoint::CheckpointEntryRecord::kNoPayload,
                     uint64_t(state == "deleted"), dim, participant});
                if (sequence != 0)
                  sequences.append({checkpoint_index, participant, sequence});
                source_row = checkpoint::checked_sum(source_row, 1);
              }
            }
          },
          storage_options, columns);
    }
    previous_watermarks = participant_watermarks[checkpoint_index];
  }
  checkpoint::reduce_checkpoint_entries(
      identities, [](const checkpoint::CheckpointEntryRecord &) {});
  checkpoint::validate_unique_records(
      sequences, "checkpoint participant mutation sequence is duplicated");
  return preserves_source_ownership;
}

namespace {

struct EntryPartitionTarget {
  pro::proxy<ops::PartitionerFacade> &partitioner;
  uint32_t partition_id;
};

std::string_view binary_text(std::span<const std::byte> bytes) {
  return bytes.empty()
             ? std::string_view{}
             : std::string_view(reinterpret_cast<const char *>(bytes.data()),
                                bytes.size());
}

checkpoint::NativeEntryScanStatistics read_checkpoint_winner_payloads(
    std::span<const std::string> uris, uint64_t version,
    const EntryPartitionTarget *target, std::string_view key_dtype,
    NativeEntryBatchVisitor visitor, bool load_optimizer_state,
    const LanceStorageOptions &storage_options,
    std::optional<uint32_t> source_participant_index) {
  if (!visitor)
    throw std::invalid_argument("native Lance entry visitor is empty");
  const auto key_type = parse_key_dtype(key_dtype);
  checkpoint::NativeEntryScanStatistics statistics;
  checkpoint::EntryPayloadSpool payloads(load_optimizer_state);
  checkpoint::ExternalSortedRecords<checkpoint::CheckpointEntryRecord>
      identities;
  checkpoint::ExternalSortedRecords<uint64_t> selected_offsets;
  std::array<bool, CheckpointStreamPolicy::kScanBatchRows> mask;
  std::array<const char *, lance_layout::kEntryColumns.size() + 1> columns{};
  size_t column_count = 0;
  for (const auto &column : lance_layout::kEntryColumns) {
    const std::string_view name(column.name);
    if (name == "physical_page_id" || name == "page_slot" ||
        (!load_optimizer_state &&
         (name.starts_with("optimizer_state_") || name == "step")))
      continue;
    columns[column_count++] = column.name;
  }
  for (size_t source = 0; source < uris.size(); ++source) {
    uint64_t source_row = 0;
    scan_dataset(
        uris[source], version,
        [&](const ArrowSchema &schema, ArrowArrayStream &stream) {
          require_schema_key_dtype(schema, key_dtype);
          const auto col =
              resolve_entry_scan_indices(schema, load_optimizer_state);
          dispatch_key_type(key_type, [&]<typename KeyT>() {
            for (;;) {
              ArrowArrayGuard batch;
              if (stream.get_next(&stream, batch.get()) != 0)
                throw std::runtime_error(
                    "checkpoint source payload scan failed");
              if (!batch->release)
                break;
              CheckpointStreamPolicy::validate_scan_batch_rows(batch->length);
              if (batch->length == 0)
                continue;
              ArrowRecordBatchView view(batch.get(), &schema);
              const auto keys = view.fixed<KeyT>(col.key);
              const auto dimensions = view.fixed<uint32_t>(col.dim);
              const auto participants =
                  view.fixed<uint32_t>(col.source_participant_index);
              const auto mutations =
                  view.fixed<uint64_t>(col.mutation_sequence);
              const auto states = view.binary(col.row_state);
              const auto values = view.binary(col.embedding_values);
              const auto value_dtypes = view.binary(col.embedding_dtype);
              const auto sequences = view.fixed<uint64_t>(col.sequence);
              const auto scores = view.fixed<uint64_t>(col.score);
              std::optional<BinaryColumnView> optimizer_values,
                  optimizer_dtypes;
              std::span<const uint32_t> optimizer_strides, steps;
              if (load_optimizer_state) {
                optimizer_values = view.binary(col.optimizer_state_values);
                optimizer_dtypes = view.binary(col.optimizer_state_dtype);
                optimizer_strides =
                    view.fixed<uint32_t>(col.optimizer_state_stride);
                steps = view.fixed<uint32_t>(col.step);
              }
              if (target)
                target->partitioner->filter_span(
                    keys, target->partition_id,
                    std::span<bool>(mask.data(), batch->length));
              for (int64_t row = 0; row < batch->length; ++row) {
                if (source_participant_index &&
                    participants[row] != *source_participant_index)
                  throw std::runtime_error(
                      "checkpoint source participant mismatch");
                const auto state = binary_text(states.row(row));
                if (state != "active" && state != "deleted")
                  throw std::runtime_error("checkpoint row state is invalid");
                uint64_t payload_offset =
                    checkpoint::CheckpointEntryRecord::kNoPayload;
                if (state == "active" && (!target || mask[row])) {
                  const auto embedding = values.row(row);
                  const auto optimizer = load_optimizer_state
                                             ? optimizer_values->row(row)
                                             : std::span<const std::byte>{};
                  if (!std::in_range<uint32_t>(embedding.size()) ||
                      !std::in_range<uint32_t>(optimizer.size()))
                    throw std::overflow_error(
                        "checkpoint row payload size overflows");
                  const checkpoint::EntryPayloadHeader header{
                      static_cast<uint64_t>(keys[row]),
                      scores[row],
                      sequences[row],
                      dimensions[row],
                      load_optimizer_state ? optimizer_strides[row] : 0,
                      load_optimizer_state ? steps[row] : 0,
                      static_cast<uint32_t>(embedding.size()),
                      static_cast<uint32_t>(optimizer.size()),
                      checkpoint::payload_dtype_code(
                          binary_text(value_dtypes.row(row))),
                      load_optimizer_state
                          ? checkpoint::payload_dtype_code(
                                binary_text(optimizer_dtypes->row(row)))
                          : checkpoint::kNoPayloadDtype};
                  payload_offset =
                      payloads.append(header, embedding, optimizer);
                }
                // Every historical row reaches reconciliation, even when its
                // payload is not requested or its latest state is a tombstone.
                identities.append({static_cast<uint64_t>(keys[row]), source,
                                   source, source_row, mutations[row],
                                   payload_offset, uint64_t(state == "deleted"),
                                   dimensions[row], participants[row]});
                source_row = checkpoint::checked_sum(source_row, 1);
              }
            }
          });
        },
        storage_options, columns.data());
    statistics.sources_scanned =
        checkpoint::checked_sum(statistics.sources_scanned, 1);
    statistics.rows_scanned =
        checkpoint::checked_sum(statistics.rows_scanned, source_row);
  }
  payloads.seal();
  checkpoint::reduce_checkpoint_entries(
      identities, [&](const checkpoint::CheckpointEntryRecord &winner) {
        if (!winner.deleted &&
            winner.payload_offset !=
                checkpoint::CheckpointEntryRecord::kNoPayload)
          selected_offsets.append(winner.payload_offset);
      });
  payloads.consume_selected(selected_offsets, std::move(visitor));
  statistics.identity_sort = identities.statistics();
  statistics.selection_sort = selected_offsets.statistics();
  statistics.payload = payloads.statistics();
  return statistics;
}

} // namespace

NativeFragmentArtifact compact_native_checkpoint(
    std::string_view target_uri, std::span<const std::string> source_uris,
    uint32_t participant_index, std::span<const uint32_t> dimensions,
    std::string_view embedding_dtype, std::string_view optimizer_dtype,
    std::string_view score_policy, std::string_view key_dtype,
    std::string_view schema_digest, bool include_page_images,
    uint32_t entries_per_page, std::string_view page_dataset_uri,
    const LanceStorageOptions &storage_options) {
  const bool has_score = score_policy != "none";
  NativeFragmentWriter writer;
  bool writer_started = false;
  auto start_writer = [&] {
    if (writer_started)
      return;
    writer.begin_fragment(
        target_uri, participant_index, dimensions, embedding_dtype,
        optimizer_dtype, score_policy, key_dtype, schema_digest, has_score,
        include_page_images, entries_per_page, page_dataset_uri);
    writer_started = true;
  };
  struct PageBatch {
    std::vector<uint64_t> keys, sequences, scores;
    std::vector<uint32_t> slots, steps;
    std::vector<std::byte> values, optimizer;
    std::optional<uint32_t> optimizer_stride;
    uint64_t next_page = 0;
  };
  std::map<uint32_t, PageBatch> pages;
  auto flush_page = [&](uint32_t dim, PageBatch &page) {
    if (page.keys.empty())
      return;
    const size_t row_bytes = dim * dtype_itemsize(embedding_dtype);
    writer.write_page_image(
        dim, page.next_page,
        static_cast<uint32_t>(
            storage::PageGeometry::aligned_slot_stride(row_bytes)),
        row_bytes, page.slots, page.keys, page.sequences, page.values,
        page.scores, page.optimizer, *page.optimizer_stride,
        optimizer_dtype.empty() ? 0 : dtype_itemsize(optimizer_dtype),
        page.steps);
    page.next_page = checkpoint::checked_sum(page.next_page, 1);
    page.keys.clear();
    page.sequences.clear();
    page.scores.clear();
    page.slots.clear();
    page.steps.clear();
    page.values.clear();
    page.optimizer.clear();
  };
  read_checkpoint_winner_payloads(
      source_uris, 0, nullptr, key_dtype,
      [&](const NativeEntryBatch &batch) {
        if (batch.embedding_dtype != embedding_dtype ||
            batch.optimizer_dtype != optimizer_dtype)
          throw std::runtime_error("checkpoint compact payload dtype mismatch");
        for (size_t row = 0; row < batch.keys.size(); ++row) {
          const uint32_t dim = batch.dimensions[row];
          if (std::find(dimensions.begin(), dimensions.end(), dim) ==
              dimensions.end())
            throw std::runtime_error("checkpoint compact dimension is invalid");
          auto &page = pages[dim];
          const uint32_t stride = batch.optimizer_strides[row];
          if (page.optimizer_stride && *page.optimizer_stride != stride)
            throw std::runtime_error(
                "checkpoint compact optimizer strides differ");
          // This invariant survives every page flush and every source boundary.
          page.optimizer_stride = stride;
        }
        start_writer();
        for (size_t row = 0; row < batch.keys.size(); ++row) {
          const uint32_t dim = batch.dimensions[row];
          auto &page = pages[dim];
          page.slots.push_back(static_cast<uint32_t>(page.keys.size()));
          page.keys.push_back(batch.keys[row]);
          page.sequences.push_back(batch.sequences[row]);
          const size_t value_begin = batch.embedding_offsets[row];
          const size_t value_end = batch.embedding_offsets[row + 1];
          page.values.insert(page.values.end(),
                             batch.embedding_values.begin() + value_begin,
                             batch.embedding_values.begin() + value_end);
          if (has_score)
            page.scores.push_back(batch.scores[row]);
          const size_t optimizer_begin = batch.optimizer_offsets[row];
          const size_t optimizer_end = batch.optimizer_offsets[row + 1];
          if (optimizer_begin != optimizer_end)
            page.optimizer.insert(
                page.optimizer.end(),
                batch.optimizer_values.begin() + optimizer_begin,
                batch.optimizer_values.begin() + optimizer_end);
          if (batch.optimizer_strides[row] != 0)
            page.steps.push_back(batch.steps[row]);
          if (page.keys.size() == entries_per_page)
            flush_page(dim, page);
        }
        return true;
      },
      true, storage_options, participant_index);
  start_writer();
  for (auto &[dim, page] : pages)
    flush_page(dim, page);
  return writer.finalize();
}

checkpoint::NativeEntryScanStatistics scan_native_checkpoint_entries(
    std::span<const std::string> uris, uint64_t version,
    pro::proxy<se::ops::PartitionerFacade> partitioner, uint32_t partition_id,
    std::string_view key_dtype, NativeEntryBatchVisitor visitor,
    bool load_optimizer_state, const LanceStorageOptions &storage_options,
    std::optional<uint32_t> source_participant_index) {
  if (!partitioner.has_value())
    throw std::invalid_argument("native Lance partitioner is null");
  const EntryPartitionTarget target{partitioner, partition_id};
  return read_checkpoint_winner_payloads(
      uris, version, &target, key_dtype, std::move(visitor),
      load_optimizer_state, storage_options, source_participant_index);
}

void scan_native_page_images(std::string_view uri, uint64_t version,
                             uint32_t participant_index,
                             std::string_view key_dtype,
                             NativePageImageVisitor visitor,
                             bool load_optimizer_state,
                             const LanceStorageOptions &storage_options) {
  if (!visitor)
    throw std::invalid_argument("native Lance page visitor is empty");
  (void)parse_key_dtype(key_dtype);
  scan_dataset(
      uri, version,
      [&](const ArrowSchema &schema, ArrowArrayStream &stream) {
        require_schema_key_dtype(schema, key_dtype);
        const auto col = resolve_page_image_scan_indices(schema);
        NativePageImageView page;
        for (;;) {
          ArrowArrayGuard batch;
          if (stream.get_next(&stream, batch.get()) != 0)
            throw std::runtime_error(
                "native Lance restore page batch read failed");
          if (!batch->release)
            break;
          CheckpointStreamPolicy::validate_scan_batch_rows(
              batch->length, CheckpointStreamPolicy::kPageImportBatchPages);
          ArrowRecordBatchView view(batch.get(), &schema);
          const auto participants =
              view.fixed<uint32_t>(col.source_participant_index);
          const auto states = view.binary(col.row_state);
          for (int64_t row = 0; row < batch->length; ++row) {
            if (participants[row] != participant_index)
              continue;
            const auto state = binary_text(states.row(row));
            if (state != "active" && state != "deleted")
              throw std::runtime_error(
                  "native Lance page row state is invalid");
            if (state == "deleted")
              continue;
            page.entries_per_page =
                view.fixed<uint32_t>(col.entries_per_page)[row];
            page.embedding_dim = view.fixed<uint32_t>(col.embedding_dim)[row];
            page.physical_page_id =
                view.fixed<uint64_t>(col.physical_page_id)[row];
            page.page_slot_stride_bytes =
                view.fixed<uint32_t>(col.page_slot_stride_bytes)[row];
            page.embedding_dtype =
                binary_text(view.binary(col.embedding_dtype).row(row));
            const auto slots = view.binary(col.slot_ordinals).row(row);
            const auto keys = view.binary(col.keys).row(row);
            const auto sequences = view.binary(col.sequences).row(row);
            const auto values = view.binary(col.embedding_values).row(row);
            const auto scores = view.binary(col.scores).row(row);
            const auto optimizer =
                view.binary(col.optimizer_state_values).row(row);
            const auto steps = view.binary(col.steps).row(row);
            page.checksum = binary_text(view.binary(col.checksum).row(row));
            if (checksum(slots, keys, sequences, values, scores, optimizer,
                         steps) != page.checksum)
              throw std::runtime_error("page image checksum mismatch");
            copy_native_values(slots, page.slot_ordinals);
            copy_native_values(keys, page.keys);
            copy_native_values(sequences, page.sequences);
            copy_native_values(scores, page.scores);
            if (steps.size() % sizeof(uint32_t) != 0)
              throw std::runtime_error("native Lance page steps are truncated");
            const size_t count = page.keys.size();
            if (count > page.entries_per_page ||
                page.slot_ordinals.size() != count ||
                page.sequences.size() != count ||
                (!page.scores.empty() && page.scores.size() != count) ||
                (!steps.empty() && steps.size() / sizeof(uint32_t) != count))
              throw std::runtime_error(
                  "native Lance page image key and payload lengths differ");
            page.values = values;
            if (load_optimizer_state) {
              page.optimizer_values = optimizer;
              copy_native_values(steps, page.steps);
            }
            page.optimizer_dtype =
                binary_text(view.binary(col.optimizer_state_dtype).row(row));
            page.optimizer_stride =
                view.fixed<uint32_t>(col.optimizer_state_stride)[row];
            if (!visitor(std::move(page)))
              return;
          }
        }
      },
      storage_options, nullptr, CheckpointStreamPolicy::kPageImportBatchPages);
}

} // namespace se
