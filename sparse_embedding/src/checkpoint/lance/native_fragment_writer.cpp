#include "sparse_embedding/src/checkpoint/lance/native_fragment_writer.hpp"

#include <lance/lance.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <random>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <type_traits>

#include "sparse_embedding/src/checkpoint/checkpoint_stream_policy.hpp"
#include "sparse_embedding/src/checkpoint/lance/layout_schema.hpp"
#include "sparse_embedding/src/common/dtypes.hpp"
#include "sparse_embedding/src/ops/partitioner_registry.hpp"

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
  const auto *offsets = static_cast<const int32_t *>(array->buffers[1]);
  const auto *data = static_cast<const std::byte *>(array->buffers[2]);
  const int64_t index = array->offset + row;
  const int32_t begin = offsets[index];
  const int32_t end = offsets[index + 1];
  if (begin < 0 || end < begin)
    throw std::runtime_error("native Lance binary offsets are invalid");
  return std::span<const std::byte>(data + begin,
                                    static_cast<size_t>(end - begin));
}

std::string binary_string(const ArrowArray *array, int64_t row) {
  const auto bytes = binary_value(array, row);
  return std::string(reinterpret_cast<const char *>(bytes.data()),
                     bytes.size());
}

template <typename T> T fixed_value(const ArrowArray *array, int64_t row) {
  const auto *values = static_cast<const T *>(array->buffers[1]);
  return values[array->offset + row];
}

void append_binary(std::vector<std::byte> &data, std::vector<uint32_t> &offsets,
                   std::span<const std::byte> value) {
  if (data.size() > UINT32_MAX - value.size())
    throw std::length_error("native Lance restore batch exceeds offset range");
  data.insert(data.end(), value.begin(), value.end());
  offsets.push_back(static_cast<uint32_t>(data.size()));
}

inline size_t find_column_index(const ArrowSchema &schema,
                                std::string_view name) {
  for (int64_t index = 0; index < schema.n_children; ++index) {
    if (schema.children[index]->name && name == schema.children[index]->name) {
      return static_cast<size_t>(index);
    }
  }
  throw std::runtime_error("native Lance schema is missing required field: " +
                           std::string(name));
}

struct ActiveKeyScanIndices {
  size_t dim = 0;
  size_t key = 0;
  size_t row_state = 0;
  size_t source_participant_index = 0;
};

inline ActiveKeyScanIndices
resolve_active_key_scan_indices(const ArrowSchema &schema) {
  return {
      .dim = find_column_index(schema, "dim"),
      .key = find_column_index(schema, "key"),
      .row_state = find_column_index(schema, "row_state"),
      .source_participant_index =
          find_column_index(schema, "source_participant_index"),
  };
}

struct KeyValidationIndices {
  size_t dim = 0;
  size_t key = 0;
  size_t row_state = 0;
};

inline KeyValidationIndices
resolve_key_validation_indices(const ArrowSchema &schema) {
  return {
      .dim = find_column_index(schema, "dim"),
      .key = find_column_index(schema, "key"),
      .row_state = find_column_index(schema, "row_state"),
  };
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
  size_t row_state = 0;
  size_t source_participant_index = 0;
};

inline EntryScanIndices resolve_entry_scan_indices(const ArrowSchema &schema) {
  return {
      .dim = find_column_index(schema, "dim"),
      .key = find_column_index(schema, "key"),
      .embedding_values = find_column_index(schema, "embedding_values"),
      .embedding_dtype = find_column_index(schema, "embedding_dtype"),
      .optimizer_state_values =
          find_column_index(schema, "optimizer_state_values"),
      .optimizer_state_dtype =
          find_column_index(schema, "optimizer_state_dtype"),
      .optimizer_state_stride =
          find_column_index(schema, "optimizer_state_stride"),
      .step = find_column_index(schema, "step"),
      .score = find_column_index(schema, "score"),
      .row_state = find_column_index(schema, "row_state"),
      .source_participant_index =
          find_column_index(schema, "source_participant_index"),
  };
}

struct PageImageScanIndices {
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
  if (dtype == "float32")
    return sizeof(float);
  if (dtype == "float16" || dtype == "bfloat16")
    return sizeof(uint16_t);
  throw std::invalid_argument("unsupported native Lance payload dtype");
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
                      for (int64_t i = 0; i < array->n_children; ++i)
                        if (array->children[i] && array->children[i]->release)
                          array->children[i]->release(array->children[i]);
                      delete[] array->children;
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
    const auto *child_schema = schema.children[child_index];
    const auto *child = source.children[child_index];
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
    const auto *bytes = static_cast<const std::byte *>(child->buffers[1]);
    std::vector<std::byte> data;
    data.reserve(selected.size() * width);
    for (const int64_t row : selected) {
      const auto *begin =
          bytes + static_cast<size_t>(child->offset + row) * width;
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
    const auto *child_schema = schema.children[child_index];
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
  std::string embedding_dtype;
  std::string optimizer_dtype;
  std::string key_dtype;
  uint32_t participant_index = 0;
  bool has_score = false;
  bool include_page_images = false;
  std::vector<char> schema_metadata;
  NativeFragmentArtifact artifact;
  uint64_t content_hash = 1469598103934665603ULL;

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
  for (int64_t i = 0; i < schema->n_children; ++i)
    if (schema->children[i] && schema->children[i]->release)
      schema->children[i]->release(schema->children[i]);
  delete[] schema->children;
  schema->children = nullptr;
  schema->release = nullptr;
}

void NativeFragmentWriter::release_array(ArrowArray *array) {
  if (!array || !array->release)
    return;
  for (int64_t i = 0; i < array->n_children; ++i)
    if (array->children[i] && array->children[i]->release)
      array->children[i]->release(array->children[i]);
  delete[] array->children;
  delete[] array->buffers;
  array->release = nullptr;
}

NativeFragmentWriter::NativeFragmentWriter() = default;
NativeFragmentWriter::~NativeFragmentWriter() = default;
NativeFragmentWriter::NativeFragmentWriter(NativeFragmentWriter &&) noexcept =
    default;
NativeFragmentWriter &
NativeFragmentWriter::operator=(NativeFragmentWriter &&) noexcept = default;

void NativeFragmentWriter::begin_fragment(
    std::string_view staging_uri, uint32_t participant_index,
    std::span<const uint32_t> dimensions, std::string_view embedding_dtype,
    std::string_view optimizer_dtype, std::string_view score_policy,
    std::string_view key_dtype, std::string_view schema_digest, bool has_score,
    bool include_page_images) {
  state_ = std::make_unique<PipelinedWriterState>();
  state_->root = std::string(staging_uri);
  state_->entries_uri =
      state_->root + "/" + std::string(lance_layout::kEntriesDataset);
  state_->pages_uri =
      state_->root + "/" + std::string(lance_layout::kPageImagesDataset);
  state_->embedding_dtype = std::string(embedding_dtype);
  state_->optimizer_dtype = std::string(optimizer_dtype);
  state_->key_dtype = std::string(key_dtype);
  state_->participant_index = participant_index;
  state_->has_score = has_score;
  state_->include_page_images = include_page_images;
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

  if (state_->include_page_images) {
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
        check_lance(lance_dataset_write(state->pages_uri.c_str(), &schema,
                                        &stream, LANCE_WRITE_CREATE, nullptr,
                                        &state->pages_dataset),
                    "native Lance fragment write pages");
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

  auto slot_bytes = bytes_of(slots);
  auto key_bytes = bytes_of(keys);
  auto sequence_bytes = bytes_of(sequences);
  auto value_bytes = std::vector<std::byte>(values.begin(), values.end());
  auto score_bytes = bytes_of(scores);
  auto optimizer_bytes =
      std::vector<std::byte>(optimizer.begin(), optimizer.end());
  auto step_bytes = bytes_of(steps);
  std::string page_digest =
      checksum(slot_bytes, key_bytes, sequence_bytes, value_bytes, score_bytes,
               optimizer_bytes, step_bytes);
  const size_t optimizer_stride_bytes = optimizer_stride * optimizer_itemsize;
  if (optimizer_stride_bytes != 0 &&
      optimizer.size() != keys.size() * optimizer_stride_bytes)
    throw std::invalid_argument(
        "native Lance optimizer payload shape mismatch");

  const size_t count = keys.size();
  std::vector<ArrowArray *> children;
  children.reserve(15);
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
  children.push_back(make_repeated_binary_column(embedding_dtype_bytes, count));
  children.push_back(
      make_strided_binary_column(optimizer, optimizer_stride_bytes, count));
  const auto optimizer_dtype_bytes = string_bytes(state_->optimizer_dtype);
  children.push_back(make_repeated_binary_column(optimizer_dtype_bytes, count));
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
  std::vector<uint64_t> last_ckpts(count, 0ULL);
  children.push_back(numeric_column(last_ckpts));
  std::vector<uint32_t> part_indices(count, state_->participant_index);
  children.push_back(numeric_column(part_indices));

  auto entries_array = make_struct_array(std::move(children), count);
  state_->artifact.entry_count += count;
  state_->entries_channel.push(std::move(entries_array));

  if (state_->include_page_images) {
    const size_t p_count = keys.empty() ? 0 : 1;
    std::vector<ArrowArray *> p_children;
    std::vector<uint32_t> part_col(p_count, state_->participant_index);
    p_children.push_back(numeric_column(part_col));
    std::vector<uint32_t> dim_col(p_count,
                                  static_cast<uint32_t>(embedding_dim));
    p_children.push_back(numeric_column(dim_col));
    std::vector<uint64_t> page_col(p_count, physical_page_id);
    p_children.push_back(numeric_column(page_col));
    std::vector<uint32_t> stride_col(p_count, page_slot_stride_bytes);
    p_children.push_back(numeric_column(stride_col));
    auto single_row = [p_count](std::span<const std::byte> val) {
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
    std::vector<uint64_t> zeros_col(p_count, 0ULL);
    p_children.push_back(numeric_column(zeros_col));

    auto page_array = make_struct_array(std::move(p_children), p_count);
    if (!slots.empty() || !keys.empty() || !values.empty())
      state_->artifact.page_count += 1;
    state_->pages_channel->push(std::move(page_array));
  } else if (!slots.empty() || !keys.empty() || !values.empty()) {
    state_->artifact.page_count += 1;
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

NativeFragmentArtifact NativeFragmentWriter::finalize() {
  if (!state_)
    throw std::runtime_error("finalize called without active fragment");

  if (state_->artifact.entry_count == 0) {
    auto schema = se::lance_layout::make_entries_schema(
        state_->key_dtype, state_->schema_metadata.empty()
                               ? nullptr
                               : state_->schema_metadata.data());
    state_->entries_channel.push(make_empty_array(schema));
    if (schema.release)
      schema.release(&schema);
  }
  state_->entries_channel.push(std::nullopt);
  state_->entries_channel.close();
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

  if (state_->include_page_images) {
    if (state_->artifact.page_count == 0) {
      auto schema = se::lance_layout::make_page_images_schema(
          state_->schema_metadata.empty() ? nullptr
                                          : state_->schema_metadata.data());
      state_->pages_channel->push(make_empty_array(schema));
      if (schema.release)
        schema.release(&schema);
    }
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
      const char *name = schema.get()->children[index]->name;
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
    if (!is_entries && schema.get()->n_children != 17)
      throw std::runtime_error(
          "native Lance compaction source schema is unsupported");

    if (!schema_initialized) {
      if (schema.get()->metadata)
        output_metadata = clone_arrow_metadata(schema.get()->metadata);
      is_entries_dataset = is_entries;
      if (is_entries && key_index.has_value()) {
        const char *fmt = schema.get()->children[*key_index]->format;
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
          if (binary_string(batch.children[*state_index], row) !=
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
        const ArrowArray *child = selected.children[col];
        const ArrowSchema *child_schema = schema.get()->children[col];
        if (child_schema->format && (child_schema->format[0] == 'z' ||
                                     child_schema->format[0] == 'u')) {
          const auto *offsets = static_cast<const int32_t *>(child->buffers[1]);
          const auto *data = static_cast<const std::byte *>(child->buffers[2]);
          for (int64_t row = 0; row < child->length; ++row) {
            const size_t begin = offsets[child->offset + row];
            const size_t end = offsets[child->offset + row + 1];
            content_hash =
                fnv1a(std::span<const std::byte>(data + begin, end - begin),
                      content_hash);
          }
          continue;
        }
        for (int64_t row = 0; row < child->length; ++row) {
          const size_t width = fixed_column_width(child_schema);
          if (width == 0)
            throw std::runtime_error(
                "native Lance compaction column format is unsupported");
          const auto *bytes = static_cast<const std::byte *>(child->buffers[1]);
          content_hash = fnv1a(
              std::span<const std::byte>(
                  bytes + static_cast<size_t>(child->offset + row) * width,
                  width),
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
    const char *const *columns = nullptr) {
  const char *const *options = storage_options.lance_options();
  std::unique_ptr<LanceDataset, DatasetDeleter> dataset(
      lance_dataset_open(std::string(uri).c_str(), options, version));
  if (!dataset)
    check_lance(-1, "native Lance restore dataset open");
  std::unique_ptr<LanceScanner, ScannerDeleter> scanner(
      lance_scanner_new(dataset.get(), columns, nullptr));
  if (!scanner)
    throw std::runtime_error("native Lance restore scanner creation failed");
  lance_scanner_set_batch_size(scanner.get(),
                               CheckpointStreamPolicy::kScanBatchRows);
  ArrowStreamGuard stream;
  if (lance_scanner_to_arrow_stream(scanner.get(), stream.get()) != 0)
    check_lance(-1, "native Lance restore scanner");
  ArrowSchemaGuard schema;
  if (stream.get()->get_schema(stream.get(), schema.get()) != 0)
    throw std::runtime_error("native Lance restore schema read failed");
  callback(*schema.get(), *stream.get());
}

struct ActiveKeyIdentity {
  uint32_t dimension;
  uint64_t key;

  friend bool operator<(const ActiveKeyIdentity &left,
                        const ActiveKeyIdentity &right) noexcept {
    return std::tie(left.dimension, left.key) <
           std::tie(right.dimension, right.key);
  }

  friend bool operator==(const ActiveKeyIdentity &left,
                         const ActiveKeyIdentity &right) noexcept {
    return left.dimension == right.dimension && left.key == right.key;
  }
};

void write_active_key(std::ostream &output, const ActiveKeyIdentity &record) {
  output.write(reinterpret_cast<const char *>(&record.dimension),
               sizeof(record.dimension));
  output.write(reinterpret_cast<const char *>(&record.key), sizeof(record.key));
  if (!output)
    throw std::runtime_error(
        "native Lance duplicate validation temporary run write failed");
}

bool read_active_key(std::istream &input, ActiveKeyIdentity &record) {
  input.read(reinterpret_cast<char *>(&record.dimension),
             sizeof(record.dimension));
  if (input.gcount() == 0 && input.eof())
    return false;
  if (input.gcount() != static_cast<std::streamsize>(sizeof(record.dimension)))
    throw std::runtime_error(
        "native Lance duplicate validation temporary run is truncated");
  input.read(reinterpret_cast<char *>(&record.key), sizeof(record.key));
  if (input.gcount() != static_cast<std::streamsize>(sizeof(record.key)))
    throw std::runtime_error(
        "native Lance duplicate validation temporary run is truncated");
  return true;
}

class TemporaryRunDirectory {
  std::filesystem::path path_;

  static std::filesystem::path create_path() {
    static std::atomic<uint64_t> sequence{0};
    const auto base = std::filesystem::temp_directory_path();
    std::random_device entropy;
    for (;;) {
      const auto now = std::chrono::steady_clock::now().time_since_epoch();
      const auto timestamp = static_cast<uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
      const uint64_t token = (static_cast<uint64_t>(entropy()) << 32) ^
                             static_cast<uint64_t>(entropy());
      const auto candidate =
          base /
          ("sparse_embedding_duplicate_validation-" +
           std::to_string(timestamp) + "-" +
           std::to_string(sequence.fetch_add(1, std::memory_order_relaxed)) +
           "-" + std::to_string(token));
      std::error_code error;
      if (std::filesystem::create_directory(candidate, error)) {
        std::filesystem::permissions(candidate,
                                     std::filesystem::perms::owner_read |
                                         std::filesystem::perms::owner_write |
                                         std::filesystem::perms::owner_exec,
                                     std::filesystem::perm_options::replace,
                                     error);
        if (error) {
          std::error_code cleanup_error;
          std::filesystem::remove(candidate, cleanup_error);
          throw std::filesystem::filesystem_error(
              "native Lance duplicate validation temporary permissions",
              candidate, error);
        }
        return candidate;
      }
      if (error)
        throw std::filesystem::filesystem_error(
            "native Lance duplicate validation temporary directory", candidate,
            error);
    }
  }

public:
  TemporaryRunDirectory() : path_(create_path()) {}
  ~TemporaryRunDirectory() {
    if (path_.empty())
      return;
    std::error_code ignored;
    std::filesystem::remove_all(path_, ignored);
  }

  TemporaryRunDirectory(const TemporaryRunDirectory &) = delete;
  TemporaryRunDirectory &operator=(const TemporaryRunDirectory &) = delete;

  const std::filesystem::path &path() const noexcept { return path_; }

  void cleanup() {
    if (path_.empty())
      return;
    std::error_code error;
    std::filesystem::remove_all(path_, error);
    if (error)
      throw std::filesystem::filesystem_error(
          "native Lance duplicate validation temporary cleanup", path_, error);
    path_.clear();
  }
};

class ExternalActiveKeyValidator {
  struct MergeNode {
    ActiveKeyIdentity record;
    size_t input_index;
  };

  struct MergeNodeGreater {
    bool operator()(const MergeNode &left,
                    const MergeNode &right) const noexcept {
      return right.record < left.record;
    }
  };

  TemporaryRunDirectory directory_;
  std::vector<ActiveKeyIdentity> buffer_;
  std::vector<std::filesystem::path> runs_;
  uint64_t next_run_id_ = 0;
  bool finished_ = false;

  std::filesystem::path next_run_path() {
    return directory_.path() /
           ("run-" + std::to_string(next_run_id_++) + ".bin");
  }

  void write_sorted_run() {
    if (buffer_.empty())
      return;
    std::ranges::sort(buffer_, [](const auto &left, const auto &right) {
      return left < right;
    });
    const auto path = next_run_path();
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output)
      throw std::runtime_error(
          "native Lance duplicate validation temporary run open failed");
    std::optional<ActiveKeyIdentity> previous;
    for (const auto &record : buffer_) {
      if (previous.has_value() && *previous == record)
        throw std::invalid_argument(
            "native Lance restore rejected duplicate active key");
      write_active_key(output, record);
      previous = record;
    }
    output.flush();
    if (!output)
      throw std::runtime_error(
          "native Lance duplicate validation temporary run flush failed");
    output.close();
    if (!output)
      throw std::runtime_error(
          "native Lance duplicate validation temporary run close failed");
    runs_.push_back(path);
    buffer_.clear();
  }

  std::filesystem::path
  merge_runs(std::span<const std::filesystem::path> input_paths) {
    if (input_paths.empty())
      throw std::invalid_argument(
          "native Lance duplicate validation cannot merge no runs");
    std::vector<std::ifstream> inputs;
    inputs.reserve(input_paths.size());
    std::priority_queue<MergeNode, std::vector<MergeNode>, MergeNodeGreater>
        queue;
    for (size_t index = 0; index < input_paths.size(); ++index) {
      inputs.emplace_back(input_paths[index], std::ios::binary);
      if (!inputs.back())
        throw std::runtime_error(
            "native Lance duplicate validation temporary run open failed");
      ActiveKeyIdentity record{};
      if (read_active_key(inputs.back(), record))
        queue.push({record, index});
    }

    const auto output_path = next_run_path();
    std::ofstream output(output_path, std::ios::binary | std::ios::trunc);
    if (!output)
      throw std::runtime_error(
          "native Lance duplicate validation temporary merge open failed");
    std::optional<ActiveKeyIdentity> previous;
    while (!queue.empty()) {
      const MergeNode node = queue.top();
      queue.pop();
      if (previous.has_value() && *previous == node.record)
        throw std::invalid_argument(
            "native Lance restore rejected duplicate active key");
      write_active_key(output, node.record);
      previous = node.record;

      ActiveKeyIdentity next{};
      if (read_active_key(inputs[node.input_index], next))
        queue.push({next, node.input_index});
    }
    output.flush();
    if (!output)
      throw std::runtime_error(
          "native Lance duplicate validation temporary merge flush failed");
    output.close();
    if (!output)
      throw std::runtime_error(
          "native Lance duplicate validation temporary merge close failed");
    inputs.clear();
    for (const auto &path : input_paths) {
      std::error_code error;
      if (!std::filesystem::remove(path, error) || error)
        throw std::filesystem::filesystem_error(
            "native Lance duplicate validation temporary run removal", path,
            error);
    }
    return output_path;
  }

  void collapse_runs() {
    if (runs_.size() < CheckpointStreamPolicy::kDuplicateValidationMergeFanIn)
      return;
    const auto merged = merge_runs(runs_);
    runs_.clear();
    runs_.push_back(merged);
  }

public:
  ExternalActiveKeyValidator() {
    static_assert(CheckpointStreamPolicy::kDuplicateValidationRunRecords > 0);
    static_assert(CheckpointStreamPolicy::kDuplicateValidationMergeFanIn > 1);
    buffer_.reserve(CheckpointStreamPolicy::kDuplicateValidationRunRecords);
  }

  ~ExternalActiveKeyValidator() = default;
  ExternalActiveKeyValidator(const ExternalActiveKeyValidator &) = delete;
  ExternalActiveKeyValidator &
  operator=(const ExternalActiveKeyValidator &) = delete;

  void append(ActiveKeyIdentity record) {
    if (finished_)
      throw std::logic_error(
          "native Lance duplicate validation received a record after finish");
    buffer_.push_back(record);
    if (buffer_.size() ==
        CheckpointStreamPolicy::kDuplicateValidationRunRecords) {
      write_sorted_run();
      collapse_runs();
    }
  }

  void finish() {
    if (finished_)
      throw std::logic_error(
          "native Lance duplicate validation finished more than once");
    write_sorted_run();
    if (runs_.size() > 1) {
      const auto merged = merge_runs(runs_);
      runs_.clear();
      runs_.push_back(merged);
    }
    directory_.cleanup();
    finished_ = true;
  }
};

} // namespace

void scan_native_active_entry_keys(std::string_view uri, uint64_t version,
                                   NativeKeyBatchVisitor visitor,
                                   std::optional<uint32_t> participant_index,
                                   const LanceStorageOptions &storage_options) {
  if (!visitor)
    throw std::invalid_argument("native Lance key visitor is empty");
  scan_dataset(
      uri, version,
      [&](const ArrowSchema &schema, ArrowArrayStream &stream) {
        const auto col = resolve_active_key_scan_indices(schema);
        for (;;) {
          ArrowArray batch{};
          if (stream.get_next(&stream, &batch) != 0)
            throw std::runtime_error(
                "native Lance restore key batch read failed");
          if (!batch.release)
            break;
          std::vector<uint64_t> keys;
          keys.reserve(static_cast<size_t>(batch.length));
          for (int64_t row = 0; row < batch.length; ++row) {
            if (binary_string(batch.children[col.row_state], row) !=
                kActiveRowState)
              continue;
            if (participant_index.has_value()) {
              const uint32_t src_idx = fixed_value<uint32_t>(
                  batch.children[col.source_participant_index], row);
              if (src_idx != *participant_index)
                continue;
            }
            keys.push_back(read_polymorphic_key(batch.children[col.key],
                                                schema.children[col.key], row));
          }
          batch.release(&batch);
          if (!keys.empty() && !visitor(keys))
            break;
        }
      },
      storage_options);
}

void validate_native_active_entry_keys(
    std::span<const std::string> uris, uint64_t version,
    const LanceStorageOptions &storage_options) {
  if (uris.empty())
    throw std::invalid_argument("native Lance key validation requires sources");
  ExternalActiveKeyValidator validator;
  const char *const columns[] = {"dim", "key", "row_state", nullptr};
  for (const auto &uri : uris) {
    scan_dataset(
        uri, version,
        [&](const ArrowSchema &schema, ArrowArrayStream &stream) {
          const auto col = resolve_key_validation_indices(schema);
          for (;;) {
            ArrowArray batch{};
            if (stream.get_next(&stream, &batch) != 0)
              throw std::runtime_error(
                  "native Lance key validation batch read failed");
            if (!batch.release)
              break;
            try {
              for (int64_t row = 0; row < batch.length; ++row) {
                if (binary_string(batch.children[col.row_state], row) !=
                    kActiveRowState)
                  continue;
                validator.append(
                    {fixed_value<uint32_t>(batch.children[col.dim], row),
                     read_polymorphic_key(batch.children[col.key],
                                          schema.children[col.key], row)});
              }
            } catch (...) {
              batch.release(&batch);
              throw;
            }
            batch.release(&batch);
          }
        },
        storage_options, columns);
  }
  validator.finish();
}

void scan_native_partition_entries(
    std::string_view uri, uint64_t version,
    pro::proxy<se::ops::PartitionerFacade> partitioner, uint32_t partition_id,
    std::string_view key_dtype, NativeEntryBatchVisitor visitor,
    bool load_optimizer_state, const LanceStorageOptions &storage_options,
    std::optional<uint32_t> source_participant_index) {
  if (!partitioner.has_value())
    throw std::invalid_argument("native Lance partitioner is null");
  if (!visitor)
    throw std::invalid_argument("native Lance entry visitor is empty");
  (void)parse_key_dtype(key_dtype);
  scan_dataset(
      uri, version,
      [&](const ArrowSchema &schema, ArrowArrayStream &stream) {
        require_schema_key_dtype(schema, key_dtype);
        const auto col = resolve_entry_scan_indices(schema);
        for (;;) {
          ArrowArray batch_array{};
          if (stream.get_next(&stream, &batch_array) != 0)
            throw std::runtime_error(
                "native Lance restore entry batch read failed");
          if (!batch_array.release)
            break;

          const int64_t batch_len = batch_array.length;
          if (batch_len == 0) {
            batch_array.release(&batch_array);
            continue;
          }

          const auto *key_col = batch_array.children[col.key];
          const auto *key_schema = schema.children[col.key];

          std::unique_ptr<bool[]> mask(new bool[batch_len]);
          std::span<bool> mask_span(mask.get(), batch_len);

          const char fmt =
              (key_schema && key_schema->format) ? key_schema->format[0] : '\0';
          switch (fmt) {
          case 'i': {
            const auto *ptr =
                static_cast<const int32_t *>(key_col->buffers[1]) +
                key_col->offset;
            partitioner->filter_span(std::span<const int32_t>(ptr, batch_len),
                                     partition_id, mask_span);
            break;
          }
          case 'I': {
            const auto *ptr =
                static_cast<const uint32_t *>(key_col->buffers[1]) +
                key_col->offset;
            partitioner->filter_span(std::span<const uint32_t>(ptr, batch_len),
                                     partition_id, mask_span);
            break;
          }
          case 'l': {
            const auto *ptr =
                static_cast<const int64_t *>(key_col->buffers[1]) +
                key_col->offset;
            partitioner->filter_span(std::span<const int64_t>(ptr, batch_len),
                                     partition_id, mask_span);
            break;
          }
          case 'L': {
            const auto *ptr =
                static_cast<const uint64_t *>(key_col->buffers[1]) +
                key_col->offset;
            partitioner->filter_span(std::span<const uint64_t>(ptr, batch_len),
                                     partition_id, mask_span);
            break;
          }
          default:
            throw std::invalid_argument(
                "unsupported Arrow format for key column: " +
                std::string(key_schema && key_schema->format
                                ? key_schema->format
                                : "null"));
          }

          NativeEntryBatch batch;
          batch.embedding_offsets.push_back(0);
          batch.optimizer_offsets.push_back(0);
          for (int64_t row = 0; row < batch_len; ++row) {
            if (!mask_span[row] ||
                binary_string(batch_array.children[col.row_state], row) !=
                    kActiveRowState)
              continue;
            if (source_participant_index.has_value() &&
                fixed_value<uint32_t>(
                    batch_array.children[col.source_participant_index], row) !=
                    *source_participant_index)
              continue;
            const uint64_t key = read_polymorphic_key(
                batch_array.children[col.key], schema.children[col.key], row);
            batch.dimensions.push_back(
                fixed_value<uint32_t>(batch_array.children[col.dim], row));
            batch.keys.push_back(key);
            const auto embedding =
                binary_value(batch_array.children[col.embedding_values], row);
            append_binary(batch.embedding_values, batch.embedding_offsets,
                          embedding);
            batch.embedding_dtype =
                binary_string(batch_array.children[col.embedding_dtype], row);
            if (load_optimizer_state) {
              const auto optimizer = binary_value(
                  batch_array.children[col.optimizer_state_values], row);
              append_binary(batch.optimizer_values, batch.optimizer_offsets,
                            optimizer);
            } else {
              batch.optimizer_offsets.push_back(
                  static_cast<uint32_t>(batch.optimizer_values.size()));
            }
            batch.optimizer_dtype = binary_string(
                batch_array.children[col.optimizer_state_dtype], row);
            batch.optimizer_strides.push_back(fixed_value<uint32_t>(
                batch_array.children[col.optimizer_state_stride], row));
            batch.steps.push_back(
                fixed_value<uint32_t>(batch_array.children[col.step], row));
            batch.scores.push_back(
                fixed_value<uint64_t>(batch_array.children[col.score], row));
          }
          batch_array.release(&batch_array);
          if (!batch.keys.empty() && !visitor(batch))
            break;
        }
      },
      storage_options);
}

void scan_native_partition_entries(
    std::string_view uri, uint64_t version, uint64_t partition_count,
    uint64_t partition_id, std::string_view key_dtype,
    NativeEntryBatchVisitor visitor, bool load_optimizer_state,
    const LanceStorageOptions &storage_options,
    std::optional<uint32_t> source_participant_index) {
  if (partition_count == 0)
    throw std::invalid_argument(
        "native Lance partition count must be positive");
  if (partition_id >= partition_count)
    throw std::invalid_argument(
        "native Lance logical partition is outside logical partition count");
  auto partitioner = se::ops::PartitionerRegistry::instance().create(
      se::ops::kModuloPartitionerName, static_cast<uint32_t>(partition_count));
  scan_native_partition_entries(uri, version, std::move(partitioner),
                                static_cast<uint32_t>(partition_id), key_dtype,
                                std::move(visitor), load_optimizer_state,
                                storage_options, source_participant_index);
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
        for (;;) {
          ArrowArray batch_array{};
          if (stream.get_next(&stream, &batch_array) != 0)
            throw std::runtime_error(
                "native Lance restore page batch read failed");
          if (!batch_array.release)
            break;
          for (int64_t row = 0; row < batch_array.length; ++row) {
            if (fixed_value<uint32_t>(
                    batch_array.children[col.source_participant_index], row) !=
                    participant_index ||
                binary_string(batch_array.children[col.row_state], row) !=
                    kActiveRowState)
              continue;
            NativePageImage page;
            page.embedding_dim = fixed_value<uint32_t>(
                batch_array.children[col.embedding_dim], row);
            page.physical_page_id = fixed_value<uint64_t>(
                batch_array.children[col.physical_page_id], row);
            page.page_slot_stride_bytes = fixed_value<uint32_t>(
                batch_array.children[col.page_slot_stride_bytes], row);
            page.embedding_dtype =
                binary_string(batch_array.children[col.embedding_dtype], row);
            auto copy_binary = [&](size_t col_idx,
                                   std::vector<std::byte> &out) {
              const auto bytes =
                  binary_value(batch_array.children[col_idx], row);
              out.assign(bytes.begin(), bytes.end());
            };
            std::vector<std::byte> slots, keys, sequences, scores, steps;
            copy_binary(col.slot_ordinals, slots);
            copy_binary(col.keys, keys);
            copy_binary(col.sequences, sequences);
            copy_binary(col.embedding_values, page.values);
            copy_binary(col.scores, scores);
            const auto optimizer_bytes = binary_value(
                batch_array.children[col.optimizer_state_values], row);
            const auto steps_bytes =
                binary_value(batch_array.children[col.steps], row);
            const auto require_fixed_width =
                [](std::span<const std::byte> bytes, size_t width,
                   const char *field) {
                  if (bytes.size() % width != 0)
                    throw std::runtime_error(
                        std::string(
                            "native Lance page image field has a truncated ") +
                        field);
                };
            require_fixed_width(slots, sizeof(uint32_t), "slot_ordinals");
            require_fixed_width(keys, sizeof(uint64_t), "keys");
            require_fixed_width(sequences, sizeof(uint64_t), "sequences");
            require_fixed_width(scores, sizeof(uint64_t), "scores");
            require_fixed_width(steps_bytes, sizeof(uint32_t), "steps");
            const size_t key_count = keys.size() / sizeof(uint64_t);
            if (slots.size() / sizeof(uint32_t) != key_count ||
                sequences.size() / sizeof(uint64_t) != key_count ||
                (!scores.empty() &&
                 scores.size() / sizeof(uint64_t) != key_count) ||
                (!steps_bytes.empty() &&
                 steps_bytes.size() / sizeof(uint32_t) != key_count))
              throw std::runtime_error(
                  "native Lance page image key and payload lengths differ");
            page.checksum =
                binary_string(batch_array.children[col.checksum], row);
            if (checksum(slots, keys, sequences, page.values, scores,
                         optimizer_bytes, steps_bytes) != page.checksum)
              throw std::runtime_error("page image checksum mismatch");
            if (load_optimizer_state) {
              page.optimizer_values.assign(optimizer_bytes.begin(),
                                           optimizer_bytes.end());
              steps.assign(steps_bytes.begin(), steps_bytes.end());
            }
            page.slot_ordinals.assign(
                reinterpret_cast<const uint32_t *>(slots.data()),
                reinterpret_cast<const uint32_t *>(slots.data() +
                                                   slots.size()));
            page.keys.assign(
                reinterpret_cast<const uint64_t *>(keys.data()),
                reinterpret_cast<const uint64_t *>(keys.data() + keys.size()));
            page.sequences.assign(
                reinterpret_cast<const uint64_t *>(sequences.data()),
                reinterpret_cast<const uint64_t *>(sequences.data() +
                                                   sequences.size()));
            page.scores.assign(
                reinterpret_cast<const uint64_t *>(scores.data()),
                reinterpret_cast<const uint64_t *>(scores.data() +
                                                   scores.size()));
            page.steps.assign(reinterpret_cast<const uint32_t *>(steps.data()),
                              reinterpret_cast<const uint32_t *>(steps.data() +
                                                                 steps.size()));
            page.optimizer_dtype = binary_string(
                batch_array.children[col.optimizer_state_dtype], row);
            page.optimizer_stride = fixed_value<uint32_t>(
                batch_array.children[col.optimizer_state_stride], row);
            if (!visitor(std::move(page))) {
              batch_array.release(&batch_array);
              return;
            }
          }
          batch_array.release(&batch_array);
        }
      },
      storage_options);
}

} // namespace se
