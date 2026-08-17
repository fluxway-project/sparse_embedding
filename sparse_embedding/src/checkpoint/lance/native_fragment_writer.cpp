#include "sparse_embedding/src/checkpoint/lance/native_fragment_writer.hpp"
#include "sparse_embedding/src/checkpoint/lance/layout_schema.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cstring>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include <lance/lance.h>

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
  if (stride != 0 && data.size() != stride * count)
    throw std::invalid_argument("native Lance payload stride mismatch");
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

std::vector<size_t> schema_indices(const ArrowSchema &schema,
                                   std::span<const std::string_view> names) {
  std::vector<size_t> indices;
  indices.reserve(names.size());
  for (const auto name : names) {
    size_t found = SIZE_MAX;
    for (int64_t index = 0; index < schema.n_children; ++index) {
      if (schema.children[index]->name &&
          name == schema.children[index]->name) {
        found = static_cast<size_t>(index);
        break;
      }
    }
    if (found == SIZE_MAX)
      throw std::runtime_error(
          "native Lance restore schema is missing a field");
    indices.push_back(found);
  }
  return indices;
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

ArrowSchema clone_schema(const ArrowSchema &source) {
  ArrowSchema clone{
      source.format,
      source.name,
      source.metadata,
      source.flags,
      source.n_children,
      source.n_children == 0
          ? nullptr
          : new ArrowSchema *[static_cast<size_t>(source.n_children)],
      nullptr,
      &NativeFragmentWriter::release_schema,
      nullptr};
  for (int64_t index = 0; index < source.n_children; ++index)
    clone.children[index] = new ArrowSchema(*source.children[index]);
  return clone;
}

struct BatchState {
  ArrowSchema schema;
  ArrowArray array;
  bool consumed = false;
};

int stream_schema(ArrowArrayStream *stream, ArrowSchema *out) {
  auto *state = static_cast<BatchState *>(stream->private_data);
  *out = state->schema;
  state->schema.release = nullptr;
  return 0;
}

int stream_next(ArrowArrayStream *stream, ArrowArray *out) {
  auto *state = static_cast<BatchState *>(stream->private_data);
  if (state->consumed) {
    *out = ArrowArray{};
    return 0;
  }
  state->consumed = true;
  *out = state->array;
  state->array.release = nullptr;
  return 0;
}

const char *stream_error(ArrowArrayStream *) { return nullptr; }

void stream_release(ArrowArrayStream *stream) {
  auto *state = static_cast<BatchState *>(stream->private_data);
  if (state) {
    if (state->schema.release)
      state->schema.release(&state->schema);
    if (state->array.release)
      state->array.release(&state->array);
    delete state;
  }
  stream->private_data = nullptr;
  stream->release = nullptr;
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

void validate_unique_source_keys(LanceDataset *source,
                                 std::span<const std::string_view> key_columns,
                                 std::unordered_set<std::string> &seen) {
  std::unique_ptr<LanceScanner, ScannerDeleter> scanner(
      lance_scanner_new(source, nullptr, nullptr));
  if (!scanner)
    throw std::runtime_error("native Lance merge scanner creation failed");
  ArrowStreamGuard stream;
  if (lance_scanner_to_arrow_stream(scanner.get(), stream.get()) != 0) {
    check_lance(-1, "native Lance merge scanner");
  }
  ArrowSchemaGuard schema;
  if (stream.get()->get_schema(stream.get(), schema.get()) != 0) {
    throw std::runtime_error("native Lance merge schema read failed");
  }
  std::vector<size_t> key_indices;
  key_indices.reserve(key_columns.size());
  for (auto key_column : key_columns) {
    size_t found = SIZE_MAX;
    for (int64_t index = 0; index < schema.get()->n_children; ++index) {
      const char *name = schema.get()->children[index]->name;
      if (name && key_column == name) {
        found = static_cast<size_t>(index);
        break;
      }
    }
    if (found == SIZE_MAX) {
      return;
    }
    key_indices.push_back(found);
  }
  for (;;) {
    ArrowArray batch{};
    if (stream.get()->get_next(stream.get(), &batch) != 0) {
      throw std::runtime_error("native Lance merge batch read failed");
    }
    if (!batch.release)
      break;
    for (int64_t row = 0; row < batch.length; ++row) {
      std::string composite;
      for (size_t index : key_indices) {
        const auto *column = batch.children[index];
        composite.append("\0", 1);
        if (std::string_view(schema.get()->children[index]->format) == "I") {
          const auto *values =
              static_cast<const uint32_t *>(column->buffers[1]);
          composite += std::to_string(values[row]);
        } else if (std::string_view(schema.get()->children[index]->format) ==
                   "L") {
          const auto *values =
              static_cast<const uint64_t *>(column->buffers[1]);
          composite += std::to_string(values[row]);
        } else {
          batch.release(&batch);
          throw std::invalid_argument(
              "native Lance merge key column has unsupported type");
        }
      }
      if (!seen.emplace(std::move(composite)).second) {
        batch.release(&batch);
        throw std::invalid_argument(
            "native Lance merge rejected duplicate checkpoint key");
      }
    }
    batch.release(&batch);
  }
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

ArrowArray *numeric_column(std::span<const uint32_t> values) {
  return make_fixed_column("I", "", bytes_of(values), sizeof(uint32_t),
                           values.size());
}
ArrowArray *numeric_column(std::span<const uint64_t> values) {
  return make_fixed_column("L", "", bytes_of(values), sizeof(uint64_t),
                           values.size());
}

void write_stream(const std::string &uri, ArrowSchema *schema,
                  ArrowArray *array, bool &created, uint64_t &version) {
  auto *state = new BatchState{clone_schema(*schema), *array, false};
  ArrowArrayStream stream{&stream_schema, &stream_next, &stream_error,
                          &stream_release, state};
  LanceDataset *dataset = nullptr;
  int32_t mode = created ? LANCE_WRITE_APPEND : LANCE_WRITE_CREATE;
  check_lance(lance_dataset_write(uri.c_str(), schema, &stream, mode, nullptr,
                                  &dataset),
              "native Lance fragment write");
  created = true;
  version = lance_dataset_version(dataset);
  lance_dataset_close(dataset);
  if (schema->release)
    schema->release(schema);
  schema->release = nullptr;
  array->release = nullptr;
}

} // namespace

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

void NativeFragmentWriter::begin_fragment(
    std::string_view staging_uri, uint32_t participant_index,
    std::span<const uint32_t> dimensions, std::string_view embedding_dtype,
    std::string_view optimizer_dtype, std::string_view score_policy,
    uint32_t protocol_version, std::string_view schema_digest, bool has_score) {
  root_ = std::string(staging_uri);
  entries_uri_ = root_ + "/entries";
  pages_uri_ = root_ + "/page_images";
  embedding_dtype_ = embedding_dtype;
  optimizer_dtype_ = optimizer_dtype;
  participant_index_ = participant_index;
  has_score_ = has_score;
  if (protocol_version != 1)
    throw std::invalid_argument("unsupported checkpoint protocol version");
  const std::string dimensions_value = join_dimensions(dimensions);
  const std::array<std::pair<std::string, std::string>, 5> metadata = {{
      {"sparse_embedding.protocol_version", std::to_string(protocol_version)},
      {"sparse_embedding.dimensions", dimensions_value},
      {"sparse_embedding.embedding_dtype", std::string(embedding_dtype)},
      {"sparse_embedding.optimizer_dtype", std::string(optimizer_dtype)},
      {"sparse_embedding.score_policy", std::string(score_policy)},
  }};
  schema_metadata_ = make_arrow_metadata(metadata);
  artifact_ = NativeFragmentArtifact{};
  artifact_.locator = root_;
  artifact_.schema_digest = std::string(schema_digest);
  content_hash_ = 1469598103934665603ULL;
  if (root_.find("://") == std::string::npos)
    std::filesystem::create_directories(root_);
}

void NativeFragmentWriter::write_entries(
    size_t dim, std::span<const uint64_t> keys,
    std::span<const std::byte> values, size_t value_stride,
    std::span<const uint64_t> scores, std::span<const std::byte> optimizer,
    size_t optimizer_stride, std::span<const uint32_t> steps,
    std::span<const uint64_t> sequences, uint64_t physical_page_id,
    std::span<const uint32_t> slots) {
  const size_t count = keys.size();
  if (values.size() != count * value_stride || sequences.size() != count ||
      slots.size() != count || (!scores.empty() && scores.size() != count) ||
      (!steps.empty() && steps.size() != count))
    throw std::invalid_argument("native Lance entry batch shape mismatch");
  if (optimizer_stride != 0 && optimizer.size() != count * optimizer_stride)
    throw std::invalid_argument("native Lance optimizer batch shape mismatch");

  std::vector<ArrowArray *> children;
  children.push_back(numeric_column(std::span<const uint32_t>{
      std::vector<uint32_t>(count, static_cast<uint32_t>(dim))}));
  children.push_back(numeric_column(keys));
  children.push_back(make_strided_binary_column(values, value_stride, count));
  const auto embedding_dtype_bytes = string_bytes(embedding_dtype_);
  children.push_back(make_repeated_binary_column(embedding_dtype_bytes, count));
  children.push_back(
      make_strided_binary_column(optimizer, optimizer_stride, count));
  const auto optimizer_dtype_bytes = string_bytes(optimizer_dtype_);
  children.push_back(make_repeated_binary_column(optimizer_dtype_bytes, count));
  children.push_back(
      numeric_column(std::span<const uint32_t>{std::vector<uint32_t>(
          count,
          optimizer_stride == 0 || optimizer_dtype_.empty()
              ? 0
              : static_cast<uint32_t>(optimizer_stride /
                                      dtype_itemsize(optimizer_dtype_)))}));
  children.push_back(numeric_column(
      steps.empty() ? std::span<const uint32_t>{std::vector<uint32_t>(count)}
                    : steps));
  children.push_back(numeric_column(
      scores.empty() ? std::span<const uint64_t>{std::vector<uint64_t>(count)}
                     : scores));
  children.push_back(numeric_column(sequences));
  children.push_back(numeric_column(std::span<const uint64_t>{
      std::vector<uint64_t>(count, physical_page_id)}));
  children.push_back(numeric_column(slots));
  const auto active_row_state = string_bytes("active");
  children.push_back(make_repeated_binary_column(active_row_state, count));
  children.push_back(
      numeric_column(std::span<const uint64_t>{std::vector<uint64_t>(count)}));
  children.push_back(numeric_column(std::span<const uint32_t>{
      std::vector<uint32_t>(count, participant_index_)}));

  auto schema = se::lance_layout::make_entries_schema(
      schema_metadata_.empty() ? nullptr : schema_metadata_.data());
  auto array = make_struct_array(std::move(children), count);
  write_stream(entries_uri_, &schema, &array, entries_created_,
               artifact_.entries_version);
  artifact_.entry_count += count;
}

void NativeFragmentWriter::write_page(
    size_t embedding_dim, uint64_t physical_page_id,
    uint32_t page_slot_stride_bytes, std::span<const uint32_t> slots,
    std::span<const uint64_t> keys, std::span<const uint64_t> sequences,
    std::span<const std::byte> values, std::span<const uint64_t> scores,
    std::span<const std::byte> optimizer, size_t optimizer_stride,
    std::span<const uint32_t> steps, std::string_view page_checksum) {
  const size_t count = keys.empty() ? 0 : 1;
  std::vector<ArrowArray *> children;
  children.push_back(numeric_column(std::span<const uint32_t>{
      count == 0 ? std::vector<uint32_t>{}
                 : std::vector<uint32_t>{participant_index_}}));
  children.push_back(numeric_column(std::span<const uint32_t>{
      count == 0
          ? std::vector<uint32_t>{}
          : std::vector<uint32_t>{static_cast<uint32_t>(embedding_dim)}}));
  children.push_back(numeric_column(std::span<const uint64_t>{
      count == 0 ? std::vector<uint64_t>{}
                 : std::vector<uint64_t>{physical_page_id}}));
  children.push_back(numeric_column(std::span<const uint32_t>{
      count == 0 ? std::vector<uint32_t>{}
                 : std::vector<uint32_t>{page_slot_stride_bytes}}));
  auto single_row = [count](std::span<const std::byte> value) {
    return count == 0 ? make_strided_binary_column({}, 0, 0)
                      : make_single_binary_column(value);
  };
  auto scalar = [&](std::string_view value) {
    const auto bytes = string_bytes(value);
    return single_row(bytes);
  };
  children.push_back(scalar(embedding_dtype_));
  const auto slot_bytes = bytes_of(slots);
  const auto key_bytes = bytes_of(keys);
  const auto sequence_bytes = bytes_of(sequences);
  const auto value_bytes = std::vector<std::byte>(values.begin(), values.end());
  const auto score_bytes = bytes_of(scores);
  const auto optimizer_bytes =
      std::vector<std::byte>(optimizer.begin(), optimizer.end());
  children.push_back(single_row(slot_bytes));
  children.push_back(single_row(key_bytes));
  children.push_back(single_row(sequence_bytes));
  children.push_back(single_row(value_bytes));
  children.push_back(single_row(score_bytes));
  children.push_back(single_row(optimizer_bytes));
  children.push_back(scalar(optimizer_dtype_));
  children.push_back(numeric_column(std::span<const uint32_t>{
      count == 0
          ? std::vector<uint32_t>{}
          : std::vector<uint32_t>{static_cast<uint32_t>(optimizer_stride)}}));
  const auto step_bytes = bytes_of(steps);
  children.push_back(single_row(step_bytes));
  children.push_back(scalar(page_checksum));
  children.push_back(scalar("active"));
  children.push_back(numeric_column(std::span<const uint64_t>{
      count == 0 ? std::vector<uint64_t>{} : std::vector<uint64_t>{0}}));

  auto schema = se::lance_layout::make_page_images_schema(
      schema_metadata_.empty() ? nullptr : schema_metadata_.data());
  auto array = make_struct_array(std::move(children), count);
  write_stream(pages_uri_, &schema, &array, pages_created_,
               artifact_.page_images_version);
  if (!slots.empty() || !keys.empty() || !values.empty())
    artifact_.page_count += 1;
}

void NativeFragmentWriter::write_page_image(
    size_t embedding_dim, uint64_t physical_page_id,
    uint32_t page_slot_stride_bytes, size_t embedding_row_bytes,
    std::span<const uint32_t> slots, std::span<const uint64_t> keys,
    std::span<const uint64_t> sequences, std::span<const std::byte> values,
    std::span<const uint64_t> scores, std::span<const std::byte> optimizer,
    size_t optimizer_stride, size_t optimizer_itemsize,
    std::span<const uint32_t> steps) {
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
  write_entries(embedding_dim, keys, values, embedding_row_bytes, scores,
                optimizer, optimizer_stride_bytes, steps, sequences,
                physical_page_id, slots);
  write_page(embedding_dim, physical_page_id, page_slot_stride_bytes, slots,
             keys, sequences, values, scores, optimizer, optimizer_stride,
             steps, page_digest);
  content_hash_ = fnv1a(slot_bytes, content_hash_);
  content_hash_ = fnv1a(key_bytes, content_hash_);
  content_hash_ = fnv1a(sequence_bytes, content_hash_);
  content_hash_ = fnv1a(value_bytes, content_hash_);
  content_hash_ = fnv1a(score_bytes, content_hash_);
  content_hash_ = fnv1a(optimizer_bytes, content_hash_);
  content_hash_ = fnv1a(step_bytes, content_hash_);
  std::array<char, 17> digest{};
  auto [end, error] =
      std::to_chars(digest.data(), digest.data() + 16, content_hash_, 16);
  if (error != std::errc{})
    throw std::runtime_error("failed to format native fragment digest");
  artifact_.content_digest.assign(digest.data(), end);
}

NativeFragmentArtifact NativeFragmentWriter::finalize() {
  if (!entries_created_) {
    write_entries(0, {}, {}, 0, {}, {}, 0, {}, {}, 0, {});
  }
  if (!pages_created_) {
    write_page(0, 0, 0, {}, {}, {}, {}, {}, {}, 0, {}, "");
  }
  if (artifact_.content_digest.empty()) {
    std::array<char, 17> digest{};
    auto [end, error] =
        std::to_chars(digest.data(), digest.data() + 16, content_hash_, 16);
    if (error != std::errc{})
      throw std::runtime_error("failed to format empty native fragment digest");
    artifact_.content_digest.assign(digest.data(), end);
  }
  artifact_.physical_bytes = 0;
  if (root_.find("://") == std::string::npos) {
    for (const auto &path :
         std::filesystem::recursive_directory_iterator(root_))
      if (path.is_regular_file())
        artifact_.physical_bytes += path.file_size();
  }
  return artifact_;
}

NativeMergeArtifact merge_native_lance_sources(
    std::string_view target_uri, std::span<const std::string> source_uris,
    std::span<const std::string_view> keys, uint64_t checkpoint_version) {
  if (source_uris.empty() || keys.empty())
    throw std::invalid_argument("native Lance merge requires sources and keys");
  NativeMergeArtifact result;
  result.locator = std::string(target_uri);
  std::unordered_set<std::string> seen_keys;
  std::unique_ptr<LanceDataset, DatasetDeleter> target(
      lance_dataset_open(std::string(target_uri).c_str(), nullptr, 0));
  if (!target) {
    const char *message = lance_last_error_message();
    std::string detail = message ? message : "unknown Lance error";
    if (message)
      lance_free_string(message);
    throw std::runtime_error("native Lance merge target open failed: " +
                             detail);
  }

  for (const auto &source_uri : source_uris) {
    std::unique_ptr<LanceDataset, DatasetDeleter> source(
        lance_dataset_open(source_uri.c_str(), nullptr, 0));
    if (!source) {
      const char *message = lance_last_error_message();
      std::string detail = message ? message : "unknown Lance error";
      if (message)
        lance_free_string(message);
      throw std::runtime_error("native Lance merge source open failed for " +
                               source_uri + ": " + detail);
    }
    validate_unique_source_keys(source.get(), keys, seen_keys);
  }

  if (checkpoint_version != 0) {
    const char *columns[] = {"row_state", "last_updated_checkpoint_version"};
    const std::string version = std::to_string(checkpoint_version);
    const char *values[] = {"'deleted'", version.c_str()};
    uint64_t updated = 0;
    check_lance(lance_dataset_update(target.get(), "row_state = 'active'",
                                     columns, values, 2, &updated),
                "native Lance checkpoint tombstone update");
    result.deleted_rows = updated;
  }

  for (const auto &source_uri : source_uris) {
    std::unique_ptr<LanceDataset, DatasetDeleter> source(
        lance_dataset_open(source_uri.c_str(), nullptr, 0));
    if (!source) {
      const char *message = lance_last_error_message();
      std::string detail = message ? message : "unknown Lance error";
      if (message)
        lance_free_string(message);
      throw std::runtime_error("native Lance merge source open failed for " +
                               source_uri + ": " + detail);
    }
    std::unique_ptr<LanceScanner, ScannerDeleter> scanner(
        lance_scanner_new(source.get(), nullptr, nullptr));
    if (!scanner) {
      throw std::runtime_error("native Lance merge scanner creation failed");
    }
    ArrowStreamGuard stream;
    if (lance_scanner_to_arrow_stream(scanner.get(), stream.get()) != 0) {
      check_lance(-1, "native Lance merge scanner");
    }
    std::vector<const char *> key_ptrs;
    key_ptrs.reserve(keys.size());
    for (auto key : keys)
      key_ptrs.push_back(key.data());
    LanceMergeInsertParams params{};
    params.when_matched = LANCE_MERGE_WHEN_MATCHED_UPDATE_ALL;
    params.when_not_matched = LANCE_MERGE_WHEN_NOT_MATCHED_INSERT_ALL;
    params.when_not_matched_by_source =
        LANCE_MERGE_WHEN_NOT_MATCHED_BY_SOURCE_KEEP;
    LanceMergeInsertResult merge_result{};
    check_lance(lance_dataset_merge_insert(target.get(), key_ptrs.data(),
                                           key_ptrs.size(), stream.get(),
                                           &params, &merge_result),
                "native Lance merge_insert");
    result.target_version = lance_dataset_version(target.get());
    result.inserted_rows += merge_result.num_inserted_rows;
    result.updated_rows += merge_result.num_updated_rows;
  }
  if (checkpoint_version != 0) {
    const char *columns[] = {"last_updated_checkpoint_version"};
    const std::string version = std::to_string(checkpoint_version);
    const char *values[] = {version.c_str()};
    uint64_t updated = 0;
    check_lance(lance_dataset_update(target.get(), "row_state = 'active'",
                                     columns, values, 1, &updated),
                "native Lance checkpoint version update");
  }
  result.target_version = lance_dataset_version(target.get());
  return result;
}

namespace {

template <typename Callback>
void scan_dataset(std::string_view uri, uint64_t version, Callback &&callback) {
  std::unique_ptr<LanceDataset, DatasetDeleter> dataset(
      lance_dataset_open(std::string(uri).c_str(), nullptr, version));
  if (!dataset)
    check_lance(-1, "native Lance restore dataset open");
  std::unique_ptr<LanceScanner, ScannerDeleter> scanner(
      lance_scanner_new(dataset.get(), nullptr, nullptr));
  if (!scanner)
    throw std::runtime_error("native Lance restore scanner creation failed");
  ArrowStreamGuard stream;
  if (lance_scanner_to_arrow_stream(scanner.get(), stream.get()) != 0)
    check_lance(-1, "native Lance restore scanner");
  ArrowSchemaGuard schema;
  if (stream.get()->get_schema(stream.get(), schema.get()) != 0)
    throw std::runtime_error("native Lance restore schema read failed");
  callback(*schema.get(), *stream.get());
}

} // namespace

void scan_native_active_entry_keys(std::string_view uri, uint64_t version,
                                   NativeKeyBatchVisitor visitor) {
  if (!visitor)
    throw std::invalid_argument("native Lance key visitor is empty");
  scan_dataset(
      uri, version, [&](const ArrowSchema &schema, ArrowArrayStream &stream) {
        constexpr std::array<std::string_view, 2> fields = {"key", "row_state"};
        const auto indices = schema_indices(schema, fields);
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
            if (binary_string(batch.children[indices[1]], row) ==
                kActiveRowState)
              keys.push_back(
                  fixed_value<uint64_t>(batch.children[indices[0]], row));
          }
          batch.release(&batch);
          if (!keys.empty() && !visitor(keys))
            break;
        }
      });
}

void scan_native_entries(std::string_view uri, uint64_t version,
                         std::span<const uint64_t> selected_keys,
                         NativeEntryBatchVisitor visitor) {
  if (!visitor)
    throw std::invalid_argument("native Lance entry visitor is empty");
  std::unordered_set<uint64_t> selected(selected_keys.begin(),
                                        selected_keys.end());
  scan_dataset(
      uri, version, [&](const ArrowSchema &schema, ArrowArrayStream &stream) {
        constexpr std::array<std::string_view, 10> fields = {
            "dim",
            "key",
            "embedding_values",
            "embedding_dtype",
            "optimizer_state_values",
            "optimizer_state_dtype",
            "optimizer_state_stride",
            "step",
            "score",
            "row_state"};
        const auto indices = schema_indices(schema, fields);
        for (;;) {
          ArrowArray batch_array{};
          if (stream.get_next(&stream, &batch_array) != 0)
            throw std::runtime_error(
                "native Lance restore entry batch read failed");
          if (!batch_array.release)
            break;
          NativeEntryBatch batch;
          batch.embedding_offsets.push_back(0);
          batch.optimizer_offsets.push_back(0);
          for (int64_t row = 0; row < batch_array.length; ++row) {
            const uint64_t key =
                fixed_value<uint64_t>(batch_array.children[indices[1]], row);
            if (binary_string(batch_array.children[indices[9]], row) !=
                    kActiveRowState ||
                (!selected.empty() && !selected.contains(key)))
              continue;
            batch.dimensions.push_back(
                fixed_value<uint32_t>(batch_array.children[indices[0]], row));
            batch.keys.push_back(key);
            const auto embedding =
                binary_value(batch_array.children[indices[2]], row);
            append_binary(batch.embedding_values, batch.embedding_offsets,
                          embedding);
            batch.embedding_dtype =
                binary_string(batch_array.children[indices[3]], row);
            const auto optimizer =
                binary_value(batch_array.children[indices[4]], row);
            append_binary(batch.optimizer_values, batch.optimizer_offsets,
                          optimizer);
            batch.optimizer_dtype =
                binary_string(batch_array.children[indices[5]], row);
            batch.optimizer_strides.push_back(
                fixed_value<uint32_t>(batch_array.children[indices[6]], row));
            batch.steps.push_back(
                fixed_value<uint32_t>(batch_array.children[indices[7]], row));
            batch.scores.push_back(
                fixed_value<uint64_t>(batch_array.children[indices[8]], row));
          }
          batch_array.release(&batch_array);
          if (!batch.keys.empty() && !visitor(batch))
            break;
        }
      });
}

void scan_native_page_images(std::string_view uri, uint64_t version,
                             uint32_t participant_index,
                             NativePageImageVisitor visitor) {
  if (!visitor)
    throw std::invalid_argument("native Lance page visitor is empty");
  scan_dataset(
      uri, version, [&](const ArrowSchema &schema, ArrowArrayStream &stream) {
        constexpr std::array<std::string_view, 16> fields = {
            "source_participant_index",
            "embedding_dim",
            "physical_page_id",
            "page_slot_stride_bytes",
            "embedding_dtype",
            "slot_ordinals",
            "keys",
            "sequences",
            "embedding_values",
            "scores",
            "optimizer_state_values",
            "optimizer_state_dtype",
            "optimizer_state_stride",
            "steps",
            "checksum",
            "row_state"};
        const auto indices = schema_indices(schema, fields);
        for (;;) {
          ArrowArray batch_array{};
          if (stream.get_next(&stream, &batch_array) != 0)
            throw std::runtime_error(
                "native Lance restore page batch read failed");
          if (!batch_array.release)
            break;
          for (int64_t row = 0; row < batch_array.length; ++row) {
            if (fixed_value<uint32_t>(batch_array.children[indices[0]], row) !=
                    participant_index ||
                binary_string(batch_array.children[indices[15]], row) !=
                    kActiveRowState)
              continue;
            NativePageImage page;
            page.embedding_dim =
                fixed_value<uint32_t>(batch_array.children[indices[1]], row);
            page.physical_page_id =
                fixed_value<uint64_t>(batch_array.children[indices[2]], row);
            page.page_slot_stride_bytes =
                fixed_value<uint32_t>(batch_array.children[indices[3]], row);
            page.embedding_dtype =
                binary_string(batch_array.children[indices[4]], row);
            auto copy_binary = [&](size_t index, std::vector<std::byte> &out) {
              const auto bytes =
                  binary_value(batch_array.children[indices[index]], row);
              out.assign(bytes.begin(), bytes.end());
            };
            std::vector<std::byte> slots, keys, sequences, scores, steps;
            copy_binary(5, slots);
            copy_binary(6, keys);
            copy_binary(7, sequences);
            copy_binary(8, page.values);
            copy_binary(9, scores);
            copy_binary(10, page.optimizer_values);
            copy_binary(13, steps);
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
            page.optimizer_dtype =
                binary_string(batch_array.children[indices[11]], row);
            page.optimizer_stride =
                fixed_value<uint32_t>(batch_array.children[indices[12]], row);
            page.checksum =
                binary_string(batch_array.children[indices[14]], row);
            if (!visitor(std::move(page))) {
              batch_array.release(&batch_array);
              return;
            }
          }
          batch_array.release(&batch_array);
        }
      });
}

} // namespace se
