#include "sparse_embedding/src/checkpoint/lance/native_entry_restore.hpp"

#include <functional>
#include <limits>
#include <string>

namespace {

void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

template <typename Function> void require_throws(Function &&function) {
  try {
    std::forward<Function>(function)();
  } catch (const std::runtime_error &) {
    return;
  }
  throw std::runtime_error("invalid restore batch was accepted");
}

template <typename T>
void append_value(std::vector<std::byte> &buffer, T value) {
  const auto bytes = std::as_bytes(std::span<const T>(&value, 1));
  buffer.insert(buffer.end(), bytes.begin(), bytes.end());
}

template <typename ParamT, typename OptT>
se::NativeEntryBatch make_batch(size_t count, size_t marker = 0) {
  se::NativeEntryBatch batch;
  batch.embedding_offsets.push_back(0);
  batch.optimizer_offsets.push_back(0);
  for (size_t row = 0; row < count; ++row) {
    const uint32_t dim = 2 + row % 2;
    batch.dimensions.push_back(dim);
    batch.keys.push_back(std::numeric_limits<uint64_t>::max() - row - marker);
    for (uint32_t value = 0; value < dim; ++value)
      append_value(batch.embedding_values, ParamT(row + value + marker));
    batch.embedding_offsets.push_back(batch.embedding_values.size());
    for (uint32_t value = 0; value < dim * 2; ++value)
      append_value(batch.optimizer_values, OptT(row + value + marker));
    batch.optimizer_offsets.push_back(batch.optimizer_values.size());
    batch.optimizer_strides.push_back(dim * 2);
    batch.steps.push_back(row + marker);
    batch.scores.push_back(dim * 10 + row + marker);
  }
  return batch;
}

template <typename KeyT, typename ParamT, typename OptT> struct CheckingStore {
  const se::NativeEntryBatch *expected = nullptr;
  const ParamT *previous_values = nullptr;
  size_t assignments = 0;
  size_t assigned_rows = 0;
  size_t score_rows = 0;
  size_t optimizer_rows = 0;
  bool reuse_values = false;
  bool scores_enabled = true;
  std::error_code assignment_error;

  bool has_score_policy() const { return scores_enabled; }

  std::error_code assign(std::span<const KeyT> keys,
                         std::span<const ParamT> values,
                         std::span<const int64_t> lengths) {
    require(expected != nullptr, "assignment outlived its scanner batch");
    require(keys.size() == expected->keys.size(), "assignment lost rows");
    require(values.size_bytes() == expected->embedding_values.size(),
            "assignment lost embedding bytes");
    require(std::memcmp(values.data(), expected->embedding_values.data(),
                        values.size_bytes()) == 0,
            "embedding bytes changed during import");
    for (size_t row = 0; row < keys.size(); ++row) {
      require(keys[row] == static_cast<KeyT>(expected->keys[row]),
              "key narrowing changed key bits");
      require(lengths[row] == expected->dimensions[row],
              "assignment lost dimension identity");
    }
    if (reuse_values && previous_values != nullptr)
      require(values.data() == previous_values,
              "same-sized batches did not reuse payload scratch");
    previous_values = values.data();
    ++assignments;
    assigned_rows += keys.size();
    return assignment_error;
  }

  std::error_code assign_scores(std::span<const KeyT> keys,
                                std::span<const uint64_t> scores,
                                std::span<const int64_t> lengths) {
    require(scores_enabled, "disabled score policy was invoked");
    for (size_t row = 0; row < keys.size(); ++row) {
      require(scores[row] == expected->scores[row], "score bytes changed");
      require(lengths[row] == expected->dimensions[row],
              "scores lost dimension identity");
    }
    score_rows += keys.size();
    return {};
  }

  void assign_opt_states(size_t dim, std::span<const KeyT> keys,
                         std::span<const OptT> values,
                         std::span<const uint32_t> steps) {
    size_t index = 0;
    const size_t stride = dim * 2;
    for (size_t row = 0; row < expected->keys.size(); ++row) {
      if (expected->dimensions[row] != dim)
        continue;
      require(index < keys.size(), "optimizer assignment lost rows");
      require(keys[index] == static_cast<KeyT>(expected->keys[row]),
              "optimizer key and dimension were separated");
      require(std::memcmp(values.data() + index * stride,
                          expected->optimizer_values.data() +
                              expected->optimizer_offsets[row],
                          stride * sizeof(OptT)) == 0,
              "optimizer bytes changed during grouping");
      if (expected->steps.empty())
        require(steps.empty(), "absent steps were synthesized");
      else
        require(steps[index] == expected->steps[row], "optimizer step changed");
      ++index;
    }
    require(index == keys.size(), "optimizer assignment added rows");
    optimizer_rows += keys.size();
  }
};

template <typename KeyT, typename ParamT, typename OptT>
void check_streaming() {
  CheckingStore<KeyT, ParamT, OptT> store;
  store.reuse_values = true;
  constexpr size_t batches = 32;
  constexpr size_t rows = se::CheckpointStreamPolicy::kScanBatchRows;
  se::restore_native_entry_batches<KeyT, ParamT, OptT>(
      store,
      [&](auto visitor) {
        for (size_t index = 0; index < batches; ++index) {
          auto batch = make_batch<ParamT, OptT>(rows, index);
          store.expected = &batch;
          require(visitor(batch), "restore cancelled a valid batch");
          require(store.assignments == index + 1 &&
                      store.assigned_rows == (index + 1) * rows &&
                      store.score_rows == store.assigned_rows &&
                      store.optimizer_rows == store.assigned_rows,
                  "restore retained a batch instead of consuming it");
          store.expected = nullptr;
        }
      },
      true);
}

void check_empty_optional_and_weight_only() {
  CheckingStore<uint64_t, float, float> store;
  auto batch = make_batch<float, float>(0);
  auto scanner = [&](auto visitor) {
    store.expected = &batch;
    visitor(batch);
  };
  se::restore_native_entry_batches<uint64_t, float, float>(store, scanner,
                                                           true);
  require(store.assignments == 0, "empty batch mutated the store");
  batch = make_batch<float, float>(7);
  batch.steps.clear();
  batch.scores.clear();
  se::restore_native_entry_batches<uint64_t, float, float>(store, scanner,
                                                           true);
  require(store.assigned_rows == 7 && store.optimizer_rows == 7 &&
              store.score_rows == 0,
          "optional metadata was not preserved");
  batch = make_batch<float, float>(9);
  CheckingStore<uint64_t, float, uint16_t> weight_store;
  weight_store.scores_enabled = false;
  weight_store.expected = &batch;
  // An incompatible optimizer item size must not matter for a weight-only load.
  se::restore_native_entry_batches<uint64_t, float, uint16_t>(
      weight_store, [&](auto visitor) { visitor(batch); }, false);
  require(weight_store.assigned_rows == 9 && weight_store.optimizer_rows == 0 &&
              weight_store.score_rows == 0,
          "weight-only restore imported auxiliary state");
}

void check_invalid_batches() {
  const std::vector<std::function<void(se::NativeEntryBatch &)>> corruptions{
      [](auto &batch) { batch.dimensions.pop_back(); },
      [](auto &batch) { batch.embedding_offsets.clear(); },
      [](auto &batch) { batch.embedding_offsets[1] = 1000000; },
      [](auto &batch) { batch.embedding_offsets[2] = 0; },
      [](auto &batch) { batch.optimizer_offsets[1] = 1000000; },
      [](auto &batch) { batch.optimizer_offsets[2] = 0; },
      [](auto &batch) { batch.optimizer_strides[0] += 1; },
      [](auto &batch) { batch.steps.pop_back(); },
      [](auto &batch) { batch.scores.pop_back(); },
  };
  for (const auto &corrupt : corruptions) {
    auto batch = make_batch<float, float>(4);
    corrupt(batch);
    CheckingStore<uint64_t, float, float> store;
    require_throws([&] {
      se::restore_native_entry_batches<uint64_t, float, float>(
          store, [&](auto visitor) { visitor(batch); }, true);
    });
    require(store.assignments == 0, "malformed batch mutated the store");
  }
  se::CheckpointStreamPolicy::validate_scan_batch_rows(0);
  se::CheckpointStreamPolicy::validate_scan_batch_rows(
      se::CheckpointStreamPolicy::kScanBatchRows);
  for (int64_t rows : {-1LL, se::CheckpointStreamPolicy::kScanBatchRows + 1LL})
    require_throws(
        [&] { se::CheckpointStreamPolicy::validate_scan_batch_rows(rows); });
  auto oversized =
      make_batch<float, float>(se::CheckpointStreamPolicy::kScanBatchRows + 1);
  CheckingStore<uint64_t, float, float> store;
  require_throws([&] {
    se::restore_native_entry_batches<uint64_t, float, float>(
        store, [&](auto visitor) { visitor(oversized); }, true);
  });
  require(store.assignments == 0, "oversized batch mutated the store");

  auto batch = make_batch<float, float>(4);
  store.expected = &batch;
  store.assignment_error = std::make_error_code(std::errc::not_enough_memory);
  require_throws([&] {
    se::restore_native_entry_batches<uint64_t, float, float>(
        store, [&](auto visitor) { visitor(batch); }, true);
  });
  require(store.score_rows == 0 && store.optimizer_rows == 0,
          "restore continued after an assignment failure");
}

} // namespace

int main() {
  check_streaming<int32_t, float, float>();
  check_streaming<uint32_t, float, uint16_t>();
  check_streaming<int64_t, uint16_t, float>();
  check_streaming<uint64_t, uint16_t, uint16_t>();
  check_empty_optional_and_weight_only();
  check_invalid_batches();
}
