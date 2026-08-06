#include "sparse_embedding/src/checkpoint/checkpoint_batch_filter.hpp"

#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

template <class Function>
void require_throws(Function &&function, const char *message) {
  try {
    std::forward<Function>(function)();
  } catch (const std::invalid_argument &) {
    return;
  }
  throw std::runtime_error(message);
}

} // namespace

int main() {
  se::FilteredCheckpointBatch<int64_t, float, float> batch;
  const std::vector<int64_t> keys = {11, 12, 13, 14};
  const std::vector<float> values = {
      1, 2, 3, 4, 5, 6, 7, 8,
  };
  const std::vector<uint64_t> scores = {3, 7, 2, 9};
  const std::vector<float> optimizer_states = {
      10, 11, 20, 21, 30, 31, 40, 41,
  };
  const std::vector<uint32_t> steps = {100, 200, 300, 400};

  require(batch.assign(keys, values, scores, optimizer_states, steps, 2, 7) ==
              2,
          "filtered checkpoint count mismatch");
  require(batch.keys == std::vector<int64_t>({12, 14}),
          "filtered checkpoint key order mismatch");
  require(batch.values == std::vector<float>({3, 4, 7, 8}),
          "filtered checkpoint value layout mismatch");
  require(batch.scores == std::vector<uint64_t>({7, 9}),
          "filtered checkpoint score layout mismatch");
  require(batch.optimizer_states == std::vector<float>({20, 21, 40, 41}),
          "filtered checkpoint optimizer layout mismatch");
  require(batch.steps == std::vector<uint32_t>({200, 400}),
          "filtered checkpoint step layout mismatch");

  const auto keys_capacity = batch.keys.capacity();
  const auto values_capacity = batch.values.capacity();
  const auto scores_capacity = batch.scores.capacity();
  const auto optimizer_capacity = batch.optimizer_states.capacity();
  const auto steps_capacity = batch.steps.capacity();
  require(batch.assign(keys, values, scores, optimizer_states, steps, 2, 7) ==
              2,
          "reused filtered checkpoint count mismatch");
  require(batch.keys.capacity() == keys_capacity &&
              batch.values.capacity() == values_capacity &&
              batch.scores.capacity() == scores_capacity &&
              batch.optimizer_states.capacity() == optimizer_capacity &&
              batch.steps.capacity() == steps_capacity,
          "filtered checkpoint batch reallocated for unchanged capacity");

  require(batch.assign(keys, values, scores, {}, {}, 2, 10) == 0,
          "empty filtered checkpoint count mismatch");
  require(batch.keys.empty() && batch.values.empty() && batch.scores.empty() &&
              batch.optimizer_states.empty() && batch.steps.empty(),
          "empty filtered checkpoint retained data");

  const std::vector<float> invalid_values = {1};
  const std::vector<uint64_t> invalid_scores = {1};
  const std::vector<float> invalid_optimizer_states = {1};
  const std::vector<uint32_t> invalid_steps = {1};
  require_throws(
      [&] { batch.assign(keys, invalid_values, scores, {}, {}, 2, 0); },
      "invalid checkpoint value shape was accepted");
  require_throws(
      [&] { batch.assign(keys, values, invalid_scores, {}, {}, 2, 0); },
      "invalid checkpoint score shape was accepted");
  require_throws(
      [&] {
        batch.assign(keys, values, scores, invalid_optimizer_states, {}, 2, 0);
      },
      "invalid checkpoint optimizer shape was accepted");
  require_throws(
      [&] { batch.assign(keys, values, scores, {}, invalid_steps, 2, 0); },
      "invalid checkpoint step shape was accepted");
}
