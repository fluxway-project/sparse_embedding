#include "sparse_embedding/src/backend/hogwild/hogwild_embedding_store.hpp"

#include <array>
#include <stdexcept>

int main() {
  using Store =
      se::HogwildEmbeddingStore<uint64_t, float, float, se::NoScorePolicy>;
  Store store({4}, {}, nullptr, se::NoScorePolicy{},
              se::HogwildStoreOptions{.thread_budget = 1});
  constexpr std::array<uint64_t, 2> keys{7, 7};
  constexpr std::array<int64_t, 2> dims{4, 4};
  std::array<float, 8> values{};
  std::array<uint8_t, 2> inserted{};
  bool assign_rejected = false;
  try {
    store.assign<uint64_t>(keys, values, dims);
  } catch (const std::invalid_argument &) {
    assign_rejected = true;
  }
  bool lookup_or_insert_rejected = false;
  try {
    store.lookup_or_insert<uint64_t>(keys, values, dims, inserted);
  } catch (const std::invalid_argument &) {
    lookup_or_insert_rejected = true;
  }
  return assign_rejected && lookup_or_insert_rejected ? 0 : 1;
}
