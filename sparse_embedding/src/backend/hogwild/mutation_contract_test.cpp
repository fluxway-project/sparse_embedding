#include "sparse_embedding/src/backend/hogwild/hogwild_embedding_store.hpp"

#include <array>

int main() {
  using Store =
      se::HogwildEmbeddingStore<uint64_t, float, float, se::NoScorePolicy>;
  constexpr std::array<uint64_t, 2> keys{7, 7};
  constexpr std::array<int64_t, 2> dims{4, 4};
  std::array<float, 8> values{};
  Store lookup_store({4}, {}, nullptr, se::NoScorePolicy{},
                     se::HogwildStoreOptions{.thread_budget = 1});
  values.fill(0.0F);
  lookup_store.lookup_or_insert<uint64_t>(keys, values, dims);
  const bool outputs_match = values[0] == values[4] && values[1] == values[5] &&
                             values[2] == values[6] && values[3] == values[7];
  return outputs_match && lookup_store.size() == 1 ? 0 : 1;
}
