#include "sparse_embedding/src/backend/hogwild/hogwild_embedding_store.hpp"
#include "sparse_embedding/src/storage/page_storage.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace {

void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

} // namespace

int main() {
  if (!se::NumaTopology::available())
    return 0;
  const size_t nodes = se::NumaTopology::node_count();
  if (nodes < 2)
    return 0;
  se::StorageOptions options;
  options.mode = se::StorageMode::kMemoryOnly;
  options.memory_limit_bytes = 64 << 20;
  auto storage = se::create_page_storage(std::move(options));
  using Store =
      se::HogwildEmbeddingStore<uint64_t, float, float, se::NoScorePolicy>;
  Store store({8}, {}, nullptr, se::NoScorePolicy{},
              se::HogwildStoreOptions{
                  .thread_budget = 2,
                  .locality = se::LocalityPolicy{se::LocalityMode::kNuma,
                                                 se::LocalityMode::kNuma},
                  .storage = storage});
  constexpr std::array<uint64_t, 8> keys{1, 2, 3, 4, 5, 6, 7, 8};
  constexpr std::array<int64_t, 8> lengths{8, 8, 8, 8, 8, 8, 8, 8};
  std::array<float, 64> inserted_values{};
  for (size_t row = 0; row < keys.size(); ++row)
    for (size_t column = 0; column < 8; ++column)
      inserted_values[row * 8 + column] = static_cast<float>(row * 8 + column);
  std::array<uint8_t, 8> inserted{};
  store.lookup_or_insert<uint64_t>(keys, inserted_values, lengths, inserted);
  for (uint8_t result : inserted)
    require(result == 1, "NUMA setup insert did not insert a unique key");
  std::array<float, 64> assigned_values{};
  for (size_t row = 0; row < keys.size(); ++row)
    for (size_t column = 0; column < 8; ++column)
      assigned_values[row * 8 + column] =
          static_cast<float>(100 + row * 8 + column);
  store.assign<uint64_t>(keys, assigned_values, lengths);
  std::array<float, 64> output{};
  store.forward_lookup<uint64_t>(keys, output, lengths);
  require(output == assigned_values,
          "NUMA-routed forward lookup changed the mapped payload");
  output.fill(0);
  store.forward_lookup<uint64_t>(keys, output, lengths);
  require(output == assigned_values,
          "NUMA-routed warm forward lookup changed the mapped payload");
  return 0;
}
