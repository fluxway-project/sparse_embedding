#include "sparse_embedding/src/backend/hogwild/hogwild_embedding_store.hpp"
#include "sparse_embedding/src/storage/page_storage.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

template <typename KeyT> void prove_signed_key_compaction() {
  se::StorageOptions options;
  options.memory_limit_bytes = 0;
  options.spill_directory = std::filesystem::temp_directory_path();
  options.max_spill_bytes = 64 << 20;
  auto storage = se::create_page_storage(std::move(options));

  using Store =
      se::HogwildEmbeddingStore<KeyT, float, float, se::NoScorePolicy>;
  Store store({8}, {}, nullptr, se::NoScorePolicy{},
              se::HogwildStoreOptions{.thread_budget = 1, .storage = storage});
  const std::array<KeyT, 6> keys{std::numeric_limits<KeyT>::min(),
                                 static_cast<KeyT>(-4097),
                                 static_cast<KeyT>(-1),
                                 static_cast<KeyT>(0),
                                 static_cast<KeyT>(4097),
                                 std::numeric_limits<KeyT>::max()};
  std::array<int64_t, keys.size()> lengths{};
  lengths.fill(8);
  std::array<float, keys.size() * 8> values{};
  for (size_t i = 0; i < values.size(); ++i)
    values[i] = static_cast<float>(i + 1);

  store.template assign<KeyT>(keys, values, lengths);
  require(store.compact_page(8, 0) == keys.size(),
          "signed-key compaction did not migrate every key");

  std::array<float, values.size()> restored{};
  store.template lookup<KeyT>(keys, restored, lengths);
  require(restored == values,
          "signed-key compaction changed a key-to-value locator");

  std::vector<KeyT> exported_keys;
  store.template export_batch<KeyT>(
      pro::make_proxy<se::FunctionFacade<bool(size_t, std::span<const KeyT>,
                                              std::span<const float>,
                                              std::span<const uint64_t>)>>(
          [&](size_t dim, std::span<const KeyT> batch_keys,
              std::span<const float>, std::span<const uint64_t>) {
            require(dim == 8, "signed-key export changed the table dimension");
            exported_keys.insert(exported_keys.end(), batch_keys.begin(),
                                 batch_keys.end());
            return true;
          }),
      keys.size());
  std::ranges::sort(exported_keys);
  auto expected_keys = keys;
  std::ranges::sort(expected_keys);
  require(std::ranges::equal(exported_keys, expected_keys),
          "signed-key export did not preserve key values");
}

} // namespace

int main() {
  prove_signed_key_compaction<int32_t>();
  prove_signed_key_compaction<int64_t>();
}
