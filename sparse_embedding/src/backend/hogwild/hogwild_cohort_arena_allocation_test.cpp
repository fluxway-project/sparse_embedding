#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include "sparse_embedding/src/backend/hogwild/hogwild_embedding_store.hpp"

namespace {

using Store =
    se::HogwildEmbeddingStore<uint64_t, float, float, se::NoScorePolicy>;

void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

uint64_t splitmix_hash(uint64_t key) {
  key ^= key >> 30;
  key *= 0xbf58476d1ce4e5b9ULL;
  key ^= key >> 27;
  key *= 0x94d049bb133111ebULL;
  return key ^ (key >> 31);
}

void prove_store_layout_and_exact_import() {
  constexpr size_t kKeyCount = 4096;
  constexpr size_t kDirectoryShards = 64;
  constexpr size_t kCohortSize = 16;
  constexpr size_t kCohortCount = kDirectoryShards / kCohortSize;
  const auto storage = se::create_page_storage();
  Store store({8}, {}, nullptr, se::NoScorePolicy{},
              se::HogwildStoreOptions{.thread_budget = kDirectoryShards,
                                      .storage = storage});
  std::vector<uint64_t> keys(kKeyCount);
  std::iota(keys.begin(), keys.end(), uint64_t{100000});
  std::vector<int64_t> lengths(kKeyCount, 8);
  std::vector<float> values(kKeyCount * 8);
  for (size_t index = 0; index < values.size(); ++index)
    values[index] = static_cast<float>(index);
  store.assign<uint64_t>(keys, values, lengths);

  std::vector<se::PageImageImport<float, float>> images;
  store.export_page_images(pro::make_proxy<se::FunctionFacade<bool(
                               size_t, se::PageImageExport<float, float> &&)>>(
      [&](size_t dim, se::PageImageExport<float, float> &&image) {
        require(dim == 8, "cohort store exported an invalid dimension");
        require(
            image.physical_page_id % kCohortCount ==
                (splitmix_hash(image.keys.front()) & (kDirectoryShards - 1)) /
                    kCohortSize,
            "cohort store page has an invalid physical cohort");
        for (uint64_t key : image.keys) {
          require(image.physical_page_id % kCohortCount ==
                      (splitmix_hash(key) & (kDirectoryShards - 1)) /
                          kCohortSize,
                  "cohort store page mixed shard cohorts");
        }
        images.push_back({dim, std::move(image)});
        return true;
      }));
  require(!images.empty(), "cohort store exported no physical pages");

  require(store.compact_page(8, images.front().image.physical_page_id) != 0,
          "cohort store did not compact a live page");
  std::unordered_set<uint64_t> live_keys;
  size_t live_key_count = 0;
  store.export_page_images(pro::make_proxy<se::FunctionFacade<bool(
                               size_t, se::PageImageExport<float, float> &&)>>(
      [&](size_t, se::PageImageExport<float, float> &&image) {
        live_key_count += image.keys.size();
        for (uint64_t key : image.keys) {
          require(live_keys.insert(key).second,
                  "cohort compaction left two live values for one key");
        }
        return true;
      }));
  require(live_key_count == keys.size() && live_keys.size() == keys.size(),
          "cohort compaction changed live key cardinality");

  Store restored({8}, {}, nullptr, se::NoScorePolicy{},
                 se::HogwildStoreOptions{.thread_budget = kDirectoryShards,
                                         .storage = storage});
  restored.import_page_images(images);
  std::vector<float> restored_values(values.size());
  restored.lookup<uint64_t>(keys, restored_values, lengths);
  require(restored_values == values,
          "cohort exact import did not preserve payload layout");
}

} // namespace

void prove_live_value_uniqueness_after_compaction() {
  prove_store_layout_and_exact_import();
}

int main() { prove_live_value_uniqueness_after_compaction(); }
