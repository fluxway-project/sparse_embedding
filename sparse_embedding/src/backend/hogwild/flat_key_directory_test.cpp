#include "sparse_embedding/src/backend/hogwild/core/flat_key_directory.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {

uint64_t splitmix(uint64_t value) {
  value ^= value >> 30;
  value *= 0xbf58476d1ce4e5b9ULL;
  value ^= value >> 27;
  value *= 0x94d049bb133111ebULL;
  value ^= value >> 31;
  return value;
}

void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

void prove_directory(size_t buckets) {
  constexpr size_t kKeys = 173;
  se::hogwild_internal::FlatKeyDirectory directory(buckets, 6);
  std::vector<uint64_t> slot_keys(kKeys + 11'100);
  const auto key_for_slot = [&](uint32_t slot_id) noexcept {
    return slot_keys[slot_id];
  };
  require(directory.bucket_count() == buckets, "bucket count was rounded");
  require(directory.uses_power_of_two_mask() ==
              ((buckets & (buckets - 1)) == 0),
          "range reduction path does not match bucket count");

  for (uint32_t key = 1; key <= kKeys; ++key) {
    slot_keys[key + 1000] = key;
    directory.insert_hashed(key, splitmix(key), key + 1000, key_for_slot);
  }
  for (uint32_t key = 1; key <= kKeys; ++key)
    require(directory.find_hashed(key, splitmix(key), key_for_slot) ==
                key + 1000,
            "inserted key was not found");

  for (uint32_t key = 3; key <= kKeys; key += 3)
    require(directory.remove_hashed(key, splitmix(key), key_for_slot) ==
                key + 1000,
            "remove returned the wrong slot");
  for (uint32_t key = 3; key <= kKeys; key += 3)
    require(
        !directory.find_hashed(key, splitmix(key), key_for_slot).has_value(),
        "removed key remained visible");
  for (uint32_t key = 10'000; key < 10'050; ++key) {
    slot_keys[key + 1000] = key;
    directory.insert_hashed(key, splitmix(key), key + 1000, key_for_slot);
  }

  se::hogwild_internal::FlatKeyDirectory clone(buckets, 6);
  size_t copied = 0;
  directory.for_each([&](uint32_t slot) {
    const uint64_t key = key_for_slot(slot);
    clone.insert_hashed(key, splitmix(key), slot, key_for_slot);
    ++copied;
  });
  require(copied == kKeys - kKeys / 3 + 50, "clone entry count is wrong");
  directory.for_each([&](uint32_t slot) {
    const uint64_t key = key_for_slot(slot);
    require(clone.find_hashed(key, splitmix(key), key_for_slot) == slot,
            "clone lost a key");
  });
}

template <typename Exception, typename Fn> void require_throws(Fn &&fn) {
  try {
    fn();
  } catch (const Exception &) {
    return;
  }
  throw std::runtime_error("operation did not throw the required exception");
}

} // namespace

int main() {
  using se::hogwild_internal::FlatKeyDirectory;
  require(FlatKeyDirectory(0, 6).bucket_count() == 2,
          "empty directory minimum is wrong");
  prove_directory(256);
  prove_directory(300);
  require(FlatKeyDirectory::bucket_count_for_key_capacity(0) == 2,
          "zero-key capacity is wrong");
  require(FlatKeyDirectory::bucket_count_for_key_capacity(65'536) == 262'144,
          "natural power-of-two capacity did not preserve the legacy mask");
  require(FlatKeyDirectory::bucket_count_for_key_capacity(81'920) == 512'000,
          "non-power-of-two capacity did not preserve the probe budget");
  require_throws<std::overflow_error>([] {
    FlatKeyDirectory::bucket_count_for_key_capacity(
        FlatKeyDirectory::maximum_key_capacity() + size_t{1});
  });

  FlatKeyDirectory validation(32, 6);
  std::array<uint64_t, 3> validation_keys{0, 1, 2};
  const auto validation_key = [&](uint32_t slot) noexcept {
    return validation_keys[slot];
  };
  validation.insert_hashed(1, splitmix(1), 1, validation_key);
  require_throws<std::logic_error>(
      [&] { validation.insert_hashed(1, splitmix(1), 2, validation_key); });
  require_throws<std::invalid_argument>(
      [&] { validation.insert_hashed(2, splitmix(2), 0, validation_key); });
  require_throws<std::overflow_error>([] {
    FlatKeyDirectory oversized(
        FlatKeyDirectory::maximum_bucket_count() + size_t{1}, 6);
  });
}
