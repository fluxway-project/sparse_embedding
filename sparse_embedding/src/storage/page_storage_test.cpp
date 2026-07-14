#include "sparse_embedding/src/storage/page_storage.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char *message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

} // namespace

int main() {
  se::StorageOptions options;
  options.memory_limit_bytes = 0;
  options.spill_directory = std::filesystem::temp_directory_path();
  options.max_spill_bytes = 1 << 20;

  auto storage = se::PageStorage::create(std::move(options));
  std::uintptr_t first_address = 0;
  {
    auto page = storage->allocate(4096, 64);
    first_address = reinterpret_cast<std::uintptr_t>(page.data());
    require(first_address % 64 == 0, "page is not cache-line aligned");
    page.data()[0] = std::byte{0x2a};
    page.data()[4095] = std::byte{0x7f};
    require(page.data()[0] == std::byte{0x2a}, "page start is not writable");
    require(page.data()[4095] == std::byte{0x7f}, "page end is not writable");

    auto stats = storage->stats();
    require(stats.memory_bytes == 0, "forced spill allocated heap memory");
    require(stats.spill_bytes == 4096, "spill bytes were not accounted");
    require(stats.spill_count == 1, "spill event was not accounted");
  }

  require(storage->stats().spill_bytes == 0,
          "released spill page remained live");
  auto reused = storage->allocate(4096, 64);
  require(reinterpret_cast<std::uintptr_t>(reused.data()) == first_address,
          "released spill extent was not reused");

  std::vector<se::StoragePage> pages;
  pages.reserve(128);
  for (size_t i = 0; i < 128; ++i) {
    pages.push_back(storage->allocate(4096, 64));
  }
  pages.clear();
  require(storage->stats().spill_bytes == reused.size(),
          "bulk spill release accounting mismatch");

  if (std::filesystem::is_directory("/dev/shm")) {
    se::StorageOptions memory_backed_options;
    memory_backed_options.memory_limit_bytes = 0;
    memory_backed_options.spill_directory = "/dev/shm";
    memory_backed_options.max_spill_bytes = 1 << 20;
    auto memory_backed =
        se::PageStorage::create(std::move(memory_backed_options));
    bool rejected = false;
    try {
      memory_backed->allocate(4096, 64);
    } catch (const std::runtime_error &) {
      rejected = true;
    }
    require(rejected, "memory-backed filesystem was accepted for spill");
  }
}
