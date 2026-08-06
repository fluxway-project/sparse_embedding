#include "sparse_embedding/src/backend/hogwild/hogwild_embedding_store.hpp"
#include "sparse_embedding/src/storage/page_storage.hpp"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <new>
#include <stdexcept>

namespace {

std::atomic<bool> count_allocations{false};
std::atomic<size_t> allocation_count{0};

void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

void *allocate(std::size_t size) {
  if (count_allocations.load(std::memory_order_relaxed))
    allocation_count.fetch_add(1, std::memory_order_relaxed);
  if (void *memory = std::malloc(size == 0 ? 1 : size))
    return memory;
  throw std::bad_alloc();
}

void *allocate_aligned(std::size_t size, std::size_t alignment) {
  if (count_allocations.load(std::memory_order_relaxed))
    allocation_count.fetch_add(1, std::memory_order_relaxed);
  const size_t rounded = (size + alignment - 1) / alignment * alignment;
  if (void *memory =
          std::aligned_alloc(alignment, rounded == 0 ? alignment : rounded))
    return memory;
  throw std::bad_alloc();
}

} // namespace

void *operator new(std::size_t size) { return allocate(size); }
void *operator new[](std::size_t size) { return allocate(size); }
void operator delete(void *memory) noexcept { std::free(memory); }
void operator delete[](void *memory) noexcept { std::free(memory); }
void operator delete(void *memory, std::size_t) noexcept { std::free(memory); }
void operator delete[](void *memory, std::size_t) noexcept {
  std::free(memory);
}
void *operator new(std::size_t size, std::align_val_t alignment) {
  return allocate_aligned(size, static_cast<size_t>(alignment));
}
void *operator new[](std::size_t size, std::align_val_t alignment) {
  return allocate_aligned(size, static_cast<size_t>(alignment));
}
void operator delete(void *memory, std::align_val_t) noexcept {
  std::free(memory);
}
void operator delete[](void *memory, std::align_val_t) noexcept {
  std::free(memory);
}
void operator delete(void *memory, std::size_t, std::align_val_t) noexcept {
  std::free(memory);
}
void operator delete[](void *memory, std::size_t, std::align_val_t) noexcept {
  std::free(memory);
}

int main() {
  se::StorageOptions options;
  options.mode = se::StorageMode::kMemoryOnly;
  options.memory_limit_bytes = 16 << 20;
  auto storage = se::create_page_storage(std::move(options));
  using Store =
      se::HogwildEmbeddingStore<uint64_t, float, float, se::NoScorePolicy>;
  Store store({8}, {}, nullptr, se::NoScorePolicy{},
              se::HogwildStoreOptions{.thread_budget = 1, .storage = storage});

  constexpr size_t kRows = 32;
  std::array<uint64_t, kRows> keys{};
  std::array<int64_t, kRows> lengths{};
  std::array<float, kRows * 8> values{};
  std::array<uint8_t, kRows> inserted{};
  for (size_t index = 0; index < kRows; ++index)
    keys[index] = index + 1;
  lengths.fill(8);
  store.lookup_or_insert<uint64_t>(keys, values, lengths, inserted);

  store.forward_lookup<uint64_t>(keys, values, lengths);
  store.forward_lookup<uint64_t>(keys, values, lengths);
  allocation_count.store(0, std::memory_order_relaxed);
  count_allocations.store(true, std::memory_order_release);
  for (size_t iteration = 0; iteration < 64; ++iteration)
    store.forward_lookup<uint64_t>(keys, values, lengths);
  count_allocations.store(false, std::memory_order_release);
  require(allocation_count.load(std::memory_order_relaxed) == 0,
          "prewarmed forward lookup allocated heap memory");
  return 0;
}
