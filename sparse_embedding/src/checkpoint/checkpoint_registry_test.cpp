#include "sparse_embedding/src/checkpoint/checkpoint_registry.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
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
  } catch (const std::exception &) {
    return;
  }
  throw std::runtime_error(message);
}

struct TestCheckpointWriter {
  void begin(std::string_view, size_t, size_t, bool,
             se::CheckpointExportOptions) {}

  void write_batch(size_t, const void *, size_t, const void *, size_t,
                   std::span<const uint64_t>, const void *, size_t,
                   std::span<const uint32_t>) {}

  uint64_t commit() { return 17; }
};

struct TestCheckpointReader {
  void open(std::string_view, size_t, uint64_t, int64_t) {}

  bool read_batch(size_t, std::vector<std::byte> &, std::vector<std::byte> &,
                  std::vector<uint64_t> &, std::vector<std::byte> &,
                  std::vector<uint32_t> &) {
    return false;
  }

  se::CheckpointVersion version_info() const { return {3, 5, 7}; }
};

se::CheckpointFactories make_test_checkpoint_factories() {
  return {
      pro::make_proxy<se::CheckpointWriterFacade, TestCheckpointWriter>(),
      pro::make_proxy<se::CheckpointReaderFacade, TestCheckpointReader>(),
  };
}

} // namespace

int main() {
  se::register_checkpoint_backend("registry_test",
                                  &make_test_checkpoint_factories);
  auto factories = se::make_checkpoint_factories("registry_test");
  require(factories.writer.has_value(), "checkpoint writer was not created");
  require(factories.reader.has_value(), "checkpoint reader was not created");
  require(factories.writer->commit() == 17,
          "checkpoint writer factory returned wrong adapter");
  require(factories.reader->version_info().version_id == 3,
          "checkpoint reader factory returned wrong adapter");

  require_throws(
      [] { se::make_checkpoint_factories("missing_registry_backend"); },
      "unknown checkpoint backend was accepted");
  require_throws(
      [] {
        se::register_checkpoint_backend("registry_test",
                                        &make_test_checkpoint_factories);
      },
      "duplicate checkpoint backend was accepted");
  require_throws(
      [] {
        se::register_checkpoint_backend("", &make_test_checkpoint_factories);
      },
      "empty checkpoint backend name was accepted");
  require_throws(
      [] { se::register_checkpoint_backend("null_registry_backend", nullptr); },
      "null checkpoint backend builder was accepted");

  constexpr size_t kThreads = 8;
  std::vector<std::thread> workers;
  workers.reserve(kThreads);
  std::atomic<size_t> ready{0};
  for (size_t worker = 0; worker < kThreads; ++worker) {
    workers.emplace_back([worker, &ready] {
      const std::string name = "registry_thread_" + std::to_string(worker);
      se::register_checkpoint_backend(name, &make_test_checkpoint_factories);
      auto local = se::make_checkpoint_factories(name);
      if (local.writer->commit() == 17)
        ready.fetch_add(1, std::memory_order_relaxed);
    });
  }
  for (auto &worker : workers)
    worker.join();
  require(ready.load(std::memory_order_relaxed) == kThreads,
          "concurrent checkpoint registry access failed");
}
