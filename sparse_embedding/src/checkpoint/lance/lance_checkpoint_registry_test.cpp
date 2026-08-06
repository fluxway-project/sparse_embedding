#include "sparse_embedding/src/checkpoint/checkpoint_registry.hpp"

#include <stdexcept>

int main() {
  auto factories = se::make_checkpoint_factories("lance");
  if (!factories.writer.has_value() || !factories.reader.has_value())
    throw std::runtime_error("Lance checkpoint adapter registration failed");
}
