#pragma once

#include "sparse_embedding/src/backend/hogwild/public/store_construction.hpp"
#include <cstddef>
#include <memory>
#include <optional>

namespace se::py::detail {

class EmbeddingStoreWrapper;

struct PyOptimizer {
  se::OptimizerConfigVariant config;
};

struct PyInitializer {
  std::shared_ptr<pro::proxy<se::InitializerBuilderFacade>> builder;
};

void construct_store(EmbeddingStoreWrapper &, const PyInitializer &,
                     const std::optional<PyOptimizer> &, size_t,
                     se::StoreConstructionKind);

} // namespace se::py::detail
