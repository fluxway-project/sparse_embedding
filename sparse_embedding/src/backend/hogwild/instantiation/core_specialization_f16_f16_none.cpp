#include "sparse_embedding/src/backend/hogwild/hogwild_embedding_store.hpp"
#include "sparse_embedding/src/backend/hogwild/store_factory_impl.inl"

template class se::HogwildEmbeddingStoreCore<se::float16_t, se::float16_t>;

SE_DEFINE_STORE_FACTORY(se::float16_t, se::float16_t, f16_f16)
