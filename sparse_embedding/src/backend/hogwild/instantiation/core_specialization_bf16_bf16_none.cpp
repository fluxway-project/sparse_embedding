#include "sparse_embedding/src/backend/hogwild/hogwild_embedding_store.hpp"
#include "sparse_embedding/src/backend/hogwild/store_factory_impl.inl"

template class se::HogwildEmbeddingStoreCore<se::bfloat16_t, se::bfloat16_t>;

SE_DEFINE_STORE_FACTORY(se::bfloat16_t, se::bfloat16_t, bf16_bf16)
