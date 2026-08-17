#include "sparse_embedding/src/backend/hogwild/hogwild_embedding_store.hpp"
#include "sparse_embedding/src/backend/hogwild/store_factory_impl.inl"

template class se::HogwildEmbeddingStoreCore<se::bfloat16_t, se::float16_t>;

SE_DEFINE_STORE_FACTORY(se::bfloat16_t, se::float16_t, bf16_f16)
