#include "sparse_embedding/src/backend/hogwild/hogwild_embedding_store.hpp"
#include "sparse_embedding/src/backend/hogwild/store_factory_impl.inl"

template class se::HogwildEmbeddingStoreCore<float, se::bfloat16_t>;

SE_DEFINE_STORE_FACTORY(float, se::bfloat16_t, f32_bf16)
