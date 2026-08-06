#pragma once

#define HOGWILD_EXPLICIT_INSTANTIATION
#include "sparse_embedding/src/backend/hogwild/hogwild_embedding_store.hpp"
#include "sparse_embedding/src/optimizer/adam_optimizer.hpp"
#include "sparse_embedding/src/optimizer/sgd_optimizer.hpp"

#define SE_INSTANTIATE_HOGWILD_STORE(KEY_T, PARAM_T, OPT_T)                    \
  template class se::HogwildEmbeddingStore<KEY_T, PARAM_T, OPT_T,              \
                                           se::NoScorePolicy>;                 \
  template class se::HogwildEmbeddingStore<KEY_T, PARAM_T, OPT_T,              \
                                           se::LfuScorePolicy>;                \
  template class se::HogwildEmbeddingStore<KEY_T, PARAM_T, OPT_T,              \
                                           se::LruScorePolicy>;                \
  template class se::HogwildEmbeddingStore<KEY_T, PARAM_T, OPT_T,              \
                                           se::ManualScorePolicy>;

#define SE_INSTANTIATE_HOGWILD_PAIR(PARAM_T, OPT_T)                            \
  SE_INSTANTIATE_HOGWILD_STORE(uint64_t, PARAM_T, OPT_T)                       \
  SE_INSTANTIATE_HOGWILD_STORE(int64_t, PARAM_T, OPT_T)                        \
  SE_INSTANTIATE_HOGWILD_STORE(uint32_t, PARAM_T, OPT_T)                       \
  SE_INSTANTIATE_HOGWILD_STORE(int32_t, PARAM_T, OPT_T)
