// Copyright 2026 The Fluxway Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#define HOGWILD_EXPLICIT_INSTANTIATION
#include "sparse_embedding/src/backend/hogwild/hogwild_embedding_store.hpp"
#include "sparse_embedding/src/optimizer/adam_optimizer.hpp"
#include "sparse_embedding/src/optimizer/sgd_optimizer.hpp"
#include <cstdint>

namespace se {

template class EmbeddingArena<float>;
template class EmbeddingArena<float16_t>;
template class EmbeddingArena<bfloat16_t>;

template class TransientArena<float>;
template class TransientArena<float16_t>;
template class TransientArena<bfloat16_t>;

#define INSTANTIATE_STORE(KEY_T, PARAM_T, OPT_T)                               \
  template class HogwildEmbeddingStore<KEY_T, PARAM_T, OPT_T, NoScorePolicy>;  \
  template class HogwildEmbeddingStore<KEY_T, PARAM_T, OPT_T, LfuScorePolicy>; \
  template class HogwildEmbeddingStore<KEY_T, PARAM_T, OPT_T, LruScorePolicy>; \
  template class HogwildEmbeddingStore<KEY_T, PARAM_T, OPT_T,                  \
                                       ManualScorePolicy>;

#define INSTANTIATE_OPT_PAIR(PARAM_T, OPT_T)                                   \
  template class AdamOptimizer<PARAM_T, OPT_T>;                                \
  template class SgdOptimizer<PARAM_T, OPT_T>;                                 \
  INSTANTIATE_STORE(uint64_t, PARAM_T, OPT_T)                                  \
  INSTANTIATE_STORE(int64_t, PARAM_T, OPT_T)                                   \
  INSTANTIATE_STORE(uint32_t, PARAM_T, OPT_T)                                  \
  INSTANTIATE_STORE(int32_t, PARAM_T, OPT_T)

#define INSTANTIATE_ALL_OPTS(PARAM_T)                                          \
  INSTANTIATE_OPT_PAIR(PARAM_T, float)                                         \
  INSTANTIATE_OPT_PAIR(PARAM_T, float16_t)                                     \
  INSTANTIATE_OPT_PAIR(PARAM_T, bfloat16_t)

INSTANTIATE_ALL_OPTS(float)
INSTANTIATE_ALL_OPTS(float16_t)
INSTANTIATE_ALL_OPTS(bfloat16_t)

} // namespace se
