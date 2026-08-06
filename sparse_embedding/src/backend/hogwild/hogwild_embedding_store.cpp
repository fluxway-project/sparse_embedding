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
namespace se {

template class EmbeddingArena<float>;
template class EmbeddingArena<float16_t>;
template class EmbeddingArena<bfloat16_t>;

template class TransientArena<float>;
template class TransientArena<float16_t>;
template class TransientArena<bfloat16_t>;

} // namespace se
