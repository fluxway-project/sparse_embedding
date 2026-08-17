#include "sparse_embedding/src/arena/transient_arena.hpp"

namespace se {

// Explicit template instantiations for TransientArena
template class TransientArena<float>;
template class TransientArena<float16_t>;
template class TransientArena<bfloat16_t>;

} // namespace se
