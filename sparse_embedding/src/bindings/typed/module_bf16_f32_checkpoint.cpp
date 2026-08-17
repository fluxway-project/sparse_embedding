#include "sparse_embedding/src/bindings/typed/module_typed_checkpoint.inl"

namespace se::py::detail {
template struct CheckpointOpsImpl<se::bfloat16_t, float>;
}
