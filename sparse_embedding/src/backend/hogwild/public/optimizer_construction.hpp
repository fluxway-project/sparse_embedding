#pragma once

#include "sparse_embedding/src/common/function.hpp"
#include "sparse_embedding/src/interface/optimizer_interface.hpp"
#include "sparse_embedding/src/optimizer/optimizer_config.hpp"
#include "sparse_embedding/src/storage/page_storage.hpp"

namespace se {

template <typename ParamT, typename OptT>
using OptimizerConstructionFactory =
    Function<OptimizerExtension<ParamT, OptT>(size_t)>;

#define SE_DECLARE_OPTIMIZER_FACTORY(NAME, P, O)                               \
  OptimizerConstructionFactory<P, O> create_optimizer_factory_##NAME(          \
      const OptimizerConfigVariant &, PageStorage);

SE_DECLARE_OPTIMIZER_FACTORY(float_float, float, float)
SE_DECLARE_OPTIMIZER_FACTORY(float_float16, float, float16_t)
SE_DECLARE_OPTIMIZER_FACTORY(float_bfloat16, float, bfloat16_t)
SE_DECLARE_OPTIMIZER_FACTORY(float16_float, float16_t, float)
SE_DECLARE_OPTIMIZER_FACTORY(float16_float16, float16_t, float16_t)
SE_DECLARE_OPTIMIZER_FACTORY(float16_bfloat16, float16_t, bfloat16_t)
SE_DECLARE_OPTIMIZER_FACTORY(bfloat16_float, bfloat16_t, float)
SE_DECLARE_OPTIMIZER_FACTORY(bfloat16_float16, bfloat16_t, float16_t)
SE_DECLARE_OPTIMIZER_FACTORY(bfloat16_bfloat16, bfloat16_t, bfloat16_t)

#undef SE_DECLARE_OPTIMIZER_FACTORY

template <typename ParamT, typename OptT>
OptimizerConstructionFactory<ParamT, OptT>
create_optimizer_factory(const OptimizerConfigVariant &config,
                         PageStorage storage);

#define SE_DEFINE_OPTIMIZER_FACTORY_DISPATCH(NAME, P, O)                       \
  template <>                                                                  \
  inline OptimizerConstructionFactory<P, O> create_optimizer_factory<P, O>(    \
      const OptimizerConfigVariant &config, PageStorage storage) {             \
    return create_optimizer_factory_##NAME(config, std::move(storage));        \
  }

SE_DEFINE_OPTIMIZER_FACTORY_DISPATCH(float_float, float, float)
SE_DEFINE_OPTIMIZER_FACTORY_DISPATCH(float_float16, float, float16_t)
SE_DEFINE_OPTIMIZER_FACTORY_DISPATCH(float_bfloat16, float, bfloat16_t)
SE_DEFINE_OPTIMIZER_FACTORY_DISPATCH(float16_float, float16_t, float)
SE_DEFINE_OPTIMIZER_FACTORY_DISPATCH(float16_float16, float16_t, float16_t)
SE_DEFINE_OPTIMIZER_FACTORY_DISPATCH(float16_bfloat16, float16_t, bfloat16_t)
SE_DEFINE_OPTIMIZER_FACTORY_DISPATCH(bfloat16_float, bfloat16_t, float)
SE_DEFINE_OPTIMIZER_FACTORY_DISPATCH(bfloat16_float16, bfloat16_t, float16_t)
SE_DEFINE_OPTIMIZER_FACTORY_DISPATCH(bfloat16_bfloat16, bfloat16_t, bfloat16_t)

#undef SE_DEFINE_OPTIMIZER_FACTORY_DISPATCH

} // namespace se
