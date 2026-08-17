#include "sparse_embedding/src/backend/hogwild/public/optimizer_construction.hpp"

#include <type_traits>

#include "sparse_embedding/src/optimizer/adam_optimizer.hpp"
#include "sparse_embedding/src/optimizer/sgd_optimizer.hpp"

namespace se::detail {

template <typename ParamT, typename OptT>
OptimizerConstructionFactory<ParamT, OptT> make_optimizer_factory(
    const OptimizerConfigVariant &config, PageStorage storage) {
  using Facade = OptimizerFacade<ParamT, OptT>;
  return pro::make_proxy<FunctionFacade<pro::proxy<Facade>(size_t)>>( 
      [config, storage = std::move(storage)](size_t dim) mutable {
        return std::visit(
            [&](const auto &optimizer_config) -> pro::proxy<Facade> {
              auto make = [&]<template <typename, typename, typename>
                          typename Optimizer>() -> pro::proxy<Facade> {
                if (optimizer_config.dynamic_lr_policy.has_value()) {
                  return std::visit(
                      [&](const auto &policy) -> pro::proxy<Facade> {
                        using LearningRate =
                            DynamicLearningRate<std::decay_t<decltype(policy)>>;
                        return pro::make_proxy<
                            Facade, Optimizer<ParamT, OptT, LearningRate>>(
                            dim, optimizer_config, storage,
                            LearningRate(optimizer_config.lr, policy));
                      },
                      *optimizer_config.dynamic_lr_policy);
                }
                return pro::make_proxy<Facade,
                                       Optimizer<ParamT, OptT,
                                                 StaticLearningRate>>(
                    dim, optimizer_config, storage,
                    StaticLearningRate(optimizer_config.lr));
              };
              if constexpr (std::is_same_v<std::decay_t<decltype(optimizer_config)>,
                                           AdamConfig>) {
                return make.template operator()<AdamOptimizer>();
              } else {
                return make.template operator()<SgdOptimizer>();
              }
            },
            config);
      });
}

}  // namespace se::detail

#define SE_DEFINE_OPTIMIZER_FACTORY(P, O, NAME)                               \
  se::OptimizerConstructionFactory<P, O>                                      \
      se::create_optimizer_factory_##NAME(const se::OptimizerConfigVariant &c,\
                                          se::PageStorage s) {                \
    return se::detail::make_optimizer_factory<P, O>(c, std::move(s));          \
  }
