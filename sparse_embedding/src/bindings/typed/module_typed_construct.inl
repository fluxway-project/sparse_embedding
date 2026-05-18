// Copyright 2026 The Fluxway Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0
// =============================================================================
// Implementation of ConstructOpsImpl<ParamT, OptT>::construct
// =============================================================================

#include "sparse_embedding/src/bindings/module_wrapper.hpp"
#include "sparse_embedding/src/backend/hogwild/hogwild_embedding_store.hpp"
#include "sparse_embedding/src/eviction/score_policy.hpp"

namespace se::py::detail {

template <typename ParamT, typename OptT>
void ConstructOpsImpl<ParamT, OptT>::construct(
    EmbeddingStoreWrapper &self, const PyInitializer &init_cfg,
    const PyOptimizer &opt_cfg, const std::string &score_policy_name,
    size_t nt, size_t max_active_keys) {
  // 1) Build the parameter-precision initializer proxy.
  pro::proxy<se::InitializerFacade<ParamT>> init_proxy;
  if constexpr (std::is_same_v<ParamT, float>) {
    init_proxy = (*init_cfg.builder)->build_f32();
  } else if constexpr (std::is_same_v<ParamT, se::float16_t>) {
    init_proxy = (*init_cfg.builder)->build_f16();
  } else if constexpr (std::is_same_v<ParamT, se::bfloat16_t>) {
    init_proxy = (*init_cfg.builder)->build_bf16();
  }

  // 2) Build the optimizer factory closure.
  using OptFacade = se::OptimizerFacade<ParamT, OptT>;
  using OptFactory = se::Function<pro::proxy<OptFacade>(size_t)>;
  OptFactory opt_factory;
  auto make_factory = [&](auto builder_fn) {
    opt_factory =
        pro::make_proxy<se::FunctionFacade<pro::proxy<OptFacade>(size_t)>>(
            [builder = opt_cfg.builder, builder_fn](size_t d) mutable {
              return builder_fn(*builder, d);
            });
  };
  if constexpr (std::is_same_v<ParamT, float> &&
                std::is_same_v<OptT, float>) {
    make_factory([](auto &b, size_t d) { return b->build_f32_f32(d); });
  } else if constexpr (std::is_same_v<ParamT, float> &&
                       std::is_same_v<OptT, se::float16_t>) {
    make_factory([](auto &b, size_t d) { return b->build_f32_f16(d); });
  } else if constexpr (std::is_same_v<ParamT, float> &&
                       std::is_same_v<OptT, se::bfloat16_t>) {
    make_factory([](auto &b, size_t d) { return b->build_f32_bf16(d); });
  } else if constexpr (std::is_same_v<ParamT, se::float16_t> &&
                       std::is_same_v<OptT, float>) {
    make_factory([](auto &b, size_t d) { return b->build_f16_f32(d); });
  } else if constexpr (std::is_same_v<ParamT, se::float16_t> &&
                       std::is_same_v<OptT, se::float16_t>) {
    make_factory([](auto &b, size_t d) { return b->build_f16_f16(d); });
  } else if constexpr (std::is_same_v<ParamT, se::float16_t> &&
                       std::is_same_v<OptT, se::bfloat16_t>) {
    make_factory([](auto &b, size_t d) { return b->build_f16_bf16(d); });
  } else if constexpr (std::is_same_v<ParamT, se::bfloat16_t> &&
                       std::is_same_v<OptT, float>) {
    make_factory([](auto &b, size_t d) { return b->build_bf16_f32(d); });
  } else if constexpr (std::is_same_v<ParamT, se::bfloat16_t> &&
                       std::is_same_v<OptT, se::float16_t>) {
    make_factory([](auto &b, size_t d) { return b->build_bf16_f16(d); });
  } else if constexpr (std::is_same_v<ParamT, se::bfloat16_t> &&
                       std::is_same_v<OptT, se::bfloat16_t>) {
    make_factory([](auto &b, size_t d) { return b->build_bf16_bf16(d); });
  }

  // 3) Pick KeyT and ScorePolicy, then materialise the concrete
  //    HogwildEmbeddingStore<KeyT, ParamT, OptT, Policy> into the variant.
  auto construct_with_key = [&]<typename KeyT>() {
    auto construct_with_policy = [&]<typename Policy>(Policy policy) {
      using StoreType =
          se::HogwildEmbeddingStore<KeyT, ParamT, OptT, Policy>;
      self.store_ = pro::make_proxy<se::EmbeddingTableFacade<ParamT, OptT>,
                                    StoreType>(
          self.dims_, nt, std::move(init_proxy), max_active_keys,
          std::move(opt_factory), std::move(policy));
    };

    if (score_policy_name == "none") {
      construct_with_policy(se::NoScorePolicy{});
    } else if (score_policy_name == "lfu") {
      construct_with_policy(se::LfuScorePolicy{});
    } else if (score_policy_name == "lru") {
      construct_with_policy(se::LruScorePolicy{});
    } else if (score_policy_name == "manual") {
      construct_with_policy(se::ManualScorePolicy{});
    } else {
      throw std::invalid_argument("unknown score policy: " +
                                  score_policy_name);
    }
  };

  if (self.key_dtype_ == "int64")
    construct_with_key.template operator()<int64_t>();
  else if (self.key_dtype_ == "uint64")
    construct_with_key.template operator()<uint64_t>();
  else if (self.key_dtype_ == "int32")
    construct_with_key.template operator()<int32_t>();
  else if (self.key_dtype_ == "uint32")
    construct_with_key.template operator()<uint32_t>();
  else
    throw std::invalid_argument("unsupported key_dtype: " + self.key_dtype_);
}

} // namespace se::py::detail
