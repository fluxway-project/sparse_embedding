#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <string_view>
#include <variant>
#include <vector>

#include <proxy/proxy.h>

#include "sparse_embedding/src/common/numeric_types.hpp"
#include "sparse_embedding/src/eviction/score_policy.hpp"
#include "sparse_embedding/src/interface/embedding_table_interface.hpp"
#include "sparse_embedding/src/interface/initializer_interface.hpp"
#include "sparse_embedding/src/optimizer/optimizer_config.hpp"
#include "sparse_embedding/src/storage/page_storage.hpp"

namespace se {

using StoreProxyVariant =
    std::variant<pro::proxy<EmbeddingTableFacade<float, float>>,
                 pro::proxy<EmbeddingTableFacade<float, float16_t>>,
                 pro::proxy<EmbeddingTableFacade<float, bfloat16_t>>,
                 pro::proxy<EmbeddingTableFacade<float16_t, float>>,
                 pro::proxy<EmbeddingTableFacade<float16_t, float16_t>>,
                 pro::proxy<EmbeddingTableFacade<float16_t, bfloat16_t>>,
                 pro::proxy<EmbeddingTableFacade<bfloat16_t, float>>,
                 pro::proxy<EmbeddingTableFacade<bfloat16_t, float16_t>>,
                 pro::proxy<EmbeddingTableFacade<bfloat16_t, bfloat16_t>>>;

enum class StoreParamDType : uint8_t { kFloat32, kFloat16, kBFloat16 };
using StoreScorePolicy = ScorePolicyKind;

struct StoreConstructionKind {
  StoreParamDType param_dtype;
  StoreParamDType optimizer_dtype;
  StoreScorePolicy score_policy;
};

struct StoreConstructionRequest {
  std::shared_ptr<pro::proxy<InitializerBuilderFacade>> initializer_builder;
  OptimizerConfigVariant optimizer_config;
};

struct StoreConstructionTarget {
  StoreProxyVariant *store;
  std::span<const size_t> dims;
  PageStorage storage;
};

using StoreFactory = void (*)(StoreConstructionTarget &,
                              StoreConstructionRequest, size_t,
                              StoreScorePolicy);

#define SE_DECLARE_STORE_FACTORY(P, O)                                         \
  void store_factory_##P##_##O(StoreConstructionTarget &,                      \
                               StoreConstructionRequest, size_t,               \
                               StoreScorePolicy);

SE_DECLARE_STORE_FACTORY(f32, f32)
SE_DECLARE_STORE_FACTORY(f32, f16)
SE_DECLARE_STORE_FACTORY(f32, bf16)
SE_DECLARE_STORE_FACTORY(f16, f32)
SE_DECLARE_STORE_FACTORY(f16, f16)
SE_DECLARE_STORE_FACTORY(f16, bf16)
SE_DECLARE_STORE_FACTORY(bf16, f32)
SE_DECLARE_STORE_FACTORY(bf16, f16)
SE_DECLARE_STORE_FACTORY(bf16, bf16)

#undef SE_DECLARE_STORE_FACTORY

void create_hogwild_store(StoreConstructionKind, StoreConstructionTarget,
                          StoreConstructionRequest, size_t thread_budget);
StoreFactory create_hogwild_store_factory(StoreConstructionKind);

StoreScorePolicy parse_store_score_policy(std::string_view name);

} // namespace se
