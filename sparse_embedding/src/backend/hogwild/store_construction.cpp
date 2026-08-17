#include "sparse_embedding/src/backend/hogwild/public/store_construction.hpp"

#include <stdexcept>
#include <string>
#include <utility>

namespace se {

StoreScorePolicy parse_store_score_policy(std::string_view name) {
  if (name == "none")
    return StoreScorePolicy::kNone;
  if (name == "lfu")
    return StoreScorePolicy::kLfu;
  if (name == "lru")
    return StoreScorePolicy::kLru;
  if (name == "manual")
    return StoreScorePolicy::kManual;
  throw std::invalid_argument("unknown score policy: " + std::string(name));
}

StoreFactory create_hogwild_store_factory(StoreConstructionKind kind) {
  if (kind.param_dtype == StoreParamDType::kFloat32 &&
      kind.optimizer_dtype == StoreParamDType::kFloat32) {
    return &store_factory_f32_f32;
  }
  if (kind.param_dtype == StoreParamDType::kFloat32 &&
      kind.optimizer_dtype == StoreParamDType::kFloat16) {
    return &store_factory_f32_f16;
  }
  if (kind.param_dtype == StoreParamDType::kFloat32 &&
      kind.optimizer_dtype == StoreParamDType::kBFloat16) {
    return &store_factory_f32_bf16;
  }
  if (kind.param_dtype == StoreParamDType::kFloat16 &&
      kind.optimizer_dtype == StoreParamDType::kFloat32) {
    return &store_factory_f16_f32;
  }
  if (kind.param_dtype == StoreParamDType::kFloat16 &&
      kind.optimizer_dtype == StoreParamDType::kFloat16) {
    return &store_factory_f16_f16;
  }
  if (kind.param_dtype == StoreParamDType::kFloat16 &&
      kind.optimizer_dtype == StoreParamDType::kBFloat16) {
    return &store_factory_f16_bf16;
  }
  if (kind.param_dtype == StoreParamDType::kBFloat16 &&
      kind.optimizer_dtype == StoreParamDType::kFloat32) {
    return &store_factory_bf16_f32;
  }
  if (kind.param_dtype == StoreParamDType::kBFloat16 &&
      kind.optimizer_dtype == StoreParamDType::kFloat16) {
    return &store_factory_bf16_f16;
  }
  if (kind.param_dtype == StoreParamDType::kBFloat16 &&
      kind.optimizer_dtype == StoreParamDType::kBFloat16) {
    return &store_factory_bf16_bf16;
  }
  throw std::invalid_argument("unsupported store construction kind");
}

void create_hogwild_store(StoreConstructionKind kind,
                          StoreConstructionTarget target,
                          StoreConstructionRequest request,
                          size_t thread_budget) {
  create_hogwild_store_factory(kind)(target, std::move(request), thread_budget,
                                     kind.score_policy);
}

} // namespace se
