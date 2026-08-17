#include "sparse_embedding/src/backend/hogwild/hogwild_embedding_store.hpp"
#include "sparse_embedding/src/backend/hogwild/public/hogwild_store_options.hpp"
#include "sparse_embedding/src/backend/hogwild/public/store_construction.hpp"
#include "sparse_embedding/src/eviction/score_policy.hpp"
#include "sparse_embedding/src/backend/hogwild/public/optimizer_construction.hpp"

#include <stdexcept>
#include <type_traits>

namespace se::detail {

template <typename ParamT>
auto make_initializer(const StoreConstructionRequest &request) {
  if constexpr (std::is_same_v<ParamT, float>)
    return (*request.initializer_builder)->build_f32();
  else if constexpr (std::is_same_v<ParamT, float16_t>)
    return (*request.initializer_builder)->build_f16();
  else
    return (*request.initializer_builder)->build_bf16();
}

template <typename ParamT, typename OptT>
void make_store(StoreConstructionTarget &target,
                StoreConstructionRequest request, size_t thread_budget,
                StoreScorePolicy score_policy) {
  auto initializer = make_initializer<ParamT>(request);
  auto optimizer = create_optimizer_factory<ParamT, OptT>(
      request.optimizer_config, target.storage);
  using Store = HogwildEmbeddingStore<uint64_t, ParamT, OptT>;
  *target.store = pro::make_proxy<EmbeddingTableFacade<ParamT, OptT>, Store>(
      std::vector<size_t>(target.dims.begin(), target.dims.end()),
      std::move(initializer), std::move(optimizer),
      score_policy,
      HogwildStoreOptions{.thread_budget = thread_budget,
                          .storage = std::move(target.storage)});
}

}  // namespace se::detail

#define SE_DEFINE_STORE_FACTORY(P, O, NAME)                                   \
  void se::store_factory_##NAME(se::StoreConstructionTarget &target,          \
                                 se::StoreConstructionRequest request,         \
                                 size_t thread_budget,                        \
                                 se::StoreScorePolicy score_policy) {          \
    se::detail::make_store<P, O>(target, std::move(request), thread_budget,   \
                                 score_policy);                                 \
  }
