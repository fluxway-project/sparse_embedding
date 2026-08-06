#include "sparse_embedding/src/backend/hogwild/store_page_eviction_proxy.hpp"

#include <cstdint>
#include <stdexcept>

namespace {

void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

} // namespace

int main() {
  se::StorePageEvictionProxy proxy;
  require(proxy.score_enabled(), "page-eviction scoring is disabled");

  uint64_t score = 0;
  require(!proxy.evaluate_page(7, score),
          "missing page evaluator unexpectedly succeeded");
  proxy.register_page(7, [] { return uint64_t{41}; });
  require(proxy.evaluate_page(7, score) && score == 41,
          "registered page evaluator returned the wrong score");
  proxy.register_page(7, [] { return uint64_t{73}; });
  require(proxy.evaluate_page(7, score) && score == 73,
          "replacement page evaluator was not published");

  proxy.register_page(9, [&proxy] {
    proxy.register_page(10, [] { return uint64_t{101}; });
    return uint64_t{99};
  });
  require(proxy.evaluate_page(9, score) && score == 99,
          "page evaluator ran while the registry mutex was held");
  require(proxy.evaluate_page(10, score) && score == 101,
          "page evaluator could not update the registry");

  proxy.unregister_page(7);
  require(!proxy.evaluate_page(7, score),
          "unregistered page evaluator remained visible");
}
