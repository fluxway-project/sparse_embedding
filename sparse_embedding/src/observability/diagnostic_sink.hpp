#pragma once

#include <cstdlib>
#include <iostream>

#include "sparse_embedding/src/observability/diagnostic_event.hpp"

namespace se {

inline void write_diagnostic_to_clog(const DiagnosticEvent &event) noexcept {
  if (event.kind != DiagnosticKind::kOptimizerLocatorRevalidation)
    std::terminate();
  std::clog << "SparseEmbedding: optimizer locator revalidation failed; "
               "the affected erased gradient was discarded key="
            << event.key << '\n';
}

} // namespace se
