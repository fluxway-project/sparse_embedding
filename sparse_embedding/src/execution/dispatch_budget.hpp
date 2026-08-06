#pragma once

#include <cstddef>
#include <limits>
#include <stdexcept>

namespace se {

inline size_t normalize_thread_budget(size_t budget, size_t automatic_budget) {
  const size_t requested = budget == 0 ? automatic_budget : budget;
  if (requested == 0)
    throw std::invalid_argument("thread budget is zero");
  return requested;
}

} // namespace se
