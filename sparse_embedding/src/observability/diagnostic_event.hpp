#pragma once

#include <cstdint>

namespace se {

enum class DiagnosticKind : uint8_t {
  kOptimizerLocatorRevalidation,
};

struct DiagnosticEvent {
  DiagnosticKind kind = DiagnosticKind::kOptimizerLocatorRevalidation;
  uint64_t key = 0;
};

using DiagnosticSink = void (*)(const DiagnosticEvent &) noexcept;

} // namespace se
