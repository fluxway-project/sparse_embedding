#pragma once

#include <cstdint>

namespace se {

enum class LocalityMode : uint8_t { kAuto, kLocal, kNuma };

struct LocalityPolicy {
  LocalityMode execution = LocalityMode::kAuto;
  LocalityMode memory = LocalityMode::kAuto;
};

} // namespace se
