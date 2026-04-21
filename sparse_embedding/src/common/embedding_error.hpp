// Copyright 2026 The Fluxway Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#pragma once

#ifndef SPARSE_EMBEDDING_EMBEDDING_ERROR_HPP_
#define SPARSE_EMBEDDING_EMBEDDING_ERROR_HPP_

#include <string>
#include <system_error>

namespace se {

enum class EmbeddingErrc {
  kSuccess = 0,
  kInvalidConfig = 1,
  kInvalidDimension = 2,
  kInvalidState = 3,
  kInternalError = 4
};

} // namespace se

namespace std {

template <> struct is_error_code_enum<se::EmbeddingErrc> : true_type {};

} // namespace std

namespace se {

class EmbeddingErrorCategoryImpl : public std::error_category {
public:
  const char *name() const noexcept override { return "EmbeddingError"; }

  std::string message(int c) const override {
    switch (static_cast<EmbeddingErrc>(c)) {
    case EmbeddingErrc::kSuccess:
      return "Success";
    case EmbeddingErrc::kInvalidConfig:
      return "Invalid configuration format or values";
    case EmbeddingErrc::kInvalidDimension:
      return "Mismatch between embedding dimension and provided buffer";
    case EmbeddingErrc::kInvalidState:
      return "Operation invalid in current state (e.g., used after shutdown)";
    case EmbeddingErrc::kInternalError:
      return "Internal backend failure";
    default:
      return "Unknown error";
    }
  }
};

inline const std::error_category &EmbeddingErrorCategory() {
  static se::EmbeddingErrorCategoryImpl category;
  return category;
}

inline std::error_code make_error_code(EmbeddingErrc e) {
  return {static_cast<int>(e), EmbeddingErrorCategory()};
}

} // namespace se

#endif // SPARSE_EMBEDDING_EMBEDDING_ERROR_HPP_
