#pragma once

#include <proxy/proxy.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>

#include "sparse_embedding/src/optimizer/optimizer_state_layout.hpp"

namespace se {

PRO_DEF_MEM_DISPATCH(OptimizerCodecLayout, layout);
PRO_DEF_MEM_DISPATCH(OptimizerCodecEncode, encode);
PRO_DEF_MEM_DISPATCH(OptimizerCodecDecode, decode);

struct OptimizerStateCodecFacade
    : pro              //
      ::facade_builder //
      ::add_convention<OptimizerCodecLayout,
                       OptimizerStateLayout() const> //
      ::add_convention<OptimizerCodecEncode,
                       void(uint32_t, std::span<std::byte>) const> //
      ::add_convention<OptimizerCodecDecode,
                       void(uint32_t,
                            std::span<const std::byte>)> //
      ::build {};

using OptimizerStateCodec = pro::proxy<OptimizerStateCodecFacade>;

inline void validate_optimizer_codec(const OptimizerStateCodec &codec,
                                     std::span<const std::byte> data,
                                     size_t expected_version) {
  const OptimizerStateLayout layout = codec->layout();
  if (layout.version() != expected_version)
    throw std::invalid_argument("optimizer state codec version mismatch");
  if (data.size() != layout.bytes_per_entry())
    throw std::invalid_argument("optimizer codec state size mismatch");
}

} // namespace se
