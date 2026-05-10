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

#ifndef SPARSE_EMBEDDING_SLOT_KEY_CODEC_HPP_
#define SPARSE_EMBEDDING_SLOT_KEY_CODEC_HPP_

#include <cstdint>
#include <stdexcept>

namespace se {

class SlotKeyCodec {
public:
  explicit SlotKeyCodec(uint8_t slot_bits = 16)
      : slot_bits_(slot_bits), key_bits_(64 - slot_bits) {
    if (slot_bits > 32) {
      throw std::invalid_argument("slot_bits must be in [0, 32], got " +
                                  std::to_string(slot_bits));
    }
    slot_mask_ =
        slot_bits_ == 0 ? 0ULL : ((1ULL << slot_bits_) - 1) << key_bits_;
    key_mask_ = key_bits_ == 64 ? ~0ULL : (1ULL << key_bits_) - 1;
  }

  uint64_t encode(uint32_t slot_id, uint64_t raw_key) const {
    if (slot_bits_ == 0)
      return raw_key;
    return (static_cast<uint64_t>(slot_id) << key_bits_) |
           (raw_key & key_mask_);
  }

  uint32_t decode_slot(uint64_t composite_key) const {
    if (slot_bits_ == 0)
      return 0;
    return static_cast<uint32_t>((composite_key >> key_bits_) &
                                 ((1ULL << slot_bits_) - 1));
  }

  uint64_t decode_key(uint64_t composite_key) const {
    return composite_key & key_mask_;
  }

  uint8_t slot_bits() const { return slot_bits_; }
  uint8_t key_bits() const { return key_bits_; }

  uint32_t max_slot_id() const {
    return slot_bits_ == 0 ? 0
                           : static_cast<uint32_t>((1ULL << slot_bits_) - 1);
  }

  uint64_t max_raw_key() const { return key_mask_; }

private:
  uint8_t slot_bits_;
  uint8_t key_bits_;
  uint64_t slot_mask_;
  uint64_t key_mask_;
};

} // namespace se

#endif // SPARSE_EMBEDDING_SLOT_KEY_CODEC_HPP_
