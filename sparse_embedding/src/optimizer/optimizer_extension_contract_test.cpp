#include "sparse_embedding/src/optimizer/optimizer_contract.hpp"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

class ShampooLikeMock {
public:
  ShampooLikeMock()
      : layout_({{"row_stats", sizeof(float), 4, 64,
                  se::OptimizerStateResidency::kHost,
                  se::OptimizerStateGranularity::kTile},
                 {"col_stats", sizeof(float), 3, 32,
                  se::OptimizerStateResidency::kHost,
                  se::OptimizerStateGranularity::kTile},
                 {"matrix_root", sizeof(float), 6, 64,
                  se::OptimizerStateResidency::kHost,
                  se::OptimizerStateGranularity::kTable}},
                3) {}

  void reserve(size_t capacity) {
    state_.resize(capacity * layout_.bytes_per_entry());
  }
  void allocate(uint32_t index) {
    if (index >= state_.size() / layout_.bytes_per_entry())
      throw std::out_of_range("mock optimizer index is not reserved");
    std::fill_n(state_.data() + index * layout_.bytes_per_entry(),
                layout_.bytes_per_entry(), std::byte{0});
  }
  void deallocate(uint32_t index) { allocate(index); }
  void clear() { state_.clear(); }

  void update(const se::OptimizerBatchView &batch, uint64_t, uint64_t) {
    if (batch.indices.empty() || batch.dimension == 0 ||
        batch.parameter_stride < batch.dimension * sizeof(float) ||
        batch.gradient_stride < batch.dimension * sizeof(float))
      throw std::invalid_argument("mock optimizer batch is empty");
  }

  se::OptimizerStateLayout state_layout() const { return layout_; }

  se::OptimizerStateCodec state_codec() const {
    return pro::make_proxy<se::OptimizerStateCodecFacade, Codec>(layout_,
                                                                 &state_);
  }

private:
  class Codec {
  public:
    Codec(se::OptimizerStateLayout layout, std::vector<std::byte> *state)
        : layout_(std::move(layout)), state_(state) {}
    se::OptimizerStateLayout layout() const { return layout_; }
    void encode(uint32_t index, std::span<std::byte> output) const {
      if (output.size() != layout_.bytes_per_entry())
        throw std::invalid_argument("mock codec encode size mismatch");
      std::memcpy(output.data(), state(index), output.size());
    }
    void decode(uint32_t index, std::span<const std::byte> input) {
      if (input.size() != layout_.bytes_per_entry())
        throw std::invalid_argument("mock codec decode size mismatch");
      std::memcpy(state(index), input.data(), input.size());
    }

  private:
    std::byte *state(uint32_t index) const {
      if (state_ == nullptr)
        throw std::logic_error("mock codec state is unavailable");
      return state_->data() + index * layout_.bytes_per_entry();
    }

    se::OptimizerStateLayout layout_;
    std::vector<std::byte> *state_ = nullptr;
  };

  se::OptimizerStateLayout layout_;
  mutable std::vector<std::byte> state_;
};

} // namespace

int main() {
  ShampooLikeMock mock;
  se::OptimizerExtension extension =
      pro::make_proxy<se::OptimizerExtensionFacade, ShampooLikeMock>(mock);
  extension->reserve(2);
  extension->allocate(1);
  const se::OptimizerStateLayout layout = extension->state_layout();
  assert(layout.blocks().size() == 3);
  assert(layout.block_offset(1) % 32 == 0);
  assert(layout.block_offset(2) % 64 == 0);
  assert(layout.bytes_per_entry() > 0);
  auto codec = extension->state_codec();
  std::vector<std::byte> encoded(layout.bytes_per_entry());
  se::validate_optimizer_codec(codec, encoded, layout.version());
  codec->encode(1, encoded);
  codec->decode(1, encoded);
  std::vector<uint32_t> indices{1};
  std::vector<std::byte> parameters(4 * sizeof(float));
  std::vector<std::byte> gradients(parameters.size());
  extension->update(
      {indices, parameters, gradients, parameters.size(), gradients.size(), 4},
      1, 0);
  extension->deallocate(1);
  extension->clear();
  try {
    se::OptimizerStateLayout invalid(
        {{"bad", sizeof(float), 1, 3, se::OptimizerStateResidency::kHost,
          se::OptimizerStateGranularity::kEntry}},
        1);
    (void)invalid;
    return 1;
  } catch (const std::invalid_argument &) {
  }
  try {
    se::OptimizerStateLayout overflow(
        {{"overflow", std::numeric_limits<size_t>::max(), 2, 1,
          se::OptimizerStateResidency::kHost,
          se::OptimizerStateGranularity::kEntry}},
        1);
    (void)overflow;
    return 1;
  } catch (const std::overflow_error &) {
  }
  return 0;
}
