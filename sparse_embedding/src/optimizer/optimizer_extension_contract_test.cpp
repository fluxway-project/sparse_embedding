#include "sparse_embedding/src/optimizer/optimizer_contract.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

class ShampooLikeMock {
public:
  void reserve(size_t) {}
  void allocate(uint32_t) {}
  void deallocate(uint32_t index) { allocate(index); }
  void clear() {}

  void update(const se::OptimizerBatchView<float> &batch, uint64_t) {
    if (!batch.valid())
      throw std::invalid_argument("mock optimizer batch is empty");
  }
};

} // namespace

int main() {
  ShampooLikeMock mock;
  se::OptimizerExtension<float, float> extension =
      pro::make_proxy<se::OptimizerExtensionFacade<float, float>,
                      ShampooLikeMock>(mock);
  extension->reserve(2);
  extension->allocate(1);
  std::vector<uint32_t> indices{1};
  std::vector<float> parameter_row(4);
  std::vector<float> gradient_row(4, 1.0F);
  std::vector<float> parameters = parameter_row;
  std::vector<float> gradients = gradient_row;
  std::vector<uint64_t> scores{0};
  std::vector<se::OptimizerBatchRow<float>> rows{
      {indices[0], parameters, gradients, scores[0]}};
  const se::OptimizerBatchView<float> view{rows, 4};
  require(view.valid(), "borrowed batch view must be valid");
  require(view.row(0).parameters.data() == parameters.data(),
          "batch view must borrow parameter storage");
  require(view.row(0).gradients.data() == gradients.data(),
          "batch view must borrow gradient storage");
  extension->update(view, 1);

  const se::OptimizerBatchView<float> empty_view{{}, 4};
  require(empty_view.valid(), "empty batch with a dimension must be valid");
  require(empty_view.size() == 0, "empty batch size must be zero");
  bool out_of_range = false;
  try {
    (void)empty_view.row(0);
  } catch (const std::out_of_range &) {
    out_of_range = true;
  }
  require(out_of_range, "batch row access must reject out-of-range positions");

  const std::array<se::OptimizerBatchRow<float>, 1> malformed_rows{rows[0]};
  const se::OptimizerBatchView<float> malformed{malformed_rows, 3};
  require(!malformed.valid(), "row dimension mismatch must be invalid");

  extension->deallocate(1);
  extension->clear();
  return 0;
}
