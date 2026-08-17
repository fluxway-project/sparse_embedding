#include "sparse_embedding/src/backend/hogwild/io/hogwild_page_image_codec.hpp"

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {
void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}
} // namespace

int main() {
  se::PageImageExport<float, float> image;
  image.embedding_dim = 2;
  image.page_slot_stride_bytes = 64;
  image.slot_ordinals = {0, 3};
  image.keys = {7, 9};
  image.seq_counts = {2, 4};
  image.values = {1, 2, 3, 4};
  image.optimizer_states = {5, 6, 7, 8};
  image.steps = {1, 2};
  require(se::hogwild::io::validate_page_image_columns(image, 2, 64, false, 2,
                                                       4096) == 2,
          "valid page image rejected");
  image.scores.push_back(1);
  try {
    (void)se::hogwild::io::validate_page_image_columns(image, 2, 64, false, 2,
                                                       4096);
    return 1;
  } catch (const std::invalid_argument &) {
  }
  return 0;
}
