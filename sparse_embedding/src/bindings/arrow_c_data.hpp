#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

struct ArrowSchema {
  const char *format;
  const char *name;
  const char *metadata;
  int64_t flags;
  int64_t n_children;
  struct ArrowSchema **children;
  struct ArrowSchema *dictionary;
  void (*release)(struct ArrowSchema *);
  void *private_data;
};

struct ArrowArray {
  int64_t length;
  int64_t null_count;
  int64_t offset;
  int64_t n_buffers;
  int64_t n_children;
  const void **buffers;
  struct ArrowArray **children;
  void *private_data;
  void (*release)(struct ArrowArray *);
};

#ifdef __cplusplus
}
#endif

namespace se {
namespace py {

template <typename T> struct ArrowVectorCtx {
  std::unique_ptr<std::vector<T>> vec;
  std::string format;
  const void *buffers[2];

  ArrowVectorCtx(std::unique_ptr<std::vector<T>> v, std::string fmt)
      : vec(std::move(v)), format(std::move(fmt)) {
    buffers[0] = nullptr;     // validity buffer
    buffers[1] = vec->data(); // data buffer
  }
};

template <typename T>
std::tuple<uintptr_t, uintptr_t>
wrap_vector_as_arrow(std::unique_ptr<std::vector<T>> data, const char *format) {
  auto *ctx = new ArrowVectorCtx<T>(std::move(data), format);

  ArrowSchema *schema = new ArrowSchema;
  schema->format = ctx->format.c_str();
  schema->name = "";
  schema->metadata = nullptr;
  schema->flags = 0;
  schema->n_children = 0;
  schema->children = nullptr;
  schema->dictionary = nullptr;
  schema->private_data = nullptr;
  schema->release = [](ArrowSchema *s) {
    s->release = nullptr;
    delete s;
  };

  ArrowArray *array = new ArrowArray;
  array->length = static_cast<int64_t>(ctx->vec->size());
  array->null_count = 0;
  array->offset = 0;
  array->n_buffers = 2;
  array->n_children = 0;
  array->buffers = ctx->buffers;
  array->children = nullptr;
  array->private_data = ctx;
  array->release = [](ArrowArray *a) {
    auto *c = static_cast<ArrowVectorCtx<T> *>(a->private_data);
    delete c;
    a->release = nullptr;
    delete a;
  };

  return {reinterpret_cast<uintptr_t>(array),
          reinterpret_cast<uintptr_t>(schema)};
}

} // namespace py
} // namespace se
